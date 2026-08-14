from __future__ import annotations

import argparse
import gc
import importlib.util
import json
from pathlib import Path
import statistics
import sys
import time

import torch
import yaml


EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[1]
BASE_EXPERIMENT_DIR = REPO_ROOT / "experiments" / "ariadnet_lut_neural_preset_setting"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))

from metrics import psnr
from models.ariadne_lut_v2 import AriadneLUTV2
from utils.checkpoint import load_checkpoint
from utils.config import load_config
from ariadne_query_adaptive import AriadneLUTQueryAdaptiveStage1


def _load_dataset_module():
    path = BASE_EXPERIMENT_DIR / "neural_preset_dataset.py"
    spec = importlib.util.spec_from_file_location("qa_bench_dataset", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def parse_args():
    p = argparse.ArgumentParser(description="Compare dense Stage1 vs Query-Adaptive Stage1 compute/quality.")
    p.add_argument("--qa-config", default=str(EXPERIMENT_DIR / "config.yaml"))
    p.add_argument("--dense-checkpoint", required=True)
    p.add_argument("--qa-checkpoint", required=True)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--max-batches", type=int, default=20)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--repeats", type=int, default=20)
    p.add_argument("--output", default=str(EXPERIMENT_DIR / "benchmark_result.json"))
    return p.parse_args()


def parameter_count(module):
    return sum(p.numel() for p in module.parameters())


def parameter_bytes(module):
    return sum(p.numel() * p.element_size() for p in module.parameters())


def model_parameter_breakdown(model):
    generators = (model.normalization_lut_generator, model.style_lut_generator)
    highres_names = (
        "up_features_16", "up_lut_16", "stage_16", "to_delta_16",
        "up_features_32", "up_lut_32", "stage_32", "to_delta_32",
        "refine_16", "refine_32",
    )
    highres = sum(
        parameter_count(getattr(generator, name))
        for generator in generators
        for name in highres_names
        if hasattr(generator, name)
    )
    return {
        "total": parameter_count(model),
        "lut_generators": sum(parameter_count(generator) for generator in generators),
        "highres_refiners": highres,
        "parameter_memory_mb": parameter_bytes(model) / (1024 ** 2),
    }


def active_stats(result):
    vals = {"norm16": [], "norm32": [], "style16": [], "style32": []}
    for state in (result["state_a"], result["state_b"]):
        npyr = state["normalization_lut_pyramid"]
        spyr = state["style_lut_pyramid"]
        vals["norm16"].append(float(npyr["level_16"].support.active_ratio.mean().cpu()))
        vals["norm32"].append(float(npyr["level_32"].support.active_ratio.mean().cpu()))
        vals["style16"].append(float(spyr["level_16"].support.active_ratio.mean().cpu()))
        vals["style32"].append(float(spyr["level_32"].support.active_ratio.mean().cpu()))
    return {k: statistics.mean(v) for k, v in vals.items()}


def timed_forward(model, a, b, warmup, repeats):
    model.eval()
    with torch.inference_mode():
        for _ in range(warmup):
            warm = model.forward_pair(a, b)
        del warm
        torch.cuda.synchronize()
        gc.collect()
        baseline = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        times = []
        for _ in range(repeats):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            result = model.forward_pair(a, b)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000.0)
            del result
        peak_incremental = max(torch.cuda.max_memory_allocated() - baseline, 0) / (1024 ** 2)
    return statistics.mean(times), statistics.pstdev(times), peak_incremental


def main():
    args = parse_args()
    qa_cfg = yaml.safe_load(Path(args.qa_config).read_text())
    base_cfg = load_config(str((REPO_ROOT / qa_cfg["base_config"]).resolve()))
    base_cfg.data.batch_size = args.batch_size
    loader, _ = _load_dataset_module().build_loader(base_cfg, str(base_cfg.data.val_split), train=False)
    device = torch.device("cuda")

    dense = AriadneLUTV2(base_cfg.model).to(device)
    load_checkpoint(args.dense_checkpoint, dense, strict=True, map_location=device)
    qa = AriadneLUTQueryAdaptiveStage1(base_cfg.model, qa_cfg["model"]).to(device)
    load_checkpoint(args.qa_checkpoint, qa, strict=True, map_location=device)

    first = next(iter(loader))
    a = first["image_a"].to(device)
    b = first["image_b"].to(device)
    dense_ms, dense_std, dense_mem = timed_forward(dense, a, b, args.warmup, args.repeats)
    gc.collect()
    torch.cuda.empty_cache()
    qa_ms, qa_std, qa_mem = timed_forward(qa, a, b, args.warmup, args.repeats)

    # Quality on a fixed number of validation batches.
    scores = {"dense": [], "qa": []}
    active_accum = []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= args.max_batches:
                break
            aa = batch["image_a"].to(device)
            bb = batch["image_b"].to(device)
            dr = dense.forward_pair(aa, bb)
            qr = qa.forward_pair(aa, bb)
            scores["dense"].append(0.5 * (float(psnr(dr["output_ab"].clamp(0,1), bb)) + float(psnr(dr["output_ba"].clamp(0,1), aa))))
            scores["qa"].append(0.5 * (float(psnr(qr["output_ab"].clamp(0,1), bb)) + float(psnr(qr["output_ba"].clamp(0,1), aa))))
            active_accum.append(active_stats(qr))

    active_mean = {
        k: statistics.mean(row[k] for row in active_accum)
        for k in active_accum[0]
    }
    dense_params = model_parameter_breakdown(dense)
    qa_params = model_parameter_breakdown(qa)
    payload = {
        "dense": {
            "parameters": dense_params,
            "forward_pair_ms_mean": dense_ms,
            "forward_pair_ms_std": dense_std,
            "peak_incremental_cuda_memory_mb": dense_mem,
            "mean_bidirectional_psnr": statistics.mean(scores["dense"]),
            "highres_refinement_fraction": 1.0,
        },
        "query_adaptive": {
            "parameters": qa_params,
            "forward_pair_ms_mean": qa_ms,
            "forward_pair_ms_std": qa_std,
            "peak_incremental_cuda_memory_mb": qa_mem,
            "mean_bidirectional_psnr": statistics.mean(scores["qa"]),
            "active_fraction": active_mean,
        },
        "ratios": {
            "latency_qa_over_dense": qa_ms / dense_ms,
            "memory_qa_over_dense": qa_mem / dense_mem,
            "parameter_qa_over_dense": qa_params["total"] / dense_params["total"],
            "highres_refiner_parameter_qa_over_dense": qa_params["highres_refiners"] / dense_params["highres_refiners"],
        },
        "note": "Active fraction is not a speedup claim. Latency is synchronized wall time; peak incremental CUDA memory is the forward peak above the allocation present immediately before timing.",
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
