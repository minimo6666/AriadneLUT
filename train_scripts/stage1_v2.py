from __future__ import annotations

from itertools import islice
import json
import math
from pathlib import Path

import torch
from torch.amp import GradScaler, autocast
from tqdm import tqdm

from datasets.coco_wds_v2 import (
    get_coco_loader_v2,
)
from losses.stage1_loss_v2 import Stage1LossV2
from metrics import psnr
from models.ariadne_lut_v2 import AriadneLUTV2
from utils.checkpoint import (
    load_checkpoint,
    save_checkpoint,
)
from utils.film_grade import FilmGradeAugmentor
from utils.image_v2 import save_labeled_grid_v2
from utils.logger import ScalarLogger

from .common import (
    move,
    prepare,
    resolve_device,
    scalar_dict,
    trainable_parameters,
)


def _mean_bidirectional_psnr_v2(
    values: dict,
) -> float:
    return 0.5 * (
        float(values["psnr_a_to_b"])
        + float(values["psnr_b_to_a"])
    )


def _add_quality_metrics_v2(
    values: dict,
    pair_result,
    synthetic_style_a: torch.Tensor,
    synthetic_style_b: torch.Tensor,
) -> None:
    output_ab = pair_result[
        "output_ab"
    ].clamp(0.0, 1.0)
    output_ba = pair_result[
        "output_ba"
    ].clamp(0.0, 1.0)

    canonical_a = pair_result[
        "state_a"
    ]["canonical"].clamp(0.0, 1.0)
    canonical_b = pair_result[
        "state_b"
    ]["canonical"].clamp(0.0, 1.0)

    image_a = synthetic_style_a.clamp(
        0.0,
        1.0,
    )
    image_b = synthetic_style_b.clamp(
        0.0,
        1.0,
    )

    input_psnr_a_to_b = float(
        psnr(image_a, image_b)
    )
    input_psnr_b_to_a = float(
        psnr(image_b, image_a)
    )

    output_psnr_a_to_b = float(
        psnr(output_ab, image_b)
    )
    output_psnr_b_to_a = float(
        psnr(output_ba, image_a)
    )

    values.update(
        {
            "input_psnr_a_to_b": (
                input_psnr_a_to_b
            ),
            "psnr_a_to_b": (
                output_psnr_a_to_b
            ),
            "psnr_gain_a_to_b": (
                output_psnr_a_to_b
                - input_psnr_a_to_b
            ),
            "input_psnr_b_to_a": (
                input_psnr_b_to_a
            ),
            "psnr_b_to_a": (
                output_psnr_b_to_a
            ),
            "psnr_gain_b_to_a": (
                output_psnr_b_to_a
                - input_psnr_b_to_a
            ),
            "canonical_psnr": float(
                psnr(
                    canonical_a,
                    canonical_b,
                )
            ),
        }
    )

    values["mean_bidirectional_psnr"] = (
        _mean_bidirectional_psnr_v2(values)
    )


def _average_scalar_dicts_v2(
    totals: dict[str, float],
    count: int,
) -> dict[str, float]:
    if count <= 0:
        raise ValueError(
            "count must be positive when averaging metrics"
        )

    return {
        key: float(value) / float(count)
        for key, value in totals.items()
    }


def _accumulate_scalars_v2(
    totals: dict[str, float],
    values: dict[str, float],
) -> None:
    for key, value in values.items():
        numeric_value = float(value)

        if math.isfinite(numeric_value):
            totals[key] = (
                totals.get(key, 0.0)
                + numeric_value
            )


def _lut_regularization_indicator_v2(
    values: dict[str, float],
) -> float:
    """
    A diagnostic indicator only.

    It combines both branches' monotonicity and range penalties. It is not
    used as an automatic stopping target because the absolute scale depends
    on the LUT regularizers and their implementation.
    """
    keys = (
        "normalization_lut_monotonicity",
        "normalization_lut_range",
        "style_lut_monotonicity",
        "style_lut_range",
    )

    return sum(
        max(float(values.get(key, 0.0)), 0.0)
        for key in keys
    )


class EarlyStoppingV2:
    """
    Metric-based Stage-1 V2 early stopping.

    Automatic stopping uses only fixed validation metrics:

    1. Plateau:
       over `plateau_window` epochs, the bidirectional validation PSNR gain is
       smaller than `psnr_min_gain`, canonical distance improves by less than
       `canonical_relative_min_gain`, and canonical active ratio changes by
       less than `active_ratio_max_change`.

    2. Clear regression:
       validation PSNR remains at least `regression_delta` below the best
       observed value for `regression_patience` consecutive epochs.

    Cross-content visual quality cannot be judged reliably by code. The
    arbitrary content/style validation images must still be reviewed before
    selecting the final checkpoint.
    """

    def __init__(self, validation_cfg):
        self.enabled = bool(
            getattr(
                validation_cfg,
                "early_stopping",
                True,
            )
        )

        self.min_epochs = int(
            getattr(
                validation_cfg,
                "early_stop_min_epochs",
                5,
            )
        )

        self.plateau_window = int(
            getattr(
                validation_cfg,
                "early_stop_plateau_window",
                3,
            )
        )

        self.psnr_min_gain = float(
            getattr(
                validation_cfg,
                "early_stop_psnr_min_gain",
                0.15,
            )
        )

        self.canonical_relative_min_gain = float(
            getattr(
                validation_cfg,
                "early_stop_canonical_relative_min_gain",
                0.05,
            )
        )

        self.active_ratio_max_change = float(
            getattr(
                validation_cfg,
                "early_stop_active_ratio_max_change",
                0.03,
            )
        )

        self.regression_delta = float(
            getattr(
                validation_cfg,
                "early_stop_regression_delta",
                0.20,
            )
        )

        self.regression_patience = int(
            getattr(
                validation_cfg,
                "early_stop_regression_patience",
                2,
            )
        )

        self.lut_warning_multiplier = float(
            getattr(
                validation_cfg,
                "lut_warning_multiplier",
                2.0,
            )
        )

        if self.min_epochs < 1:
            raise ValueError(
                "early_stop_min_epochs must be >= 1"
            )

        if self.plateau_window < 1:
            raise ValueError(
                "early_stop_plateau_window must be >= 1"
            )

        if self.regression_patience < 1:
            raise ValueError(
                "early_stop_regression_patience must be >= 1"
            )

        self.history: list[dict[str, float | int]] = []
        self.best_psnr = float("-inf")
        self.best_epoch = -1
        self.regression_count = 0
        self.minimum_lut_indicator = float("inf")

    def state_dict(self) -> dict:
        return {
            "enabled": self.enabled,
            "min_epochs": self.min_epochs,
            "plateau_window": self.plateau_window,
            "psnr_min_gain": self.psnr_min_gain,
            "canonical_relative_min_gain": (
                self.canonical_relative_min_gain
            ),
            "active_ratio_max_change": (
                self.active_ratio_max_change
            ),
            "regression_delta": self.regression_delta,
            "regression_patience": (
                self.regression_patience
            ),
            "lut_warning_multiplier": (
                self.lut_warning_multiplier
            ),
            "best_psnr": self.best_psnr,
            "best_epoch": self.best_epoch,
            "regression_count": self.regression_count,
            "minimum_lut_indicator": (
                self.minimum_lut_indicator
            ),
            "history": self.history,
        }

    def load_state_dict(self, state: dict) -> None:
        self.best_psnr = float(
            state.get(
                "best_psnr",
                self.best_psnr,
            )
        )
        self.best_epoch = int(
            state.get(
                "best_epoch",
                self.best_epoch,
            )
        )
        self.regression_count = int(
            state.get(
                "regression_count",
                self.regression_count,
            )
        )
        self.minimum_lut_indicator = float(
            state.get(
                "minimum_lut_indicator",
                self.minimum_lut_indicator,
            )
        )

        loaded_history = state.get(
            "history",
            [],
        )

        if isinstance(loaded_history, list):
            self.history = loaded_history

    def save(self, path: Path) -> None:
        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        temporary_path = path.with_suffix(
            path.suffix + ".tmp"
        )

        temporary_path.write_text(
            json.dumps(
                self.state_dict(),
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        temporary_path.replace(path)

    def load(self, path: Path) -> None:
        if not path.exists():
            return

        state = json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )
        self.load_state_dict(state)

    def update(
        self,
        epoch: int,
        validation_values: dict[str, float],
    ) -> tuple[bool, str, dict[str, float]]:
        current_psnr = float(
            validation_values[
                "mean_bidirectional_psnr"
            ]
        )
        canonical_distance = float(
            validation_values[
                "canonical_distance"
            ]
        )
        canonical_active_ratio = float(
            validation_values[
                "canonical_active_ratio"
            ]
        )
        lut_indicator = (
            _lut_regularization_indicator_v2(
                validation_values
            )
        )

        if current_psnr > self.best_psnr:
            self.best_psnr = current_psnr
            self.best_epoch = int(epoch)

        if (
            current_psnr
            < self.best_psnr - self.regression_delta
        ):
            self.regression_count += 1
        else:
            self.regression_count = 0

        self.minimum_lut_indicator = min(
            self.minimum_lut_indicator,
            lut_indicator,
        )

        record = {
            "epoch": int(epoch),
            "mean_bidirectional_psnr": current_psnr,
            "canonical_distance": canonical_distance,
            "canonical_active_ratio": (
                canonical_active_ratio
            ),
            "lut_regularization_indicator": (
                lut_indicator
            ),
        }
        self.history.append(record)

        diagnostics = {
            "early_stop_best_psnr": self.best_psnr,
            "early_stop_best_epoch": float(
                self.best_epoch
            ),
            "early_stop_regression_count": float(
                self.regression_count
            ),
            "early_stop_window_psnr_gain": float(
                "nan"
            ),
            "early_stop_window_canonical_relative_gain": (
                float("nan")
            ),
            "early_stop_window_active_ratio_change": (
                float("nan")
            ),
            "early_stop_plateau_detected": 0.0,
            "early_stop_lut_warning": 0.0,
        }

        if (
            math.isfinite(
                self.minimum_lut_indicator
            )
            and self.minimum_lut_indicator > 0.0
            and lut_indicator
            > (
                self.minimum_lut_indicator
                * self.lut_warning_multiplier
            )
        ):
            diagnostics[
                "early_stop_lut_warning"
            ] = 1.0

        completed_epochs = int(epoch) + 1

        regression_stop = (
            completed_epochs >= self.min_epochs
            and self.regression_count
            >= self.regression_patience
        )

        plateau_stop = False

        required_history = (
            self.plateau_window + 1
        )

        if (
            completed_epochs >= self.min_epochs
            and len(self.history)
            >= required_history
        ):
            reference = self.history[
                -required_history
            ]

            psnr_gain = (
                current_psnr
                - float(
                    reference[
                        "mean_bidirectional_psnr"
                    ]
                )
            )

            reference_distance = max(
                abs(
                    float(
                        reference[
                            "canonical_distance"
                        ]
                    )
                ),
                1e-12,
            )

            canonical_relative_gain = (
                float(
                    reference[
                        "canonical_distance"
                    ]
                )
                - canonical_distance
            ) / reference_distance

            active_ratio_change = abs(
                canonical_active_ratio
                - float(
                    reference[
                        "canonical_active_ratio"
                    ]
                )
            )

            plateau_detected = (
                psnr_gain
                < self.psnr_min_gain
                and canonical_relative_gain
                < self.canonical_relative_min_gain
                and active_ratio_change
                < self.active_ratio_max_change
            )

            diagnostics.update(
                {
                    "early_stop_window_psnr_gain": (
                        psnr_gain
                    ),
                    "early_stop_window_canonical_relative_gain": (
                        canonical_relative_gain
                    ),
                    "early_stop_window_active_ratio_change": (
                        active_ratio_change
                    ),
                    "early_stop_plateau_detected": (
                        1.0
                        if plateau_detected
                        else 0.0
                    ),
                }
            )

            plateau_stop = plateau_detected

        if not self.enabled:
            return (
                False,
                "early stopping disabled",
                diagnostics,
            )

        if regression_stop:
            reason = (
                "validation reconstruction regressed for "
                f"{self.regression_count} consecutive epoch(s): "
                f"current={current_psnr:.4f} dB, "
                f"best={self.best_psnr:.4f} dB at "
                f"epoch {self.best_epoch}"
            )
            return True, reason, diagnostics

        if plateau_stop:
            reason = (
                "validation entered a joint plateau over "
                f"{self.plateau_window} epoch(s): "
                f"PSNR gain="
                f"{diagnostics['early_stop_window_psnr_gain']:.4f} dB, "
                f"canonical relative gain="
                f"{diagnostics['early_stop_window_canonical_relative_gain']:.4f}, "
                f"active-ratio change="
                f"{diagnostics['early_stop_window_active_ratio_change']:.4f}"
            )
            return True, reason, diagnostics

        return False, "continue training", diagnostics


@torch.no_grad()
def validate_v2(
    model,
    loader,
    criterion,
    augmentor,
    cfg,
    device,
    epoch,
    paths,
):
    """
    Save three complementary validation views:

    1. Same-content bidirectional reconstruction with GT.
    2. Canonicalization comparison for the two source grades.
    3. Different-content qualitative transfer:
       content + unrelated style -> stylized, with no GT.
    """
    model.eval()

    totals = {}
    number_of_batches = 0

    reconstruction_visual = None
    canonical_visual = None
    arbitrary_visual = None

    # Keep synthetic validation grades and arbitrary pairings fixed across
    # epochs so numerical and visual changes remain directly comparable.
    validation_seed = int(
        getattr(
            cfg.validation,
            "seed",
            12345,
        )
    )
    torch.manual_seed(validation_seed)

    for batch in islice(
        loader,
        int(cfg.data.val_steps),
    ):
        content = move(batch[0], device)

        synthetic_style_a, synthetic_style_b = (
            augmentor.two_views(
                content,
                shared=False,
            )
        )

        pair_result = model.forward_pair(
            synthetic_style_a,
            synthetic_style_b,
        )

        losses = criterion(
            pair_result=pair_result,
            image_a=synthetic_style_a,
            image_b=synthetic_style_b,
        )

        values = scalar_dict(losses)

        _add_quality_metrics_v2(
            values=values,
            pair_result=pair_result,
            synthetic_style_a=synthetic_style_a,
            synthetic_style_b=synthetic_style_b,
        )

        _accumulate_scalars_v2(
            totals,
            values,
        )

        number_of_batches += 1

        if reconstruction_visual is None:
            reconstruction_visual = (
                content,
                synthetic_style_a,
                synthetic_style_b,
                pair_result["output_ab"],
                synthetic_style_b,
                pair_result["output_ba"],
                synthetic_style_a,
            )

            canonical_visual = (
                synthetic_style_a,
                pair_result[
                    "state_a"
                ]["canonical"],
                synthetic_style_b,
                pair_result[
                    "state_b"
                ]["canonical"],
            )

            if content.shape[0] >= 2:
                # Roll the batch so every style reference comes from a
                # different source image.
                style_indices = torch.roll(
                    torch.arange(
                        content.shape[0],
                        device=content.device,
                    ),
                    shifts=1,
                    dims=0,
                )

                arbitrary_content = (
                    synthetic_style_a
                )
                arbitrary_style = (
                    synthetic_style_b[
                        style_indices
                    ]
                )

                arbitrary_result = model(
                    arbitrary_content,
                    arbitrary_style,
                )

                arbitrary_visual = (
                    arbitrary_content,
                    arbitrary_style,
                    arbitrary_result["output"],
                )

    if number_of_batches == 0:
        raise RuntimeError(
            "Stage-1 V2 validation loader produced no batches."
        )

    averages = _average_scalar_dicts_v2(
        totals,
        number_of_batches,
    )

    maximum_visual_items = int(
        getattr(
            cfg.validation,
            "max_visual_items",
            2,
        )
    )

    if reconstruction_visual is not None:
        save_labeled_grid_v2(
            tensors=reconstruction_visual,
            path=(
                paths["images"]
                / (
                    "val_stage1_v2_reconstruction_"
                    f"epoch_{epoch:04d}.png"
                )
            ),
            labels=[
                "content",
                "synthetic_style_A_on_same_content",
                "synthetic_style_B_on_same_content",
                "style_A_predict_style_B",
                "ground_truth_style_B",
                "style_B_predict_style_A",
                "ground_truth_style_A",
            ],
            title=(
                "Stage 1 V2: bidirectional reconstruction "
                f"- epoch {epoch}"
            ),
            max_items=maximum_visual_items,
        )

    if canonical_visual is not None:
        save_labeled_grid_v2(
            tensors=canonical_visual,
            path=(
                paths["images"]
                / (
                    "val_stage1_v2_canonical_"
                    f"epoch_{epoch:04d}.png"
                )
            ),
            labels=[
                "synthetic_style_A",
                "canonical_A",
                "synthetic_style_B",
                "canonical_B",
            ],
            title=(
                "Stage 1 V2: same content mapped into "
                f"canonical space - epoch {epoch}"
            ),
            max_items=maximum_visual_items,
        )

    if arbitrary_visual is not None:
        save_labeled_grid_v2(
            tensors=arbitrary_visual,
            path=(
                paths["images"]
                / (
                    "val_stage1_v2_arbitrary_transfer_"
                    f"epoch_{epoch:04d}.png"
                )
            ),
            labels=[
                "content",
                "style",
                "stylized",
            ],
            title=(
                "Stage 1 V2: different-content style transfer "
                f"(qualitative, no GT) - epoch {epoch}"
            ),
            max_items=maximum_visual_items,
        )
    else:
        print(
            "[Stage 1 V2] Arbitrary content-style validation "
            "was skipped because validation batch_size < 2."
        )

    return averages


def run(cfg):
    device = resolve_device(cfg.device)
    paths = prepare(cfg)

    model = AriadneLUTV2(cfg.model).to(device)
    criterion = Stage1LossV2(cfg).to(device)
    augmentor = FilmGradeAugmentor(
        cfg.augmentation
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.train.learning_rate),
        weight_decay=float(
            cfg.train.weight_decay
        ),
    )

    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(cfg.train.epochs),
        )
    )

    amp_enabled = (
        bool(cfg.train.amp)
        and device.type == "cuda"
    )

    scaler = GradScaler(
        "cuda",
        enabled=amp_enabled,
    )

    logger = ScalarLogger(paths["logs"])

    start_epoch = 0
    global_step = 0
    best_metric = float("-inf")

    early_stopper = EarlyStoppingV2(
        cfg.validation
    )
    early_stopping_state_path = (
        Path(paths["logs"])
        / "early_stopping_v2.json"
    )

    if cfg.train.resume:
        checkpoint = load_checkpoint(
            cfg.train.resume,
            model,
            optimizer,
            scheduler,
            scaler,
            map_location=device,
        )

        start_epoch = int(
            checkpoint.get("epoch", -1)
        ) + 1
        global_step = int(
            checkpoint.get(
                "global_step",
                0,
            )
        )
        best_metric = float(
            checkpoint.get(
                "best_metric",
                best_metric,
            )
        )

        early_stopper.load(
            early_stopping_state_path
        )

    print(
        "Stage 1 V2 trainable parameters: "
        f"{trainable_parameters(model):,}"
    )

    print(
        "Stage 1 V2 early stopping:",
        f"enabled={early_stopper.enabled},",
        f"min_epochs={early_stopper.min_epochs},",
        f"plateau_window={early_stopper.plateau_window},",
        f"psnr_min_gain={early_stopper.psnr_min_gain:.3f} dB,",
        "canonical_relative_min_gain="
        f"{early_stopper.canonical_relative_min_gain:.3f},",
        "regression="
        f"{early_stopper.regression_delta:.3f} dB x "
        f"{early_stopper.regression_patience} epochs",
    )

    train_loader = get_coco_loader_v2(
        cfg,
        "train",
    )
    validation_loader = get_coco_loader_v2(
        cfg,
        "val",
    )

    try:
        for epoch in range(
            start_epoch,
            int(cfg.train.epochs),
        ):
            model.train()

            progress = tqdm(
                islice(
                    train_loader,
                    int(
                        cfg.data.steps_per_epoch
                    ),
                ),
                total=int(
                    cfg.data.steps_per_epoch
                ),
                desc=f"Stage1V2 {epoch:03d}",
            )

            seen_batches = 0
            train_totals: dict[str, float] = {}

            for batch in progress:
                seen_batches += 1
                content = move(
                    batch[0],
                    device,
                )

                synthetic_style_a, synthetic_style_b = (
                    augmentor.two_views(
                        content,
                        shared=False,
                    )
                )

                optimizer.zero_grad(
                    set_to_none=True
                )

                with autocast(
                    device_type=device.type,
                    enabled=amp_enabled,
                ):
                    pair_result = (
                        model.forward_pair(
                            synthetic_style_a,
                            synthetic_style_b,
                        )
                    )

                    losses = criterion(
                        pair_result=pair_result,
                        image_a=synthetic_style_a,
                        image_b=synthetic_style_b,
                    )

                scaler.scale(
                    losses["total"]
                ).backward()

                scaler.unscale_(optimizer)

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    float(cfg.train.grad_clip),
                )

                scaler.step(optimizer)
                scaler.update()

                global_step += 1

                values = scalar_dict(losses)

                _add_quality_metrics_v2(
                    values=values,
                    pair_result=pair_result,
                    synthetic_style_a=(
                        synthetic_style_a
                    ),
                    synthetic_style_b=(
                        synthetic_style_b
                    ),
                )

                _accumulate_scalars_v2(
                    train_totals,
                    values,
                )

                progress.set_postfix(
                    total=(
                        f"{values['total']:.4f}"
                    ),
                    a2b=(
                        f"{values['psnr_a_to_b']:.2f}"
                    ),
                    b2a=(
                        f"{values['psnr_b_to_a']:.2f}"
                    ),
                    canonical=(
                        f"{values['canonical_psnr']:.2f}"
                    ),
                )

                if (
                    global_step
                    % int(cfg.train.log_every)
                    == 0
                ):
                    values["lr"] = (
                        optimizer
                        .param_groups[0]["lr"]
                    )
                    logger.log(
                        values,
                        global_step,
                        "train",
                    )

                if (
                    global_step
                    % int(cfg.train.image_every)
                    == 0
                ):
                    save_labeled_grid_v2(
                        tensors=(
                            content,
                            synthetic_style_a,
                            synthetic_style_b,
                            pair_result[
                                "output_ab"
                            ],
                            synthetic_style_b,
                            pair_result[
                                "output_ba"
                            ],
                            synthetic_style_a,
                        ),
                        path=(
                            paths["images"]
                            / (
                                "train_stage1_v2_"
                                f"step_{global_step:08d}.png"
                            )
                        ),
                        labels=[
                            "content",
                            "synthetic_style_A_on_same_content",
                            "synthetic_style_B_on_same_content",
                            "style_A_predict_style_B",
                            "ground_truth_style_B",
                            "style_B_predict_style_A",
                            "ground_truth_style_A",
                        ],
                        title=(
                            "Stage 1 V2 training "
                            f"- step {global_step}"
                        ),
                        max_items=int(
                            getattr(
                                cfg.validation,
                                "max_visual_items",
                                2,
                            )
                        ),
                    )

                    save_labeled_grid_v2(
                        tensors=(
                            synthetic_style_a,
                            pair_result[
                                "state_a"
                            ]["canonical"],
                            synthetic_style_b,
                            pair_result[
                                "state_b"
                            ]["canonical"],
                        ),
                        path=(
                            paths["images"]
                            / (
                                "train_stage1_v2_canonical_"
                                f"step_{global_step:08d}.png"
                            )
                        ),
                        labels=[
                            "synthetic_style_A",
                            "canonical_A",
                            "synthetic_style_B",
                            "canonical_B",
                        ],
                        title=(
                            "Stage 1 V2 canonicalization "
                            f"- step {global_step}"
                        ),
                        max_items=int(
                            getattr(
                                cfg.validation,
                                "max_visual_items",
                                2,
                            )
                        ),
                    )

            if seen_batches == 0:
                raise RuntimeError(
                    "Stage-1 V2 training loader produced no batches."
                )

            train_epoch_values = (
                _average_scalar_dicts_v2(
                    train_totals,
                    seen_batches,
                )
            )

            scheduler.step()

            validation_values = validate_v2(
                model=model,
                loader=validation_loader,
                criterion=criterion,
                augmentor=augmentor,
                cfg=cfg,
                device=device,
                epoch=epoch,
                paths=paths,
            )

            train_mean_psnr = float(
                train_epoch_values[
                    "mean_bidirectional_psnr"
                ]
            )
            validation_mean_psnr = float(
                validation_values[
                    "mean_bidirectional_psnr"
                ]
            )

            validation_values[
                "train_epoch_mean_bidirectional_psnr"
            ] = train_mean_psnr

            validation_values[
                "train_val_psnr_gap"
            ] = (
                train_mean_psnr
                - validation_mean_psnr
            )

            (
                should_stop,
                stopping_reason,
                stopping_diagnostics,
            ) = early_stopper.update(
                epoch=epoch,
                validation_values=validation_values,
            )

            validation_values.update(
                stopping_diagnostics
            )

            logger.log(
                validation_values,
                global_step,
                "val",
            )

            print(
                "Validation V2:",
                " ".join(
                    f"{key}={value:.5f}"
                    for key, value
                    in validation_values.items()
                    if math.isfinite(float(value))
                ),
            )

            print(
                "Early stopping V2:",
                stopping_reason,
            )

            if (
                stopping_diagnostics[
                    "early_stop_lut_warning"
                ]
                > 0.5
            ):
                print(
                    "[Stage 1 V2 warning] LUT monotonicity/range "
                    "indicator is more than "
                    f"{early_stopper.lut_warning_multiplier:.2f}x "
                    "its best observed level. Inspect the current "
                    "LUT outputs and arbitrary-transfer images."
                )

            checkpoint_metric = (
                validation_mean_psnr
            )

            if checkpoint_metric > best_metric:
                best_metric = checkpoint_metric

                save_checkpoint(
                    paths["checkpoints"]
                    / "best.pth",
                    model,
                    optimizer,
                    scheduler,
                    scaler,
                    epoch,
                    global_step,
                    best_metric,
                    extra={
                        "stage": "stage1_v2",
                        "canonical_psnr": (
                            validation_values[
                                "canonical_psnr"
                            ]
                        ),
                        "canonical_distance": (
                            validation_values[
                                "canonical_distance"
                            ]
                        ),
                        "mean_bidirectional_psnr": (
                            validation_mean_psnr
                        ),
                        "train_val_psnr_gap": (
                            validation_values[
                                "train_val_psnr_gap"
                            ]
                        ),
                    },
                )

            save_checkpoint(
                paths["checkpoints"]
                / "latest.pth",
                model,
                optimizer,
                scheduler,
                scaler,
                epoch,
                global_step,
                best_metric,
                extra={
                    "stage": "stage1_v2",
                    "canonical_psnr": (
                        validation_values[
                            "canonical_psnr"
                        ]
                    ),
                    "canonical_distance": (
                        validation_values[
                            "canonical_distance"
                        ]
                    ),
                    "mean_bidirectional_psnr": (
                        validation_mean_psnr
                    ),
                    "train_val_psnr_gap": (
                        validation_values[
                            "train_val_psnr_gap"
                        ]
                    ),
                    "early_stopping_reason": (
                        stopping_reason
                    ),
                },
            )

            if (
                (epoch + 1)
                % int(
                    cfg.train.checkpoint_every
                )
                == 0
            ):
                save_checkpoint(
                    paths["checkpoints"]
                    / f"epoch_{epoch:04d}.pth",
                    model,
                    optimizer,
                    scheduler,
                    scaler,
                    epoch,
                    global_step,
                    best_metric,
                    extra={
                        "stage": "stage1_v2",
                        "canonical_psnr": (
                            validation_values[
                                "canonical_psnr"
                            ]
                        ),
                        "canonical_distance": (
                            validation_values[
                                "canonical_distance"
                            ]
                        ),
                        "mean_bidirectional_psnr": (
                            validation_mean_psnr
                        ),
                        "train_val_psnr_gap": (
                            validation_values[
                                "train_val_psnr_gap"
                            ]
                        ),
                        "early_stopping_reason": (
                            stopping_reason
                        ),
                    },
                )

            early_stopper.save(
                early_stopping_state_path
            )

            if should_stop:
                print(
                    "\n[Stage 1 V2] Training stopped early."
                )
                print(
                    f"Reason: {stopping_reason}"
                )
                print(
                    "Use checkpoints/best.pth as the primary "
                    "quantitative checkpoint, then compare its "
                    "canonical and arbitrary-transfer validation "
                    "images before making the final selection."
                )
                break
    finally:
        logger.close()
