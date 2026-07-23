from __future__ import annotations

import argparse

from utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train AriadneLUT Stage 1 V2 without "
            "modifying the original Stage-1 entrypoint."
        )
    )
    parser.add_argument(
        "--config",
        required=True,
    )
    arguments = parser.parse_args()

    cfg = load_config(arguments.config)
    stage = str(cfg.stage).lower()

    if stage != "stage1_v2":
        raise ValueError(
            "train_v2.py currently supports only "
            f"stage1_v2, but received: {cfg.stage}"
        )

    from train_scripts.stage1_v2 import run

    run(cfg)


if __name__ == "__main__":
    main()
