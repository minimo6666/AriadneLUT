from __future__ import annotations

from pathlib import Path
import glob


def find_tars(root: str, split: str) -> list[str]:
    candidates = [
        str(Path(root) / split / "*.tar"),
        str(Path(root) / "*.tar"),
    ]
    for pattern in candidates:
        files = sorted(glob.glob(pattern))
        if files:
            return files
    raise FileNotFoundError(
        "No WebDataset shards found. Checked: " + ", ".join(candidates)
    )
