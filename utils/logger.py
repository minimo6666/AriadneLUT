from __future__ import annotations

from pathlib import Path
import csv


class ScalarLogger:
    """CSV scalar logger that can safely add new metric columns mid-run."""

    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _read_existing(path: Path) -> tuple[list[str], list[dict[str, str]]]:
        if not path.exists() or path.stat().st_size == 0:
            return [], []
        with path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            return list(reader.fieldnames or []), list(reader)

    @staticmethod
    def _rewrite(
        path: Path,
        fieldnames: list[str],
        rows: list[dict[str, object]],
    ) -> None:
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=fieldnames,
                extrasaction="ignore",
            )
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
        temporary.replace(path)

    def log(self, values: dict, step: int, split: str):
        path = self.root / f"{split}.csv"
        row = {
            "step": int(step),
            **{key: float(value) for key, value in values.items()},
        }

        existing_fields, existing_rows = self._read_existing(path)
        if not existing_fields:
            self._rewrite(path, list(row.keys()), [row])
            return

        new_fields = [key for key in row.keys() if key not in existing_fields]
        if new_fields:
            fields = existing_fields + new_fields
            self._rewrite(path, fields, [*existing_rows, row])
            return

        with path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=existing_fields,
                extrasaction="ignore",
            )
            writer.writerow(row)

    def close(self):
        pass
