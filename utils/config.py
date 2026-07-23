from __future__ import annotations

from pathlib import Path
import yaml


class AttrDict(dict):
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    def __setattr__(self, key, value):
        self[key] = value


def _convert(value):
    if isinstance(value, dict):
        return AttrDict({key: _convert(item) for key, item in value.items()})
    if isinstance(value, list):
        return [_convert(item) for item in value]
    return value


def load_config(path: str):
    path_obj = Path(path)
    with path_obj.open("r", encoding="utf-8") as handle:
        config = _convert(yaml.safe_load(handle))
    config.config_path = str(path_obj)
    return config
