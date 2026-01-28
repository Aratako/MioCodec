from __future__ import annotations

import json5
from pathlib import Path
from typing import Iterable


class JsonHParams:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            if isinstance(value, dict):
                value = JsonHParams(**value)
            self[key] = value

    def keys(self):
        return self.__dict__.keys()

    def items(self):
        return self.__dict__.items()

    def values(self):
        return self.__dict__.values()

    def __len__(self):
        return len(self.__dict__)

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        return setattr(self, key, value)

    def __contains__(self, key):
        return key in self.__dict__

    def __repr__(self):
        return self.__dict__.__repr__()


DEFAULT_PUPU_CONFIG = {
    "preprocess": {"n_mel": 128},
    "model": {
        "pupuvocoder": {
            "resblock": "1",
            "upsample_rates": [8, 8, 2, 2, 2],
            "upsample_kernel_sizes": [16, 16, 4, 4, 4],
            "upsample_initial_channel": 512,
            "resblock_kernel_sizes": [3, 7, 11],
            "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        }
    },
}

def override_config(base_config: dict, new_config: dict) -> dict:
    for key, value in new_config.items():
        if isinstance(value, dict):
            if key not in base_config:
                base_config[key] = {}
            base_config[key] = override_config(base_config[key], value)
        else:
            base_config[key] = value
    return base_config


def get_lowercase_keys_config(cfg: dict) -> dict:
    updated_cfg = {}
    for key, value in cfg.items():
        if isinstance(value, dict):
            value = get_lowercase_keys_config(value)
        updated_cfg[key.lower()] = value
    return updated_cfg


def _iter_search_roots(config_path: Path, search_roots: Iterable[Path]) -> Iterable[Path]:
    yielded: set[Path] = set()
    for parent in config_path.parents:
        if parent not in yielded:
            yielded.add(parent)
            yield parent
    for root in search_roots:
        if root not in yielded:
            yielded.add(root)
            yield root


def _resolve_base_config(
    base_config: str, config_path: Path, search_roots: Iterable[Path]
) -> Path:
    candidate = Path(base_config)
    if candidate.is_absolute() and candidate.exists():
        return candidate

    candidate = config_path.parent / base_config
    if candidate.exists():
        return candidate

    for root in _iter_search_roots(config_path, search_roots):
        candidate = root / base_config
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"base_config '{base_config}' not found relative to {config_path} or search roots."
    )


def _load_config(
    config_path: Path, lowercase: bool, search_roots: Iterable[Path]
) -> dict:
    data = json5.loads(config_path.read_text())
    if "base_config" in data:
        base_path = _resolve_base_config(data["base_config"], config_path, search_roots)
        base_config = _load_config(base_path, lowercase=lowercase, search_roots=search_roots)
        data = override_config(base_config, data)
    if lowercase:
        data = get_lowercase_keys_config(data)
    return data


def load_config(
    config_path: str | Path,
    lowercase: bool = False,
    search_roots: Iterable[str | Path] = (),
) -> JsonHParams:
    path = Path(config_path)
    roots = [Path(root) for root in search_roots if root]
    config = _load_config(path, lowercase=lowercase, search_roots=roots)
    return JsonHParams(**config)
