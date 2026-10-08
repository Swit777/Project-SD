from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any


DEFAULT_CONFIG: dict[str, Any] = {
    "generation": {
        "seed_start": 1,
        "n_seeds": 90,
        "sizes": [8, 10, 12],
        "obstacle_densities": [0.08, 0.16, 0.24, 0.32, 0.40, 0.48],
        "agents": ["random", "greedy", "bfs"],
        "random_episodes": 6,
        "max_steps_factor": 2.7,
    },
    "training": {
        "test_size": 0.25,
        "random_state": 42,
    },
}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    if path is None:
        return copy.deepcopy(DEFAULT_CONFIG)

    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as fh:
        user_config = json.load(fh)

    return _deep_merge(DEFAULT_CONFIG, user_config)

