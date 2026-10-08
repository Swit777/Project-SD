from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs" / "player_experience.json"


def load_config(path: Path | str = DEFAULT_CONFIG) -> dict:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    sample = config["sampling"]
    evaluation = config["evaluation"]
    if sample["max_players"] < 12 or sample["min_responses"] < 1:
        raise ValueError("Sampling requires at least 12 players and positive min_responses")
    if sample["max_responses_per_player"] < sample["min_responses"]:
        raise ValueError("max_responses_per_player must be >= min_responses")
    windows = config["features"]["windows_minutes"]
    if not windows or len(set(windows)) != len(windows) or any(not isinstance(w, int) or w <= 0 for w in windows):
        raise ValueError("Feature windows must be distinct positive integer minutes")
    if not 0 < evaluation["test_fraction"] < 0.5 or not 0 < evaluation["validation_fraction"] < 0.5:
        raise ValueError("Test and validation fractions must be between 0 and 0.5")
    if evaluation["test_fraction"] + evaluation["validation_fraction"] >= 0.8:
        raise ValueError("At least 20% of players must remain in training")
    for key in ["cv_folds", "bootstrap_repeats", "trees", "min_samples_leaf"]:
        if not isinstance(evaluation[key], int) or evaluation[key] < (2 if key == "cv_folds" else 1):
            raise ValueError(f"Invalid evaluation setting: {key}")
    if not isinstance(config["seed"], int) or not 0 <= config["seed"] < 2**32 - 1:
        raise ValueError("seed must be a non-negative 32-bit integer")
    return config
