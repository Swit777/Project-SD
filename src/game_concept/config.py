from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs/game_concept.json"


def load_config(path=DEFAULT_CONFIG):
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if config["sample_games"] < 100 or config["minimum_age_days"] < 0:
        raise ValueError("At least 100 games and nonnegative game ages are required")
    if config["request_interval_seconds"] < 1.5 or not 0 <= config["scrape_games"] <= 1000:
        raise ValueError("Use a polite request interval >= 1.5 seconds and a bounded scrape size")
    fractions = [config["evaluation"][k] for k in ["test_fraction", "validation_fraction", "calibration_fraction"]]
    if any(not 0 < value < 0.4 for value in fractions) or sum(fractions) >= 0.8:
        raise ValueError("Invalid independent experiment split fractions")
    if config["evaluation"]["iterations"] < 2 or config["evaluation"]["bootstrap_repeats"] < 1:
        raise ValueError("Invalid model or bootstrap settings")
    if not 1 <= config["review_sample_size"] <= 100 or config["evaluation"]["min_samples_leaf"] < 2:
        raise ValueError("Review samples must be 1..100 and tree leaves require at least 2 games")
    return config
