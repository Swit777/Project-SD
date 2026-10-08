from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit

from .features import TARGET


def split_players(data: pd.DataFrame, seed: int, test_fraction: float, validation_fraction: float) -> dict[str, np.ndarray]:
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_fraction, random_state=seed)
    development, test = next(splitter.split(data, groups=data["player_id"]))
    validation_split = GroupShuffleSplit(n_splits=1, test_size=validation_fraction / (1 - test_fraction), random_state=seed + 1)
    train_local, val_local = next(validation_split.split(data.iloc[development], groups=data.iloc[development]["player_id"]))
    split = {"train": development[train_local], "validation": development[val_local], "test": test}
    player_sets = {key: set(data.iloc[value]["player_id"]) for key, value in split.items()}
    if any(player_sets[a] & player_sets[b] for a, b in [("train", "validation"), ("train", "test"), ("validation", "test")]):
        raise AssertionError("A player occurs in multiple splits")
    return split


def player_weights(data: pd.DataFrame) -> np.ndarray:
    counts = data.groupby("player_id")["player_id"].transform("size").to_numpy()
    weights = 1.0 / counts
    return weights * len(weights) / weights.sum()


def regression_metrics(data: pd.DataFrame, prediction: np.ndarray) -> dict[str, float]:
    y = data[TARGET].to_numpy(dtype=float)
    errors = pd.DataFrame({"player_id": data["player_id"].to_numpy(), "error": np.abs(y - prediction)})
    return {"mae": float(mean_absolute_error(y, prediction)),
            "player_mae": float(errors.groupby("player_id")["error"].mean().mean()),
            "rmse": float(np.sqrt(mean_squared_error(y, prediction))),
            "r2": float(r2_score(y, prediction))}


def paired_player_bootstrap(data: pd.DataFrame, baseline: np.ndarray, dynamic: np.ndarray, seed: int, repeats: int) -> dict:
    y = data[TARGET].to_numpy(dtype=float)
    errors = pd.DataFrame({"player_id": data["player_id"].to_numpy(),
                           "baseline": np.abs(y - baseline), "dynamic": np.abs(y - dynamic)})
    per_player = errors.groupby("player_id")[["baseline", "dynamic"]].mean()
    delta = (per_player["baseline"] - per_player["dynamic"]).to_numpy()
    rng = np.random.default_rng(seed)
    samples = np.array([rng.choice(delta, len(delta), replace=True).mean() for _ in range(repeats)])
    low, high = np.quantile(samples, [0.025, 0.975])
    verdict = "supported" if low > 0 else "dynamic_worse" if high < 0 else "inconclusive"
    return {"comparison": "random_forest_aggregate vs random_forest_dynamic",
            "unit": "player", "delta_definition": "aggregate player-MAE minus dynamic player-MAE",
            "delta_mae": float(delta.mean()), "ci_low": float(low), "ci_high": float(high),
            "confidence": 0.95, "bootstrap_repeats": repeats, "players": len(delta), "verdict": verdict}
