from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.metrics import accuracy_score, balanced_accuracy_score, log_loss
from sklearn.model_selection import GroupShuffleSplit


def developer_components(data: pd.DataFrame) -> np.ndarray:
    owners = {}
    edges = []
    for i, value in enumerate(data.developer_group.astype(str)):
        for developer in value.split("|"):
            developer = developer.strip().casefold()
            if developer in owners:
                edges.extend([(i, owners[developer]), (owners[developer], i)])
            else:
                owners[developer] = i
    if edges:
        row, column = zip(*edges)
        graph = coo_matrix((np.ones(len(edges)), (row, column)), shape=(len(data), len(data)))
    else:
        graph = coo_matrix((len(data), len(data)))
    return connected_components(graph, directed=False)[1]


def split_games(data: pd.DataFrame, config: dict) -> dict[str, np.ndarray]:
    settings, seed = config["evaluation"], config["seed"]
    def divide(indices, fraction, offset):
        local = data.iloc[indices]
        a, b = next(GroupShuffleSplit(n_splits=1, test_size=fraction, random_state=seed + offset)
                    .split(local, groups=local.split_group))
        return indices[a], indices[b]
    remaining, test = divide(np.arange(len(data)), settings["test_fraction"], 0)
    remaining, validation = divide(remaining, settings["validation_fraction"] / (1 - settings["test_fraction"]), 1)
    train, calibration = divide(remaining, settings["calibration_fraction"] /
                                 (1 - settings["test_fraction"] - settings["validation_fraction"]), 2)
    parts = {"train": train, "validation": validation, "calibration": calibration, "test": test}
    sets = [set(data.iloc[indices].split_group) for indices in parts.values()]
    if any(sets[i] & sets[j] for i in range(4) for j in range(i + 1, 4)):
        raise AssertionError("Developer leakage between independent experiment parts")
    return parts


def observation_weights(data: pd.DataFrame) -> np.ndarray:
    counts = data.groupby("split_group").split_group.transform("size").to_numpy()
    weights = 1 / counts
    return weights * len(weights) / weights.sum()


def row_log_loss(y, probabilities):
    return -np.log(np.clip(probabilities[np.arange(len(y)), np.asarray(y, dtype=int)], 1e-12, 1))


def classification_metrics(data: pd.DataFrame, probabilities: np.ndarray) -> dict:
    y = data.owners_band.to_numpy(dtype=int)
    errors = pd.DataFrame({"group": data.split_group.to_numpy(), "loss": row_log_loss(y, probabilities)})
    one_hot = np.eye(4)[y]
    return {"log_loss": float(log_loss(y, probabilities, labels=list(range(4)))),
            "group_log_loss": float(errors.groupby("group").loss.mean().mean()),
            "accuracy": float(accuracy_score(y, probabilities.argmax(axis=1))),
            "balanced_accuracy": float(balanced_accuracy_score(y, probabilities.argmax(axis=1))),
            "brier": float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))}


def paired_bootstrap(data: pd.DataFrame, additive: np.ndarray, interactions: np.ndarray, seed: int, repeats: int) -> dict:
    y = data.owners_band.to_numpy(dtype=int)
    frame = pd.DataFrame({"group": data.split_group.to_numpy(),
                         "difference": row_log_loss(y, additive) - row_log_loss(y, interactions)})
    differences = frame.groupby("group").difference.mean().to_numpy()
    rng = np.random.default_rng(seed)
    draws = np.array([rng.choice(differences, len(differences), replace=True).mean() for _ in range(repeats)])
    low, high = np.quantile(draws, [0.025, 0.975])
    return {"delta_log_loss": float(differences.mean()), "ci_low": float(low), "ci_high": float(high),
            "unit": "connected developer group", "groups": len(differences), "repeats": repeats,
            "comparison": "matched HGB additive vs interactions after independent calibration",
            "verdict": "supported" if low > 0 else "interactions_worse" if high < 0 else "inconclusive"}


def threshold_reliability(data: pd.DataFrame, probabilities: np.ndarray, bins=10) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fixed-bin, game-weighted calibration diagnostics, not a model selection rule."""
    if bins < 2 or len(data) == 0 or probabilities.shape != (len(data), 4):
        raise ValueError("Expected >=2 bins and one four-band distribution per game")
    if not np.isfinite(probabilities).all() or (probabilities < 0).any() or (probabilities > 1).any() or not np.allclose(probabilities.sum(axis=1), 1):
        raise ValueError("Invalid probability distributions")
    rows, summaries = [], []
    for band, threshold in [(1, 20000), (2, 100000), (3, 500000)]:
        p = probabilities[:, band:].sum(axis=1)
        actual = data.owners_band.to_numpy(dtype=int) >= band
        positions = np.minimum((p * bins).astype(int), bins - 1)
        gaps = []
        for index in range(bins):
            chosen = positions == index
            n = int(chosen.sum())
            predicted = float(p[chosen].mean()) if n else np.nan
            observed = float(actual[chosen].mean()) if n else np.nan
            if n:
                gaps.append(n * abs(predicted - observed))
            rows.append({"threshold": threshold, "bin": index, "bin_lower": index / bins,
                         "bin_upper": (index + 1) / bins, "games": n,
                         "developer_groups": int(data.loc[chosen, "split_group"].nunique()),
                         "mean_probability": predicted, "observed_fraction": observed})
        summaries.append({"threshold": threshold, "games": len(data), "bins": bins,
                          "ece": float(sum(gaps) / len(data)), "binary_brier": float(np.mean((p - actual) ** 2))})
    return pd.DataFrame(rows), pd.DataFrame(summaries)
