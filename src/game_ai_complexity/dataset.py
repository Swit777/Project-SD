from __future__ import annotations

from pathlib import Path

import pandas as pd

from .agents import run_agent
from .features import FEATURE_COLUMNS, extract_features
from .grid import generate_level, stable_int_seed


ID_COLUMNS = ["level_id", "seed"]
AGENT_COLUMNS = ["agent"]
TARGET_COLUMNS = ["success", "success_rate", "steps", "reward", "episodes", "difficulty"]
INFERENCE_COLUMNS = ID_COLUMNS + FEATURE_COLUMNS + AGENT_COLUMNS
REQUIRED_COLUMNS = INFERENCE_COLUMNS + TARGET_COLUMNS


def level_id(seed: int, size: int, obstacle_density: float) -> str:
    return f"lvl_{stable_int_seed(seed, size, f'{obstacle_density:.4f}') % 10**12:012d}"


def generate_dataset(config: dict) -> pd.DataFrame:
    gen = config["generation"]
    rows = []
    seed_start = int(gen["seed_start"])
    n_seeds = int(gen["n_seeds"])

    for seed in range(seed_start, seed_start + n_seeds):
        for size in gen["sizes"]:
            for density in gen["obstacle_densities"]:
                max_steps = max(1, int(round(size * float(gen["max_steps_factor"]))))
                level = generate_level(seed, int(size), float(density))
                features = extract_features(level, max_steps)
                base = {
                    "level_id": level_id(seed, int(size), float(density)),
                    "seed": seed,
                    **features,
                }
                for agent in gen["agents"]:
                    result = run_agent(
                        level,
                        agent,
                        max_steps,
                        int(gen["random_episodes"]),
                        int(gen.get("q_learning_episodes", 70)),
                        int(gen.get("q_learning_eval_episodes", 5)),
                    )
                    rows.append({**base, **result.to_row()})

    return pd.DataFrame(rows, columns=REQUIRED_COLUMNS)


def validate_dataset(data: pd.DataFrame, require_target: bool = False) -> list[str]:
    errors = []
    missing = [column for column in REQUIRED_COLUMNS if column not in data.columns]
    if missing:
        errors.append(f"Missing required columns: {', '.join(missing)}")

    if not missing:
        if data.empty:
            errors.append("Dataset is empty")
        if require_target and data["success"].nunique() < 2:
            errors.append("Target column 'success' must contain at least two classes")
        if (data["success_rate"] < 0).any() or (data["success_rate"] > 1).any():
            errors.append("'success_rate' must be between 0 and 1")
        if (data["difficulty"] < 0).any() or (data["difficulty"] > 1).any():
            errors.append("'difficulty' must be between 0 and 1")

    return errors


def validate_inference_dataset(data: pd.DataFrame) -> list[str]:
    errors = []
    missing = [column for column in INFERENCE_COLUMNS if column not in data.columns]
    if missing:
        errors.append(f"Missing required input columns: {', '.join(missing)}")
        return errors

    if data.empty:
        errors.append("Dataset is empty")

    numeric_columns = ["seed"] + FEATURE_COLUMNS
    for column in numeric_columns:
        values = pd.to_numeric(data[column], errors="coerce")
        if values.isna().any():
            errors.append(f"Column '{column}' must be numeric")

    if data["agent"].astype(str).str.strip().eq("").any():
        errors.append("Column 'agent' must not contain empty values")

    return errors


def save_dataset(data: pd.DataFrame, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(path, index=False, encoding="utf-8")
    return path
