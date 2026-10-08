from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd

from .grid import generate_level, shortest_path


def _level_matrix(seed: int, size: int, density: float) -> np.ndarray:
    level = generate_level(seed, size, density)
    matrix = np.array(level.grid, dtype=float)
    for row, col in shortest_path(level) or []:
        matrix[row, col] = 0.5
    matrix[level.start] = 0.25
    matrix[level.goal] = 0.75
    return matrix


def plot_level_examples(data: pd.DataFrame, output_dir: str | Path) -> Path:
    output = Path(output_dir)
    plots = output / "plots"
    plots.mkdir(parents=True, exist_ok=True)

    level_data = (
        data.groupby("level_id", as_index=False)
        .agg(
            seed=("seed", "first"),
            size=("size", "first"),
            requested_obstacle_density=("requested_obstacle_density", "first"),
            difficulty=("difficulty", "mean"),
            shortest_path_length=("shortest_path_length", "first"),
            directness=("directness", "first"),
        )
    )

    targets = [0.25, 0.60, 0.90]
    titles = ["Easy example", "Medium example", "Hard example"]
    selected = []
    used = set()
    for target in targets:
        candidates = level_data.loc[~level_data["level_id"].isin(used)].copy()
        candidates["distance"] = (candidates["difficulty"] - target).abs()
        row = candidates.sort_values("distance").iloc[0]
        used.add(row["level_id"])
        selected.append(row)

    colors = mcolors.ListedColormap(["#f4f7fb", "#49e2a4", "#08a8c7", "#ff6b6b", "#263044"])
    bounds = [-0.1, 0.1, 0.35, 0.65, 0.85, 1.1]
    norm = mcolors.BoundaryNorm(bounds, colors.N)

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.8))
    for ax, row, title in zip(axes, selected, titles):
        matrix = _level_matrix(int(row["seed"]), int(row["size"]), float(row["requested_obstacle_density"]))
        ax.imshow(matrix, cmap=colors, norm=norm)
        ax.set_title(f"{title}\ndifficulty={row['difficulty']:.2f}, path={row['shortest_path_length']:.0f}")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    fig.suptitle("Reproducible level examples selected from generated dataset", y=1.02)
    fig.tight_layout()
    path = plots / "level_examples.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path
