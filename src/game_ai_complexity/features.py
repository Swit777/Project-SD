from __future__ import annotations

from statistics import mean

from .grid import GridLevel, open_neighbors, reachable_cells, shortest_path


FEATURE_COLUMNS = [
    "size",
    "requested_obstacle_density",
    "actual_obstacle_density",
    "open_cell_ratio",
    "wall_count",
    "has_path",
    "shortest_path_length",
    "shortest_path_norm",
    "manhattan_distance",
    "directness",
    "accessible_area_ratio",
    "dead_end_ratio",
    "branch_cell_ratio",
    "mean_degree",
    "start_degree",
    "goal_degree",
    "max_steps",
]


def extract_features(level: GridLevel, max_steps: int) -> dict[str, float | int]:
    total_cells = level.size * level.size
    wall_count = sum(cell for row in level.grid for cell in row)
    open_count = total_cells - wall_count
    actual_density = wall_count / total_cells

    path = shortest_path(level)
    has_path = int(path is not None)
    shortest_len = len(path) - 1 if path else total_cells + 1
    shortest_norm = shortest_len / total_cells
    manhattan = abs(level.goal[0] - level.start[0]) + abs(level.goal[1] - level.start[1])
    directness = manhattan / shortest_len if path and shortest_len > 0 else 0.0

    reachable = reachable_cells(level)
    accessible_ratio = len(reachable) / open_count if open_count else 0.0

    degrees = []
    for row in range(level.size):
        for col in range(level.size):
            cell = (row, col)
            if level.is_open(cell):
                degrees.append(len(open_neighbors(level, cell)))

    dead_ends = sum(1 for degree in degrees if degree <= 1)
    branches = sum(1 for degree in degrees if degree >= 3)

    return {
        "size": level.size,
        "requested_obstacle_density": level.requested_obstacle_density,
        "actual_obstacle_density": actual_density,
        "open_cell_ratio": open_count / total_cells,
        "wall_count": wall_count,
        "has_path": has_path,
        "shortest_path_length": shortest_len,
        "shortest_path_norm": shortest_norm,
        "manhattan_distance": manhattan,
        "directness": directness,
        "accessible_area_ratio": accessible_ratio,
        "dead_end_ratio": dead_ends / open_count if open_count else 0.0,
        "branch_cell_ratio": branches / open_count if open_count else 0.0,
        "mean_degree": mean(degrees) if degrees else 0.0,
        "start_degree": len(open_neighbors(level, level.start)),
        "goal_degree": len(open_neighbors(level, level.goal)),
        "max_steps": max_steps,
    }

