from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import hashlib
import random
from typing import Iterable


Coord = tuple[int, int]


@dataclass(frozen=True)
class GridLevel:
    seed: int
    size: int
    requested_obstacle_density: float
    grid: tuple[tuple[int, ...], ...]
    start: Coord
    goal: Coord

    def is_open(self, cell: Coord) -> bool:
        row, col = cell
        return 0 <= row < self.size and 0 <= col < self.size and self.grid[row][col] == 0


def stable_int_seed(*parts: object) -> int:
    raw = "::".join(str(part) for part in parts).encode("utf-8")
    return int(hashlib.sha256(raw).hexdigest()[:16], 16)


def generate_level(seed: int, size: int, obstacle_density: float) -> GridLevel:
    if size < 4:
        raise ValueError("size must be >= 4")
    if not 0 <= obstacle_density < 0.75:
        raise ValueError("obstacle_density must be in [0, 0.75)")

    rng = random.Random(stable_int_seed(seed, size, f"{obstacle_density:.4f}"))
    start = (0, 0)
    goal = (size - 1, size - 1)
    rows: list[tuple[int, ...]] = []

    for row in range(size):
        values = []
        for col in range(size):
            cell = (row, col)
            if cell in (start, goal):
                values.append(0)
            else:
                values.append(1 if rng.random() < obstacle_density else 0)
        rows.append(tuple(values))

    return GridLevel(
        seed=seed,
        size=size,
        requested_obstacle_density=obstacle_density,
        grid=tuple(rows),
        start=start,
        goal=goal,
    )


def neighbors(cell: Coord, size: int) -> Iterable[Coord]:
    row, col = cell
    for nr, nc in ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)):
        if 0 <= nr < size and 0 <= nc < size:
            yield nr, nc


def open_neighbors(level: GridLevel, cell: Coord) -> list[Coord]:
    return [candidate for candidate in neighbors(cell, level.size) if level.is_open(candidate)]


def reachable_cells(level: GridLevel, start: Coord | None = None) -> set[Coord]:
    origin = level.start if start is None else start
    if not level.is_open(origin):
        return set()

    queue: deque[Coord] = deque([origin])
    seen = {origin}
    while queue:
        cell = queue.popleft()
        for candidate in open_neighbors(level, cell):
            if candidate not in seen:
                seen.add(candidate)
                queue.append(candidate)
    return seen


def shortest_path(level: GridLevel) -> list[Coord] | None:
    if not level.is_open(level.start) or not level.is_open(level.goal):
        return None

    queue: deque[Coord] = deque([level.start])
    parent: dict[Coord, Coord | None] = {level.start: None}

    while queue:
        cell = queue.popleft()
        if cell == level.goal:
            break
        for candidate in open_neighbors(level, cell):
            if candidate not in parent:
                parent[candidate] = cell
                queue.append(candidate)

    if level.goal not in parent:
        return None

    path = []
    current: Coord | None = level.goal
    while current is not None:
        path.append(current)
        current = parent[current]
    path.reverse()
    return path


def render_ascii(level: GridLevel) -> str:
    path = set(shortest_path(level) or [])
    rows = []
    for row in range(level.size):
        chars = []
        for col in range(level.size):
            cell = (row, col)
            if cell == level.start:
                chars.append("S")
            elif cell == level.goal:
                chars.append("G")
            elif level.grid[row][col] == 1:
                chars.append("#")
            elif cell in path:
                chars.append("*")
            else:
                chars.append(".")
        rows.append("".join(chars))
    return "\n".join(rows)

