from __future__ import annotations

from dataclasses import dataclass
import heapq
import random

from .grid import Coord, GridLevel, open_neighbors, shortest_path, stable_int_seed


@dataclass(frozen=True)
class AgentResult:
    agent: str
    success: int
    success_rate: float
    steps: float
    reward: float
    episodes: int

    def to_row(self) -> dict[str, float | int | str]:
        return {
            "agent": self.agent,
            "success": self.success,
            "success_rate": self.success_rate,
            "steps": self.steps,
            "reward": self.reward,
            "episodes": self.episodes,
            "difficulty": 1.0 - self.success_rate,
        }


def _reward(success: bool, steps: int, max_steps: int) -> float:
    if success:
        return 1.0 - (steps / max_steps) * 0.25
    return -1.0


def run_bfs_agent(level: GridLevel, max_steps: int) -> AgentResult:
    path = shortest_path(level)
    if not path:
        return AgentResult("bfs", 0, 0.0, float(max_steps), -1.0, 1)

    steps = len(path) - 1
    success = steps <= max_steps
    return AgentResult("bfs", int(success), float(success), float(min(steps, max_steps)), _reward(success, steps, max_steps), 1)


def _manhattan(left: Coord, right: Coord) -> int:
    return abs(left[0] - right[0]) + abs(left[1] - right[1])


def _reconstruct_path(parent: dict[Coord, Coord | None], goal: Coord) -> list[Coord]:
    path: list[Coord] = []
    current: Coord | None = goal
    while current is not None:
        path.append(current)
        current = parent[current]
    path.reverse()
    return path


def astar_path(level: GridLevel) -> list[Coord] | None:
    if not level.is_open(level.start) or not level.is_open(level.goal):
        return None

    frontier: list[tuple[int, int, Coord]] = []
    heapq.heappush(frontier, (_manhattan(level.start, level.goal), 0, level.start))
    parent: dict[Coord, Coord | None] = {level.start: None}
    cost_so_far: dict[Coord, int] = {level.start: 0}

    while frontier:
        _, cost, current = heapq.heappop(frontier)
        if current == level.goal:
            return _reconstruct_path(parent, level.goal)

        for candidate in open_neighbors(level, current):
            new_cost = cost + 1
            if candidate not in cost_so_far or new_cost < cost_so_far[candidate]:
                cost_so_far[candidate] = new_cost
                priority = new_cost + _manhattan(candidate, level.goal)
                parent[candidate] = current
                heapq.heappush(frontier, (priority, new_cost, candidate))

    return None


def run_astar_agent(level: GridLevel, max_steps: int) -> AgentResult:
    path = astar_path(level)
    if not path:
        return AgentResult("astar", 0, 0.0, float(max_steps), -1.0, 1)

    steps = len(path) - 1
    success = steps <= max_steps
    return AgentResult("astar", int(success), float(success), float(min(steps, max_steps)), _reward(success, steps, max_steps), 1)


def run_greedy_agent(level: GridLevel, max_steps: int) -> AgentResult:
    current = level.start
    visits = {current: 1}

    for step in range(max_steps + 1):
        if current == level.goal:
            return AgentResult("greedy", 1, 1.0, float(step), _reward(True, step, max_steps), 1)

        candidates = open_neighbors(level, current)
        if not candidates:
            break

        candidates.sort(
            key=lambda cell: (
                abs(cell[0] - level.goal[0]) + abs(cell[1] - level.goal[1]),
                visits.get(cell, 0),
                cell[0],
                cell[1],
            )
        )
        current = candidates[0]
        visits[current] = visits.get(current, 0) + 1

    return AgentResult("greedy", 0, 0.0, float(max_steps), -1.0, 1)


def _move(level: GridLevel, state: Coord, action: int) -> tuple[Coord, float, bool]:
    row, col = state
    moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    dr, dc = moves[action]
    candidate = (row + dr, col + dc)

    if candidate == level.goal and level.is_open(candidate):
        return candidate, 1.0, True
    if not level.is_open(candidate):
        return state, -0.22, False
    return candidate, -0.025, False


def run_q_learning_agent(
    level: GridLevel,
    max_steps: int,
    train_episodes: int,
    eval_episodes: int,
) -> AgentResult:
    rng = random.Random(stable_int_seed(level.seed, level.size, level.requested_obstacle_density, "q-learning"))
    q_values: dict[tuple[Coord, int], float] = {}
    actions = [0, 1, 2, 3]
    alpha = 0.35
    gamma = 0.92
    epsilon_start = 0.35
    epsilon_end = 0.05

    if not level.is_open(level.start) or not level.is_open(level.goal):
        return AgentResult("q_learning", 0, 0.0, float(max_steps), -1.0, eval_episodes)

    for episode in range(max(1, train_episodes)):
        state = level.start
        frac = episode / max(1, train_episodes - 1)
        epsilon = epsilon_start + (epsilon_end - epsilon_start) * frac

        for _ in range(max_steps):
            if rng.random() < epsilon:
                action = rng.choice(actions)
            else:
                action = max(actions, key=lambda act: (q_values.get((state, act), 0.0), -act))

            next_state, reward, done = _move(level, state, action)
            current = q_values.get((state, action), 0.0)
            future = max(q_values.get((next_state, act), 0.0) for act in actions)
            q_values[(state, action)] = current + alpha * (reward + gamma * future - current)
            state = next_state
            if done:
                break

    successes = 0
    total_steps = 0
    total_reward = 0.0

    for episode in range(max(1, eval_episodes)):
        state = level.start
        episode_steps = max_steps
        success = False

        for step in range(max_steps + 1):
            if state == level.goal:
                episode_steps = step
                success = True
                break

            if rng.random() < 0.03:
                action = rng.choice(actions)
            else:
                action = max(actions, key=lambda act: (q_values.get((state, act), 0.0), -act))
            state, _, _ = _move(level, state, action)

        successes += int(success)
        total_steps += episode_steps
        total_reward += _reward(success, episode_steps, max_steps)

    episodes = max(1, eval_episodes)
    success_rate = successes / episodes
    return AgentResult(
        "q_learning",
        int(success_rate >= 0.5),
        success_rate,
        total_steps / episodes,
        total_reward / episodes,
        episodes,
    )


def run_random_agent(level: GridLevel, max_steps: int, episodes: int) -> AgentResult:
    rng = random.Random(stable_int_seed(level.seed, level.size, level.requested_obstacle_density, "random-agent"))
    successes = 0
    total_steps = 0
    total_reward = 0.0

    for _ in range(episodes):
        current = level.start
        episode_steps = max_steps
        success = False

        for step in range(max_steps + 1):
            if current == level.goal:
                success = True
                episode_steps = step
                break
            candidates = open_neighbors(level, current)
            if not candidates:
                episode_steps = step
                break
            current = rng.choice(candidates)

        successes += int(success)
        total_steps += episode_steps
        total_reward += _reward(success, episode_steps, max_steps)

    success_rate = successes / episodes if episodes else 0.0
    return AgentResult(
        "random",
        int(success_rate >= 0.5),
        success_rate,
        total_steps / episodes if episodes else float(max_steps),
        total_reward / episodes if episodes else -1.0,
        episodes,
    )


def run_agent(
    level: GridLevel,
    agent: str,
    max_steps: int,
    random_episodes: int,
    q_learning_episodes: int = 70,
    q_learning_eval_episodes: int = 5,
) -> AgentResult:
    if agent == "bfs":
        return run_bfs_agent(level, max_steps)
    if agent == "astar":
        return run_astar_agent(level, max_steps)
    if agent == "greedy":
        return run_greedy_agent(level, max_steps)
    if agent == "random":
        return run_random_agent(level, max_steps, random_episodes)
    if agent == "q_learning":
        return run_q_learning_agent(level, max_steps, q_learning_episodes, q_learning_eval_episodes)
    raise ValueError(f"Unknown agent: {agent}")
