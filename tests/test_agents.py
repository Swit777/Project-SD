from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from game_ai_complexity.agents import astar_path, run_astar_agent, run_q_learning_agent
from game_ai_complexity.grid import GridLevel


class AgentTests(unittest.TestCase):
    def empty_level(self):
        grid = tuple(tuple(0 for _ in range(5)) for _ in range(5))
        return GridLevel(1, 5, 0.0, grid, (0, 0), (4, 4))

    def test_astar_finds_shortest_path_on_empty_grid(self):
        level = self.empty_level()
        path = astar_path(level)
        self.assertIsNotNone(path)
        self.assertEqual(len(path) - 1, 8)

    def test_astar_agent_succeeds_on_empty_grid(self):
        result = run_astar_agent(self.empty_level(), max_steps=12)
        self.assertEqual(result.success, 1)
        self.assertEqual(result.agent, "astar")

    def test_q_learning_agent_returns_valid_result(self):
        result = run_q_learning_agent(self.empty_level(), max_steps=14, train_episodes=8, eval_episodes=2)
        self.assertEqual(result.agent, "q_learning")
        self.assertGreaterEqual(result.success_rate, 0.0)
        self.assertLessEqual(result.success_rate, 1.0)


if __name__ == "__main__":
    unittest.main()
