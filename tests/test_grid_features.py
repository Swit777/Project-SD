from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from game_ai_complexity.features import extract_features
from game_ai_complexity.grid import GridLevel, generate_level, shortest_path


class GridFeatureTests(unittest.TestCase):
    def test_generation_is_reproducible(self):
        first = generate_level(seed=10, size=8, obstacle_density=0.25)
        second = generate_level(seed=10, size=8, obstacle_density=0.25)
        self.assertEqual(first.grid, second.grid)
        self.assertEqual(first.start, (0, 0))
        self.assertEqual(first.goal, (7, 7))

    def test_shortest_path_on_empty_grid(self):
        grid = tuple(tuple(0 for _ in range(5)) for _ in range(5))
        level = GridLevel(1, 5, 0.0, grid, (0, 0), (4, 4))
        path = shortest_path(level)
        self.assertIsNotNone(path)
        self.assertEqual(len(path) - 1, 8)

    def test_features_do_not_contain_missing_values(self):
        level = generate_level(seed=7, size=8, obstacle_density=0.35)
        features = extract_features(level, max_steps=22)
        self.assertTrue(all(value is not None for value in features.values()))
        self.assertGreaterEqual(features["actual_obstacle_density"], 0.0)
        self.assertLessEqual(features["actual_obstacle_density"], 1.0)


if __name__ == "__main__":
    unittest.main()

