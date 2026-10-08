from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from game_ai_complexity.config import load_config
from game_ai_complexity.dataset import INFERENCE_COLUMNS, REQUIRED_COLUMNS, generate_dataset, validate_dataset, validate_inference_dataset
from game_ai_complexity.models import train_models
from game_ai_complexity.predict import predict_success


class DatasetModelTests(unittest.TestCase):
    def small_config(self):
        config = load_config()
        config["generation"]["seed_start"] = 1
        config["generation"]["n_seeds"] = 12
        config["generation"]["sizes"] = [6, 8]
        config["generation"]["obstacle_densities"] = [0.05, 0.25, 0.50]
        config["generation"]["agents"] = ["greedy", "astar", "q_learning"]
        config["generation"]["random_episodes"] = 2
        config["generation"]["q_learning_episodes"] = 20
        config["generation"]["q_learning_eval_episodes"] = 2
        config["generation"]["max_steps_factor"] = 3.2
        return config

    def test_dataset_schema_and_target(self):
        data = generate_dataset(self.small_config())
        self.assertEqual(list(data.columns), REQUIRED_COLUMNS)
        self.assertFalse(validate_dataset(data, require_target=True))
        self.assertGreater(data["success"].nunique(), 1)

    def test_train_models_smoke(self):
        data = generate_dataset(self.small_config())
        with tempfile.TemporaryDirectory() as tmp:
            result = train_models(data, tmp, random_state=5, test_size=0.3)
            self.assertIn("random_forest", result["metrics"])
            self.assertIn("best_model", result)
            self.assertGreaterEqual(result["metrics"][result["best_model"]]["roc_auc"], 0.5)

    def test_predict_on_feature_only_dataset(self):
        data = generate_dataset(self.small_config())
        with tempfile.TemporaryDirectory() as tmp:
            train_models(data, tmp, random_state=5, test_size=0.3)
            new_data = data[INFERENCE_COLUMNS].head(10).copy()
            self.assertFalse(validate_inference_dataset(new_data))
            predictions = predict_success(new_data, Path(tmp) / "best_model.pkl")
            self.assertIn("predicted_difficulty", predictions.columns)
            self.assertEqual(len(predictions), 10)
            self.assertTrue(predictions["predicted_success_probability"].between(0, 1).all())


if __name__ == "__main__":
    unittest.main()
