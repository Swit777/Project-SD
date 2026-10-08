import io
from pathlib import Path
import unittest
from unittest.mock import patch

import pandas as pd
from streamlit.testing.v1 import AppTest
import sys

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/processed/game_concept/games.csv"
MODEL = ROOT / "reports/game_concept/models/concept_model.joblib"
sys.path.insert(0, str(ROOT / "src"))
from game_concept.mechanics import KEYS


@unittest.skipUnless(MODEL.exists() and DATA.exists(), "Run the Steam pipeline before UI integration tests")
class ConceptAppTests(unittest.TestCase):
    def test_boots_and_handles_empty_catalog(self):
        app = AppTest.from_file(str(ROOT / "app.py")).run(timeout=30)
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.tabs), 4)
        app.multiselect(key="catalog_genres").set_value([]).run(timeout=30)
        self.assertEqual(len(app.exception), 0)

    def test_all_mechanics_does_not_return_false_success(self):
        app = AppTest.from_file(str(ROOT / "app.py")).run(timeout=30)
        mechanics = app.multiselect(key="concept_mechanics")
        mechanics.set_value(KEYS).run(timeout=30)
        app.button(key="forecast").click().run(timeout=30)
        self.assertEqual(len(app.exception), 0)
        prediction = app.session_state["concept_result"][1]
        self.assertEqual(prediction["support"]["status"], "abstain")
        self.assertIsNone(prediction["probabilities"])

    def test_invalid_upload_is_reported(self):
        with patch("streamlit.file_uploader", return_value=io.BytesIO(b"wrong\n1\n")):
            app = AppTest.from_file(str(ROOT / "app.py")).run(timeout=30)
        self.assertEqual(len(app.exception), 0)
        self.assertTrue(any("Missing columns" in error.value for error in app.error))

    def test_feature_only_upload_runs_without_targets(self):
        import sys
        sys.path.insert(0, str(ROOT / "src"))
        from game_concept.dataset import feature_columns
        payload = pd.read_csv(DATA)[feature_columns()].head(5).to_csv(index=False).encode()
        with patch("streamlit.file_uploader", return_value=io.BytesIO(payload)):
            app = AppTest.from_file(str(ROOT / "app.py")).run(timeout=30)
            app.button(key="predict_upload").click().run(timeout=30)
        self.assertEqual(len(app.exception), 0)
        predictions = app.session_state["uploaded_predictions"][1]
        self.assertEqual(len(predictions), 5)
        self.assertTrue(predictions.support_status.isin(["supported", "limited", "abstain"]).all())

    def test_comparison_and_lime_work_with_real_model(self):
        app = AppTest.from_file(str(ROOT / "app.py")).run(timeout=30)
        app.button(key="compare").click().run(timeout=30)
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.session_state["comparison"][1]), 36)
        app.button(key="lime").click().run(timeout=30)
        self.assertEqual(len(app.exception), 0)
        self.assertGreater(len(app.session_state["lime_explanation"][1]), 0)


if __name__ == "__main__":
    unittest.main()
