import io
from pathlib import Path
import unittest
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports/player_experience"


@unittest.skipUnless((REPORTS / "models/best_model.joblib").exists(), "Run the real pipeline before integration UI tests")
class PlayerAppTests(unittest.TestCase):
    def test_app_loads_all_views_and_handles_empty_filter(self):
        app = AppTest.from_file(str(ROOT / "player_experience_app.py")).run(timeout=30)
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.tabs), 4)
        self.assertEqual(len(app.metric), 6)
        app.multiselect(key="modes_filter").set_value([]).run(timeout=30)
        self.assertEqual(len(app.exception), 0)

    def test_invalid_upload_is_explained_without_crash(self):
        with patch("streamlit.file_uploader", return_value=io.BytesIO(b"wrong_column\n1\n")):
            app = AppTest.from_file(str(ROOT / "player_experience_app.py")).run(timeout=30)
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.error), 1)
        self.assertIn("Missing columns", app.error[0].value)

    def test_real_feature_only_upload_and_prediction(self):
        payload = (REPORTS / "example_prediction_input.csv").read_bytes()
        with patch("streamlit.file_uploader", return_value=io.BytesIO(payload)):
            app = AppTest.from_file(str(ROOT / "player_experience_app.py")).run(timeout=30)
            self.assertEqual(len(app.exception), 0)
            app.button(key="analyze_upload").click().run(timeout=30)
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.error), 0)
        _, prediction = app.session_state["upload_prediction"]
        self.assertEqual(len(prediction), 50)
        self.assertTrue(prediction.predicted_enjoyment.between(0, 100).all())


if __name__ == "__main__":
    unittest.main()
