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
        self.assertEqual(len(app.error), 0)
        self.assertGreaterEqual(len(app.get("plotly_chart")), 8)
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

    def test_analogue_filters_do_not_change_model_probabilities(self):
        app = AppTest.from_file(str(ROOT / "app.py")).run(timeout=30)
        before = app.session_state["concept_result"][1]["probabilities"]
        app.selectbox(key="analogue_reviews_close").set_value(500).run(timeout=30)
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(before, app.session_state["concept_result"][1]["probabilities"])
        self.assertTrue(app.session_state["analogue_results"].review_count.ge(500).all())
        self.assertTrue(app.session_state["analogue_results"].mechanic_coverage.ge(0.5).all())

    def test_train_analogues_stay_within_training_partition(self):
        from game_concept.predict import load_bundle
        app = AppTest.from_file(str(ROOT / "app.py")).run(timeout=30)
        # Streamlit exposes a segmented control as a button group in AppTest.
        app.button_group(key="analogue_source").set_value("Только train").run(timeout=30)
        self.assertEqual(len(app.exception), 0)
        ids = set(app.session_state["analogue_results"].appid)
        self.assertTrue(ids.issubset(set(load_bundle(MODEL)["reference"].appid)))

    def test_broader_profile_exposes_popular_references_without_changing_forecast(self):
        app = AppTest.from_file(str(ROOT / "app.py")).run(timeout=30)
        before = app.session_state["concept_result"][1]["probabilities"]
        app.button_group(key="analogue_profile_mode").set_value("Рыночные референсы").run(timeout=30)
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(before, app.session_state["concept_result"][1]["probabilities"])
        rows = app.session_state["analogue_results"]
        self.assertTrue(rows.review_count.is_monotonic_decreasing)
        self.assertFalse(app.checkbox(key="analogue_modes_market").value)

    def test_game_anchor_changes_retrieval_not_forecast(self):
        app = AppTest.from_file(str(ROOT / "app.py")).run(timeout=30)
        before = app.session_state["concept_result"][1]["probabilities"]
        app.text_input(key="analogue_anchor_query").set_value("Stardew Valley").run(timeout=30)
        app.selectbox(key="analogue_anchor").set_value(413150).run(timeout=30)
        self.assertEqual(len(app.exception), 0)
        rows = app.session_state["analogue_results"]
        self.assertTrue(rows.tag_similarity.notna().all())
        self.assertNotIn(413150, rows.appid.tolist())
        self.assertEqual(before, app.session_state["concept_result"][1]["probabilities"])


if __name__ == "__main__":
    unittest.main()
