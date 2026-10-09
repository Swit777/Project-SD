import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from urllib.parse import parse_qs, urlparse

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from game_concept.analogues import find_analogues, owner_evidence, review_stats, structured_mode
from game_concept.analogue_catalog import build_analogue_catalog
from game_concept.charts import analogue_landscape, classification_errors, mechanic_matrix
from game_concept.config import load_config
from game_concept.live_reviews import parse_review_totals, refresh_review_totals
from game_concept.predict import concept_features
from game_concept.sources import SNAPSHOT_SHA256
from game_concept.tag_similarity import TagIndex


def game(appid, genres=None, mechanics=None, reviews=100, categories=None, price=15):
    row = concept_features(genres or ["Indie", "RPG"], mechanics or ["crafting", "co_op"], price).iloc[0].to_dict()
    row.update(appid=appid, name=f"Game {appid}", price_usd=price, age_days=1000, owners_lower=0, owners_upper=20000,
               positive_reviews=reviews, negative_reviews=0, categories_json=json.dumps(categories if categories is not None else ["Online Co-op"]),
               genres_json=json.dumps(genres or ["Indie", "RPG"]))
    return row


class AnalogueTests(unittest.TestCase):
    def setUp(self):
        self.concept = concept_features(["Indie", "RPG"], ["crafting", "co_op"])

    def test_indie_alone_cannot_match_a_different_core_genre(self):
        rows, _ = find_analogues(self.concept, pd.DataFrame([game(1, genres=["Indie", "Sports"]), game(2)]))
        self.assertEqual(rows.appid.tolist(), [2])

    def test_multiplayer_claim_alone_does_not_confirm_mode(self):
        rows, _ = find_analogues(self.concept, pd.DataFrame([game(1, categories=["Single-player"]), game(2)]))
        self.assertEqual(rows.appid.tolist(), [2])
        rows, _ = find_analogues(self.concept, pd.DataFrame([game(1, categories=["Single-player"])]), require_modes=False)
        self.assertEqual(len(rows), 1)

    def test_unknown_categories_do_not_confirm_mode(self):
        frame = pd.DataFrame([game(1)]).drop(columns="categories_json")
        self.assertTrue(find_analogues(self.concept, frame)[0].empty)

    def test_review_filter_is_explicit_and_does_not_fabricate_fallback(self):
        frame = pd.DataFrame([game(1, reviews=1), game(2, reviews=30)])
        rows, meta = find_analogues(self.concept, frame, min_reviews=30)
        self.assertEqual(rows.appid.tolist(), [2])
        self.assertEqual(meta["matched_before_reviews"], 2)
        self.assertEqual(meta["excluded_low_or_unknown_reviews"], 1)
        self.assertTrue(find_analogues(self.concept, frame, min_reviews=100)[0].empty)

    def test_reviews_and_owner_outcomes_do_not_change_relevance(self):
        frame = pd.DataFrame([game(1), game(2, reviews=1000000)])
        before = find_analogues(self.concept, frame)[0]
        frame["positive_reviews"] = [1000000, 100]
        frame["owners_lower"], frame["owners_upper"] = 500000, 1000000
        after = find_analogues(self.concept, frame)[0]
        self.assertEqual(before.appid.tolist(), after.appid.tolist())
        np.testing.assert_allclose(before.similarity, after.similarity)

    def test_review_unknown_and_invalid_are_not_zero(self):
        frame = pd.DataFrame({"positive_reviews": [0, np.nan, -1, 1.5, np.inf], "negative_reviews": [0, 1, 0, 0, 0]})
        stats = review_stats(frame)
        self.assertEqual(stats.review_count.iloc[0], 0)
        self.assertTrue(stats.review_count.iloc[1:].isna().all())
        self.assertTrue(stats.positive_share.isna().all())

    def test_unknown_review_counts_are_excluded_only_when_filtered(self):
        frame = pd.DataFrame([game(1)]).drop(columns=["positive_reviews", "negative_reviews"])
        self.assertEqual(len(find_analogues(self.concept, frame, min_reviews=0)[0]), 1)
        self.assertTrue(find_analogues(self.concept, frame, min_reviews=1)[0].empty)

    def test_partial_match_explains_missing_mechanics(self):
        frame = pd.DataFrame([game(1, mechanics=["co_op"])])
        rows, _ = find_analogues(self.concept, frame, min_coverage=0.5)
        self.assertEqual(json.loads(rows.iloc[0].missing_mechanics), ["crafting"])
        self.assertEqual(rows.iloc[0].mechanic_coverage, 0.5)
        self.assertTrue(find_analogues(self.concept, frame, min_coverage=1)[0].empty)

    def test_prices_constrain_comparability(self):
        frame = pd.DataFrame([game(1, price=150), game(2, price=15)])
        self.assertEqual(find_analogues(self.concept, frame, price_factor=3)[0].appid.tolist(), [2])

    def test_shuffle_preserves_ties_and_score_bounds(self):
        frame = pd.DataFrame([game(3), game(1), game(2)])
        rows, _ = find_analogues(self.concept, frame)
        shuffled, _ = find_analogues(self.concept, frame.sample(frac=1, random_state=7))
        self.assertEqual(rows.appid.tolist(), [1, 2, 3])
        pd.testing.assert_frame_equal(rows, shuffled)
        self.assertTrue(rows.similarity.between(0, 100).all())

    def test_self_match_is_excluded(self):
        concept = self.concept.assign(appid=1)
        self.assertEqual(find_analogues(concept, pd.DataFrame([game(1), game(2)]))[0].appid.tolist(), [2])

    def test_no_mechanics_and_no_candidates_are_valid(self):
        concept = concept_features(["RPG"], [])
        rows, _ = find_analogues(concept, pd.DataFrame([game(1)]))
        self.assertTrue(np.isfinite(rows.similarity).all())
        rows, meta = find_analogues(concept, pd.DataFrame([game(1)]).iloc[:0])
        self.assertTrue(rows.empty)
        self.assertEqual(meta["eligible_games"], 0)

    def test_lower_owner_band_is_not_presented_as_sales(self):
        status, note = owner_evidence(0, 20000)
        self.assertEqual(status, "low_resolution")
        self.assertIn("неизвестно", note)
        self.assertEqual(owner_evidence(np.nan, np.nan)[0], "unknown")
        self.assertEqual(owner_evidence(0, 0)[0], "unknown")
        self.assertIn("не продажи", owner_evidence(20000, 50000)[1])

    def test_modes_use_exact_categories_not_mentions(self):
        self.assertTrue(structured_mode(["Online Co-op"], "co_op"))
        self.assertFalse(structured_mode(["Single-player", "No Online Co-op"], "co_op"))
        self.assertFalse(structured_mode("invalid json", "co_op"))

    def test_charts_keep_zero_reviews_and_confusion_counts(self):
        rows, _ = find_analogues(self.concept, pd.DataFrame([game(1, reviews=0), game(2)]))
        fig = analogue_landscape(rows, 15)
        self.assertEqual(len(fig.data[0].x), 2)
        self.assertIsNotNone(mechanic_matrix(rows, ["crafting", "co_op"]))
        predictions = pd.DataFrame({"owners_band": [0, 1, 3], "m_p0": [1, 1, 0], "m_p1": [0, 0, 0], "m_p2": [0, 0, 0], "m_p3": [0, 0, 1]})
        fig, matrix = classification_errors(predictions, "m")
        self.assertEqual(matrix.sum(), 3)
        self.assertEqual(matrix[1, 0], 1)
        self.assertTrue(np.isfinite(fig.data[0].z).all())

    def test_tag_anchor_prefers_format_over_popularity(self):
        frame = pd.DataFrame([game(1, reviews=100000), game(2, reviews=50), game(3)])
        frame["tags_json"] = [json.dumps(["Horror", "First-Person"]), json.dumps(["Farming Sim", "Pixel Graphics"]), json.dumps(["Farming Sim", "Pixel Graphics"])]
        scores = TagIndex(frame).scores(["Farming Sim", "Pixel Graphics"])
        rows, _ = find_analogues(self.concept, frame, tag_scores=scores, exclude_appid=3)
        self.assertEqual(rows.iloc[0].appid, 2)
        self.assertNotIn(3, rows.appid.tolist())
        strict, _ = find_analogues(self.concept, frame, required_tags=("Pixel Graphics",))
        self.assertEqual(set(strict.appid), {2, 3})

    def test_empty_or_unknown_tags_do_not_invent_semantic_match(self):
        frame = pd.DataFrame([game(1)])
        self.assertIsNone(TagIndex(frame).scores(["Unknown tag"]))
        frame["tags_json"] = '["RPG", "Pixel Graphics"]'
        self.assertIsNone(TagIndex(frame).scores(["Unknown tag"]))
        self.assertIsNone(TagIndex(frame).scores(["RPG"]))


class LiveReviewsTests(unittest.TestCase):
    def test_missing_summary_is_not_zero(self):
        for payload in [{}, {"response": {}}, {"response": {"query_summary": {"num_reviews": 1}}}]:
            with self.assertRaises(ValueError):
                parse_review_totals(payload)

    def test_validate_totals_not_the_one_review_sample(self):
        payload = {"response": {"query_summary": {"num_reviews": 1, "total_reviews": 100, "total_positive": 80, "total_negative": 20}}}
        self.assertEqual(parse_review_totals(payload)["total_reviews"], 100)
        payload["response"]["query_summary"]["total_positive"] = 99
        with self.assertRaises(ValueError):
            parse_review_totals(payload)

    def test_live_fetch_saves_only_aggregate_and_does_not_touch_training(self):
        with tempfile.TemporaryDirectory() as directory, patch("game_concept.live_reviews.CachedHttp") as client:
            body = {"response": {"query_summary": {"total_reviews": 10, "total_positive": 7, "total_negative": 3},
                                 "reviews": [{"author": {"steamid": "personal-id"}}]}}
            client.return_value.get.return_value = (json.dumps(body).encode(), {"captured_at_utc": "2026-10-08", "sha256": "hash"})
            result = refresh_review_totals(Path(directory), 42)
            params = json.loads(parse_qs(urlparse(client.return_value.get.call_args.args[0]).query)["input_json"][0])
            self.assertEqual(params["purchase_type"], 1)
            self.assertEqual(params["languages"], ["all"])
            self.assertEqual(result["total_reviews"], 10)
            self.assertNotIn("personal-id", json.dumps(result))
            self.assertFalse((Path(directory) / "data/processed/game_concept/games.csv").exists())


class DiscoveryCatalogTests(unittest.TestCase):
    def test_separate_catalog_retains_unknown_owners_and_repairs_header(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "data/raw/steam_snapshot/games.csv"
            source.parent.mkdir(parents=True)
            columns = ["AppID", "Name", "Release date", "Estimated owners", "Price", "DiscountDLC count", "About the game",
                       "Header image", "Positive", "Negative", "Genres", "Tags", "Categories", "Developers", "Median playtime forever"]
            with source.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(columns)
                for appid, name, price in [(1, "Example", 10), (2, "Example Demo", 10), (3, "Free", 0)]:
                    writer.writerow([appid, name, "Jan 1, 2020", "0 - 0", price, 50, 2,
                        "Crafting gear and cooperative play with friends in an enormous world full of adventures and mysteries.",
                        "", 100, 20, "Indie,RPG", "Crafting", "Online Co-op", "Studio", 0])
            with patch("game_concept.analogue_catalog.sha256_file", return_value=SNAPSHOT_SHA256):
                manifest = build_analogue_catalog(root, load_config())
            data = pd.read_csv(root / "data/processed/game_concept/analogue_catalog.csv")
            self.assertEqual(manifest["catalog_games"], 1)
            self.assertEqual(data.iloc[0].price_usd, 20)
            self.assertEqual(data.iloc[0].mode_co_op_structured, 1)
            self.assertTrue(data.owners_lower.isna().all())
            self.assertFalse((root / "data/processed/game_concept/games.csv").exists())


if __name__ == "__main__":
    unittest.main()
