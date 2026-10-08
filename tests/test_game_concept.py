import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from game_concept.config import load_config
from game_concept.dataset import feature_columns, owner_band, owner_bounds, prepare_snapshot, validate_dataset
from game_concept.evaluation import developer_components, paired_bootstrap, split_games
from game_concept.mechanics import KEYS, extract_feature_claims, extract_mechanics, plain_text
from game_concept.models import train_experiment
from game_concept.predict import compare_components, concept_features, explain_lime, forecast, load_bundle
from game_concept.reports import write_reports
from game_concept.scraper import parse_store_page, scrape_games, summarize_reviews
from game_concept.sources import CachedHttp


def synthetic_games(n=400):
    rng = np.random.default_rng(42)
    rows = []
    bounds = [(0, 20000), (20000, 50000), (100000, 200000), (500000, 1000000)]
    for i in range(n):
        mechanics = [key for key in ["crafting", "base_building", "puzzles"] if rng.random() > 0.4]
        row = concept_features(["Indie", "RPG"], mechanics, float(rng.uniform(1, 30)), float(rng.uniform(1, 10))).iloc[0].to_dict()
        band = i % 4
        row.update(appid=i + 1, name=f"Game {i}", developer_group=f"studio_{i}", release_date=f"2020-{i % 12 + 1:02d}-01",
                   owners_band=band, owners_lower=bounds[band][0], owners_upper=bounds[band][1],
                   playtime_median_hours=float(rng.uniform(1, 20)))
        rows.append(row)
    return pd.DataFrame(rows)


class MechanicsTests(unittest.TestCase):
    def test_broad_tags_do_not_prove_specific_mechanics(self):
        flags, _ = extract_mechanics("", ["Building", "Trading Card Game", "Creature Collector"], [], "source")
        for key in ["base_building", "trading", "taming"]:
            self.assertEqual(flags[f"mechanic_{key}"], 0)

    def test_exact_tags_and_structured_categories(self):
        flags, evidence = extract_mechanics("", ["Crafting", "Not a crafting tag"], ["Online Co-op"], "source")
        self.assertEqual(flags["mechanic_crafting"], 1)
        self.assertEqual(flags["mechanic_co_op"], 1)
        self.assertTrue({"tag", "category"}.issubset({row["source_type"] for row in evidence}))

    def test_negation_and_nonliteral_crafting(self):
        text = "No crafting or procedural generation. Craft your destiny. Solve puzzles and build your base."
        flags, evidence = extract_mechanics(text, [], [], "source")
        self.assertEqual(flags["mechanic_crafting"], 0)
        self.assertEqual(flags["mechanic_procedural_generation"], 0)
        self.assertEqual(flags["mechanic_puzzles"], 1)
        self.assertEqual(flags["mechanic_base_building"], 1)
        self.assertTrue(any(row["negated"] for row in evidence))

    def test_html_scripts_are_not_evidence(self):
        flags, _ = extract_mechanics("<script>crafting</script><p>Explore a vast world.</p>", [], [], "source")
        self.assertEqual(flags["mechanic_crafting"], 0)
        self.assertNotIn("crafting", plain_text("<script>crafting</script>ok"))

    def test_distinct_feature_claims_are_candidates(self):
        claims = extract_feature_claims("<ul><li>Build your base using floating islands and movable rooms.</li><li>No need to craft your own weapons in this adventure.</li></ul>")
        self.assertTrue(any("floating islands" in row for row in claims))
        self.assertTrue(all(len(row) <= 250 for row in claims))


class DatasetTests(unittest.TestCase):
    def test_real_csv_header_repair_preserves_price_and_description(self):
        import csv
        columns = ["AppID", "Name", "Release date", "Estimated owners", "Price", "Discount", "DLC count", "About the game",
                   "Supported languages", "Header image", "Windows", "Mac", "Linux", "Positive", "Negative",
                   "Average playtime forever", "Median playtime forever", "Developers", "Publishers", "Categories", "Genres", "Tags"]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "data/raw/steam_snapshot/games.csv"
            source.parent.mkdir(parents=True)
            with source.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow([c for c in columns if c != "DLC count"][:5] + ["DiscountDLC count"] + columns[7:])
                for i in range(40):
                    writer.writerow([i + 1, f"Example {i}", "Jan 1, 2020", "20,000 - 50,000", 12.5, 50, 2,
                                     "Craft your own weapons and solve challenging puzzles in this magical adventure of discovery.",
                                     "['English']", "", True, False, False, 1, 0, 0, 120, f"studio_{i}", "publisher", "Single-player", "Indie,RPG", "Crafting"])
            from game_concept.sources import SNAPSHOT_SHA256
            with patch("game_concept.dataset.sha256_file", return_value=SNAPSHOT_SHA256):
                data, audit = prepare_snapshot(root, load_config())
            self.assertTrue(audit["header_repaired"])
            self.assertEqual(len(data), 40)
            self.assertTrue(data.price_usd.eq(25).all())
            self.assertTrue(data.playtime_median_hours.eq(2).all())
            self.assertTrue(data.mechanic_crafting.eq(1).all())

    def test_changed_source_is_rejected_before_parsing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "data/raw/steam_snapshot/games.csv"
            source.parent.mkdir(parents=True)
            source.write_bytes(b"modified data")
            with self.assertRaisesRegex(ValueError, "checksum"):
                prepare_snapshot(root, load_config())

    def test_owner_ranges_are_not_fake_midpoint_sales(self):
        self.assertEqual(owner_bounds("20,000 - 50,000"), (20000, 50000))
        self.assertTrue(np.isnan(owner_bounds("0 - 0")[0]))
        self.assertEqual(owner_band(20000, 50000), 1)
        self.assertIsNone(owner_band(10000, 50000))
        self.assertIsNone(owner_band(0, np.inf))

    def test_training_and_prediction_schemas(self):
        data = synthetic_games(40)
        self.assertEqual(validate_dataset(data), [])
        self.assertEqual(validate_dataset(data[feature_columns()], training=False), [])
        self.assertTrue(validate_dataset(data[feature_columns()], training=True))
        forbidden = {"positive_reviews", "negative_reviews", "owners_lower", "owners_upper", "playtime_median_hours", "appid"}
        self.assertFalse(forbidden & set(feature_columns()))

    def test_invalid_numeric_and_derived_features(self):
        data = concept_features(["Indie"], ["crafting"])
        data.loc[0, "mechanic_count"] = 100
        data.loc[0, "language_count"] = -1
        data.loc[0, "log_price"] = np.inf
        errors = " ".join(validate_dataset(data, training=False))
        self.assertIn("finite", errors)
        self.assertIn("mechanic_count", errors)
        self.assertIn("language_count", errors)

    def test_duplicate_and_mislabeled_targets_rejected(self):
        data = synthetic_games(40)
        data.loc[1, "appid"] = data.loc[0, "appid"]
        data.loc[0, "owners_band"] = 3
        errors = " ".join(validate_dataset(data))
        self.assertIn("unique", errors)
        self.assertIn("Owner bands", errors)

    def test_concept_errors(self):
        for arguments in [([], []), (["Unknown"], []), (["Indie"], ["unknown"]), (["Indie"], [], -5)]:
            with self.assertRaises(ValueError):
                concept_features(*arguments)

    def test_developer_graph_handles_shared_teams_transitively(self):
        data = pd.DataFrame({"developer_group": ["A|B", "b|c", "C", "D"]})
        groups = developer_components(data)
        self.assertEqual(len(set(groups[:3])), 1)
        self.assertNotEqual(groups[0], groups[3])

    def test_developer_splits_are_independent_and_repeatable(self):
        data = synthetic_games(120)
        data["split_group"] = developer_components(data)
        parts = split_games(data, load_config())
        again = split_games(data, load_config())
        for key in parts:
            np.testing.assert_array_equal(parts[key], again[key])
        for a in parts:
            for b in parts:
                if a != b:
                    self.assertFalse(set(data.iloc[parts[a]].split_group) & set(data.iloc[parts[b]].split_group))

    def test_bootstrap_identical_models_does_not_claim_improvement(self):
        data = synthetic_games(120)
        data["split_group"] = developer_components(data)
        probabilities = np.full((len(data), 4), 0.25)
        result = paired_bootstrap(data, probabilities, probabilities, 42, 100)
        self.assertEqual(result["delta_log_loss"], 0)
        self.assertEqual(result["verdict"], "inconclusive")


class SteamTests(unittest.TestCase):
    def test_public_html_and_gate_detection(self):
        body = b'<div id="appHubAppName">Example</div><div id="game_area_description">Crafting</div><a class="app_tag">Crafting</a><img class="game_header_image_full" src="image.jpg">'
        result = parse_store_page(body)
        self.assertEqual(result["name"], "Example")
        self.assertEqual(result["tags"], ["Crafting"])
        self.assertFalse(result["page_unavailable_or_gated"])
        self.assertTrue(parse_store_page(b"<p>Age verification</p>")["page_unavailable_or_gated"])

    def test_review_aggregates_do_not_export_user_identifiers(self):
        reviews = [{"steam_purchase": True, "author": {"steamid": "private-id", "playtime_forever": 120}},
                   {"steam_purchase": True, "received_for_free": True, "author": {"playtime_forever": 600}},
                   {"steam_purchase": True, "author": {"playtime_forever": 0}}]
        result = summarize_reviews({"response": {"reviews": reviews}})
        self.assertEqual(result["sample_median_hours"], 2)
        self.assertFalse(result["is_retention"])
        self.assertNotIn("private-id", json.dumps(result))

    def test_cache_is_verified_and_refresh_preserves_original(self):
        with tempfile.TemporaryDirectory() as directory:
            client = CachedHttp(Path(directory))
            response = Mock(status_code=200, content=b"first", headers={})
            client.session.get = Mock(return_value=response)
            url = "https://store.steampowered.com/api/appdetails?appids=1"
            with patch("game_concept.sources.time.sleep"):
                body, meta = client.get(url)
                self.assertEqual(client.get(url)[0], body)
                self.assertEqual(client.session.get.call_count, 1)
                client.refresh = True
                response.content = b"second"
                client.get(url)
            self.assertEqual((Path(directory) / "snapshots" / f"{meta['sha256']}.body").read_bytes(), b"first")
            client.refresh = False
            key = hashlib.sha256(url.encode()).hexdigest()
            (Path(directory) / f"{key}.body").write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "checksum"):
                client.get(url)

    def test_no_access_or_rate_limit_bypass(self):
        with tempfile.TemporaryDirectory() as directory:
            client = CachedHttp(Path(directory))
            for url in ["https://example.org/", "http://store.steampowered.com/", "https://user:pass@store.steampowered.com/"]:
                with self.assertRaises(ValueError):
                    client.get(url)
            denied = Mock(status_code=403)
            client.session.get = Mock(return_value=denied)
            with patch("game_concept.sources.time.sleep"), self.assertRaises(PermissionError):
                client.get("https://store.steampowered.com/app/1/")
            limited = Mock(status_code=429, headers={"Retry-After": "600"})
            client.session.get = Mock(return_value=limited)
            with patch("game_concept.sources.time.sleep"), self.assertRaisesRegex(RuntimeError, "pause"):
                client.get("https://store.steampowered.com/app/1/")

    def test_scraper_upserts_without_erasing_other_games(self):
        def fake_get(url):
            if "appdetails" in url:
                appid = re_search_appid(url)
                content = json.dumps({str(appid): {"success": True, "data": {"type": "game", "name": "Example", "about_the_game": "Craft your own weapons in a magical world.", "genres": [{"description": "RPG"}]}}}).encode()
            elif "GetAppReviews" in url:
                content = b'{"response": {"reviews": []}}'
            else:
                content = b'<div id="game_area_description">Craft your own weapons.</div><a class="app_tag">Crafting</a>'
            return content, {"captured_at_utc": "2026-10-07T00:00:00+00:00", "sha256": hashlib.sha256(content).hexdigest()}
        with tempfile.TemporaryDirectory() as directory, patch("game_concept.scraper.CachedHttp") as http:
            http.return_value.get.side_effect = fake_get
            first = scrape_games(Path(directory), load_config(), [100])
            second = scrape_games(Path(directory), load_config(), [200])
            self.assertEqual(first["successful"], 1)
            self.assertEqual(second["total_cached_games"], 2)
            data = pd.read_csv(Path(directory) / "data/processed/game_concept/live_games.csv")
            self.assertEqual(set(data.appid), {100, 200})
            self.assertIn("crafting", data.iloc[0].mechanics)


def re_search_appid(url):
    from urllib.parse import parse_qs, urlparse
    return int(parse_qs(urlparse(url).query)["appids"][0])


class ExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.output = Path(cls.temp.name)
        cls.data = synthetic_games()
        config = load_config()
        config["evaluation"].update(iterations=8, bootstrap_repeats=100, min_samples_leaf=5)
        cls.result = train_experiment(cls.data, config, cls.output)
        cls.bundle = load_bundle(cls.output / "models/concept_model.joblib")

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_all_models_and_finite_scores(self):
        metrics = self.result["market_metrics"]
        self.assertEqual(len(metrics), 5)
        self.assertTrue(np.isfinite(metrics.group_log_loss).all())
        self.assertEqual(len(self.result["playtime_metrics"]), 3)

    def test_forecast_is_calibrated_distribution_not_sales(self):
        prediction = forecast(concept_features(["Indie", "RPG"], ["crafting"]), self.bundle)
        self.assertAlmostEqual(sum(prediction["probabilities"]), 1)
        self.assertFalse(prediction["is_actual_sales_forecast"])
        self.assertFalse(prediction["is_retention_forecast"])
        self.assertGreaterEqual(prediction["playtime"]["high_hours"], prediction["playtime"]["low_hours"])
        self.assertTrue(set(prediction["analogues"].appid).issubset(set(self.bundle["reference"].appid)))

    def test_all_mechanics_abstains_instead_of_100_percent(self):
        prediction = forecast(concept_features(["Indie"], KEYS), self.bundle)
        self.assertEqual(prediction["support"]["status"], "abstain")
        self.assertIsNone(prediction["probabilities"])

    def test_unknown_model_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown fitted model"):
            forecast(concept_features(["Indie"], []), self.bundle, "made_up")

    def test_component_comparison_checks_full_model_disagreement(self):
        rows = compare_components(concept_features(["Indie", "RPG"], ["crafting"]), self.bundle)
        self.assertEqual(len(rows), len(KEYS))
        accepted = rows[rows.support_status.ne("abstain")]
        self.assertTrue(accepted.full_models.eq(3).all())
        self.assertTrue((accepted.model_min_delta_pp <= accepted.model_max_delta_pp).all())
        self.assertTrue(rows.loc[rows.support_status.eq("abstain"), "delta_pp"].isna().all())

    def test_genre_only_comparison_does_not_invent_mechanic_effect(self):
        rows = compare_components(concept_features(["Indie", "RPG"], ["crafting"]), self.bundle, "genre_only")
        self.assertTrue(rows.loc[rows.support_status.ne("abstain"), "delta_pp"].eq(0).all())

    def test_component_disagreement_is_not_a_confident_recommendation(self):
        columns = feature_columns()
        position = columns.index("mechanic_crafting")
        class FixedMechanicModel:
            classes_ = np.arange(4)
            def __init__(self, sign):
                self.sign = sign
            def predict_proba(self, values):
                effect = self.sign * 0.1 * values[:, position]
                return np.column_stack([0.7 - effect, 0.2 + effect, np.full(len(values), 0.08), np.full(len(values), 0.02)])
        bundle = dict(self.bundle)
        bundle["all_market"] = dict(self.bundle["all_market"])
        for name, sign in [("hgb_additive", 1), ("logistic_additive", -1), ("hgb_interactions", 1)]:
            bundle["all_market"][name] = {"model": FixedMechanicModel(sign), "columns": columns}
        result = compare_components(concept_features(["Indie", "RPG"], []), bundle, "hgb_additive")
        row = result[result.mechanic.eq("crafting")].iloc[0]
        self.assertEqual(row.agreement, "mixed")
        self.assertAlmostEqual(row.model_min_delta_pp, -10)
        self.assertAlmostEqual(row.model_max_delta_pp, 10)

    def test_numeric_string_prediction_is_normalized(self):
        data = concept_features(["Indie", "RPG"], ["crafting"])
        regular = forecast(data, self.bundle)
        stringified = forecast(data.astype(str), self.bundle)
        np.testing.assert_allclose(regular["probabilities"], stringified["probabilities"])

    def test_calibration_diagnostics_have_both_stages_and_three_thresholds(self):
        curves = self.result["reliability"]
        self.assertEqual(set(curves.stage), {"before", "after"})
        self.assertEqual(set(curves.threshold), {20000, 100000, 500000})
        self.assertEqual(len(self.result["calibration_metrics"]), 30)

    def test_permutation_importance_preserves_derived_mechanic_count(self):
        importance = self.result["importance"]
        count = importance[importance.feature.eq("mechanic_count")].iloc[0]
        self.assertTrue(count.derived_input)
        self.assertAlmostEqual(count.log_loss_increase, 0)
        self.assertAlmostEqual(count["std"], 0)

    def test_unobserved_genre_abstains(self):
        prediction = forecast(concept_features(["Sports"], ["crafting"]), self.bundle)
        self.assertEqual(prediction["support"]["status"], "abstain")
        self.assertIn("unobserved_genre", prediction["support"]["reasons"])

    def test_lime_is_real_and_finite(self):
        explanation, meta = explain_lime(concept_features(["Indie", "RPG"], ["crafting"]), self.bundle, "hgb_additive", samples=500)
        self.assertGreater(len(explanation), 0)
        self.assertTrue(np.isfinite(explanation.local_weight).all())
        self.assertTrue(np.isfinite(meta["local_surrogate_r2"]))

    def test_custom_dataset_report_does_not_claim_official_provenance(self):
        write_reports(self.data, self.result, self.output)
        report = (self.output / "research_report_ru.md").read_text(encoding="utf-8")
        self.assertIn("Пользовательский набор", report)
        self.assertNotIn("Данные подготовлены из снимка", report)
        self.assertIsNone(self.result["metadata"]["reference_dataset"])


if __name__ == "__main__":
    unittest.main()
