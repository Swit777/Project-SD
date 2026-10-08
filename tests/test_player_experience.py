from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from player_experience.config import load_config
from player_experience.evaluation import paired_player_bootstrap, player_weights, split_players
from player_experience.features import extract_features, feature_sets, validate_dataset
from player_experience.models import predict_dataset, train_experiment
from player_experience.prepare import _read_parquet, _save_parquet
from player_experience.reports import write_reports
from player_experience.source import download_sources, sha256_file
from unittest.mock import patch


def utc(seconds):
    return pd.Timestamp("2023-01-01", tz="UTC") + pd.Timedelta(seconds=seconds)


def telemetry():
    prompts = pd.DataFrame([{
        "response_id": "r1", "player_id": "p1", "timestamp": utc(300), "enjoyment": 70,
        "job": "VAN", "mode": "Career", "session_minutes": 5,
        "campaign_progress": 0.1, "level_progress": 0.4,
    }])
    events = pd.DataFrame([{
        "player_id": "p1", "timestamp": utc(t), "progress": i / 10,
        "washer": "basic", "nozzle": "wide",
    } for i, t in enumerate([30, 130, 250, 280, 300, 400])])
    boundaries = pd.DataFrame([{
        "player_id": "p1", "timestamp": utc(t), "kind": "job_started",
        "job": "OLD" if t == 0 else "VAN", "mode": "Career",
    } for t in [0, 120]])
    return prompts, events, boundaries


def training_fixture():
    rng = np.random.default_rng(51)
    columns = feature_sets([1, 5, 15])["dynamic"]
    data = pd.DataFrame({c: rng.uniform(0, 1, 72) for c in columns})
    for c in ["job", "mode", "washer", "nozzle"]:
        data[c] = "known"
    data["player_id"] = [f"p{i // 3:02d}" for i in range(72)]
    data["response_id"] = [f"r{i}" for i in range(72)]
    data["timestamp"] = [utc(i).isoformat() for i in range(72)]
    data["enjoyment"] = 30 + data["w1_rate"] * 45 + rng.normal(0, 2, 72)
    data.loc[0:4, "overall_gap_mean"] = np.nan
    return data


class FeatureTests(unittest.TestCase):
    def test_past_only_and_no_cross_episode(self):
        data, audit = extract_features(*telemetry(), [1, 5, 15])
        row = data.iloc[0]
        self.assertEqual(row.subtasks_total, 3)
        self.assertEqual(row.w1_count, 2)
        self.assertEqual(row.w5_count, 3)
        self.assertEqual(row.seconds_since_subtask, 20)
        self.assertEqual(row.episode_minutes, 3)
        self.assertAlmostEqual(row.w5_coverage, 0.6)
        self.assertTrue(audit["equal_timestamp_events_excluded"])

    def test_future_changes_do_not_change_features(self):
        prompts, events, boundaries = telemetry()
        expected, _ = extract_features(prompts, events, boundaries, [1, 5, 15])
        events.loc[events.timestamp >= utc(300), "progress"] = 999
        events.loc[events.timestamp >= utc(300), "washer"] = "future"
        actual, _ = extract_features(prompts, events, boundaries, [1, 5, 15])
        pd.testing.assert_frame_equal(expected, actual)

    def test_exited_and_mismatched_jobs_are_excluded(self):
        prompts, events, boundaries = telemetry()
        boundaries.loc[1, "kind"] = "job_exited"
        data, audit = extract_features(prompts, events, boundaries, [1])
        self.assertTrue(data.empty)
        self.assertEqual(audit["missing_episode"], 1)
        boundaries.loc[1, "kind"] = "job_started"
        boundaries.loc[1, "job"] = "DIFFERENT"
        _, audit = extract_features(prompts, events, boundaries, [1])
        self.assertEqual(audit["episode_context_mismatch"], 1)

    def test_no_completions_have_missing_not_fake_gaps(self):
        prompts, events, boundaries = telemetry()
        data, _ = extract_features(prompts, events.iloc[:0], boundaries, [1])
        self.assertEqual(data.iloc[0].w1_count, 0)
        self.assertTrue(pd.isna(data.iloc[0].w1_gap_mean))

    def test_parquet_cache_matches_raw_features(self):
        frames = telemetry()
        expected, _ = extract_features(*frames, [1, 5, 15])
        with tempfile.TemporaryDirectory() as tmp:
            cached = []
            for i, frame in enumerate(frames):
                path = Path(tmp) / f"{i}.parquet"
                _save_parquet(frame, path)
                cached.append(_read_parquet(path))
            actual, _ = extract_features(*cached, [1, 5, 15])
        pd.testing.assert_frame_equal(expected, actual)


class ProtocolTests(unittest.TestCase):
    def test_players_are_independent_and_splits_repeatable(self):
        data = training_fixture()
        split = split_players(data, 42, 0.2, 0.2)
        repeat = split_players(data, 42, 0.2, 0.2)
        groups = [set(data.iloc[indices].player_id) for indices in split.values()]
        self.assertFalse(groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
        self.assertEqual(sum(len(i) for i in split.values()), len(data))
        for name in split:
            np.testing.assert_array_equal(split[name], repeat[name])

    def test_each_player_has_equal_total_training_weight(self):
        data = pd.DataFrame({"player_id": ["a", "a", "a", "b", "c", "c"]})
        data["weight"] = player_weights(data)
        self.assertEqual(data.groupby("player_id").weight.sum().nunique(), 1)

    def test_bootstrap_is_paired_on_players_not_rows(self):
        data = pd.DataFrame({"player_id": ["a", "a", "a", "b"], "enjoyment": [0, 0, 0, 0]})
        result = paired_player_bootstrap(data, np.array([2, 2, 2, 6]), np.zeros(4), 42, 200)
        self.assertEqual(result["delta_mae"], 4)
        self.assertEqual(result["players"], 2)
        self.assertEqual(result["verdict"], "supported")
        equal = paired_player_bootstrap(data, np.zeros(4), np.zeros(4), 42, 100)
        self.assertEqual(equal["verdict"], "inconclusive")

    def test_invalid_data_rejected(self):
        data = training_fixture()
        self.assertFalse(validate_dataset(data, [1, 5, 15]))
        data.loc[0, "enjoyment"] = 101
        data.loc[1, "response_id"] = data.loc[0, "response_id"]
        data.loc[0, "w1_rate"] = np.inf
        errors = validate_dataset(data, [1, 5, 15])
        self.assertTrue(any("between 0 and 100" in e for e in errors))
        self.assertTrue(any("unique" in e for e in errors))
        self.assertTrue(any("finite" in e for e in errors))
        self.assertTrue(validate_dataset(data.iloc[:0], [1, 5, 15], require_target=False))

    def test_invalid_configuration_rejected(self):
        import json
        config = load_config()
        config["evaluation"]["bootstrap_repeats"] = 0
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text(json.dumps(config))
            with self.assertRaises(ValueError):
                load_config(path)

    def test_training_reports_and_unseen_category_prediction(self):
        config = load_config()
        config["evaluation"].update(trees=8, min_samples_leaf=2, bootstrap_repeats=100, cv_folds=2)
        data = training_fixture()
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            result = train_experiment(data, output, config)
            write_reports(data, result, output)
            self.assertEqual(len(result["metrics"]), 6)
            self.assertEqual(result["metadata"]["dataset"]["license"], "Not verified")
            self.assertEqual(len(result["cv"]), 6)
            self.assertTrue((output / "research_report_ru.md").exists())
            self.assertTrue((output / "figures/model_comparison.png").exists())
            incoming = data[feature_sets([1, 5, 15])["dynamic"]].head(5).copy()
            incoming["job"] = "unseen_job"
            prediction = predict_dataset(incoming, output / "models/best_model.joblib")
            self.assertEqual(len(prediction), 5)
            self.assertTrue(prediction.predicted_enjoyment.between(0, 100).all())
            self.assertTrue((prediction.interval_low <= prediction.predicted_enjoyment).all())
            self.assertTrue((prediction.interval_high >= prediction.predicted_enjoyment).all())

    def test_existing_source_checksum_is_verified(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            source = directory / "tiny.csv"
            source.write_bytes(b"x\n1\n")
            spec = {"tiny.csv": {"url": "https://example.invalid", "bytes": source.stat().st_size,
                                  "sha256": sha256_file(source)}}
            with patch("player_experience.source.FILES", spec):
                manifest = download_sources(directory)
                self.assertEqual(manifest["files"]["tiny.csv"]["sha256"], sha256_file(source))
                source.write_bytes(b"x\n2\n")
                with self.assertRaisesRegex(ValueError, "wrong checksum"):
                    download_sources(directory)


if __name__ == "__main__":
    unittest.main()
