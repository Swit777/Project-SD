import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from game_concept.annotation import create_annotation_tasks, current_annotation, evaluate_annotations, load_annotation, save_reviews
from game_concept.evaluation import threshold_reliability
from game_concept.dataset import feature_columns
from game_concept.models import mechanic_permutation_importance
from game_concept.predict import concept_features
from game_concept.mechanics import KEYS, extract_mechanics
from game_concept.sources import sha256_file


class ExtractionRegressionTests(unittest.TestCase):
    def test_negative_mention_does_not_hide_later_positive(self):
        flags, evidence = extract_mechanics("No crafting at launch. Later updates add crafting.", [], [], "source")
        self.assertEqual(flags["mechanic_crafting"], 1)
        self.assertEqual({e["negated"] for e in evidence if e["mechanic"] == "crafting"}, {False, True})

    def test_same_sentence_contrast_does_not_hide_positive(self):
        flags, _ = extract_mechanics("No crafting system, but you can craft weapons.", [], [], "source")
        self.assertEqual(flags["mechanic_crafting"], 1)

    def test_not_only_is_not_negation(self):
        flags, _ = extract_mechanics("Not only crafting but also procedural generation.", [], [], "source")
        self.assertEqual(flags["mechanic_crafting"], 1)
        self.assertEqual(flags["mechanic_procedural_generation"], 1)

    def test_repeated_mentions_are_bounded(self):
        flags, evidence = extract_mechanics("Crafting is available. " * 500, [], [], "source")
        self.assertEqual(flags["mechanic_crafting"], 1)
        self.assertEqual(len([e for e in evidence if e["mechanic"] == "crafting"]), 1)


class ReliabilityTests(unittest.TestCase):
    def test_permutation_subsampling_does_not_invent_importance_for_constant_model(self):
        from sklearn.dummy import DummyClassifier
        features = concept_features(["Indie"], ["crafting"])
        data = pd.concat([features] * 1500, ignore_index=True)
        data["owners_band"] = np.arange(len(data)) % 4
        model = DummyClassifier(strategy="prior").fit(np.zeros((116, len(feature_columns()))), [0] * 100 + [1] * 10 + [2] * 5 + [3])
        importance = mechanic_permutation_importance(model, data, feature_columns(), 42)
        self.assertTrue(np.allclose(importance.log_loss_increase, 0))
        self.assertTrue(importance.evaluation_games.eq(1200).all())

    def test_perfect_distributions_have_zero_error(self):
        data = pd.DataFrame({"owners_band": [0, 1, 2, 3], "split_group": [0, 0, 1, 2]})
        curve, metrics = threshold_reliability(data, np.eye(4))
        self.assertTrue(metrics.ece.eq(0).all())
        self.assertTrue(metrics.binary_brier.eq(0).all())
        self.assertTrue(curve.groupby("threshold").games.sum().eq(4).all())
        self.assertEqual(curve[curve.threshold.eq(20000) & curve.bin.eq(9)].games.iloc[0], 3)

    def test_uniform_predictions_expose_overconfidence(self):
        data = pd.DataFrame({"owners_band": [0] * 10, "split_group": range(10)})
        _, metrics = threshold_reliability(data, np.full((10, 4), 0.25))
        self.assertAlmostEqual(metrics.iloc[0].ece, 0.75)
        self.assertAlmostEqual(metrics.iloc[0].binary_brier, 0.75 ** 2)

    def test_invalid_distributions_rejected(self):
        data = pd.DataFrame({"owners_band": [0], "split_group": [0]})
        with self.assertRaises(ValueError):
            threshold_reliability(data, np.full((1, 4), 0.8))


class AnnotationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        data_dir = self.root / "data/processed/game_concept"
        cache = self.root / "data/interim/game_concept"
        data_dir.mkdir(parents=True)
        cache.mkdir(parents=True)
        rows, descriptions = [], []
        for i in range(8):
            rows.append({"appid": i + 1, **{f"mechanic_{key}": int(key == "crafting") for key in KEYS}})
            descriptions.append({"appid": i + 1, "name": f"Game {i}", "description": "Crafting is available.", "tags": ["Crafting"], "categories": [], "source_url": "https://store.steampowered.com/app/1/"})
        pd.DataFrame(rows).to_csv(data_dir / "games.csv", index=False)
        (cache / "catalog.jsonl").write_text("\n".join(json.dumps(d) for d in descriptions), encoding="utf-8")
        (data_dir / "audit.json").write_text(json.dumps({"data_sha256": sha256_file(data_dir / "games.csv"), "catalog_sha256": sha256_file(cache / "catalog.jsonl")}), encoding="utf-8")
        self.directory = create_annotation_tasks(self.root, games=3)
        self.tasks, _, _ = load_annotation(self.directory)

    def tearDown(self):
        self.temp.cleanup()

    def test_tasks_are_repeatable_blind_to_outcomes_and_empty(self):
        self.assertEqual(create_annotation_tasks(self.root, games=3), self.directory)
        self.assertEqual(current_annotation(self.root), self.directory)
        self.assertEqual(len(self.tasks), 3 * len(KEYS))
        self.assertNotIn("owners_band", self.tasks)
        summary, _ = evaluate_annotations(self.tasks, self.tasks[["task_id", "expert_label"]])
        self.assertEqual(summary["status"], "not_reviewed")
        self.assertEqual(summary["reviewed"], 0)
        self.assertIsNone(summary["precision"])

    def test_partial_labels_do_not_turn_other_tasks_into_negatives(self):
        reviews = self.tasks.head(1)[["task_id", "expert_label"]].copy()
        reviews["expert_label"] = "1"
        summary = save_reviews(self.directory, reviews)
        self.assertEqual(summary["reviewed"], 1)
        self.assertEqual(summary["unreviewed"], len(self.tasks) - 1)
        self.assertEqual(summary["precision"], 1)
        self.assertEqual(summary["fully_labeled_games"], 0)
        create_annotation_tasks(self.root, games=3)
        stored, _, _ = load_annotation(self.directory)
        self.assertEqual(stored.expert_label.eq("1").sum(), 1)

    def test_recall_detects_missed_claims_and_unclear_is_excluded(self):
        rows = self.tasks.head(3).copy()
        rows["predicted"] = [1, 0, 1]
        reviews = rows[["task_id", "expert_label"]].copy()
        reviews["expert_label"] = ["1", "1", "unclear"]
        summary, _ = evaluate_annotations(rows, reviews)
        self.assertEqual(summary["tp"], 1)
        self.assertEqual(summary["fn"], 1)
        self.assertEqual(summary["recall"], 0.5)
        self.assertEqual(summary["unclear"], 1)

    def test_duplicates_foreign_ids_and_invalid_labels_rejected(self):
        reviews = self.tasks.head(1)[["task_id", "expert_label"]].copy()
        for invalid in [pd.concat([reviews, reviews]), reviews.assign(task_id="foreign"), reviews.assign(expert_label=2)]:
            with self.assertRaises(ValueError):
                save_reviews(self.directory, invalid)

    def test_modified_source_and_task_files_rejected(self):
        path = self.directory / "tasks.csv"
        path.write_bytes(path.read_bytes() + b"modified")
        with self.assertRaisesRegex(ValueError, "checksum"):
            load_annotation(self.directory)
        source = self.root / "data/interim/game_concept/catalog.jsonl"
        source.write_bytes(b"modified")
        with self.assertRaisesRegex(ValueError, "audit"):
            create_annotation_tasks(self.root, games=3)


if __name__ == "__main__":
    unittest.main()
