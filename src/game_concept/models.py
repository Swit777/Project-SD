from __future__ import annotations

import importlib.metadata
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.frozen import FrozenEstimator
from sklearn.inspection import permutation_importance
from sklearn.linear_model import BayesianRidge, LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .dataset import GENRE_COLUMNS, MECHANIC_COLUMNS, feature_columns, validate_dataset
from .evaluation import classification_metrics, developer_components, observation_weights, paired_bootstrap, split_games, threshold_reliability
from .mechanics import TAXONOMY_VERSION
from .sources import DATASET_ID, REVISION, sha256_file

MODEL_SPECS = {"prior": "genre", "genre_only": "genre", "logistic_additive": "full",
               "hgb_additive": "full", "hgb_interactions": "full"}


def make_classifier(name: str, config: dict):
    if name == "prior":
        return DummyClassifier(strategy="prior")
    if name == "logistic_additive":
        return make_pipeline(StandardScaler(), LogisticRegression(C=0.4, max_iter=1000, random_state=config["seed"]))
    settings = config["evaluation"]
    return HistGradientBoostingClassifier(max_iter=settings["iterations"], max_leaf_nodes=15,
        min_samples_leaf=settings["min_samples_leaf"], l2_regularization=5, learning_rate=0.07,
        interaction_cst="no_interactions" if name == "hgb_additive" else None,
        early_stopping=False, random_state=config["seed"])


def fit_classifier(model, x, y, weights):
    if isinstance(model, Pipeline):
        model.fit(x, y, logisticregression__sample_weight=weights)
    else:
        model.fit(x, y, sample_weight=weights)
    return model


def probabilities(model, x):
    raw = model.predict_proba(x)
    result = np.zeros((len(x), 4))
    result[:, model.classes_.astype(int)] = raw
    return result


def mechanic_permutation_importance(model, data: pd.DataFrame, columns: list[str], seed: int) -> pd.DataFrame:
    sample = data.sample(n=min(len(data), 1200), random_state=seed)
    def consistent_score(estimator, values, labels):
        values = values.copy()
        # Keep the derived count valid, and compare on exactly the same sample.
        values[:, columns.index("mechanic_count")] = values[:, [columns.index(c) for c in MECHANIC_COLUMNS]].sum(axis=1)
        return -log_loss(labels, probabilities(estimator, values), labels=list(range(4)))
    importance = permutation_importance(model, sample[columns].to_numpy(dtype=float),
        sample.owners_band.to_numpy(dtype=int), scoring=consistent_score, n_repeats=3,
        random_state=seed, n_jobs=1)
    frame = pd.DataFrame({"feature": columns, "log_loss_increase": importance.importances_mean,
                          "std": importance.importances_std}).sort_values("log_loss_increase", ascending=False)
    frame["derived_input"] = frame.feature.eq("mechanic_count")
    frame["evaluation_games"] = len(sample)
    return frame


def train_experiment(data: pd.DataFrame, config: dict, output: Path, input_path: Path | None = None) -> dict:
    errors = validate_dataset(data)
    if errors:
        raise ValueError("; ".join(errors))
    official_path = Path(__file__).resolve().parents[2] / "data/processed/game_concept/games.csv"
    official = input_path is not None and input_path.resolve() == official_path
    input_hash = sha256_file(input_path) if input_path else None
    if official:
        audit = json.loads((official_path.parent / "audit.json").read_text(encoding="utf-8"))
        if input_hash != audit["data_sha256"]:
            raise ValueError("Prepared dataset checksum differs from its audit; use a separate path for custom data")
        if not data.equals(pd.read_csv(input_path)):
            raise ValueError("Training data differs from the declared input file")
    data = data.copy().reset_index(drop=True)
    for column in feature_columns() + ["owners_band", "playtime_median_hours"]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data["split_group"] = developer_components(data)
    if data.split_group.nunique() < 20 or len(data) < 100:
        raise ValueError("Not enough independent games/developers")
    split = split_games(data, config)
    parts = {name: data.iloc[indices] for name, indices in split.items()}
    for name in ["train", "validation", "calibration"]:
        if set(parts[name].owners_band.unique()) != set(range(4)):
            raise ValueError(f"All four owner bands are required in {name}; increase the dataset")
    output.mkdir(parents=True, exist_ok=True)
    (output / "models").mkdir(exist_ok=True)
    metrics, fitted, test_predictions = [], {}, {}
    reliability_rows, calibration_rows = [], []
    uncalibrated_validation = {}
    with threadpool_limits(limits=1):
        for name, mode in MODEL_SPECS.items():
            columns = feature_columns(mode)
            x_train = parts["train"][columns].to_numpy(dtype=float)
            model = fit_classifier(make_classifier(name, config), x_train, parts["train"].owners_band.to_numpy(dtype=int), observation_weights(parts["train"]))
            val = probabilities(model, parts["validation"][columns].to_numpy(dtype=float))
            uncalibrated_validation[name] = classification_metrics(parts["validation"], val)
            raw_test = probabilities(model, parts["test"][columns].to_numpy(dtype=float))
            if name != "prior":
                calibrated = CalibratedClassifierCV(FrozenEstimator(model), method="sigmoid", n_jobs=1)
                calibrated.fit(parts["calibration"][columns].to_numpy(dtype=float), parts["calibration"].owners_band.to_numpy(dtype=int))
            else:
                calibrated = model
            test_prob = probabilities(calibrated, parts["test"][columns].to_numpy(dtype=float))
            test_predictions[name] = test_prob
            fitted[name] = {"model": calibrated, "columns": columns}
            for stage, probs in [("before", raw_test), ("after", test_prob)]:
                curve, calibration_summary = threshold_reliability(parts["test"], probs)
                reliability_rows.append(curve.assign(model=name, stage=stage))
                calibration_rows.append(calibration_summary.assign(model=name, stage=stage,
                    group_log_loss=classification_metrics(parts["test"], probs)["group_log_loss"]))
            metrics.append({"model": name, "validation_group_log_loss": uncalibrated_validation[name]["group_log_loss"],
                            **classification_metrics(parts["test"], test_prob)})
            print(f"Trained {name}", flush=True)
        best_name = min(uncalibrated_validation, key=lambda name: uncalibrated_validation[name]["group_log_loss"])
        hypothesis = paired_bootstrap(parts["test"], test_predictions["hgb_additive"], test_predictions["hgb_interactions"],
                                      config["seed"], config["evaluation"]["bootstrap_repeats"])
        print(f"Interaction hypothesis: {hypothesis['verdict']}, delta={hypothesis['delta_log_loss']:.4f}", flush=True)
        full = fitted["hgb_interactions"]
        importance_frame = mechanic_permutation_importance(full["model"], parts["test"], full["columns"], config["seed"])
        # A chronological backtest tests a different, harder shift; it does not select the model.
        chronological = data.sort_values(["release_date", "appid"])
        time_test = chronological.iloc[int(len(data) * 0.8):]
        time_train = chronological.iloc[:int(len(data) * 0.8)]
        time_train = time_train[~time_train.split_group.isin(time_test.split_group)]
        temporal_rows = []
        if len(time_train) >= 100 and time_train.owners_band.nunique() == 4:
            for name in ["hgb_additive", "hgb_interactions"]:
                columns = feature_columns()
                model = fit_classifier(make_classifier(name, config), time_train[columns].to_numpy(dtype=float),
                                       time_train.owners_band.to_numpy(dtype=int), observation_weights(time_train))
                pred = probabilities(model, time_test[columns].to_numpy(dtype=float))
                temporal_rows.append({"model": name, "train_games": len(time_train), "test_games": len(time_test),
                                      "calibrated": False, **classification_metrics(time_test, pred)})
        playtime_bundle, playtime_rows, playtime_meta = train_playtime(parts, config)
    metrics_frame = pd.DataFrame(metrics)
    metrics_frame.to_csv(output / "market_metrics.csv", index=False)
    pd.concat(reliability_rows, ignore_index=True).to_csv(output / "reliability_bins.csv", index=False)
    pd.concat(calibration_rows, ignore_index=True).to_csv(output / "calibration_metrics.csv", index=False)
    importance_frame.to_csv(output / "feature_importance.csv", index=False)
    pd.DataFrame(temporal_rows).to_csv(output / "temporal_backtest.csv", index=False)
    pd.DataFrame(playtime_rows).to_csv(output / "playtime_metrics.csv", index=False)
    split_rows = []
    counts = {}
    for name, part in parts.items():
        counts[name] = {"games": len(part), "developer_groups": int(part.split_group.nunique()),
                        "playtime_available": int(part.playtime_median_hours.notna().sum())}
        split_rows.extend({"appid": int(row.appid), "developer_group": int(row.split_group), "split": name} for row in part.itertuples())
    pd.DataFrame(split_rows).to_csv(output / "splits.csv", index=False)
    predictions = parts["test"][["appid", "name", "owners_band", "owners_lower", "owners_upper", "split_group"]].copy()
    for name, probs in test_predictions.items():
        for band in range(4):
            predictions[f"{name}_p{band}"] = probs[:, band]
    predictions.to_csv(output / "test_predictions.csv", index=False)
    reference = parts["train"].copy()
    matrix = reference[MECHANIC_COLUMNS].to_numpy(dtype=int)
    genre_matrix = reference[GENRE_COLUMNS].to_numpy(dtype=int)
    support = {"games": len(reference), "mechanic_counts": matrix.sum(axis=0).tolist(),
               "genre_counts": genre_matrix.sum(axis=0).tolist(),
               "genre_mechanic_counts": (genre_matrix.T @ matrix).tolist(),
               "pair_counts": (matrix.T @ matrix).tolist(), "max_mechanic_count": int(reference.mechanic_count.max()),
               "p95_mechanic_count": float(reference.mechanic_count.quantile(0.95)),
               "price_range": np.expm1(reference.log_price.quantile([0.01, 0.99])).tolist(),
               "age_range_years": np.expm1(reference.log_age.quantile([0.01, 0.99])).tolist()}
    metadata = {"selected_model": best_name, "selection": "uncalibrated validation developer-balanced log loss",
        "calibration": "independent developer groups; sigmoid probabilities", "hypothesis": hypothesis,
        "calibration_diagnostics": "Independent test; 10 fixed bins, game-weighted threshold ECE/Brier; exploratory, not selection",
        "permutation_importance": "Derived mechanic_count recomputed for every perturbation; not an independent feature or causal effect",
        "splits": counts, "playtime": playtime_meta, "config": config, "taxonomy_version": TAXONOMY_VERSION,
        "support": support, "reference_dataset": DATASET_ID if official else None, "snapshot_revision": REVISION if official else None,
        "input_sha256": input_hash,
        "code_sha256": {path.name: sha256_file(path) for path in sorted(Path(__file__).parent.glob("*.py"))},
        "versions": {name: importlib.metadata.version(name) for name in ["numpy", "pandas", "scikit-learn", "scipy", "requests", "beautifulsoup4", "lime", "joblib", "threadpoolctl"]},
        "limitations": ["Owner-band probabilities predict SteamSpy estimates, not actual sales",
                        "Playtime is conditional on available positive source estimates; not retention",
                        "Snapshot features are post-release; this is not validated launch-time or annual sales forecasting",
                        "Scenario differences are observational predictions, not causal effects",
                        "Missing mechanics mean no evidence found, not demonstrated absence",
                        "Marketing, execution quality and budgets are unobserved"]}
    metadata["input_origin"] = "pinned Steam snapshot" if official else "user-provided; provenance unverified"
    bundle = {"market": fitted[best_name], "all_market": fitted, "playtime": playtime_bundle,
              "reference": reference, "metadata": metadata}
    temporary_model = output / "models/concept_model.joblib.tmp"
    joblib.dump(bundle, temporary_model)
    temporary_model.replace(output / "models/concept_model.joblib")
    (output / "run_manifest.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return {"metadata": metadata, "market_metrics": metrics_frame, "importance": importance_frame,
            "temporal": pd.DataFrame(temporal_rows), "playtime_metrics": pd.DataFrame(playtime_rows), "predictions": predictions,
            "reliability": pd.concat(reliability_rows, ignore_index=True), "calibration_metrics": pd.concat(calibration_rows, ignore_index=True)}


def train_playtime(parts: dict, config: dict):
    observed = {name: part[part.playtime_median_hours.notna() & part.playtime_median_hours.gt(0)] for name, part in parts.items()}
    counts = {name: len(part) for name, part in observed.items()}
    if min(counts.values()) < 12 or counts["train"] < 80:
        return None, [], {"status": "insufficient_data", "counts": counts, "is_retention": False}
    columns = feature_columns()
    settings = config["evaluation"]
    models = {"median": DummyRegressor(strategy="median"), "bayesian_ridge": make_pipeline(StandardScaler(), BayesianRidge()),
              "hgb_interactions": HistGradientBoostingRegressor(max_iter=settings["iterations"], max_leaf_nodes=15,
                 min_samples_leaf=settings["min_samples_leaf"], l2_regularization=5, early_stopping=False, random_state=config["seed"])}
    rows, fitted, validation_scores = [], {}, {}
    for name, model in models.items():
        x = observed["train"][columns].to_numpy(dtype=float)
        y = np.log1p(observed["train"].playtime_median_hours.to_numpy(dtype=float))
        weights = observation_weights(observed["train"])
        if name == "bayesian_ridge":
            model.fit(x, y, bayesianridge__sample_weight=weights)
        else:
            model.fit(x, y, sample_weight=weights)
        fitted[name] = model
        val_y = np.log1p(observed["validation"].playtime_median_hours.to_numpy(dtype=float))
        val_prediction = model.predict(observed["validation"][columns].to_numpy(dtype=float))
        validation_scores[name] = float(np.mean(np.abs(val_y - val_prediction)))
        test_y = np.log1p(observed["test"].playtime_median_hours.to_numpy(dtype=float))
        pred = np.maximum(model.predict(observed["test"][columns].to_numpy(dtype=float)), 0)
        rows.append({"model": name, "validation_mae_log_hours": validation_scores[name],
                     "test_mae_log_hours": float(np.mean(np.abs(test_y - pred))),
                     "test_mae_hours": float(np.mean(np.abs(np.expm1(test_y) - np.expm1(pred))))})
    best_name = min(validation_scores, key=validation_scores.get)
    best = fitted[best_name]
    calibration_y = np.log1p(observed["calibration"].playtime_median_hours.to_numpy(dtype=float))
    residual = np.abs(calibration_y - np.maximum(best.predict(observed["calibration"][columns].to_numpy(dtype=float)), 0))
    rank = min(int(np.ceil((len(residual) + 1) * 0.9)), len(residual))
    radius = float(np.sort(residual)[rank - 1])
    test_y = np.log1p(observed["test"].playtime_median_hours.to_numpy(dtype=float))
    test_prediction = np.maximum(best.predict(observed["test"][columns].to_numpy(dtype=float)), 0)
    coverage = float(np.mean(np.abs(test_y - test_prediction) <= radius))
    meta = {"status": "available", "selected_model": best_name, "counts": counts, "radius_log_hours": radius,
            "empirical_test_coverage": coverage, "nominal_fraction": 0.9, "is_retention": False,
            "interval": "split absolute residual on log-hours; empirical, developer observations can be dependent"}
    return {"model": best, "columns": columns, "radius": radius}, rows, meta
