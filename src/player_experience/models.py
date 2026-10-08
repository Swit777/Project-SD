from __future__ import annotations

import importlib.metadata
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import BayesianRidge, Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .evaluation import paired_player_bootstrap, player_weights, regression_metrics, split_players
from .config import ROOT
from .features import CATEGORICAL_FEATURES, IDENTIFIERS, TARGET, feature_sets, validate_dataset
from .source import ARTICLE_URL, SOURCE_DOI, sha256_file


MODEL_SPECS = {
    "mean_baseline": ("mean", "aggregate"),
    "median_baseline": ("median", "aggregate"),
    "ridge_aggregate": ("ridge", "aggregate"),
    "random_forest_aggregate": ("forest", "aggregate"),
    "random_forest_dynamic": ("forest", "dynamic"),
    "bayesian_ridge_dynamic": ("bayesian", "dynamic"),
}


def _preprocessor(columns: list[str]) -> ColumnTransformer:
    categorical = [c for c in columns if c in CATEGORICAL_FEATURES]
    numeric = [c for c in columns if c not in categorical]
    return ColumnTransformer([
        ("numeric", Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
                              ("scale", StandardScaler())]), numeric),
        ("categorical", Pipeline([("impute", SimpleImputer(strategy="constant", fill_value="unknown")),
                                  ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), categorical),
    ])


def make_model(kind: str, columns: list[str], config: dict) -> Pipeline:
    settings = config["evaluation"]
    if kind == "forest":
        model = RandomForestRegressor(n_estimators=settings["trees"], min_samples_leaf=settings["min_samples_leaf"],
                                      max_depth=14, random_state=config["seed"], n_jobs=1)
    elif kind == "ridge":
        model = Ridge(alpha=30.0)
    elif kind == "bayesian":
        model = BayesianRidge()
    else:
        model = DummyRegressor(strategy="median" if kind == "median" else "mean")
    return Pipeline([("preprocess", _preprocessor(columns)), ("model", model)])


def _normalise(data: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    data = data.copy().reset_index(drop=True)
    for column in CATEGORICAL_FEATURES:
        data[column] = data[column].fillna("unknown").astype(str)
    for column in feature_sets(windows)["dynamic"]:
        if column not in CATEGORICAL_FEATURES:
            data[column] = pd.to_numeric(data[column], errors="coerce")
    if TARGET in data:
        data[TARGET] = pd.to_numeric(data[TARGET], errors="coerce")
    if "player_id" in data:
        data["player_id"] = data["player_id"].astype(str)
    return data


def fit_model(model: Pipeline, data: pd.DataFrame, columns: list[str]) -> Pipeline:
    return model.fit(data[columns], data[TARGET], model__sample_weight=player_weights(data))


def train_experiment(data: pd.DataFrame, output: Path, config: dict, input_path: Path | None = None) -> dict:
    windows = config["features"]["windows_minutes"]
    errors = validate_dataset(data, windows)
    if errors:
        raise ValueError("; ".join(errors))
    if data[TARGET].nunique() < 2:
        raise ValueError("At least two distinct enjoyment ratings are required")
    original = input_path is not None and input_path.resolve() == (ROOT / "data/processed/player_experience.csv").resolve()
    preparation_audit = None
    if original:
        audit_path = ROOT / "data/processed/player_experience_audit.json"
        if not audit_path.exists():
            raise ValueError("Preparation audit missing; rerun prepare or use a separate input path")
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        if audit.get("data_sha256") != sha256_file(input_path):
            raise ValueError("Prepared data differs from its audit; rerun prepare or use a separate input path")
        preparation_audit = {k: v for k, v in audit.items() if k != "selected_players"}
    data = _normalise(data, windows)
    output.mkdir(parents=True, exist_ok=True)
    (output / "models").mkdir(exist_ok=True)
    settings = config["evaluation"]
    split = split_players(data, config["seed"], settings["test_fraction"], settings["validation_fraction"])
    train, validation, test = [data.iloc[split[name]] for name in ["train", "validation", "test"]]
    sets = feature_sets(windows)
    predictions = test[IDENTIFIERS + [TARGET]].copy()
    validation_metrics, test_metrics, fitted = {}, {}, {}
    for name, (kind, feature_set) in MODEL_SPECS.items():
        print(f"Training {name} ({len(train):,} observations)", flush=True)
        columns = sets[feature_set]
        model = fit_model(make_model(kind, columns, config), train, columns)
        val_prediction = np.clip(model.predict(validation[columns]), 0, 100)
        test_prediction = np.clip(model.predict(test[columns]), 0, 100)
        validation_metrics[name] = regression_metrics(validation, val_prediction)
        test_metrics[name] = regression_metrics(test, test_prediction)
        predictions[name] = test_prediction
        fitted[name] = model
    best_name = min(validation_metrics, key=lambda name: validation_metrics[name]["player_mae"])
    best_columns = sets[MODEL_SPECS[best_name][1]]
    best = fitted[best_name]
    val_residual = np.abs(validation[TARGET].to_numpy() - np.clip(best.predict(validation[best_columns]), 0, 100))
    interval_radius = float(np.quantile(val_residual, 0.90, method="higher"))
    predictions["prediction"] = predictions[best_name]
    predictions["interval_low"] = np.maximum(predictions["prediction"] - interval_radius, 0)
    predictions["interval_high"] = np.minimum(predictions["prediction"] + interval_radius, 100)
    interval_coverage = float(((predictions[TARGET] >= predictions["interval_low"]) &
                               (predictions[TARGET] <= predictions["interval_high"])).mean())
    hypothesis = paired_player_bootstrap(test, predictions["random_forest_aggregate"].to_numpy(),
                                        predictions["random_forest_dynamic"].to_numpy(), config["seed"], settings["bootstrap_repeats"])
    print(f"Primary comparison: delta MAE={hypothesis['delta_mae']:.3f}, verdict={hypothesis['verdict']}", flush=True)
    split_records = []
    split_counts = {}
    for name, indices in split.items():
        players = sorted(data.iloc[indices]["player_id"].unique())
        split_counts[name] = {"observations": len(indices), "players": len(players)}
        split_records.extend({"player_id": player, "split": name} for player in players)
    pd.DataFrame(split_records).to_csv(output / "player_splits.csv", index=False)
    development = data.iloc[np.concatenate([split["train"], split["validation"]])]
    cv = GroupKFold(n_splits=min(settings["cv_folds"], development["player_id"].nunique()))
    cv_rows = []
    for fold, (training, held_out) in enumerate(cv.split(development, groups=development["player_id"]), 1):
        for name in ["ridge_aggregate", "random_forest_aggregate", "random_forest_dynamic"]:
            kind, feature_set = MODEL_SPECS[name]
            columns = sets[feature_set]
            model = fit_model(make_model(kind, columns, config), development.iloc[training], columns)
            fold_prediction = np.clip(model.predict(development.iloc[held_out][columns]), 0, 100)
            cv_rows.append({"fold": fold, "model": name, **regression_metrics(development.iloc[held_out], fold_prediction)})
        print(f"Completed development CV fold {fold}", flush=True)
    cv_frame = pd.DataFrame(cv_rows)
    cv_frame.to_csv(output / "cross_validation.csv", index=False)
    window_rows = []
    for window in windows:
        columns = sets["aggregate"] + [c for c in sets["dynamic"] if c.startswith(f"w{window}_")]
        model = fit_model(make_model("forest", columns, config), train, columns)
        pred = np.clip(model.predict(test[columns]), 0, 100)
        window_rows.append({"window_minutes": window, **regression_metrics(test, pred)})
    pd.DataFrame(window_rows).to_csv(output / "window_ablation.csv", index=False)
    importance = permutation_importance(fitted["random_forest_dynamic"], test[sets["dynamic"]], test[TARGET],
                                       scoring="neg_mean_absolute_error", n_repeats=5, max_samples=min(len(test), 2000),
                                       random_state=config["seed"], n_jobs=1)
    importance_frame = pd.DataFrame({"feature": sets["dynamic"], "mae_increase": importance.importances_mean,
                                    "std": importance.importances_std}).sort_values("mae_increase", ascending=False)
    importance_frame.to_csv(output / "feature_importance.csv", index=False)
    predictions.to_csv(output / "test_predictions.csv", index=False)
    metrics_frame = pd.DataFrame([{"model": name, "feature_set": MODEL_SPECS[name][1],
                                   "validation_player_mae": validation_metrics[name]["player_mae"],
                                   **test_metrics[name]} for name in MODEL_SPECS])
    metrics_frame.to_csv(output / "metrics.csv", index=False)
    metadata = {"best_model": best_name, "selection_criterion": "validation player-balanced MAE",
                "splits": split_counts, "test_metrics": test_metrics, "validation_metrics": validation_metrics,
                "hypothesis": hypothesis, "interval": {"method": "validation absolute-residual percentile",
                    "nominal_fraction": 0.90, "radius": interval_radius, "test_coverage": interval_coverage,
                    "limitation": "Descriptive interval; repeated observations are not independent. No coverage guarantee."},
                "target_scale": [0, 100], "windows_minutes": windows, "config": config}
    metadata["dataset"] = {"origin": "PowerWash Simulator / pinned OSF release" if original else "User-provided prepared table; provenance not verified",
                           "license": "CC0-1.0" if original else "Not verified"}
    if preparation_audit is not None:
        metadata["preparation_audit"] = preparation_audit
    versions = {name: importlib.metadata.version(name) for name in ["numpy", "pandas", "scikit-learn", "scipy", "duckdb", "streamlit", "joblib", "matplotlib", "plotly", "openpyxl"]}
    code_files = sorted(Path(__file__).parent.glob("*.py"))
    metadata["reproducibility"] = {"source_doi": SOURCE_DOI, "article": ARTICLE_URL, "versions": versions,
        "input_sha256": sha256_file(input_path) if input_path else None,
        "code_sha256": {path.name: sha256_file(path) for path in code_files}, "seed": config["seed"]}
    joblib.dump({"model": best, "columns": best_columns, "metadata": metadata}, output / "models" / "best_model.joblib")
    (output / "run_manifest.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"metadata": metadata, "metrics": metrics_frame, "predictions": predictions, "cv": cv_frame,
            "importance": importance_frame, "window_ablation": pd.DataFrame(window_rows)}


def predict_dataset(data: pd.DataFrame, model_path: Path) -> pd.DataFrame:
    bundle = joblib.load(model_path)
    errors = validate_dataset(data, bundle["metadata"]["windows_minutes"], require_target=False)
    if errors:
        raise ValueError("; ".join(errors))
    normalised = _normalise(data, bundle["metadata"]["windows_minutes"])
    result = data[[c for c in IDENTIFIERS if c in data]].copy()
    result["predicted_enjoyment"] = np.clip(bundle["model"].predict(normalised[bundle["columns"]]), 0, 100)
    radius = bundle["metadata"]["interval"]["radius"]
    result["interval_low"] = np.maximum(result["predicted_enjoyment"] - radius, 0)
    result["interval_high"] = np.minimum(result["predicted_enjoyment"] + radius, 100)
    return result
