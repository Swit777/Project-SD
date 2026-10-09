from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .dataset import BAND_LABELS, GAME_GENRES, GENRE_COLUMNS, MECHANIC_COLUMNS, feature_columns, validate_dataset
from .mechanics import KEYS, LABELS
from .models import probabilities
from .analogues import find_analogues


def concept_features(genres: list[str], mechanics: list[str], price_usd: float = 14.99,
                     reference_age_years: float = 3.0, language_count: int = 2,
                     windows=True, mac=False, linux=False) -> pd.DataFrame:
    if not genres or any(g not in GAME_GENRES for g in genres):
        raise ValueError("Choose supported Steam genres")
    if any(m not in KEYS for m in mechanics):
        raise ValueError("Unknown mechanic identifier")
    if not np.isfinite(price_usd) or price_usd <= 0 or not np.isfinite(reference_age_years) or reference_age_years <= 0:
        raise ValueError("Positive finite price and reference game age are required")
    if not isinstance(language_count, int) or language_count < 1:
        raise ValueError("At least one supported language is required")
    selected = set(mechanics)
    row = {"log_price": np.log1p(price_usd), "log_age": np.log1p(reference_age_years),
           "language_count": language_count, "windows": int(windows), "mac": int(mac), "linux": int(linux),
           "mechanic_count": len(selected)}
    row.update({column: int(genre in genres) for column, genre in zip(GENRE_COLUMNS, GAME_GENRES)})
    row.update({f"mechanic_{key}": int(key in selected) for key in KEYS})
    data = pd.DataFrame([row], columns=feature_columns())
    errors = validate_dataset(data, training=False)
    if errors:
        raise ValueError("; ".join(errors))
    return data


def assess_support(data: pd.DataFrame, bundle: dict) -> dict:
    row = data.iloc[0]
    support = bundle["metadata"]["support"]
    selected = np.flatnonzero(row[MECHANIC_COLUMNS].to_numpy(dtype=int))
    counts = np.asarray(support["mechanic_counts"])
    pairs = np.asarray(support["pair_counts"])
    rare = [KEYS[i] for i in selected if counts[i] < 15]
    absent_pairs = [(KEYS[a], KEYS[b], int(pairs[a, b])) for i, a in enumerate(selected) for b in selected[i + 1:] if pairs[a, b] < 5]
    pair_count = len(selected) * (len(selected) - 1) / 2
    reasons = []
    limit = min(support["max_mechanic_count"], max(8, int(support["p95_mechanic_count"] + 3)))
    if len(selected) > limit:
        reasons.append("mechanic_density_outside_reference")
    if any(counts[i] < 3 for i in selected):
        reasons.append("unobserved_mechanic")
    if pair_count and len(absent_pairs) / pair_count > 0.5:
        reasons.append("unsupported_combination")
    genre_ids = np.flatnonzero(row[GENRE_COLUMNS].to_numpy(dtype=int))
    genre_counts = np.asarray(support["genre_counts"])
    genre_mechanic = np.asarray(support["genre_mechanic_counts"])
    if not any(genre_counts[i] >= 3 for i in genre_ids):
        reasons.append("unobserved_genre")
    rare_genre_mechanics = [KEYS[i] for i in selected if genre_mechanic[genre_ids, i].max() < 5]
    if len(selected) and len(rare_genre_mechanics) / len(selected) > 0.5:
        reasons.append("unsupported_genre_mechanics")
    price, age = np.expm1(row.log_price), np.expm1(row.log_age)
    if price < support["price_range"][0] * 0.5 or price > support["price_range"][1] * 1.5:
        reasons.append("price_outside_reference")
    if age < support["age_range_years"][0] * 0.5 or age > support["age_range_years"][1] * 1.5:
        reasons.append("age_outside_reference")
    return {"status": "abstain" if reasons else "limited" if rare or absent_pairs or rare_genre_mechanics else "supported",
            "reasons": reasons, "rare_mechanics": rare, "rare_pairs": absent_pairs[:30],
            "rare_genre_mechanics": rare_genre_mechanics,
            "density_limit": limit, "selected_mechanics": len(selected),
            "definition": "Training support is not a measurement of design quality"}


def nearest_analogues(data: pd.DataFrame, reference: pd.DataFrame, limit=8) -> pd.DataFrame:
    return find_analogues(data, reference, limit=limit)[0]


def forecast(data: pd.DataFrame, bundle: dict, model_name: str | None = None) -> dict:
    errors = validate_dataset(data, training=False)
    if errors or len(data) != 1:
        raise ValueError("; ".join(errors) if errors else "One concept is required")
    if model_name is not None and model_name not in bundle["all_market"]:
        raise ValueError("Unknown fitted model")
    data = data.copy()
    data[feature_columns()] = data[feature_columns()].apply(pd.to_numeric)
    support = assess_support(data, bundle)
    result = {"support": support, "model": model_name or bundle["metadata"]["selected_model"],
              "analogues": nearest_analogues(data, bundle["reference"]), "probabilities": None, "playtime": None,
              "is_actual_sales_forecast": False, "is_retention_forecast": False}
    if support["status"] == "abstain":
        return result
    market = bundle["all_market"].get(model_name, bundle["market"]) if model_name else bundle["market"]
    with threadpool_limits(limits=1):
        probs = probabilities(market["model"], data[market["columns"]].to_numpy(dtype=float))[0]
        result["probabilities"] = probs.tolist()
        result["band_labels"] = BAND_LABELS
        result["p_at_least_20k"] = float(probs[1:].sum())
        result["p_at_least_100k"] = float(probs[2:].sum())
        result["p_at_least_500k"] = float(probs[3])
        if bundle.get("playtime"):
            model = bundle["playtime"]
            value = max(float(model["model"].predict(data[model["columns"]].to_numpy(dtype=float))[0]), 0)
            radius = model["radius"]
            result["playtime"] = {"median_hours_estimate": float(np.expm1(value)),
                "low_hours": float(np.expm1(max(value - radius, 0))), "high_hours": float(np.expm1(value + radius)),
                "meaning": "Available SteamSpy median-hours estimate for a comparable game; not owner retention"}
    return result


def compare_components(data: pd.DataFrame, bundle: dict, model_name: str | None = None) -> pd.DataFrame:
    baseline = forecast(data, bundle, model_name)
    if baseline["probabilities"] is None:
        raise ValueError("Cannot compare components of an unsupported baseline")
    data = data[feature_columns()].apply(pd.to_numeric).copy()
    rows, scenarios = [], []
    for key in KEYS:
        variant = data.copy()
        variant.loc[variant.index[0], f"mechanic_{key}"] = 1 - int(data.iloc[0][f"mechanic_{key}"])
        variant["mechanic_count"] = variant[MECHANIC_COLUMNS].sum(axis=1)
        support = assess_support(variant, bundle)
        rows.append({"mechanic": key, "change": "remove" if data.iloc[0][f"mechanic_{key}"] else "add",
                     "support_status": support["status"], "support_reasons": ", ".join(support["reasons"]),
                     "delta_pp": np.nan, "model_min_delta_pp": np.nan, "model_max_delta_pp": np.nan,
                     "agreement": "unavailable", "full_models": 0})
        scenarios.append(variant)
    batch = pd.concat([data] + scenarios, ignore_index=True)
    primary = model_name or bundle["metadata"]["selected_model"]
    full_names = [name for name in ["logistic_additive", "hgb_additive", "hgb_interactions"] if name in bundle["all_market"]]
    if not full_names:
        raise ValueError("Full-feature models are required for component comparison")
    predictions = {}
    with threadpool_limits(limits=1):
        for name in dict.fromkeys([primary] + full_names):
            model = bundle["all_market"][name]
            predictions[name] = probabilities(model["model"], batch[model["columns"]].to_numpy(dtype=float))[:, 1:].sum(axis=1)
    for i, row in enumerate(rows, 1):
        if row["support_status"] == "abstain":
            continue
        delta = (predictions[primary][i] - predictions[primary][0]) * 100
        differences = np.array([(predictions[name][i] - predictions[name][0]) * 100 for name in full_names])
        positive, negative = np.any(differences > 0.25), np.any(differences < -0.25)
        agreement = "mixed" if positive and negative else "near_zero" if not positive and not negative else "all_positive" if np.all(differences > 0.25) else "all_negative" if np.all(differences < -0.25) else "partly_neutral"
        row.update(delta_pp=float(delta), model_min_delta_pp=float(differences.min()), model_max_delta_pp=float(differences.max()),
                   agreement=agreement, full_models=len(full_names))
    return pd.DataFrame(rows)


def explain_lime(data: pd.DataFrame, bundle: dict, model_name: str | None = None, samples=1800) -> tuple[pd.DataFrame, dict]:
    from lime.lime_tabular import LimeTabularExplainer

    if forecast(data, bundle, model_name)["support"]["status"] == "abstain":
        raise ValueError("No model explanation for an unsupported concept")
    data = data[feature_columns()].apply(pd.to_numeric)
    market = bundle["all_market"].get(model_name, bundle["market"]) if model_name else bundle["market"]
    columns = market["columns"]
    categorical = [i for i, c in enumerate(columns) if c in GENRE_COLUMNS + MECHANIC_COLUMNS + ["windows", "mac", "linux"]]
    feature_names = [LABELS.get(c.removeprefix("mechanic_"), c.removeprefix("genre_")) for c in columns]
    reference = bundle["reference"][columns].to_numpy(dtype=float)
    explainer = LimeTabularExplainer(reference, feature_names=feature_names, class_names=BAND_LABELS,
        categorical_features=categorical, discretize_continuous=True, mode="classification",
        random_state=bundle["metadata"]["config"]["seed"])
    def predict(values):
        if "mechanic_count" in columns:
            values = values.copy()
            values[:, columns.index("mechanic_count")] = values[:, [columns.index(c) for c in MECHANIC_COLUMNS]].sum(axis=1)
        return probabilities(market["model"], values)
    with threadpool_limits(limits=1):
        explanation = explainer.explain_instance(data[columns].to_numpy(dtype=float)[0], predict,
            labels=(1,), num_features=12, num_samples=samples)
    frame = pd.DataFrame(explanation.as_list(label=1), columns=["feature_condition", "local_weight"])
    return frame, {"method": "LIME", "explained_band": BAND_LABELS[1], "local_surrogate_r2": float(explanation.score),
                   "samples": samples, "limitation": "Local approximation, not causal evidence; perturbations can be outside observed combinations"}


def load_bundle(path: Path) -> dict:
    return joblib.load(path)
