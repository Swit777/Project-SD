from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .dataset import GENRE_COLUMNS, MECHANIC_COLUMNS
from .mechanics import CATALOG, KEYS
from .tag_similarity import tag_tokens

RETRIEVAL_VERSION = "3.0"


def structured_mode(categories, key):
    if isinstance(categories, str):
        try:
            categories = json.loads(categories)
        except (ValueError, TypeError):
            return False
    if not isinstance(categories, list):
        return False
    aliases = next(entry[3] for entry in CATALOG if entry[0] == key)
    return bool({str(c).casefold() for c in categories} & {c.casefold() for c in aliases})


def review_stats(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    counts = []
    for column in ["positive_reviews", "negative_reviews"]:
        values = pd.to_numeric(result.get(column, pd.Series(np.nan, index=result.index)), errors="coerce")
        counts.append(values.where(np.isfinite(values) & values.ge(0) & values.mod(1).eq(0)))
    result["review_count"] = counts[0] + counts[1]
    result["positive_share"] = counts[0] / result.review_count.where(result.review_count.gt(0))
    return result


def find_analogues(data: pd.DataFrame, reference: pd.DataFrame, *, limit: int | None = 12,
                   min_reviews: int = 0, min_coverage: float = 0.5, require_modes: bool = True,
                   price_factor: float | None = None, tag_scores: pd.Series | None = None,
                   required_tags: tuple[str, ...] = (), exclude_appid: int | None = None) -> tuple[pd.DataFrame, dict]:
    """Deterministic retrieval; outcome values never contribute to the similarity score."""
    if len(data) != 1 or min_reviews < 0 or not 0 <= min_coverage <= 1:
        raise ValueError("One concept and valid review/coverage thresholds are required")
    if limit is not None and (not isinstance(limit, int) or limit < 0):
        raise ValueError("limit must be a nonnegative integer or None")
    if price_factor is not None and (not np.isfinite(price_factor) or price_factor < 1):
        raise ValueError("price_factor must be finite and >= 1")
    row = data.iloc[0]
    result = review_stats(reference).reset_index(drop=True)
    required = GENRE_COLUMNS + MECHANIC_COLUMNS + ["log_price", "log_age"]
    missing = set(required) - set(result)
    if missing:
        raise ValueError("Missing analogue columns: " + ", ".join(sorted(missing)))
    flags = result[required].apply(pd.to_numeric, errors="coerce")
    a = row[MECHANIC_COLUMNS].to_numpy(dtype=float)
    b = flags[MECHANIC_COLUMNS].to_numpy(dtype=float)
    selected = int(a.sum())
    intersection = np.minimum(a, b).sum(axis=1)
    coverage = intersection / selected if selected else np.ones(len(result))
    overlap = intersection / np.maximum(np.maximum(a, b).sum(axis=1), 1)
    g = row[GENRE_COLUMNS].to_numpy(dtype=float)
    h = flags[GENRE_COLUMNS].to_numpy(dtype=float)
    weights = np.array([0.1 if c == "genre_indie" else 1.0 for c in GENRE_COLUMNS])
    genre_similarity = (np.minimum(g, h) * weights).sum(axis=1) / np.maximum((np.maximum(g, h) * weights).sum(axis=1), 0.1)
    core = [c for c in GENRE_COLUMNS if c != "genre_indie" and float(row[c]) == 1]
    genre_match = flags[core].sum(axis=1).gt(0) if core else (h * g).sum(axis=1) > 0
    mode_columns = [f"mechanic_{k}" for k in ["co_op", "pvp"] if float(row[f"mechanic_{k}"]) == 1]
    modes_match = pd.Series(True, index=result.index)
    if require_modes:
        for column in mode_columns:
            key = column.removeprefix("mechanic_")
            if f"mode_{key}_structured" in result:
                modes_match &= result[f"mode_{key}_structured"].eq(1)
            elif "categories_json" in result:
                modes_match &= result.categories_json.map(lambda c: structured_mode(c, key))
            else:
                # Unknown structured evidence must not confirm a multiplayer mode.
                modes_match &= False
    price_similarity = np.exp(-np.abs(flags.log_price - float(row.log_price)))
    age_similarity = np.exp(-np.abs(flags.log_age - float(row.log_age)))
    score = ((0.55 * (0.7 * coverage + 0.3 * overlap) + 0.25 * genre_similarity) if selected
             else 0.8 * genre_similarity) + 0.12 * price_similarity + 0.08 * age_similarity
    result["structural_similarity"] = score * 100
    result["tag_similarity"] = np.nan
    if tag_scores is not None:
        similarity = result.appid.map(tag_scores).fillna(0).clip(0, 1)
        result["tag_similarity"] = similarity
        score = 0.45 * score + 0.55 * similarity
    valid = np.isfinite(flags).all(axis=1) & flags[GENRE_COLUMNS + MECHANIC_COLUMNS].isin([0, 1]).all(axis=1)
    matched = valid & genre_match & modes_match & (coverage >= min_coverage)
    if exclude_appid is not None:
        matched &= result.appid.ne(exclude_appid)
    if required_tags:
        wanted = {t.casefold() for t in required_tags}
        matched &= result.get("tags_json", pd.Series("[]", index=result.index)).map(lambda v: wanted.issubset(tag_tokens(v)))
    if "appid" in data and "appid" in result:
        matched &= result.appid.ne(row.appid)
    if price_factor is not None:
        ratio = np.expm1(flags.log_price) / np.expm1(float(row.log_price))
        matched &= ratio.between(1 / price_factor, price_factor)
    reviewed = result.review_count.ge(min_reviews) if min_reviews else pd.Series(True, index=result.index)
    diagnostics = {"reference_games": len(result), "matched_before_reviews": int(matched.sum()),
                   "excluded_low_or_unknown_reviews": int((matched & ~reviewed).sum()),
                   "eligible_games": int((matched & reviewed).sum()), "min_reviews": min_reviews,
                   "min_coverage": min_coverage, "retrieval_version": RETRIEVAL_VERSION}
    diagnostics["tag_query_available"] = tag_scores is not None
    result["similarity"] = score * 100
    result["distance"] = 1 - score
    result["mechanic_coverage"] = coverage
    result["genre_similarity"] = genre_similarity
    result["price_similarity"] = price_similarity
    result["age_similarity"] = age_similarity
    if "appid" not in result:
        result["appid"] = result.index
    # AppID is only a stable tie-breaker, not an input to relevance.
    result = result.loc[matched & reviewed].sort_values(["similarity", "appid"], ascending=[False, True]).drop_duplicates("appid")
    diagnostics["eligible_games"] = len(result)
    if limit is not None:
        result = result.head(limit)
    values = b[result.index]
    result["matched_mechanics"] = [json.dumps([k for i, k in enumerate(KEYS) if a[i] and v[i] == 1]) for v in values]
    result["missing_mechanics"] = [json.dumps([k for i, k in enumerate(KEYS) if a[i] and v[i] != 1]) for v in values]
    return result.reset_index(drop=True), diagnostics


def owner_evidence(lower, upper) -> tuple[str, str]:
    try:
        lo, hi = float(lower), float(upper)
    except (ValueError, TypeError):
        return "unknown", "Нет оценки владельцев"
    if not np.isfinite([lo, hi]).all() or lo < 0 or hi <= lo:
        return "unknown", "Нет оценки владельцев"
    if lo == 0 and hi <= 20000:
        return "low_resolution", "Нижняя категория SteamSpy (<20 тыс.); точное число неизвестно"
    return "estimate", f"SteamSpy: {lo:,.0f}–{hi:,.0f} владельцев; не продажи"
