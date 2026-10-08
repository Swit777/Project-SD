from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .mechanics import KEYS, TAXONOMY_VERSION, extract_mechanics, extract_feature_claims
from .sources import REVISION, SNAPSHOT_SHA256, sha256_file

GAME_GENRES = ["Action", "Adventure", "Casual", "Indie", "RPG", "Simulation", "Sports", "Strategy", "Racing", "Massively Multiplayer"]
BAND_LABELS = ["0-20k", "20k-100k", "100k-500k", "500k+"]
MECHANIC_COLUMNS = [f"mechanic_{key}" for key in KEYS]
GENRE_COLUMNS = [f"genre_{genre.lower().replace(' ', '_')}" for genre in GAME_GENRES]
CONTEXT_COLUMNS = ["log_price", "log_age", "language_count", "windows", "mac", "linux"]


def list_field(value) -> list[str]:
    if not isinstance(value, str):
        return list(value) if isinstance(value, (list, tuple)) else []
    if not value.strip():
        return []
    return [part.strip() for part in next(csv.reader([value])) if part.strip()]


def owner_bounds(value: str) -> tuple[float, float]:
    match = re.fullmatch(r"\s*([\d,]+)\s*(?:-|\.\.)\s*([\d,]+)\s*", str(value))
    if not match:
        return np.nan, np.nan
    lo, hi = [int(s.replace(",", "")) for s in match.groups()]
    return (lo, hi) if 0 <= lo < hi else (np.nan, np.nan)


def owner_band(lower: float, upper: float) -> int | None:
    if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
        return None
    for i, (lo, hi) in enumerate([(0, 20000), (20000, 100000), (100000, 500000), (500000, np.inf)]):
        if lower >= lo and upper <= hi:
            return i
    return None


def feature_columns(mode: str = "full") -> list[str]:
    columns = CONTEXT_COLUMNS + GENRE_COLUMNS
    return columns if mode == "genre" else columns + MECHANIC_COLUMNS + ["mechanic_count"]


def validate_dataset(data: pd.DataFrame, training=True) -> list[str]:
    required = feature_columns() + (["appid", "name", "developer_group", "release_date", "owners_band", "owners_lower", "owners_upper", "playtime_median_hours"] if training else [])
    missing = sorted(set(required) - set(data))
    if missing:
        return ["Missing columns: " + ", ".join(missing)]
    if data.empty:
        return ["Empty dataset"]
    errors = []
    for column in feature_columns():
        numeric = pd.to_numeric(data[column], errors="coerce")
        if numeric.isna().any() or np.isinf(numeric).any():
            errors.append(f"{column}: finite numeric values required")
    for column in GENRE_COLUMNS + MECHANIC_COLUMNS + ["windows", "mac", "linux"]:
        if not pd.to_numeric(data[column], errors="coerce").isin([0, 1]).all():
            errors.append(f"{column}: expected 0 or 1")
    if not np.allclose(pd.to_numeric(data.mechanic_count, errors="coerce"), data[MECHANIC_COLUMNS].apply(pd.to_numeric, errors="coerce").sum(axis=1)):
        errors.append("mechanic_count must equal the sum of mechanic flags")
    if (data[GENRE_COLUMNS].apply(pd.to_numeric, errors="coerce").sum(axis=1) == 0).any():
        errors.append("At least one supported genre is required")
    for column in ["log_price", "log_age"]:
        if pd.to_numeric(data[column], errors="coerce").le(0).any():
            errors.append(f"{column}: positive values required for paid, released games")
    languages = pd.to_numeric(data.language_count, errors="coerce")
    if (languages.lt(1) | languages.mod(1).ne(0)).any():
        errors.append("language_count must be a positive integer")
    if training:
        if data.appid.isna().any() or data.appid.duplicated().any():
            errors.append("appid must be nonmissing and unique")
        if data.developer_group.isna().any() or data.developer_group.astype(str).str.strip().eq("").any():
            errors.append("developer_group cannot be missing")
        if data.developer_group.nunique() < 20:
            errors.append("At least 20 independent developer groups are required")
        if pd.to_datetime(data.release_date, errors="coerce").isna().any():
            errors.append("release_date must contain valid dates")
        if not pd.to_numeric(data.owners_band, errors="coerce").isin(range(4)).all():
            errors.append("owners_band must be one of 0, 1, 2, 3")
        for row in data[["owners_lower", "owners_upper", "owners_band"]].itertuples(index=False):
            try:
                expected = owner_band(float(row.owners_lower), float(row.owners_upper))
                if expected is None or expected != int(row.owners_band):
                    errors.append("Owner bands must agree with finite, ordered source bounds")
                    break
            except (TypeError, ValueError):
                errors.append("Invalid owner bounds")
                break
        hours = pd.to_numeric(data.playtime_median_hours, errors="coerce")
        if (data.playtime_median_hours.notna() & hours.isna()).any() or np.isinf(hours).any() or (hours < 0).any():
            errors.append("Playtime must be nonnegative, finite or missing")
    return list(dict.fromkeys(errors))


def prepare_snapshot(root: Path, config: dict) -> tuple[pd.DataFrame, dict]:
    source = root / "data/raw/steam_snapshot/games.csv"
    if not source.exists():
        raise FileNotFoundError("Download the pinned Steam snapshot first")
    if sha256_file(source) != SNAPSHOT_SHA256:
        raise ValueError("Steam source checksum mismatch")
    with source.open(encoding="utf-8", newline="") as stream:
        header = next(csv.reader(stream))
    repaired = "DiscountDLC count" in header
    if repaired:
        at = header.index("DiscountDLC count")
        header[at:at + 1] = ["Discount", "DLC count"]
    columns = ["AppID", "Name", "Release date", "Estimated owners", "Price", "Discount", "About the game",
               "Supported languages", "Header image", "Windows", "Mac", "Linux", "Positive", "Negative",
               "Average playtime forever", "Median playtime forever", "Developers", "Publishers", "Categories", "Genres", "Tags"]
    pools = []
    audit = {"source_rows": 0, "eligible_rows": 0, "header_repaired": repaired, "sampling": "uniform lowest SHA-256 priorities by appid + seed"}
    cutoff = pd.Timestamp(config["snapshot_date"]) - pd.Timedelta(days=config["minimum_age_days"])
    for chunk in pd.read_csv(source, names=header, skiprows=1, usecols=columns, chunksize=8000, low_memory=False, keep_default_na=False):
        audit["source_rows"] += len(chunk)
        chunk["parsed_date"] = pd.to_datetime(chunk["Release date"], format="mixed", errors="coerce")
        bounds = chunk["Estimated owners"].map(owner_bounds)
        chunk["owners_lower"] = [pair[0] for pair in bounds]
        chunk["owners_upper"] = [pair[1] for pair in bounds]
        chunk["owners_band"] = [owner_band(*pair) for pair in bounds]
        eligible = (pd.to_numeric(chunk["Price"], errors="coerce").gt(0) & chunk.parsed_date.le(cutoff)
                    & chunk.parsed_date.ge(pd.Timestamp("2010-01-01")) & chunk.owners_band.notna()
                    & chunk["Developers"].str.strip().ne("") & chunk["About the game"].str.len().ge(80)
                    & chunk["Genres"].map(lambda x: bool(set(list_field(x)) & set(GAME_GENRES)))
                    & ~chunk["Name"].str.contains(r"\b(?:playtest|demo|soundtrack|dedicated server)\b", case=False, regex=True))
        chunk = chunk.loc[eligible].copy()
        audit["eligible_rows"] += len(chunk)
        chunk["priority"] = chunk["AppID"].map(lambda appid: hashlib.sha256(f"{config['seed']}:{appid}".encode()).hexdigest())
        pools.append(chunk)
        pool = pd.concat(pools).sort_values("priority").head(config["sample_games"])
        pools = [pool]
        print(f"Scanned {audit['source_rows']:,} Steam records", flush=True)
    selected = pools[0].sort_values("AppID").drop_duplicates("AppID")
    raw_records, evidence, rows, claims = [], [], [], []
    for item in selected.to_dict("records"):
        appid = int(item["AppID"])
        genres, categories, tags = [list_field(item[c]) for c in ["Genres", "Categories", "Tags"]]
        url = f"https://store.steampowered.com/app/{appid}/"
        flags, entries = extract_mechanics(item["About the game"], tags, categories, url)
        evidence.extend({"appid": appid, **entry} for entry in entries)
        claims.extend({"appid": appid, "claim": claim, "source_url": url, "source_kind": "steam_snapshot",
                       "status": "candidate_for_manual_annotation"} for claim in extract_feature_claims(item["About the game"]))
        price = float(item["Price"])
        discount = pd.to_numeric(item["Discount"], errors="coerce")
        base_price = price / (1 - discount / 100) if np.isfinite(discount) and 0 <= discount < 100 else price
        age_days = (pd.Timestamp(config["snapshot_date"]) - item["parsed_date"]).days
        import ast
        try:
            languages = ast.literal_eval(item["Supported languages"])
            language_count = max(len(languages), 1) if isinstance(languages, list) else 1
        except (ValueError, SyntaxError):
            language_count = 1
        median = pd.to_numeric(item["Median playtime forever"], errors="coerce")
        mean = pd.to_numeric(item["Average playtime forever"], errors="coerce")
        row = {"appid": appid, "name": item["Name"], "release_date": item["parsed_date"].date().isoformat(),
            "developer_group": "|".join(sorted(d.casefold() for d in list_field(item["Developers"]))),
            "developers": item["Developers"], "publishers": item["Publishers"],
            "price_usd": base_price, "snapshot_price_usd": price, "age_days": age_days,
            "log_price": np.log1p(base_price), "log_age": np.log1p(age_days / 365.25),
            "language_count": language_count, "genres_json": json.dumps(genres), "tags_json": json.dumps(tags),
            "categories_json": json.dumps(categories), "mechanics_json": json.dumps([key for key in KEYS if flags[f"mechanic_{key}"]]),
            "source_url": url, "header_image": item["Header image"], "snapshot_date": config["snapshot_date"],
            "owners_lower": item["owners_lower"], "owners_upper": item["owners_upper"], "owners_band": int(item["owners_band"]),
            "playtime_median_hours": float(median / 60) if np.isfinite(median) and median > 0 else np.nan,
            "playtime_average_hours": float(mean / 60) if np.isfinite(mean) and mean > 0 else np.nan,
            "positive_reviews": pd.to_numeric(item["Positive"], errors="coerce"),
            "negative_reviews": pd.to_numeric(item["Negative"], errors="coerce"), **flags}
        for platform in ["windows", "mac", "linux"]:
            row[platform] = int(str(item[platform.title()]).lower() == "true")
        row.update({column: int(genre in genres) for column, genre in zip(GENRE_COLUMNS, GAME_GENRES)})
        row["mechanic_count"] = sum(flags.values())
        rows.append(row)
        raw_records.append({"appid": appid, "name": item["Name"], "description": item["About the game"],
                            "genres": genres, "categories": categories, "tags": tags, "source_url": url})
    data = pd.DataFrame(rows)
    errors = validate_dataset(data)
    if errors:
        raise ValueError("; ".join(errors))
    output = root / "data/processed/game_concept"
    output.mkdir(parents=True, exist_ok=True)
    data.to_csv(output / "games.csv", index=False)
    pd.DataFrame(evidence).to_csv(output / "mechanic_evidence.csv", index=False)
    pd.DataFrame(claims, columns=["appid", "claim", "source_url", "source_kind", "status"]).to_csv(output / "feature_claims.csv", index=False)
    cache = root / "data/interim/game_concept"
    cache.mkdir(parents=True, exist_ok=True)
    with (cache / "catalog.jsonl").open("w", encoding="utf-8") as stream:
        for item in raw_records:
            stream.write(json.dumps(item, ensure_ascii=False) + "\n")
    audit.update({"prepared_games": len(data), "developer_groups": int(data.developer_group.nunique()),
        "games_with_playtime": int(data.playtime_median_hours.notna().sum()), "evidence_rows": len(evidence),
        "feature_claims": len(claims), "taxonomy_version": TAXONOMY_VERSION, "snapshot_revision": REVISION, "snapshot_sha256": SNAPSHOT_SHA256,
        "catalog_sha256": sha256_file(cache / "catalog.jsonl"),
        "data_sha256": sha256_file(output / "games.csv"), "config": config,
        "band_counts": data.owners_band.value_counts().sort_index().to_dict(),
        "zero_playtime_policy": "Zero source values are missing/unreliable, not demonstrated zero engagement",
        "limitations": ["Owner estimates are not unit sales", "Time played is not cohort retention",
                        "Current descriptions/tags are not historical launch-time features", "Snapshot date is the file version date, not a timestamp for every record"]})
    (output / "audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(f"Prepared {len(data):,} games and {len(evidence):,} mechanic evidence entries", flush=True)
    return data, audit
