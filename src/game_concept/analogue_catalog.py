from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .dataset import GAME_GENRES, GENRE_COLUMNS, list_field, owner_bounds
from .mechanics import KEYS, TAXONOMY_VERSION, extract_mechanics
from .sources import SNAPSHOT_SHA256, sha256_file
from .analogues import structured_mode


def build_analogue_catalog(root: Path, config: dict) -> dict:
    """Build a separate discovery index, without changing the research sample or its labels."""
    source = root / "data/raw/steam_snapshot/games.csv"
    if sha256_file(source) != SNAPSHOT_SHA256:
        raise ValueError("Steam source checksum mismatch")
    with source.open(encoding="utf-8", newline="") as stream:
        header = next(csv.reader(stream))
    if "DiscountDLC count" in header:
        at = header.index("DiscountDLC count")
        header[at:at + 1] = ["Discount", "DLC count"]
    columns = ["AppID", "Name", "Release date", "Estimated owners", "Price", "Discount", "About the game",
               "Header image", "Positive", "Negative", "Genres", "Tags", "Categories", "Developers",
               "Median playtime forever"]
    cutoff = pd.Timestamp(config["snapshot_date"]) - pd.Timedelta(days=config["minimum_age_days"])
    rows, scanned = [], 0
    for chunk in pd.read_csv(source, names=header, skiprows=1, usecols=columns, chunksize=8000,
                             low_memory=False, keep_default_na=False):
        scanned += len(chunk)
        dates = pd.to_datetime(chunk["Release date"], format="mixed", errors="coerce")
        eligible = (pd.to_numeric(chunk.Price, errors="coerce").gt(0) & dates.between("2010-01-01", cutoff)
                    & chunk.Developers.str.strip().ne("") & chunk["About the game"].str.len().ge(80)
                    & chunk.Genres.map(lambda v: bool(set(list_field(v)) & set(GAME_GENRES)))
                    & ~chunk.Name.str.contains(r"\b(?:playtest|demo|soundtrack|dedicated server)\b", case=False, regex=True))
        chunk = chunk.loc[eligible].copy()
        chunk["parsed_date"] = dates.loc[eligible]
        for item in chunk.to_dict("records"):
            appid = int(item["AppID"])
            url = f"https://store.steampowered.com/app/{appid}/"
            genres, tags, categories = [list_field(item[c]) for c in ["Genres", "Tags", "Categories"]]
            flags, _ = extract_mechanics(item["About the game"], tags, categories, url)
            discount = pd.to_numeric(item["Discount"], errors="coerce")
            price = float(item["Price"])
            base = price / (1 - discount / 100) if np.isfinite(discount) and 0 <= discount < 100 else price
            age = (pd.Timestamp(config["snapshot_date"]) - item["parsed_date"]).days
            lo, hi = owner_bounds(item["Estimated owners"])
            hours = pd.to_numeric(item["Median playtime forever"], errors="coerce")
            rows.append({"appid": appid, "name": item["Name"], "price_usd": base, "age_days": age,
                "release_date": item["parsed_date"].date().isoformat(), "developers": item["Developers"],
                "log_price": np.log1p(base), "log_age": np.log1p(age / 365.25), "snapshot_date": config["snapshot_date"],
                "genres_json": json.dumps(genres), "tags_json": json.dumps(tags), "categories_json": json.dumps(categories),
                "mode_co_op_structured": int(structured_mode(categories, "co_op")),
                "mode_pvp_structured": int(structured_mode(categories, "pvp")),
                "mechanics_json": json.dumps([k for k in KEYS if flags[f"mechanic_{k}"]]),
                "positive_reviews": pd.to_numeric(item["Positive"], errors="coerce"),
                "negative_reviews": pd.to_numeric(item["Negative"], errors="coerce"),
                "owners_lower": lo, "owners_upper": hi, "source_url": url, "header_image": item["Header image"],
                "playtime_median_hours": hours / 60 if np.isfinite(hours) and hours > 0 else np.nan,
                **{c: int(g in genres) for c, g in zip(GENRE_COLUMNS, GAME_GENRES)}, **flags})
        print(f"Analogue index: {scanned:,} scanned, {len(rows):,} eligible", flush=True)
    if not rows:
        raise ValueError("No eligible games for the analogue catalog")
    result = pd.DataFrame(rows).drop_duplicates("appid").sort_values("appid")
    output = root / "data/processed/game_concept"
    output.mkdir(parents=True, exist_ok=True)
    temporary = output / "analogue_catalog.csv.part"
    result.to_csv(temporary, index=False)
    temporary.replace(output / "analogue_catalog.csv")
    meta = {"source_rows": scanned, "catalog_games": len(result), "snapshot_sha256": SNAPSHOT_SHA256,
            "catalog_sha256": sha256_file(output / "analogue_catalog.csv"), "snapshot_date": config["snapshot_date"],
            "taxonomy_version": TAXONOMY_VERSION, "catalog_schema": 3, "minimum_age_days": config["minimum_age_days"],
            "purpose": "Discovery only; not used for fitting, calibration, test selection or probabilities",
            "owner_policy": "Missing owner estimates retained as unknown; no review-to-sales conversion",
            "filters": "Paid; released 2010+; minimum age; developer; description >=80; supported genre; no demo/playtest/soundtrack/server"}
    (output / "analogue_catalog_manifest.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def enrich_analogue_tags(root: Path):
    """Upgrade a verified discovery index without repeating mechanic extraction."""
    source = root / "data/raw/steam_snapshot/games.csv"
    destination = root / "data/processed/game_concept/analogue_catalog.csv"
    manifest_path = destination.with_name("analogue_catalog_manifest.json")
    meta = json.loads(manifest_path.read_text(encoding="utf-8"))
    if sha256_file(source) != SNAPSHOT_SHA256 or sha256_file(destination) != meta["catalog_sha256"]:
        raise ValueError("Source or catalog checksum mismatch")
    with source.open(encoding="utf-8", newline="") as stream:
        header = next(csv.reader(stream))
    if "DiscountDLC count" in header:
        at = header.index("DiscountDLC count")
        header[at:at + 1] = ["Discount", "DLC count"]
    tags = pd.read_csv(source, names=header, skiprows=1, usecols=["AppID", "Tags"], keep_default_na=False)
    tags["tags_json"] = tags.Tags.map(lambda v: json.dumps(list_field(v)))
    data = pd.read_csv(destination).drop(columns="tags_json", errors="ignore")
    data = data.merge(tags[["AppID", "tags_json"]].rename(columns={"AppID": "appid"}), on="appid", how="left", validate="one_to_one")
    if data.tags_json.isna().any():
        raise ValueError("Incomplete tag enrichment")
    temporary = destination.with_suffix(".csv.part")
    data.to_csv(temporary, index=False)
    temporary.replace(destination)
    meta.update(catalog_schema=3, catalog_sha256=sha256_file(destination))
    manifest_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta
