from __future__ import annotations

import json
import re
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup

from .mechanics import TAXONOMY_VERSION, extract_mechanics, extract_feature_claims, plain_text
from .sources import CachedHttp, utc_now

ANCHORS = [413150, 105600, 1145360, 1245620, 1086940, 1091500]


def parse_store_page(body: bytes) -> dict:
    soup = BeautifulSoup(body, "html.parser")
    title = soup.select_one("#appHubAppName") or soup.select_one(".apphub_AppName")
    description = soup.select_one("#game_area_description")
    image = soup.select_one("img.game_header_image_full")
    gated = description is None
    return {"name": title.get_text(" ", strip=True) if title else "",
            "tags": list(dict.fromkeys(a.get_text(" ", strip=True) for a in soup.select("a.app_tag") if a.get_text(strip=True) != "+")),
            "description": description.get_text(" ", strip=True) if description else "",
            "header_image": image.get("src", "") if image else "", "page_unavailable_or_gated": gated}


def summarize_reviews(response: dict) -> dict:
    data = response.get("response", response)
    reviews = data.get("reviews", [])
    paid = [r for r in reviews if r.get("steam_purchase") and not r.get("received_for_free")]
    times = [r.get("author", {}).get("playtime_forever") for r in paid]
    valid = [float(value) / 60 for value in times if isinstance(value, (int, float)) and value > 0]
    return {"sample_reviews": len(reviews), "sample_paid_reviews": len(paid),
            "sample_median_hours": float(np.median(valid)) if valid else None,
            "query_summary": data.get("query_summary", {}),
            "sampling": "most recent public reviews; not representative of all owners",
            "is_retention": False}


def scrape_games(root: Path, config: dict, appids: list[int] | None = None, refresh=False) -> dict:
    catalog_path = root / "data/processed/game_concept/games.csv"
    if appids is None:
        catalog = pd.read_csv(catalog_path)
        rng = np.random.default_rng(config["seed"])
        chosen = []
        # This is a stratified verification sample, not the distribution used for forecasting.
        for _, group in catalog.groupby("owners_band", sort=True):
            count = min(len(group), max(config["scrape_games"] // 4, 1))
            chosen.extend(int(value) for value in rng.choice(group.appid, count, replace=False))
        appids = list(dict.fromkeys(ANCHORS + chosen))[:config["scrape_games"]]
    if any(not isinstance(appid, int) or not 0 < appid < 2**32 for appid in appids):
        raise ValueError("Steam AppIDs must be positive 32-bit integers")
    if len(appids) > 1000:
        raise ValueError("Scrape batches are bounded to 1000 games")
    http = CachedHttp(root / "data/raw/steam_live/http", config["request_interval_seconds"], refresh)
    http.check_robots()
    output = root / "data/processed/game_concept"
    output.mkdir(parents=True, exist_ok=True)
    records, evidence, errors = [], [], []
    for i, appid in enumerate(appids, 1):
        print(f"Scraping Steam {i}/{len(appids)}: {appid}", flush=True)
        try:
            api_url = f"https://store.steampowered.com/api/appdetails?appids={appid}&l=english&cc=us"
            body, api_meta = http.get(api_url)
            entry = json.loads(body).get(str(appid), {})
            if not entry.get("success") or entry.get("data", {}).get("type") != "game":
                raise ValueError("Steam reports missing/unsupported app or non-game content")
            details = entry["data"]
            page_url = f"https://store.steampowered.com/app/{appid}/?l=english&cc=us"
            page, page_meta = http.get(page_url)
            parsed = parse_store_page(page)
            tags = parsed["tags"]
            genres = [g["description"] for g in details.get("genres", [])]
            categories = [c["description"] for c in details.get("categories", [])]
            description = plain_text(details.get("about_the_game", details.get("detailed_description", "")))
            flags, entries = extract_mechanics(description, tags, categories, page_url, "steam_live")
            evidence.extend({"appid": appid, "captured_at_utc": api_meta["captured_at_utc"], **e} for e in entries)
            review_summary, review_error = {}, None
            try:
                parameters = {"appid": appid, "filter": 1, "num_per_page": config["review_sample_size"],
                              "languages": ["all"], "purchase_type": 0}
                review_url = "https://api.steampowered.com/IUserReviewsService/GetAppReviews/v1/?input_json=" + quote(json.dumps(parameters))
                review_body, review_meta = http.get(review_url)
                review_summary = summarize_reviews(json.loads(review_body))
            except (ValueError, OSError, RuntimeError) as error:
                review_error = str(error)
            price = details.get("price_overview", {})
            record = {"appid": appid, "name": details.get("name", parsed["name"]), "source_url": page_url,
                      "taxonomy_version": TAXONOMY_VERSION,
                      "captured_at_utc": api_meta["captured_at_utc"], "source_kind": "steam_live",
                      "genres": genres, "categories": categories, "tags": tags, "mechanics": [k.removeprefix("mechanic_") for k, v in flags.items() if v],
                      "description": description, "price_currency": price.get("currency"),
                      "feature_claims": extract_feature_claims(details.get("about_the_game", description)),
                      "mechanic_evidence": entries,
                      "price_initial": price.get("initial", 0) / 100, "price_current": price.get("final", 0) / 100,
                      "is_free": details.get("is_free"), "release_date": details.get("release_date", {}),
                      "developers": details.get("developers", []), "publishers": details.get("publishers", []),
                      "header_image": details.get("header_image", parsed["header_image"]),
                      "page_unavailable_or_gated": parsed["page_unavailable_or_gated"],
                      "api_sha256": api_meta["sha256"], "page_sha256": page_meta["sha256"],
                      "review_sample": review_summary, "review_error": review_error}
            records.append(record)
            raw_path = root / "data/raw/steam_live/games"
            raw_path.mkdir(parents=True, exist_ok=True)
            (raw_path / f"{appid}.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        except (ValueError, OSError, RuntimeError) as error:
            errors.append({"appid": appid, "error": str(error)})
            print(f"Recorded scrape failure: {error}", flush=True)
    all_records = [json.loads(path.read_text(encoding="utf-8")) for path in sorted((root / "data/raw/steam_live/games").glob("*.json"))]
    if all_records:
        rows = [{k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v
                 for k, v in record.items() if k not in ["description", "mechanic_evidence"]} for record in all_records]
        pd.DataFrame(rows).to_csv(output / "live_games.csv", index=False)
        all_evidence = []
        for record in all_records:
            entries = record.get("mechanic_evidence")
            if entries is None:
                _, entries = extract_mechanics(record["description"], record["tags"], record["categories"], record["source_url"], "steam_live")
            all_evidence.extend({"appid": record["appid"], "captured_at_utc": record["captured_at_utc"], **entry} for entry in entries)
        pd.DataFrame(all_evidence).to_csv(output / "live_mechanic_evidence.csv", index=False)
    manifest = {"requested": len(appids), "successful": len(records), "errors": errors,
        "taxonomy_version": TAXONOMY_VERSION,
        "appids": appids, "total_cached_games": len(all_records), "completed_at_utc": utc_now(), "mode": "stratified audit; not training labels",
        "store_html_unavailable": sum(bool(r["page_unavailable_or_gated"]) for r in records),
        "fallback": "Public API metadata is retained; unavailable HTML tags remain unknown",
        "retention_available": False, "sales_available": False,
        "limitations": ["Steam public pages do not disclose unit sales", "Review playtime is a biased sample",
                        "No authentication, age-gate, rate-limit or CAPTCHA bypass", "Current enrichment is not merged into older training labels"]}
    (output / "scrape_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
