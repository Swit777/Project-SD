from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import quote

from .sources import CachedHttp


def parse_review_totals(payload: dict) -> dict:
    summary = payload.get("response", {}).get("query_summary")
    if not isinstance(summary, dict):
        raise ValueError("Steam did not return a review summary")
    totals = {}
    for key in ["total_reviews", "total_positive", "total_negative"]:
        value = summary.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("Missing or invalid Steam review totals")
        totals[key] = value
    if totals["total_positive"] + totals["total_negative"] != totals["total_reviews"]:
        raise ValueError("Steam review totals are inconsistent")
    return totals


def refresh_review_totals(root: Path, appid: int, interval=2.0) -> dict:
    if isinstance(appid, bool) or not isinstance(appid, int) or not 0 < appid < 2**32:
        raise ValueError("A positive Steam AppID is required")
    parameters = {"appid": appid, "filter": 1, "languages": ["all"], "review_type": 0,
                  "purchase_type": 1, "num_per_page": 1, "filter_offtopic_activity": True}
    url = "https://api.steampowered.com/IUserReviewsService/GetAppReviews/v1/?input_json=" + quote(json.dumps(parameters))
    client = CachedHttp(root / "data/raw/steam_live/review_totals_http", interval, refresh=True)
    body, meta = client.get(url)
    result = {"appid": appid, **parse_review_totals(json.loads(body)), "captured_at_utc": meta["captured_at_utc"],
              "source_url": url, "response_sha256": meta["sha256"], "parameters": parameters,
              "meaning": "Public review totals, all languages and purchase types; not owners or sales"}
    destination = root / "data/processed/game_concept/review_totals"
    destination.mkdir(parents=True, exist_ok=True)
    (destination / f"{appid}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result
