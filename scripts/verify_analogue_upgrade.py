"""Compare retrieval constraints with the old ranking without claiming human-judged relevance."""
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from game_concept.analogues import find_analogues, review_stats, structured_mode
from game_concept.dataset import GENRE_COLUMNS, MECHANIC_COLUMNS
from game_concept.predict import concept_features, load_bundle
from game_concept.sources import sha256_file
from game_concept.tag_similarity import TagIndex


def old_analogues(features, reference, limit=12):
    flags = GENRE_COLUMNS + MECHANIC_COLUMNS
    a = features.iloc[0][flags].to_numpy(dtype=float)
    b = reference[flags].to_numpy(dtype=float)
    distance = 1 - np.minimum(a, b).sum(axis=1) / np.maximum(np.maximum(a, b).sum(axis=1), 1)
    distance += 0.12 * np.abs(reference.log_price.to_numpy() - features.iloc[0].log_price)
    distance += 0.08 * np.abs(reference.log_age.to_numpy() - features.iloc[0].log_age)
    return review_stats(reference.iloc[np.argsort(distance, kind="stable")[:limit]])


def summarize(rows, keys):
    return {"games": len(rows), "median_reviews": float(rows.review_count.median()) if len(rows) else None,
            "below_30_reviews": int(rows.review_count.lt(30).sum()),
            "mean_selected_coverage": float(rows[[f"mechanic_{k}" for k in keys]].mean(axis=1).mean()) if len(rows) else None,
            "unconfirmed_multiplayer": int(sum(any(not structured_mode(r.categories_json, k) for k in keys if k in ["co_op", "pvp"]) for r in rows.itertuples()))}


def main():
    output = ROOT / "reports/game_concept/analogue_upgrade"
    output.mkdir(parents=True, exist_ok=True)
    market = pd.read_csv(ROOT / "data/processed/game_concept/analogue_catalog.csv")
    bundle = load_bundle(ROOT / "reports/game_concept/models/concept_model.joblib")
    cases = [("rpg_coop_crafting", ["Indie", "RPG"], ["crafting", "co_op"]),
             ("strategy_cards", ["Indie", "Strategy"], ["deck_building", "turn_based_combat"]),
             ("adventure_puzzles", ["Adventure"], ["puzzles", "exploration"]),
             ("action_platformer", ["Action"], ["platforming", "boss_battles"]),
             ("simulation_automation", ["Simulation"], ["base_building", "automation"])]
    summaries, examples = {}, []
    for name, genres, keys in cases:
        features = concept_features(genres, keys)
        old = old_analogues(features, bundle["reference"])
        new, diagnostics = find_analogues(features, market, min_reviews=30, min_coverage=1, price_factor=3, limit=12)
        balanced, balanced_meta = find_analogues(features, market, min_reviews=30, min_coverage=0.5, limit=12)
        broad, broad_meta = find_analogues(features, market, min_reviews=100, min_coverage=0.25, require_modes=False, limit=None)
        broad = broad.sort_values(["review_count", "similarity", "appid"], ascending=[False, False, True]).head(12)
        assert new.review_count.ge(30).all() and new.mechanic_coverage.eq(1).all()
        summaries[name] = {"old": summarize(old, keys), "new": summarize(new, keys), "diagnostics": diagnostics,
            "balanced": summarize(balanced, keys), "balanced_diagnostics": balanced_meta,
            "broad": summarize(broad, keys), "broad_diagnostics": broad_meta}
        for method, frame in [("old_train_jaccard", old), ("new_market_structured", new), ("new_market_balanced", balanced), ("new_market_visible", broad)]:
            rows = frame[["appid", "name", "review_count", "owners_lower", "owners_upper"]].copy()
            rows["case"], rows["method"] = name, method
            examples.append(rows)
    anchor = market.loc[market.appid.eq(413150)].iloc[0]
    tags = TagIndex(market).scores(json.loads(anchor.tags_json))
    anchored, anchored_meta = find_analogues(concept_features(["Indie", "RPG"], ["crafting", "co_op"]), market,
        min_reviews=30, tag_scores=tags, exclude_appid=413150, limit=12)
    anchored[["appid", "name", "similarity", "tag_similarity", "structural_similarity", "review_count"]].to_csv(output / "anchor_stardew.csv", index=False, encoding="utf-8-sig")
    frozen = json.loads((ROOT / "reports/game_concept/article/article_manifest.json").read_text(encoding="utf-8"))
    preserved = {}
    for path, digest in frozen["inputs_sha256"].items():
        if path.startswith("reports/game_concept/") and (path.endswith(".csv") or path.endswith(".joblib") or path.endswith("run_manifest.json")):
            preserved[path] = sha256_file(ROOT / path) == digest
    preserved["training_data"] = sha256_file(ROOT / "data/processed/game_concept/games.csv") == frozen["experiment_input_sha256"]
    assert all(preserved.values()), preserved
    result = {"cases": summaries, "anchor_diagnostics": anchored_meta, "research_artifacts_unchanged": preserved,
              "depth": 12, "catalog_games": len(market), "article_update_date": "2026-10-09",
              "retrieval_code_sha256": {p.name: sha256_file(p) for p in (ROOT / "src/game_concept").glob("*.py")},
              "limitations": "Constraint compliance, not independently judged retrieval precision; filtered market analogues are not representative success frequencies"}
    (output / "comparison.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.concat(examples, ignore_index=True).to_csv(output / "examples.csv", index=False, encoding="utf-8-sig")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
