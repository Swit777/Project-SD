from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from .mechanics import KEYS, TAXONOMY_VERSION
from .sources import sha256_file


def create_annotation_tasks(root: Path, games=40, seed=42) -> Path:
    if not isinstance(games, int) or not 1 <= games <= 200:
        raise ValueError("Audit samples must contain 1..200 games")
    prepared = root / "data/processed/game_concept"
    source = prepared / "games.csv"
    catalog_path = root / "data/interim/game_concept/catalog.jsonl"
    audit = json.loads((prepared / "audit.json").read_text(encoding="utf-8"))
    dataset_hash, catalog_hash = sha256_file(source), sha256_file(catalog_path)
    if audit["data_sha256"] != dataset_hash or audit.get("catalog_sha256") != catalog_hash:
        raise ValueError("Annotation sources do not match their audit; prepare the verified snapshot first")
    data = pd.read_csv(source)
    selected = sorted(data.appid, key=lambda appid: hashlib.sha256(f"audit:{seed}:{appid}".encode()).hexdigest())[:games]
    manifest = {"dataset_sha256": dataset_hash, "catalog_sha256": catalog_hash, "taxonomy_version": TAXONOMY_VERSION,
                "sample_games": len(selected), "seed": seed, "sampling": "uniform SHA-256 priorities, independent of outcome",
                "target": "mechanic claims supported by provided Steam metadata; not verified gameplay quality"}
    fingerprint = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()[:16]
    base = root / "reports/game_concept/annotations"
    output = base / fingerprint
    output.mkdir(parents=True, exist_ok=True)
    if not (output / "tasks.csv").exists():
        catalog = {}
        with catalog_path.open(encoding="utf-8") as stream:
            for line in stream:
                game = json.loads(line)
                if game["appid"] in selected:
                    catalog[game["appid"]] = game
        if set(catalog) != set(selected):
            raise ValueError("Annotation descriptions are incomplete")
        lookup = data.set_index("appid")
        tasks = []
        for appid in selected:
            for key in KEYS:
                task_id = hashlib.sha256(f"{fingerprint}:{appid}:{key}".encode()).hexdigest()[:24]
                tasks.append({"task_id": task_id, "appid": appid, "mechanic": key,
                              "predicted": int(lookup.loc[appid, f"mechanic_{key}"]), "expert_label": "", "expert_notes": ""})
        pd.DataFrame(tasks).to_csv(output / "tasks.csv", index=False)
        (output / "games.json").write_text(json.dumps(list(catalog.values()), ensure_ascii=False, indent=2), encoding="utf-8")
        manifest["tasks_sha256"] = sha256_file(output / "tasks.csv")
        manifest["games_sha256"] = sha256_file(output / "games.json")
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (base / "current.json").write_text(json.dumps({"directory": fingerprint}), encoding="utf-8")
    return output


def current_annotation(root: Path) -> Path | None:
    base = root / "reports/game_concept/annotations"
    path = base / "current.json"
    if not path.exists():
        return None
    name = json.loads(path.read_text(encoding="utf-8"))["directory"]
    if not isinstance(name, str) or len(name) != 16 or any(c not in "0123456789abcdef" for c in name):
        raise ValueError("Invalid annotation directory")
    return base / name


def load_annotation(directory: Path) -> tuple[pd.DataFrame, list[dict], dict]:
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    for name, key in [("tasks.csv", "tasks_sha256"), ("games.json", "games_sha256")]:
        if sha256_file(directory / name) != manifest[key]:
            raise ValueError("Annotation task checksum mismatch")
    tasks = pd.read_csv(directory / "tasks.csv", keep_default_na=False)
    games = json.loads((directory / "games.json").read_text(encoding="utf-8"))
    reviews = pd.read_csv(directory / "reviews.csv", keep_default_na=False) if (directory / "reviews.csv").exists() else tasks[["task_id", "expert_label", "expert_notes"]]
    reviews = validate_reviews(tasks, reviews)
    tasks = tasks.drop(columns=["expert_label", "expert_notes"]).merge(reviews, on="task_id", how="left")
    tasks["expert_label"] = tasks.expert_label.fillna("")
    tasks["expert_notes"] = tasks.expert_notes.fillna("")
    return tasks, games, manifest


def validate_reviews(tasks: pd.DataFrame, reviews: pd.DataFrame) -> pd.DataFrame:
    if not {"task_id", "expert_label"}.issubset(reviews):
        raise ValueError("Review CSV requires task_id and expert_label")
    if reviews.task_id.isna().any() or reviews.task_id.duplicated().any():
        raise ValueError("Review task IDs must be nonmissing and unique")
    if not set(reviews.task_id).issubset(set(tasks.task_id)):
        raise ValueError("Reviews belong to different tasks or dataset version")
    result = reviews[["task_id", "expert_label"]].copy()
    labels = []
    for value in result.expert_label:
        if pd.isna(value) or str(value).strip() == "":
            labels.append("")
        elif str(value).strip().casefold() == "unclear":
            labels.append("unclear")
        else:
            try:
                number = float(value)
            except (TypeError, ValueError) as error:
                raise ValueError("Expert labels must be 0, 1, unclear or blank") from error
            if number not in (0, 1):
                raise ValueError("Expert labels must be 0, 1, unclear or blank")
            labels.append(str(int(number)))
    result["expert_label"] = labels
    result["expert_notes"] = reviews.expert_notes.fillna("").astype(str) if "expert_notes" in reviews else ""
    return result


def evaluate_annotations(tasks: pd.DataFrame, reviews: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    reviews = validate_reviews(tasks, reviews)
    merged = tasks.drop(columns=[c for c in ["expert_label", "expert_notes"] if c in tasks]).merge(reviews, on="task_id", how="left")
    known = merged.expert_label.isin(["0", "1"])
    measured = merged[known].copy()
    measured["actual"] = measured.expert_label.astype(int)
    def scores(frame):
        tp = int(((frame.predicted == 1) & (frame.actual == 1)).sum())
        fp = int(((frame.predicted == 1) & (frame.actual == 0)).sum())
        fn = int(((frame.predicted == 0) & (frame.actual == 1)).sum())
        tn = int(((frame.predicted == 0) & (frame.actual == 0)).sum())
        return {"reviewed": len(frame), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                "precision": tp / (tp + fp) if tp + fp else None,
                "recall": tp / (tp + fn) if tp + fn else None,
                "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None}
    counts = merged.assign(known=known).groupby("appid").known.agg(["sum", "size"])
    summary = {"status": "partially_reviewed" if known.any() else "not_reviewed", "tasks": len(tasks),
               "games": int(tasks.appid.nunique()), "reviewed_games": int(measured.appid.nunique()),
               "fully_labeled_games": int((counts["sum"] == counts["size"]).sum()),
               "unclear": int(merged.expert_label.eq("unclear").sum()), "unreviewed": int(merged.expert_label.isna().sum() + merged.expert_label.eq("").sum()),
               **scores(measured), "limitations": ["Metrics concern reviewed Steam claims, not verified gameplay",
                    "Partial or selective annotation cannot establish corpus-wide precision/recall", "One annotator; inter-rater agreement unmeasured"]}
    if known.all():
        summary["status"] = "fully_labeled"
    per_mechanic = pd.DataFrame([{"mechanic": key, **scores(measured[measured.mechanic.eq(key)])} for key in KEYS])
    return summary, per_mechanic


def save_reviews(directory: Path, reviews: pd.DataFrame, replace=False) -> dict:
    tasks, _, _ = load_annotation(directory)
    incoming = validate_reviews(tasks, reviews)
    stored = tasks[["task_id", "expert_label", "expert_notes"]]
    combined = incoming if replace else pd.concat([stored[~stored.task_id.isin(incoming.task_id)], incoming], ignore_index=True)
    temporary = directory / "reviews.csv.tmp"
    combined.to_csv(temporary, index=False)
    temporary.replace(directory / "reviews.csv")
    summary, details = evaluate_annotations(tasks, combined)
    (directory / "annotation_metrics.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    details.to_csv(directory / "mechanic_metrics.csv", index=False)
    return summary
