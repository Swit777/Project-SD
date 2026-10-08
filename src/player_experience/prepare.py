from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from .features import extract_features, validate_dataset
from .source import FILES, sha256_file


PREPARATION_VERSION = 2


def _canonical(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.rename(columns={
        "pid": "player_id", "Time_utc": "timestamp", "CurrentJobName": "job",
        "CurrentGameMode": "mode", "CurrentWasher": "washer", "CurrentNozzle": "nozzle",
        "LevelProgressionAmount": "progress", "EventName": "kind",
    })
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    return frame.dropna(subset=["player_id", "timestamp"])


def _read_member(archive: zipfile.ZipFile, name: str, columns: list[str], players: set | None = None) -> pd.DataFrame:
    parts = []
    count = 0
    with archive.open("data/" + name + ".csv") as stream:
        for chunk in pd.read_csv(stream, usecols=columns, chunksize=250000, low_memory=False):
            count += len(chunk)
            if players is not None:
                chunk = chunk.loc[chunk["pid"].isin(players)]
            if len(chunk):
                parts.append(chunk)
            if name == "subtask_completed" and count % 1000000 == 0:
                print(f"Scanned {count:,} subtask events", flush=True)
    print(f"Read {name}: {count:,} source rows", flush=True)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=columns)


def _save_parquet(frame: pd.DataFrame, path: Path) -> None:
    with duckdb.connect() as connection:
        connection.register("prepared_frame", frame)
        connection.execute("COPY prepared_frame TO ? (FORMAT PARQUET)", [str(path)])


def _read_parquet(path: Path) -> pd.DataFrame:
    with duckdb.connect() as connection:
        connection.execute("SET TimeZone='UTC'")
        frame = connection.execute("SELECT * FROM read_parquet(?)", [str(path)]).df()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True).dt.as_unit("ns")
    return frame


def prepare_dataset(root: Path, config: dict, force: bool = False) -> tuple[pd.DataFrame, dict]:
    raw = root / "data" / "raw" / "powerwash"
    cache = root / "data" / "interim" / "powerwash"
    output = root / "data" / "processed" / "player_experience.csv"
    cache.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    cache_spec = {"sampling": config["sampling"], "seed": config["seed"],
                  "source_sha256": FILES["data.zip"]["sha256"], "version": PREPARATION_VERSION}
    signature = hashlib.sha256(json.dumps(cache_spec, sort_keys=True).encode()).hexdigest()
    manifest_path = cache / "manifest.json"
    names = ["prompts", "subtasks", "boundaries"]
    cached = not force and manifest_path.exists() and all((cache / f"{name}.parquet").exists() for name in names)
    if cached:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        cached = manifest.get("signature") == signature
        if cached:
            for name in names:
                if sha256_file(cache / f"{name}.parquet") != manifest["cache_sha256"][name]:
                    raise ValueError("Preparation cache checksum mismatch; rerun prepare --force")
    if cached:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        prompts, subtasks, boundaries = [_read_parquet(cache / f"{name}.parquet") for name in names]
        print("Using verified preparation cache", flush=True)
    else:
        archive_path = raw / "data.zip"
        if not archive_path.exists():
            raise FileNotFoundError("Run 'python run_pipeline.py download' first")
        if sha256_file(archive_path) != FILES["data.zip"]["sha256"]:
            raise ValueError("Raw data checksum differs from the pinned OSF release")
        with zipfile.ZipFile(archive_path) as archive:
            prompt_columns = ["pid", "Time_utc", "LastStudyPromptType", "response", "CurrentGameMode",
                              "CurrentJobName", "CurrentSessionLength", "CampaignProgressionAmount", "LevelProgressionAmount"]
            prompts = _canonical(_read_member(archive, "study_prompt_answered", prompt_columns))
            total = len(prompts)
            prompts["response"] = pd.to_numeric(prompts["response"], errors="coerce")
            prompts = prompts.loc[
                prompts["LastStudyPromptType"].eq("Enjoyment") & prompts["response"].between(0, 1000)
                & prompts["job"].notna() & prompts["mode"].isin(["Career", "FreePlay", "Challenge", "Special"])
            ].copy()
            prompts = prompts.drop_duplicates(["player_id", "timestamp"])
            eligible_counts = prompts.groupby("player_id").size()
            eligible = sorted(eligible_counts[eligible_counts >= config["sampling"]["min_responses"]].index)
            if len(eligible) < 12:
                raise ValueError("Not enough eligible players")
            rng = np.random.default_rng(config["seed"])
            selected = sorted(rng.choice(eligible, size=min(config["sampling"]["max_players"], len(eligible)), replace=False))
            selected_set = set(selected)
            prompts = prompts.loc[prompts["player_id"].isin(selected_set)].sort_values(["player_id", "timestamp"])
            capped = []
            for _, group in prompts.groupby("player_id", sort=True):
                take = min(len(group), config["sampling"]["max_responses_per_player"])
                capped.append(group.iloc[np.sort(rng.choice(len(group), take, replace=False))])
            prompts = pd.concat(capped).rename(columns={"response": "enjoyment", "CurrentSessionLength": "session_minutes",
                                                      "CampaignProgressionAmount": "campaign_progress", "progress": "level_progress"})
            prompts["enjoyment"] /= 10.0
            prompts["response_id"] = prompts["player_id"] + ":" + prompts["timestamp"].astype(str)
            subtasks = _canonical(_read_member(archive, "subtask_completed",
                ["pid", "Time_utc", "LevelProgressionAmount", "CurrentWasher", "CurrentNozzle"], selected_set))
            boundary_parts = []
            for name in ["job_started", "job_resumed", "job_exited", "exited_game", "player_logged_in"]:
                columns = ["pid", "Time_utc", "EventName"]
                if name != "player_logged_in":
                    columns += ["CurrentJobName", "CurrentGameMode"]
                part = _canonical(_read_member(archive, name, columns, selected_set))
                for column in ["job", "mode"]:
                    if column not in part:
                        part[column] = None
                boundary_parts.append(part)
            boundaries = pd.concat(boundary_parts, ignore_index=True)
            manifest = {"signature": signature, "source_prompts": total, "eligible_players": len(eligible),
                        "selected_players": selected, "sampling": config["sampling"], "seed": config["seed"],
                        "source_sha256": FILES["data.zip"]["sha256"], "version": PREPARATION_VERSION}
        for name, frame in zip(names, [prompts, subtasks, boundaries]):
            _save_parquet(frame, cache / f"{name}.parquet")
        manifest["cache_sha256"] = {name: sha256_file(cache / f"{name}.parquet") for name in names}
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    data, feature_audit = extract_features(prompts, subtasks, boundaries, config["features"]["windows_minutes"])
    errors = validate_dataset(data, config["features"]["windows_minutes"])
    if errors:
        raise ValueError("; ".join(errors))
    data.to_csv(output, index=False, encoding="utf-8")
    audit = {**feature_audit, "eligible_players": manifest["eligible_players"], "source_prompts": manifest["source_prompts"],
             "target": "Enjoyment / 10, scale 0-100", "source_sha256": manifest["source_sha256"],
             "windows_minutes": config["features"]["windows_minutes"], "data_sha256": sha256_file(output),
             "sampling_seed": config["seed"], "selected_players": manifest["selected_players"]}
    (output.parent / "player_experience_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(f"Prepared {len(data):,} observations from {data['player_id'].nunique():,} players", flush=True)
    return data, audit
