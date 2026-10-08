from __future__ import annotations

import numpy as np
import pandas as pd


CATEGORICAL_FEATURES = ["job", "mode", "washer", "nozzle"]
STATIC_FEATURES = [
    "session_minutes", "episode_minutes", "campaign_progress", "level_progress",
    "subtasks_total", "overall_rate", "overall_gap_mean",
]
HISTORY_GLOBAL = ["seconds_since_subtask", "recent_rate_change"]
WINDOW_STATS = ["count", "rate", "gap_mean", "gap_cv", "progress_gain", "coverage"]
IDENTIFIERS = ["response_id", "player_id", "timestamp"]
TARGET = "enjoyment"


def feature_sets(windows: list[int]) -> dict[str, list[str]]:
    static = STATIC_FEATURES + CATEGORICAL_FEATURES
    dynamic = HISTORY_GLOBAL + [f"w{w}_{name}" for w in windows for name in WINDOW_STATS]
    return {"aggregate": static, "dynamic": static + dynamic}


def validate_dataset(data: pd.DataFrame, windows: list[int], require_target: bool = True) -> list[str]:
    required = feature_sets(windows)["dynamic"] + (IDENTIFIERS + [TARGET] if require_target else [])
    missing = sorted(set(required) - set(data.columns))
    if missing:
        return ["Missing columns: " + ", ".join(missing)]
    if data.empty:
        return ["The dataset is empty"]
    errors = []
    numeric = [c for c in required if c not in CATEGORICAL_FEATURES + IDENTIFIERS]
    for column in numeric:
        converted = pd.to_numeric(data[column], errors="coerce")
        if (data[column].notna() & converted.isna()).any() or np.isinf(converted).any():
            errors.append(f"{column}: expected finite numeric values or missing values")
    if require_target:
        target = pd.to_numeric(data[TARGET], errors="coerce")
        if target.isna().any() or not target.between(0, 100).all():
            errors.append("enjoyment must be present and between 0 and 100")
        if data[IDENTIFIERS].isna().any().any():
            errors.append("Response IDs, player IDs and timestamps cannot be missing")
        if data["response_id"].duplicated().any():
            errors.append("response_id must be unique")
        if pd.to_datetime(data["timestamp"], utc=True, errors="coerce").isna().any():
            errors.append("timestamp must contain valid dates")
        if data["player_id"].nunique() < 12:
            errors.append("At least 12 distinct players are required for independent splits")
    return errors


def extract_features(
    prompts: pd.DataFrame,
    subtasks: pd.DataFrame,
    boundaries: pd.DataFrame,
    windows: list[int],
) -> tuple[pd.DataFrame, dict]:
    """Use past completions inside the current job episode; never cross a job boundary."""
    prompts, subtasks, boundaries = [frame.copy() for frame in (prompts, subtasks, boundaries)]
    for frame in (prompts, subtasks, boundaries):
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True).dt.as_unit("ns")
    event_groups = {p: g.sort_values("timestamp", kind="stable") for p, g in subtasks.groupby("player_id")}
    boundary_groups = {p: g.sort_values("timestamp", kind="stable") for p, g in boundaries.groupby("player_id")}
    rows = []
    audit = {"input_responses": len(prompts), "missing_episode": 0, "episode_context_mismatch": 0}
    for player, player_prompts in prompts.groupby("player_id", sort=True):
        events = event_groups.get(player)
        starts = boundary_groups.get(player)
        if starts is None:
            audit["missing_episode"] += len(player_prompts)
            continue
        bt = starts["timestamp"].astype("int64").to_numpy() / 1e9
        if events is None:
            events = pd.DataFrame(columns=["timestamp", "progress", "washer", "nozzle"])
            et = np.array([], dtype=float)
        else:
            et = events["timestamp"].astype("int64").to_numpy() / 1e9
        ep = pd.to_numeric(events["progress"], errors="coerce").to_numpy(dtype=float)
        washers = events["washer"].fillna("unknown").to_numpy()
        nozzles = events["nozzle"].fillna("unknown").to_numpy()
        for prompt in player_prompts.itertuples(index=False):
            now = prompt.timestamp.timestamp()
            # Equal timestamps are excluded because source precision is one second.
            bidx = int(np.searchsorted(bt, now, side="left") - 1)
            if bidx < 0 or starts.iloc[bidx]["kind"] not in ("job_started", "job_resumed"):
                audit["missing_episode"] += 1
                continue
            boundary = starts.iloc[bidx]
            if boundary["job"] != prompt.job or boundary["mode"] != prompt.mode:
                audit["episode_context_mismatch"] += 1
                continue
            episode_start = bt[bidx]
            lo = int(np.searchsorted(et, episode_start, side="left"))
            hi = int(np.searchsorted(et, now, side="left"))
            count = hi - lo
            duration = max(now - episode_start, 1)
            gaps = np.diff(et[lo:hi])
            row = {
                "response_id": prompt.response_id,
                "player_id": player,
                "timestamp": prompt.timestamp.isoformat(),
                TARGET: prompt.enjoyment,
                "job": prompt.job,
                "mode": prompt.mode,
                "washer": washers[hi - 1] if count else "unknown",
                "nozzle": nozzles[hi - 1] if count else "unknown",
                "session_minutes": prompt.session_minutes,
                "episode_minutes": duration / 60,
                "campaign_progress": prompt.campaign_progress,
                "level_progress": prompt.level_progress,
                "subtasks_total": count,
                "overall_rate": count / (duration / 60),
                "overall_gap_mean": float(gaps.mean()) if len(gaps) else np.nan,
                "seconds_since_subtask": min(now - et[hi - 1], 3600) if count else np.nan,
            }
            for window in windows:
                left = int(np.searchsorted(et, max(episode_start, now - window * 60), side="left"))
                times = et[left:hi]
                local_gaps = np.diff(times)
                coverage = min(duration, window * 60)
                observed_progress = ep[left:hi]
                finite_progress = observed_progress[np.isfinite(observed_progress)]
                row.update({
                    f"w{window}_count": len(times),
                    f"w{window}_rate": len(times) / (max(coverage, 1) / 60),
                    f"w{window}_gap_mean": float(local_gaps.mean()) if len(local_gaps) else np.nan,
                    f"w{window}_gap_cv": float(local_gaps.std() / local_gaps.mean())
                    if len(local_gaps) >= 2 and local_gaps.mean() > 0 else np.nan,
                    f"w{window}_progress_gain": float(max(finite_progress[-1] - finite_progress[0], 0))
                    if len(finite_progress) >= 2 else 0.0,
                    f"w{window}_coverage": coverage / (window * 60),
                })
            shortest, longest = min(windows), max(windows)
            row["recent_rate_change"] = row[f"w{shortest}_rate"] - row[f"w{longest}_rate"]
            rows.append(row)
    columns = IDENTIFIERS + [TARGET] + feature_sets(windows)["dynamic"]
    data = pd.DataFrame(rows, columns=columns).sort_values(["player_id", "timestamp"]).reset_index(drop=True)
    audit.update({"prepared_responses": len(data), "players": int(data["player_id"].nunique()),
                  "equal_timestamp_events_excluded": True, "cross_episode_history_excluded": True})
    return data, audit
