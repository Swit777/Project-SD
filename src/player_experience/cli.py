from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .config import DEFAULT_CONFIG, ROOT, load_config
from .models import predict_dataset, train_experiment
from .prepare import prepare_dataset
from .source import download_sources


def main() -> None:
    parser = argparse.ArgumentParser(description="Public-data player-experience research")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("download", help="Download and checksum the pinned public OSF release")
    prepare = commands.add_parser("prepare", help="Extract past-only features from real telemetry")
    prepare.add_argument("--force", action="store_true")
    train = commands.add_parser("train", help="Run the independent-player experiment")
    train.add_argument("--input", type=Path, default=ROOT / "data/processed/player_experience.csv")
    train.add_argument("--output", type=Path, default=ROOT / "reports/player_experience")
    commands.add_parser("run-all", help="Download, prepare, train and write research reports")
    predict = commands.add_parser("predict", help="Estimate enjoyment for a same-schema CSV")
    predict.add_argument("--input", type=Path, required=True)
    predict.add_argument("--output", type=Path, required=True)
    predict.add_argument("--model", type=Path, default=ROOT / "reports/player_experience/models/best_model.joblib")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.command in ("download", "run-all"):
        download_sources(ROOT / "data/raw/powerwash")
    if args.command in ("prepare", "run-all"):
        prepare_dataset(ROOT, config, force=getattr(args, "force", False))
    if args.command in ("train", "run-all"):
        from .reports import write_reports

        source = args.input if args.command == "train" else ROOT / "data/processed/player_experience.csv"
        output = args.output if args.command == "train" else ROOT / "reports/player_experience"
        data = pd.read_csv(source)
        result = train_experiment(data, output, config, input_path=source)
        write_reports(data, result, output)
        print(f"Selected on validation: {result['metadata']['best_model']}", flush=True)
        print(f"Reports: {output.resolve()}", flush=True)
    if args.command == "predict":
        predictions = predict_dataset(pd.read_csv(args.input), args.model)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        predictions.to_csv(args.output, index=False)
        print(f"Saved {len(predictions)} predictions to {args.output}", flush=True)


if __name__ == "__main__":
    main()
