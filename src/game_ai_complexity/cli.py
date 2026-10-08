from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .config import load_config
from .dataset import generate_dataset, save_dataset, validate_dataset
from .models import train_models
from .predict import predict_success
from .reproducibility import write_run_manifest


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs" / "default.json"
DEFAULT_DATA = ROOT / "data" / "processed" / "levels_results.csv"
DEFAULT_REPORTS = ROOT / "reports"
DEFAULT_MODEL = DEFAULT_REPORTS / "best_model.pkl"
DEFAULT_PREDICTIONS = DEFAULT_REPORTS / "new_dataset_predictions.csv"


def cmd_generate(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    data = generate_dataset(config)
    errors = validate_dataset(data)
    if errors:
        raise SystemExit("; ".join(errors))
    path = save_dataset(data, args.output)
    print(f"Generated {len(data)} rows -> {path}")


def cmd_train(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    data = pd.read_csv(args.data)
    result = train_models(
        data,
        args.reports,
        random_state=int(config["training"]["random_state"]),
        test_size=float(config["training"]["test_size"]),
        cv_folds=int(config["training"].get("cv_folds", 3)),
    )
    best = result["best_model"]
    score = result["metrics"][best]["roc_auc"]
    write_run_manifest(args.reports, config, args.data)
    print(f"Best model: {best} | ROC-AUC={score:.3f}")
    print(f"Reports -> {args.reports}")


def cmd_run_all(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    data = generate_dataset(config)
    save_dataset(data, args.output)
    result = train_models(
        data,
        args.reports,
        random_state=int(config["training"]["random_state"]),
        test_size=float(config["training"]["test_size"]),
        cv_folds=int(config["training"].get("cv_folds", 3)),
    )
    best = result["best_model"]
    score = result["metrics"][best]["roc_auc"]
    write_run_manifest(args.reports, config, args.output)
    print(f"Generated {len(data)} rows -> {args.output}")
    print(f"Best model: {best} | ROC-AUC={score:.3f}")
    print(f"Reports -> {args.reports}")


def cmd_predict(args: argparse.Namespace) -> None:
    data = pd.read_csv(args.input)
    predictions = predict_success(data, args.model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.output, index=False, encoding="utf-8")
    print(f"Predicted {len(predictions)} rows -> {args.output}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Game AI level complexity pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    generate = sub.add_parser("generate", help="Generate reproducible level dataset")
    generate.add_argument("--config", default=DEFAULT_CONFIG, type=Path)
    generate.add_argument("--output", default=DEFAULT_DATA, type=Path)
    generate.set_defaults(func=cmd_generate)

    train = sub.add_parser("train", help="Train models on a generated or uploaded dataset")
    train.add_argument("--config", default=DEFAULT_CONFIG, type=Path)
    train.add_argument("--data", default=DEFAULT_DATA, type=Path)
    train.add_argument("--reports", default=DEFAULT_REPORTS, type=Path)
    train.set_defaults(func=cmd_train)

    run_all = sub.add_parser("run-all", help="Generate data and train all models")
    run_all.add_argument("--config", default=DEFAULT_CONFIG, type=Path)
    run_all.add_argument("--output", default=DEFAULT_DATA, type=Path)
    run_all.add_argument("--reports", default=DEFAULT_REPORTS, type=Path)
    run_all.set_defaults(func=cmd_run_all)

    predict = sub.add_parser("predict", help="Predict success probability for a new CSV")
    predict.add_argument("--input", required=True, type=Path)
    predict.add_argument("--model", default=DEFAULT_MODEL, type=Path)
    predict.add_argument("--output", default=DEFAULT_PREDICTIONS, type=Path)
    predict.set_defaults(func=cmd_predict)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)
