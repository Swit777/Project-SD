from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .config import ROOT, DEFAULT_CONFIG, load_config
from .dataset import prepare_snapshot
from .scraper import scrape_games
from .sources import download_snapshot


def main():
    parser = argparse.ArgumentParser(description="Steam mechanics and game-concept research")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("download")
    sub.add_parser("prepare")
    sub.add_parser("analogues-catalog")
    sub.add_parser("analogues-tags")
    scrape = sub.add_parser("scrape")
    scrape.add_argument("--appids", type=int, nargs="+")
    scrape.add_argument("--refresh", action="store_true")
    train = sub.add_parser("train")
    train.add_argument("--input", type=Path, default=ROOT / "data/processed/game_concept/games.csv")
    train.add_argument("--output", type=Path, default=ROOT / "reports/game_concept")
    sub.add_parser("run-all")
    audit = sub.add_parser("audit-create")
    audit.add_argument("--games", type=int, default=40)
    review = sub.add_parser("audit-evaluate")
    review.add_argument("--input", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    if args.command == "analogues-tags":
        from .analogue_catalog import enrich_analogue_tags
        print(enrich_analogue_tags(ROOT), flush=True)
    if args.command == "audit-create":
        from .annotation import create_annotation_tasks
        print(create_annotation_tasks(ROOT, args.games, config["seed"]), flush=True)
    if args.command == "audit-evaluate":
        import json
        from .annotation import current_annotation, save_reviews
        directory = current_annotation(ROOT)
        if directory is None:
            parser.error("Create annotation tasks first")
        print(json.dumps(save_reviews(directory, pd.read_csv(args.input, keep_default_na=False)), indent=2), flush=True)
    if args.command in ("download", "run-all"):
        download_snapshot(ROOT / "data/raw/steam_snapshot")
    if args.command in ("prepare", "run-all"):
        prepare_snapshot(ROOT, config)
    if args.command in ("analogues-catalog", "run-all"):
        from .analogue_catalog import build_analogue_catalog
        build_analogue_catalog(ROOT, config)
    if args.command in ("scrape", "run-all"):
        manifest = scrape_games(ROOT, config, getattr(args, "appids", None), getattr(args, "refresh", False))
        print(f"Steam scrape: {manifest['successful']}/{manifest['requested']} successful", flush=True)
    if args.command in ("train", "run-all"):
        from .models import train_experiment
        from .reports import write_reports
        source = args.input if args.command == "train" else ROOT / "data/processed/game_concept/games.csv"
        output = args.output if args.command == "train" else ROOT / "reports/game_concept"
        data = pd.read_csv(source)
        result = train_experiment(data, config, output, source)
        write_reports(data, result, output)
        print(f"Research results: {output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
