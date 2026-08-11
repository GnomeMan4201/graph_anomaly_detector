#!/usr/bin/env python3
"""
main.py — Graph-Based Anomaly Detection Pipeline
─────────────────────────────────────────────────
Usage
  # generate synthetic data and run
  python main.py --synthetic

  # run on existing JSONL file
  python main.py --input /path/to/events.jsonl

  # custom config overrides
  python main.py --synthetic --contamination 0.08 --flag-threshold 0.50

  # deterministic synthetic run
  python main.py --synthetic --seed 42
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from config import Config
from data.synthetic_generator import SyntheticDataGenerator
from pipeline.ingestion import DataIngestionLayer
from pipeline.graph_builder import GraphBuilder
from pipeline.feature_extraction import FeatureExtractor
from pipeline.modeling import AnomalyModeler
from pipeline.scoring import FraudScorer
from pipeline.explainability import ExplainabilityEngine
from pipeline.output import OutputLayer
from utils.logger import get_logger

log = get_logger("main")


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Graph-based anomaly and coordinated-behaviour detector."
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--synthetic", action="store_true",
        help="Generate and use a synthetic interaction dataset.",
    )
    src.add_argument(
        "--input", metavar="PATH",
        help="Path to .jsonl or .csv interaction file.",
    )

    p.add_argument("--seed", type=int, default=42,
                   help="Synthetic generator seed (default: 42).")
    p.add_argument("--n-normal", type=int, default=200,
                   help="Number of normal users in synthetic data (default: 200).")
    p.add_argument("--n-clusters", type=int, default=5,
                   help="Number of bot clusters (default: 5).")
    p.add_argument("--n-days", type=int, default=7,
                   help="Simulation window in days (default: 7).")

    p.add_argument("--contamination", type=float, default=0.10)
    p.add_argument("--dbscan-eps", type=float, default=0.60)
    p.add_argument("--dbscan-min-samples", type=int, default=3)
    p.add_argument("--flag-threshold", type=float, default=0.55)
    p.add_argument("--top-n", type=int, default=30)

    p.add_argument("--output-dir", default="output",
                   help="Directory for JSON result files (default: ./output).")
    p.add_argument("--save-synthetic", metavar="PATH",
                   help="If --synthetic, also save generated events to this JSONL path.")

    return p.parse_args(argv)


def run(args: argparse.Namespace) -> None:
    t0 = time.perf_counter()

    cfg = Config(
        isolation_forest_contamination=args.contamination,
        dbscan_eps=args.dbscan_eps,
        dbscan_min_samples=args.dbscan_min_samples,
        flag_threshold=args.flag_threshold,
        top_n_suspicious=args.top_n,
    )
    cfg.validate()
    log.info(
        "config: contamination=%.2f  dbscan_eps=%.2f  flag_threshold=%.2f",
        cfg.isolation_forest_contamination,
        cfg.dbscan_eps,
        cfg.flag_threshold,
    )

    ground_truth_bots = []

    if args.synthetic:
        log.info(
            "generating synthetic dataset: seed=%d  n_normal=%d  n_clusters=%d  n_days=%d",
            args.seed, args.n_normal, args.n_clusters, args.n_days,
        )
        gen = SyntheticDataGenerator(
            seed=args.seed,
            n_normal_users=args.n_normal,
            n_bot_clusters=args.n_clusters,
            n_days=args.n_days,
        )
        records = gen.generate()
        ground_truth_bots = gen.ground_truth_bots()
        log.info(
            "synthetic: %d events  |  %d known bot IDs",
            len(records), len(ground_truth_bots),
        )

        if args.save_synthetic:
            n = gen.save_jsonl(args.save_synthetic)
            log.info("saved %d synthetic events → %s", n, args.save_synthetic)

        ingestion_layer = DataIngestionLayer()
        df = ingestion_layer.ingest_records(records)

    else:
        if not Path(args.input).exists():
            log.error("input file not found: %s", args.input)
            sys.exit(1)
        ingestion_layer = DataIngestionLayer()
        df = ingestion_layer.ingest(args.input)

    if df.empty:
        log.error("no usable records after ingestion — aborting")
        sys.exit(1)

    log.info(
        "dataset: %d interactions  %d unique sources  %d unique targets",
        len(df),
        df["user_id"].nunique(),
        df["target_id"].nunique(),
    )

    builder = GraphBuilder()
    G = builder.build(df)

    extractor = FeatureExtractor(cfg)
    feat_df = extractor.extract(G, df)

    modeler = AnomalyModeler(cfg)
    result = modeler.fit_predict(feat_df)

    scorer = FraudScorer(cfg)
    score_df = scorer.score(feat_df, result)

    explainer = ExplainabilityEngine(cfg)
    explanations = explainer.explain_all(feat_df, score_df, result)

    run_meta = {
        "source": "synthetic" if args.synthetic else args.input,
        "n_events": len(df),
        "n_nodes": len(feat_df),
        "known_bots": ground_truth_bots,
        "config": {
            "seed": args.seed if args.synthetic else None,
            "n_normal": args.n_normal if args.synthetic else None,
            "n_clusters": args.n_clusters if args.synthetic else None,
            "n_days": args.n_days if args.synthetic else None,
            "contamination": args.contamination,
            "dbscan_eps": args.dbscan_eps,
            "dbscan_min_samples": args.dbscan_min_samples,
            "flag_threshold": args.flag_threshold,
            "top_n": args.top_n,
        },
    }

    output_layer = OutputLayer(cfg, output_dir=args.output_dir)
    paths = output_layer.write(score_df, explanations, result, feat_df, run_meta)

    elapsed = time.perf_counter() - t0
    log.info("pipeline complete in %.2fs", elapsed)

    if ground_truth_bots:
        _print_detection_stats(score_df, ground_truth_bots)

    log.info("output files:")
    for label, path in paths.items():
        log.info("  %-20s  %s", label, path)


def _print_detection_stats(score_df: "pd.DataFrame", known_bots: list) -> None:
    import pandas as pd

    flagged = set(score_df[score_df["flagged"]].index.astype(str))
    bots = set(str(b) for b in known_bots)

    tp = len(flagged & bots)
    fp = len(flagged - bots)
    fn = len(bots - flagged)
    tn = len(set(score_df.index.astype(str)) - flagged - bots)

    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)

    width = 60
    print(f"\n{'─' * width}")
    print("  GROUND TRUTH EVALUATION (synthetic bots only)")
    print(f"{'─' * width}")
    print(f"  Known bots  : {len(bots)}")
    print(f"  Flagged     : {len(flagged)}")
    print(f"  TP={tp}  FP={fp}  FN={fn}  TN={tn}")
    print(f"  Precision   : {precision:.3f}")
    print(f"  Recall      : {recall:.3f}")
    print(f"  F1          : {f1:.3f}")
    print(f"{'─' * width}\n")


if __name__ == "__main__":
    run(parse_args())
