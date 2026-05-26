#!/usr/bin/env python3
"""
replay.py — Real-World Botnet Detection Pipeline
─────────────────────────────────────────────────
Runs the full detection pipeline against actual researcher-collected datasets.

Usage examples

  # DEV.to follower audit dataset
  python replay.py \\
      --platform devto \\
      --data /path/to/devto_bot_audit_full.csv \\
      --labels /path/to/flagged_usernames.txt \\
      --target gnomeman4201

  # DEV.to raw JSON from API
  python replay.py \\
      --platform devto \\
      --data /path/to/followers_raw.json \\
      --labels /path/to/flagged_usernames.txt \\
      --target gnomeman4201

  # GitHub following-list similarity analysis
  python replay.py \\
      --platform github \\
      --data /path/to/following_lists.json \\
      --labels /path/to/confirmed_bots.txt \\
      --jaccard-threshold 0.50

  # Tune detection threshold
  python replay.py \\
      --platform devto \\
      --data devto_bot_audit_full.csv \\
      --labels flagged_usernames.txt \\
      --threshold 0.45 \\
      --contamination 0.15

Differences from main.py (synthetic mode)
  - Loads real platform data via adapters instead of synthetic generator
  - Merges social features (follow_ratio, s3_batch_density, etc.) into
    the feature matrix alongside graph features
  - Runs BotnetEvaluator against ground-truth labels
  - Graph feature extraction is skipped when the graph is a degenerate
    star topology (all nodes pointing to one target) and --social-only
    flag is set
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from adapters          import DevToAdapter, GitHubAdapter
from config            import Config
from evaluation        import BotnetEvaluator
from pipeline.graph_builder       import GraphBuilder
from pipeline.feature_extraction  import FeatureExtractor
from pipeline.social_features     import SocialFeatureExtractor
from pipeline.modeling            import AnomalyModeler
from pipeline.scoring             import FraudScorer
from pipeline.bot_scorer          import DirectBotScorer
from pipeline.explainability      import ExplainabilityEngine
from pipeline.output              import OutputLayer
from utils.logger      import get_logger

log = get_logger("replay")


# ─── CLI ─────────────────────────────────────────────────────────────────────

def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Replay real botnet dataset through the detection pipeline."
    )
    p.add_argument("--platform", required=True, choices=["devto", "github"],
                   help="Source platform.")
    p.add_argument("--data",     required=True, metavar="PATH",
                   help="Path to account CSV/JSON or following-list JSON.")
    p.add_argument("--labels",   metavar="PATH",
                   help="Ground truth labels file (txt/csv/json).")
    p.add_argument("--target",   default=None,
                   help="Target account username (DEV.to: your handle).")

    # GitHub specific
    p.add_argument("--accounts",          metavar="PATH",
                   help="GitHub: separate account metadata CSV/JSON.")
    p.add_argument("--jaccard-threshold", type=float, default=0.50,
                   help="Jaccard similarity threshold for GitHub edges (default 0.50).")

    # Model tuning
    p.add_argument("--contamination",    type=float, default=0.10)
    p.add_argument("--dbscan-eps",       type=float, default=0.60)
    p.add_argument("--dbscan-min-samples", type=int, default=3)
    p.add_argument("--threshold",        type=float, default=0.55,
                   help="Fraud score flag threshold (default 0.55).")
    p.add_argument("--top-n",            type=int,   default=50)

    # Feature mode
    p.add_argument("--social-only", action="store_true",
                   help="Use only social features (skip graph centrality). "
                        "Recommended for star-topology datasets like DEV.to followers.")
    p.add_argument("--no-graph",    action="store_true",
                   help="Alias for --social-only.")

    # I/O
    p.add_argument("--output-dir", default="output")
    return p.parse_args(argv)


# ─── pipeline ─────────────────────────────────────────────────────────────────

def run(args: argparse.Namespace) -> None:
    t0 = time.perf_counter()

    # ── 0. Config ─────────────────────────────────────────────────────────────
    cfg = Config(
        isolation_forest_contamination = args.contamination,
        dbscan_eps                     = args.dbscan_eps,
        dbscan_min_samples             = args.dbscan_min_samples,
        flag_threshold                 = args.threshold,
        top_n_suspicious               = args.top_n,
    )
    cfg.validate()
    skip_graph = args.social_only or args.no_graph
    log.info(
        "config: platform=%s  threshold=%.2f  social_only=%s",
        args.platform, cfg.flag_threshold, skip_graph,
    )

    # ── 1. Load data via adapter ──────────────────────────────────────────────
    if args.platform == "devto":
        adapter  = DevToAdapter(target_username=args.target or "target")
        edges_df, nodes_df = adapter.load(args.data)

    elif args.platform == "github":
        adapter = GitHubAdapter(
            target_username   = args.target,
            jaccard_threshold = args.jaccard_threshold,
        )
        if Path(args.data).suffix == ".json":
            # Assume following-lists format
            edges_df, nodes_df = adapter.load_following_lists(
                args.data, accounts_path=args.accounts
            )
        else:
            edges_df, nodes_df = adapter.load_accounts(args.data)

    else:
        log.error("unknown platform: %s", args.platform)
        sys.exit(1)

    if edges_df.empty and nodes_df.empty:
        log.error("no data loaded — check input file")
        sys.exit(1)

    log.info(
        "loaded: %d accounts, %d edges",
        len(nodes_df), len(edges_df),
    )

    # ── 2. Social features ────────────────────────────────────────────────────
    social_extractor = SocialFeatureExtractor()
    social_feats = social_extractor.extract(nodes_df)

    # ── 3. Graph features (optional) ──────────────────────────────────────────
    if skip_graph or edges_df.empty:
        log.info("skipping graph feature extraction (social-only mode)")
        feat_df = social_feats
    else:
        # Align edges_df to canonical schema expected by existing pipeline
        canonical = _ensure_canonical(edges_df)
        builder   = GraphBuilder()
        G         = builder.build(canonical)

        graph_extractor = FeatureExtractor(cfg)
        graph_feats     = graph_extractor.extract(G, canonical)

        # Merge: social features take priority; graph features fill in where present
        feat_df = social_feats.join(graph_feats, how="left", rsuffix="_graph")
        feat_df = feat_df.fillna(0.0)

    if feat_df.empty:
        log.error("feature matrix is empty — nothing to model")
        sys.exit(1)

    log.info("feature matrix: %d × %d", feat_df.shape[0], feat_df.shape[1])

    # Override model_features to match whatever is actually in feat_df
    cfg_override = Config(
        isolation_forest_contamination = cfg.isolation_forest_contamination,
        dbscan_eps                     = cfg.dbscan_eps,
        dbscan_min_samples             = cfg.dbscan_min_samples,
        flag_threshold                 = cfg.flag_threshold,
        top_n_suspicious               = cfg.top_n_suspicious,
        model_features                 = [c for c in feat_df.columns
                                          if feat_df[c].dtype in (float, "float64", "int64")],
    )

    # ── 4. Modeling ───────────────────────────────────────────────────────────
    modeler  = AnomalyModeler(cfg_override)
    result   = modeler.fit_predict(feat_df)

    # ── 5. Fraud scoring ──────────────────────────────────────────────────────
    if skip_graph:
        # Social-only mode: IsolationForest inverts when bots are the majority
        # class (treats the dense bot cluster as "normal").  Use the direct
        # rule-based scorer exclusively — it correctly weights known bot signals
        # without being confused by the majority-class distribution.
        direct_scorer  = DirectBotScorer()
        direct_scores  = direct_scorer.score(feat_df, result.cluster_labels)

        score_df = pd.DataFrame({
            "anomaly_score":            result.anomaly_scores.reindex(feat_df.index),
            "direct_bot_score":         direct_scores.reindex(feat_df.index),
            "centrality_deviation":     0.0,
            "cluster_density_anomaly":  0.0,
            "fraud_score":              direct_scores.reindex(feat_df.index).fillna(0).clip(0, 1),
            "flagged":                  direct_scores.reindex(feat_df.index).fillna(0) >= cfg.flag_threshold,
        }, index=feat_df.index)

        n_flagged = int(score_df["flagged"].sum())
        log.info(
            "social scoring (direct+IF blend): flagged=%d / %d  (threshold=%.2f)",
            n_flagged, len(score_df), cfg.flag_threshold,
        )
    else:
        scorer   = FraudScorer(cfg_override)
        score_df = scorer.score(feat_df, result)

    # ── 6. Explainability ─────────────────────────────────────────────────────
    explainer    = ExplainabilityEngine(cfg_override)
    explanations = explainer.explain_all(feat_df, score_df, result)

    # ── 7. Output ─────────────────────────────────────────────────────────────
    run_meta = {
        "platform":    args.platform,
        "data":        args.data,
        "n_accounts":  len(feat_df),
        "social_only": skip_graph,
    }
    output_layer = OutputLayer(cfg_override, output_dir=args.output_dir)
    paths = output_layer.write(score_df, explanations, result, feat_df, run_meta)

    # ── 8. Evaluation against ground truth ───────────────────────────────────
    if args.labels:
        evaluator = BotnetEvaluator(output_dir=args.output_dir)
        evaluator.load_labels(args.labels)
        evaluator.evaluate(
            score_df      = score_df,
            cluster_labels= result.cluster_labels,
            threshold     = cfg.flag_threshold,
        )

    elapsed = time.perf_counter() - t0
    log.info("replay complete in %.2fs", elapsed)
    for label, path in paths.items():
        log.info("  %-22s  %s", label, path)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _ensure_canonical(df: pd.DataFrame) -> pd.DataFrame:
    """
    Make sure edges_df has the four required canonical columns.
    Fill missing ones with sensible defaults.
    """
    out = df.copy()
    if "user_id"     not in out.columns: out["user_id"]     = "unknown"
    if "target_id"   not in out.columns: out["target_id"]   = "unknown"
    if "action_type" not in out.columns: out["action_type"] = "follow"
    if "timestamp"   not in out.columns: out["timestamp"]   = 0.0
    out["timestamp"] = pd.to_numeric(out["timestamp"], errors="coerce").fillna(0.0)
    return out[["user_id", "target_id", "action_type", "timestamp"]]


# ─── entrypoint ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run(parse_args())
