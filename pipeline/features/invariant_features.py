"""
FeatureLayer
────────────
Layer 1 of the three-layer architecture.

CONTRACT: this layer has NO access to ground truth labels.
Any feature derived from label files, audit scores, or post-hoc
researcher judgement is FORBIDDEN here.

Leakage audit
─────────────
Removed from prior implementation:
  ✗ prior_audit_score    — directly from researcher's Score column
  ✗ following_one        — dataset-collection artefact (all followers = 1)
  ✗ DirectBotScorer weights — hand-tuned on labelled data

Added (all unsupervised):
  ✓ temporal_entropy     — timing structure
  ✓ coactivation_score   — synchrony with peers
  ✓ bipartite motifs     — structural graph patterns
  ✓ target_entropy_norm  — normalised diversity of targeting
  ✓ cross_node_jaccard   — pairwise target-set similarity
  ✓ social metadata      — profile completeness, account age, etc.
                           (these describe the node, not the label)

Output: a single feature DataFrame (index = node_id) ready for
the detection layer.  The DataFrame carries no label column.
"""
from __future__ import annotations

from typing import List, Optional

import networkx as nx
import numpy as np
import pandas as pd

from pipeline.features       import temporal_features, motif_features, diversity_features
from pipeline.graph_builder  import GraphBuilder
from utils.logger            import get_logger

log = get_logger(__name__)


class FeatureLayer:
    """
    Parameters
    ──────────
    use_temporal  : include temporal entropy features (default True)
    use_motifs    : include graph motif features (default True)
    use_diversity : include interaction diversity features (default True)
    use_social    : include account-level social metadata (default True)
    """

    def __init__(
        self,
        use_temporal:  bool = True,
        use_motifs:    bool = True,
        use_diversity: bool = True,
        use_social:    bool = True,
    ) -> None:
        self.use_temporal  = use_temporal
        self.use_motifs    = use_motifs
        self.use_diversity = use_diversity
        self.use_social    = use_social

    def fit_transform(
        self,
        df:       pd.DataFrame,          # canonical edge DataFrame
        nodes_df: Optional[pd.DataFrame] = None,  # account metadata (NO audit score)
        G:        Optional[nx.DiGraph]   = None,
    ) -> pd.DataFrame:
        """
        df       : canonical interactions (user_id, target_id, action_type, timestamp)
        nodes_df : optional account metadata — used ONLY for non-label fields:
                   following_count, followers_count, bio_empty, default_avatar,
                   username_hash_suffix, joined_ts, s3_id
                   The 'audit_score' / 'Score' column is DROPPED if present.
        G        : pre-built graph (built from df if not provided)

        Returns feature DataFrame, index = node_id.
        """
        # Sanitize: drop any label-derived columns from nodes_df
        if nodes_df is not None:
            nodes_df = self._sanitize_nodes(nodes_df)

        if G is None:
            builder = GraphBuilder()
            G       = builder.build(df)

        active_nodes = [n for n in G.nodes() if G.out_degree(n) > 0]
        log.info("feature_layer: %d active nodes, %d edges", len(active_nodes), G.number_of_edges())

        frames = []

        # ── temporal features ─────────────────────────────────────────────────
        if self.use_temporal:
            log.debug("computing temporal entropy features ...")
            tf = temporal_features(df, nodes=active_nodes)
            frames.append(tf)

        # ── motif features ────────────────────────────────────────────────────
        if self.use_motifs:
            log.debug("computing graph motif features ...")
            mf = motif_features(G, nodes=active_nodes)
            frames.append(mf)

        # ── diversity features ────────────────────────────────────────────────
        if self.use_diversity:
            log.debug("computing diversity features ...")
            df_div = diversity_features(df, nodes=active_nodes)
            frames.append(df_div)

        # ── social metadata features (no labels) ──────────────────────────────
        if self.use_social and nodes_df is not None and not nodes_df.empty:
            sf = self._social_features(nodes_df, active_nodes)
            frames.append(sf)

        # ── graph centrality (basic) ───────────────────────────────────────────
        cf = self._centrality_features(G, active_nodes)
        frames.append(cf)

        if not frames:
            raise RuntimeError("No feature groups enabled.")

        feat_df = frames[0]
        for f in frames[1:]:
            feat_df = feat_df.join(f, how="outer")

        feat_df = feat_df.replace([np.inf, -np.inf], np.nan)
        feat_df = feat_df.fillna(feat_df.median())

        log.info(
            "feature_layer: output %d × %d (nodes × features)",
            feat_df.shape[0], feat_df.shape[1],
        )
        return feat_df

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _sanitize_nodes(nodes_df: pd.DataFrame) -> pd.DataFrame:
        """Drop any column that could encode ground-truth label information."""
        FORBIDDEN = {
            "audit_score", "score", "Score", "BotScore",
            "prior_audit_score", "label", "is_bot", "reasons", "Reasons",
        }
        drop_cols = [c for c in nodes_df.columns if c in FORBIDDEN
                     or c.lower() in {f.lower() for f in FORBIDDEN}]
        if drop_cols:
            log.info("feature_layer: dropped label-derived columns: %s", drop_cols)
            nodes_df = nodes_df.drop(columns=drop_cols)
        return nodes_df

    @staticmethod
    def _social_features(
        nodes_df:     pd.DataFrame,
        active_nodes: List[str],
    ) -> pd.DataFrame:
        """
        Extract account-level social metadata.
        NO audit score. NO researcher labels.
        """
        import time

        cols = {
            "following_count", "followers_count", "articles_count",
            "comments_count", "bio_empty", "default_avatar",
            "username_hash_suffix", "s3_id", "joined_ts",
        }
        available = [c for c in cols if c in nodes_df.columns]
        if not available:
            return pd.DataFrame(index=active_nodes)

        sub = nodes_df.reindex(active_nodes)[available].copy()

        # Derived: follow ratio (no labels needed)
        if "following_count" in sub and "followers_count" in sub:
            sub["follow_ratio"] = (
                sub["following_count"].fillna(0) /
                (sub["followers_count"].fillna(0) + 1.0)
            )

        # Derived: profile completeness (no labels needed)
        completeness_parts = []
        if "bio_empty"      in sub: completeness_parts.append(1 - sub["bio_empty"].fillna(1))
        if "default_avatar" in sub: completeness_parts.append(1 - sub["default_avatar"].fillna(1))
        if "articles_count" in sub: completeness_parts.append((sub["articles_count"].fillna(0) > 0).astype(float))
        if "comments_count" in sub: completeness_parts.append((sub["comments_count"].fillna(0) > 0).astype(float))
        if completeness_parts:
            sub["profile_completeness"] = sum(completeness_parts) / len(completeness_parts)

        # Derived: account age (normalised, no labels)
        if "joined_ts" in sub:
            now = time.time()
            age_days = (now - sub["joined_ts"].fillna(now)) / 86_400.0
            sub["account_age_days_norm"] = 1.0 - (age_days.clip(0, 3650) / 3650.0)

        # Derived: S3 batch density (no labels)
        if "s3_id" in sub:
            s3 = sub["s3_id"].fillna(-1).astype(float)
            valid = s3[s3 > 0].values
            if len(valid) > 0:
                density = np.zeros(len(s3))
                s3_vals = s3.values
                for i, sid in enumerate(s3_vals):
                    if sid <= 0:
                        continue
                    lo = np.searchsorted(valid, sid - 5_000)
                    hi = np.searchsorted(valid, sid + 5_000, side="right")
                    density[i] = (hi - lo - 1) / max(len(s3), 1)
                sub["s3_batch_density"] = density

        # Drop raw columns that are superseded by derived ones
        sub = sub.drop(columns=[c for c in ["s3_id", "joined_ts"] if c in sub.columns], errors="ignore")
        return sub.fillna(0.0)

    @staticmethod
    def _centrality_features(G: nx.DiGraph, nodes: List[str]) -> pd.DataFrame:
        """Basic graph centrality — pure topology, no labels."""
        out_deg = dict(G.out_degree())
        in_deg  = dict(G.in_degree())
        pr      = nx.pagerank(G, alpha=0.85, weight="weight")

        k_approx = min(200, len(G.nodes()))
        bc = nx.betweenness_centrality(G, k=k_approx, normalized=True, weight="weight")

        G_und  = G.to_undirected()
        clust  = nx.clustering(G_und)

        rows = {}
        for node in nodes:
            od = out_deg.get(node, 0)
            id_ = in_deg.get(node, 0)
            rows[node] = {
                "out_degree":      od,
                "in_degree":       id_,
                "degree_ratio":    od / (id_ + 1),
                "pagerank":        pr.get(node, 0.0),
                "betweenness":     bc.get(node, 0.0),
                "clustering_coeff":clust.get(node, 0.0),
            }
        return pd.DataFrame.from_dict(rows, orient="index")
