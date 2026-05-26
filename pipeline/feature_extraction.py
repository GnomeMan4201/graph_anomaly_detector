"""
FeatureExtractor
────────────────
Computes a feature matrix (pandas DataFrame, index = node_id) covering:

  Graph features
    out_degree, in_degree, degree_ratio, clustering_coeff,
    pagerank, betweenness

  Temporal features (derived from raw event timestamps)
    inter_event_variance, burstiness, activity_rate_1h,
    activity_rate_24h, total_events

  Behavioural features
    repetition_ratio, target_diversity, neighbor_overlap

All NaN / inf values are filled (strategy: 0 for rate/variance features,
column-mean for centrality features that should never be absent).
"""
from __future__ import annotations

from typing import Dict, List

import networkx as nx
import numpy as np
import pandas as pd

from config import Config
from utils.logger import get_logger

log = get_logger(__name__)


class FeatureExtractor:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg

    # ── public api ────────────────────────────────────────────────────────────

    def extract(self, G: nx.DiGraph, df: pd.DataFrame) -> pd.DataFrame:
        """
        Returns a DataFrame where index = node_id and columns = features.
        Only nodes that appear as *source* (user_id) are included in the
        feature matrix; pure targets with no out-edges get skipped.
        """
        nodes = [n for n in G.nodes() if G.out_degree(n) > 0]
        log.info("extracting features for %d active nodes", len(nodes))

        graph_feats   = self._graph_features(G, nodes)
        temporal_feats= self._temporal_features(df, nodes)
        behav_feats   = self._behavioural_features(df, G, nodes)

        feat_df = (
            pd.DataFrame(graph_feats)
            .join(pd.DataFrame(temporal_feats))
            .join(pd.DataFrame(behav_feats))
        )
        feat_df = self._sanitize(feat_df)

        log.info(
            "feature matrix: %d nodes × %d features",
            feat_df.shape[0],
            feat_df.shape[1],
        )
        return feat_df

    # ── graph features ────────────────────────────────────────────────────────

    def _graph_features(
        self, G: nx.DiGraph, nodes: List[str]
    ) -> pd.DataFrame:
        log.debug("computing graph centrality features ...")

        out_deg = dict(G.out_degree())
        in_deg  = dict(G.in_degree())

        # Undirected projection for clustering coefficient
        G_und = G.to_undirected()
        clust = nx.clustering(G_und)

        pr = nx.pagerank(G, alpha=self.cfg.pagerank_alpha, weight="weight")

        # Approximate betweenness: k=min(cfg.k, n) random pivots
        k_approx = min(self.cfg.betweenness_approx_k, len(G.nodes()))
        bc = nx.betweenness_centrality(
            G, k=k_approx, normalized=True, weight="weight"
        )

        rows = {}
        for node in nodes:
            od = out_deg.get(node, 0)
            id_ = in_deg.get(node, 0)
            rows[node] = {
                "out_degree":      od,
                "in_degree":       id_,
                # degree_ratio > 1 means the node follows far more than it is followed
                "degree_ratio":    od / (id_ + 1),
                "clustering_coeff": clust.get(node, 0.0),
                "pagerank":        pr.get(node, 0.0),
                "betweenness":     bc.get(node, 0.0),
            }
        return pd.DataFrame.from_dict(rows, orient="index")

    # ── temporal features ─────────────────────────────────────────────────────

    def _temporal_features(
        self, df: pd.DataFrame, nodes: List[str]
    ) -> pd.DataFrame:
        log.debug("computing temporal features ...")

        # Pre-group to avoid repeated filtering
        grouped = df.groupby("user_id")["timestamp"].apply(
            lambda s: np.sort(s.values)
        ).to_dict()

        max_ts = df["timestamp"].max()

        rows = {}
        for node in nodes:
            ts = grouped.get(node, np.array([]))
            rows[node] = self._temporal_for_node(ts, max_ts)
        return pd.DataFrame.from_dict(rows, orient="index")

    @staticmethod
    def _temporal_for_node(ts: np.ndarray, max_ts: float) -> Dict:
        n = len(ts)
        if n < 2:
            return {
                "inter_event_variance": 0.0,
                "burstiness":           0.0,
                "activity_rate_1h":     float(n),
                "activity_rate_24h":    float(n),
                "total_events":         float(n),
            }

        iet   = np.diff(ts)                  # inter-event times
        mu    = iet.mean()
        sigma = iet.std()

        # Goh-Barabási burstiness:  B = (σ−μ)/(σ+μ) ∈ (-1, +1)
        burstiness = (sigma - mu) / (sigma + mu + 1e-12)

        return {
            "inter_event_variance": float(sigma ** 2),
            "burstiness":           float(burstiness),
            "activity_rate_1h":     float(np.sum(ts > (max_ts - 3_600))),
            "activity_rate_24h":    float(np.sum(ts > (max_ts - 86_400))),
            "total_events":         float(n),
        }

    # ── behavioural features ──────────────────────────────────────────────────

    def _behavioural_features(
        self, df: pd.DataFrame, G: nx.DiGraph, nodes: List[str]
    ) -> pd.DataFrame:
        log.debug("computing behavioural features ...")

        # Pre-compute successor sets (limit to 500 for scale)
        succ: Dict[str, set] = {
            n: set(list(G.successors(n))[:500]) for n in G.nodes()
        }
        grouped_targets = df.groupby("user_id")["target_id"].apply(list).to_dict()

        rows = {}
        for node in nodes:
            targets = grouped_targets.get(node, [])
            rows[node] = self._behav_for_node(node, targets, succ)
        return pd.DataFrame.from_dict(rows, orient="index")

    @staticmethod
    def _behav_for_node(
        node: str,
        targets: List[str],
        succ: Dict[str, set],
    ) -> Dict:
        total = len(targets)
        if total == 0:
            return {
                "repetition_ratio": 0.0,
                "target_diversity": 0.0,
                "neighbor_overlap": 0.0,
            }

        from collections import Counter
        counts  = Counter(targets)
        unique  = len(counts)
        repeated = sum(c - 1 for c in counts.values() if c > 1)

        repetition_ratio = repeated / total
        target_diversity = unique / total

        # Neighbor overlap: mean Jaccard(out_nbrs(node), out_nbrs(nbr))
        # Sample up to 20 neighbours to bound cost
        node_out = succ.get(node, set())
        if not node_out:
            neighbor_overlap = 0.0
        else:
            sample    = list(node_out)[:20]
            overlaps  = []
            for nbr in sample:
                nbr_out = succ.get(nbr, set())
                union   = node_out | nbr_out
                if union:
                    overlaps.append(len(node_out & nbr_out) / len(union))
            neighbor_overlap = float(np.mean(overlaps)) if overlaps else 0.0

        return {
            "repetition_ratio": float(repetition_ratio),
            "target_diversity": float(target_diversity),
            "neighbor_overlap": float(neighbor_overlap),
        }

    # ── sanitization ─────────────────────────────────────────────────────────

    @staticmethod
    def _sanitize(df: pd.DataFrame) -> pd.DataFrame:
        """Replace inf/NaN.  Rate/variance cols → 0; centrality cols → col mean."""
        zero_fill_cols = {
            "inter_event_variance", "burstiness", "activity_rate_1h",
            "activity_rate_24h", "total_events", "repetition_ratio",
            "target_diversity", "neighbor_overlap", "degree_ratio",
        }
        for col in df.columns:
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)
            if col in zero_fill_cols:
                df[col] = df[col].fillna(0.0)
            else:
                df[col] = df[col].fillna(df[col].mean())
        return df
