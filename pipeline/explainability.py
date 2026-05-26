"""
ExplainabilityEngine
────────────────────
For each flagged node produces:

  top_features      list of (feature_name, node_value, baseline_mean, z_score)
                    sorted by |z_score| descending, top-N returned
  deviation_summary human-readable string describing the key signals
  cluster_context   why the cluster assignment is suspicious (if relevant)

The approach is a simple z-score attribution — no external SHAP dependency.
Each feature's contribution to the anomaly decision is approximated by its
z-score relative to the *clean baseline* (non-flagged nodes).  This matches
the intuition: "this node is anomalous because feature X is Z standard
deviations from normal".
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from config import Config
from pipeline.modeling import ModelResult
from utils.logger import get_logger

log = get_logger(__name__)


class ExplainabilityEngine:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg

    # ── public api ────────────────────────────────────────────────────────────

    def explain_all(
        self,
        feat_df:     pd.DataFrame,
        score_df:    pd.DataFrame,
        result:      ModelResult,
    ) -> Dict[str, Dict]:
        """
        Returns explanations for all flagged nodes.
        Key = node_id, value = explanation dict.
        """
        flagged = score_df[score_df["flagged"]].index.tolist()
        if not flagged:
            log.info("no flagged nodes to explain")
            return {}

        # Baseline statistics from non-flagged nodes
        clean_nodes = score_df[~score_df["flagged"]].index
        clean_feats = feat_df.loc[feat_df.index.intersection(clean_nodes)]

        baseline_mean = clean_feats.mean()
        baseline_std  = clean_feats.std().replace(0, 1e-12).clip(lower=1e-12).clip(lower=1e-12)

        cluster_stats = self._cluster_stats(result)

        explanations: Dict[str, Dict] = {}
        for node in flagged:
            if node not in feat_df.index:
                continue
            explanations[node] = self._explain_node(
                node, feat_df, score_df, result,
                baseline_mean, baseline_std, cluster_stats,
            )

        log.info("generated explanations for %d flagged nodes", len(explanations))
        return explanations

    # ── per-node explanation ──────────────────────────────────────────────────

    def _explain_node(
        self,
        node:          str,
        feat_df:       pd.DataFrame,
        score_df:      pd.DataFrame,
        result:        ModelResult,
        baseline_mean: pd.Series,
        baseline_std:  pd.Series,
        cluster_stats: Dict[int, Dict],
    ) -> Dict:
        node_feats = feat_df.loc[node]
        # Guard: loc on a non-unique index returns a DataFrame
        if isinstance(node_feats, pd.DataFrame):
            node_feats = node_feats.iloc[0]
        z_scores   = ((node_feats - baseline_mean) / baseline_std).clip(-50, 50)
        if isinstance(z_scores, pd.DataFrame):
            z_scores = z_scores.iloc[0]

        # Top-N features by absolute z-score
        top_features = self._top_features(node_feats, baseline_mean, z_scores)

        # Cluster context
        cluster_id      = int(result.cluster_labels.get(node, -99))
        cluster_context = self._cluster_context(node, cluster_id, cluster_stats)

        # Deviation summary in natural language
        deviation_summary = self._deviation_summary(top_features, cluster_context)

        return {
            "node_id":          node,
            "fraud_score":      float(score_df.loc[node, "fraud_score"]),
            "anomaly_score":    float(score_df.loc[node, "anomaly_score"]),
            "cluster_id":       cluster_id,
            "top_features":     top_features,
            "deviation_summary": deviation_summary,
            "cluster_context":  cluster_context,
        }

    def _top_features(
        self,
        node_feats:    pd.Series,
        baseline_mean: pd.Series,
        z_scores:      pd.Series,
    ) -> List[Dict]:
        # Only include features that are in our model feature list
        relevant = [f for f in self.cfg.model_features if f in z_scores.index]
        z_clamped = z_scores.clip(-99, 99)
        z_sub    = pd.Series(z_clamped[relevant], dtype=float).sort_values(
            key=np.abs, ascending=False
        )

        top: List[Dict] = []
        for feat in z_sub.index[: self.cfg.top_features_n]:
            top.append({
                "feature":        feat,
                "node_value":     float(node_feats[feat]),
                "baseline_mean":  float(baseline_mean[feat]),
                "z_score":        float(z_scores[feat]),
                "direction":      "HIGH" if z_scores[feat] > 0 else "LOW",
            })
        return top

    @staticmethod
    def _cluster_context(
        node:          str,
        cluster_id:    int,
        cluster_stats: Dict[int, Dict],
    ) -> Dict:
        if cluster_id == -1:
            return {
                "cluster_id":  -1,
                "description": "DBSCAN noise: node does not belong to any dense cluster; "
                               "pattern is highly irregular relative to all neighbours.",
            }
        stats = cluster_stats.get(cluster_id, {})
        size  = stats.get("size", 0)
        mean_a= stats.get("mean_anomaly", 0.0)
        desc  = (
            f"Cluster {cluster_id} has {size} member(s) with mean anomaly "
            f"score {mean_a:.3f}."
        )
        if size < 5:
            desc += "  Very small cluster — consistent with a coordinated bot ring."
        elif mean_a > 0.6:
            desc += "  High mean anomaly across cluster members — coordinated behaviour likely."
        return {
            "cluster_id":       cluster_id,
            "cluster_size":     size,
            "cluster_mean_anomaly": mean_a,
            "description":      desc,
        }

    @staticmethod
    def _deviation_summary(
        top_features: List[Dict],
        cluster_context: Dict,
    ) -> str:
        parts = []
        for f in top_features:
            direction = "elevated" if f["direction"] == "HIGH" else "depressed"
            parts.append(
                f"{f['feature']} is {direction} "
                f"(z={f['z_score']:+.2f}, "
                f"node={f['node_value']:.3g}, "
                f"baseline={f['baseline_mean']:.3g})"
            )
        summary = "; ".join(parts)
        if cluster_context.get("cluster_id") == -1:
            summary += ".  Node is DBSCAN outlier (no cluster)."
        elif cluster_context.get("cluster_size", 99) < 5:
            summary += f".  Member of small cluster (id={cluster_context['cluster_id']})."
        return summary

    # ── cluster-level stats ───────────────────────────────────────────────────

    @staticmethod
    def _cluster_stats(result: ModelResult) -> Dict[int, Dict]:
        stats: Dict[int, Dict] = {}
        for label in result.cluster_labels.unique():
            if label == -1:
                continue
            members = result.cluster_labels[result.cluster_labels == label].index
            stats[int(label)] = {
                "size":         len(members),
                "mean_anomaly": float(result.anomaly_scores[members].mean()),
                "members":      members.tolist(),
            }
        return stats
