"""
FraudScorer
───────────
Combines three signal streams into a single fraud score ∈ [0, 1]:

  signal_1  anomaly_score          from IsolationForest (already [0,1])
  signal_2  centrality_deviation   z-score–based deviation in graph centrality
  signal_3  cluster_density_anomaly  DBSCAN noise penalty + small-cluster penalty

Weights are read from Config.scoring_weights (must sum to 1.0).
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from config import Config
from pipeline.modeling import ModelResult
from utils.logger import get_logger

log = get_logger(__name__)

_CENTRALITY_COLS = ("pagerank", "betweenness", "out_degree", "degree_ratio")


class FraudScorer:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg

    # ── public api ────────────────────────────────────────────────────────────

    def score(
        self,
        feat_df: pd.DataFrame,
        result: ModelResult,
    ) -> pd.DataFrame:
        """
        Returns a DataFrame indexed by node_id with columns:
          anomaly_score, centrality_deviation, cluster_density_anomaly,
          fraud_score, flagged
        """
        s1 = result.anomaly_scores.rename("anomaly_score")
        s2 = self._centrality_deviation(feat_df)
        s3 = self._cluster_density_anomaly(result)

        scores = pd.concat([s1, s2, s3], axis=1).reindex(feat_df.index)

        w = self.cfg.scoring_weights
        scores["fraud_score"] = (
            w["anomaly_score"]           * scores["anomaly_score"]
            + w["centrality_deviation"]  * scores["centrality_deviation"]
            + w["cluster_density_anomaly"] * scores["cluster_density_anomaly"]
        ).clip(0.0, 1.0)

        scores["flagged"] = scores["fraud_score"] >= self.cfg.flag_threshold

        n_flagged = int(scores["flagged"].sum())
        log.info(
            "fraud scoring: flagged=%d / %d  (threshold=%.2f)",
            n_flagged, len(scores), self.cfg.flag_threshold,
        )
        return scores

    # ── signal components ─────────────────────────────────────────────────────

    def _centrality_deviation(self, feat_df: pd.DataFrame) -> pd.Series:
        """
        For each node compute the max absolute z-score across centrality
        columns, then normalise to [0, 1] via sigmoid.

        High out-degree relative to in-degree, or abnormal pagerank/betweenness,
        push this signal up.
        """
        cols = [c for c in _CENTRALITY_COLS if c in feat_df.columns]
        if not cols:
            return pd.Series(0.0, index=feat_df.index, name="centrality_deviation")

        sub = feat_df[cols].copy()
        z   = (sub - sub.mean()) / (sub.std() + 1e-12)

        # Take the maximum absolute z-score per node across the centrality dims
        max_abs_z = z.abs().max(axis=1)

        # Sigmoid normalisation: centre on z=3 so clear outliers push toward 1
        normalised = 1.0 / (1.0 + np.exp(-(max_abs_z - 3.0)))
        return normalised.rename("centrality_deviation")

    @staticmethod
    def _cluster_density_anomaly(result: ModelResult) -> pd.Series:
        """
        Noise nodes (-1) → score 1.0
        Nodes in very small clusters (size < 5) → proportionally elevated score
        All other nodes → 0.0
        """
        labels = result.cluster_labels
        cluster_sizes: Dict[int, int] = labels.value_counts().to_dict()

        scores: pd.Series = pd.Series(0.0, index=labels.index)

        for node, label in labels.items():
            if label == -1:
                scores[node] = 1.0
            else:
                size = cluster_sizes.get(label, 1)
                # Small clusters are suspicious (tight coordinated rings)
                if size < 5:
                    scores[node] = 1.0 - (size / 5.0)   # e.g. size=2 → 0.6
        return scores.rename("cluster_density_anomaly")
