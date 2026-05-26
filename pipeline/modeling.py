"""
AnomalyModeler
──────────────
Runs two independent approaches on the feature matrix:

  1. IsolationForest  → per-node anomaly score ∈ [0, 1]  (1 = most anomalous)
  2. DBSCAN           → cluster assignment (-1 = noise/outlier)

Both are fitted on the same scaled feature matrix.  StandardScaler is used
so that high-magnitude features (total_events) do not dominate distance-based
methods.

ModelResult namedtuple is returned so downstream layers have a stable contract.
"""
from __future__ import annotations

from typing import Dict, NamedTuple

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from config import Config
from utils.logger import get_logger

log = get_logger(__name__)


class ModelResult(NamedTuple):
    anomaly_scores:    pd.Series    # index=node_id, values ∈ [0,1]
    cluster_labels:    pd.Series    # index=node_id, values int (-1=noise)
    raw_if_scores:     pd.Series    # raw IsolationForest decision scores
    scaled_features:   pd.DataFrame # features after StandardScaler


class AnomalyModeler:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self._scaler = StandardScaler()
        self._iso_forest = IsolationForest(
            n_estimators=cfg.isolation_forest_n_estimators,
            contamination=cfg.isolation_forest_contamination,
            random_state=cfg.isolation_forest_random_state,
            n_jobs=-1,
        )
        self._dbscan = DBSCAN(
            eps=cfg.dbscan_eps,
            min_samples=cfg.dbscan_min_samples,
            n_jobs=-1,
        )

    # ── public api ────────────────────────────────────────────────────────────

    def fit_predict(self, feat_df: pd.DataFrame) -> ModelResult:
        """
        feat_df : feature matrix, index = node_id.
        Uses only cfg.model_features columns that are present in the matrix.
        """
        feature_cols = [c for c in self.cfg.model_features if c in feat_df.columns]
        missing = set(self.cfg.model_features) - set(feature_cols)
        if missing:
            log.warning("feature(s) missing from matrix, skipping: %s", missing)

        X_raw = feat_df[feature_cols].values.astype(np.float64)
        X     = self._scaler.fit_transform(X_raw)

        scaled_df = pd.DataFrame(X, index=feat_df.index, columns=feature_cols)

        # ── IsolationForest ───────────────────────────────────────────────────
        # score_samples: lower (more negative) = more anomalous
        if_raw = self._iso_forest.fit(X).score_samples(X)
        anomaly_scores = self._minmax_invert(if_raw)

        # ── DBSCAN ────────────────────────────────────────────────────────────
        cluster_labels = self._dbscan.fit_predict(X)

        log.info(
            "IsolationForest: contamination=%.2f  n_anomalous=%d",
            self.cfg.isolation_forest_contamination,
            int((anomaly_scores >= 0.5).sum()),
        )
        n_clusters  = len(set(cluster_labels) - {-1})
        n_noise     = int((cluster_labels == -1).sum())
        log.info(
            "DBSCAN: eps=%.2f  min_samples=%d  clusters=%d  noise=%d",
            self.cfg.dbscan_eps,
            self.cfg.dbscan_min_samples,
            n_clusters,
            n_noise,
        )

        return ModelResult(
            anomaly_scores  = pd.Series(anomaly_scores,  index=feat_df.index),
            cluster_labels  = pd.Series(cluster_labels,  index=feat_df.index),
            raw_if_scores   = pd.Series(if_raw,           index=feat_df.index),
            scaled_features = scaled_df,
        )

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _minmax_invert(arr: np.ndarray) -> np.ndarray:
        """
        Normalise arr to [0, 1] and invert so that the most anomalous
        (lowest raw score) maps to 1.0.
        """
        mn, mx = arr.min(), arr.max()
        if mx - mn < 1e-12:
            return np.zeros_like(arr)
        return 1.0 - (arr - mn) / (mx - mn)

    def cluster_stats(self, result: ModelResult) -> Dict[int, Dict]:
        """
        Returns per-cluster statistics keyed by cluster_id.
        Cluster -1 = DBSCAN noise.
        """
        stats: Dict[int, Dict] = {}
        for label in sorted(result.cluster_labels.unique()):
            members = result.cluster_labels[result.cluster_labels == label].index.tolist()
            member_scores = result.anomaly_scores[members]
            stats[label] = {
                "size":          len(members),
                "mean_anomaly":  float(member_scores.mean()),
                "max_anomaly":   float(member_scores.max()),
                "members":       members,
                "is_noise":      label == -1,
            }
        return stats
