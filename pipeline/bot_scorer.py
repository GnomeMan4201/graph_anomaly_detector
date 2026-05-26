"""
DirectBotScorer
───────────────
Rule-based fraud scorer derived from known botnet signal patterns.

Used instead of (or combined with) IsolationForest when the bot fraction
is too high for unsupervised anomaly detection to work correctly.
IsolationForest assumes bots are the MINORITY outlier class.  When bots
are the majority (e.g. DEV.to dataset: 63.7%), it treats the bot cluster
as "normal" and flags outlier humans instead.

This scorer directly maps feature values to a fraud signal using weights
informed by the research findings:

  profile_completeness   weight=0.20  (inverted — empty profile → high score)
  username_hash_suffix   weight=0.10  (_hex suffix pattern)
  bio_empty              weight=0.08  (redundant with completeness but additive)
  s3_batch_density       weight=0.07  (batch creation clustering)

Cluster-level upgrade: after individual scoring, the cluster centroid is
evaluated.  Clusters whose centroid looks bot-like get their members'
scores boosted.  This correctly flags the large pure-bot clusters that
DBSCAN separates correctly but the anomaly scorer misses.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from utils.logger import get_logger

log = get_logger(__name__)

# Default weights for individual-level bot signals
_DEFAULT_WEIGHTS: Dict[str, float] = {
    "profile_completeness":  0.25,   # inverted — empty profile → high score
    "username_hash_suffix":  0.12,   # _hex suffix pattern
    "bio_empty":             0.06,   # no bio
    "s3_batch_density":      0.04,   # batch creation (same S3 ID window)
    "account_age_days":      0.03,   # fresh accounts score higher (already inverted)
}

# Cluster-level centroid threshold: clusters whose centroid bot_score
# exceeds this value are considered bot clusters and members get boosted.
_CLUSTER_BOT_CENTROID_THRESHOLD = 0.50


class DirectBotScorer:
    """
    Parameters
    ──────────
    weights       : override the default per-feature weights
    cluster_boost : how much to add to cluster members when their cluster
                    centroid is bot-like (default 0.15)
    """

    def __init__(
        self,
        weights: Dict[str, float] | None = None,
        cluster_boost: float = 0.15,
    ) -> None:
        self.weights       = weights or _DEFAULT_WEIGHTS
        self.cluster_boost = cluster_boost

    def score(
        self,
        feat_df:        pd.DataFrame,
        cluster_labels: pd.Series,
    ) -> pd.Series:
        """
        Returns a bot_score Series ∈ [0, 1], indexed like feat_df.
        """
        scores = pd.Series(0.0, index=feat_df.index)
        total_weight = 0.0

        for feature, weight in self.weights.items():
            if feature not in feat_df.columns:
                continue
            col = feat_df[feature].fillna(0.0).astype(float)

            # Auto-exclude zero-variance features: a feature that is constant
            # in a follower spike dataset) carries no discriminative signal
            # within that cohort and should not contribute to scoring.
            if col.std() < 1e-6:
                log.debug(
                    "feature '%s' has zero variance across cohort — excluded from scoring",
                    feature,
                )
                continue

            # profile_completeness is inverted: high completeness = low bot score
            if feature == "profile_completeness":
                col = 1.0 - col

            # account_age_days is already inverted in SocialFeatureExtractor
            # (recent = high value = high risk) — use as-is

            scores += weight * col.clip(0.0, 1.0)
            total_weight += weight

        # Renormalise by actual weight used (handles excluded features)
        if total_weight > 1e-9:
            scores = scores / total_weight

        # Normalise range to [0,1]
        mx = scores.max()
        if mx > 1e-9:
            scores = scores / mx

        # Cluster-level boost: if a cluster's mean bot_score is above threshold,
        # boost all members (captures the large tight bot clusters)
        scores = self._apply_cluster_boost(scores, cluster_labels)

        n_above = int((scores >= 0.50).sum())
        log.info(
            "direct bot scoring: mean=%.3f  >0.5=%d / %d  "
            "(effective_weight=%.2f)",
            float(scores.mean()), n_above, len(scores), total_weight,
        )
        return scores

    def _apply_cluster_boost(
        self,
        scores:         pd.Series,
        cluster_labels: pd.Series,
    ) -> pd.Series:
        out = scores.copy()
        for label in cluster_labels.unique():
            if label == -1:
                continue
            members      = cluster_labels[cluster_labels == label].index
            members_in   = scores.index.intersection(members)
            centroid_score = float(scores.loc[members_in].mean())
            if centroid_score >= _CLUSTER_BOT_CENTROID_THRESHOLD:
                log.debug(
                    "cluster %d centroid=%.3f → boosting %d members",
                    label, centroid_score, len(members_in),
                )
                out.loc[members_in] = (
                    out.loc[members_in] + self.cluster_boost
                ).clip(0.0, 1.0)
        return out
