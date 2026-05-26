"""
SocialFeatureExtractor
──────────────────────
Computes features from social platform account metadata that are invisible
to pure graph analysis but are the PRIMARY discriminators for commercial
follower-inflation botnets.

Derived from actual signals in the DEV.to / GitHub investigations:

                           across all 1,409 audited accounts in the DEV.to dataset)
  follow_ratio           : following / (followers + 1)  — bots push this to ∞
  profile_completeness   : composite of bio, avatar, articles, comments
  s3_id                  : creation-order proxy from DEV.to avatar URL
  s3_batch_density       : accounts within ±batch_window S3 IDs (batch detection)
  creation_wave_id       : KMeans cluster on join_ts (synchronised batch signal)
  account_age_days       : days since join (fresh accounts = higher risk)
  username_hash_suffix   : regex match for `_[hex]{6,}` username pattern
  jaccard_mean_similarity: mean following-list Jaccard to other cluster members
                           (GitHub-specific, 0 if not available)

All features are normalised to floats.  Missing / None values are filled
with column medians (neutral assumption) rather than zero, to avoid biasing
the anomaly model toward accounts with missing metadata.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

from utils.logger import get_logger

log = get_logger(__name__)

# S3 ID window for batch density computation
DEFAULT_BATCH_WINDOW = 5_000


class SocialFeatureExtractor:
    """
    Parameters
    ──────────
    batch_window     : S3 ID radius for batch density (default 5000)
    n_creation_waves : number of KMeans clusters for join_ts (default 5)
    """

    def __init__(
        self,
        batch_window:      int = DEFAULT_BATCH_WINDOW,
        n_creation_waves:  int = 5,
    ) -> None:
        self.batch_window     = batch_window
        self.n_creation_waves = n_creation_waves

    # ── public api ────────────────────────────────────────────────────────────

    def extract(self, nodes_df: pd.DataFrame) -> pd.DataFrame:
        """
        nodes_df index = username/user_id.
        Expected columns (all optional — missing ones produce 0/NaN):
          following_count, followers_count, articles_count, comments_count,
          bio_empty, default_avatar, username_hash_suffix, s3_id, joined_ts,
          audit_score, jaccard_mean_similarity

        Returns a feature DataFrame with the same index.
        """
        if nodes_df.empty:
            return pd.DataFrame()

        feats = pd.DataFrame(index=nodes_df.index)

            self._col(nodes_df, "following_count") == 1
        ).astype(float)

        following = self._col(nodes_df, "following_count")
        followers = self._col(nodes_df, "followers_count")
        feats["follow_ratio"] = following / (followers + 1.0)

        feats["profile_completeness"] = self._profile_completeness(nodes_df)

        feats["bio_empty"]            = self._col(nodes_df, "bio_empty").astype(float)
        feats["default_avatar"]       = self._col(nodes_df, "default_avatar").astype(float)
        feats["username_hash_suffix"]  = self._col(nodes_df, "username_hash_suffix").astype(float)

        # articles / comments as raw counts (log-scaled to reduce range)
        feats["log_articles"]  = np.log1p(self._col(nodes_df, "articles_count"))
        feats["log_comments"]  = np.log1p(self._col(nodes_df, "comments_count"))
        feats["log_following"] = np.log1p(following)
        feats["log_followers"] = np.log1p(followers)

        # S3-ID features (DEV.to specific)
        if "s3_id" in nodes_df.columns:
            feats["s3_id_norm"]        = self._normalize_s3(nodes_df["s3_id"])
            feats["s3_batch_density"]  = self._s3_batch_density(nodes_df["s3_id"])
        else:
            feats["s3_id_norm"]       = 0.0
            feats["s3_batch_density"] = 0.0

        # Temporal features
        if "joined_ts" in nodes_df.columns:
            feats["account_age_days"] = self._account_age(nodes_df["joined_ts"])
            feats["creation_wave_id"] = self._creation_wave(nodes_df["joined_ts"])
        else:
            feats["account_age_days"] = 365.0  # assume non-fresh if unknown
            feats["creation_wave_id"] = 0.0

        # GitHub Jaccard similarity
        if "jaccard_mean_similarity" in nodes_df.columns:
            feats["jaccard_mean_similarity"] = (
                nodes_df["jaccard_mean_similarity"].fillna(0.0).astype(float)
            )
        else:
            feats["jaccard_mean_similarity"] = 0.0

        # Audit score passthrough (if available from the devto-botnet-hunter CSV)
        if "audit_score" in nodes_df.columns:
                nodes_df["audit_score"].replace(-1, np.nan).fillna(0.0).astype(float)
            )
        else:

        feats = self._fill_missing(feats)
        log.info(
            "social features: %d accounts × %d features",
            feats.shape[0], feats.shape[1],
        )
        return feats

    # ── feature computations ──────────────────────────────────────────────────

    def _profile_completeness(self, df: pd.DataFrame) -> pd.Series:
        """
        Score 0–1: fraction of profile fields that are populated.
        Fields: bio present, non-default avatar, ≥1 article, ≥1 comment.
        """
        bio_ok    = (1 - self._col(df, "bio_empty"))
        avatar_ok = (1 - self._col(df, "default_avatar"))
        art_ok    = (self._col(df, "articles_count") > 0).astype(float)
        cmt_ok    = (self._col(df, "comments_count") > 0).astype(float)
        return (bio_ok + avatar_ok + art_ok + cmt_ok) / 4.0

    def _normalize_s3(self, s3_series: pd.Series) -> pd.Series:
        """Min-max normalise S3 IDs to [0, 1]."""
        s3 = s3_series.fillna(0.0).astype(float)
        mn, mx = s3.min(), s3.max()
        if mx - mn < 1e-9:
            return pd.Series(0.0, index=s3.index)
        return (s3 - mn) / (mx - mn)

    def _s3_batch_density(self, s3_series: pd.Series) -> pd.Series:
        """
        For each account, count how many other accounts have an S3 ID within
        ±batch_window.  High density signals a creation batch.
        Normalised by total account count.
        """
        s3 = s3_series.fillna(-1).astype(float).values
        valid = s3 > 0
        n = len(s3)
        density = np.zeros(n)

        # Only operate on valid S3 IDs; sort for efficient windowed count
        valid_ids   = np.sort(s3[valid])
        valid_idx   = np.where(valid)[0]

        for pos, i in enumerate(valid_idx):
            lo = np.searchsorted(valid_ids, s3[i] - self.batch_window, side="left")
            hi = np.searchsorted(valid_ids, s3[i] + self.batch_window, side="right")
            density[i] = (hi - lo - 1) / max(n, 1)   # subtract self, normalise

        return pd.Series(density, index=s3_series.index)

    def _account_age(self, ts_series: pd.Series) -> pd.Series:
        """Days since account creation.  More recent = higher risk score."""
        import time
        now = time.time()
        age = ts_series.fillna(now).astype(float)
        days = (now - age) / 86_400.0
        # Clip to reasonable range, invert so fresh accounts score high
        days = days.clip(lower=0, upper=3650)
        # Normalise: 0 days → 1.0 (most suspicious), 3650 days → 0.0
        return 1.0 - (days / 3650.0)

    def _creation_wave(self, ts_series: pd.Series) -> pd.Series:
        """
        KMeans cluster ID on join timestamps (n=n_creation_waves).
        Accounts in the same cluster were created in the same batch.
        Returns normalised cluster ID [0,1].
        """
        ts = ts_series.fillna(ts_series.median()).astype(float).values
        if len(ts) < self.n_creation_waves:
            return pd.Series(0.0, index=ts_series.index)

        try:
            km = KMeans(
                n_clusters=self.n_creation_waves,
                random_state=42,
                n_init=10,
            )
            labels = km.fit_predict(ts.reshape(-1, 1)).astype(float)
            # Normalise cluster IDs to [0, 1]
            labels = (labels - labels.min()) / (labels.max() - labels.min() + 1e-9)
        except Exception:
            labels = np.zeros(len(ts))

        return pd.Series(labels, index=ts_series.index)

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _col(df: pd.DataFrame, col: str) -> pd.Series:
        if col in df.columns:
            return df[col].fillna(0).astype(float)
        return pd.Series(0.0, index=df.index)

    @staticmethod
    def _fill_missing(df: pd.DataFrame) -> pd.DataFrame:
        for col in df.columns:
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)
            df[col] = df[col].fillna(df[col].median())
        return df

    # ── feature list for downstream ───────────────────────────────────────────

    @staticmethod
    def feature_names() -> List[str]:
        return [
            "follow_ratio",
            "profile_completeness",
            "bio_empty",
            "default_avatar",
            "username_hash_suffix",
            "log_articles",
            "log_comments",
            "log_following",
            "log_followers",
            "s3_id_norm",
            "s3_batch_density",
            "account_age_days",
            "creation_wave_id",
            "jaccard_mean_similarity",
        ]
