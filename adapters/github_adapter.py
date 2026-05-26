"""
GitHubAdapter
─────────────
Converts GitHub follower/following data into canonical edge + node format.

Two input modes
  1. Account snapshot CSV/JSON — columns: username, created_at,
     followers, following, public_repos, bio, avatar_url
  2. Following-list JSON — dict mapping username → [list of followed accounts].
     When following lists are available, pairwise Jaccard similarity is
     computed and similarity edges are added to the graph.

The Jaccard similarity graph is the primary detection surface for the
GitHub botnet described in the DEV.to investigation:
  canestein, hazexone, domcomit, kylehyne, jaderytm,
  vierystein, hanyvert, mariwatts, lynewinter
  — Jaccard ≥ 0.98 across ~29,800 following entries each.

Graph schema
  nodes  = GitHub accounts
  edges  = follow events  (source→target, action_type="follow")
         + similarity edges (sourceA→sourceB, action_type="similar_following",
                              weight=jaccard_score)  — optional, high-signal
"""
from __future__ import annotations

import csv
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from utils.logger import get_logger

log = get_logger(__name__)

# Jaccard threshold above which two accounts get a similarity edge
DEFAULT_JACCARD_THRESHOLD = 0.50


class GitHubAdapter:
    def __init__(
        self,
        target_username: Optional[str] = None,
        jaccard_threshold: float = DEFAULT_JACCARD_THRESHOLD,
    ) -> None:
        self.target            = target_username
        self.jaccard_threshold = jaccard_threshold

    # ── public api ────────────────────────────────────────────────────────────

    def load_accounts(self, path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load account snapshot (CSV or JSON).
        Returns (edges_df, nodes_df).
        If target_username is set, follow-edges point to that target.
        Otherwise edges are omitted (nodes only, for similarity analysis).
        """
        p = Path(path)
        if p.suffix == ".csv":
            raw = self._load_csv(p)
        else:
            raw = self._load_json_accounts(p)

        nodes_df = self._build_nodes(raw)
        edges_df = self._build_follow_edges(nodes_df) if self.target else pd.DataFrame()
        log.info("github adapter: %d accounts loaded", len(nodes_df))
        return edges_df, nodes_df

    def load_following_lists(
        self, path: str, accounts_path: Optional[str] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load following-list JSON: {username: [followed_accounts, ...]}
        Computes pairwise Jaccard similarity and generates similarity edges
        for pairs above jaccard_threshold.

        Returns (edges_df, nodes_df).  nodes_df includes jaccard_mean_similarity
        per account (how similar it is to the rest of the dataset on average).
        """
        p = Path(path)
        with p.open(encoding="utf-8") as f:
            following_lists: Dict[str, List[str]] = json.load(f)

        # Optionally merge account metadata
        if accounts_path:
            _, nodes_df = self.load_accounts(accounts_path)
        else:
            nodes_df = pd.DataFrame(
                {"username": list(following_lists.keys())}
            ).set_index("username")

        # Pairwise Jaccard
        similarity_edges, mean_sims = self._compute_jaccard_edges(following_lists)

        # Attach mean similarity as a node feature
        nodes_df["jaccard_mean_similarity"] = pd.Series(mean_sims)
        nodes_df["jaccard_mean_similarity"] = nodes_df["jaccard_mean_similarity"].fillna(0.0)

        # Build edge frame: similarity edges + optional follow edges
        frames = [similarity_edges]
        if self.target:
            frames.append(self._build_follow_edges(nodes_df))
        edges_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

        log.info(
            "github adapter: %d accounts, %d similarity edges (threshold=%.2f)",
            len(nodes_df), len(similarity_edges), self.jaccard_threshold,
        )
        return edges_df, nodes_df

    # ── loaders ───────────────────────────────────────────────────────────────

    @staticmethod
    def _load_csv(p: Path) -> List[Dict]:
        with p.open(newline="", encoding="utf-8-sig") as f:
            return [dict(r) for r in csv.DictReader(f)]

    @staticmethod
    def _load_json_accounts(p: Path) -> List[Dict]:
        with p.open(encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else [data]

    # ── node building ─────────────────────────────────────────────────────────

    @staticmethod
    def _build_nodes(rows: List[Dict]) -> pd.DataFrame:
        _aliases = {
            "login": "username", "name": "username",
            "created_at": "joined_at",
            "followers": "followers_count",
            "following": "following_count",
            "public_repos": "repos_count",
            "bio": "bio",
            "avatar_url": "avatar_url",
        }
        records = []
        for raw in rows:
            r = {_aliases.get(k, k): v for k, v in raw.items()}
            username = str(r.get("username", "")).strip()
            if not username:
                continue

            joined_ts = None
            joined_raw = r.get("joined_at", "")
            if joined_raw:
                for fmt in (
                    "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S",
                    "%Y-%m-%d %H:%M:%S", "%Y-%m-%d",
                ):
                    try:
                        dt = datetime.strptime(str(joined_raw).strip(), fmt)
                        joined_ts = dt.replace(tzinfo=timezone.utc).timestamp()
                        break
                    except ValueError:
                        continue

            try:
                following = int(float(str(r.get("following_count", 0) or 0)))
            except (ValueError, TypeError):
                following = 0
            try:
                followers = int(float(str(r.get("followers_count", 0) or 0)))
            except (ValueError, TypeError):
                followers = 0
            try:
                repos = int(float(str(r.get("repos_count", 0) or 0)))
            except (ValueError, TypeError):
                repos = 0

            bio = str(r.get("bio", "") or "").strip()

            records.append({
                "username":        username,
                "joined_ts":       joined_ts,
                "following_count": following,
                "followers_count": followers,
                "repos_count":     repos,
                "bio_empty":       1 if not bio else 0,
            })

        df = pd.DataFrame(records)
        return df.set_index("username") if not df.empty else df

    # ── edge building ─────────────────────────────────────────────────────────

    def _build_follow_edges(self, nodes_df: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for username, node in nodes_df.iterrows():
            ts = node.get("joined_ts") or 0.0
            rows.append({
                "user_id":     username,
                "target_id":   self.target,
                "action_type": "follow",
                "timestamp":   float(ts),
            })
        return pd.DataFrame(rows)

    def _compute_jaccard_edges(
        self, following_lists: Dict[str, List[str]]
    ) -> Tuple[pd.DataFrame, Dict[str, float]]:
        """
        Compute pairwise Jaccard similarity between all following lists.
        Returns similarity edge DataFrame + per-account mean similarity dict.

        Uses set intersection; scales to ~thousands of accounts × tens-of-
        thousands of following entries.  For very large datasets, switch to
        MinHash LSH (add datasketch to requirements).
        """
        accounts = list(following_lists.keys())
        sets: Dict[str, Set[str]] = {a: set(following_lists[a]) for a in accounts}
        n = len(accounts)

        edge_rows = []
        sum_sim: Dict[str, float] = {a: 0.0 for a in accounts}
        count: Dict[str, int]     = {a: 0   for a in accounts}

        for i in range(n):
            a = accounts[i]
            sa = sets[a]
            for j in range(i + 1, n):
                b = accounts[j]
                sb = sets[b]
                union_size = len(sa | sb)
                if union_size == 0:
                    continue
                jaccard = len(sa & sb) / union_size
                sum_sim[a] += jaccard
                sum_sim[b] += jaccard
                count[a]   += 1
                count[b]   += 1

                if jaccard >= self.jaccard_threshold:
                    # Use the join_timestamp of the earlier account as edge ts
                    edge_rows.append({
                        "user_id":     a,
                        "target_id":   b,
                        "action_type": "similar_following",
                        "timestamp":   0.0,   # no meaningful timestamp
                        "weight":      round(jaccard, 6),
                    })

        mean_sims = {
            a: sum_sim[a] / count[a] if count[a] > 0 else 0.0
            for a in accounts
        }
        edges_df = pd.DataFrame(edge_rows) if edge_rows else pd.DataFrame(
            columns=["user_id", "target_id", "action_type", "timestamp", "weight"]
        )
        log.info(
            "jaccard: %d account pairs, %d edges above threshold %.2f",
            n * (n - 1) // 2, len(edges_df), self.jaccard_threshold,
        )
        return edges_df, mean_sims
