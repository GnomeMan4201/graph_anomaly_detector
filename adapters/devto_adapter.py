"""
DevToAdapter
────────────
Converts DEV.to follower data (CSV or JSON from the API) into:

  edges_df     canonical interaction format (user_id, target_id,
               action_type="follow", timestamp)
  nodes_df     per-account metadata used by SocialFeatureExtractor

Supported input formats
  CSV  — devto_bot_audit_full.csv from devto-botnet-hunter
  JSON — followers_raw.json (raw DEV.to API response)
  JSONL — one API profile object per line

Column aliases handled
  The CSV uses capitalised keys (Username, JoinedDate, Following, …);
  the API JSON uses snake_case (username, joined_at, following_count, …).
  Both are normalised to a single canonical schema here.
"""
from __future__ import annotations

import csv
import json
import re
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from utils.logger import get_logger

log = get_logger(__name__)

# ── column alias map (source key → canonical key) ─────────────────────────────

_CSV_ALIASES: Dict[str, str] = {
    # username
    "username":          "username",
    "Username":          "username",
    "user":              "username",
    # following
    "following":         "following_count",
    "Following":         "following_count",
    "following_count":   "following_count",
    # followers
    "followers":         "followers_count",
    "Followers":         "followers_count",
    "followers_count":   "followers_count",
    # articles
    "articles":          "articles_count",
    "Articles":          "articles_count",
    "public_articles_count": "articles_count",
    # bio
    "bio":               "bio",
    "Bio":               "bio",
    "summary":           "bio",
    # avatar
    "profile_image":     "avatar_url",
    "ProfileImage":      "avatar_url",
    "avatar":            "avatar_url",
    # join date
    "joined_at":         "joined_at",
    "JoinedDate":        "joined_at",
    "join_date":         "joined_at",
    "created_at":        "joined_at",
    # comments
    "comments_count":    "comments_count",
    "Comments":          "comments_count",
    # heuristic score (from audit tool)
    "score":             "audit_score",
    "Score":             "audit_score",
    "BotScore":          "audit_score",
    # reasons
    "reasons":           "audit_reasons",
    "Reasons":           "audit_reasons",
}


class DevToAdapter:
    """
    Parameters
    ──────────
    target_username : the account that was followed (becomes edge target_id)
    """

    def __init__(self, target_username: str = "unknown_target") -> None:
        self.target = target_username

    # ── public api ────────────────────────────────────────────────────────────

    def load(self, path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Auto-detect format and return (edges_df, nodes_df).
        """
        p = Path(path)
        if p.suffix == ".csv":
            raw = self._load_csv(p)
        elif p.suffix in (".json", ".jsonl"):
            raw = self._load_json(p)
        else:
            raise ValueError(f"Unsupported format: {p.suffix}")

        nodes_df = self._build_nodes(raw)
        edges_df = self._build_edges(nodes_df)
        log.info(
            "devto adapter: loaded %d accounts → %d edges (target=%s)",
            len(nodes_df), len(edges_df), self.target,
        )
        return edges_df, nodes_df

    # ── loaders ───────────────────────────────────────────────────────────────

    def _load_csv(self, p: Path) -> List[Dict]:
        rows = []
        with p.open(newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(self._remap(dict(row)))
        return rows

    def _load_json(self, p: Path) -> List[Dict]:
        rows = []
        with p.open(encoding="utf-8") as f:
            content = f.read().strip()
            # Try array first, then JSONL
            if content.startswith("["):
                raw_list = json.loads(content)
            else:
                raw_list = [json.loads(line) for line in content.splitlines() if line.strip()]
        for item in raw_list:
            rows.append(self._remap(item))
        return rows

    @staticmethod
    def _remap(raw: Dict) -> Dict:
        out = {}
        for k, v in raw.items():
            canonical = _CSV_ALIASES.get(k, k)
            out[canonical] = v
        return out

    # ── node building ─────────────────────────────────────────────────────────

    def _build_nodes(self, rows: List[Dict]) -> pd.DataFrame:
        records = []
        for r in rows:
            username = str(r.get("username", "")).strip()
            if not username:
                continue

            joined_at   = self._parse_date(r.get("joined_at", ""))
            avatar_url  = str(r.get("avatar_url", ""))
            s3_id       = self._extract_s3_id(avatar_url)
            bio         = str(r.get("bio", "") or "").strip()
            following   = self._safe_int(r.get("following_count", 0))
            followers   = self._safe_int(r.get("followers_count", 0))
            articles    = self._safe_int(r.get("articles_count", 0))
            comments    = self._safe_int(r.get("comments_count", 0))
            audit_score = self._safe_int(r.get("audit_score", -1))

            records.append({
                "username":       username,
                "joined_at":      joined_at,
                "joined_ts":      joined_at.timestamp() if joined_at else None,
                "s3_id":          s3_id,
                "following_count": following,
                "followers_count": followers,
                "articles_count": articles,
                "comments_count": comments,
                "bio_empty":      1 if not bio else 0,
                "default_avatar": 1 if self._is_default_avatar(avatar_url) else 0,
                "username_hash_suffix": 1 if re.search(r"_[a-f0-9]{6,}$", username) else 0,
                "audit_score":    audit_score,
                "avatar_url":     avatar_url,
            })

        df = pd.DataFrame(records)
        if not df.empty:
            df = df.set_index("username")
        log.debug("nodes_df shape: %s", df.shape)
        return df

    # ── edge building ─────────────────────────────────────────────────────────

    def _build_edges(self, nodes_df: pd.DataFrame) -> pd.DataFrame:
        """
        One edge per follower: follower → target_username.
        Timestamp = account join date (best available proxy).
        """
        rows = []
        for username, node in nodes_df.iterrows():
            ts = node.get("joined_ts")
            if ts is None:
                # Fall back to S3 ID as a rough ordering proxy (not a real timestamp)
                s3 = node.get("s3_id")
                ts = float(s3) if s3 else 0.0

            rows.append({
                "user_id":     username,
                "target_id":   self.target,
                "action_type": "follow",
                "timestamp":   float(ts),
            })
        return pd.DataFrame(rows)

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _extract_s3_id(avatar_url: str) -> Optional[int]:
        """
        DEV.to avatar URLs encode the S3 path: …/profile_image/{id}/…
        The URL may be URL-encoded once or twice through the CDN proxy.
        """
        try:
            decoded = urllib.parse.unquote(urllib.parse.unquote(avatar_url))
            m = re.search(r"/profile_image/(\d+)/", decoded)
            if m:
                return int(m.group(1))
        except Exception:
            pass
        return None

    @staticmethod
    def _is_default_avatar(url: str) -> bool:
        return bool(
            not url
            or "default_profile_image" in url.lower()
            or url.endswith("/assets/images/default.png")
        )

    @staticmethod
    def _parse_date(val: object) -> Optional[datetime]:
        if not val or str(val).strip() in ("", "nan", "None"):
            return None
        s = str(val).strip()
        for fmt in (
            "%Y-%m-%dT%H:%M:%S.%fZ",
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%d %H:%M:%S",
            "%B %d, %Y",
            "%b %d, %Y",
            "%Y-%m-%d",
            "%m/%d/%Y",
        ):
            try:
                dt = datetime.strptime(s, fmt)
                return dt.replace(tzinfo=timezone.utc)
            except ValueError:
                continue
        return None

    @staticmethod
    def _safe_int(val: object) -> int:
        try:
            return int(float(str(val).strip() or "0"))
        except (ValueError, TypeError):
            return 0
