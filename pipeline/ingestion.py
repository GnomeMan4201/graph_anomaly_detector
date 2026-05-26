"""
DataIngestionLayer
──────────────────
Accepts .jsonl / .json (newline-delimited) or .csv.

Canonical output schema
  user_id      str
  target_id    str
  action_type  str (lower-cased)
  timestamp    float (unix seconds)

Any extra columns in the source are preserved.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from utils.logger import get_logger

log = get_logger(__name__)

# ─── field aliases accepted at ingestion ─────────────────────────────────────

_ALIASES: Dict[str, str] = {
    # user_id
    "user":       "user_id",
    "source":     "user_id",
    "from":       "user_id",
    "from_user":  "user_id",
    "actor":      "user_id",
    # target_id
    "target":     "target_id",
    "to":         "target_id",
    "to_user":    "target_id",
    "dest":       "target_id",
    # action_type
    "action":     "action_type",
    "type":       "action_type",
    "event_type": "action_type",
    "event":      "action_type",
    # timestamp
    "time":       "timestamp",
    "ts":         "timestamp",
    "created_at": "timestamp",
    "occurred_at":"timestamp",
    "date":       "timestamp",
}

_REQUIRED = ("user_id", "target_id", "action_type", "timestamp")


class DataIngestionLayer:
    """
    Thread-unsafe by design (single-pipeline use).
    Call reset() to reuse the instance for a second dataset.
    """

    def __init__(self) -> None:
        self._seen: set[str] = set()
        self._accepted = 0
        self._rejected_missing = 0
        self._rejected_dedup   = 0
        self._rejected_selfloop = 0

    # ── public api ────────────────────────────────────────────────────────────

    def ingest(self, path: str) -> pd.DataFrame:
        p = Path(path)
        if p.suffix in (".jsonl", ".json"):
            return self._from_jsonl(p)
        if p.suffix == ".csv":
            return self._from_csv(p)
        raise ValueError(f"Unsupported file format: {p.suffix!r}")

    def ingest_records(self, records: List[Dict[str, Any]]) -> pd.DataFrame:
        """Ingest from in-memory list (useful for synthetic data injection)."""
        rows = [self._normalize(r) for r in records]
        rows = [r for r in rows if r is not None]
        df = pd.DataFrame(rows)
        self._log_stats(len(records))
        return df

    def reset(self) -> None:
        self._seen.clear()
        self._accepted = self._rejected_missing = 0
        self._rejected_dedup = self._rejected_selfloop = 0

    # ── internal ──────────────────────────────────────────────────────────────

    def _from_jsonl(self, p: Path) -> pd.DataFrame:
        rows = []
        with p.open() as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError as exc:
                    log.warning("line %d: JSON parse error: %s", lineno, exc)
                    continue
                row = self._normalize(raw)
                if row:
                    rows.append(row)
        df = pd.DataFrame(rows)
        self._log_stats(lineno)
        return df

    def _from_csv(self, p: Path) -> pd.DataFrame:
        rows = []
        n_raw = 0
        with p.open(newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                n_raw += 1
                normalized = self._normalize(dict(row))
                if normalized:
                    rows.append(normalized)
        df = pd.DataFrame(rows)
        self._log_stats(n_raw)
        return df

    def _normalize(self, raw: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        # Remap aliased keys
        record: Dict[str, Any] = {}
        for k, v in raw.items():
            canonical = _ALIASES.get(k, k)
            record[canonical] = v

        # Validate required fields
        for field in _REQUIRED:
            if field not in record or record[field] is None or record[field] == "":
                self._rejected_missing += 1
                return None

        # Coerce types
        record["user_id"]    = str(record["user_id"]).strip()
        record["target_id"]  = str(record["target_id"]).strip()
        record["action_type"]= str(record["action_type"]).strip().lower()

        ts = record["timestamp"]
        record["timestamp"]  = self._parse_timestamp(ts)
        if record["timestamp"] is None:
            self._rejected_missing += 1
            return None

        # Drop self-loops
        if record["user_id"] == record["target_id"]:
            self._rejected_selfloop += 1
            return None

        # Deduplication
        key = (
            f"{record['user_id']}|{record['target_id']}"
            f"|{record['action_type']}|{record['timestamp']}"
        )
        h = hashlib.blake2s(key.encode(), digest_size=8).hexdigest()
        if h in self._seen:
            self._rejected_dedup += 1
            return None
        self._seen.add(h)

        self._accepted += 1
        return record

    @staticmethod
    def _parse_timestamp(ts: Any) -> Optional[float]:
        if isinstance(ts, (int, float)):
            return float(ts)
        if isinstance(ts, str):
            ts = ts.strip()
            # Try numeric string first
            try:
                return float(ts)
            except ValueError:
                pass
            # Try ISO 8601
            from datetime import datetime, timezone
            for fmt in (
                "%Y-%m-%dT%H:%M:%S%z",
                "%Y-%m-%dT%H:%M:%S",
                "%Y-%m-%d %H:%M:%S",
                "%Y-%m-%d",
            ):
                try:
                    dt = datetime.strptime(ts, fmt)
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                    return dt.timestamp()
                except ValueError:
                    continue
        return None

    def _log_stats(self, n_raw: int) -> None:
        log.info(
            "ingestion: raw=%d  accepted=%d  dup=%d  missing=%d  selfloop=%d",
            n_raw,
            self._accepted,
            self._rejected_dedup,
            self._rejected_missing,
            self._rejected_selfloop,
        )
