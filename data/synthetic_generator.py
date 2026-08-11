"""
Synthetic interaction dataset generator.

Behaviour profiles
──────────────────
  normal       : organic random interactions, spread over the day,
                 Poisson arrival rate, broad target diversity.
  bot_cluster  : coordinated rings. All bots in a cluster target the
                 same small victim set and fire within a tight burst
                 window (±burst_jitter_s seconds of a shared trigger).
  burst_user   : otherwise-normal user who occasionally dumps a large
                 number of actions in a very short window.
"""
from __future__ import annotations

import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

ACTION_TYPES = ["follow", "like", "comment", "share", "repost"]

_HOURLY_WEIGHT = [
    0.2, 0.1, 0.1, 0.1, 0.1, 0.2,
    0.4, 0.7, 0.9, 1.0, 1.0, 0.9,
    0.8, 0.9, 0.9, 0.8, 0.7, 0.9,
    1.0, 1.0, 0.9, 0.7, 0.5, 0.3,
]


def _random_ts_in_day(day_offset: int, rng: np.random.RandomState) -> float:
    """Return a unix timestamp in the given day, weighted by hour-of-day."""
    hour = rng.choice(24, p=np.array(_HOURLY_WEIGHT) / sum(_HOURLY_WEIGHT))
    minute = rng.randint(0, 60)
    second = rng.randint(0, 60)
    base = datetime(2024, 1, 1, tzinfo=timezone.utc).timestamp()
    return base + day_offset * 86_400 + hour * 3_600 + minute * 60 + second


class SyntheticDataGenerator:
    """
    Parameters
    ──────────
    seed              : reproducibility seed
    n_normal_users    : organic users
    n_bot_clusters    : number of coordinated bot rings
    bots_per_cluster  : bots per ring
    n_target_users    : popular accounts bots target (victim pool)
    n_burst_users     : organic users with periodic burst behaviour
    n_days            : simulation window
    burst_jitter_s    : seconds spread for bot synchronised bursts
    """

    def __init__(
        self,
        seed: int = 42,
        n_normal_users: int = 200,
        n_bot_clusters: int = 5,
        bots_per_cluster: int = 10,
        n_target_users: int = 50,
        n_burst_users: int = 20,
        n_days: int = 7,
        burst_jitter_s: int = 180,
    ) -> None:
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        random.seed(seed)

        self.n_normal_users = n_normal_users
        self.n_bot_clusters = n_bot_clusters
        self.bots_per_cluster = bots_per_cluster
        self.n_target_users = n_target_users
        self.n_burst_users = n_burst_users
        self.n_days = n_days
        self.burst_jitter_s = burst_jitter_s

        self._uid = 0
        self.normal_users: List[str] = self._make_ids("U", n_normal_users)
        self.target_users: List[str] = self._make_ids("T", n_target_users)

        self.bot_clusters: List[tuple] = []
        for c in range(n_bot_clusters):
            bots = self._make_ids(f"BOT{c}_", bots_per_cluster)
            victims = random.sample(self.target_users, k=min(5, n_target_users))
            self.bot_clusters.append((f"cluster_{c}", bots, victims))

        self.burst_users: List[str] = random.sample(
            self.normal_users, k=min(n_burst_users, n_normal_users)
        )

        self._all_accounts: List[str] = (
            self.normal_users
            + self.target_users
            + [b for _, bots, _ in self.bot_clusters for b in bots]
        )

    def _make_ids(self, prefix: str, n: int) -> List[str]:
        ids = [f"{prefix}{self._uid + i:05d}" for i in range(n)]
        self._uid += n
        return ids

    def _normal_events(self) -> List[Dict]:
        events = []
        for user in self.normal_users:
            is_burst = user in self.burst_users
            for day in range(self.n_days):
                n_events = int(self.rng.poisson(lam=15 if not is_burst else 12))
                targets = self.rng.choice(
                    [u for u in self._all_accounts if u != user],
                    size=n_events,
                    replace=True,
                )
                for target in targets:
                    events.append({
                        "user_id": user,
                        "target_id": str(target),
                        "action_type": random.choice(ACTION_TYPES),
                        "timestamp": _random_ts_in_day(day, self.rng),
                    })

                if is_burst and self.rng.random() < 0.25:
                    burst_anchor = _random_ts_in_day(day, self.rng)
                    burst_size = int(self.rng.uniform(30, 80))
                    burst_target = random.choice(
                        [u for u in self._all_accounts if u != user]
                    )
                    for _ in range(burst_size):
                        jitter = self.rng.uniform(0, self.burst_jitter_s * 2)
                        events.append({
                            "user_id": user,
                            "target_id": burst_target,
                            "action_type": random.choice(["like", "follow"]),
                            "timestamp": burst_anchor + jitter,
                        })
        return events

    def _bot_cluster_events(self) -> List[Dict]:
        events = []
        for cluster_id, bots, victims in self.bot_clusters:
            for day in range(self.n_days):
                n_rounds = int(self.rng.randint(2, 6))
                for _ in range(n_rounds):
                    anchor = _random_ts_in_day(day, self.rng)
                    burst_action = random.choice(["follow", "like"])
                    for bot in bots:
                        for victim in victims:
                            jitter = self.rng.uniform(
                                -self.burst_jitter_s, self.burst_jitter_s
                            )
                            events.append({
                                "user_id": bot,
                                "target_id": victim,
                                "action_type": burst_action,
                                "timestamp": anchor + jitter,
                                "_cluster": cluster_id,
                            })
        return events

    def generate(self) -> List[Dict]:
        """Return full event list, shuffled."""
        events = self._normal_events() + self._bot_cluster_events()
        random.shuffle(events)
        return events

    def ground_truth_bots(self) -> List[str]:
        """Return flat list of all synthetic bot user IDs."""
        return [b for _, bots, _ in self.bot_clusters for b in bots]

    def save_jsonl(self, path: str, events: Optional[List[Dict]] = None) -> int:
        """
        Write synthetic events to JSONL and return the record count.

        Pass the exact records returned by ``generate()`` when preserving an
        analyzed run. If ``events`` is omitted, a new dataset is generated.
        Internal ground-truth labels are stripped from the exported file.
        """
        records = events if events is not None else self.generate()
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w") as f:
            for event in records:
                record = {k: v for k, v in event.items() if not k.startswith("_")}
                f.write(json.dumps(record) + "\n")
        return len(records)
