"""
BotMutator
──────────
Generates adversarially mutated versions of a bot dataset to test whether
the detection pipeline survives feature drift.

Mutation strategies modelled on real operator evasion tactics:

  bio_injection         : add a plausible bio string to bot accounts
  target_spread         : bots follow N targets instead of 1
                          (defeats following_one=1 heuristic)
  timestamp_jitter      : add Gaussian noise to event timestamps
                          (defeats burst_regularity detection if jitter > window)
  username_normalization: replace _hexsuffix patterns with clean names
                          (defeats username_hash_suffix heuristic)
  follower_seeding      : inject fake followers for each bot
                          (defeats fan_in_ratio = 0 signal)
  action_diversification: vary action types across follow/like/comment
                          (defeats action_entropy = low signal)
  combined              : all of the above at moderate levels
  full_adversarial      : all at maximum levels

Each mutation returns a modified copy of the interaction DataFrame and
optionally modified nodes_df — it never modifies the originals.
"""
from __future__ import annotations

import random
import string
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from utils.logger import get_logger

log = get_logger(__name__)

ACTION_POOL = ["follow", "like", "comment", "share", "repost"]

BIO_POOL = [
    "Software developer and tech enthusiast.",
    "Learning to code every day.",
    "Open source contributor.",
    "Building things on the web.",
    "Exploring new technologies.",
    "Developer • Writer • Learner",
    "Passionate about clean code.",
    "Full stack developer.",
    "Always be learning.",
    "Tech, coffee, code.",
]


@dataclass
class MutationConfig:
    name:                    str
    bio_injection_rate:      float = 0.0   # fraction of bots that get a bio
    target_spread:           int   = 1     # max additional targets per bot
    timestamp_jitter_std:    float = 0.0   # seconds of Gaussian jitter
    username_normalize_rate: float = 0.0   # fraction with hex suffix removed
    follower_seed_count:     int   = 0     # fake followers injected per bot
    action_diversify_rate:   float = 0.0   # fraction of actions diversified
    description:             str   = ""


MUTATION_SUITE: List[MutationConfig] = [
    MutationConfig(
        "baseline",
        description="No mutation — establishes detection baseline.",
    ),
    MutationConfig(
        "bio_injection",
        bio_injection_rate=0.90,
        description="90% of bots get a plausible bio. Defeats empty-bio heuristic.",
    ),
    MutationConfig(
        "target_spread_3",
        target_spread=3,
        description="Bots follow 3 targets instead of 1. Defeats following_one=1 signal.",
    ),
    MutationConfig(
        "target_spread_10",
        target_spread=10,
        description="Bots follow 10 targets. Tests normalized target entropy robustness.",
    ),
    MutationConfig(
        "timestamp_jitter_1h",
        timestamp_jitter_std=3_600,
        description="±1h timestamp jitter. Tests burst_regularity robustness.",
    ),
    MutationConfig(
        "timestamp_jitter_6h",
        timestamp_jitter_std=21_600,
        description="±6h jitter. Heavy temporal obfuscation.",
    ),
    MutationConfig(
        "username_cleanup",
        username_normalize_rate=1.0,
        description="All hex suffixes removed. Defeats username_hash_suffix.",
    ),
    MutationConfig(
        "follower_seeding_10",
        follower_seed_count=10,
        description="Each bot gets 10 fake followers. Tests fan_in_ratio robustness.",
    ),
    MutationConfig(
        "action_diversify",
        action_diversify_rate=0.8,
        description="80% of actions randomised across all types. Tests action_entropy.",
    ),
    MutationConfig(
        "combined_moderate",
        bio_injection_rate=0.8,
        target_spread=3,
        timestamp_jitter_std=3_600,
        username_normalize_rate=0.5,
        follower_seed_count=5,
        action_diversify_rate=0.5,
        description="Moderate combined evasion across all dimensions.",
    ),
    MutationConfig(
        "full_adversarial",
        bio_injection_rate=1.0,
        target_spread=10,
        timestamp_jitter_std=14_400,
        username_normalize_rate=1.0,
        follower_seed_count=20,
        action_diversify_rate=0.9,
        description="Maximum evasion. Tests structural robustness limits.",
    ),
]


class BotMutator:
    def __init__(self, seed: int = 42) -> None:
        self.rng = np.random.RandomState(seed)
        random.seed(seed)

    def mutate(
        self,
        edges_df:   pd.DataFrame,
        bot_ids:    List[str],
        cfg:        MutationConfig,
        nodes_df:   Optional[pd.DataFrame] = None,
        victim_pool: Optional[List[str]]   = None,
    ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """
        Apply a mutation config to the bot subset of edges_df.
        Returns (mutated_edges_df, mutated_nodes_df).
        """
        bot_set    = set(bot_ids)
        edges_out  = edges_df.copy()
        nodes_out  = nodes_df.copy() if nodes_df is not None else None

        if not bot_ids:
            return edges_out, nodes_out

        log.info("mutator: applying '%s' to %d bots", cfg.name, len(bot_ids))

        # ── timestamp jitter ─────────────────────────────────────────────────
        if cfg.timestamp_jitter_std > 0:
            bot_mask = edges_out["user_id"].isin(bot_set)
            jitter   = self.rng.normal(0, cfg.timestamp_jitter_std,
                                       int(bot_mask.sum()))
            edges_out.loc[bot_mask, "timestamp"] = (
                edges_out.loc[bot_mask, "timestamp"].values + jitter
            )

        # ── target spread ─────────────────────────────────────────────────────
        if cfg.target_spread > 1 and victim_pool:
            new_rows = []
            for bot in bot_ids:
                extra_victims = random.sample(
                    [v for v in victim_pool if v != bot],
                    k=min(cfg.target_spread - 1, len(victim_pool) - 1),
                )
                for victim in extra_victims:
                    ts = edges_out.loc[edges_out["user_id"] == bot, "timestamp"]
                    base_ts = float(ts.mean()) if len(ts) else 0.0
                    new_rows.append({
                        "user_id":     bot,
                        "target_id":   victim,
                        "action_type": random.choice(["follow", "like"]),
                        "timestamp":   base_ts + self.rng.uniform(-300, 300),
                    })
            if new_rows:
                edges_out = pd.concat(
                    [edges_out, pd.DataFrame(new_rows)], ignore_index=True
                )

        # ── action diversification ────────────────────────────────────────────
        if cfg.action_diversify_rate > 0:
            bot_mask = edges_out["user_id"].isin(bot_set)
            indices  = edges_out.index[bot_mask].tolist()
            n_change = int(len(indices) * cfg.action_diversify_rate)
            change_idx = self.rng.choice(indices, n_change, replace=False)
            edges_out.loc[change_idx, "action_type"] = [
                random.choice(ACTION_POOL) for _ in range(n_change)
            ]

        # ── node-level mutations ──────────────────────────────────────────────
        if nodes_out is not None:
            bots_in_nodes = [b for b in bot_ids if b in nodes_out.index]

            # Bio injection
            if cfg.bio_injection_rate > 0 and "bio_empty" in nodes_out.columns:
                n_inject = int(len(bots_in_nodes) * cfg.bio_injection_rate)
                inject_ids = self.rng.choice(bots_in_nodes, n_inject, replace=False)
                nodes_out.loc[inject_ids, "bio_empty"] = 0

            # Username normalization
            if cfg.username_normalize_rate > 0 and "username_hash_suffix" in nodes_out.columns:
                n_clean = int(len(bots_in_nodes) * cfg.username_normalize_rate)
                clean_ids = self.rng.choice(bots_in_nodes, n_clean, replace=False)
                nodes_out.loc[clean_ids, "username_hash_suffix"] = 0

            # Follower seeding
            if cfg.follower_seed_count > 0 and "followers_count" in nodes_out.columns:
                nodes_out.loc[bots_in_nodes, "followers_count"] = (
                    nodes_out.loc[bots_in_nodes, "followers_count"].fillna(0)
                    + cfg.follower_seed_count
                )

        return edges_out, nodes_out
