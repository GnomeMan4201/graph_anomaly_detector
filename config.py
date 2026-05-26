from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class Config:
    # ── Scoring weights (must sum to 1.0) ─────────────────────────────────────
    scoring_weights: Dict[str, float] = field(default_factory=lambda: {
        "anomaly_score":           0.45,
        "centrality_deviation":    0.25,
        "cluster_density_anomaly": 0.30,
    })

    # ── Temporal ──────────────────────────────────────────────────────────────
    time_window_hours: int   = 24       # default window for temporal slicing
    burst_window_seconds: int = 3_600   # window for burst activity rate

    # ── IsolationForest ───────────────────────────────────────────────────────
    isolation_forest_contamination: float = 0.10
    isolation_forest_n_estimators:  int   = 200
    isolation_forest_random_state:  int   = 42

    # ── DBSCAN ────────────────────────────────────────────────────────────────
    dbscan_eps:         float = 0.6
    dbscan_min_samples: int   = 3

    # ── Graph ─────────────────────────────────────────────────────────────────
    betweenness_approx_k: int = 200     # use approximate betweenness for scale
    pagerank_alpha:       float = 0.85

    # ── Output ────────────────────────────────────────────────────────────────
    flag_threshold:    float = 0.55     # nodes above this score are flagged
    top_n_suspicious:  int   = 30
    top_features_n:    int   = 3        # features per explanation

    # ── Features used by modeling layer (ordered, must match extraction) ──────
    model_features: List[str] = field(default_factory=lambda: [
        "out_degree",
        "in_degree",
        "degree_ratio",           # out/in, signals aggressive following
        "clustering_coeff",
        "pagerank",
        "betweenness",
        "inter_event_variance",
        "burstiness",
        "activity_rate_1h",
        "activity_rate_24h",
        "total_events",
        "repetition_ratio",
        "target_diversity",
        "neighbor_overlap",
    ])

    def validate(self) -> None:
        total = sum(self.scoring_weights.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"scoring_weights must sum to 1.0, got {total:.4f}"
            )
