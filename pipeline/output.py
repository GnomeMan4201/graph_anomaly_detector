"""
OutputLayer
───────────
Formats and persists pipeline results:

  ranked_nodes.json        scored + explained node list, sorted by fraud_score
  cluster_summary.json     per-cluster stats and member list
  full_results.json        everything in one document (for downstream consumers)

Also prints a console summary table.
"""
from __future__ import annotations

import json
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from config import Config
from pipeline.modeling import ModelResult
from utils.logger import get_logger

log = get_logger(__name__)


class OutputLayer:
    def __init__(self, cfg: Config, output_dir: str = "output") -> None:
        self.cfg        = cfg
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ── public api ────────────────────────────────────────────────────────────

    def write(
        self,
        score_df:     pd.DataFrame,
        explanations: Dict[str, Dict],
        result:       ModelResult,
        feat_df:      pd.DataFrame,
        run_meta:     Optional[Dict] = None,
    ) -> Dict[str, str]:
        """
        Write all output files.  Returns dict of {label: filepath}.
        """
        ranked       = self._build_ranked_list(score_df, explanations)
        cluster_summ = self._build_cluster_summary(result, score_df)
        full         = self._build_full_results(
            ranked, cluster_summ, run_meta or {}
        )

        paths = {
            "ranked_nodes":    self._dump("ranked_nodes.json", ranked),
            "cluster_summary": self._dump("cluster_summary.json", cluster_summ),
            "full_results":    self._dump("full_results.json", full),
        }

        self._print_console_summary(ranked, cluster_summ)
        return paths

    # ── builders ──────────────────────────────────────────────────────────────

    def _build_ranked_list(
        self,
        score_df:     pd.DataFrame,
        explanations: Dict[str, Dict],
    ) -> List[Dict]:
        top_n = self.cfg.top_n_suspicious
        top   = score_df.sort_values("fraud_score", ascending=False).head(top_n)

        rows = []
        for node_id, row in top.iterrows():
            expl = explanations.get(str(node_id), {})
            rows.append({
                "rank":               len(rows) + 1,
                "node_id":            node_id,
                "fraud_score":        round(float(row["fraud_score"]), 4),
                "anomaly_score":      round(float(row["anomaly_score"]), 4),
                "centrality_deviation":round(float(row["centrality_deviation"]), 4),
                "cluster_density_anomaly": round(
                    float(row["cluster_density_anomaly"]), 4
                ),
                "flagged":            bool(row["flagged"]),
                "cluster_id":         expl.get("cluster_id"),
                "top_features":       expl.get("top_features", []),
                "deviation_summary":  expl.get("deviation_summary", ""),
                "cluster_context":    expl.get("cluster_context", {}),
            })
        return rows

    @staticmethod
    def _build_cluster_summary(
        result:   ModelResult,
        score_df: pd.DataFrame,
    ) -> Dict[str, Any]:
        summary: Dict[int, Any] = {}
        for label in sorted(result.cluster_labels.unique()):
            members = (
                result.cluster_labels[result.cluster_labels == label]
                .index.tolist()
            )
            member_scores = score_df.loc[
                score_df.index.intersection(members), "fraud_score"
            ]
            summary[int(label)] = {
                "cluster_id":         int(label),
                "is_noise_cluster":   label == -1,
                "size":               len(members),
                "mean_fraud_score":   round(float(member_scores.mean()), 4)
                                      if not member_scores.empty else 0.0,
                "max_fraud_score":    round(float(member_scores.max()), 4)
                                      if not member_scores.empty else 0.0,
                "flagged_members":    int(
                    (member_scores >= 0.55).sum()
                ) if not member_scores.empty else 0,
                "members":            members[:50],   # truncate for readability
            }
        return summary

    @staticmethod
    def _build_full_results(
        ranked:       List[Dict],
        cluster_summ: Dict,
        run_meta:     Dict,
    ) -> Dict:
        return {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "run_meta":     run_meta,
            "ranked_nodes": ranked,
            "clusters":     cluster_summ,
        }

    # ── IO ────────────────────────────────────────────────────────────────────

    def _dump(self, filename: str, data: Any) -> str:
        path = self.output_dir / filename
        with path.open("w") as f:
            json.dump(data, f, indent=2, default=str)
        log.info("wrote %s", path)
        return str(path)

    # ── console output ────────────────────────────────────────────────────────

    @staticmethod
    def _print_console_summary(
        ranked:       List[Dict],
        cluster_summ: Dict,
    ) -> None:
        width = 80
        sep   = "─" * width

        print(f"\n{'═' * width}")
        print(f"  GRAPH ANOMALY DETECTION — RESULTS SUMMARY")
        print(f"{'═' * width}")

        # Ranked node table
        n_flagged = sum(1 for r in ranked if r["flagged"])
        print(f"\n  TOP SUSPICIOUS NODES  (showing top {len(ranked)}, {n_flagged} flagged)\n")
        print(f"  {'RANK':<5} {'NODE_ID':<18} {'FRAUD':>7} {'IF_SCORE':>9} "
              f"{'CENT_DEV':>9} {'CLUSTER':>8}  {'FLAG':>5}")
        print(f"  {sep}")
        for r in ranked[:20]:
            flag_str = "⚑ YES" if r["flagged"] else "   no"
            cid      = str(r.get("cluster_id", "?"))
            print(
                f"  {r['rank']:<5} {str(r['node_id']):<18} "
                f"{r['fraud_score']:>7.4f} {r['anomaly_score']:>9.4f} "
                f"{r['centrality_deviation']:>9.4f} {cid:>8}  {flag_str:>5}"
            )

        # Cluster summary
        print(f"\n{sep}\n  CLUSTER SUMMARY\n{sep}")
        print(f"  {'CLUSTER':>9} {'SIZE':>6} {'MEAN_FRAUD':>11} "
              f"{'MAX_FRAUD':>10} {'FLAGGED':>8}  {'TYPE':>10}")
        print(f"  {sep}")
        for cid, cs in sorted(cluster_summ.items()):
            kind = "NOISE" if cs["is_noise_cluster"] else "cluster"
            print(
                f"  {cid:>9} {cs['size']:>6} "
                f"{cs['mean_fraud_score']:>11.4f} "
                f"{cs['max_fraud_score']:>10.4f} "
                f"{cs['flagged_members']:>8}  {kind:>10}"
            )

        # Explanation preview for top-3
        print(f"\n{sep}\n  TOP-3 EXPLANATIONS\n{sep}")
        for r in ranked[:3]:
            print(f"\n  Node: {r['node_id']}  fraud_score={r['fraud_score']:.4f}")
            for feat in r["top_features"]:
                print(
                    f"    ● {feat['feature']:<28} "
                    f"z={feat['z_score']:+.2f}  "
                    f"val={feat['node_value']:.3g}  "
                    f"baseline={feat['baseline_mean']:.3g}"
                )
            if r.get("deviation_summary"):
                wrapped = textwrap.fill(
                    r["deviation_summary"], width=74,
                    initial_indent="    ", subsequent_indent="      "
                )
                print(f"\n{wrapped}")

        print(f"\n{'═' * width}\n")
