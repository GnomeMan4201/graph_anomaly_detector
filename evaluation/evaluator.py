"""
BotnetEvaluator
───────────────
Measures detection quality against researcher-verified ground truth.

Ground truth formats accepted
  txt  — one username per line (flagged_usernames.txt from devto-botnet-hunter)
  csv  — must have a 'username' column and a boolean/int label column
  json — {"username": true/false, ...}  OR  [{"username": ..., "label": ...}]

Metrics produced
  precision / recall / F1 at a configurable score threshold
  ROC-AUC and PR-AUC (threshold-free)
  Cluster purity per detected cluster
  Per-wave breakdown if wave labels are provided

All output is also written to evaluation_report.json in the output directory.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from utils.logger import get_logger

log = get_logger(__name__)


class BotnetEvaluator:
    def __init__(self, output_dir: str = "output") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._labels: Dict[str, bool] = {}

    # ── ground truth loading ──────────────────────────────────────────────────

    def load_labels(self, path: str, label_col: str = "is_bot") -> int:
        """
        Load ground truth labels.  Returns number of positive labels loaded.
        """
        p = Path(path)
        if p.suffix == ".txt":
            self._labels = self._load_txt(p)
        elif p.suffix == ".csv":
            self._labels = self._load_csv_labels(p, label_col)
        elif p.suffix == ".json":
            self._labels = self._load_json_labels(p, label_col)
        else:
            raise ValueError(f"Unsupported label format: {p.suffix}")

        n_pos = sum(self._labels.values())
        log.info(
            "ground truth: %d total labels, %d positive (bots), %d negative",
            len(self._labels), n_pos, len(self._labels) - n_pos,
        )
        return n_pos

    def add_labels(self, bot_usernames: List[str]) -> None:
        """Add known-bot labels from a list (e.g. from synthetic data)."""
        for u in bot_usernames:
            self._labels[str(u)] = True

    # ── evaluation ────────────────────────────────────────────────────────────

    def evaluate(
        self,
        score_df:         pd.DataFrame,
        cluster_labels:   Optional[pd.Series] = None,
        threshold:        float = 0.55,
        wave_col:         Optional[str] = None,
        nodes_df:         Optional[pd.DataFrame] = None,
    ) -> Dict:
        """
        score_df must have a 'fraud_score' column and index = username.
        cluster_labels: pd.Series index=username, values=int cluster ID.

        Returns a results dict and writes evaluation_report.json.
        """
        if not self._labels:
            log.warning("no ground truth labels loaded — skipping evaluation")
            return {}

        # Align labels with scored nodes
        all_nodes = score_df.index.astype(str).tolist()
        y_true = np.array([self._labels.get(u, False) for u in all_nodes], dtype=int)
        y_score= score_df["fraud_score"].reindex(score_df.index).fillna(0.0).values
        y_pred = (y_score >= threshold).astype(int)

        # Only evaluate nodes that appear in ground truth
        labeled_mask = np.array([u in self._labels for u in all_nodes])
        if labeled_mask.sum() == 0:
            log.warning("none of the scored nodes match ground truth labels")
            return {}

        y_true_l  = y_true[labeled_mask]
        y_pred_l  = y_pred[labeled_mask]
        y_score_l = y_score[labeled_mask]

        binary_metrics = self._binary_metrics(y_true_l, y_pred_l, y_score_l)

        # Threshold sweep
        thresholds = np.arange(0.3, 0.95, 0.05)
        threshold_sweep = self._threshold_sweep(y_true_l, y_score_l, thresholds)

        # Cluster purity
        cluster_purity = {}
        if cluster_labels is not None:
            cluster_purity = self._cluster_purity(cluster_labels, all_nodes)

        results = {
            "threshold":         threshold,
            "n_labeled":         int(labeled_mask.sum()),
            "n_positive":        int(y_true_l.sum()),
            "n_flagged":         int(y_pred_l.sum()),
            **binary_metrics,
            "threshold_sweep":   threshold_sweep,
            "cluster_purity":    cluster_purity,
        }

        self._print_report(results)
        self._write_report(results)
        return results

    # ── metric computations ───────────────────────────────────────────────────

    @staticmethod
    def _binary_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_score: np.ndarray,
    ) -> Dict:
        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        fn = int(((y_pred == 0) & (y_true == 1)).sum())
        tn = int(((y_pred == 0) & (y_true == 0)).sum())

        precision = tp / (tp + fp + 1e-9)
        recall    = tp / (tp + fn + 1e-9)
        f1        = 2 * precision * recall / (precision + recall + 1e-9)
        fpr       = fp / (fp + tn + 1e-9)

        # AUC (trapezoidal)
        try:
            from sklearn.metrics import roc_auc_score, average_precision_score
            roc_auc = float(roc_auc_score(y_true, y_score))
            pr_auc  = float(average_precision_score(y_true, y_score))
        except Exception:
            roc_auc = pr_auc = 0.0

        return {
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": round(precision, 4),
            "recall":    round(recall,    4),
            "f1":        round(f1,        4),
            "fpr":       round(fpr,       4),
            "roc_auc":   round(roc_auc,   4),
            "pr_auc":    round(pr_auc,    4),
        }

    @staticmethod
    def _threshold_sweep(
        y_true: np.ndarray,
        y_score: np.ndarray,
        thresholds: np.ndarray,
    ) -> List[Dict]:
        rows = []
        for t in thresholds:
            y_pred = (y_score >= t).astype(int)
            tp = int(((y_pred == 1) & (y_true == 1)).sum())
            fp = int(((y_pred == 1) & (y_true == 0)).sum())
            fn = int(((y_pred == 0) & (y_true == 1)).sum())
            prec = tp / (tp + fp + 1e-9)
            rec  = tp / (tp + fn + 1e-9)
            f1   = 2 * prec * rec / (prec + rec + 1e-9)
            rows.append({
                "threshold": round(float(t), 2),
                "precision": round(prec, 4),
                "recall":    round(rec,  4),
                "f1":        round(f1,   4),
                "flagged":   int(y_pred.sum()),
            })
        return rows

    def _cluster_purity(
        self,
        cluster_labels: pd.Series,
        all_nodes: List[str],
    ) -> Dict[int, Dict]:
        """
        For each detected cluster: what fraction of members are known bots?
        """
        purity: Dict[int, Dict] = {}
        for label in sorted(cluster_labels.unique()):
            members = cluster_labels[cluster_labels == label].index.astype(str).tolist()
            n_bots  = sum(1 for u in members if self._labels.get(u, False))
            n_total = len(members)
            purity[int(label)] = {
                "cluster_id":    int(label),
                "size":          n_total,
                "known_bots":    n_bots,
                "purity":        round(n_bots / (n_total + 1e-9), 4),
                "is_noise":      label == -1,
            }
        return purity

    # ── label loaders ─────────────────────────────────────────────────────────

    @staticmethod
    def _load_txt(p: Path) -> Dict[str, bool]:
        labels = {}
        with p.open() as f:
            for line in f:
                u = line.strip()
                if u and not u.startswith("#"):
                    labels[u] = True
        return labels

    @staticmethod
    def _load_csv_labels(p: Path, label_col: str) -> Dict[str, bool]:
        labels = {}
        with p.open(newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                username = row.get("username") or row.get("Username") or ""
                if not username:
                    continue
                if label_col in row:
                    val = str(row[label_col]).strip().lower()
                    labels[username.strip()] = val in ("1", "true", "yes", "bot")
                else:
                    # No label column — treat presence in file as positive
                    labels[username.strip()] = True
        return labels

    @staticmethod
    def _load_json_labels(p: Path, label_col: str) -> Dict[str, bool]:
        labels = {}
        with p.open() as f:
            data = json.load(f)
        if isinstance(data, dict):
            for u, v in data.items():
                labels[u] = bool(v)
        elif isinstance(data, list):
            for item in data:
                u = item.get("username", "")
                v = item.get(label_col, item.get("is_bot", False))
                if u:
                    labels[u] = bool(v)
        return labels

    # ── output ────────────────────────────────────────────────────────────────

    @staticmethod
    def _print_report(r: Dict) -> None:
        width = 60
        print(f"\n{'─' * width}")
        print("  GROUND TRUTH EVALUATION")
        print(f"{'─' * width}")
        print(f"  Labeled nodes  : {r['n_labeled']}")
        print(f"  Known bots     : {r['n_positive']}")
        print(f"  Flagged        : {r['n_flagged']}")
        print(f"  Threshold      : {r['threshold']:.2f}")
        print(f"  TP={r['tp']}  FP={r['fp']}  FN={r['fn']}  TN={r['tn']}")
        print(f"  Precision      : {r['precision']:.4f}")
        print(f"  Recall         : {r['recall']:.4f}")
        print(f"  F1             : {r['f1']:.4f}")
        print(f"  ROC-AUC        : {r['roc_auc']:.4f}")
        print(f"  PR-AUC         : {r['pr_auc']:.4f}")

        if r.get("cluster_purity"):
            print(f"\n  CLUSTER PURITY")
            print(f"  {'CID':>6} {'SIZE':>6} {'BOTS':>6} {'PURITY':>8}  TYPE")
            for cid, cp in sorted(r["cluster_purity"].items()):
                kind = "NOISE" if cp["is_noise"] else "cluster"
                print(
                    f"  {cid:>6} {cp['size']:>6} {cp['known_bots']:>6} "
                    f"{cp['purity']:>8.4f}  {kind}"
                )

        # Best F1 from threshold sweep
        if r.get("threshold_sweep"):
            best = max(r["threshold_sweep"], key=lambda x: x["f1"])
            print(
                f"\n  Best F1 @ threshold={best['threshold']:.2f}: "
                f"P={best['precision']:.4f}  R={best['recall']:.4f}  "
                f"F1={best['f1']:.4f}  flagged={best['flagged']}"
            )
        print(f"{'─' * width}\n")

    def _write_report(self, r: Dict) -> None:
        path = self.output_dir / "evaluation_report.json"
        with path.open("w") as f:
            json.dump(r, f, indent=2, default=str)
        log.info("evaluation report written to %s", path)
