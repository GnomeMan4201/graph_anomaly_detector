# Graph Anomaly Detector

[![CI](https://github.com/GnomeMan4201/graph_anomaly_detector/actions/workflows/ci.yml/badge.svg)](https://github.com/GnomeMan4201/graph_anomaly_detector/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](#requirements)
[![Local first](https://img.shields.io/badge/analysis-local--first-222222)](#data-and-scope)

**Graph-based anomaly and coordinated-behavior detection pipeline with explainable scoring, clustering, and synthetic ground-truth evaluation.**

---

## Overview

Graph Anomaly Detector turns interaction records into a behavioral graph, extracts structural and social features, models anomalous nodes, scores suspicious activity, and writes explainable JSON results for downstream investigation.

The repository is designed around a reproducible local pipeline rather than a hosted service. It can run against your own JSONL/CSV interaction data or generate a deterministic synthetic dataset with known coordinated clusters so the complete path can be exercised without external data.

```text
ingest interactions
      ↓
build graph
      ↓
extract structural + social features
      ↓
Isolation Forest + DBSCAN modeling
      ↓
combine anomaly / centrality / cluster signals
      ↓
explain suspicious nodes
      ↓
ranked JSON + cluster summaries
```

---

## What it produces

Each run writes three primary artifacts to the selected output directory:

| File | Purpose |
|---|---|
| `ranked_nodes.json` | Suspicious nodes ranked by composite anomaly score with explanations |
| `cluster_summary.json` | Per-cluster size, member, score, and noise-cluster summaries |
| `full_results.json` | Machine-readable result bundle with the run configuration and source metadata |

Synthetic runs also print precision, recall, and F1 against the generator's known bot identities. Those metrics apply only to the synthetic scenario used for that run; they are not claims about arbitrary real-world datasets.

---

## Quick start

### Requirements

- Python 3.10+
- Linux/macOS recommended; Windows should work with a standard Python environment
- no external service or API key required for synthetic runs

### Install

```bash
git clone https://github.com/GnomeMan4201/graph_anomaly_detector.git
cd graph_anomaly_detector
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Deterministic smoke run

```bash
python main.py \
  --synthetic \
  --seed 42 \
  --n-normal 60 \
  --n-clusters 2 \
  --n-days 2 \
  --top-n 10 \
  --save-synthetic output/smoke/input.jsonl \
  --output-dir output/smoke
```

The exported `input.jsonl` is the exact synthetic event set analyzed in that run. `full_results.json` records the seed and detector configuration so the result can be tied back to the conditions that produced it.

CI runs the same deterministic path and verifies the expected artifacts and run metadata.

---

## Run on your own data

```bash
python main.py --input /path/to/events.jsonl
```

CSV is also supported:

```bash
python main.py --input /path/to/events.csv
```

Use separate output directories for separate experiments:

```bash
python main.py \
  --input data/events.jsonl \
  --output-dir output/experiment-01
```

---

## Model controls

```bash
python main.py \
  --synthetic \
  --seed 42 \
  --contamination 0.08 \
  --dbscan-eps 0.40 \
  --dbscan-min-samples 4 \
  --flag-threshold 0.50
```

| Option | Meaning |
|---|---|
| `--seed` | Synthetic generator seed; recorded in run metadata |
| `--contamination` | Expected anomalous fraction used by Isolation Forest |
| `--dbscan-eps` | DBSCAN neighborhood radius |
| `--dbscan-min-samples` | Minimum samples for a dense DBSCAN region |
| `--flag-threshold` | Composite score threshold used to flag a node and count flagged cluster members |
| `--top-n` | Number of suspicious nodes retained in ranked output |

Do not tune these values against the answer key of the same dataset and then report the resulting score as an unbiased evaluation. Use held-out or separately generated data when comparing parameter sets.

---

## Synthetic experiment contract

The synthetic generator is deterministic for a fixed seed and configuration. For any result you want another person to inspect or reproduce, preserve:

- the `--seed`
- generator dimensions (`--n-normal`, `--n-clusters`, `--n-days`)
- detector configuration
- exact exported input JSONL
- complete output directory
- code revision/commit

`--save-synthetic` preserves the same records used by the analysis rather than generating a second dataset for export.

The synthetic generator is useful for regression testing and controlled threshold experiments because the coordinated identities are known in advance. It is deliberately not evidence that the same performance will transfer to real social graphs.

---

## Architecture

```text
main.py
├── data/
│   └── synthetic_generator.py
├── pipeline/
│   ├── ingestion.py
│   ├── graph_builder.py
│   ├── feature_extraction.py
│   ├── social_features.py
│   ├── modeling.py
│   ├── scoring.py
│   ├── explainability.py
│   └── output.py
├── utils/
└── config.py
```

This separation keeps ingestion, feature engineering, modeling, scoring, and explanation independently inspectable rather than burying the decision path in one script.

---

## Explainability

The detector does not emit only a binary label. Ranked nodes include the strongest feature deviations and cluster context used to explain why a node surfaced. Anomaly scores should be treated as triage signals that lead back to inspectable evidence, not as attribution by themselves.

---

## Data and scope

This project detects unusual graph behavior and coordinated patterns. It does **not** establish identity, intent, maliciousness, or common control on its own.

For real investigations, preserve the distinction between:

- observation: measurable interaction or graph structure
- anomaly: deviation from the modeled baseline
- coordination hypothesis: a pattern consistent with organized behavior
- attribution: a separate evidentiary claim requiring additional sources

Input datasets can contain sensitive identifiers or behavioral histories. Keep raw data and generated outputs local unless you have a reason and authorization to share them.

---

## Dependencies

Runtime dependencies are intentionally small and explicit:

- NetworkX
- pandas
- NumPy
- scikit-learn
- SciPy

See `requirements.txt` for version floors.

---

## Author

**GnomeMan4201** — independent security researcher / badBANANA research.
