# Security Policy

## Reporting a vulnerability

Do **not** publish a suspected vulnerability that could expose imported datasets, local output, credentials, or host files.

Preferred reporting path:

1. Use **Security → Report a vulnerability** for this repository when GitHub private vulnerability reporting is available.
2. Otherwise email **badbanana@proton.me** with the subject `graph_anomaly_detector security report`.

Include the affected commit/version, input shape needed to reproduce the issue, expected and observed behavior, impact, and any proposed mitigation. Prefer synthetic fixtures over real investigation data.

## Security-relevant scope

Reports are especially useful for issues involving:

- unsafe CSV/JSON/file-path handling;
- malformed input causing unintended local file access or code execution;
- output-path traversal or overwriting files outside the selected output directory;
- unsafe deserialization or dependency behavior;
- adapters leaking imported investigation data;
- CI/test fixtures containing non-sanitized real-world data;
- dependency issues with a meaningful exploit path.

Detection accuracy, threshold selection, and false positives are analytical-quality issues unless they also create a security boundary failure.

## Investigation data

Input snapshots and following-list datasets can contain identifiers and relationship data. Keep real investigation datasets out of public issues and fixtures unless they have been intentionally sanitized for publication. The deterministic synthetic generator should be used for security and regression tests whenever possible.

## Supported state

Report findings against the current default branch or identify the exact historical commit affected. Synthetic benchmark behavior and live investigation accuracy are separate concerns and should be reported separately.

## Disclosure

I aim to acknowledge reproducible reports within seven days. Validation and remediation timing depends on severity and reproducibility; no fixed remediation deadline is promised before triage.

Reporter credit is welcome unless anonymity is requested.
