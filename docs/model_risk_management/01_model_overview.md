# 1. Model Overview

## 1.1 Model Purpose

This model rank-orders commercial real estate (CRE) mortgage loans by refinancing distress probability 24 months before maturity. The output enables credit risk teams to prioritize workout efforts, allocate loss reserves, and direct special servicing attention to the loans most likely to fail refinancing.

## 1.2 Intended Users

- CRE credit risk teams (loan-level triage)
- Special servicers (workout prioritization)
- Portfolio managers (aggregate risk reporting)
- Bank examiners and internal audit (via this MRM documentation)

## 1.3 In-Scope Decisions

| Decision | Model Role |
|----------|-----------|
| Loan-level triage | Rank loans for human credit review; highest-PD loans reviewed first |
| Portfolio-level risk aggregation | Aggregate PD-weighted exposure for management reporting |
| Stress test scenario analysis | Re-score portfolio under macro shocks (8 predefined scenarios) |
| Reserve staging | Directional input to CECL staging; not sole determinant |

## 1.4 Out-of-Scope Decisions

The model output must NOT be used as the sole basis for:

- **Loan pricing or origination approval** — the model is trained on maturity risk, not origination risk
- **Individual credit decisions without human review** — model provides a priority ranking, not a binary approve/deny
- **Regulatory capital calculations (CECL / IFRS 9 lifetime PD)** — this model provides directional input only; final PD estimates for capital purposes require independent validation against realized loss data
- **Disposition decisions (foreclosure, note sale)** — these require qualitative assessment of borrower, asset, and market factors beyond model scope

## 1.5 Governance Framework

This documentation follows the structure prescribed by:
- **SR 11-7** (Federal Reserve Board, April 2011): Guidance on Model Risk Management
- **OCC 2011-12** (Comptroller of the Currency): Supervisory Guidance on Model Risk Management

The model lifecycle is tracked in MLflow (experiment: `cre_distress`, registry: `cre_distress_classifier`). All training runs, hyperparameters, metrics, and artifacts are versioned and auditable.

## 1.6 Model Inventory

| Field | Value |
|-------|-------|
| Model Name | CRE Refinancing Distress Classifier |
| Registry Name | `cre_distress_classifier` |
| Current Version | v4 (MLflow) |
| Model Type | XGBoost Binary Classifier + Cox PH Survival |
| Development Date | July 2026 |
| Last Validation | July 2026 (initial) |
| Next Scheduled Review | October 2026 (quarterly) |
| Materiality | High (credit risk triage) |
| Model Tier | Tier 2 (directional input, not sole determinant) |
