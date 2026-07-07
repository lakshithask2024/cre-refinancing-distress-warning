# 5. Limitations and Conservative Use Guidance

## 5.1 Known Limitations

| # | Limitation | Impact | Mitigation |
|---|-----------|--------|-----------|
| 1 | **Synthetic loan portfolio** | High | Distributions calibrated to CBRE/MBA industry averages, but tail behavior and correlation structures may differ from real CMBS tapes. Full re-validation required before production deployment on real data. |
| 2 | **Deterministic property value model** | Medium | NOI / cap_rate approach ignores comparable sales adjustments, submarket-specific vacancy trends, physical asset quality, and in-place lease rollover schedules. |
| 3 | **Small test set (88 loans)** | Medium | Limits statistical power on subgroup analysis (e.g., cannot reliably assess performance on hotel-only or specific metro subsets). Confidence intervals on AUC are wide. |
| 4 | **Historical market data cutoff** | Medium | Trained on market conditions through 2025-06. Performance on post-2025 vintages or novel market regimes (e.g., deflation, negative rates) is untested. |
| 5 | **No borrower-side data** | Medium | No guarantor financial strength, cross-collateralization structure, or sponsor track record. Model captures property economics only. A loan with a strong sponsor may refinance despite marginal metrics. |
| 6 | **No workout / recovery data** | Medium | Model predicts refinancing distress, not loss-given-default or recovery timing. Cannot estimate $ loss or optimal disposition strategy. |
| 7 | **Idiosyncratic shocks are synthetic** | Low-Medium | The stochastic noise (NOI volatility, tenant loss events, surprise defaults) is calibrated to plausible magnitudes but not to observed historical volatility of real properties. |
| 8 | **Metro encoding learnability** | Low | Target-encoded metro variable captures geographic risk concentration but may reflect data artifacts in a synthetic portfolio rather than true submarket fundamentals. |

## 5.2 Conservative Use Guidance

### Required Human Review
Human credit review is REQUIRED before any adverse action on a loan flagged by this model. The model provides a priority ranking for review — it does not make the decision.

### Recommended PD Thresholds

| Threshold | Action |
|-----------|--------|
| PD ≥ 0.70 | High priority review — assign to senior workout officer within 30 days |
| PD 0.50–0.70 | Medium priority — include in quarterly portfolio review |
| PD < 0.50 | Standard monitoring — no immediate action required |

The 0.70 threshold (not 0.50) is recommended for high-priority review to control false-positive burden on the workout team.

### Prohibited Uses
- Do NOT use as sole input to regulatory capital calculations (CECL / IFRS 9)
- Do NOT use for loan pricing or origination decisions
- Do NOT use for automated foreclosure or note-sale triggers

### Flag for Enhanced Scrutiny
Flag any loan where the SHAP top feature is `metro_encoded` — this indicates the model's prediction relies heavily on geographic classification rather than property-specific economics. Such loans should receive independent credit assessment regardless of the model's PD output.

### Stress Test Interpretation
Stress test results represent conditional scenarios ("if rates rise 200 bps, then..."), not forecasts. Do not interpret stressed PDs as unconditional default probabilities.

## 5.3 Model Limitations vs. Rule-Based System

The relationship between the XGBoost model and the rule-based distress tier system (critical/high/medium/low) is documented in [`docs/analysis/distress_tier_diagnostics.md`](../analysis/distress_tier_diagnostics.md), Section 7. Key principle:

> Rules set the boundaries. ML prioritizes within them. Humans decide.

## 5.4 Reference Documents

- [`docs/analysis/modeling_journey.md`](../analysis/modeling_journey.md) — Full leakage detection and resolution history
- [`docs/analysis/distress_tier_diagnostics.md`](../analysis/distress_tier_diagnostics.md) — Rule-based vs ML usage guidance
