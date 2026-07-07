# 6. Ongoing Monitoring Plan

## 6.1 Performance Monitoring

### Metrics Tracked

| Metric | Frequency | Alert Threshold | Rationale |
|--------|-----------|-----------------|-----------|
| **Population Stability Index (PSI)** on top-5 features | Monthly | PSI > 0.25 | Detects input data drift that may degrade predictions |
| **Kolmogorov-Smirnov (KS)** on predicted PD distribution | Monthly | KS > 0.10 vs. prior month | Detects output distribution shift |
| **Rolling 6-month AUC** on newly-labeled data | Quarterly | AUC < 0.75 | Detects concept drift (relationship between features and outcomes has changed) |
| **SHAP concentration ratio** (max single-feature % of total |SHAP|) | Quarterly | Any feature > 40% | Detects model over-reliance on a single variable |
| **Calibration deviation** (mean PD vs observed distress rate by decile) | Quarterly | Any decile deviates > 15pp | Detects probability calibration degradation |

### Top-5 Features for PSI Monitoring
1. `cap_rate_at_Tobs`
2. `rate_gap_bps_at_Tobs`
3. `current_dscr_at_Tobs`
4. `current_ltv_at_Tobs`
5. `treasury_10y_at_Tobs`

## 6.2 Retraining Triggers

| Trigger | Action | Timeline |
|---------|--------|----------|
| PSI > 0.25 on any top-5 feature | Investigate data source; retrain if structural shift confirmed | Within 30 days of detection |
| Rolling AUC drops below 0.75 | Mandatory retraining and revalidation | Within 60 days |
| Material change to lending policy | Assess whether model assumptions still hold; retrain if needed | Within 90 days of policy change |
| New property type enters portfolio | Retrain with expanded feature categories | Before scoring new type |
| Quarterly scheduled retraining | Default cadence regardless of triggers | Every 90 days |

## 6.3 Retraining Procedure

1. Ingest updated market data (FRED rates, cap rate benchmarks)
2. Re-run `build_training_frame()` with latest Gold data
3. Execute Optuna HPO (20 trials minimum)
4. Compare new model's test AUC to incumbent's; require ≥ 95% of incumbent performance
5. Review SHAP feature importance for unexpected rank changes
6. Log new model to MLflow with version increment
7. Promote to Staging alias; production switchover pending validation signoff
8. Archive previous version (do not delete)

## 6.4 Governance

| Role | Responsibility | Placeholder |
|------|---------------|-------------|
| **Model Owner** | Day-to-day performance monitoring, retraining execution, first-line issue resolution | [CRE Credit Risk Analytics Lead] |
| **Independent Validator** | Annual full validation, methodology review, challenge function | [Model Validation / Quantitative Risk] |
| **Business Sponsor** | Defines model purpose, approves use-case boundaries, escalation authority | [Head of CRE Lending / Chief Credit Officer] |

### Review Cadence
- **Monthly**: PSI and KS automated checks (model owner)
- **Quarterly**: Performance review with rolling AUC, retraining if triggered (model owner + validator)
- **Annually**: Full independent validation including methodology review, backtesting, and benchmark comparison (independent validator)

### Escalation Path
Model owner → Model validation → Risk committee (for material issues: AUC below 0.70, or model used outside approved scope)

## 6.5 Documentation Maintenance

This MRM documentation is updated upon:
- Any model retraining (update Section 4 metrics)
- Any methodology change (update Section 3)
- Any new limitation discovered (update Section 5)
- Annual review (full document refresh)

Version history tracked in git. All changes require review before merge to main branch.
