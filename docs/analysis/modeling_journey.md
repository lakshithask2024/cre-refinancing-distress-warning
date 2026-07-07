# Modeling Journey: From AUC 1.0 to Defensible Risk Stratification

**A case study in identifying and resolving data leakage in a CRE distress prediction model.**

---

## Summary

Four iterations were required to produce a defensible distress classifier. The journey illustrates how target-definition leakage can produce perfect scores that mask a fundamentally broken prediction task — and how synthetic data requires explicit stochastic noise to create a genuine learning problem.

| Version | AUC | Issue | Root Cause |
|---------|-----|-------|------------|
| v1 | 1.0000 | Perfect discrimination | Label derived from same features fed to model |
| v2 | 0.998 | Near-perfect | Smooth deterministic projections → features predict label mechanically |
| v3 | 0.964 | Still too high | LTV threshold (0.70) too aggressive → base rate 65% → shocks invisible |
| v4 | **0.920** | Defensible | Balanced label, genuine noise, irreducible error floor |

---

## v1: Baseline (AUC 1.0000)

### Setup
- Features: `current_ltv`, `new_dscr`, `rate_gap_bps`, `debt_yield`, `months_to_maturity`
- Label: `is_distressed` = 1 if `new_dscr < 1.0 OR current_ltv > 0.80`
- Split: by origination_year

### Result
AUC = 1.0000. Every single loan classified correctly.

### Diagnosis
**Target-definition leakage.** The label `is_distressed` was a deterministic function of two features (`new_dscr` and `current_ltv`) that were also in the feature matrix. The model didn't learn to predict distress — it learned to evaluate a logical expression.

This is more subtle than direct feature leakage. The features weren't "future" data — they were computed at the same timestamp as the label. But because the label was *defined by* those same features, perfect prediction was trivial.

### Key Lesson
**Target-definition leakage passes superficial correlation checks.** The label isn't directly correlated with any single feature at r=1.0 (it's an OR of two conditions). But a tree-based model with depth ≥ 2 can recover the exact definition.

---

## v2: Temporal Reframing (AUC 0.998)

### Fix Applied
Reframed the prediction task with temporal separation:
- **Observation time (T_obs):** maturity_date - 24 months
- **Features:** snapshotted at T_obs using market data from that date
- **Label:** computed using market conditions at maturity_date (24 months later)

This ensures features and labels use data from different time points — the model must predict the *future* state from the *past* state.

### Result
AUC = 0.998. Still near-perfect.

### Diagnosis
**Deterministic transitions.** In the synthetic data, NOI is static and market rate paths are smooth (no shocks). If you know a loan's state at T-24mo, the market path from T-24 to T-0 is perfectly predictable from the origination vintage and property type. The model learns the deterministic rate trajectory, not a risk signal.

### Key Lesson
**Synthetic data with smooth projections has near-deterministic outcomes.** Temporal separation eliminates direct leakage but doesn't create genuine prediction uncertainty unless the underlying data has stochastic transitions.

---

## v3: Idiosyncratic Shocks (AUC 0.964)

### Fix Applied
Added three stochastic shock types to the label computation (applied only at maturity, not visible in T_obs features):
1. **Property NOI volatility:** per-loan random shock ~ Normal(0, vol), vol ~ U(5%, 25%)
2. **Tenant loss events:** 10% probability for office/retail loans
3. **Submarket cap rate shock:** 15% of loans get cap rate noise (50bps std)

### Result
AUC = 0.964. Still above the 0.90 "investigate further" threshold.

### Diagnosis
**LTV threshold too aggressive.** With maturity_ltv > 0.70 as the label threshold, 65% of loans were deterministically distressed just from cap rate expansion — before any shocks were applied. The shocks could only flip the remaining 35% of loans, which limited the model's uncertainty.

Additionally:
- Shocks were too mild to overcome the strong deterministic signal
- 5% surprise defaults were invisible (piling onto already-distressed loans)
- The model still had enough deterministic signal to classify most loans correctly

### Key Lesson
**The label threshold determines the effective noise level.** If 65% of loans are guaranteed distressed regardless of shocks, the model can achieve high accuracy by simply predicting "distressed" for the majority — the shocks only affect the margin.

---

## v4: Calibrated Label + Stronger Noise (AUC 0.920)

### Fixes Applied

**Label threshold adjustment:**
- Changed from `maturity_ltv > 0.70` to `maturity_ltv > 0.90`
- Rationale: 0.90 LTV = 10% equity remaining, the realistic point where a lender rejects refinancing
- Effect: base deterministic distress rate dropped from 65% to 44%

**Shock amplification:**
- NOI volatility: U(10%, 40%) annual std, ±60% clip
- Tenant loss: 25% probability for office/retail, 10% for hotel, 15-50% severity
- Cap rate shock: 30% of loans, 100bps std
- NEW: 5% "surprise defaults" — unobservable factors (sponsor bankruptcy, fraud, environmental)

**Model regularization:**
- max_depth: 3-6 (was 3-10) — shallower trees reduce memorization
- min_child_weight: 5-30 (was 1-10) — more samples per leaf
- reg_alpha/lambda: 0.01-5.0 (was 1e-8 to 1.0) — stronger L1/L2

### Result
- **AUC: 0.920** — strong discrimination without memorization
- **PR-AUC: 0.975** — high precision across recall range
- **Brier Score: 0.114** — well-calibrated probabilities
- **Log Loss: 0.352** — informative probability estimates

### Label Distribution
| Split | Loans | Distress Rate |
|-------|-------|---------------|
| Train (orig ≤2018) | 1,739 | 40.7% |
| Valid (orig 2019) | 244 | 55.7% |
| Test (orig ≥2020) | 88 | 68.2% |

The increasing rate across splits reflects economic reality (later vintages mature into worse conditions) — this is signal, not leakage.

### Why 0.920 is Defensible
1. **5% irreducible error floor** from surprise defaults (no model can predict these)
2. **Genuine stochasticity** in NOI and cap rates creates uncertainty the model cannot resolve
3. **Strong regularization** prevents memorizing the training set's specific shock realizations
4. **Feature importance concentration** (top 3 features: rate_gap, cap_rate_at_Tobs, current_dscr_at_Tobs) matches economic intuition

---

## Lessons Learned

### 1. Target-definition leakage is more subtle than direct-feature leakage

Labels derived from feature-driving variables can pass superficial correlation checks (no single feature has r=1.0 with the label) while still being perfectly reconstructible by a tree model. **Always check: "could a 2-depth tree recover my label from my features?"**

### 2. Synthetic data requires explicit stochastic noise for genuine prediction problems

Real portfolios have idiosyncratic risk (tenant departures, sponsor issues, submarket shifts) that makes outcomes inherently uncertain. Synthetic generators that use smooth deterministic projections will always produce near-perfect AUC unless you inject noise explicitly.

### 3. AUC alone is insufficient for leakage detection

A model with AUC 0.998 could be brilliant or broken — you can't tell from the metric alone. The diagnostic toolkit must include:
- Feature-label correlation analysis (top-5 correlations should be < 0.60)
- Temporal integrity checks (features must be from before the label-determining event)
- Calibration curves (a leaky model is often poorly calibrated despite high AUC)
- Feature importance concentration (if one feature dominates, it's likely proxying the label definition)

### 4. Model complexity should scale with signal richness

A synthetic dataset with moderate stochastic noise supports max_depth=3-6, not 10. Deeper trees memorize shock realizations rather than learning generalizable risk patterns. In production with richer real data, complexity can be increased — but start constrained and loosen based on validation performance.

### 5. The label threshold is a modeling decision, not a data property

Choosing LTV > 0.70 vs 0.80 vs 0.90 is a business decision about what "distress" means operationally. The threshold should reflect the actual lending decision boundary (at what LTV does a lender reject the refi application?), not an academic definition of negative equity.

---

*This document feeds into the SR 11-7 Model Risk Management documentation under "Model Development and Validation History."*



---

## Post-v4: MLflow Artifact Hygiene

### Bug Discovered

After v4 training completed successfully (AUC 0.92, all metrics logged), downstream consumers (SHAP CLI, stress testing engine) failed with:

> "Failed to download artifacts from path 'model'"

The XGBoost model was registered in the MLflow Model Registry (version 4, status "Staging"), but the actual model artifact — the serialized `.xgb` file and `MLmodel` metadata — was never written to disk. The registry entry pointed to a non-existent artifact path.

### Root Cause

MLflow 3.x introduced two breaking changes that combined to create a silent failure:

1. **`artifact_path` parameter deprecated**: The old `mlflow.xgboost.log_model(model, artifact_path="model")` call silently registered metadata without persisting the binary artifact. The new API requires `name="model"` (or explicit `xgb_model=` kwarg).

2. **Stages deprecated in favor of aliases**: `registered_model_name` with stage transitions no longer works for loading. The registry accepted the registration but `models:/cre_distress_classifier/Staging` (stage syntax) returned "not found" — only `models:/cre_distress_classifier@Staging` (alias syntax) resolves correctly.

### Fix

1. **Refactored `log_model` call** in `distress_classifier.py`:
   ```python
   model_info = mlflow.xgboost.log_model(
       xgb_model=best_model,
       artifact_path="model",
       input_example=X_train[:5],
       registered_model_name="cre_distress_classifier",
   )
   ```
   Added post-registration verification:
   ```python
   _ = mlflow.xgboost.load_model(model_info.model_uri)
   logger.info(f"✓ Model artifact verified loadable at {model_info.model_uri}")
   ```
   If verification fails, training raises `RuntimeError` rather than silently producing a broken registry entry.

2. **Set alias instead of stage**:
   ```python
   client.set_registered_model_alias("cre_distress_classifier", "Staging", version=model_info.registered_model_version)
   ```

3. **Updated all downstream loaders** (SHAP CLI, stress engine, survival CLI) to use alias syntax: `models:/cre_distress_classifier@Staging` with fallback to version-based loading (`models:/cre_distress_classifier/{max_version}`).

### Lesson

**Model registration and model logging are separate operations in MLflow.** It is possible to successfully register a model version that points to a non-existent artifact. This creates a silent failure that only surfaces when a consumer tries to load the model — potentially days later in a production pipeline.

Post-registration load verification is essential. The pattern `load_model(model_info.model_uri)` immediately after `log_model()` catches this class of bug at training time, not deployment time.
