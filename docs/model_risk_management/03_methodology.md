# 3. Methodology

## 3.1 Model Architecture

The system uses two complementary models:

| Model | Purpose | Framework |
|-------|---------|-----------|
| **XGBoost Binary Classifier** | Predict probability of refinancing distress at maturity | xgboost 2.x, scikit-learn |
| **Cox Proportional Hazards** | Estimate time-to-distress (survival analysis) | lifelines 0.28+ |

The XGBoost model is the primary scoring engine. The Cox PH model provides a complementary temporal dimension — "when" in addition to "whether."

## 3.2 Feature Set (24 Features)

### Structural (One-Hot Encoded)
| Feature | Categories | Rationale |
|---------|-----------|-----------|
| property_type | office, retail, industrial, multifamily, hotel | Different sectors have distinct cap rate dynamics and distress profiles |
| sponsor_credit_tier | A, B, C | Proxy for sponsor's ability to inject equity at refi |
| amortization_type | interest_only, amortizing | IO loans have no paydown → higher maturity LTV |
| balloon_flag | True, False | Balloon maturity creates hard refinancing deadline |

### Target-Encoded
| Feature | Method | Rationale |
|---------|--------|-----------|
| metro_encoded | Mean distress rate per metro (train-only) | Captures geographic risk concentration without high-cardinality one-hot |

### Origination Characteristics
| Feature | Description |
|---------|-------------|
| ltv_at_origination | Leverage at close (higher = riskier from inception) |
| dscr_at_origination | Cash flow coverage at close |
| note_rate | Coupon rate (locked at origination) |
| log_original_balance | Log-transformed loan size |
| occupancy_pct | Property occupancy at origination |

### Current State at T_obs (24 Months Before Maturity)
| Feature | Description |
|---------|-------------|
| current_ltv_at_Tobs | Market-implied LTV using cap_rate_at_Tobs |
| current_dscr_at_Tobs | DSCR at projected refi rate (T_obs market) |
| rate_gap_bps_at_Tobs | Refi rate minus note rate (in basis points) |
| months_since_origination_at_Tobs | Loan age at observation |

### Market Context at T_obs
| Feature | Description |
|---------|-------------|
| treasury_10y_at_Tobs | Treasury 10Y rate at observation time |
| cap_rate_at_Tobs | National cap rate for property type at T_obs |
| cap_rate_delta_since_origination_at_Tobs | Cap rate change from origination to T_obs |

## 3.3 Hyperparameter Selection

Hyperparameters selected via Optuna Bayesian optimization (20 trials, TPE sampler, maximizing validation AUC):

| Parameter | Value | Search Range | Rationale |
|-----------|-------|-------------|-----------|
| max_depth | 3 | [3, 6] | Shallow trees prevent memorizing shock realizations |
| learning_rate | 0.0144 | [0.01, 0.3] log-uniform | Low rate with more estimators for stable convergence |
| n_estimators | 205 | [100, 500] | Sufficient capacity without overfitting |
| min_child_weight | 16 | [5, 30] | Requires 16+ samples per leaf; prevents noisy splits |
| subsample | 0.606 | [0.6, 1.0] | Row subsampling reduces variance |
| colsample_bytree | 0.833 | [0.6, 1.0] | Feature subsampling decorrelates trees |
| reg_alpha (L1) | 0.291 | [0.01, 5.0] log-uniform | Sparsity-inducing regularization |
| reg_lambda (L2) | 0.334 | [0.01, 5.0] log-uniform | Ridge regularization prevents large weights |
| scale_pos_weight | 1.30 | (computed) | Corrects for 43.4% positive class (mild imbalance) |

## 3.4 Class Imbalance Handling

Training label distribution: 40.7% distressed (positive), 59.3% healthy (negative). This is a moderate imbalance addressed via `scale_pos_weight = n_negative / n_positive = 1.30`. No oversampling or undersampling applied — the imbalance is mild enough for direct weighting.

## 3.5 Architecture Choice Rationale

| Choice | Rationale |
|--------|-----------|
| XGBoost over logistic regression | Captures nonlinear interactions (e.g., high LTV compounds with IO amortization and secondary metro) |
| XGBoost over deep learning | Tabular data; interpretable via SHAP; fast training (minutes); no benefit from architecture complexity on 24 features |
| Shallow depth (3) + strong regularization | Intentional constraint on synthetic data with moderate stochastic signal; prevents overfitting shock realizations |
| Cox PH as complement | Provides time-to-event dimension; concordance index is more informative than AUC for loan prioritization |
| SHAP TreeExplainer | Exact Shapley values (not approximations); additive guarantee; enables per-loan human-readable explanations |

## 3.6 Time-Based Split (Anti-Leakage)

| Split | Origination Year | Loans | Label Rate |
|-------|-----------------|-------|-----------|
| Train | ≤ 2018 | 1,739 | 40.7% |
| Validation | 2019 | 244 | 55.7% |
| Test | ≥ 2020 | 88 | 68.2% |

The increasing distress rate across splits reflects economic reality: later vintages mature into a higher-rate, wider-cap-rate environment. This is signal, not leakage — the model must generalize from benign-era training to stressed-era evaluation.

No random shuffling. No cross-validation (which would violate temporal ordering). The split is deterministic and reproducible.
