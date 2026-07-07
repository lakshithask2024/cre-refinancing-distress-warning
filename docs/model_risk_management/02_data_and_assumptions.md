# 2. Data Description and Assumptions

## 2.1 Data Lineage

### Bronze Layer (Raw Ingestion)
- **Synthetic loan portfolio**: 10,000 CMBS-style loans, origination years 2015–2022, 5 property types (office 30%, multifamily 25%, industrial 20%, retail 16%, hotel 10%), 15 metro areas. Generated with seed=42 for reproducibility.
- **Treasury rates**: Daily US Treasury 10Y and 5Y from FRED (2015-01 to 2025-06). Monthly granularity in the processed series.
- **Cap rates**: Quarterly national averages by property type (2015-Q1 to 2025-Q4), calibrated from published CBRE/JLL benchmarks. Metro-level adjustments (±80 bps) based on gateway/secondary market spreads.
- **SOFR**: Monthly rate series (2018-05 to 2025-06).

### Silver Layer (Cleaned & Enriched)
- PySpark transformations: null handling, deduplication, type casting, outlier flagging
- Feature engineering: joins loan data with market data; computes current_value, current_ltv, refinance_rate, rate_gap, new_dscr, debt_yield, months_to_maturity

### Gold Layer (Consumption)
- dbt models (4 tables): loan_current_state, loan_distress_history, market_distress_index, maturity_wall
- Additional Gold outputs: stress_test_results (8 scenarios), loan_survival_predictions, loan_shap_explanations

## 2.2 Key Modeling Assumptions

| # | Assumption | Rationale | Impact if Violated |
|---|-----------|-----------|-------------------|
| 1 | Property value = NOI / cap_rate (income capitalization) | Standard CRE valuation approach for stabilized properties | Overestimates value for distressed assets, underestimates for value-add |
| 2 | Refinancing threshold: LTV ≤ 90% (with borrower cash-in) | Reflects realistic point where lenders reject; below 90% the borrower retains incentive to refinance | Lower threshold (70%) would flag 65% of portfolio as false positives |
| 3 | Distress label: DSCR < 1.0 OR LTV > 0.90 at maturity | Dual-trigger captures both coverage failure and equity insufficiency | Single-trigger would miss compound distress cases |
| 4 | Observation horizon: T_obs = maturity_date - 24 months | Provides actionable lead time for workout preparation | Shorter horizon (12 months) would be more accurate but less actionable |
| 5 | NOI projection: static base + idiosyncratic shocks | No trend growth/decay; shocks (σ = 10-40% annually) add uncertainty | Real NOI trends vary by property type and cycle stage |
| 6 | Feature snapshot: all market features taken at T_obs, not today | Prevents temporal leakage; model predicts future from past | Production scoring must look up market conditions at each loan's specific T_obs |
| 7 | Idiosyncratic shocks: property-specific NOI volatility, tenant loss, submarket cap rate noise, 5% surprise defaults | Creates genuine prediction uncertainty; prevents deterministic label-feature relationship | Without shocks, AUC = 0.998 (leakage artifact on synthetic data) |

## 2.3 Known Data Limitations

| Limitation | Severity | Mitigation |
|-----------|----------|-----------|
| Synthetic portfolio (not real CMBS tape data) | High | Distributions calibrated to industry averages; re-validation required on real data before production |
| Cap rates are national averages with fixed metro adjustments | Medium | Does not capture intra-year submarket movements or property-specific repricing |
| No borrower-specific factors | Medium | No guarantor strength, cross-collateralization, or sponsor track record; model captures property economics only |
| No loan modification / workout / recovery data | Medium | Model predicts refi distress, not loss-given-default or recovery timing |
| Historical market data cutoff at 2025-06 | Medium | Performance on post-2025 market conditions is untested; requires quarterly revalidation |
| No property-level physical condition data | Low | No deferred maintenance, environmental risk, or capex reserve information |

## 2.4 Data Quality Controls

Automated data quality gates validate each layer transition:
- **Bronze → Silver**: row count > 0, null rate < 5% on critical fields, no duplicate loan_ids
- **Silver → Gold**: referential integrity, accepted value ranges, uniqueness tests (dbt)
- **Feature engineering output**: top-5 feature correlations with label checked < 0.90 (leakage guard)
- **Model training input**: no NaN in feature matrix, binary labels only, non-degenerate split distributions
