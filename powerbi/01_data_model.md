# Power BI Data Model Specification

## Star Schema Overview

```
                    dim_date
                       │
           dim_metro   │   dim_property_type
               │       │        │
               └───dim_loan─────┘
                       │
        ┌──────────────┼──────────────────────────┐
        │              │              │            │
  fact_loan_current  fact_loan_history  fact_survival  fact_shap_top_features
                                          │
                              fact_stress_results
                                    │
                              dim_scenario
```

---

## Fact Tables

### fact_loan_current
**Grain:** One row per loan (latest assessment)

| Column | Type | Description |
|--------|------|-------------|
| loan_id | Text | Unique loan identifier (FK → dim_loan) |
| distress_tier | Text | critical / high / medium / low |
| current_ltv | Decimal | Market-implied loan-to-value |
| new_dscr | Decimal | DSCR at projected refinance rate |
| refinance_rate | Decimal | Projected refi rate (decimal) |
| rate_gap_bps | Decimal | Rate gap in basis points |
| debt_yield | Decimal | NOI / balance |
| current_cap_rate | Decimal | Current cap rate (%) |
| current_value | Currency | Implied property value ($) |
| months_to_maturity | Decimal | Months until maturity |
| is_matured | Boolean | True if already past maturity |
| dscr_change | Decimal | new_dscr - dscr_at_origination |
| ltv_change | Decimal | current_ltv - ltv_at_origination |
| current_balance | Currency | Outstanding principal ($) |
| noi_annual | Currency | Net operating income ($) |

### fact_loan_history
**Grain:** One row per loan per snapshot

| Column | Type | Description |
|--------|------|-------------|
| loan_id | Text | FK → dim_loan |
| is_distressed | Integer | 1 = distressed, 0 = healthy |
| distress_tier | Text | Tier at snapshot time |
| current_ltv | Decimal | LTV at snapshot |
| new_dscr | Decimal | DSCR at snapshot |
| rate_gap_bps | Decimal | Rate gap at snapshot |
| dscr_severity_score | Decimal | Continuous severity (0=healthy, >1=severe) |
| ltv_severity_score | Decimal | Continuous severity |
| snapshot_at | DateTime | Snapshot timestamp |

### fact_stress_results
**Grain:** One row per loan × scenario

| Column | Type | Description |
|--------|------|-------------|
| loan_id | Text | FK → dim_loan |
| scenario_name | Text | FK → dim_scenario |
| stressed_pd | Decimal | Probability of distress under scenario |
| stressed_distress_tier | Text | Tier under scenario |
| stressed_ltv | Decimal | LTV under scenario |
| stressed_dscr | Decimal | DSCR under scenario |
| stressed_refinance_gap | Decimal | Rate gap under scenario (bps) |
| baseline_pd | Decimal | Baseline probability (for delta) |
| delta_pd | Decimal | stressed_pd - baseline_pd |
| current_balance | Currency | UPB ($) for weighting |

### fact_shap_top_features
**Grain:** One row per loan × feature (top 5 per loan)

| Column | Type | Description |
|--------|------|-------------|
| loan_id | Text | FK → dim_loan |
| predicted_pd | Decimal | Model prediction for this loan |
| feature_name | Text | Feature contributing to prediction |
| feature_value | Decimal | Feature value for this loan |
| shap_value | Decimal | SHAP contribution (+ = increases risk) |
| rank_in_loan | Integer | 1 = most important, 5 = least |

### fact_survival
**Grain:** One row per loan

| Column | Type | Description |
|--------|------|-------------|
| loan_id | Text | FK → dim_loan |
| predicted_median_months_to_distress | Decimal | Median time to event |
| predicted_survival_prob_12m | Decimal | P(survive 12 months) |
| predicted_survival_prob_24m | Decimal | P(survive 24 months) |
| predicted_survival_prob_36m | Decimal | P(survive 36 months) |

---

## Dimension Tables

### dim_loan
| Column | Type | Description |
|--------|------|-------------|
| loan_id | Text | Primary key |
| property_type | Text | FK → dim_property_type |
| metro_area | Text | FK → dim_metro[metro] |
| sponsor_credit_tier | Text | A / B / C |
| amortization_type | Text | interest_only / amortizing |
| balloon_flag | Text | True / False |
| origination_date | Date | Loan origination |
| maturity_date | Date | Loan maturity |
| origination_year | Integer | Year of origination |
| original_balance | Currency | Balance at origination ($) |
| ltv_at_origination | Decimal | LTV at close |
| dscr_at_origination | Decimal | DSCR at close |
| note_rate | Decimal | Coupon rate |
| loan_purpose | Text | acquisition / refinance |
| loan_term_years | Integer | Original term |

### dim_scenario
| Column | Type | Description |
|--------|------|-------------|
| scenario_name | Text | Primary key |
| description | Text | Human-readable label |
| rate_shock_bps | Integer | Rate delta in bps |
| cap_rate_shock_bps | Integer | Cap rate delta in bps |
| noi_shock_pct | Decimal | NOI multiplier (0 = no shock) |
| property_type_filter | Text | "all" or specific type |
| severity_rank | Integer | 0 = baseline, higher = more severe |

### dim_property_type
| Column | Type | Description |
|--------|------|-------------|
| property_type | Text | Primary key (office, retail, etc.) |
| sector | Text | Display name |
| risk_tier | Text | high / medium-high / medium / low |

### dim_metro
| Column | Type | Description |
|--------|------|-------------|
| metro | Text | Primary key |
| region | Text | Geographic region |

### dim_date
| Column | Type | Description |
|--------|------|-------------|
| date | Date | Primary key (daily, 2015-01-01 to 2028-12-31) |
| year | Integer | Calendar year |
| quarter | Integer | 1-4 |
| quarter_name | Text | Q1, Q2, Q3, Q4 |
| month | Integer | 1-12 |
| month_name | Text | January, February, etc. |
| is_month_end | Boolean | Last day of month |

---

## Relationships

| From | To | Cardinality | Direction |
|------|----|-------------|-----------|
| dim_loan[loan_id] | fact_loan_current[loan_id] | 1:1 | Single |
| dim_loan[loan_id] | fact_loan_history[loan_id] | 1:* | Single |
| dim_loan[loan_id] | fact_stress_results[loan_id] | 1:* | Single |
| dim_loan[loan_id] | fact_shap_top_features[loan_id] | 1:* | Single |
| dim_loan[loan_id] | fact_survival[loan_id] | 1:1 | Single |
| dim_scenario[scenario_name] | fact_stress_results[scenario_name] | 1:* | Single |
| dim_property_type[property_type] | dim_loan[property_type] | 1:* | Single |
| dim_metro[metro] | dim_loan[metro_area] | 1:* | Single |
| dim_date[date] | fact_loan_history[snapshot_at] | 1:* | Single |

**Mark dim_date as the Date Table** in Power BI (Modeling → Mark as Date Table).

---

## Storage Mode

| Environment | Mode | Notes |
|-------------|------|-------|
| File-based (reviewer) | Import | Load from data/exports/powerbi/*.parquet or *.csv |
| Databricks (production) | DirectQuery | Connect to Gold Delta tables via SQL Warehouse |
