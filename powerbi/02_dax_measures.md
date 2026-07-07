# DAX Measures Library

All measures organized by folder. Copy-paste directly into Power BI Desktop.

---

## Portfolio Measures

```dax
// ─── Portfolio KPIs ──────────────────────────────────────────────────────────

Total UPB =
SUMX(fact_loan_current, fact_loan_current[current_balance])

Loan Count =
COUNTROWS(fact_loan_current)

Distressed Loan Count =
CALCULATE(
    COUNTROWS(fact_loan_current),
    fact_loan_current[distress_tier] IN {"critical", "high"}
)

Distressed UPB =
CALCULATE(
    SUMX(fact_loan_current, fact_loan_current[current_balance]),
    fact_loan_current[distress_tier] IN {"critical", "high"}
)

% Distressed by Count =
DIVIDE([Distressed Loan Count], [Loan Count], 0)

% Distressed by UPB =
DIVIDE([Distressed UPB], [Total UPB], 0)

Weighted Avg DSCR =
DIVIDE(
    SUMX(fact_loan_current, fact_loan_current[new_dscr] * fact_loan_current[current_balance]),
    [Total UPB],
    0
)

Weighted Avg LTV =
DIVIDE(
    SUMX(fact_loan_current, fact_loan_current[current_ltv] * fact_loan_current[current_balance]),
    [Total UPB],
    0
)

Weighted Avg Refinance Gap =
DIVIDE(
    SUMX(fact_loan_current, fact_loan_current[rate_gap_bps] * fact_loan_current[current_balance]),
    [Total UPB],
    0
)
```

---

## Stress Test Measures

```dax
// ─── Stress Testing (context-aware on dim_scenario) ──────────────────────────

Stressed % Distressed =
DIVIDE(
    CALCULATE(
        COUNTROWS(fact_stress_results),
        fact_stress_results[stressed_pd] > 0.5
    ),
    COUNTROWS(fact_stress_results),
    0
)

Stressed Distressed UPB =
CALCULATE(
    SUMX(fact_stress_results, fact_stress_results[current_balance]),
    fact_stress_results[stressed_pd] > 0.5
)

Baseline % Distressed =
VAR BaselineLoans =
    CALCULATE(
        COUNTROWS(fact_stress_results),
        fact_stress_results[scenario_name] = "baseline"
    )
VAR BaselineDistressed =
    CALCULATE(
        COUNTROWS(fact_stress_results),
        fact_stress_results[scenario_name] = "baseline",
        fact_stress_results[stressed_pd] > 0.5
    )
RETURN DIVIDE(BaselineDistressed, BaselineLoans, 0)

Delta % Distressed vs Baseline =
[Stressed % Distressed] - [Baseline % Distressed]

Delta Distressed UPB vs Baseline =
VAR BaselineUPB =
    CALCULATE(
        SUMX(fact_stress_results, fact_stress_results[current_balance]),
        fact_stress_results[scenario_name] = "baseline",
        fact_stress_results[stressed_pd] > 0.5
    )
RETURN [Stressed Distressed UPB] - BaselineUPB

Loans New To Distress =
CALCULATE(
    COUNTROWS(fact_stress_results),
    fact_stress_results[stressed_pd] > 0.5,
    fact_stress_results[baseline_pd] <= 0.5
)
```

---

## Time-Based Measures

```dax
// ─── Maturity Wall & Time Intelligence ───────────────────────────────────────

UPB Maturing in Selected Quarter =
CALCULATE(
    SUMX(dim_loan, dim_loan[original_balance]),
    FILTER(
        dim_loan,
        YEAR(dim_loan[maturity_date]) = SELECTEDVALUE(dim_date[year])
        && QUARTER(dim_loan[maturity_date]) = SELECTEDVALUE(dim_date[quarter])
    )
)

Distressed UPB Maturing =
CALCULATE(
    SUMX(fact_loan_current, fact_loan_current[current_balance]),
    fact_loan_current[distress_tier] IN {"critical", "high"},
    FILTER(
        dim_loan,
        YEAR(dim_loan[maturity_date]) = SELECTEDVALUE(dim_date[year])
        && QUARTER(dim_loan[maturity_date]) = SELECTEDVALUE(dim_date[quarter])
    )
)

% Maturity Wall Distressed =
DIVIDE([Distressed UPB Maturing], [UPB Maturing in Selected Quarter], 0)

YoY Distress Growth =
VAR CurrentYear = SELECTEDVALUE(dim_date[year])
VAR CurrentDistressed =
    CALCULATE([% Distressed by Count], dim_loan[origination_year] = CurrentYear)
VAR PriorDistressed =
    CALCULATE([% Distressed by Count], dim_loan[origination_year] = CurrentYear - 1)
RETURN CurrentDistressed - PriorDistressed
```

---

## Explainability Measures (Drill-Through)

```dax
// ─── Loan-Level Explainability ───────────────────────────────────────────────

Top SHAP Driver =
VAR SelectedLoan = SELECTEDVALUE(dim_loan[loan_id])
RETURN
CALCULATE(
    FIRSTNONBLANK(fact_shap_top_features[feature_name], 1),
    fact_shap_top_features[loan_id] = SelectedLoan,
    fact_shap_top_features[rank_in_loan] = 1
)

Distress Probability (Selected Loan) =
CALCULATE(
    AVERAGE(fact_loan_current[new_dscr]),
    -- Use actual predicted PD if available from SHAP table
    USERELATIONSHIP(fact_shap_top_features[loan_id], dim_loan[loan_id])
)

Median Months To Distress (Selected Loan) =
VAR SelectedLoan = SELECTEDVALUE(dim_loan[loan_id])
RETURN
CALCULATE(
    VALUES(fact_survival[predicted_median_months_to_distress]),
    fact_survival[loan_id] = SelectedLoan
)

SHAP Feature Contribution =
// Use on bar chart with feature_name on axis, shap_value as value
SUMX(
    fact_shap_top_features,
    fact_shap_top_features[shap_value]
)
```

---

## Formatting Notes

| Measure | Format | Conditional Formatting |
|---------|--------|----------------------|
| % Distressed | Percentage, 1 decimal | Red > 50%, Yellow > 30%, Green < 30% |
| UPB measures | Currency, millions ($M) | — |
| DSCR | Number, 2 decimals | Red < 1.0, Yellow < 1.25, Green ≥ 1.25 |
| LTV | Percentage, 1 decimal | Red > 90%, Yellow > 70%, Green ≤ 70% |
| Delta | Percentage, 1 decimal + sign | Always show + or - |
