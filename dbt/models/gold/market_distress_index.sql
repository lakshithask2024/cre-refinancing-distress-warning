/*
    Gold Model: market_distress_index
    ═══════════════════════════════════════════════════════════════════════════════
    Aggregated distress metrics by metro × property_type × vintage_year.
    Provides market-level view of refinancing stress concentration.

    Grain: (metro_area, property_type, origination_year)
    Source: silver.loan_features
    Purpose: Market-level heatmap, concentration risk analysis, dashboard filters.
*/

WITH loan_features AS (
    SELECT *
    FROM {{ source('silver', 'loan_features') }}
),

market_agg AS (
    SELECT
        -- Dimensions
        metro_area,
        property_type,
        origination_year,

        -- Volume metrics
        COUNT(*) AS loan_count,
        SUM(current_balance) AS total_upb,

        -- Distress rate
        SUM(CASE WHEN _refi_stressed = 1 OR _refi_stressed = TRUE THEN 1 ELSE 0 END) AS distressed_loan_count,
        CAST(SUM(CASE WHEN _refi_stressed = 1 OR _refi_stressed = TRUE THEN 1 ELSE 0 END) AS FLOAT)
            / NULLIF(COUNT(*), 0) AS pct_distressed_loans,

        -- UPB at risk (balance of distressed loans)
        SUM(CASE WHEN _refi_stressed = 1 OR _refi_stressed = TRUE THEN current_balance ELSE 0 END) AS total_upb_at_risk,
        CAST(SUM(CASE WHEN _refi_stressed = 1 OR _refi_stressed = TRUE THEN current_balance ELSE 0 END) AS FLOAT)
            / NULLIF(SUM(current_balance), 0) AS pct_upb_at_risk,

        -- Refinance gap metrics
        AVG(rate_gap_bps) AS avg_refinance_gap_bps,
        MAX(rate_gap_bps) AS max_refinance_gap_bps,

        -- DSCR metrics (weighted by balance)
        SUM(new_dscr * current_balance) / NULLIF(SUM(current_balance), 0) AS weighted_avg_dscr,
        AVG(new_dscr) AS simple_avg_dscr,
        MIN(new_dscr) AS min_dscr,

        -- LTV metrics
        SUM(current_ltv * current_balance) / NULLIF(SUM(current_balance), 0) AS weighted_avg_ltv,
        AVG(current_ltv) AS simple_avg_ltv,
        MAX(current_ltv) AS max_ltv,

        -- Debt yield
        AVG(debt_yield) AS avg_debt_yield,

        -- Tier distribution
        SUM(CASE WHEN new_dscr < 1.0 AND current_ltv > 0.80 THEN 1 ELSE 0 END) AS critical_count,
        SUM(CASE WHEN (new_dscr < 1.0 OR current_ltv > 0.80)
                  AND NOT (new_dscr < 1.0 AND current_ltv > 0.80) THEN 1 ELSE 0 END) AS high_count,
        SUM(CASE WHEN (new_dscr < 1.25 OR current_ltv > 0.70)
                  AND NOT (new_dscr < 1.0 OR current_ltv > 0.80) THEN 1 ELSE 0 END) AS medium_count

    FROM loan_features
    GROUP BY metro_area, property_type, origination_year
)

SELECT * FROM market_agg
ORDER BY pct_distressed_loans DESC, total_upb_at_risk DESC
