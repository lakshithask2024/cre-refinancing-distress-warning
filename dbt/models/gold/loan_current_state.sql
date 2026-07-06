/*
    Gold Model: loan_current_state
    ═══════════════════════════════════════════════════════════════════════════════
    One row per loan representing the most recent distress assessment.

    Grain: loan_id (unique)
    Source: silver.loan_features
    Purpose: Primary consumption table for distress dashboard and XGBoost model input.
*/

WITH loan_features AS (
    SELECT *
    FROM {{ source('silver', 'loan_features') }}
),

ranked AS (
    SELECT
        *,
        ROW_NUMBER() OVER (
            PARTITION BY loan_id
            ORDER BY _feature_computed_at DESC
        ) AS _row_rank
    FROM loan_features
),

current_state AS (
    SELECT
        -- Identifiers
        loan_id,
        deal_name,
        property_id,

        -- Loan terms
        origination_date,
        maturity_date,
        origination_year,
        original_balance,
        current_balance,
        note_rate,
        amortization_type,
        balloon_flag,
        loan_term_years,
        loan_purpose,

        -- Property
        property_type,
        metro_area,
        submarket,
        occupancy_pct,
        noi_annual,
        property_value_at_origination,

        -- Origination underwriting
        ltv_at_origination,
        dscr_at_origination,
        sponsor_credit_tier,

        -- Current market-derived metrics
        current_cap_rate,
        current_value,
        current_ltv,
        refinance_rate,
        refinance_rate_pct,
        rate_gap,
        rate_gap_bps,
        new_dscr,
        annual_debt_service_refi,
        debt_yield,
        debt_yield_pct,

        -- Time
        months_to_maturity,
        is_matured,

        -- Distress flags
        _refi_stressed,
        _rate_gap_positive,

        -- Distress tier classification
        CASE
            WHEN new_dscr < 1.0 AND current_ltv > 0.80 THEN 'critical'
            WHEN new_dscr < 1.0 OR current_ltv > 0.80 THEN 'high'
            WHEN new_dscr < 1.25 OR current_ltv > 0.70 THEN 'medium'
            ELSE 'low'
        END AS distress_tier,

        -- DSCR change from origination
        new_dscr - dscr_at_origination AS dscr_change,

        -- LTV change from origination
        current_ltv - ltv_at_origination AS ltv_change,

        -- Processing metadata
        _feature_computed_at

    FROM ranked
    WHERE _row_rank = 1
)

SELECT * FROM current_state
