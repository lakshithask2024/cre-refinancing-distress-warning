/*
    Gold Model: loan_distress_history
    ═══════════════════════════════════════════════════════════════════════════════
    Full time series of distress metrics per loan snapshot.
    Each row represents one assessment point for a loan.

    Grain: (loan_id, _feature_computed_at)
    Source: silver.loan_features
    Purpose: Feeds survival model (Milestone 6) and temporal analysis.

    Note: In production with incremental runs, this model appends new snapshots.
    In sandbox (single snapshot), it captures the current state as the first
    history point.
*/

{{
    config(
        materialized='table',
        tags=['gold', 'history']
    )
}}

WITH loan_features AS (
    SELECT *
    FROM {{ source('silver', 'loan_features') }}
),

distress_history AS (
    SELECT
        -- Identifiers
        loan_id,
        property_type,
        metro_area,
        origination_year,

        -- Loan context
        original_balance,
        current_balance,
        note_rate,
        amortization_type,
        maturity_date,

        -- Distress metrics (time-varying)
        current_ltv,
        new_dscr,
        rate_gap,
        rate_gap_bps,
        debt_yield,
        current_cap_rate,
        refinance_rate,
        months_to_maturity,

        -- Distress event flags
        _refi_stressed AS is_distressed,
        CASE WHEN new_dscr < 1.0 THEN 1 ELSE 0 END AS dscr_below_1,
        CASE WHEN current_ltv > 0.80 THEN 1 ELSE 0 END AS ltv_above_80,
        CASE WHEN new_dscr < 1.0 AND current_ltv > 0.80 THEN 1 ELSE 0 END AS dual_trigger,

        -- Distress tier
        CASE
            WHEN new_dscr < 1.0 AND current_ltv > 0.80 THEN 'critical'
            WHEN new_dscr < 1.0 OR current_ltv > 0.80 THEN 'high'
            WHEN new_dscr < 1.25 OR current_ltv > 0.70 THEN 'medium'
            ELSE 'low'
        END AS distress_tier,

        -- Severity scores (continuous, for survival model)
        CASE
            WHEN new_dscr >= 1.5 THEN 0.0
            WHEN new_dscr >= 1.0 THEN (1.5 - new_dscr) / 0.5
            ELSE 1.0 + (1.0 - new_dscr)
        END AS dscr_severity_score,

        CASE
            WHEN current_ltv <= 0.60 THEN 0.0
            WHEN current_ltv <= 0.80 THEN (current_ltv - 0.60) / 0.20
            ELSE 1.0 + (current_ltv - 0.80) / 0.20
        END AS ltv_severity_score,

        -- Snapshot timestamp
        _feature_computed_at AS snapshot_at

    FROM loan_features
)

SELECT * FROM distress_history
