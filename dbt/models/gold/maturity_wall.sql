/*
    Gold Model: maturity_wall
    ═══════════════════════════════════════════════════════════════════════════════
    Loans grouped by maturity year and quarter, showing the "wall" of upcoming
    maturities and their distress risk profile.

    Grain: (maturity_year, maturity_quarter, property_type)
    Source: silver.loan_features
    Purpose: Maturity wall visualization, timeline risk assessment.

    Filters: Only loans maturing 2024–2028 (configurable via dbt vars).
*/

WITH loan_features AS (
    SELECT *
    FROM {{ source('silver', 'loan_features') }}
),

with_maturity_period AS (
    SELECT
        *,
        CAST(SUBSTR(maturity_date, 1, 4) AS INTEGER) AS maturity_year,
        CASE
            WHEN CAST(SUBSTR(maturity_date, 6, 2) AS INTEGER) <= 3 THEN 1
            WHEN CAST(SUBSTR(maturity_date, 6, 2) AS INTEGER) <= 6 THEN 2
            WHEN CAST(SUBSTR(maturity_date, 6, 2) AS INTEGER) <= 9 THEN 3
            ELSE 4
        END AS maturity_quarter
    FROM loan_features
    WHERE CAST(SUBSTR(maturity_date, 1, 4) AS INTEGER) >= 2024
      AND CAST(SUBSTR(maturity_date, 1, 4) AS INTEGER) <= 2028
),

maturity_wall AS (
    SELECT
        -- Time dimensions
        maturity_year,
        maturity_quarter,
        maturity_year || '-Q' || maturity_quarter AS maturity_period,
        property_type,

        -- Volume
        COUNT(*) AS count_loans_maturing,
        SUM(current_balance) AS total_upb_maturing,
        AVG(current_balance) AS avg_loan_size,

        -- Distress profile
        SUM(CASE WHEN _refi_stressed = 1 OR _refi_stressed = TRUE THEN 1 ELSE 0 END) AS count_distressed,
        CAST(SUM(CASE WHEN _refi_stressed = 1 OR _refi_stressed = TRUE THEN 1 ELSE 0 END) AS FLOAT)
            / NULLIF(COUNT(*), 0) AS pct_distressed,

        -- Distressed UPB
        SUM(CASE WHEN _refi_stressed = 1 OR _refi_stressed = TRUE THEN current_balance ELSE 0 END) AS upb_distressed,

        -- Risk metrics for maturing cohort
        AVG(new_dscr) AS avg_dscr,
        AVG(current_ltv) AS avg_ltv,
        AVG(rate_gap_bps) AS avg_rate_gap_bps,

        -- Tier breakdown
        SUM(CASE WHEN new_dscr < 1.0 AND current_ltv > 0.80 THEN 1 ELSE 0 END) AS critical_count,
        SUM(CASE WHEN (new_dscr < 1.0 OR current_ltv > 0.80)
                  AND NOT (new_dscr < 1.0 AND current_ltv > 0.80) THEN 1 ELSE 0 END) AS high_count

    FROM with_maturity_period
    GROUP BY maturity_year, maturity_quarter, maturity_period, property_type
)

SELECT * FROM maturity_wall
ORDER BY maturity_year, maturity_quarter, property_type
