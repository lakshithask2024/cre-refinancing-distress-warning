# Distress Tier Diagnostic Analysis

**Date:** 2026-07-06  
**Portfolio:** 5,000 synthetic CMBS loans (origination 2015–2022)  
**Reference market:** Current rates as of mid-2025 (Treasury 10Y = 4.05%, property-type-specific cap rates)

---

## 1. Tier Assignment Logic

The `distress_tier` column in `gold.loan_current_state` is assigned using a dual-trigger framework based on refinance viability:

```sql
CASE
  WHEN new_dscr < 1.0  AND  current_ltv > 0.80  THEN 'critical'
  WHEN new_dscr < 1.0  OR   current_ltv > 0.80  THEN 'high'
  WHEN new_dscr < 1.25 OR   current_ltv > 0.70  THEN 'medium'
  ELSE                                                'low'
END
```

| Tier | Condition | Interpretation |
|------|-----------|----------------|
| **critical** | DSCR < 1.0 **AND** LTV > 80% | Dual-trigger: cannot service refi debt AND collateral insufficient |
| **high** | DSCR < 1.0 **OR** LTV > 80% | Single-trigger: one of the two refinance gates fails |
| **medium** | DSCR < 1.25 **OR** LTV > 70% | Approaching distress: thin coverage or elevated leverage |
| **low** | Neither condition | Both metrics in acceptable range for refinancing |

These thresholds reflect standard CRE lending underwriting gates. A DSCR of 1.0 is the break-even point for debt service coverage; lenders typically require 1.25x minimum for new originations. An LTV of 80% is the conventional maximum for CMBS underwriting.

---

## 2. Portfolio-Level Tier Distribution

| Tier | Count | Percentage |
|------|-------|------------|
| critical | 1,872 | 37.4% |
| high | 1,061 | 21.2% |
| medium | 801 | 16.0% |
| low | 1,266 | 25.3% |

**Overall distress rate (critical + high): 58.6%**

---

## 3. Underlying Feature Distributions

### 3.1 new_dscr (DSCR at Projected Refinance Rate)

The refinance DSCR is computed as: `NOI / annual_debt_service(current_balance, refinance_rate)`.

| Statistic | Value |
|-----------|-------|
| Mean | 1.111x |
| Median | 1.102x |
| Std Dev | 0.302x |
| Min | 0.377x |
| Max | 2.444x |

**DSCR bucket distribution:**

| Range | Count | % | Assessment |
|-------|-------|---|------------|
| < 0.80 | 827 | 16.5% | Severely impaired — cannot cover even 80% of debt service |
| 0.80 – 1.00 | 1,056 | 21.1% | Below breakeven — shortfall at refinance |
| 1.00 – 1.25 | 1,498 | 30.0% | Thin coverage — vulnerable to any further deterioration |
| 1.25 – 1.50 | 1,059 | 21.2% | Adequate — meets minimum lender threshold |
| ≥ 1.50 | 560 | 11.2% | Comfortable — unlikely refinancing difficulty |

**37.7% of loans have new_dscr < 1.0** — they literally cannot cover debt service at the projected refinance rate.

### 3.2 current_ltv (Market-Implied Loan-to-Value)

Current LTV is computed as: `current_balance / (NOI / current_market_cap_rate)`.

| Statistic | Value |
|-----------|-------|
| Mean | 0.960 |
| Median | 0.884 |
| Std Dev | 0.402 |
| Min | 0.297 |
| Max | 3.064 |

**LTV bucket distribution:**

| Range | Count | % | Assessment |
|-------|-------|---|------------|
| < 0.60 | 958 | 19.2% | Well-collateralized — significant equity cushion |
| 0.60 – 0.70 | 578 | 11.6% | Moderate — standard CRE leverage |
| 0.70 – 0.80 | 541 | 10.8% | Approaching concern — limited equity buffer |
| 0.80 – 1.00 | 962 | 19.2% | Over-leveraged — negative equity building |
| ≥ 1.00 | 1,961 | 39.2% | Underwater — property worth less than debt |

**58.5% of loans have current_ltv > 0.80.** This is the dominant distress driver.

### 3.3 Trigger Overlap (Venn Diagram)

| Condition | Count | % | Tier Assignment |
|-----------|-------|---|-----------------|
| BOTH dscr < 1.0 AND ltv > 0.80 | 1,872 | 37.4% | → critical |
| ONLY dscr < 1.0 (ltv ≤ 0.80) | 11 | 0.2% | → high |
| ONLY ltv > 0.80 (dscr ≥ 1.0) | 1,050 | 21.0% | → high |
| Neither | 2,067 | 41.3% | → medium or low |

**Key insight:** The high critical rate is overwhelmingly driven by LTV deterioration. Nearly all loans with DSCR < 1.0 also have LTV > 0.80 (1,872 out of 1,883). This makes economic sense: the same cap rate expansion that increases LTV (by reducing property value) also reduces NOI coverage ratios. The two triggers are highly correlated, not independent.

---

## 4. Distress Tier by Property Type

| Property Type | Total | Critical | High | Medium | Low | Avg DSCR | Avg LTV |
|---------------|-------|----------|------|--------|-----|----------|---------|
| **office** | 1,478 | 57.6% | 31.5% | 8.1% | 2.8% | 0.958x | 1.250 |
| multifamily | 1,292 | 36.3% | 11.6% | 22.4% | 29.7% | 1.103x | 0.856 |
| retail | 780 | 34.9% | 27.2% | 17.3% | 20.6% | 1.124x | 0.967 |
| hotel | 463 | 29.2% | 41.3% | 15.1% | 14.5% | 1.165x | 1.016 |
| **industrial** | 987 | 14.6% | 4.3% | 19.0% | 62.1% | 1.315x | 0.633 |

**Actual ordering (worst → best by critical %):** office > multifamily > retail > hotel > industrial

### 4.1 Office: Worst-Performing Sector (57.6% Critical)

Office is the most distressed sector, consistent with market consensus post-2022:
- Average LTV of **1.25** means the typical office loan is 25% underwater
- Average DSCR of **0.96x** — the sector as a whole cannot cover refi debt service
- Only **2.8%** of office loans are classified "low risk"

**Economic drivers:**
- Remote/hybrid work reduced office demand → higher vacancy → suppressed NOI growth
- Cap rates expanded from ~6.1% (2021 trough) to 7.6% (current) — a 150bp expansion representing ~25% value decline
- Most office loans are interest-only (65%), so no principal paydown to cushion the LTV blow
- The 200+ bps rate gap (avg origination rate 3.5-4.5% vs. refi rate 6.55%) destroys DSCR

### 4.2 Industrial: Best-Performing Sector (62.1% Low)

Industrial shows the least distress, reflecting strong post-pandemic fundamentals:
- Average LTV of **0.633** — significant equity cushion remains
- Average DSCR of **1.315x** — comfortable debt service coverage
- Cap rates expanded least among all sectors (5.25% current vs. 4.3% in 2021 — still tight relative to history)
- Higher proportion of amortizing loans (70%) means principal paydown further reduces LTV

### 4.3 Surprise Finding: Multifamily (36.3%) Ranks Worse Than Hotel (29.2%)

**Expected:** Hotel should be riskier than multifamily given its operational volatility and COVID-era distress history.

**Actual:** Multifamily has a higher critical rate than hotel. The explanation is mechanical:

1. **Relative cap rate compression at origination:** Multifamily cap rates compressed to historically extreme levels during 2020-2021 (reaching 4.6% nationally). This meant loans originated in that window were underwritten at very high valuations.

2. **Subsequent relative expansion:** Multifamily cap rates have since expanded to 5.55% — a ~95bp move from the 2021 trough. While the absolute cap rate is still lower than hotel, the *relative* expansion is larger as a percentage of the starting point (~21% value decline).

3. **Hotel entered the cycle at wider levels:** Hotel cap rates were already elevated during peak origination years (8.2% in late 2020 due to COVID disruption). Loans originated at those wide spreads are less vulnerable to the current cap rate environment (8.4%) because the cap rate expansion is minimal (~20bp from the 2020 post-shock level).

4. **Portfolio vintage composition:** The synthetic portfolio has a high concentration of 2020-2021 vintage multifamily loans, which were originated at the tightest cap rates and are now experiencing the largest relative value declines.

**In a real portfolio**, hotel would likely show more idiosyncratic distress from operational cash flow volatility. The synthetic generator uses static NOI, which understates hotel risk from RevPAR declines and overstates multifamily risk from value correction alone.

---

## 5. Assessment: Are the Thresholds Appropriate?

### The 37% critical rate is genuine, not threshold-driven.

**Evidence:**
- The DSCR threshold of 1.0x is the mathematical breakeven — below this, the borrower literally cannot pay debt service from property income. This is not conservative; it's definitional.
- The LTV threshold of 0.80 is the standard maximum for CMBS origination. Exceeding it means the borrower has minimal or negative equity — they may rationally choose to default rather than refinance.
- The high critical rate reflects the specific macroeconomic scenario being modeled: a portfolio of low-rate-era originations facing a 200-300bp rate shock and significant cap rate expansion. This is not a hypothetical — it describes actual market conditions facing CRE lenders in 2024-2025.

### If the distress rate seems high, consider:

1. **This is the whole point of the system** — it's an early-warning tool designed to identify distress before maturity. A portfolio with only 10% distress wouldn't need such a tool.

2. **Industry estimates align:** Multiple industry sources (MBA, Trepp, CBRE) estimated 20-40% of office maturities in 2024-2025 faced refinancing difficulties. Our 57.6% office distress rate is at the aggressive end but defensible given our rate assumptions.

3. **The "low" tier (25.3%) represents genuinely safe loans** — industrial/multifamily assets with strong coverage and conservative origination leverage. These would refinance without difficulty.

---

## 6. Implications for Modeling (Milestones 5-6)

This analysis informs several modeling decisions:

1. **Class imbalance:** With 37% in the target class ("critical"), this is NOT a rare-event problem. Standard XGBoost without extreme oversampling should work.

2. **Feature importance:** Expect `current_ltv` and `rate_gap_bps` to dominate SHAP values. The model may learn that LTV is the primary separator (given 99.4% overlap between DSCR<1 and LTV>80).

3. **Property type as interaction:** The model should capture that the same LTV level carries different risk by property type (office at 0.85 LTV is likely distressed; industrial at 0.85 LTV may be fine given stronger NOI growth trajectory).

4. **Survival analysis:** The `months_to_maturity` feature creates a natural time-to-event structure. Loans maturing in 2024-2025 with critical metrics are immediate risks; loans maturing in 2027-2028 may improve if rates decline.

---

*This document will be incorporated into the SR 11-7 Model Risk Management documentation (Milestone 9) under the "Model Assumptions and Limitations" and "Data Description" sections.*



---

## 7. Model vs. Rule: When to Use Each

The system provides two complementary risk assessment approaches. Understanding when to use each is critical for effective portfolio management.

### Rule-Based Tiers (Gold layer: `distress_tier`)

**Characteristics:**
- Deterministic, transparent, fully auditable
- Based on well-understood financial metrics (DSCR, LTV)
- Identical inputs always produce identical outputs
- No model risk — just threshold logic

**Best for:**
- **Policy triggers:** "All critical-tier loans require monthly reporting"
- **Reserve staging under CECL:** Assign higher loss provisions based on tier
- **Regulatory reporting:** Auditors can independently verify tier assignments
- **Hard limits:** "No new lending to properties in critical tier"

**Limitation:** Cannot capture interaction effects. A 0.85 LTV office loan in a secondary metro with an IO structure and C-tier sponsor is far riskier than a 0.85 LTV industrial loan in a gateway market — but both get the same tier under the rule.

### ML Distress Probability (XGBoost classifier output)

**Characteristics:**
- Probabilistic (0.0-1.0 continuous score)
- Captures multivariate interactions and nonlinearities
- Trained to predict forward-looking maturity outcomes
- Requires ongoing validation and governance (SR 11-7)

**Best for:**
- **Rank-ordering within a tier:** "Among 500 critical-tier loans, which 50 should the workout team call first?"
- **Portfolio-level loss estimation:** Aggregate probability-weighted exposure
- **Early warning at the margin:** Identifying medium-tier loans approaching distress
- **SHAP-based explanations:** Understanding *why* a specific loan is high-risk

**Limitation:** Model risk (potential for degradation, drift, or misspecification). Requires MRM documentation, ongoing monitoring, and periodic revalidation.

### Recommended Integrated Usage

```
┌─────────────────────────────────────────────────────────────┐
│  STEP 1: RULE-BASED STAGING                                  │
│  Assign distress_tier (critical/high/medium/low)             │
│  → Hard policy triggers activate                             │
│  → Reserve provisioning by tier                              │
└─────────────────────────────┬───────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 2: ML-BASED PRIORITIZATION (within each tier)          │
│  Score loans with XGBoost distress probability               │
│  → Rank-order within tier for workout attention              │
│  → Surface SHAP explanations for credit officer review       │
└─────────────────────────────┬───────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 3: HUMAN CREDIT REVIEW                                 │
│  Credit committee evaluates top-ranked loans                 │
│  → Incorporates qualitative factors (sponsor relationship,   │
│    market intelligence, renovation plans)                     │
│  → Makes disposition decision (modify, extend, foreclose)    │
└─────────────────────────────────────────────────────────────┘
```

**Key principle:** Rules set the boundaries. ML prioritizes within them. Humans decide.

---

*This integrated framework ensures the model adds value (better prioritization) while maintaining the transparency and auditability required by bank examiners and SR 11-7 governance.*
