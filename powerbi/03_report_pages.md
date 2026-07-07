# Power BI Report Pages — Design Specifications

---

## Page 1: Executive Summary

**Purpose:** Single-screen portfolio health overview for C-suite / investment committee.

### Layout (1280×720)

| Position | Visual | Fields | Notes |
|----------|--------|--------|-------|
| Top row (4 cards) | KPI Cards | Total UPB, % Distressed by UPB, Weighted Avg DSCR, Weighted Avg LTV | Conditional format: DSCR red <1.0, LTV red >80% |
| Middle left | Stacked Column | X: maturity_year+quarter, Y: UPB, Legend: distress_tier | "Maturity Wall" — shows upcoming refinancing volume by risk tier |
| Middle right | Tornado/Bar | Y: scenario_name (sorted by severity_rank), X: Delta Distressed UPB vs Baseline | Stress test impact comparison |
| Bottom | Clustered Bar | Y: metro_area (top 10 by distressed UPB), X: Distressed UPB | Geographic concentration |

### Slicers
- Property Type (dropdown)
- Origination Year (range slider 2015–2022)

---

## Page 2: Portfolio Breakdown

**Purpose:** Segment-level analysis for portfolio managers.

### Layout

| Position | Visual | Fields | Notes |
|----------|--------|--------|-------|
| Main (full width) | Matrix | Rows: metro_area, Columns: property_type, Values: % Distressed by Count | Conditional formatting: background color gradient (green→red) |
| Bottom left | Clustered Column | X: origination_year, Y: Loan Count, Legend: distress_tier | Vintage analysis |
| Bottom right | Donut | Values: Loan Count by sponsor_credit_tier | Sponsor quality mix |

### Slicers
- Amortization Type (IO / Amortizing)
- Maturity Year range

---

## Page 3: Loan Explorer

**Purpose:** Searchable loan table for analysts. Entry point to drill-through.

### Layout

| Position | Visual | Fields | Notes |
|----------|--------|--------|-------|
| Full page | Table | loan_id, property_type, metro_area, distress_tier, current_ltv, new_dscr, rate_gap_bps, current_balance, months_to_maturity | Sortable on all columns |

### Conditional Formatting
- distress_tier: critical=red bg, high=orange, medium=yellow, low=green
- current_ltv: gradient red >90%
- new_dscr: gradient red <1.0

### Drill-Through
- Right-click any loan_id → "Loan Detail" page
- Pass loan_id as filter context

### Slicers
- Property Type, Metro, Distress Tier (all as dropdown)
- Months to Maturity (range -24 to 120)

---

## Page 4: Loan Detail (Drill-Through)

**Purpose:** Deep-dive on a single loan. Hidden from tab navigation.

**Drill-through field:** dim_loan[loan_id]

### Layout

| Position | Visual | Fields | Notes |
|----------|--------|--------|-------|
| Top row | Cards (6) | loan_id, property_type, metro, origination_date, maturity_date, original_balance | Static loan attributes |
| Row 2 left | Gauge | Distress Probability | 0–100%, threshold at 50% and 70% |
| Row 2 right | Card | Median Months To Distress | From fact_survival |
| Middle | Clustered Bar (horizontal) | Y: feature_name, X: shap_value | SHAP top 5 drivers, colored by direction |
| Bottom | Small Multiples Grid | Y: stressed_pd, Category: scenario_name | 8 cards showing PD under each stress scenario |

---

## Page 5: Stress Test Comparison

**Purpose:** Compare scenarios side-by-side. Identify loans newly at risk under stress.

### Layout

| Position | Visual | Fields | Notes |
|----------|--------|--------|-------|
| Top | Slicer | dim_scenario[scenario_name] | Single-select, defaults to "combined_severe" |
| Row 1 | KPI Cards (3) | Stressed % Distressed, Delta % Distressed vs Baseline, Loans New To Distress | Impact summary for selected scenario |
| Middle | Scatter | X: baseline_pd, Y: stressed_pd, Size: current_balance, Color: property_type | Each dot = one loan. Above diagonal = worsened under stress |
| Bottom | Table | loan_id, property_type, baseline_pd, stressed_pd, delta_pd | Filtered to Loans New To Distress (baseline_pd ≤0.5, stressed_pd >0.5) |

### Interactions
- Clicking scenario slicer updates all visuals
- Scatter plot: hovering shows loan_id tooltip with property_type, metro, balance

---

## Cross-Page Interactions

- All pages respect Property Type and Metro slicers (synced)
- Drill-through from Page 3 → Page 4 passes loan_id
- Back button on Page 4 returns to Page 3
- Bookmarks: "Office Focus" (property_type = office), "Near-Term Maturities" (months_to_maturity < 24)
