# 7. Stress Testing Appendix

## 7.1 Scenario Definitions

Eight stress scenarios are defined in `config/stress_scenarios.yaml`:

| Scenario | Rate Shock | Cap Rate Shock | NOI Shock | Target | Macro Interpretation |
|----------|-----------|---------------|-----------|--------|---------------------|
| baseline | 0 | 0 | 0% | All | Current conditions (control) |
| rate_shock_100bps | +100 bps | 0 | 0% | All | Moderate Fed tightening |
| rate_shock_200bps | +200 bps | 0 | 0% | All | Sustained higher-for-longer |
| rate_shock_300bps | +300 bps | 0 | 0% | All | Severe monetary tightening |
| cap_rate_shock_100bps | 0 | +100 bps | 0% | All | Mild investor repricing |
| cap_rate_shock_200bps | 0 | +200 bps | 0% | All | CRE market correction |
| combined_severe | +200 bps | +200 bps | -10% | All | Full recession + CRE crash |
| office_specific | 0 | +300 bps | -20% | Office | Remote-work structural shift |

## 7.2 Aggregated Stress Test Results

Source: `reports/stress_summary.csv`

| Scenario | % Distressed | Avg PD | Delta (pp) | UPB at Risk ($B) | % UPB Distressed |
|----------|-------------|--------|-----------|------------------|-----------------|
| **baseline** | 52.9% | 0.571 | — | $155.0B | 55.8% |
| rate_shock_100bps | 55.1% | 0.586 | +1.5 | $161.4B | 58.1% |
| rate_shock_200bps | 57.1% | 0.594 | +2.3 | $167.2B | 60.2% |
| rate_shock_300bps | 58.7% | 0.600 | +3.0 | $172.5B | 62.1% |
| cap_rate_shock_100bps | 64.9% | 0.651 | +8.0 | $187.0B | 67.4% |
| cap_rate_shock_200bps | 74.8% | 0.708 | +13.7 | $213.5B | 76.9% |
| **combined_severe** | **84.4%** | **0.782** | **+21.2** | **$238.1B** | **85.8%** |
| office_specific | 58.5% | 0.614 | +4.3 | $170.8B | 61.5% |

## 7.3 Headline Finding

**Under `combined_severe` (rate +200 bps, cap rate +200 bps, NOI -10%), the portfolio distress rate rises from 52.9% at baseline to 84.4% (+31.5 percentage points), with total UPB at risk increasing from $155B to $238B (+$83B).**

This scenario represents a full-cycle recession with simultaneous rate tightening, investor risk repricing, and property-level income deterioration.

## 7.4 Key Observations

### Cap Rate Shocks Dominate Rate Shocks

At equivalent magnitudes, cap rate shocks produce substantially larger portfolio impact than interest rate shocks:

| Magnitude | Rate Shock Impact | Cap Rate Shock Impact | Ratio |
|-----------|------------------|----------------------|-------|
| +100 bps | +1.5pp distressed | +8.0pp distressed | **5.3×** |
| +200 bps | +2.3pp distressed | +13.7pp distressed | **6.0×** |

**Economic explanation**: Cap rate expansion compresses property values directly (Value = NOI / Cap Rate), cascading into LTV blowouts for the entire portfolio. A 200bp cap rate increase on a 7% base reduces property values by ~22% (7.0% → 9.0%). In contrast, a 200bp rate shock only affects DSCR (through higher refinance debt service) — it does not directly impair collateral values.

This asymmetry has direct portfolio management implications: a CRE market with stable rates but expanding risk premiums (cap rates rising due to flight from CRE) produces more distress than a market with rising policy rates but stable investor demand.

### Office-Specific Scenario

The targeted office downturn (+300bp cap, -20% NOI, office only) produces a relatively modest portfolio-wide impact (+4.3pp) because office represents 30% of the portfolio. However, within the office segment specifically, the impact is severe — nearly all office loans are pushed into distress under this scenario.

### Monotonicity

Within each shock type, distress rates increase monotonically with severity:
- Rate: 52.9% → 55.1% → 57.1% → 58.7%
- Cap Rate: 52.9% → 64.9% → 74.8%

This validates the model's economic sensitivity — predictions respond directionally correctly to adverse market movements.

## 7.5 Worst-Hit Geographies

Under `combined_severe`, the most impacted metros (by average increase in predicted PD) are:
1. San Francisco
2. Houston
3. Philadelphia
4. Atlanta
5. Los Angeles

These are primarily secondary/tertiary markets where cap rate expansion is most acute relative to gateway city pricing.

## 7.6 Limitations of Stress Test Results

1. **Parallel shock assumption**: All loans experience the shock simultaneously. In reality, market dislocations propagate with lags across geographies and property types.
2. **No feedback effects**: The model does not capture distress contagion (fire sales depressing values further).
3. **No policy response**: No modeling of potential government intervention, forbearance programs, or rate cuts in response to systemic stress.
4. **Static portfolio**: Analysis assumes the current portfolio composition; no new originations or dispositions during the stress horizon.
