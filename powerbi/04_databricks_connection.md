# Power BI → Databricks Connection Guide (Production)

## Prerequisites

- Databricks workspace with Unity Catalog enabled
- SQL Warehouse (Serverless or Classic) in **Running** state
- Personal Access Token (PAT) generated for your user
- Gold Delta tables materialized in the `gold` schema
- Power BI Desktop (latest version, November 2023+)

---

## Step 1: Get Connection Details from Databricks

1. Navigate to **SQL Warehouses** in the Databricks sidebar
2. Click your warehouse → **Connection details** tab
3. Copy:
   - **Server Hostname**: e.g., `adb-1234567890.azuredatabricks.net`
   - **HTTP Path**: e.g., `/sql/1.0/warehouses/abc123def456`

---

## Step 2: Connect from Power BI Desktop

1. Open Power BI Desktop → **Get Data** → Search "Databricks"
2. Select **Azure Databricks** (or **Databricks** for AWS/GCP)
3. Enter:
   - **Server Hostname**: (from Step 1)
   - **HTTP Path**: (from Step 1)
4. Authentication: **Personal Access Token**
   - Paste your Databricks PAT
5. Click **Connect**

---

## Step 3: Select Tables

In the Navigator pane, expand your catalog → `gold` schema.

Select these tables:
- `loan_current_state` → rename to `fact_loan_current`
- `loan_distress_history` → rename to `fact_loan_history`
- `stress_test_results` → rename to `fact_stress_results`
- `loan_shap_explanations` → rename to `fact_shap_top_features`
- `loan_survival_predictions` → rename to `fact_survival`

For dimension tables, create them in Power Query or load from a separate reference schema.

---

## Step 4: Choose Data Mode

| Portfolio Size | Recommended Mode | Rationale |
|----------------|-----------------|-----------|
| < 100K loans | **Import** | Fast interactions, full DAX support, scheduled refresh |
| 100K – 1M loans | **Import** with incremental refresh | Reduces refresh time |
| > 1M loans | **DirectQuery** | No data movement, always live, slightly slower interactions |

For DirectQuery:
- Ensure your SQL Warehouse has **Auto-Stop** disabled (or a long idle timeout)
- Create aggregation tables for portfolio-level KPIs to reduce query volume

---

## Step 5: Publish and Schedule Refresh

1. **Publish** to Power BI Service (Workspace → Publish)
2. Configure **Gateway** if on-premises (not needed for Databricks cloud)
3. Set **Scheduled Refresh**:
   - Frequency: Daily (after pipeline runs)
   - Time: 6:00 AM (after overnight batch completes)
4. Configure **Credentials** in dataset settings (same PAT)

---

## Performance Tips

1. **Reduce imported columns**: Only import columns used in visuals (not all 40+ Gold columns)
2. **Pre-aggregate**: Use the `maturity_wall` and `market_distress_index` Gold tables for summary pages
3. **Partition by scenario**: For stress results, consider importing only selected scenarios
4. **Use Databricks SQL caching**: Enable result caching on the SQL Warehouse for repeated queries
5. **Star schema**: Build dimension tables in the Gold schema (dbt) rather than Power Query for DirectQuery compatibility

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "Connection timed out" | Ensure SQL Warehouse is running (not auto-stopped) |
| "Authentication failed" | Regenerate PAT in Databricks → User Settings → Access Tokens |
| "Table not found" | Verify catalog/schema path; Unity Catalog requires 3-part naming |
| Slow DirectQuery | Add aggregation tables or switch to Import mode |
| "Data type mismatch" | Ensure Delta table schema matches expected types (no mixed int/float) |
