# Power BI Reviewer Quickstart (File-Based)

For reviewers without Databricks access: load pre-exported Parquet/CSV files directly into Power BI Desktop.

---

## Step 1: Generate Export Files

```bash
cd cre-refinancing-distress-warning
python -m src.exports.powerbi_exporter
```

This produces files in `data/exports/powerbi/`:
```
data/exports/powerbi/
├── fact_loan_current.parquet       (and .csv)
├── fact_loan_history.parquet       (and .csv)
├── fact_stress_results.parquet     (and .csv)
├── fact_shap_top_features.parquet  (and .csv)
├── fact_survival.parquet           (and .csv)
├── dim_loan.parquet                (and .csv)
├── dim_scenario.parquet            (and .csv)
├── dim_property_type.parquet       (and .csv)
├── dim_metro.parquet               (and .csv)
└── dim_date.parquet                (and .csv)
```

---

## Step 2: Load into Power BI Desktop

### Option A: Load Parquet files (preferred)

1. Open Power BI Desktop
2. **Get Data** → **Parquet**
3. Navigate to `data/exports/powerbi/`
4. Select one file at a time (e.g., `fact_loan_current.parquet`)
5. Repeat for all 10 tables

### Option B: Load CSV files (fallback)

1. **Get Data** → **Text/CSV**
2. Navigate to `data/exports/powerbi/`
3. Select `fact_loan_current.csv`
4. Ensure delimiter is comma, encoding UTF-8
5. Repeat for all tables

### Option C: Load from folder (bulk)

1. **Get Data** → **Folder**
2. Point to `data/exports/powerbi/`
3. Power BI will list all files — filter to `.parquet` or `.csv`
4. Combine and load

---

## Step 3: Set Up Relationships

After loading all tables, go to **Model View** and create relationships per `01_data_model.md`:

1. `dim_loan[loan_id]` → `fact_loan_current[loan_id]` (1:1)
2. `dim_loan[loan_id]` → `fact_stress_results[loan_id]` (1:*)
3. `dim_scenario[scenario_name]` → `fact_stress_results[scenario_name]` (1:*)
4. `dim_property_type[property_type]` → `dim_loan[property_type]` (1:*)
5. `dim_metro[metro]` → `dim_loan[metro_area]` (1:*)

Mark `dim_date` as the Date Table.

---

## Step 4: Add DAX Measures

Open `02_dax_measures.md` and copy each measure into the Power BI formula bar.

Create a "Measures" table (Enter Data → empty table → rename to "_Measures") to organize them.

---

## Step 5: Build Report Pages

Follow `03_report_pages.md` for page layouts, visual types, and slicer configurations.

---

## Notes

- The file-based approach supports **full Import mode** only (no DirectQuery on flat files)
- For production with live data, switch to the Databricks connection pattern (`04_databricks_connection.md`)
- Refresh by re-running the export pipeline and reopening in Power BI
