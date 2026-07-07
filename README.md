# CRE Refinancing Distress Early-Warning System

A medallion-architecture data platform that ingests commercial real estate loan data, models refinancing distress probability, runs interest-rate and cap-rate stress scenarios, and outputs a portfolio-level risk dashboard for CRE lenders and CMBS investors.

---

## Business Problem

Roughly **$1.5 trillion** of commercial real estate debt matures in the United States through 2027. Much of it consists of office loans originated during a period of historically low interest rates and elevated property valuations.

Banks, CMBS bondholders, and special servicers need to identify which loans are unlikely to refinance successfully — **before maturity** — so they can act early: pursue loan modifications, build reserves, or position assets for disposition.

Manual loan-by-loan analysis does not scale to portfolios containing thousands of loans. This system automates the identification and prioritization of refinancing distress risk across large commercial real estate portfolios.

### Model Purpose and Positioning

The XGBoost distress classifier is a **risk stratification tool** — it rank-orders loans by refinancing distress probability 24 months before maturity, enabling credit risk teams to:

- Prioritize workout efforts (focus attention on the top quintile first)
- Stage reserve allocations under CECL (assign higher loss provisions to high-probability loans)
- Route loans to special servicing before maturity triggers a hard default

**This is NOT a standalone default-decision engine.** The model output feeds into human-led credit review, analogous to how banks use PD models for allowance staging — not for automatic loan foreclosure. All disposition decisions remain with the credit committee.

Model governance follows the **SR 11-7 / OCC 2011-12** framework for model risk management. See [`docs/model_risk_management/`](docs/model_risk_management/) for the full MRM documentation package.

### Model Performance (v4)

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **AUC-ROC** | 0.920 | Strong rank-ordering: the model reliably separates high-risk from low-risk loans |
| **PR-AUC** | 0.975 | High precision at all recall levels; few false negatives in the top-ranked cohort |
| **Brier Score** | 0.114 | Well-calibrated probabilities; a 60% prediction corresponds to ~60% observed distress |
| **Log Loss** | 0.352 | Proper scoring rule confirms the model's probability estimates are informative |

*Note: Trained on synthetic data with idiosyncratic shocks. Production deployment on real CMBS tape data would require re-training and re-validation.*

### What This System Does

1. **Ingests** a mix of real market data (Treasury rates, market cap rates) and synthetic CMBS-style loan data into a Delta Lake bronze layer.
2. **Cleans, joins, and enriches** data through a silver layer using PySpark and dbt.
3. **Produces a gold layer** with two consumption models:
   - Loan-level distress probability scores (XGBoost classifier + survival analysis for time-to-distress)
   - Market-level distress indices (aggregate risk by metro and vintage)
4. **Runs stress-test scenarios**: rate shocks (+100/+200/+300 bps) and cap-rate expansion shocks (+100/+200 bps) — recomputes refinance viability under each.
5. **Outputs a ranked portfolio dashboard** flagging loans by distress tier with SHAP explanations.
6. **Ships with SR 11-7 style** model risk management documentation.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          DATA SOURCES                                        │
│  ┌──────────────────┐   ┌──────────────────────┐                           │
│  │ Synthetic Loan   │   │ Market Data APIs     │                           │
│  │ Generator        │   │ (FRED, Cap Rates)    │                           │
│  └────────┬─────────┘   └──────────┬───────────┘                           │
└───────────┼──────────────────────────┼──────────────────────────────────────┘
            │                          │
            ▼                          ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                       BRONZE LAYER (Raw Ingestion)                           │
│  Delta Lake: bronze_loans, bronze_treasury_rates, bronze_cap_rates          │
└────────────────────────────────────┬────────────────────────────────────────┘
                                     │  PySpark ETL
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                       SILVER LAYER (Cleaned & Enriched)                      │
│  Delta Lake: silver_loans, silver_market_rates, silver_loan_features        │
└────────────────────────────────────┬────────────────────────────────────────┘
                                     │  dbt + ML Scoring
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                       GOLD LAYER (Consumption Models)                        │
│  gold_loan_scores │ gold_market_indices │ gold_stress_results               │
└────────────────────────────────────┬────────────────────────────────────────┘
                                     │
              ┌──────────────────────┼──────────────────────┐
              ▼                      ▼                      ▼
┌──────────────────┐    ┌────────────────────┐    ┌─────────────────────┐
│ ML Layer         │    │ Stress Engine      │    │ Power BI Dashboard  │
│ XGBoost+Survival │    │ Rate/Cap Scenarios │    │ Direct or Export    │
└──────────────────┘    └────────────────────┘    └─────────────────────┘
```

---

## Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| Runtime | Python 3.11 | Primary language |
| Compute | PySpark 3.5 | Distributed data processing |
| Storage | Delta Lake 3.x | ACID lakehouse tables |
| Transform | dbt-spark | SQL transformations (Silver → Gold) |
| ML — Classification | XGBoost | Distress probability |
| ML — Survival | lifelines | Time-to-distress |
| ML — Tracking | MLflow | Experiments & model registry |
| Explainability | SHAP | Feature attribution |
| Tuning | Optuna | Hyperparameter optimization |
| Dashboard | Power BI | Portfolio risk visualization |
| Testing | pytest | Unit & integration tests |
| Linting | ruff | Code quality |
| Types | mypy | Static type analysis |

---

## Quickstart

### Prerequisites

- Python 3.11+
- Java 11+ (required by PySpark)
- A FRED API key ([get one here](https://fred.stlouisfed.org/docs/api/api_key.html))

### Installation

```bash
# Clone the repository
git clone <repo-url> cre-distress-warning
cd cre-distress-warning

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# Install all dependencies (core + dev)
pip install -e ".[dev]"

# Configure environment
cp .env.example .env
# Edit .env and add your FRED_API_KEY
```

### Run the Pipeline

```bash
# Full end-to-end pipeline (Bronze → Silver → Gold → Model → Stress → Export)
python scripts/run_pipeline.py

# Or run individual stages:
python scripts/run_pipeline.py --stage bronze
python scripts/run_pipeline.py --stage silver
python scripts/run_pipeline.py --stage gold
python scripts/run_pipeline.py --stage model
python scripts/run_pipeline.py --stage stress
python scripts/run_pipeline.py --stage export
```

### Run Tests

```bash
# All tests
pytest

# Unit tests only
pytest tests/unit/ -m unit

# With coverage report
pytest --cov=src --cov-report=html
```

---

## Training the Model

### Prerequisites

Verify all ML libraries are installed:

```bash
python -c "import xgboost, optuna, mlflow, shap"
```

If any import fails, install with:

```bash
pip install xgboost optuna mlflow shap scikit-learn pandas numpy pyarrow
```

### Training Command

```bash
python -m src.models.train_cli --experiment-name cre_distress
```

Options:
- `--n-trials 20` — Number of Optuna HPO trials (default: 20)
- `--seed 42` — Random seed for reproducibility
- `--gold-path data/gold/loan_distress_history` — Path to Gold table

### Expected Runtime

~3-5 minutes on a laptop for 20 Optuna trials on a 5,000-loan portfolio.

### Inspecting Results

```bash
# Launch the MLflow tracking UI
mlflow ui
# Opens at http://localhost:5000
```

The training run logs:
- All hyperparameters (best + per-trial)
- Test metrics: AUC, PR-AUC, Brier score, log loss
- Artifacts: feature importance plot, calibration curve, confusion matrix
- Registered model: `cre_distress_classifier` in MLflow Model Registry

### Metrics Output

Machine-readable evaluation summary:
```
models/evaluation/distress_classifier_metrics.json
```

---

### Code Quality

```bash
# Lint
ruff check .

# Format
ruff format .

# Type check
mypy src/
```

---

## Repository Structure

```
cre-distress-warning/
├── README.md                         # This file
├── pyproject.toml                    # Project metadata & dependencies
├── .gitignore                        # VCS ignore rules
├── .env.example                      # Environment variable template
│
├── config/                           # Configuration
│   ├── __init__.py
│   ├── settings.py                   # Pydantic settings loader
│   ├── loan_generator.yaml           # Synthetic data parameters
│   └── stress_scenarios.yaml         # Stress scenario definitions
│
├── data/                             # Data lake (gitignored)
│   ├── bronze/                       # Raw ingested data
│   ├── silver/                       # Cleaned & enriched
│   ├── gold/                         # Consumption layer
│   └── exports/
│       └── powerbi/                  # Parquet/CSV for Power BI
│
├── src/                              # Source code
│   ├── __init__.py
│   ├── ingestion/                    # Bronze layer ETL
│   ├── transformations/              # Silver layer PySpark jobs
│   ├── models/                       # ML models (XGBoost, survival)
│   ├── stress_testing/               # Stress scenario engine
│   ├── explainability/               # SHAP explanations
│   ├── exports/                      # Power BI export pipeline
│   └── utils/                        # Shared utilities
│
├── dbt/                              # dbt project (Silver → Gold)
│   ├── dbt_project.yml
│   ├── profiles.yml.example
│   └── models/
│       ├── staging/
│       └── marts/
│
├── notebooks/                        # Exploratory analysis
├── powerbi/                          # Dashboard documentation
│   └── reports/                      # .pbix files
│
├── docs/                             # Documentation
│   ├── model_risk_management/        # SR 11-7 documentation
│   └── screenshots/                  # Dashboard screenshots
│
├── tests/                            # Test suite
│   ├── unit/                         # Fast, isolated tests
│   └── integration/                  # End-to-end pipeline tests
│
└── scripts/                          # CLI entry points
    ├── run_pipeline.py               # Pipeline orchestrator
    ├── run_dbt.sh                    # dbt Gold layer build (with Python fallback)
    └── run_gold_models.py            # Sandbox Gold materializer (SQLite-based)
```

---

## Environment Notes

**dbt models use DuckDB in sandbox environments and Spark in Databricks production — both targets are defined in `dbt/profiles.yml.example`.**

- **Sandbox (`--target sandbox`):** dbt-duckdb reads Silver Parquet files and writes Gold tables locally.
- **Production (`--target production`):** dbt-spark connects to a Databricks SQL Warehouse, reads from the Unity Catalog `silver` schema, and writes Delta tables to the `gold` schema.
- SQL is ANSI-compatible across both adapters; any dialect-specific adjustments are noted in `dbt/profiles.yml.example`.

---

## Configuration

The system is configured through:

1. **Environment variables** (`.env` file) — API keys, paths, Spark settings
2. **YAML files** (`config/`) — Loan generation params, stress scenarios
3. **Pydantic validation** (`config/settings.py`) — Type-safe, fails fast on misconfiguration

See `.env.example` for all available environment variables.

---

## Dashboard

The Power BI dashboard supports two connectivity patterns:

| Pattern | Use Case | Setup |
|---------|----------|-------|
| **Direct Databricks** | Production | Connect Power BI to Databricks SQL endpoint via ODBC |
| **File-based import** | Review / Development | Load Parquet/CSV exports from `data/exports/powerbi/` |

Dashboard documentation lives in `powerbi/` including star schema spec, DAX measures, and page layouts.

---

## Model Risk Management

This system includes SR 11-7 compliant model documentation in `docs/model_risk_management/`, covering:

- Model purpose and methodology
- Assumptions and limitations
- Validation results and back-testing
- Ongoing monitoring plan
- Model inventory card

---

## License

MIT
