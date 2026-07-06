"""
Synthetic CMBS Loan Portfolio Generator
========================================

Generates a realistic CMBS-style loan portfolio with configurable size and
property-type-specific statistical distributions. Designed to produce data
suitable for distress modeling and stress testing.

Output: Delta Lake table (Parquet fallback) at data/bronze/loans/

CLI Usage:
    python -m src.ingestion.synthetic_loan_generator --output data/bronze/loans/
    python -m src.ingestion.synthetic_loan_generator --size 5000 --seed 42
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import random
import uuid
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

from src.utils.yaml_compat import load_yaml_file

logger = logging.getLogger(__name__)

# ─── Constants ────────────────────────────────────────────────────────────────

CONFIG_DIR = Path(__file__).resolve().parent.parent.parent / "config"
DEFAULT_OUTPUT = Path(__file__).resolve().parent.parent.parent / "data" / "bronze" / "loans"

# Historical US Treasury 10Y approximate annual averages (for note rate correlation)
TREASURY_10Y_BY_YEAR: dict[int, float] = {
    2015: 2.14,
    2016: 1.84,
    2017: 2.33,
    2018: 2.91,
    2019: 2.14,
    2020: 0.89,
    2021: 1.45,
    2022: 2.95,
    2023: 3.96,
    2024: 4.25,
    2025: 4.10,
}

# 25 US metro areas for loan generation
METROS_25 = [
    "New York",
    "Los Angeles",
    "Chicago",
    "Dallas",
    "San Francisco",
    "Miami",
    "Washington DC",
    "Boston",
    "Atlanta",
    "Seattle",
    "Denver",
    "Houston",
    "Phoenix",
    "Philadelphia",
    "Minneapolis",
    "Austin",
    "Nashville",
    "Charlotte",
    "San Diego",
    "Portland",
    "Tampa",
    "Raleigh",
    "Salt Lake City",
    "Las Vegas",
    "Orlando",
]

# Submarket suffixes per property type
SUBMARKETS: dict[str, list[str]] = {
    "Office": ["CBD", "Midtown", "Suburban", "Urban Fringe", "Tech Corridor"],
    "Retail": ["Power Center", "Neighborhood", "Lifestyle", "Outlet", "Strip"],
    "Multifamily": ["Garden", "Mid-Rise", "High-Rise", "Student", "Suburban"],
    "Industrial": ["Warehouse", "Distribution", "Flex", "Cold Storage", "Last-Mile"],
    "Hotel": ["Full Service", "Select Service", "Extended Stay", "Resort", "Convention"],
}


# ─── Data Classes ─────────────────────────────────────────────────────────────


@dataclass
class LoanRecord:
    """A single CMBS-style loan record."""

    # Loan identifiers
    loan_id: str
    deal_name: str

    # Dates
    origination_date: str  # ISO format
    maturity_date: str  # ISO format
    origination_year: int

    # Loan financials
    original_balance: float
    current_balance: float
    note_rate: float  # decimal (e.g., 0.045 = 4.5%)
    amortization_type: str  # "interest_only" or "amortizing"
    balloon_flag: bool
    loan_term_years: int
    loan_purpose: str  # "acquisition" or "refinance"

    # Property info
    property_id: str
    property_type: str
    metro_area: str
    submarket: str
    occupancy_pct: float  # 0-1
    noi_annual: float
    property_value_at_origination: float

    # Underwriting metrics
    ltv_at_origination: float  # 0-1
    dscr_at_origination: float

    # Sponsor
    sponsor_credit_tier: str  # "A", "B", or "C"

    # Ingestion metadata
    ingested_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    source: str = "synthetic_generator"
    source_version: str = "1.0.0"


# ─── Configuration Loading ────────────────────────────────────────────────────


def load_config(config_path: Path | None = None) -> dict[str, Any]:
    """Load loan generator configuration from YAML."""
    if config_path is None:
        config_path = CONFIG_DIR / "loan_generator.yaml"

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    config = load_yaml_file(config_path)
    return config


# ─── Statistical Helpers ──────────────────────────────────────────────────────


def clamp(value: float, min_val: float, max_val: float) -> float:
    """Clamp a value between min and max."""
    return max(min_val, min(max_val, value))


def sample_normal_clamped(
    rng: random.Random, mean: float, std: float, min_val: float, max_val: float
) -> float:
    """Sample from a normal distribution, clamped to [min_val, max_val]."""
    value = rng.gauss(mean, std)
    return clamp(value, min_val, max_val)


def sample_lognormal_clamped(
    rng: random.Random, log_mean: float, log_std: float, min_val: float, max_val: float
) -> float:
    """Sample from a lognormal distribution, clamped to [min_val, max_val]."""
    value = math.exp(rng.gauss(log_mean, log_std))
    return clamp(value, min_val, max_val)


# ─── Core Generator ──────────────────────────────────────────────────────────


class SyntheticLoanGenerator:
    """
    Generates a synthetic CMBS-style loan portfolio.

    Uses property-type-specific distributions from config/loan_generator.yaml
    and correlates note rates with historical Treasury yields.
    """

    def __init__(self, config: dict[str, Any] | None = None, seed: int | None = None):
        """Initialize the generator with config and random seed."""
        self.config = config or load_config()
        self.seed = seed or self.config["defaults"].get("random_seed", 42)
        self.rng = random.Random(self.seed)

        # Extract config sections
        self.defaults = self.config["defaults"]
        self.property_types = self.config["property_types"]
        self.loan_amount_config = self.config["loan_amount"]
        self.metros = self.config.get("metros", METROS_25[:15])

        # Build property type weights for sampling
        self._property_type_names: list[str] = []
        self._property_type_weights: list[float] = []
        for ptype, pconfig in self.property_types.items():
            self._property_type_names.append(ptype)
            self._property_type_weights.append(pconfig["weight"])

    def generate_portfolio(self, size: int | None = None) -> list[LoanRecord]:
        """Generate a full loan portfolio."""
        portfolio_size = size or self.defaults["portfolio_size"]
        logger.info(f"Generating {portfolio_size} synthetic CMBS loans (seed={self.seed})...")

        loans: list[LoanRecord] = []
        used_loan_ids: set[str] = set()
        deal_counter = 1

        for i in range(portfolio_size):
            loan = self._generate_single_loan(i, used_loan_ids, deal_counter)
            loans.append(loan)
            used_loan_ids.add(loan.loan_id)

            # Group ~20–50 loans per deal
            if (i + 1) % self.rng.randint(20, 50) == 0:
                deal_counter += 1

            if (i + 1) % 1000 == 0:
                logger.info(f"  Generated {i + 1}/{portfolio_size} loans")

        logger.info(f"Portfolio generation complete: {len(loans)} loans")
        return loans

    def _generate_single_loan(
        self, idx: int, used_ids: set[str], deal_num: int
    ) -> LoanRecord:
        """Generate a single loan record with realistic distributions."""
        # Select property type based on configured weights
        property_type = self.rng.choices(
            self._property_type_names,
            weights=self._property_type_weights,
            k=1,
        )[0]
        property_type_lower = property_type.lower()
        ptype_config = self.property_types[property_type]

        # Generate unique loan ID
        loan_id = self._unique_id(used_ids)
        property_id = f"PROP-{''.join(self.rng.choices('0123456789ABCDEF', k=10))}"
        deal_name = f"CMBS-{2015 + (deal_num % 11)}-{deal_num:04d}"

        # Origination timing
        vintage_start = self.defaults["vintage_range"]["start_year"]
        vintage_end = self.defaults["vintage_range"]["end_year"]
        origination_year = self.rng.randint(vintage_start, min(vintage_end, 2022))
        origination_month = self.rng.randint(1, 12)
        origination_day = self.rng.randint(1, 28)
        origination_date = date(origination_year, origination_month, origination_day)

        # Loan term
        term_choices = self.defaults["term_years"]
        term_years = self.rng.choices(term_choices, weights=[0.35, 0.40, 0.25], k=1)[0]
        maturity_date = date(
            origination_year + term_years, origination_month, origination_day
        )

        # Loan amount (lognormal)
        original_balance = sample_lognormal_clamped(
            self.rng,
            self.loan_amount_config["log_mean"],
            self.loan_amount_config["log_std"],
            self.loan_amount_config["min"],
            self.loan_amount_config["max"],
        )
        original_balance = round(original_balance / 100_000) * 100_000  # Round to $100K

        # LTV at origination
        ltv_config = ptype_config["ltv"]
        ltv_at_origination = sample_normal_clamped(
            self.rng, ltv_config["mean"], ltv_config["std"], ltv_config["min"], ltv_config["max"]
        )

        # Property value implied by LTV
        property_value = original_balance / ltv_at_origination

        # Note rate: correlated with Treasury 10Y at origination + property-type spread
        # (Moved before DSCR/NOI so we can back-solve NOI from DSCR)
        treasury_rate = TREASURY_10Y_BY_YEAR.get(origination_year, 2.5)
        spread_config = ptype_config["coupon_spread_bps"]
        spread_bps = sample_normal_clamped(
            self.rng,
            spread_config["mean"],
            spread_config["std"],
            spread_config["mean"] - 2 * spread_config["std"],
            spread_config["mean"] + 2 * spread_config["std"],
        )
        note_rate = (treasury_rate + spread_bps / 100.0) / 100.0
        note_rate = clamp(note_rate, 0.025, 0.09)

        # Amortization type (IO more common for office/hotel, amortizing for industrial)
        # (Moved before DSCR/NOI so debt service formula depends on it)
        io_probability = {
            "Office": 0.65,
            "Retail": 0.45,
            "Multifamily": 0.50,
            "Industrial": 0.30,
            "Hotel": 0.55,
        }.get(property_type, 0.50)
        amortization_type = "interest_only" if self.rng.random() < io_probability else "amortizing"

        # DSCR at origination — sampled from target distribution
        # Then NOI is BACK-SOLVED so that DSCR = NOI / annual_debt_service.
        # This ensures economic consistency: when rates rise at refi,
        # new_dscr = NOI / higher_debt_service < dscr_at_origination.
        dscr_config = ptype_config["dscr"]
        dscr_at_origination = sample_normal_clamped(
            self.rng,
            dscr_config["mean"],
            dscr_config["std"],
            dscr_config["min"],
            dscr_config["max"],
        )

        # Compute annual debt service at origination terms
        if amortization_type == "interest_only":
            annual_debt_service_orig = original_balance * note_rate
        else:
            # Fully amortizing over 30 years (360 monthly payments)
            monthly_rate = note_rate / 12.0
            if monthly_rate > 0:
                pmt_factor = (
                    monthly_rate * (1 + monthly_rate) ** 360
                ) / ((1 + monthly_rate) ** 360 - 1)
                annual_debt_service_orig = original_balance * pmt_factor * 12
            else:
                annual_debt_service_orig = original_balance / 30.0

        # Back-solve NOI from DSCR: NOI = DSCR × annual_debt_service
        noi_annual = dscr_at_origination * annual_debt_service_orig
        noi_annual = round(noi_annual, -3)  # Round to nearest $1K

        # Balloon flag (most CMBS loans are balloon)
        balloon_flag = self.rng.random() < 0.92

        # Current balance (slight paydown for amortizing, none for IO)
        years_elapsed = min(2025 - origination_year, term_years)
        if amortization_type == "amortizing" and years_elapsed > 0:
            # Approximate amortization over 30-year schedule
            monthly_rate = note_rate / 12
            total_payments_30y = 360
            payments_made = years_elapsed * 12
            if monthly_rate > 0:
                factor = (
                    (1 + monthly_rate) ** total_payments_30y
                    - (1 + monthly_rate) ** payments_made
                ) / ((1 + monthly_rate) ** total_payments_30y - 1)
                current_balance = original_balance * max(factor, 0.5)
            else:
                current_balance = original_balance
        else:
            current_balance = original_balance

        current_balance = round(current_balance, -3)

        # Occupancy
        occ_config = ptype_config["occupancy"]
        occupancy_pct = sample_normal_clamped(
            self.rng, occ_config["mean"], occ_config["std"], occ_config["min"], occ_config["max"]
        )

        # Metro and submarket
        metro_area = self.rng.choice(self.metros if len(self.metros) >= 15 else METROS_25)
        submarket_options = SUBMARKETS.get(property_type, ["General"])
        submarket = f"{metro_area} - {self.rng.choice(submarket_options)}"

        # Loan purpose
        loan_purpose = self.rng.choices(
            ["acquisition", "refinance"], weights=[0.55, 0.45], k=1
        )[0]

        # Sponsor credit tier
        sponsor_credit_tier = self.rng.choices(
            ["A", "B", "C"], weights=[0.30, 0.50, 0.20], k=1
        )[0]

        return LoanRecord(
            loan_id=loan_id,
            deal_name=deal_name,
            origination_date=origination_date.isoformat(),
            maturity_date=maturity_date.isoformat(),
            origination_year=origination_year,
            original_balance=original_balance,
            current_balance=current_balance,
            note_rate=round(note_rate, 6),
            amortization_type=amortization_type,
            balloon_flag=balloon_flag,
            loan_term_years=term_years,
            loan_purpose=loan_purpose,
            property_id=property_id,
            property_type=property_type_lower,
            metro_area=metro_area,
            submarket=submarket,
            occupancy_pct=round(occupancy_pct, 4),
            noi_annual=noi_annual,
            property_value_at_origination=round(property_value, -3),
            ltv_at_origination=round(ltv_at_origination, 4),
            dscr_at_origination=round(dscr_at_origination, 4),
            sponsor_credit_tier=sponsor_credit_tier,
        )

    def _unique_id(self, used: set[str]) -> str:
        """Generate a unique loan ID using the seeded RNG for reproducibility."""
        while True:
            hex_chars = "0123456789ABCDEF"
            random_hex = "".join(self.rng.choices(hex_chars, k=12))
            lid = f"LN-{random_hex}"
            if lid not in used:
                return lid


# ─── Writers ──────────────────────────────────────────────────────────────────


def write_delta(loans: list[LoanRecord], output_path: Path) -> None:
    """
    Write loans to Delta Lake format.

    Falls back to Parquet (via pyarrow) or JSON-lines if Delta/Spark unavailable.
    """
    output_path.mkdir(parents=True, exist_ok=True)

    records = [asdict(loan) for loan in loans]

    # Attempt 1: Delta Lake via pyspark
    try:
        from pyspark.sql import SparkSession
        from pyspark.sql.types import (
            StructType, StructField, StringType, DoubleType, IntegerType, BooleanType,
        )
        from delta import configure_spark_with_delta_pip

        # Explicit schema to avoid type inference issues with Delta
        loan_schema = StructType([
            StructField("loan_id", StringType(), False),
            StructField("deal_name", StringType(), True),
            StructField("origination_date", StringType(), True),
            StructField("maturity_date", StringType(), True),
            StructField("origination_year", IntegerType(), True),
            StructField("original_balance", DoubleType(), True),
            StructField("current_balance", DoubleType(), True),
            StructField("note_rate", DoubleType(), True),
            StructField("amortization_type", StringType(), True),
            StructField("balloon_flag", BooleanType(), True),
            StructField("loan_term_years", IntegerType(), True),
            StructField("loan_purpose", StringType(), True),
            StructField("property_id", StringType(), True),
            StructField("property_type", StringType(), True),
            StructField("metro_area", StringType(), True),
            StructField("submarket", StringType(), True),
            StructField("occupancy_pct", DoubleType(), True),
            StructField("noi_annual", DoubleType(), True),
            StructField("property_value_at_origination", DoubleType(), True),
            StructField("ltv_at_origination", DoubleType(), True),
            StructField("dscr_at_origination", DoubleType(), True),
            StructField("sponsor_credit_tier", StringType(), True),
            StructField("ingested_at", StringType(), True),
            StructField("source", StringType(), True),
            StructField("source_version", StringType(), True),
        ])

        # Ensure numeric types are consistent (cast ints to float where schema expects Double)
        for rec in records:
            rec["original_balance"] = float(rec["original_balance"])
            rec["current_balance"] = float(rec["current_balance"])
            rec["noi_annual"] = float(rec["noi_annual"])
            rec["property_value_at_origination"] = float(rec["property_value_at_origination"])
            rec["origination_year"] = int(rec["origination_year"])
            rec["loan_term_years"] = int(rec["loan_term_years"])

        builder = (
            SparkSession.builder.master("local[*]")
            .appName("cre-loan-generator")
            .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
            .config(
                "spark.sql.catalog.spark_catalog",
                "org.apache.spark.sql.delta.catalog.DeltaCatalog",
            )
        )
        spark = configure_spark_with_delta_pip(builder).getOrCreate()
        df = spark.createDataFrame(records, schema=loan_schema)
        (
            df.write.format("delta")
            .mode("overwrite")
            .partitionBy("origination_year")
            .save(str(output_path))
        )
        logger.info(f"Written {len(loans)} loans as Delta Lake to {output_path}")
        spark.stop()
        return
    except ImportError:
        logger.info("PySpark not available, trying pyarrow Parquet fallback...")
    except Exception as e:
        logger.warning(f"Delta write failed ({e}), trying Parquet fallback...")

    # Attempt 2: Parquet via pyarrow
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq

        table = pa.Table.from_pylist(records)
        pq.write_to_dataset(
            table,
            root_path=str(output_path),
            partition_cols=["origination_year"],
        )
        logger.info(f"Written {len(loans)} loans as Parquet to {output_path}")
        return
    except ImportError:
        logger.info("pyarrow not available, using Delta writer fallback...")

    # Attempt 3: Pure-Python Delta writer (produces _delta_log/)
    from src.utils.delta_writer import DeltaWriter

    writer = DeltaWriter(output_path)
    writer.write(records, partition_by="origination_year", mode="overwrite")
    logger.info(f"Written {len(loans)} loans as Delta (pure-Python) to {output_path}")


def _write_jsonl_partitioned(
    records: list[dict[str, Any]], output_path: Path, partition_col: str
) -> None:
    """Write records as partitioned JSON-lines files (universal fallback)."""
    # Group by partition column
    partitions: dict[Any, list[dict[str, Any]]] = {}
    for record in records:
        key = record[partition_col]
        partitions.setdefault(key, []).append(record)

    for partition_value, partition_records in partitions.items():
        partition_dir = output_path / f"{partition_col}={partition_value}"
        partition_dir.mkdir(parents=True, exist_ok=True)
        file_path = partition_dir / "data.jsonl"
        with open(file_path, "w") as f:
            for record in partition_records:
                f.write(json.dumps(record, default=str) + "\n")

    # Write schema file for reference
    schema_path = output_path / "_schema.json"
    if records:
        schema = {k: type(v).__name__ for k, v in records[0].items()}
        with open(schema_path, "w") as f:
            json.dump(schema, f, indent=2)

    # Write metadata
    metadata_path = output_path / "_metadata.json"
    metadata = {
        "format": "jsonl_partitioned",
        "partition_col": partition_col,
        "total_records": len(records),
        "partitions": {str(k): len(v) for k, v in partitions.items()},
        "generated_at": datetime.utcnow().isoformat(),
    }
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(
        f"Written {len(records)} loans as partitioned JSONL to {output_path} "
        f"({len(partitions)} partitions)"
    )


def write_csv(loans: list[LoanRecord], output_path: Path) -> None:
    """Write loans as a single CSV file (for debugging/review)."""
    output_path.mkdir(parents=True, exist_ok=True)
    csv_path = output_path / "loans.csv"
    records = [asdict(loan) for loan in loans]

    if not records:
        return

    fieldnames = list(records[0].keys())
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    logger.info(f"Written {len(records)} loans as CSV to {csv_path}")


# ─── CLI Entry Point ──────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic CMBS loan portfolio for the bronze layer.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output directory for Delta/Parquet/JSONL (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=None,
        help="Number of loans to generate (overrides config; default: from config)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (overrides config; default: from config)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to loan_generator.yaml config file",
    )
    parser.add_argument(
        "--csv",
        action="store_true",
        help="Also write a CSV file alongside the primary output",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    return parser.parse_args()


def main() -> None:
    """Main entry point for the synthetic loan generator."""
    args = parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Load config
    config = load_config(args.config)

    # Override defaults from CLI
    size = args.size or config["defaults"]["portfolio_size"]
    seed = args.seed or config["defaults"]["random_seed"]

    # Generate
    generator = SyntheticLoanGenerator(config=config, seed=seed)
    loans = generator.generate_portfolio(size=size)

    # Write output
    write_delta(loans, args.output)

    if args.csv:
        write_csv(loans, args.output)

    # Summary statistics
    logger.info("─── Portfolio Summary ───")
    logger.info(f"  Total loans: {len(loans)}")
    logger.info(f"  Unique loan IDs: {len(set(l.loan_id for l in loans))}")

    # Property type breakdown
    type_counts: dict[str, int] = {}
    for loan in loans:
        type_counts[loan.property_type] = type_counts.get(loan.property_type, 0) + 1
    for ptype, count in sorted(type_counts.items()):
        logger.info(f"  {ptype}: {count} ({count/len(loans)*100:.1f}%)")

    # Balance stats
    balances = [l.original_balance for l in loans]
    avg_balance = sum(balances) / len(balances)
    logger.info(f"  Avg balance: ${avg_balance:,.0f}")
    logger.info(f"  Total portfolio: ${sum(balances):,.0f}")

    # LTV/DSCR stats
    ltvs = [l.ltv_at_origination for l in loans]
    dscrs = [l.dscr_at_origination for l in loans]
    logger.info(f"  Avg LTV: {sum(ltvs)/len(ltvs):.3f}")
    logger.info(f"  Avg DSCR: {sum(dscrs)/len(dscrs):.3f}")


if __name__ == "__main__":
    main()
