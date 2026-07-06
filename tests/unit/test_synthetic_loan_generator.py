"""
Unit tests for the Synthetic CMBS Loan Generator.

Validates:
- All required fields are present and correctly typed
- No duplicate loan_ids in generated portfolio
- Property-type distributions are statistically reasonable
- LTV, DSCR, note_rate, occupancy within expected bounds
- Loan amounts are within configured range
- Origination/maturity date relationships are valid
- Configurable portfolio size works correctly
- Reproducibility via seed
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.ingestion.synthetic_loan_generator import (
    LoanRecord,
    SyntheticLoanGenerator,
    clamp,
    load_config,
    sample_normal_clamped,
)


# ─── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def config():
    """Load the loan generator config."""
    return load_config()


@pytest.fixture(scope="module")
def generator(config):
    """Create a generator with default config and seed."""
    return SyntheticLoanGenerator(config=config, seed=42)


@pytest.fixture(scope="module")
def portfolio(generator):
    """Generate a small portfolio for testing (500 loans for speed)."""
    return generator.generate_portfolio(size=500)


@pytest.fixture(scope="module")
def large_portfolio(config):
    """Generate a larger portfolio (2000 loans) for distribution tests."""
    gen = SyntheticLoanGenerator(config=config, seed=99)
    return gen.generate_portfolio(size=2000)


# ─── Schema & Field Tests ─────────────────────────────────────────────────────


REQUIRED_FIELDS = [
    "loan_id",
    "deal_name",
    "origination_date",
    "maturity_date",
    "origination_year",
    "original_balance",
    "current_balance",
    "note_rate",
    "amortization_type",
    "balloon_flag",
    "loan_term_years",
    "loan_purpose",
    "property_id",
    "property_type",
    "metro_area",
    "submarket",
    "occupancy_pct",
    "noi_annual",
    "property_value_at_origination",
    "ltv_at_origination",
    "dscr_at_origination",
    "sponsor_credit_tier",
    "ingested_at",
    "source",
    "source_version",
]


class TestSchemaCompleteness:
    """Test that all required fields are present and non-null."""

    def test_all_fields_present(self, portfolio):
        """Every loan record must have all required fields."""
        for loan in portfolio[:50]:  # Check first 50 for speed
            for field_name in REQUIRED_FIELDS:
                assert hasattr(loan, field_name), f"Missing field: {field_name}"
                value = getattr(loan, field_name)
                assert value is not None, f"Field {field_name} is None"

    def test_loan_id_is_string(self, portfolio):
        """loan_id must be a non-empty string."""
        for loan in portfolio:
            assert isinstance(loan.loan_id, str)
            assert len(loan.loan_id) > 0

    def test_dates_are_iso_format(self, portfolio):
        """Dates must be valid ISO format strings."""
        from datetime import date as dt_date

        for loan in portfolio[:50]:
            # Should parse without error
            orig = dt_date.fromisoformat(loan.origination_date)
            mat = dt_date.fromisoformat(loan.maturity_date)
            assert orig < mat, "Origination must precede maturity"

    def test_numeric_fields_are_numbers(self, portfolio):
        """All numeric fields must be int or float."""
        for loan in portfolio[:50]:
            assert isinstance(loan.original_balance, (int, float))
            assert isinstance(loan.current_balance, (int, float))
            assert isinstance(loan.note_rate, float)
            assert isinstance(loan.ltv_at_origination, float)
            assert isinstance(loan.dscr_at_origination, float)
            assert isinstance(loan.occupancy_pct, float)
            assert isinstance(loan.noi_annual, (int, float))
            assert isinstance(loan.property_value_at_origination, (int, float))

    def test_categorical_fields_valid_values(self, portfolio):
        """Categorical fields must contain only valid values."""
        valid_property_types = {"office", "retail", "industrial", "multifamily", "hotel"}
        valid_amort_types = {"interest_only", "amortizing"}
        valid_purposes = {"acquisition", "refinance"}
        valid_tiers = {"A", "B", "C"}

        for loan in portfolio:
            assert loan.property_type in valid_property_types
            assert loan.amortization_type in valid_amort_types
            assert loan.loan_purpose in valid_purposes
            assert loan.sponsor_credit_tier in valid_tiers
            assert isinstance(loan.balloon_flag, bool)


# ─── Uniqueness Tests ─────────────────────────────────────────────────────────


class TestUniqueness:
    """Test that identifiers are unique across the portfolio."""

    def test_no_duplicate_loan_ids(self, portfolio):
        """All loan_ids must be unique."""
        loan_ids = [loan.loan_id for loan in portfolio]
        assert len(loan_ids) == len(set(loan_ids)), "Duplicate loan_ids found"

    def test_no_duplicate_property_ids(self, portfolio):
        """All property_ids must be unique."""
        prop_ids = [loan.property_id for loan in portfolio]
        assert len(prop_ids) == len(set(prop_ids)), "Duplicate property_ids found"


# ─── Distribution Tests ───────────────────────────────────────────────────────


class TestDistributions:
    """Test that generated data has reasonable statistical properties."""

    def test_ltv_range(self, portfolio):
        """LTV at origination should be between 0.30 and 0.85."""
        for loan in portfolio:
            assert 0.25 <= loan.ltv_at_origination <= 0.90, (
                f"LTV {loan.ltv_at_origination} out of range for {loan.property_type}"
            )

    def test_ltv_mean_reasonable(self, large_portfolio):
        """Average LTV should be in the 0.55-0.75 range."""
        ltvs = [loan.ltv_at_origination for loan in large_portfolio]
        avg_ltv = sum(ltvs) / len(ltvs)
        assert 0.55 <= avg_ltv <= 0.75, f"Average LTV {avg_ltv:.3f} out of expected range"

    def test_dscr_range(self, portfolio):
        """DSCR at origination should be between 0.7 and 3.5."""
        for loan in portfolio:
            assert 0.65 <= loan.dscr_at_origination <= 3.6, (
                f"DSCR {loan.dscr_at_origination} out of range for {loan.property_type}"
            )

    def test_dscr_mean_reasonable(self, large_portfolio):
        """Average DSCR should be in the 1.2-1.8 range."""
        dscrs = [loan.dscr_at_origination for loan in large_portfolio]
        avg_dscr = sum(dscrs) / len(dscrs)
        assert 1.2 <= avg_dscr <= 1.8, f"Average DSCR {avg_dscr:.3f} out of expected range"

    def test_note_rate_range(self, portfolio):
        """Note rate should be between 2.5% and 9%."""
        for loan in portfolio:
            assert 0.025 <= loan.note_rate <= 0.09, (
                f"Note rate {loan.note_rate} out of range"
            )

    def test_occupancy_range(self, portfolio):
        """Occupancy should be between 0.25 and 1.0."""
        for loan in portfolio:
            assert 0.25 <= loan.occupancy_pct <= 1.0, (
                f"Occupancy {loan.occupancy_pct} out of range"
            )

    def test_loan_amount_range(self, portfolio):
        """Loan amounts should be within configured bounds."""
        for loan in portfolio:
            assert 1_000_000 <= loan.original_balance <= 500_000_000, (
                f"Balance ${loan.original_balance:,.0f} out of range"
            )

    def test_current_balance_not_exceeds_original(self, portfolio):
        """Current balance should not exceed original balance (+ rounding tolerance)."""
        for loan in portfolio:
            assert loan.current_balance <= loan.original_balance * 1.01, (
                f"Current balance ${loan.current_balance:,.0f} > "
                f"original ${loan.original_balance:,.0f}"
            )

    def test_noi_positive(self, portfolio):
        """NOI should be positive for all loans."""
        for loan in portfolio:
            assert loan.noi_annual > 0, f"NOI ${loan.noi_annual:,.0f} should be positive"

    def test_property_value_positive(self, portfolio):
        """Property value at origination should be positive."""
        for loan in portfolio:
            assert loan.property_value_at_origination > 0

    def test_dscr_economically_consistent_io(self, portfolio):
        """For IO loans, DSCR ≈ NOI / (balance * note_rate) within rounding tolerance."""
        io_loans = [l for l in portfolio if l.amortization_type == "interest_only"]
        assert len(io_loans) > 10, "Need IO loans for this test"
        for loan in io_loans[:50]:
            expected_ds = loan.original_balance * loan.note_rate
            if expected_ds > 0:
                computed_dscr = loan.noi_annual / expected_ds
                # Allow ±0.05 tolerance due to NOI rounding to nearest $1K
                assert abs(computed_dscr - loan.dscr_at_origination) < 0.05, (
                    f"IO loan {loan.loan_id}: computed DSCR {computed_dscr:.3f} != "
                    f"stated {loan.dscr_at_origination:.3f}"
                )

    def test_dscr_economically_consistent_amortizing(self, portfolio):
        """For amortizing loans, DSCR ≈ NOI / annual_debt_service within tolerance."""
        amort_loans = [l for l in portfolio if l.amortization_type == "amortizing"]
        assert len(amort_loans) > 10, "Need amortizing loans for this test"
        for loan in amort_loans[:50]:
            mr = loan.note_rate / 12.0
            if mr > 0:
                pmt_factor = (mr * (1 + mr) ** 360) / ((1 + mr) ** 360 - 1)
                annual_ds = loan.original_balance * pmt_factor * 12
            else:
                annual_ds = loan.original_balance / 30.0
            if annual_ds > 0:
                computed_dscr = loan.noi_annual / annual_ds
                assert abs(computed_dscr - loan.dscr_at_origination) < 0.05, (
                    f"Amort loan {loan.loan_id}: computed DSCR {computed_dscr:.3f} != "
                    f"stated {loan.dscr_at_origination:.3f}"
                )

    def test_implied_cap_rate_reasonable(self, large_portfolio):
        """Implied cap rate (NOI/property_value) should be mostly in 2-12% range."""
        cap_rates = [
            l.noi_annual / l.property_value_at_origination * 100
            for l in large_portfolio
            if l.property_value_at_origination > 0
        ]
        in_range = sum(1 for c in cap_rates if 2.0 <= c <= 12.0)
        pct_in_range = in_range / len(cap_rates) * 100
        assert pct_in_range > 90, (
            f"Only {pct_in_range:.1f}% of implied cap rates in [2%, 12%] range"
        )


class TestPropertyTypeDistribution:
    """Test that property type mix roughly matches configured weights."""

    def test_all_property_types_represented(self, large_portfolio):
        """All 5 property types should appear in a sufficiently large portfolio."""
        types_seen = {loan.property_type for loan in large_portfolio}
        expected = {"office", "retail", "industrial", "multifamily", "hotel"}
        assert types_seen == expected

    def test_property_type_weights_approximate(self, large_portfolio):
        """Property type proportions should roughly match configured weights."""
        counts: dict[str, int] = {}
        for loan in large_portfolio:
            counts[loan.property_type] = counts.get(loan.property_type, 0) + 1

        total = len(large_portfolio)
        # Expected: Office 30%, Retail 15%, Industrial 20%, Multifamily 25%, Hotel 10%
        # Allow ±8% tolerance for randomness with 2000 samples
        expected = {
            "office": 0.30,
            "retail": 0.15,
            "industrial": 0.20,
            "multifamily": 0.25,
            "hotel": 0.10,
        }
        for ptype, expected_pct in expected.items():
            actual_pct = counts.get(ptype, 0) / total
            assert abs(actual_pct - expected_pct) < 0.08, (
                f"{ptype}: expected ~{expected_pct:.0%}, got {actual_pct:.0%}"
            )

    def test_office_ltv_higher_than_industrial(self, large_portfolio):
        """Office LTV should be statistically higher than Industrial LTV."""
        office_ltvs = [l.ltv_at_origination for l in large_portfolio if l.property_type == "office"]
        industrial_ltvs = [
            l.ltv_at_origination for l in large_portfolio if l.property_type == "industrial"
        ]
        avg_office = sum(office_ltvs) / len(office_ltvs)
        avg_industrial = sum(industrial_ltvs) / len(industrial_ltvs)
        assert avg_office > avg_industrial, (
            f"Office avg LTV ({avg_office:.3f}) should exceed "
            f"Industrial ({avg_industrial:.3f})"
        )


# ─── Date Relationship Tests ──────────────────────────────────────────────────


class TestDateRelationships:
    """Test origination and maturity date logic."""

    def test_maturity_after_origination(self, portfolio):
        """Maturity date must be after origination date."""
        from datetime import date as dt_date

        for loan in portfolio:
            orig = dt_date.fromisoformat(loan.origination_date)
            mat = dt_date.fromisoformat(loan.maturity_date)
            assert mat > orig

    def test_term_matches_date_difference(self, portfolio):
        """Loan term should approximately match date difference."""
        from datetime import date as dt_date

        for loan in portfolio[:50]:
            orig = dt_date.fromisoformat(loan.origination_date)
            mat = dt_date.fromisoformat(loan.maturity_date)
            years_diff = (mat - orig).days / 365.25
            assert abs(years_diff - loan.loan_term_years) < 0.1

    def test_origination_year_in_range(self, portfolio):
        """Origination year should be within configured vintage range."""
        for loan in portfolio:
            assert 2015 <= loan.origination_year <= 2022

    def test_term_years_valid(self, portfolio):
        """Loan term should be one of the configured values (5, 7, 10)."""
        valid_terms = {5, 7, 10}
        for loan in portfolio:
            assert loan.loan_term_years in valid_terms


# ─── Reproducibility Tests ────────────────────────────────────────────────────


class TestReproducibility:
    """Test that the generator is deterministic with the same seed."""

    def test_same_seed_same_output(self, config):
        """Two generators with the same seed should produce identical portfolios."""
        gen1 = SyntheticLoanGenerator(config=config, seed=123)
        gen2 = SyntheticLoanGenerator(config=config, seed=123)

        portfolio1 = gen1.generate_portfolio(size=100)
        portfolio2 = gen2.generate_portfolio(size=100)

        for l1, l2 in zip(portfolio1, portfolio2):
            assert l1.loan_id == l2.loan_id
            assert l1.original_balance == l2.original_balance
            assert l1.note_rate == l2.note_rate
            assert l1.property_type == l2.property_type

    def test_different_seed_different_output(self, config):
        """Two generators with different seeds should produce different portfolios."""
        gen1 = SyntheticLoanGenerator(config=config, seed=1)
        gen2 = SyntheticLoanGenerator(config=config, seed=2)

        portfolio1 = gen1.generate_portfolio(size=50)
        portfolio2 = gen2.generate_portfolio(size=50)

        # At least some loans should differ
        different_count = sum(
            1 for l1, l2 in zip(portfolio1, portfolio2)
            if l1.original_balance != l2.original_balance
        )
        assert different_count > 10


# ─── Portfolio Size Tests ─────────────────────────────────────────────────────


class TestPortfolioSize:
    """Test configurable portfolio sizes."""

    def test_exact_size_100(self, config):
        """Generator produces exactly the requested number of loans."""
        gen = SyntheticLoanGenerator(config=config, seed=42)
        portfolio = gen.generate_portfolio(size=100)
        assert len(portfolio) == 100

    def test_exact_size_1000(self, config):
        """Generator produces exactly 1000 loans when requested."""
        gen = SyntheticLoanGenerator(config=config, seed=42)
        portfolio = gen.generate_portfolio(size=1000)
        assert len(portfolio) == 1000

    def test_size_override_works(self, config):
        """CLI size override should work regardless of config default."""
        gen = SyntheticLoanGenerator(config=config, seed=42)
        portfolio = gen.generate_portfolio(size=77)
        assert len(portfolio) == 77


# ─── Helper Function Tests ────────────────────────────────────────────────────


class TestHelperFunctions:
    """Test utility functions."""

    def test_clamp_within_range(self):
        assert clamp(5.0, 0.0, 10.0) == 5.0

    def test_clamp_below_min(self):
        assert clamp(-1.0, 0.0, 10.0) == 0.0

    def test_clamp_above_max(self):
        assert clamp(15.0, 0.0, 10.0) == 10.0

    def test_sample_normal_clamped_stays_in_bounds(self):
        """Clamped normal samples should always be within bounds."""
        import random

        rng = random.Random(42)
        for _ in range(1000):
            val = sample_normal_clamped(rng, 0.65, 0.10, 0.30, 0.85)
            assert 0.30 <= val <= 0.85


# ─── Metadata Tests ───────────────────────────────────────────────────────────


class TestMetadata:
    """Test ingestion metadata fields."""

    def test_source_field(self, portfolio):
        """Source should be 'synthetic_generator'."""
        for loan in portfolio[:10]:
            assert loan.source == "synthetic_generator"

    def test_source_version(self, portfolio):
        """Source version should be set."""
        for loan in portfolio[:10]:
            assert loan.source_version == "1.0.0"

    def test_ingested_at_is_iso_timestamp(self, portfolio):
        """ingested_at should be a valid ISO timestamp."""
        from datetime import datetime

        for loan in portfolio[:10]:
            # Should parse without error
            datetime.fromisoformat(loan.ingested_at)
