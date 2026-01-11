"""Tests for generate_capex_schedule_df with variable growth rates."""

import pytest
from src.constants.value_chain_depreciation_schedules import generate_capex_schedule_df


class TestGenerateCapexScheduleWithVariableGrowth:
    """Test suite for variable growth rate functionality."""

    def test_uniform_growth_rate_as_float(self):
        """Test that a single float growth rate works (backward compatibility)."""
        schedule = generate_capex_schedule_df(
            nvda_rev_in_start_year=100_000_000_000,  # $100B
            growth_rate=0.15,  # 15% uniform growth
            start_year=2025,
            end_year=2027,
            chips_life=5,
            datacenter_life=20,
            power_life=25
        )

        # Verify schedule was created
        assert schedule is not None
        df = schedule.depreciation_accounting_schedule
        assert not df.empty
        assert 2025 in df.index
        assert 2027 in df.index

        # Verify growth is applied uniformly (each year should be 15% higher than previous)
        chips_2025 = df.loc[2025, 'chips']
        chips_2026 = df.loc[2026, 'chips']
        chips_2027 = df.loc[2027, 'chips']

        # Year 2026 should be ~15% higher than 2025
        assert chips_2026 / chips_2025 == pytest.approx(1.15, rel=0.01)
        # Year 2027 should be ~15% higher than 2026
        assert chips_2027 / chips_2026 == pytest.approx(1.15, rel=0.01)

    def test_variable_growth_rates_as_dict(self):
        """Test that variable growth rates per year work correctly."""
        schedule = generate_capex_schedule_df(
            nvda_rev_in_start_year=100_000_000_000,  # $100B
            growth_rate={
                2025: 0.10,  # 10% growth in 2025
                2026: 0.20,  # 20% growth in 2026
                2027: 0.05   # 5% growth in 2027
            },
            start_year=2025,
            end_year=2027,
            chips_life=5,
            datacenter_life=20,
            power_life=25
        )

        # Verify schedule was created
        assert schedule is not None
        df = schedule.depreciation_accounting_schedule
        assert not df.empty

        # Verify variable growth is applied correctly
        chips_2025 = df.loc[2025, 'chips']
        chips_2026 = df.loc[2026, 'chips']
        chips_2027 = df.loc[2027, 'chips']

        # Year 2026 should be ~20% higher than 2025 (applying 2026's rate)
        assert chips_2026 / chips_2025 == pytest.approx(1.20, rel=0.01)
        # Year 2027 should be ~5% higher than 2026 (applying 2027's rate)
        assert chips_2027 / chips_2026 == pytest.approx(1.05, rel=0.01)

    def test_missing_years_default_to_zero_growth(self):
        """Test that missing years in dict default to 0% growth."""
        schedule = generate_capex_schedule_df(
            nvda_rev_in_start_year=100_000_000_000,  # $100B
            growth_rate={
                2025: 0.10,  # Only specify 2025
                # 2026 and 2027 should default to 0%
            },
            start_year=2025,
            end_year=2027,
            chips_life=5,
            datacenter_life=20,
            power_life=25
        )

        df = schedule.depreciation_accounting_schedule

        chips_2025 = df.loc[2025, 'chips']
        chips_2026 = df.loc[2026, 'chips']
        chips_2027 = df.loc[2027, 'chips']

        # Year 2026 should be same as 2025 (0% growth)
        assert chips_2026 == pytest.approx(chips_2025, rel=0.01)
        # Year 2027 should be same as 2026 (0% growth)
        assert chips_2027 == pytest.approx(chips_2026, rel=0.01)

    def test_zero_growth_rate(self):
        """Test that zero growth rate produces constant revenue."""
        schedule = generate_capex_schedule_df(
            nvda_rev_in_start_year=100_000_000_000,
            growth_rate=0.0,  # No growth
            start_year=2025,
            end_year=2027,
            chips_life=5,
            datacenter_life=20,
            power_life=25
        )

        df = schedule.depreciation_accounting_schedule

        # All years should have same chip capex (flat revenue)
        chips_2025 = df.loc[2025, 'chips']
        chips_2026 = df.loc[2026, 'chips']
        chips_2027 = df.loc[2027, 'chips']

        assert chips_2025 == pytest.approx(chips_2026, rel=0.01)
        assert chips_2026 == pytest.approx(chips_2027, rel=0.01)

    def test_negative_growth_rate(self):
        """Test that negative growth rate produces declining revenue."""
        schedule = generate_capex_schedule_df(
            nvda_rev_in_start_year=100_000_000_000,
            growth_rate=-0.10,  # -10% decline
            start_year=2025,
            end_year=2027,
            chips_life=5,
            datacenter_life=20,
            power_life=25
        )

        df = schedule.depreciation_accounting_schedule

        chips_2025 = df.loc[2025, 'chips']
        chips_2026 = df.loc[2026, 'chips']
        chips_2027 = df.loc[2027, 'chips']

        # Each year should be 10% lower
        assert chips_2026 / chips_2025 == pytest.approx(0.90, rel=0.01)
        assert chips_2027 / chips_2026 == pytest.approx(0.90, rel=0.01)

    def test_invalid_growth_rate_below_negative_100(self):
        """Test that growth rates below -100% raise an error."""
        with pytest.raises(ValueError, match="cannot be less than -100%"):
            generate_capex_schedule_df(
                nvda_rev_in_start_year=100_000_000_000,
                growth_rate=-1.5,  # -150% is invalid
                start_year=2025,
                end_year=2027
            )

    def test_invalid_dict_growth_rate_below_negative_100(self):
        """Test that invalid dict growth rates raise an error."""
        with pytest.raises(ValueError, match="cannot be less than -100%"):
            generate_capex_schedule_df(
                nvda_rev_in_start_year=100_000_000_000,
                growth_rate={2025: 0.10, 2026: -1.5},  # -150% is invalid
                start_year=2025,
                end_year=2027
            )

    def test_all_asset_types_affected_by_growth(self):
        """Test that growth affects chips, datacenter, and power proportionally."""
        schedule = generate_capex_schedule_df(
            nvda_rev_in_start_year=100_000_000_000,
            growth_rate=0.20,  # 20% growth
            start_year=2025,
            end_year=2026,
            chips_life=5,
            datacenter_life=20,
            power_life=25
        )

        df = schedule.depreciation_accounting_schedule

        # Check all asset types grow proportionally
        for asset_type in ['chips', 'datacenter', 'power']:
            value_2025 = df.loc[2025, asset_type]
            value_2026 = df.loc[2026, asset_type]
            growth_ratio = value_2026 / value_2025

            # All should grow by ~20%
            assert growth_ratio == pytest.approx(1.20, rel=0.01)
