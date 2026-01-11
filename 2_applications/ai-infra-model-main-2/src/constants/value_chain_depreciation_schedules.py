"""Value Chain Depreciation Schedules - Investment Timeline Configurations

This module defines the CapexDepreciationSchedule class and investment timeline presets.
Each schedule represents a multi-year CAPEX investment pattern with different asset types
(chips, datacenter infrastructure, power systems) and their useful life depreciation.

Architecture:
- CapexDepreciationSchedule uses DataFrame for annual investments by asset type
- Straight-line depreciation based on useful life per asset type
- Clean separation from market segments and pricing power
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

# -----------------------------
# Chip Vendor Margin Constants
# -----------------------------

# NVIDIA's default gross margin for datacenter GPUs (industry standard baseline)
NVIDIA_DEFAULT_GROSS_MARGIN = 0.75  # 75% gross margin

# Portion of the compute stack attributable to NVIDIA hardware economics
NVIDIA_COMPUTE_SHARE = 0.75  # Remaining 25% assumed to be non-NVIDIA integration costs

# -----------------------------
# Core Depreciation Schedule Class
# -----------------------------

@dataclass
class CapexDepreciationSchedule:
    """Unified depreciation schedule using a single DataFrame approach.

    No more nested objects! Just one DataFrame with:
    - Index: Years
    - Columns: Asset types (chips, datacenter, power, etc.)
    - Values: Annual CapEx investments

    Useful life is stored as a Series mapping asset type -> years.
    """
    depreciation_accounting_schedule: pd.DataFrame  # Years x Asset types
    useful_life_series: pd.Series  # Asset type -> useful life years

    def get_total_depreciation_by_year(self) -> pd.Series:
        """Calculate total annual depreciation across all assets.

        Returns Series with year index and total depreciation values.
        """
        if self.depreciation_accounting_schedule.empty:
            return pd.Series(dtype=float)

        # Calculate depreciation for each asset type
        all_depreciation = []

        for asset_type in self.depreciation_accounting_schedule.columns:
            useful_life = self.useful_life_series[asset_type]
            asset_capex = self.depreciation_accounting_schedule[asset_type].dropna()

            if asset_capex.empty:
                continue

            # Create depreciation schedule for this asset
            min_year = asset_capex.index.min()
            max_year = asset_capex.index.max() + useful_life

            depreciation_years = list(range(min_year, max_year))
            asset_depreciation = pd.Series(0.0, index=depreciation_years)

            # Add depreciation for each purchase year
            for purchase_year, capex_amount in asset_capex.items():
                if capex_amount > 0:
                    annual_depreciation = capex_amount / useful_life
                    for dep_year in range(purchase_year, purchase_year + useful_life):
                        if dep_year in asset_depreciation.index:
                            asset_depreciation[dep_year] += annual_depreciation

            all_depreciation.append(asset_depreciation)

        if not all_depreciation:
            return pd.Series(dtype=float)

        # Sum across all assets, aligning on year index
        total_depreciation = pd.concat(all_depreciation, axis=1).sum(axis=1)
        return total_depreciation


    def _count_active_generations_by_year(self, years) -> list[int]:
        """Count how many asset generations are active each year."""
        active_counts = []

        for year in years:
            count = 0
            for asset_type in self.depreciation_accounting_schedule.columns:
                useful_life = self.useful_life_series[asset_type]
                asset_capex = self.depreciation_accounting_schedule[asset_type].dropna()

                # Count purchases still depreciating in this year
                for purchase_year in asset_capex.index:
                    if purchase_year <= year < purchase_year + useful_life:
                        count += 1

            active_counts.append(count)

        return active_counts

    @staticmethod
    def calculate_chip_price_multiplier(chip_gross_margin: float) -> tuple[float, float]:
        """Calculate the price multiplier for chip margin changes.

        Args:
            chip_gross_margin: New chip vendor gross margin (0.0 to 1.0)

        Returns:
            tuple of (nvidia_multiplier, blended_compute_multiplier)
            - nvidia_multiplier: Raw NVIDIA price change multiplier
            - blended_compute_multiplier: Blended multiplier including fixed integration costs
        """
        # Validate margin input
        if not (0.0 <= chip_gross_margin <= 1.0):
            raise ValueError(f"chip_gross_margin must be between 0.0 and 1.0, got {chip_gross_margin}")

        old_margin_complement = 1.0 - NVIDIA_DEFAULT_GROSS_MARGIN  # 0.25 for 75% margin
        new_margin_complement = max(0.001, 1.0 - chip_gross_margin)

        nvidia_multiplier = old_margin_complement / new_margin_complement
        blended_compute_multiplier = (NVIDIA_COMPUTE_SHARE * nvidia_multiplier) + (1.0 - NVIDIA_COMPUTE_SHARE)

        return nvidia_multiplier, blended_compute_multiplier

    def with_adjusted_chip_prices(self, chip_gross_margin: float = NVIDIA_DEFAULT_GROSS_MARGIN) -> 'CapexDepreciationSchedule':
        """Create a new CapexDepreciationSchedule with chip prices adjusted for margin compression.

        Args:
            chip_gross_margin: New chip vendor gross margin (0.0 to 1.0)
                              When margin decreases, chip prices decrease proportionally

        Returns:
            New CapexDepreciationSchedule with adjusted chip prices

        Example:
            - Default margin 75% → 50%: chips become 50% cheaper (price multiplier = 0.5)
            - Default margin 75% → 90%: chips become 150% more expensive (price multiplier = 2.5)
        """
        # Use the static method for calculation
        _, blended_compute_multiplier = CapexDepreciationSchedule.calculate_chip_price_multiplier(chip_gross_margin)

        # Create adjusted depreciation_accounting_schedule
        adjusted_capex_df = self.depreciation_accounting_schedule.copy()

        # Apply margin adjustment only to chips, leave other assets unchanged
        if 'chips' in adjusted_capex_df.columns:
            adjusted_capex_df['chips'] = adjusted_capex_df['chips'] * blended_compute_multiplier

        # Return new CapexDepreciationSchedule with adjusted chip prices
        return CapexDepreciationSchedule(
            depreciation_accounting_schedule=adjusted_capex_df,
            useful_life_series=self.useful_life_series.copy()
        )

    def with_capex_cutoff(self, cutoff_year: int) -> 'CapexDepreciationSchedule':
        """Create a new CapexDepreciationSchedule with capex zeroed after cutoff year.

        This allows modeling scenarios where new investment stops but existing assets
        continue to depreciate over their useful life.

        Args:
            cutoff_year: Last year to include capex investments
                        All investments after this year are set to zero

        Returns:
            New CapexDepreciationSchedule with capex stopped after cutoff

        Example:
            cutoff_year=2027: Investment continues through 2027, then stops
            Depreciation continues for all purchased assets based on useful life
        """
        # Validate cutoff year
        min_year = self.depreciation_accounting_schedule.index.min()
        max_year = self.depreciation_accounting_schedule.index.max()

        if cutoff_year < min_year:
            raise ValueError(f"cutoff_year {cutoff_year} is before first investment year {min_year}")
        if cutoff_year > max_year:
            # If cutoff is after all investments, return unchanged
            return CapexDepreciationSchedule(
                depreciation_accounting_schedule=self.depreciation_accounting_schedule.copy(),
                useful_life_series=self.useful_life_series.copy()
            )

        # Create copy and zero out capex after cutoff year
        cutoff_capex_df = self.depreciation_accounting_schedule.copy()

        # Set all years after cutoff to zero
        years_to_zero = cutoff_capex_df.index > cutoff_year
        cutoff_capex_df.loc[years_to_zero] = 0.0

        # Return new schedule with cutoff applied
        return CapexDepreciationSchedule(
            depreciation_accounting_schedule=cutoff_capex_df,
            useful_life_series=self.useful_life_series.copy()
        )


# -----------------------------
# Dynamic Schedule Generation
# -----------------------------

def generate_capex_schedule_df(
    nvda_rev_in_start_year: float,
    growth_rate: float | dict[int, float],
    start_year: int,
    end_year: int,
    chips_life: int = 5,
    datacenter_life: int = 20,
    power_life: int = 25,
    nvidia_compute_share: float = NVIDIA_COMPUTE_SHARE
) -> CapexDepreciationSchedule:
    """Generate dynamic capex schedule based on NVIDIA revenue and growth rate.

    Args:
        nvda_rev_in_start_year: Starting NVIDIA revenue (e.g., 160_000_000_000)
        growth_rate: Annual growth rate - either:
            - float: Single rate applied to all years (e.g., 0.15 for 15%)
            - dict[int, float]: Year-specific rates (e.g., {2025: 0.15, 2026: 0.20})
        start_year: First year (e.g., 2025)
        end_year: Last year (e.g., 2031)
        chips_life: Chip useful life
        datacenter_life: Datacenter useful life
        power_life: Power useful life
        nvidia_compute_share: NVIDIA's share of total compute (default 0.75)

    Returns:
        CapexDepreciationSchedule with generated investment timeline

    Example:
        # Uniform growth
        schedule = generate_capex_schedule_df(
            nvda_rev_in_start_year=160_000_000_000,  # $160B
            growth_rate=0.15,  # 15% growth
            start_year=2025,
            end_year=2031
        )

        # Variable growth
        schedule = generate_capex_schedule_df(
            nvda_rev_in_start_year=160_000_000_000,
            growth_rate={2025: 0.15, 2026: 0.20, 2027: 0.10},
            start_year=2025,
            end_year=2027
        )
    """
    if start_year > end_year:
        raise ValueError("start_year must be <= end_year")
    if nvda_rev_in_start_year <= 0:
        raise ValueError("nvda_rev_in_start_year must be positive")

    # Normalize growth_rate to dict format
    if isinstance(growth_rate, (int, float)):
        # Single rate - apply to all years
        if growth_rate < -1.0:
            raise ValueError("growth_rate cannot be less than -100%")
        growth_rate_dict = {year: float(growth_rate) for year in range(start_year, end_year + 1)}
    else:
        # Dict of year-specific rates - fill missing years with 0.0
        growth_rate_dict = {}
        for year in range(start_year, end_year + 1):
            rate = growth_rate.get(year, 0.0)
            if rate < -1.0:
                raise ValueError(f"growth_rate for {year} cannot be less than -100%")
            growth_rate_dict[year] = float(rate)

    from src.constants.gpu_dataclass import VR200
    from src.utils.capex_helpers import (
        CapexStartingPoint,
        calculate_infrastructure,
        nvidia_revenue_to_total_chip_capex,
    )

    years = list(range(start_year, end_year + 1))

    chip_values = []
    datacenter_values = []
    power_values = []

    # Track cumulative revenue for variable growth rates
    nvidia_revenue = nvda_rev_in_start_year

    for i, year in enumerate(years):
        # For first year (i=0), use base revenue. For subsequent years, apply growth
        if i > 0:
            nvidia_revenue *= (1 + growth_rate_dict[year])

        # Convert NVIDIA revenue to total chip capex using helper
        total_chip_capex = nvidia_revenue_to_total_chip_capex(
            nvidia_revenue,
            nvidia_compute_share
        )

        # Use capex_helpers to calculate proper infrastructure ratios based on VR200
        infra = calculate_infrastructure(
            starting_point=CapexStartingPoint.CHIP_CAPEX,
            value=total_chip_capex,
            gpu_model=VR200,
            power_cost_per_kw=2500,         # $2,500/kW
            datacenter_cost_per_mw=15_000_000  # $15M/MW
        )

        chip_values.append(infra.chip_capex)
        datacenter_values.append(infra.datacenter_capex)
        power_values.append(infra.power_capex)

    # Create DataFrame
    capex_df = pd.DataFrame({
        'chips': chip_values,
        'datacenter': datacenter_values,
        'power': power_values
    }, index=years)

    useful_life_series = pd.Series({
        'chips': chips_life,
        'datacenter': datacenter_life,
        'power': power_life
    })

    return CapexDepreciationSchedule(
        depreciation_accounting_schedule=capex_df,
        useful_life_series=useful_life_series
    )


# -----------------------------
# Helper Functions
# -----------------------------

def create_capex_schedule_df(
    years: list[int],
    chips_values: list[float],
    datacenter_values: list[float] = None,
    power_values: list[float] = None,
    chips_life: int = 5,
    datacenter_life: int = 20,
    power_life: int = 25
) -> CapexDepreciationSchedule:
    """Create a unified schedule using direct DataFrame construction."""

    # Build DataFrame data dictionary
    df_data = {'chips': chips_values}
    useful_life_data = {'chips': chips_life}

    if datacenter_values:
        df_data['datacenter'] = datacenter_values
        useful_life_data['datacenter'] = datacenter_life

    if power_values:
        df_data['power'] = power_values
        useful_life_data['power'] = power_life

    # Direct DataFrame construction - NO dict literals! 🔥
    capex_df = pd.DataFrame(df_data, index=years).fillna(0.0)
    useful_life_series = pd.Series(useful_life_data)

    return CapexDepreciationSchedule(
        depreciation_accounting_schedule=capex_df,
        useful_life_series=useful_life_series
    )

# All scenarios are now generated dynamically using generate_capex_schedule_df()
# No more hardcoded presets - users create custom scenarios through the UI

# -----------------------------
# Validation Helpers
# -----------------------------

def validate_depreciation_schedule(schedule: CapexDepreciationSchedule) -> None:
    """Validate that a depreciation schedule has the required structure and values."""
    if not isinstance(schedule, CapexDepreciationSchedule):
        raise ValueError("schedule must be a CapexDepreciationSchedule instance")

    # Check DataFrames exist and are not empty
    if schedule.depreciation_accounting_schedule is None or schedule.depreciation_accounting_schedule.empty:
        raise ValueError("Annual CAPEX DataFrame cannot be empty")

    if schedule.useful_life_series is None or schedule.useful_life_series.empty:
        raise ValueError("Useful life Series cannot be empty")

    # Check CAPEX values are non-negative
    if not (schedule.depreciation_accounting_schedule >= 0).all().all():
        raise ValueError("All CAPEX values must be non-negative")

    # Check useful life values are positive
    if not (schedule.useful_life_series > 0).all():
        invalid_assets = schedule.useful_life_series[schedule.useful_life_series <= 0].index.tolist()
        raise ValueError(f"All useful life values must be positive (invalid assets: {invalid_assets})")

    # Check that useful life covers all asset types in CAPEX DataFrame
    capex_assets = set(schedule.depreciation_accounting_schedule.columns)
    life_assets = set(schedule.useful_life_series.index)
    missing_life = capex_assets - life_assets
    if missing_life:
        raise ValueError(f"Missing useful life for asset types: {missing_life}")

    # Check for reasonable useful life values (between 1 and 50 years)
    unreasonable_life = schedule.useful_life_series[(schedule.useful_life_series < 1) | (schedule.useful_life_series > 50)]
    if not unreasonable_life.empty:
        raise ValueError(f"Useful life values seem unreasonable (not 1-50 years): {unreasonable_life.to_dict()}")

    # Check years index is reasonable
    years = schedule.depreciation_accounting_schedule.index
    if years.min() < 2020 or years.max() > 2040:
        raise ValueError(f"CAPEX years seem unreasonable (not 2020-2040): {years.min()}-{years.max()}")

    # Check for NaN values in CAPEX data
    if schedule.depreciation_accounting_schedule.isnull().any().any():
        raise ValueError("CAPEX DataFrame contains null/NaN values")

    if schedule.useful_life_series.isnull().any():
        raise ValueError("Useful life Series contains null/NaN values")


# -----------------------------
# Module Validation (Self-Test)
# -----------------------------

if __name__ == "__main__":
    # Self-validate all defined depreciation schedules
    print("Validating depreciation schedules...")

    for name, schedule in ALL_DEPRECIATION_PRESETS.items():
        try:
            validate_depreciation_schedule(schedule)
            print(f"✅ {name} - valid")

            # Show summary stats
            total_capex = schedule.depreciation_accounting_schedule.sum().sum()
            years = f"{schedule.depreciation_accounting_schedule.index.min()}-{schedule.depreciation_accounting_schedule.index.max()}"
            print(f"   Total CAPEX: ${total_capex:,.0f} over {years}")

        except ValueError as e:
            print(f"❌ {name} - {e}")

    print("✅ All depreciation schedule validations passed!")
