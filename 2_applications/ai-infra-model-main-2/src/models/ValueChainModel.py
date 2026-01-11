"""Value Chain Model - DataFrame-Based Architecture

This module implements the ValueChainModel class with explicit DataFrame parameters.
No more nested dataclasses - clean separation of market segments, depreciation schedules,
markup presets, and market toggles.

Architecture:
- start_market_segments: pd.DataFrame - starting customer segments
- target_market_segments: pd.DataFrame - target customer segments for interpolation
- depreciation_schedule: CapexDepreciationSchedule - investment timeline
- markups: pd.Series - pricing power configuration
- toggles: list[ValueChainToggle] - market forces (optional)
"""

from __future__ import annotations

from typing import List, Optional

import pandas as pd

from src.constants.value_chain_depreciation_schedules import (
    NVIDIA_DEFAULT_GROSS_MARGIN,
    CapexDepreciationSchedule,
    validate_depreciation_schedule,
)
from src.constants.value_chain_market_segments import validate_segment_dataframe
from src.constants.value_chain_markups import validate_markups_series
from src.constants.value_chain_toggles import ValueChainToggle, validate_toggles_list


class ValueChainModel:
    """Modern value chain model with explicit DataFrame parameters.

    Purpose:
    - Use CapexDepreciationSchedule for realistic compute hardware investment timeline
    - Apply markup multipliers through the value chain (GPU → Cloud → Model → App)
    - Support both single-year snapshots and multi-year timeline analysis
    - Process market segments as modular DataFrames for flexible scenario composition

    Architecture:
    - market_segments concatenated into internal DataFrame for vectorized operations
    - Toggles applied to markups in __init__ for performance
    - Clean separation of infrastructure costs, user economics, and market dynamics
    """

    def __init__(
        self,
        start_market_segments: pd.DataFrame,
        target_market_segments: pd.DataFrame,
        depreciation_schedule: CapexDepreciationSchedule,
        base_markups: pd.Series,
        toggles: Optional[List[ValueChainToggle]] = None,
        chip_vendor_margin: float = NVIDIA_DEFAULT_GROSS_MARGIN,
    ) -> None:
        """Initialize ValueChainModel with explicit parameters.

        Args:
            start_market_segments: Starting customer segments and cohort composition
            target_market_segments: Target customer segments for interpolation over time
            depreciation_schedule: Investment timeline with asset depreciation
            base_markups: Base pricing power multipliers by value chain layer (cloud/model/app)
            toggles: Optional market force adjustments that modify layer-specific markups
                    (e.g., open model competition → lower model markup, neocloud fragmentation → lower cloud markup)
                    ⚠️ NOTE: Currently always None in practice (see value_chain_app.py:93)
                    Infrastructure exists but toggles are not used in UI.
        """
        # Validate inputs using functions from constants modules
        validate_segment_dataframe(start_market_segments)
        validate_segment_dataframe(target_market_segments)
        validate_markups_series(base_markups)
        validate_depreciation_schedule(depreciation_schedule)
        if toggles:
            validate_toggles_list(toggles)


        # Store components directly
        self.depreciation_schedule = depreciation_schedule.with_adjusted_chip_prices(chip_vendor_margin)
        self.toggles: List[ValueChainToggle] = toggles or []
        self.chip_vendor_margin = chip_vendor_margin

        # Apply market force adjustments to base markups
        # Toggles apply multipliers to specific layers (e.g., open model share → lower model markup)
        if self.toggles:
            adjusted_markups = self._apply_toggle_multipliers(base_markups, self.toggles)
        else:
            adjusted_markups = base_markups.copy()

        # Store both base and market-adjusted markups for transparency
        # base_markup: original pricing power assumptions
        # adjusted_markup: after applying toggle adjustments (used in calculations)
        self.pricing_df = pd.DataFrame({
            'base_markup': base_markups,
            'adjusted_markup': adjusted_markups
        }, index=['cloud', 'model', 'app'])

        # Store both start and target market segments
        self.start_market_segments = start_market_segments.copy()
        self.target_market_segments = target_market_segments.copy()

        # Extract timeline from depreciation schedule
        schedule_df = self.depreciation_schedule.depreciation_accounting_schedule
        # Find first and last years with non-zero depreciation
        non_zero_years = schedule_df[schedule_df.sum(axis=1) > 0].index
        self.start_year = int(non_zero_years.min())
        self.end_year = int(non_zero_years.max())

        # Use start segments as default for single-year calculations
        self._internal_scenario_df = start_market_segments.copy()

    def _apply_toggle_multipliers(self, base_markups: pd.Series, toggles: List[ValueChainToggle]) -> pd.Series:
        """Apply toggle multipliers to base markups.

        Each toggle applies a multiplier to its target markup layer based on market conditions.
        """
        # Work on a copy
        adjusted = base_markups.copy()
        allowed_keys = {"cloud", "model", "app"}  # 3-layer model

        # Validate base markups structure
        keys = set(adjusted.index)
        missing = allowed_keys - keys
        extra = keys - allowed_keys
        if missing or extra:
            raise ValueError(
                f"Markups Series must contain exactly {sorted(allowed_keys)}; "
                f"missing={sorted(missing)} extra={sorted(extra)}"
            )

        # Apply each toggle's multiplier to its target layer
        for toggle in toggles:
            multiplier = float(toggle.get_multiplier())
            if multiplier <= 0.0:
                raise ValueError(f"Toggle '{toggle.label}' produced non-positive multiplier {multiplier}")

            layer = toggle.markup_layer
            if layer not in allowed_keys:
                raise AttributeError(
                    f"Toggle '{toggle.label}' has invalid markup_layer: {layer} (expected one of {sorted(allowed_keys)})"
                )

            # Apply multiplier to the specific layer
            adjusted[layer] *= multiplier

        # Validate final markups are positive (allow < 1 for loss-leaders)
        if not all(float(v) > 0.0 for v in adjusted.values):
            raise ValueError("Resulting markups must be > 0 (k may be < 1 for loss‑leaders)")

        return adjusted

    # -----------------------------
    # Market Segment Interpolation
    # -----------------------------


    # -----------------------------
    # Infrastructure Cost Calculations
    # -----------------------------


    def _calculate_layer_metrics(self, cost_base: float, markup: float) -> tuple[float, float]:
        """Calculate revenue and profit for a single layer."""

        revenue = markup * cost_base
        profit = revenue - cost_base
        return revenue, profit

    def _calculate_all_layers(self, year: int = 2025) -> pd.DataFrame:
        """Calculate full value chain for a given year (3-layer model).

        Returns DataFrame with layers as index and metrics as columns:
        - markup: markup multiplier for this layer
        - revenue: total revenue at this layer
        - profit: profit generated by this layer
        """
        k = self.pricing_df['adjusted_markup']
        depreciation_series = self.depreciation_schedule.get_total_depreciation_by_year()
        compute_cost = float(depreciation_series[year]) if year in depreciation_series.index else 0.0

        # Calculate each layer sequentially (3-layer model)
        cloud_revenue, cloud_profit = self._calculate_layer_metrics(compute_cost, k["cloud"])
        model_revenue, model_profit = self._calculate_layer_metrics(cloud_revenue, k["model"])
        app_revenue, app_profit = self._calculate_layer_metrics(model_revenue, k["app"])

        # Direct DataFrame construction
        return pd.DataFrame(
            data=[
                (k["cloud"], cloud_revenue, cloud_profit),
                (k["model"], model_revenue, model_profit),
                (k["app"], app_revenue, app_profit)
            ],
            columns=["markup", "revenue", "profit"],
            index=["cloud", "model", "app"]
        )

    # -----------------------------
    # User Revenue Calculations (DataFrame-Based)
    # -----------------------------

    def _calculate_segment_economics(self, segments_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate economics for any segments DataFrame (single or multi-year).

        Works with:
        - Single year segments (no 'year' column)
        - Multi-year interpolated segments (has 'year' column)

        Args:
            segments_df: DataFrame with columns: cohort_share, total_addressable_users,
                        arpu, and optionally cost_to_service, year

        Returns:
            Same DataFrame with added columns: users, monthly_revenue, annual_revenue,
            and optionally monthly_cost, annual_cost, annual_profit
        """
        df = segments_df.copy()

        # Core calculation (always the same regardless of single/multi-year)
        df['users'] = (df['total_addressable_users'] * df['cohort_share']).astype(int)
        df['monthly_revenue'] = df['users'] * df['arpu']

        # Cost calculations (optional, only if cost_to_service exists)
        if 'cost_to_service' in df.columns:
            df['monthly_cost'] = df['users'] * df['cost_to_service']

        # Annual calculations (maintain original column order)
        df['annual_revenue'] = df['monthly_revenue'] * 12
        if 'cost_to_service' in df.columns:
            df['annual_cost'] = df['monthly_cost'] * 12
            df['annual_profit'] = df['annual_revenue'] - df['annual_cost']

        return df

    def _calculate_cohort_distribution_and_economics(self) -> pd.DataFrame:
        """Calculate user distribution and economics using centralized calculation."""
        return self._calculate_segment_economics(self._internal_scenario_df)

    def _calculate_user_economics_summary(self) -> pd.Series:
        """Calculate total user economics across all cohorts and segments."""
        cohort_economics = self._calculate_cohort_distribution_and_economics()

        # Sum across all cohorts and segments
        total_monthly_revenue = cohort_economics['monthly_revenue'].sum()
        total_monthly_cost = cohort_economics['monthly_cost'].sum()
        total_annual_revenue = cohort_economics['annual_revenue'].sum()
        total_annual_cost = cohort_economics['annual_cost'].sum()

        return pd.Series({
            "monthly_user_revenue": total_monthly_revenue,
            "annual_user_revenue": total_annual_revenue,
            "monthly_user_cost": total_monthly_cost,
            "annual_user_cost": total_annual_cost,
            "annual_user_profit": total_annual_revenue - total_annual_cost,
            "total_active_users": cohort_economics['users'].sum()
        })

    def _calculate_capex_depreciation_summary(
        self,
        start_year: int | None = None,
        end_year: int | None = None,
    ) -> dict[str, float | int | None]:
        """Summarize capex deployed and depreciation recognized within a year window."""

        schedule_df = self.depreciation_schedule.depreciation_accounting_schedule
        if schedule_df.empty:
            return {
                "period_start_year": None,
                "period_end_year": None,
                "total_capex_committed": 0.0,
                "total_depreciation": 0.0,
            }

        schedule_years = schedule_df.index
        default_start = int(schedule_years.min())
        default_end = int(schedule_years.max())

        window_start = start_year if start_year is not None else default_start
        window_end = end_year if end_year is not None else default_end

        if window_start > window_end:
            raise ValueError("start_year cannot be greater than end_year")

        # Clip to available schedule years to avoid KeyErrors when using loc
        window_start = max(window_start, default_start)

        capex_slice = schedule_df.loc[window_start:window_end]
        total_capex_committed = float(capex_slice.fillna(0.0).sum().sum())

        depreciation_series = self.depreciation_schedule.get_total_depreciation_by_year()
        if depreciation_series.empty:
            total_depreciation = 0.0
        else:
            depreciation_window = depreciation_series[(depreciation_series.index >= window_start) & (depreciation_series.index <= window_end)]
            total_depreciation = float(depreciation_window.sum())

        return {
            "period_start_year": window_start,
            "period_end_year": window_end,
            "total_capex_committed": total_capex_committed,
            "total_depreciation": total_depreciation,
        }

    # -----------------------------
    # Market Segment Interpolation
    # -----------------------------

    @staticmethod
    def _fill_missing_cohorts_in_both_dataframes(
        start_df: pd.DataFrame,
        end_df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Fill missing cohorts in both DataFrames with zero cohort_share.

        Args:
            start_df: Starting segments DataFrame
            end_df: Ending segments DataFrame

        Returns:
            Tuple of (filled_start_df, filled_end_df) with all combinations present in both
        """
        start_df = start_df.copy()
        end_df = end_df.copy()

        # Get all unique segment-cohort combinations from both DataFrames
        start_combinations = set(zip(start_df['segment'], start_df['cohort_name']))
        end_combinations = set(zip(end_df['segment'], end_df['cohort_name']))

        # Add missing cohorts to start_df (those present in end but not start)
        missing_in_start = end_combinations - start_combinations
        for segment, cohort_name in missing_in_start:
            template_row = end_df[
                (end_df['segment'] == segment) &
                (end_df['cohort_name'] == cohort_name)
            ].iloc[0].copy()
            template_row['cohort_share'] = 0.0
            start_df = pd.concat([start_df, template_row.to_frame().T], ignore_index=True)

        # Add missing cohorts to end_df (those present in start but not end)
        missing_in_end = start_combinations - end_combinations
        for segment, cohort_name in missing_in_end:
            template_row = start_df[
                (start_df['segment'] == segment) &
                (start_df['cohort_name'] == cohort_name)
            ].iloc[0].copy()
            template_row['cohort_share'] = 0.0
            end_df = pd.concat([end_df, template_row.to_frame().T], ignore_index=True)

        return start_df, end_df

    @staticmethod
    def _create_interpolated_row(
        start_cohort: pd.Series,
        end_cohort: pd.Series,
        year: int,
        segment_name: str,
        cohort_name: str,
        start_year: int,
        end_year: int
    ) -> dict:
        """Create a single interpolated row for a cohort in a specific year."""
        # Calculate interpolation weight (0.0 at start_year, 1.0 at end_year)
        if start_year == end_year:
            # Single year case - use start values
            interpolation_weight = 0.0
        else:
            interpolation_weight = (year - start_year) / (end_year - start_year)

        # Interpolate cohort_share (userbase) between start and end values
        interpolated_cohort_share = (
            start_cohort['cohort_share'] * (1 - interpolation_weight) +
            end_cohort['cohort_share'] * interpolation_weight
        )

        # Interpolate total_addressable_users (market size) between start and end values
        interpolated_tam = (
            start_cohort['total_addressable_users'] * (1 - interpolation_weight) +
            end_cohort['total_addressable_users'] * interpolation_weight
        )

        # Interpolate ARPU (average revenue per user) between start and end values
        interpolated_arpu = (
            start_cohort['arpu'] * (1 - interpolation_weight) +
            end_cohort['arpu'] * interpolation_weight
        )

        # Interpolate cost_to_service between start and end values
        interpolated_cost = (
            start_cohort['cost_to_service'] * (1 - interpolation_weight) +
            end_cohort['cost_to_service'] * interpolation_weight
        )

        return {
            'year': year,
            'segment': segment_name,
            'cohort_name': cohort_name,
            'cohort_share': interpolated_cohort_share,
            'arpu': interpolated_arpu,  # Interpolate ARPU
            'cost_to_service': interpolated_cost,  # Interpolate cost
            'total_addressable_users': interpolated_tam,  # Interpolate TAM
        }

    @staticmethod
    def interpolate_market_segments_over_time(
        start_segments: pd.DataFrame,
        end_segments: pd.DataFrame,
        start_year: int,
        end_year: int
    ) -> pd.DataFrame:
        """Linearly interpolate market segments between start and end years.

        Interpolates userbase (cohort_share) while keeping ARPU, cost_to_service,
        and total_addressable_users constant.

        Args:
            start_segments: DataFrame with concatenated starting segments
            end_segments: DataFrame with concatenated ending segments
            start_year: Starting year (e.g., 2025)
            end_year: Ending year (e.g., 2030)

        Returns:
            DataFrame with columns: year, segment, cohort_name, cohort_share,
            arpu, cost_to_service, total_addressable_users
        """
        if start_year > end_year:
            raise ValueError("start_year must be less than or equal to end_year")

        if start_segments.empty or end_segments.empty:
            raise ValueError("Both start_segments and end_segments must be non-empty")

        # Handle missing cohorts by inserting them with zero cohort_share
        start_df, end_df = ValueChainModel._fill_missing_cohorts_in_both_dataframes(
            start_segments, end_segments
        )

        # Since we've added all missing cohorts, we can iterate through all combinations
        # Generate all years in the range
        years = list(range(start_year, end_year + 1))

        # Build interpolated data
        interpolated_data = []

        # Get all unique segment-cohort combinations (they should now exist in both DataFrames)
        all_combinations = set(zip(start_df['segment'], start_df['cohort_name']))

        for segment_name, cohort_name in all_combinations:
            # Find the matching cohort rows in both start and end DataFrames
            # These will be used to interpolate cohort_share while keeping other values constant
            start_cohort = start_df[
                (start_df['segment'] == segment_name) &
                (start_df['cohort_name'] == cohort_name)
            ].iloc[0]
            end_cohort = end_df[
                (end_df['segment'] == segment_name) &
                (end_df['cohort_name'] == cohort_name)
            ].iloc[0]

            for year in years:
                interpolated_row = ValueChainModel._create_interpolated_row(
                    start_cohort, end_cohort, year, segment_name, cohort_name,
                    start_year, end_year
                )
                interpolated_data.append(interpolated_row)

        return pd.DataFrame(interpolated_data)

    # -----------------------------
    # Multi-Year Timeline Analysis
    # -----------------------------

    def _calculate_revenue_from_interpolated_segments(self, interpolated_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate revenue timeline from interpolated market segments.

        Args:
            interpolated_df: DataFrame from get_interpolated_timeline() with interpolated cohort shares

        Returns:
            DataFrame with columns: year, segment, annual_revenue, cohort_share
        """
        if interpolated_df.empty:
            return pd.DataFrame(columns=['year', 'segment', 'annual_revenue', 'cohort_share'])

        # Use centralized calculation (vectorized, not loops!)
        df_with_economics = self._calculate_segment_economics(interpolated_df)

        # Aggregate by year and segment
        result = df_with_economics.groupby(['year', 'segment']).agg({
            'annual_revenue': 'sum',
            'cohort_share': 'sum'
        }).reset_index()

        return result

    def compute_value_chain_timeline(self) -> pd.DataFrame:
        """Compute multi-year value chain analysis using depreciation schedule.

        Simplified 3-layer model (no GPU maker complexity):
        - Cloud revenue = total_depreciation * k_cloud (covers all infra costs)
        - Model revenue = cloud_revenue * k_model
        - App revenue = model_revenue * k_app

        Returns DataFrame with years and value chain metrics.
        """
        depreciation_series = self.depreciation_schedule.get_total_depreciation_by_year()

        if depreciation_series.empty:
            return pd.DataFrame()

        # Get market-adjusted markup values (3-layer model)
        adjusted_markups = self.pricing_df['adjusted_markup']
        k_cloud = adjusted_markups["cloud"]
        k_model = adjusted_markups["model"]
        k_app = adjusted_markups["app"]

        # Sequential calculation - simplified 3-layer model (revenues required to achieve target margins)
        required_cloud_revenue_for_margins = depreciation_series * k_cloud
        required_model_revenue_for_margins = required_cloud_revenue_for_margins * k_model
        required_app_revenue_for_margins = required_model_revenue_for_margins * k_app

        # Total profit = final required app revenue - original cost base
        total_profit = required_app_revenue_for_margins - depreciation_series

        # Calculate shortfall based on required revenues (not actual interpolated market revenues)
        # This shortfall represents: aggregate_capex - required_app_revenue_for_margins
        # Note: For funding analysis, use get_funding_shortfall_timeline() with interpolated_projected_revenue
        shortfall = abs(depreciation_series) - required_app_revenue_for_margins

        timeline_data = {
            'year': depreciation_series.index,
            'depreciation': depreciation_series.values,
            'required_cloud_revenue_for_margins': required_cloud_revenue_for_margins.values,
            'required_model_revenue_for_margins': required_model_revenue_for_margins.values,
            'required_app_revenue_for_margins': required_app_revenue_for_margins.values,
            'total_profit': total_profit.values,
            'shortfall': shortfall.values,
        }

        return pd.DataFrame(timeline_data)

    def get_funding_shortfall_timeline(self, interpolated_revenue_timeline: pd.DataFrame = None) -> pd.DataFrame:
        """Get annual funding shortfall: aggregate_capex - interpolated_projected_revenue.

        Args:
            interpolated_revenue_timeline: DataFrame with year, annual_revenue from market interpolation

        Returns:
            DataFrame with columns: year, shortfall
        """
        # Get actual capex investment amounts (not depreciation)
        capex_schedule_df = self.depreciation_schedule.depreciation_accounting_schedule
        if capex_schedule_df.empty:
            return pd.DataFrame(columns=['year', 'shortfall'])

        # Calculate total capex investment per year
        total_capex_by_year = capex_schedule_df.sum(axis=1)  # Sum across all asset types per year

        if interpolated_revenue_timeline is None or interpolated_revenue_timeline.empty:
            # Fallback to timeline calculation if no market revenue provided
            timeline = self.compute_value_chain_timeline()
            return timeline[['year', 'shortfall']].copy()

        # Aggregate interpolated projected revenue by year
        market_revenue_by_year = interpolated_revenue_timeline.groupby('year')['annual_revenue'].sum()

        # Vectorized shortfall calculation: aggregate_capex - interpolated_projected_revenue
        shortfall_df = pd.DataFrame({
            'year': total_capex_by_year.index,
            'aggregate_capex': total_capex_by_year.values  # Use actual capex investment amounts
        })

        # Prepare interpolated projected revenue data
        market_rev_df = market_revenue_by_year.reset_index().rename(
            columns={'annual_revenue': 'interpolated_projected_revenue'}
        )

        # Merge with interpolated projected revenues (left join to keep all capex years)
        shortfall_df = shortfall_df.merge(market_rev_df, on='year', how='left').fillna(0)

        # Calculate shortfall: aggregate_capex - interpolated_projected_revenue
        shortfall_df['shortfall'] = shortfall_df['aggregate_capex'] - shortfall_df['interpolated_projected_revenue']

        return shortfall_df[['year', 'shortfall']]

    # -----------------------------
    # Public Interface Methods
    # -----------------------------


    def run_full_analysis(self) -> dict:
        """Run complete multi-year analysis.

        Returns dictionary with timeline and summary metrics.
        """
        # Multi-year timeline
        timeline_df = self.compute_value_chain_timeline()

        # User economics (consistent across years)
        user_economics = self._calculate_user_economics_summary()
        cohort_breakdown = self._calculate_cohort_distribution_and_economics()
        capex_summary = self._calculate_capex_depreciation_summary()

        # Interpolation moved to display layer to use proper display years

        return {
            "timeline": timeline_df,
            "user_economics_summary": user_economics,
            "cohort_breakdown": cohort_breakdown,
            "model_configuration": {
                "num_segments": len(self._internal_scenario_df['segment'].unique()),
                "num_cohorts": len(self._internal_scenario_df),
                "base_markups": self.pricing_df['base_markup'].to_dict(),
                "adjusted_markups": self.pricing_df['adjusted_markup'].to_dict(),
                "num_toggles_applied": len(self.toggles),
            },
            "capex_depreciation_summary": capex_summary,
        }
