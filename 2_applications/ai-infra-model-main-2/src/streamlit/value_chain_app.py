"""Streamlit interface for the value chain analysis workflow."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

import streamlit as st
from src.constants.value_chain_depreciation_schedules import (
    CapexDepreciationSchedule,
    generate_capex_schedule_df,
)
from src.constants.value_chain_market_segments import (
    ALL_MARKET_SCENARIOS,
    TODAY_MARKET_SEGMENTS,
)
from src.constants.value_chain_markups import make_markups_from_margins
from src.models.ValueChainModel import ValueChainModel
from src.streamlit.value_chain import ModelConfigurationDict, UserEconomicsSummaryDict
from src.streamlit.value_chain.sidebar import build_sidebar
from src.streamlit.value_chain.tab_depreciation import render_depreciation_tab
from src.streamlit.value_chain.tab_funding import render_funding_tab
from src.streamlit.value_chain.tab_market import (
    render_market_tab,
)


def _get_target_market_segments(market_label: str) -> pd.DataFrame:
    """Map market scenario label to predefined market segments."""
    scenario = ALL_MARKET_SCENARIOS.get(market_label)
    return scenario["segments"]


@dataclass
class ValueChainAnalysisResults:
    """Validated outputs returned by the value chain model."""

    timeline: pd.DataFrame
    cohort_breakdown: pd.DataFrame
    user_economics_summary: UserEconomicsSummaryDict
    model_configuration: ModelConfigurationDict
    capex_depreciation_summary: dict
    interpolated_revenue_timeline: Optional[pd.DataFrame] = None


st.set_page_config(
    page_title="ROAI: Value Chain Analysis",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _create_custom_depreciation_schedule(
    chips_useful_life: int,
    datacenter_useful_life: int,
    power_useful_life: int,
    custom_scenario_params: dict,
) -> CapexDepreciationSchedule:
    """Generate depreciation schedule from custom growth parameters."""

    # Always generate new schedule from custom growth parameters
    schedule = generate_capex_schedule_df(
        nvda_rev_in_start_year=custom_scenario_params["nvda_start_revenue"],
        growth_rate=custom_scenario_params["growth_rates"],
        start_year=custom_scenario_params["start_year"],
        end_year=custom_scenario_params["end_year"],
        chips_life=chips_useful_life,
        datacenter_life=datacenter_useful_life,
        power_life=power_useful_life
    )

    return schedule


def _run_value_chain_analysis(
    start_market_segments: pd.DataFrame,
    target_market_segments: pd.DataFrame,
    depreciation_schedule: CapexDepreciationSchedule,
    cloud_margin: float,
    model_margin: float,
    app_margin: float,
    chip_vendor_margin: float,
) -> tuple[ValueChainAnalysisResults, 'ValueChainModel']:
    """Execute the value chain model and return validated results with model."""

    model = ValueChainModel(
        start_market_segments=start_market_segments,
        target_market_segments=target_market_segments,
        depreciation_schedule=depreciation_schedule,
        base_markups=make_markups_from_margins(cloud_margin, model_margin, app_margin),
        toggles=None,
        chip_vendor_margin=chip_vendor_margin,
    )

    raw_results = model.run_full_analysis()

    results = ValueChainAnalysisResults(
        timeline=raw_results["timeline"],
        cohort_breakdown=raw_results["cohort_breakdown"],
        user_economics_summary=raw_results["user_economics_summary"],
        model_configuration=raw_results["model_configuration"],
        capex_depreciation_summary=raw_results["capex_depreciation_summary"],
        interpolated_revenue_timeline=None,  # Now calculated in display layer
    )

    return results, model


def _render_analysis_sections(
    results: ValueChainAnalysisResults,
    depreciation_schedule: CapexDepreciationSchedule,
    investment_label: str,
    market_label: str,
    chips_useful_life: int,
    datacenter_useful_life: int,
    power_useful_life: int,
    model: 'ValueChainModel',
    custom_scenario_params: dict,
    fcf_growth_rate: float,
) -> None:
    """Render sections without tabs to test scroll behavior."""

    # Use user-controlled display end year
    max_display_year = custom_scenario_params["display_end_year"]

    render_depreciation_tab(
        depreciation_schedule,
        investment_label,
        chips_useful_life,
        datacenter_useful_life,
        power_useful_life,
        timeline=results.timeline,
        config=results.model_configuration,
        max_display_year=max_display_year,
    )

    # Calculate interpolated timeline using display years (not depreciation years)

    # Get first year with depreciation to start interpolation
    timeline_with_depreciation = results.timeline[results.timeline["depreciation"] > 0]
    if not timeline_with_depreciation.empty:
        first_year = int(timeline_with_depreciation["year"].min())

        # Use static method directly with proper display years
        interpolated_segments = model.interpolate_market_segments_over_time(
            start_segments=TODAY_MARKET_SEGMENTS,
            end_segments=model.target_market_segments,
            start_year=first_year,
            end_year=max_display_year
        )
        interpolated_timeline = model._calculate_revenue_from_interpolated_segments(interpolated_segments)
    else:
        interpolated_timeline = None

    render_market_tab(
        results.cohort_breakdown,
        results.user_economics_summary,
        market_label,
        interpolated_timeline=interpolated_timeline,
        start_scenario_name="Today",
        target_scenario_name=market_label,
        timeline=results.timeline
    )

    render_funding_tab(model, interpolated_timeline, fcf_growth_rate=fcf_growth_rate)


def main() -> None:
    """Entry point for the Streamlit dashboard."""

    st.title("Value Chain Analysis Dashboard")
    st.markdown(
        "- How much revenue must the ecosystem generate to pay back infrastructure spend?\n"
        "- What assumptions (users cohorts, adoption, ARPU) are needed to reach that revenue?\n"
        "- Pricing power across the value chain (GPUs → cloud → model → app): how can we expect that to evolve?"
    )

    # Simplified sidebar without session state management
    config, custom_segments, investment_label, market_label, custom_scenario_params = build_sidebar(existing_config=None)

    # Always use custom scenario generation
    depreciation_schedule = _create_custom_depreciation_schedule(
        chips_useful_life=config.chips_useful_life,
        datacenter_useful_life=config.datacenter_useful_life,
        power_useful_life=config.power_useful_life,
        custom_scenario_params=custom_scenario_params,
    )

    # Apply chip pricing adjustments to the depreciation schedule
    depreciation_schedule = depreciation_schedule.with_adjusted_chip_prices(config.chip_gross_margin)

    # Use custom_segments from sidebar (contains user-edited TAM, ARPU, cohort share)
    # Both start and target use custom_segments for pie chart and revenue calculations
    results, model = _run_value_chain_analysis(
        start_market_segments=custom_segments,  # Used for display/cohort breakdown
        target_market_segments=custom_segments,  # End point for interpolation
        depreciation_schedule=depreciation_schedule,
        cloud_margin=config.cloud_margin,
        model_margin=config.model_margin,
        app_margin=config.app_margin,
        chip_vendor_margin=config.chip_gross_margin,
    )

    _render_analysis_sections(
        results,
        depreciation_schedule,
        investment_label,
        market_label,
        config.chips_useful_life,
        config.datacenter_useful_life,
        config.power_useful_life,
        model,
        custom_scenario_params,
        config.fcf_growth_rate,
    )


if __name__ == "__main__":  # pragma: no cover - Streamlit entrypoint
    main()
