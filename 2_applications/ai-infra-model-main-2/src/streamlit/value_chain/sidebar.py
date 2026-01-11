"""Sidebar configuration and session-state helpers for the Value Chain app.

The sidebar is the primary control surface for the value chain analysis workflow.
This module keeps the Streamlit rendering concerns isolated so that the rest of the
application can reason about a clean, typed configuration object.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import pandas as pd

import streamlit as st
from src.constants.value_chain_depreciation_schedules import (
    NVIDIA_COMPUTE_SHARE,
    NVIDIA_DEFAULT_GROSS_MARGIN,
)
from src.constants.value_chain_market_segments import ALL_MARKET_SCENARIOS
from src.constants.value_chain_markups import (
    DEFAULT_APP_MARGIN,
    DEFAULT_CLOUD_MARGIN,
    DEFAULT_MODEL_MARGIN,
)
from src.streamlit.value_chain import get_segment_display_name

DEFAULT_MARKET_SCENARIO_KEY = "Total Adoption Case"
COMPUTE_COST_ANCHOR = 100.0


@dataclass
class SidebarConfig:
    """Typed container describing the selections made in the sidebar."""

    market_cohort_scenario: str
    chips_useful_life: int
    datacenter_useful_life: int
    power_useful_life: int
    cloud_margin: float
    model_margin: float
    app_margin: float
    chip_gross_margin: float
    fcf_growth_rate: float


def build_default_sidebar_config() -> SidebarConfig:
    """Return default sidebar configuration aligned with initial UI selections."""

    if DEFAULT_MARKET_SCENARIO_KEY not in ALL_MARKET_SCENARIOS:
        raise KeyError(
            f"Default market scenario '{DEFAULT_MARKET_SCENARIO_KEY}' not found in ALL_MARKET_SCENARIOS."
        )

    return SidebarConfig(
        market_cohort_scenario=DEFAULT_MARKET_SCENARIO_KEY,
        chips_useful_life=5,  # Default chip useful life
        datacenter_useful_life=20,  # Default datacenter useful life
        power_useful_life=25,  # Default power useful life
        cloud_margin=DEFAULT_CLOUD_MARGIN,
        model_margin=DEFAULT_MODEL_MARGIN,
        app_margin=DEFAULT_APP_MARGIN,
        chip_gross_margin=NVIDIA_DEFAULT_GROSS_MARGIN,
        fcf_growth_rate=0.08,  # Default 8% FCF growth
    )


def _create_sidebar_controls_for_segment_df(segment_df: pd.DataFrame, tam: int, scenario_slug: str) -> pd.DataFrame:
    """Create editable controls for a single segment DataFrame and return the override DataFrame."""

    rows = []
    segment_name = segment_df['segment'].iloc[0]  # Should be same for all cohorts

    for _, row in segment_df.iterrows():
        cohort_name = row["cohort_name"]

        st.write(f"**{cohort_name}**")

        col1, col2 = st.columns(2)
        with col1:
            cohort_share = (
                st.number_input(
                    "% of TAM",
                    min_value=0.0,
                    max_value=100.0,
                    value=float(row["cohort_share"] * 100),
                    step=0.1,
                    format="%.1f",
                    key=f"{segment_name}_{cohort_name}_share_{scenario_slug}",
                )
                / 100.0
            )

        with col2:
            arpu = st.number_input(
                "ARPU ($/month)",
                min_value=0.0,
                value=float(row["arpu"]),
                step=0.1,
                format="%.2f",
                key=f"{segment_name}_{cohort_name}_arpu_{scenario_slug}",
            )

        rows.append(
            {
                "cohort_name": cohort_name,
                "cohort_share": cohort_share,
                "arpu": arpu,
                "cost_to_service": row["cost_to_service"],
                "segment": segment_name,
                "total_addressable_users": tam,
            }
        )

    return pd.DataFrame(rows)


def _create_sidebar_controls_for_all_segments(
    scenario_key: str,
    market_segments: pd.DataFrame,
) -> pd.DataFrame:
    """Collect overrides for every segment in the selected scenario."""

    segment_overrides = []
    scenario_slug = _slugify_scenario_key(scenario_key)

    # Process each unique segment
    for segment_name in market_segments['segment'].unique():
        segment_df = market_segments[market_segments['segment'] == segment_name]
        tam = segment_df['total_addressable_users'].iloc[0]  # Should be same for all cohorts
        display_name = get_segment_display_name(segment_name)

        with st.sidebar.expander(display_name, expanded=False):
            tam_millions = st.number_input(
                "TAM (Millions)",
                min_value=1,
                value=tam // 1_000_000,
                step=1,
                format="%d",
                key=f"{segment_name}_tam_{scenario_slug}",
            )
            tam_actual = tam_millions * 1_000_000
            st.markdown("---")
            custom_df = _create_sidebar_controls_for_segment_df(segment_df, tam_actual, scenario_slug)
            segment_overrides.append(custom_df)

    # Concatenate all segment DataFrames
    if segment_overrides:
        return pd.concat(segment_overrides, ignore_index=True)
    else:
        # Return empty DataFrame with expected columns if no overrides
        return pd.DataFrame(columns=['segment', 'cohort_name', 'cohort_share', 'arpu', 'cost_to_service', 'total_addressable_users'])


def _render_sidebar_custom_growth_section() -> dict:
    """Render custom growth scenario controls and return parameters."""

    nvda_start_revenue = st.sidebar.number_input(
        "NVIDIA Revenue in Start Year ($B)",
        min_value=50.0,
        max_value=1000.0,
        value=161.25,  # Default to $161.25B (gives $215B total chip capex like Zero Growth)
        step=1.0,
        help="Starting NVIDIA datacenter revenue"
    ) * 1_000_000_000  # Convert to actual dollars

    col1, col2 = st.sidebar.columns(2)
    with col1:
        start_year = st.sidebar.selectbox(
            "Start Year",
            options=[2024, 2025, 2026],
            index=1,  # Default to 2025
            help="First year of investment"
        )

    with col2:
        end_year = st.sidebar.selectbox(
            "End Year",
            options=list(range(start_year + 4, start_year + 8)),
            index=2,  # Default to start_year + 6
            help="Last year of investment"
        )

    # Variable growth rates section in collapsible expander
    with st.sidebar.expander("Annual Growth Rates", expanded=False):
        st.write("Set growth rate for each year (applied to following year):")
        growth_rates = {}
        # Default growth rates by year
        default_growth_by_year = {
            2026: 37.0,
            2027: 16.0,
            2028: 12.0,
            2029: 8.0,
            2030: 6.0,
            2031: 5.0,
        }
        # Skip start_year since it's the base year (no growth applied)
        for year in range(start_year + 1, end_year + 1):
            default_value = default_growth_by_year.get(year, 5.0)  # Fall back to 5% for years beyond 2031
            growth_rate_pct = st.number_input(
                f"{year} Growth Rate (%)",
                min_value=-20.0,
                max_value=50.0,
                value=default_value,
                step=1.0,
                format="%.1f",
                help=f"Growth rate applied from {year-1} to {year}",
                key=f"growth_rate_{year}"
            )
            growth_rates[year] = growth_rate_pct / 100  # Convert to decimal

    st.sidebar.markdown("---")
    st.sidebar.write("**Display Range**")

    col3, col4 = st.sidebar.columns(2)
    with col3:
        display_start_year = st.sidebar.number_input(
            "Display Start",
            min_value=start_year - 5,
            max_value=end_year,
            value=start_year,
            step=1,
            help="First year to display in charts"
        )

    with col4:
        display_end_year = st.sidebar.number_input(
            "Display End",
            min_value=start_year + 1,
            max_value=end_year + 20,
            value=start_year + 6,
            step=1,
            help="Last year to display in charts"
        )

    # Revenue preview removed per user request

    return {
        "nvda_start_revenue": nvda_start_revenue,
        "growth_rates": growth_rates,
        "start_year": start_year,
        "end_year": end_year,
        "display_start_year": display_start_year,
        "display_end_year": display_end_year
    }




def _render_sidebar_useful_life_section(seeded_config: SidebarConfig) -> Tuple[int, int, int]:
    """Collect user overrides for useful life assumptions."""

    st.sidebar.subheader("Depreciation Assumptions")

    chips_useful_life = st.sidebar.number_input(
        "Chips Useful Life (years)",
        value=int(seeded_config.chips_useful_life),
        min_value=1,
        help="Number of years for chip depreciation",
    )

    datacenter_useful_life = st.sidebar.number_input(
        "Datacenter Useful Life (years)",
        value=int(seeded_config.datacenter_useful_life),
        min_value=1,
        help="Number of years for datacenter infrastructure depreciation",
    )

    power_useful_life = st.sidebar.number_input(
        "Power Useful Life (years)",
        value=int(seeded_config.power_useful_life),
        min_value=1,
        help="Number of years for power infrastructure depreciation",
    )

    return int(chips_useful_life), int(datacenter_useful_life), int(power_useful_life)


def _render_sidebar_margin_section(seeded_config: SidebarConfig) -> Tuple[float, float, float]:
    """Collect pricing power adjustments for each layer of the stack."""

    st.sidebar.subheader("Base Margins")
    st.sidebar.write("Adjust pricing power at each layer:")

    cloud_margin = (
        st.sidebar.number_input(
            "Cloud Margin (%)",
            value=float(seeded_config.cloud_margin * 100),
            min_value=0.0,
            max_value=100.0,
            help="Margin percentage for cloud layer",
        )
        / 100.0
    )

    model_margin = (
        st.sidebar.number_input(
            "Model Margin (%)",
            value=float(seeded_config.model_margin * 100),
            min_value=0.0,
            max_value=100.0,
            help="Margin percentage for model layer",
        )
        / 100.0
    )

    app_margin = (
        st.sidebar.number_input(
            "App Margin (%)",
            value=float(seeded_config.app_margin * 100),
            min_value=0.0,
            max_value=100.0,
            help="Margin percentage for app layer",
        )
        / 100.0
    )

    return float(cloud_margin), float(model_margin), float(app_margin)


def _render_sidebar_chip_vendor_section(seeded_config: SidebarConfig) -> float:
    """Render chip vendor margin controls and display delta messaging."""

    st.sidebar.subheader("Chip Vendor Pricing")
    st.sidebar.write("Adjust chip vendor gross margin:")

    chip_gross_margin = (
        st.sidebar.number_input(
            "Chip Vendor Gross Margin (%)",
            value=float(seeded_config.chip_gross_margin * 100),
            min_value=0.0,
            max_value=100.0,
            step=1.0,
            help="NVIDIA's gross margin on datacenter GPUs. Lower margins → cheaper chips → lower infrastructure costs throughout value chain",
        )
        / 100.0
    )

    def _format_currency(value: float, decimals: int = 2) -> str:
        """Return markdown-safe currency string."""
        return f"\\${value:,.{decimals}f}"

    old_margin_complement = 1.0 - NVIDIA_DEFAULT_GROSS_MARGIN
    new_margin_complement = max(0.001, 1.0 - chip_gross_margin)
    price_multiplier_nvidia = old_margin_complement / new_margin_complement

    baseline_total_cost = COMPUTE_COST_ANCHOR
    nvidia_share = NVIDIA_COMPUTE_SHARE
    baseline_nvidia_component = baseline_total_cost * nvidia_share
    baseline_other_component = baseline_total_cost * (1.0 - nvidia_share)

    adjusted_nvidia_component = baseline_nvidia_component * price_multiplier_nvidia
    adjusted_total_cost = adjusted_nvidia_component + baseline_other_component
    total_multiplier = adjusted_total_cost / baseline_total_cost if baseline_total_cost else 1.0

    total_change_pct = (total_multiplier - 1.0) * 100
    nvidia_change_pct = (price_multiplier_nvidia - 1.0) * 100

    if chip_gross_margin != NVIDIA_DEFAULT_GROSS_MARGIN:
        if total_multiplier < 1.0:
            st.sidebar.success(
                f"Compute stack is {abs(total_change_pct):.0f}% cheaper (NVIDIA hardware {abs(nvidia_change_pct):.0f}%)."
            )
        else:
            st.sidebar.warning(
                f"Compute stack is {total_change_pct:.0f}% more expensive (NVIDIA hardware {nvidia_change_pct:.0f}%)."
            )

    impact_expanded = chip_gross_margin != NVIDIA_DEFAULT_GROSS_MARGIN
    with st.sidebar.expander("Compute Pricing Math", expanded=impact_expanded):
        st.markdown(
            (
                f"**Assumed compute budget:** {_format_currency(baseline_total_cost, decimals=0)}\n"
                f"- NVIDIA share ({nvidia_share:.0%}): {_format_currency(baseline_nvidia_component)} → {_format_currency(adjusted_nvidia_component)} "
                f"({nvidia_change_pct:+.1f}%)\n"
                f"- Non-NVIDIA integration ({(1.0 - nvidia_share):.0%}): {_format_currency(baseline_other_component)} (unchanged)\n"
                f"- Total compute spend: {_format_currency(baseline_total_cost)} → {_format_currency(adjusted_total_cost)} "
                f"({total_change_pct:+.1f}%)\n"
            )
        )

    return float(chip_gross_margin)


def _render_sidebar_fcf_growth_section(seeded_config: SidebarConfig) -> float:
    """Render FCF growth rate control for funding analysis."""

    st.sidebar.subheader("Funding Assumptions")

    fcf_growth_rate = (
        st.sidebar.number_input(
            "Annual Non-AI FCF Growth (%)",
            value=float(seeded_config.fcf_growth_rate * 100),
            min_value=0.0,
            max_value=50.0,
            step=1.0,
            help="Annual growth rate for hyperscaler non-AI free cash flow from 2025 baseline",
        )
        / 100.0
    )

    return float(fcf_growth_rate)


def _render_sidebar_market_segment_section(
    seeded_config: SidebarConfig,
) -> Tuple[str, pd.DataFrame]:
    """Render market segment controls and return scenario key plus overrides."""

    st.sidebar.subheader("Market Segments")
    cohort_keys = list(ALL_MARKET_SCENARIOS.keys())
    selected_index = cohort_keys.index(seeded_config.market_cohort_scenario)

    selected_key = st.sidebar.selectbox(
        "Select Market Cohort Scenario:",
        options=cohort_keys,
        index=selected_index,
        help="Choose user adoption and revenue assumptions across market segments (separate from investment scenarios)",
    )

    scenario_info = ALL_MARKET_SCENARIOS[selected_key]

    custom_segments = _create_sidebar_controls_for_all_segments(selected_key, scenario_info["segments"])
    return selected_key, custom_segments


def build_sidebar(existing_config: Optional[SidebarConfig] = None) -> Tuple[SidebarConfig, pd.DataFrame, str, str, dict]:
    """Render the sidebar and return user selections."""

    st.sidebar.header("Value Chain Configuration")
    seeded_config = existing_config or build_default_sidebar_config()

    # Always show custom growth scenario parameters
    custom_scenario_params = _render_sidebar_custom_growth_section()
    chips_useful_life, datacenter_useful_life, power_useful_life = _render_sidebar_useful_life_section(seeded_config)
    cloud_margin, model_margin, app_margin = _render_sidebar_margin_section(seeded_config)
    chip_gross_margin = _render_sidebar_chip_vendor_section(seeded_config)
    fcf_growth_rate = _render_sidebar_fcf_growth_section(seeded_config)
    selected_market_key, custom_segments = _render_sidebar_market_segment_section(seeded_config)

    updated_config = SidebarConfig(
        market_cohort_scenario=selected_market_key,
        chips_useful_life=chips_useful_life,
        datacenter_useful_life=datacenter_useful_life,
        power_useful_life=power_useful_life,
        cloud_margin=cloud_margin,
        model_margin=model_margin,
        app_margin=app_margin,
        chip_gross_margin=chip_gross_margin,
        fcf_growth_rate=fcf_growth_rate,
    )

    # Return "Custom Growth Scenario" as the investment label for consistency
    return updated_config, custom_segments, "Custom Growth Scenario", selected_market_key, custom_scenario_params


__all__ = [
    "SidebarConfig",
    "build_default_sidebar_config",
    "build_sidebar",
]
def _slugify_scenario_key(key: str) -> str:
    """Create a stable slug for scenario keys when building Streamlit widget ids."""

    return key.lower().replace(" ", "_")
