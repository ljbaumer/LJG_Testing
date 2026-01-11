"""Sidebar configuration for the labor value share dashboard."""

from __future__ import annotations

from dataclasses import dataclass

import streamlit as st
from src.constants.labor_share_global_assumptions import (
    BASELINE_GLOBAL_GDP_USD,
    BASELINE_GLOBAL_KNOWLEDGE_WORKERS,
    BASELINE_LABOR_SHARE_PERCENTAGES,
    BASELINE_TRADITIONAL_SOFTWARE_SPEND_USD,
    BASELINE_VALUE_CAPTURE,
    LaborSharePercentages,
    ValueCaptureBreakdown,
)


@dataclass
class SidebarConfig:
    """Collected user inputs from the sidebar."""

    global_gdp_usd: float
    total_knowledge_workers: float
    traditional_software_spend_usd: float
    labor_shares: LaborSharePercentages
    value_capture: ValueCaptureBreakdown


def _parse_numeric_text(value: str, *, label: str) -> float:
    """Parse numeric input from a text box or stop the app with an error if invalid."""
    cleaned = value.replace(",", "").strip()
    if cleaned == "":
        st.error(f"{label} cannot be empty. Please enter a number.")
        st.stop()
    try:
        return float(cleaned)
    except ValueError as exc:
        st.error(f"{label} must be a valid number. {exc}")
        st.stop()


def _parse_percentage_input(
    label: str,
    default_fraction: float,
    *,
    key: str,
    help_text: str | None = None,
    override_default: float | None = None,
) -> float:
    """Collect a percentage via text input (0-100) and return it as a fraction (0-1)."""
    if override_default is not None:
        default_fraction = override_default
    default_pct = f"{default_fraction * 100:.2f}"
    user_value = st.text_input(label, value=default_pct, key=key, help=help_text)
    percentage = _parse_numeric_text(user_value, label=label)
    fraction = percentage / 100.0
    if not 0.0 <= fraction <= 1.0:
        st.error(f"{label} must be between 0 and 100. Received {percentage:,.2f}.")
        st.stop()
    return fraction


def build_sidebar(existing_config: SidebarConfig | None = None) -> SidebarConfig:
    """Render the sidebar inputs and return the collected configuration."""

    default_global_gdp_trillions = (
        existing_config.global_gdp_usd / 1_000_000_000_000
        if existing_config
        else BASELINE_GLOBAL_GDP_USD / 1_000_000_000_000
    )

    default_knowledge_workers_millions = (
        existing_config.total_knowledge_workers / 1_000_000
        if existing_config
        else BASELINE_GLOBAL_KNOWLEDGE_WORKERS / 1_000_000
    )

    default_traditional_spend_billions = (
        existing_config.traditional_software_spend_usd / 1_000_000_000
        if existing_config
        else BASELINE_TRADITIONAL_SOFTWARE_SPEND_USD / 1_000_000_000
    )

    default_labor_shares = (
        existing_config.labor_shares
        if existing_config
        else BASELINE_LABOR_SHARE_PERCENTAGES
    )
    default_value_capture = (
        existing_config.value_capture
        if existing_config
        else BASELINE_VALUE_CAPTURE
    )

    with st.sidebar:
        st.header("Assumptions")

        gdp_input = st.text_input(
            "Global GDP (USD, trillions)",
            value=f"{default_global_gdp_trillions:.1f}",
            help="Nominal global GDP expressed in trillions of USD.",
            key="global_gdp_trillions",
        )
        global_gdp_trillions = _parse_numeric_text(gdp_input, label="Global GDP")
        global_gdp_usd = global_gdp_trillions * 1_000_000_000_000

        knowledge_workers_input = st.text_input(
            "Knowledge Workers (millions)",
            value=f"{default_knowledge_workers_millions:.1f}",
            help="Total global knowledge workers counted in millions.",
            key="knowledge_workers_millions",
        )
        total_knowledge_workers = (
            _parse_numeric_text(knowledge_workers_input, label="Knowledge Workers")
            * 1_000_000
        )

        traditional_spend_input = st.text_input(
            "Traditional Software Spend (USD, billions)",
            value=f"{default_traditional_spend_billions:.1f}",
            help="Annual legacy software/SaaS spend attributed to knowledge workers.",
            key="traditional_software_spend_billions",
        )
        traditional_software_spend_usd = (
            _parse_numeric_text(
                traditional_spend_input, label="Traditional Software Spend"
            )
            * 1_000_000_000
        )

        st.divider()
        st.subheader("Labor Share Structure")
        labor_share = _parse_percentage_input(
            "Labor Share of GDP (%)",
            BASELINE_LABOR_SHARE_PERCENTAGES.labor_share_of_gdp,
            key="labor_share_pct",
            help_text="Portion of GDP paid out as labor compensation.",
            override_default=default_labor_shares.labor_share_of_gdp,
        )
        knowledge_share = _parse_percentage_input(
            "Knowledge Work Share of Labor (%)",
            BASELINE_LABOR_SHARE_PERCENTAGES.knowledge_work_share,
            key="knowledge_share_pct",
            help_text="Share of labor compensation attributable to knowledge work.",
            override_default=default_labor_shares.knowledge_work_share,
        )
        ai_efficiency_gain = _parse_percentage_input(
            "AI Efficiency Gain (%)",
            BASELINE_LABOR_SHARE_PERCENTAGES.ai_efficiency_gain_pct,
            key="ai_efficiency_gain_pct",
            help_text="Productivity lift applied to knowledge worker compensation.",
            override_default=default_labor_shares.ai_efficiency_gain_pct,
        )

        st.divider()
        st.subheader("Value Capture Split")
        consumer_share = _parse_percentage_input(
            "Consumer Surplus Share (%)",
            BASELINE_VALUE_CAPTURE.consumer_surplus,
            key="consumer_share_pct",
            override_default=default_value_capture.consumer_surplus,
        )
        worker_share = _parse_percentage_input(
            "Worker Surplus Share (%)",
            BASELINE_VALUE_CAPTURE.worker_surplus,
            key="worker_share_pct",
            override_default=default_value_capture.worker_surplus,
        )
        revenue_uplift_share = _parse_percentage_input(
            "Revenue Uplift Share (%)",
            BASELINE_VALUE_CAPTURE.revenue_uplift,
            key="revenue_uplift_share_pct",
            override_default=default_value_capture.revenue_uplift,
        )
        cost_reduction_share = _parse_percentage_input(
            "Cost Reduction Share (%)",
            BASELINE_VALUE_CAPTURE.cost_reduction,
            key="cost_reduction_share_pct",
            override_default=default_value_capture.cost_reduction,
        )
        software_capture_share = _parse_percentage_input(
            "Software Capture Share (%)",
            BASELINE_VALUE_CAPTURE.software_capture,
            key="software_capture_share_pct",
            override_default=default_value_capture.software_capture,
        )

        total_capture_share = (
            consumer_share
            + worker_share
            + revenue_uplift_share
            + cost_reduction_share
            + software_capture_share
        )
        if not 0.99 <= total_capture_share <= 1.01:
            st.error(
                "Value capture shares must sum to 100%. "
                f"Current total: {total_capture_share * 100:.2f}%"
            )
            st.stop()

    labor_shares = LaborSharePercentages(
        labor_share_of_gdp=labor_share,
        knowledge_work_share=knowledge_share,
        ai_efficiency_gain_pct=ai_efficiency_gain,
    )
    value_capture = ValueCaptureBreakdown(
        consumer_surplus=consumer_share,
        worker_surplus=worker_share,
        revenue_uplift=revenue_uplift_share,
        cost_reduction=cost_reduction_share,
        software_capture=software_capture_share,
    )

    return SidebarConfig(
        global_gdp_usd=global_gdp_usd,
        total_knowledge_workers=total_knowledge_workers,
        traditional_software_spend_usd=traditional_software_spend_usd,
        labor_shares=labor_shares,
        value_capture=value_capture,
    )
