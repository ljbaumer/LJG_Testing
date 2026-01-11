"""Market and revenue analysis tab utilities for the Value Chain app."""

from typing import Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

import streamlit as st
from src.constants.value_chain_market_segments import ALL_MARKET_SCENARIOS
from src.streamlit.value_chain import (
    UserEconomicsSummaryDict,
    apply_value_chain_theme,
    get_segment_display_name,
)
from src.utils.streamlit_app_helpers import calculate_chart_tick_intervals, format_number_to_string


def _create_cohort_pie_chart(cohort_breakdown: pd.DataFrame, total_revenue: float):
    df = cohort_breakdown.copy()
    df = df[df["annual_revenue"] > 0]

    def normalize_cohort_name(name: str) -> str:
        if "Free" in name:
            return "Consumer Free"
        if name in ["Consumer Paid", "APAC+ROW Paid"]:
            return "Consumer Paid"
        return name

    df["normalized_cohort"] = df["cohort_name"].apply(normalize_cohort_name)
    consolidated = df.groupby("normalized_cohort").agg({"annual_revenue": "sum"}).reset_index()

    revenue_str = format_number_to_string(total_revenue, is_currency=True, escape_markdown=False)
    chart_title = f"Eventual Annual Revenue: {revenue_str}"

    fig = px.pie(consolidated, values="annual_revenue", names="normalized_cohort", title=chart_title)
    fig.update_traces(
        textposition="inside",
        textinfo="percent+label",
        hovertemplate="<b>%{label}</b><br>Revenue: %{customdata}<br>Percentage: %{percent}<extra></extra>",
        customdata=[format_number_to_string(rev) for rev in consolidated["annual_revenue"]],
    )
    apply_value_chain_theme(
        fig,
        height=400,
        legend={
            "orientation": "v",
            "yanchor": "middle",
            "y": 0.5,
            "xanchor": "left",
            "x": 1.05,
        },
        margin={"l": 40, "r": 140, "t": 60, "b": 40},
    )
    return fig




def _render_scenario_header(scenario_label: Optional[str]) -> None:
    """Display scenario name and description."""

    if scenario_label and scenario_label in ALL_MARKET_SCENARIOS:
        scenario_info = ALL_MARKET_SCENARIOS[scenario_label]
        description = scenario_info["description"]

        st.header(f"Projecting Revenue in the {scenario_label}")
        st.markdown(f"{description}")


def _create_market_interpolation_chart(
    interpolated_timeline: pd.DataFrame,
    start_scenario_name: str,
    target_scenario_name: str
) -> go.Figure:
    """Create stacked bar chart showing revenue growth over time.

    Args:
        interpolated_timeline: DataFrame with year, segment, annual_revenue columns
        start_scenario_name: Name of starting scenario (e.g., "Today")
        target_scenario_name: Name of target scenario (e.g., "Bull Case")
    """
    # Pivot data for stacking
    pivot_df = interpolated_timeline.pivot(
        index='year',
        columns='segment',
        values='annual_revenue'
    ).fillna(0)

    # Create stacked bar chart
    fig = go.Figure()

    # Define colors for each segment
    segment_colors = {
        'us_canada_europe_consumers': '#1f77b4',
        'apac_row_consumers': '#ff7f0e',
        'professional': '#2ca02c',
        'programmer': '#d62728'
    }

    # Order segments by total revenue (biggest at bottom for better stacking)
    segment_totals = pivot_df.sum().sort_values(ascending=True)  # Smallest first = bottom of stack
    ordered_segments = segment_totals.index

    # Add bars for each segment in revenue-ordered sequence
    for segment in ordered_segments:
        display_name = get_segment_display_name(segment)
        fig.add_trace(go.Bar(
            name=display_name,
            x=pivot_df.index,
            y=pivot_df[segment],
            marker_color=segment_colors.get(segment, '#666666'),
            hovertemplate=(
                f"<b>{display_name}</b><br>"
                "Year: %{x}<br>"
                "Revenue: $%{y:,.0f}<br>"
                "<extra></extra>"
            )
        ))

    # Calculate total revenue per year for display
    total_revenue_by_year = pivot_df.sum(axis=1)
    max_value = total_revenue_by_year.max() if not total_revenue_by_year.empty else 1

    # Create clean tick intervals using format_number_to_string
    tick_values = calculate_chart_tick_intervals(max_value)
    tick_texts = [format_number_to_string(val, is_currency=True, escape_markdown=False) for val in tick_values]

    # Update layout
    fig.update_layout(
        title=f"Revenue Trajectory: {start_scenario_name} → {target_scenario_name}",
        xaxis_title="",
        yaxis_title="Annual Revenue ($)",
        barmode='stack',
        hovermode='x unified',
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.15,
            xanchor="center",
            x=0.5
        ),
        yaxis=dict(
            tickmode="array",
            tickvals=tick_values,
            ticktext=tick_texts,
            gridcolor='rgba(128, 128, 128, 0.2)'
        ),
        xaxis=dict(
            dtick=1,  # Show every year
            tickformat='d'  # Display as integer
        )
    )

    # Extend y-axis range to prevent text cutoff (15% margin)
    fig.update_yaxes(range=[0, max_value * 1.15])

    # Add total revenue labels on top of bars
    fig.add_trace(
        go.Scatter(
            x=pivot_df.index,
            y=total_revenue_by_year * 1.05,  # Position above the bars
            mode="text",
            text=[format_number_to_string(total, is_currency=True, escape_markdown=False) for total in total_revenue_by_year],
            textposition="top center",
            textfont={"size": 14, "color": "#37474f"},
            showlegend=False,
            hoverinfo="skip",
        )
    )

    apply_value_chain_theme(fig)
    return fig


def _create_revenue_and_net_chart(
    interpolated_timeline: pd.DataFrame,
    timeline: pd.DataFrame,
    start_scenario_name: str,
    target_scenario_name: str
) -> go.Figure:
    """Create stacked bar chart showing revenue breakdown: costs vs net profit.

    Args:
        interpolated_timeline: DataFrame with year, segment, annual_revenue columns
        timeline: DataFrame with year, depreciation, required_app_revenue_for_margins columns
        start_scenario_name: Name of starting scenario (e.g., "Today")
        target_scenario_name: Name of target scenario (e.g., "Bull Case")
    """
    # Get revenue by year from interpolated timeline
    revenue_by_year = interpolated_timeline.groupby('year')['annual_revenue'].sum().reset_index()
    revenue_by_year = revenue_by_year.rename(columns={'annual_revenue': 'total_revenue'})

    # Get depreciation by year from timeline
    depreciation_by_year = timeline[['year', 'depreciation']].copy()

    # Merge revenue and depreciation
    merged_df = pd.merge(revenue_by_year, depreciation_by_year, on='year', how='left').fillna(0)

    # Calculate net profit (revenue - depreciation)
    merged_df['net_profit'] = merged_df['total_revenue'] - merged_df['depreciation']

    # Create the chart
    fig = go.Figure()

    # For each year, we want to show:
    # 1. Depreciation (bottom part) - in blue
    # 2. Net profit/loss (top part) - green if positive, red if negative

    # First, add the depreciation bars (bottom of stack)
    fig.add_trace(go.Bar(
        name='Annualized Depreciation',
        x=merged_df['year'],
        y=merged_df['depreciation'],
        marker_color='#1f77b4',  # Blue for costs (same as consumer segment)
        hovertemplate=(
            "<b>Annualized Depreciation</b><br>"
            "Year: %{x}<br>"
            "Costs: %{customdata}<br>"
            "<extra></extra>"
        ),
        customdata=[format_number_to_string(cost, is_currency=True) for cost in merged_df['depreciation']]
    ))

    # Add gross profit bars (only for positive values)
    positive_profit_df = merged_df[merged_df['net_profit'] > 0]
    if not positive_profit_df.empty:
        fig.add_trace(go.Bar(
            name='Gross Profit',
            x=positive_profit_df['year'],
            y=positive_profit_df['net_profit'],
            marker_color='#2ca02c',  # Green for profit
            hovertemplate=(
                "<b>Gross Profit</b><br>"
                "Year: %{x}<br>"
                "Gross: %{customdata[0]}<br>"
                "Revenue: %{customdata[1]}<br>"
                "Costs: %{customdata[2]}<br>"
                "<extra></extra>"
            ),
            customdata=list(zip(
                [format_number_to_string(net, is_currency=True) for net in positive_profit_df['net_profit']],
                [format_number_to_string(rev, is_currency=True) for rev in positive_profit_df['total_revenue']],
                [format_number_to_string(cost, is_currency=True) for cost in positive_profit_df['depreciation']]
            ))
        ))

    # Add net loss bars (only for negative values)
    negative_profit_df = merged_df[merged_df['net_profit'] < 0]
    if not negative_profit_df.empty:
        fig.add_trace(go.Bar(
            name='Net Loss',
            x=negative_profit_df['year'],
            y=negative_profit_df['net_profit'],  # Negative values will show below zero
            marker_color='#d62728',  # Red for loss
            hovertemplate=(
                "<b>Net Loss</b><br>"
                "Year: %{x}<br>"
                "Net: %{customdata[0]}<br>"
                "Revenue: %{customdata[1]}<br>"
                "Costs: %{customdata[2]}<br>"
                "<extra></extra>"
            ),
            customdata=list(zip(
                [format_number_to_string(net, is_currency=True) for net in negative_profit_df['net_profit']],
                [format_number_to_string(rev, is_currency=True) for rev in negative_profit_df['total_revenue']],
                [format_number_to_string(cost, is_currency=True) for cost in negative_profit_df['depreciation']]
            ))
        ))

    # Calculate net profit/loss for display
    net_profit_loss = merged_df['total_revenue'] - merged_df['depreciation']

    # For shortfall chart, Y-axis should be based on:
    # - Maximum: the larger of max depreciation or max net profit
    # - Minimum: the most negative net loss
    max_depreciation = merged_df['depreciation'].max() if not merged_df.empty else 1
    max_net_profit = net_profit_loss.max() if not net_profit_loss.empty else 0
    max_value = max(max_depreciation, max_net_profit)
    min_value = net_profit_loss.min() if not net_profit_loss.empty else 0

    # Use tick interval calculation based on the max value (like revenue chart)
    import math

    # Calculate tick interval using helper
    tick_values_positive = calculate_chart_tick_intervals(max_value)
    tick_interval = tick_values_positive[1] if len(tick_values_positive) > 1 else max_value / 8

    # Create ticks from min_value to max_value
    min_tick = math.floor(min_value / tick_interval) * tick_interval
    max_tick = math.ceil(max_value / tick_interval) * tick_interval

    tick_values = []
    current = min_tick
    while current <= max_tick:
        tick_values.append(current)
        current += tick_interval

    # Ensure zero is included if not already present
    if 0 not in tick_values:
        tick_values.append(0)
        tick_values.sort()

    tick_texts = [format_number_to_string(val, is_currency=True, escape_markdown=False) for val in tick_values]

    # Update layout
    fig.update_layout(
        title=f"Shortfall / Gross Profit on Annual Depreciation: {start_scenario_name} → {target_scenario_name}",
        xaxis_title="",
        yaxis_title="Amount ($)",
        barmode='relative',  # Allows negative bars to show below zero
        hovermode='x unified',
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.15,
            xanchor="center",
            x=0.5
        ),
        yaxis=dict(
            tickmode="array",
            tickvals=tick_values,
            ticktext=tick_texts,
            gridcolor='rgba(128, 128, 128, 0.2)',
            zeroline=True,
            zerolinecolor='rgba(128, 128, 128, 0.5)',
            zerolinewidth=2
        ),
        xaxis=dict(
            dtick=1,  # Show every year
            tickformat='d'  # Display as integer
        )
    )

    # Extend y-axis range to prevent text cutoff (15% margin above max positive value)
    fig.update_yaxes(range=[min_value * 1.05, max_value * 1.15])

    # Position labels appropriately: above total bar for profits, above depreciation for losses
    y_positions = []
    for dep, net in zip(merged_df['depreciation'], net_profit_loss):
        if net >= 0:
            # Positive: position above the total bar (depreciation + net_profit)
            y_positions.append((dep + net) * 1.05)
        else:
            # Negative: position above the depreciation bar
            y_positions.append(dep * 1.05)

    fig.add_trace(
        go.Scatter(
            x=merged_df['year'],
            y=y_positions,
            mode="text",
            text=[format_number_to_string(net, is_currency=True, escape_markdown=False) if net >= 0 else f"({format_number_to_string(abs(net), is_currency=True, escape_markdown=False)})" for net in net_profit_loss],
            textposition="top center",
            textfont={"size": 14, "color": "#37474f"},
            showlegend=False,
            hoverinfo="skip",
        )
    )

    apply_value_chain_theme(fig)
    return fig


def render_market_tab(
    cohort_breakdown: pd.DataFrame,
    user_economics: UserEconomicsSummaryDict,
    scenario_label: Optional[str] = None,
    interpolated_timeline: Optional[pd.DataFrame] = None,
    start_scenario_name: Optional[str] = None,
    target_scenario_name: Optional[str] = None,
    timeline: Optional[pd.DataFrame] = None,
) -> None:
    """Render the market segment analysis tab."""

    total_revenue = cohort_breakdown["annual_revenue"].sum()

    _render_scenario_header(scenario_label)

    pie_chart = _create_cohort_pie_chart(cohort_breakdown, total_revenue)
    st.plotly_chart(pie_chart, width="stretch")


        # Add interpolation chart at the bottom
    if interpolated_timeline is not None and not interpolated_timeline.empty:
        interpolation_chart = _create_market_interpolation_chart(
            interpolated_timeline,
            start_scenario_name or "Start",
            target_scenario_name or "Target"
        )
        st.plotly_chart(interpolation_chart, use_container_width=True)

        # Add revenue and net profit/loss chart underneath
        if timeline is not None and not timeline.empty:
            revenue_net_chart = _create_revenue_and_net_chart(
                interpolated_timeline,
                timeline,
                start_scenario_name or "Start",
                target_scenario_name or "Target"
            )
            st.plotly_chart(revenue_net_chart, use_container_width=True)


__all__ = ["render_market_tab"]
