"""Depreciation timeline tab for the Value Chain Streamlit app."""

from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd
import plotly.graph_objects as go

import streamlit as st
from src.constants.value_chain_depreciation_schedules import CapexDepreciationSchedule
from src.streamlit.value_chain import (
    VALUE_CHAIN_DEPRECIATION_COLORS,
    VALUE_CHAIN_LAYER_COLORS,
    ModelConfigurationDict,
    apply_value_chain_theme,
)
from src.utils.streamlit_app_helpers import calculate_chart_tick_intervals, format_number_to_string

ASSET_TYPE_ALIASES = {
    "chip": "chips",
    "chips": "chips",
    "compute": "chips",
    "compute_hw": "chips",
    "datacenter": "datacenter",
    "data_center": "datacenter",
    "datacentre": "datacenter",
    "facility": "datacenter",
    "power": "power",
}

ASSET_DISPLAY_NAMES = {
    "chips": "Compute",
    "datacenter": "Datacenter",
    "power": "Power",
}

SCENARIO_COLOR_SCALES = {
    "Zero Growth": [
        [0.0, "#f8f9fa"],
        [0.001, "#e3f2fd"],
        [0.25, "#bbdefb"],
        [0.5, "#64b5f6"],
        [0.75, "#2196f3"],
        [1.0, "#1565c0"],
    ],
    "Continued Growth": [
        [0.0, "#f0f7ff"],
        [0.001, "#dce9ff"],
        [0.25, "#b6d4ff"],
        [0.5, "#73b3ff"],
        [0.75, "#438ef4"],
        [1.0, "#1f5fbf"],
    ],
    "Slowdown": [
        [0.0, "#f1f6ff"],
        [0.001, "#e0ecff"],
        [0.25, "#bcd6ff"],
        [0.5, "#89b4ff"],
        [0.75, "#4e87ef"],
        [1.0, "#2d5ec7"],
    ],
    "Consensus NVIDIA Growth": [
        [0.0, "#eef4ff"],
        [0.001, "#d5e3ff"],
        [0.25, "#adc6ff"],
        [0.5, "#7aa4ff"],
        [0.75, "#4f7de6"],
        [1.0, "#2651b3"],
    ],
}

DEFAULT_COLOR_SCALE = SCENARIO_COLOR_SCALES["Zero Growth"]

COLORBAR_TICKS = [0, 40_000_000_000, 80_000_000_000, 120_000_000_000, 160_000_000_000]
COLORBAR_TICKTEXT = ["$0", "$40B", "$80B", "$120B", "$160B"]
COLORBAR_MAX = COLORBAR_TICKS[-1]
GROUP_LABEL_COLOR = "#37474f"
HEATMAP_TEXT_SIZE = 12


def _recalculate_total_depreciation_by_year(
    depreciation_accounting_schedule: pd.DataFrame,
    useful_life_series: pd.Series,
) -> pd.Series:
    """Recalculate total depreciation by year using updated useful life values."""

    all_depreciation = []

    for asset_type in depreciation_accounting_schedule.columns:
        useful_life = int(useful_life_series[asset_type])
        asset_capex = depreciation_accounting_schedule[asset_type].dropna()

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


def _normalize_asset_type(asset_type: str) -> str:
    """Return canonical asset type key used across displays and color mapping."""

    normalized = asset_type.strip().lower()
    return ASSET_TYPE_ALIASES.get(normalized, normalized)


def _get_asset_display_name(asset_type: str) -> str:
    """Return human-readable display name for an asset type."""

    normalized = _normalize_asset_type(asset_type)
    return ASSET_DISPLAY_NAMES.get(normalized, asset_type.replace("_", " ").title())


def _strip_trailing_zero_suffix(text: str) -> str:
    """Remove trailing .0 from compact number strings (e.g., $40.0B → $40B)."""

    cleaned = text
    for suffix in ("B", "M", "K", "T"):
        cleaned = cleaned.replace(f".0{suffix}", f"{suffix}")
    if cleaned.endswith(".0"):
        cleaned = cleaned[:-2]
    return cleaned



def _format_capex_snapshot_title(
    total_capex: float,
    total_depreciation: float,
    start_year: int,
    end_year: int,
) -> str:
    """Return compact snapshot title for the depreciation gantt chart."""

    capex_display = format_number_to_string(total_capex, is_currency=True, escape_markdown=False)
    depreciation_display = format_number_to_string(total_depreciation, is_currency=True, escape_markdown=False)
    return f"{capex_display} CAPEX Deployed, {depreciation_display} Depreciation Recognized ({start_year}-{end_year})"


def _create_depreciation_matrix(
    depreciation_accounting_schedule: pd.DataFrame,
    useful_life_series: pd.Series,
    years_range: Sequence[int],
) -> Tuple[
    List[List[float]],
    List[List[str]],
    List[int],
    List[str],
    List[List[str]],
    Dict[str, Tuple[int, int]],
]:
    """Create heatmap inputs describing depreciation for each asset investment."""

    depreciation_matrix: List[List[float]] = []
    text_matrix: List[List[str]] = []
    hover_matrix: List[List[str]] = []
    axis_labels: List[str] = []
    group_bounds: Dict[str, Dict[str, int]] = {}

    for asset_type in depreciation_accounting_schedule.columns.tolist():
        useful_life = useful_life_series[asset_type]
        asset_capex = depreciation_accounting_schedule[asset_type]
        asset_display = _get_asset_display_name(asset_type)

        for investment_year in asset_capex.index:
            investment_amount = float(asset_capex[investment_year])
            if investment_amount <= 0:
                continue

            annual_depreciation = investment_amount / useful_life
            capex_display = format_number_to_string(investment_amount, is_currency=True, escape_markdown=False)
            row_depreciation: List[float] = []
            row_text: List[str] = []
            row_hover: List[str] = []

            for year in years_range:
                if investment_year <= year < investment_year + useful_life:
                    row_depreciation.append(annual_depreciation)
                    row_text.append(format_number_to_string(annual_depreciation, is_currency=True, escape_markdown=False))
                else:
                    row_depreciation.append(0.0)
                    row_text.append("")
                row_hover.append(
                    f"{asset_display} · {investment_year}<br>Capex {capex_display}"
                )

            depreciation_matrix.append(row_depreciation)
            text_matrix.append(row_text)
            hover_matrix.append(row_hover)
            axis_labels.append(f"{investment_year} · {capex_display}")

            row_index = len(depreciation_matrix) - 1
            if asset_display in group_bounds:
                group_bounds[asset_display]["end"] = row_index
            else:
                group_bounds[asset_display] = {"start": row_index, "end": row_index}

    row_positions = list(range(len(depreciation_matrix)))
    group_bounds_compact = {
        label: (bounds["start"], bounds["end"])
        for label, bounds in group_bounds.items()
    }

    return (
        depreciation_matrix,
        text_matrix,
        row_positions,
        axis_labels,
        hover_matrix,
        group_bounds_compact,
    )



def _get_colorscale_for_scenario(scenario_label: str) -> List[List[str]]:
    """Return a colorscale tuned for the selected investment scenario."""

    return SCENARIO_COLOR_SCALES.get(scenario_label, DEFAULT_COLOR_SCALE)


def _create_gantt_chart(
    depreciation_matrix: Sequence[Sequence[float]],
    text_matrix: Sequence[Sequence[str]],
    row_positions: Sequence[int],
    axis_labels: Sequence[str],
    hover_matrix: Sequence[Sequence[str]],
    years_range: Sequence[int],
    group_bounds: Dict[str, Tuple[int, int]],
    *,
    scenario_label: str,
    total_capex: float | None = None,
    total_depreciation: float | None = None,
    start_year: int | None = None,
    end_year: int | None = None,
) -> go.Figure:
    """Render a heatmap-style gantt chart showing depreciation coverage."""

    custom_colorscale = _get_colorscale_for_scenario(scenario_label)

    heatmap = go.Heatmap(
        z=depreciation_matrix,
        x=years_range,
        y=row_positions,
        text=text_matrix,
        customdata=hover_matrix,
        texttemplate="%{text}",
        textfont={"size": HEATMAP_TEXT_SIZE},
        colorscale=custom_colorscale,
        zmin=0,
        zmax=COLORBAR_MAX,
        hovertemplate="<b>%{customdata}</b><br>Year %{x}<br>Annual Depreciation %{text}<extra></extra>",
        colorbar=dict(
            title=dict(text="Annual Depreciation", side="top", font={"size": 12}),
            tickmode="array",
            tickvals=COLORBAR_TICKS,
            ticktext=COLORBAR_TICKTEXT,
            orientation="h",
            len=0.9,
            thickness=12,
            x=0.5,
            xanchor="center",
            y=-0.08,
            yanchor="bottom",
            tickfont={"size": 10},
            outlinewidth=0,
        ),
    )

    fig = go.Figure(data=heatmap)

    snapshot_title = None
    if (
        total_capex is not None
        and total_depreciation is not None
        and start_year is not None
        and end_year is not None
    ):
        snapshot_title = _format_capex_snapshot_title(total_capex, total_depreciation, start_year, end_year)

    fig.update_layout(
        title=snapshot_title or "Infrastructure Depreciation Timeline",
        xaxis_title=None,
        yaxis_title=None,
        height=max(420, len(axis_labels) * 50),  # 50px per row for consistent row heights
        margin=dict(l=240, r=50, t=80, b=120),
        xaxis=dict(
            side="top",
            tickformat="d",
            tickmode="array",
            tickvals=years_range,
            ticktext=[str(year) for year in years_range],
            tickfont={"size": 12, "color": "#000000"},
        ),
        yaxis=dict(
            autorange="reversed",
            tickmode="array",
            tickvals=row_positions,
            ticktext=axis_labels,
            tickfont={"size": 12, "color": "#000000"},
            ticks="",
        ),
    )

    for label, (start, end) in group_bounds.items():
        center = (start + end) / 2
        fig.add_annotation(
            xref="paper",
            x=-0.22,
            y=center,
            yref="y",
            text=label,
            showarrow=False,
            xanchor="center",
            align="center",
            textangle=-90,
            font={"size": 12, "color": GROUP_LABEL_COLOR},
        )

    fig.update_yaxes(showgrid=False)
    return fig


def _calculate_annual_depreciation_by_asset(
    depreciation_accounting_schedule: pd.DataFrame,
    useful_life_series: pd.Series,
    years_range: Sequence[int],
    max_display_year: int,
) -> Dict[str, pd.Series]:
    """Aggregate depreciation totals per asset type for each year."""

    depreciation_by_asset: Dict[str, pd.Series] = {}

    for asset_type in depreciation_accounting_schedule.columns.tolist():
        useful_life = useful_life_series[asset_type]
        asset_capex = depreciation_accounting_schedule[asset_type]
        series = pd.Series(0.0, index=years_range)

        for investment_year in asset_capex.index:
            investment_amount = float(asset_capex[investment_year])
            if investment_amount <= 0:
                continue

            annual_depreciation = investment_amount / useful_life
            for depreciation_year in range(
                investment_year,
                min(investment_year + useful_life, max_display_year + 1),
            ):
                if depreciation_year in years_range:
                    series[depreciation_year] += annual_depreciation

        depreciation_by_asset[asset_type] = series

    return depreciation_by_asset


def _create_annual_capex_investment_chart(
    depreciation_schedule: CapexDepreciationSchedule
) -> go.Figure:
    """Create stacked bar chart showing annual capex investments by asset type."""

    capex_df = depreciation_schedule.depreciation_accounting_schedule
    years = capex_df.index.tolist()

    fig = go.Figure()

    # Add bars for each asset type
    for asset_type in capex_df.columns:
        values = capex_df[asset_type].tolist()
        display_name = _get_asset_display_name(asset_type)
        normalized_type = _normalize_asset_type(asset_type)
        color = VALUE_CHAIN_DEPRECIATION_COLORS.get(normalized_type, "#636efa")

        fig.add_trace(go.Bar(
            name=display_name,
            x=years,
            y=values,
            marker_color=color,
            hovertemplate=f"<b>{display_name}</b><br>Year %{{x}}<br>Investment %{{customdata}}<extra></extra>",
            customdata=[format_number_to_string(val, is_currency=True, escape_markdown=False) for val in values]
        ))

    # Calculate totals for text overlay
    total_by_year = [sum(capex_df.loc[year].values) for year in years]

    # Calculate total CapEx across all years
    total_capex_all_years = sum(total_by_year)

    # Calculate total power capacity (GW) from power CapEx
    # Power CapEx = power_mw * 1000 * power_cost_per_kw
    # Therefore: power_mw = power_capex / (1000 * power_cost_per_kw)
    # We use DEFAULT_POWER_COST_PER_KW = 2500 from the generation
    DEFAULT_POWER_COST_PER_KW = 2500
    power_column = 'power'
    total_power_gw = 0.0
    if power_column in capex_df.columns:
        total_power_capex = capex_df[power_column].sum()
        total_power_mw = total_power_capex / (1000 * DEFAULT_POWER_COST_PER_KW)
        total_power_gw = total_power_mw / 1000

    # Calculate power by year for labels
    power_gw_by_year = []
    if power_column in capex_df.columns:
        for year in years:
            power_capex = capex_df.loc[year, power_column]
            power_mw = power_capex / (1000 * DEFAULT_POWER_COST_PER_KW)
            power_gw = power_mw / 1000
            power_gw_by_year.append(power_gw)
    else:
        power_gw_by_year = [0.0] * len(years)

    # Create 5-10 clean tick intervals at multiples of 5 billion
    max_value = max(total_by_year) if total_by_year else 1
    tick_values = calculate_chart_tick_intervals(max_value)
    tick_texts = [format_number_to_string(val, is_currency=True, escape_markdown=False) for val in tick_values]

    # Build title with total CapEx and total power
    total_capex_display = format_number_to_string(total_capex_all_years, is_currency=True, escape_markdown=False)
    total_power_display = f"{total_power_gw:.1f}GW" if total_power_gw > 0 else "0GW"
    chart_title = f"Aggregate AI Capex (Total: {total_capex_display}, {total_power_display})"

    apply_value_chain_theme(
        fig,
        title=chart_title,
        height=420,
        barmode="stack",
        xaxis=dict(tickformat="d", tickfont={"color": "#000000"}),
        xaxis_title="",
        yaxis=dict(tickmode="array", tickvals=tick_values, ticktext=tick_texts, tickfont={"color": "#000000"}),
        yaxis_title="Annual Investment",
        font={"size": 13},
    )

    # Extend y-axis range to prevent text cutoff (15% margin instead of 5%)
    max_y_value = max(total_by_year) if total_by_year else 1
    fig.update_yaxes(range=[0, max_y_value * 1.15])

    # Add total labels on top with power (GW) next to dollar figures
    label_texts = []
    for i, (total, power_gw) in enumerate(zip(total_by_year, power_gw_by_year)):
        dollar_text = format_number_to_string(total, is_currency=True, escape_markdown=False)
        power_text = f" ({power_gw:.1f}GW)" if power_gw > 0 else ""
        label_texts.append(f"{dollar_text}{power_text}")

    fig.add_trace(go.Scatter(
        x=years,
        y=[total * 1.05 for total in total_by_year],
        mode="text",
        text=label_texts,
        textposition="top center",
        showlegend=False,
        hoverinfo="skip",
        textfont=dict(size=12, color=GROUP_LABEL_COLOR)
    ))

    return fig


def _create_stacked_depreciation_chart(
    depreciation_by_asset: Dict[str, pd.Series],
    years_range: Sequence[int],
) -> go.Figure:
    """Build stacked bar chart showing annual depreciation by asset type."""

    asset_types = list(depreciation_by_asset.keys())
    fig = go.Figure()

    for asset_type in asset_types:
        normalized_type = _normalize_asset_type(asset_type)
        depreciation_values = [depreciation_by_asset[asset_type][year] for year in years_range]
        display_name = _get_asset_display_name(asset_type)
        color = VALUE_CHAIN_DEPRECIATION_COLORS.get(normalized_type, "#636efa")

        fig.add_trace(
            go.Bar(
                name=display_name,
                x=years_range,
                y=depreciation_values,
                marker_color=color,
                hovertemplate=(
                    f"<b>{display_name}</b><br>Year %{{x}}<br>Depreciation %{{customdata}}<extra></extra>"
                ),
                customdata=[format_number_to_string(val, is_currency=True, escape_markdown=False) for val in depreciation_values],
            )
        )

    total_by_year = [sum(depreciation_by_asset[asset][year] for asset in asset_types) for year in years_range]
    max_value = max(total_by_year) if total_by_year else 1

    # Create 5-10 clean tick intervals at multiples of 5 billion
    tick_values = calculate_chart_tick_intervals(max_value)
    tick_texts = [format_number_to_string(val, is_currency=True, escape_markdown=False) for val in tick_values]

    apply_value_chain_theme(
        fig,
        title="Annual Depreciation Totals",
        height=420,
        barmode="stack",
        xaxis=dict(tickformat="d", tickfont={"color": "#000000"}),
        xaxis_title="",
        yaxis=dict(tickmode="array", tickvals=tick_values, ticktext=tick_texts, tickfont={"color": "#000000"}),
        yaxis_title="Annual Depreciation",
        font={"size": 13},
    )

    # Extend y-axis range to prevent text cutoff (15% margin)
    fig.update_yaxes(range=[0, max_value * 1.15])

    fig.add_trace(
        go.Scatter(
            x=years_range,
            y=[total * 1.05 for total in total_by_year],
            mode="text",
            text=[format_number_to_string(total, is_currency=True, escape_markdown=False) for total in total_by_year],
            textposition="top center",
            textfont={"size": 14, "color": GROUP_LABEL_COLOR},
            showlegend=False,
            hoverinfo="skip",
        )
    )

    return fig


def _calculate_required_margin_percentages(config: ModelConfigurationDict) -> tuple[float, float, float]:
    """Return the required margin percentages for each value chain layer."""

    adjusted_markups = config["adjusted_markups"]
    cloud_margin_pct = (1 - (1 / adjusted_markups["cloud"])) * 100
    model_margin_pct = (1 - (1 / adjusted_markups["model"])) * 100
    app_margin_pct = (1 - (1 / adjusted_markups["app"])) * 100
    return cloud_margin_pct, model_margin_pct, app_margin_pct


def _create_required_revenue_per_dollar_chart(
    config: ModelConfigurationDict,
    latest_data: pd.Series,
    margin_percentages: tuple[float, float, float],
) -> go.Figure:
    """Plot normalized stack showing dollars required per infrastructure dollar."""

    cloud_margin_pct, model_margin_pct, app_margin_pct = margin_percentages

    base_cost = latest_data["depreciation"]
    cloud_revenue = latest_data["required_cloud_revenue_for_margins"]
    model_revenue = latest_data["required_model_revenue_for_margins"]
    app_revenue = latest_data["required_app_revenue_for_margins"]

    layers = [
        f"Cloud<br>({cloud_margin_pct:.0f}% Required Margin)",
        f"Model Provider<br>({model_margin_pct:.0f}% Required Margin)",
        f"App<br>({app_margin_pct:.0f}% Required Margin)",
    ]

    # Always normalize to $1.00 infrastructure cost regardless of actual depreciation values
    normalization_denominator = base_cost if base_cost > 0 else 1.0
    revenues_norm = [cloud_revenue / normalization_denominator, model_revenue / normalization_denominator, app_revenue / normalization_denominator]
    costs_norm = [1.0, cloud_revenue / normalization_denominator, model_revenue / normalization_denominator]
    profits_norm = [revenue - cost for revenue, cost in zip(revenues_norm, costs_norm)]

    fig = go.Figure()
    cost_labels = [
        "$1.00 Infrastructure Cost",
        f"${costs_norm[1]:.2f} Cost Paid to Cloud",
        f"${costs_norm[2]:.2f} Cost Paid to Model Provider",
    ]
    profit_labels = [
        f"${profits_norm[0]:.2f} Required Cloud Gross Margin",
        f"${profits_norm[1]:.2f} Required Model Provider Gross Margin",
        f"${profits_norm[2]:.2f} Required App Gross Margin",
    ]

    fig.add_trace(
        go.Bar(
            name="Costs",
            x=layers,
            y=costs_norm,
            marker_color=VALUE_CHAIN_LAYER_COLORS["infrastructure"],
            text=cost_labels,
            textposition="inside",
            textfont=dict(color="white", size=12),
        )
    )

    fig.add_trace(
        go.Bar(
            name="Required Gross Margin",
            x=layers,
            y=profits_norm,
            marker_color=VALUE_CHAIN_LAYER_COLORS["cloud_margin"],
            text=profit_labels,
            textposition="inside",
            textfont=dict(color="black", size=12),
        )
    )

    fig.add_shape(
        type="line",
        xref="x",
        yref="y",
        x0=0,
        y0=revenues_norm[0],
        x1=1,
        y1=costs_norm[1],
        line=dict(color="black", width=2, dash="dash"),
    )

    fig.add_shape(
        type="line",
        xref="x",
        yref="y",
        x0=1,
        y0=revenues_norm[1],
        x1=2,
        y1=costs_norm[2],
        line=dict(color="black", width=2, dash="dash"),
    )

    revenue_multiplier = app_revenue / base_cost if base_cost > 0 else 0
    apply_value_chain_theme(
        fig,
        title=f"${revenue_multiplier:.2f} required revenue needed to support every $1 of infrastructure cost at assumed margins",
        height=400,
        barmode="stack",
        yaxis_tickformat="$,.2f",
        xaxis=dict(tickfont=dict(color="#000000", size=12)),
        xaxis_title=dict(text="Value Chain Layer", font=dict(color="#000000")),
        yaxis_title=dict(text="Dollars per $1 of Infrastructure Cost", font=dict(color="#000000")),
        margin={"l": 60, "r": 40, "t": 80, "b": 120},
        legend={"orientation": "h", "yanchor": "top", "y": -0.35, "xanchor": "center", "x": 0.5}
    )

    return fig


def _create_required_revenue_timeline_chart(timeline: pd.DataFrame, max_display_year: int) -> Optional[go.Figure]:
    """Render multi-year stacked required revenue chart."""

    # Use exact same dynamic year range as depreciation charts: first investment year to max_display_year
    timeline_with_depreciation = timeline[timeline["depreciation"] > 0]
    if timeline_with_depreciation.empty:
        min_year = timeline["year"].min()
    else:
        min_year = timeline_with_depreciation["year"].min()

    display_timeline = timeline[(timeline["year"] >= min_year) & (timeline["year"] <= max_display_year)].copy()

    if display_timeline.empty:
        st.warning(f"No data available for display years {min_year}-{max_display_year}")
        return None

    display_timeline = display_timeline.assign(
        cloud_required_margin=display_timeline["required_cloud_revenue_for_margins"] - display_timeline["depreciation"],
        model_required_margin=display_timeline["required_model_revenue_for_margins"] - display_timeline["required_cloud_revenue_for_margins"],
        app_required_margin=display_timeline["required_app_revenue_for_margins"] - display_timeline["required_model_revenue_for_margins"],
    )

    years = display_timeline["year"].tolist()
    infrastructure_costs = display_timeline["depreciation"].tolist()
    cloud_required_margins = display_timeline["cloud_required_margin"].tolist()
    model_required_margins = display_timeline["model_required_margin"].tolist()
    app_required_margins = display_timeline["app_required_margin"].tolist()

    fig = go.Figure()

    # Always show infrastructure costs
    fig.add_trace(
        go.Bar(
            name="Depreciated Infrastructure",
            x=years,
            y=infrastructure_costs,
            marker_color=VALUE_CHAIN_LAYER_COLORS["infrastructure"],
            text=[format_number_to_string(cost, is_currency=True, escape_markdown=False) for cost in infrastructure_costs],
            textposition="inside",
            textfont=dict(color="white", size=10),
        )
    )

    # Only show cloud margin if it has meaningful values
    if any(margin > 0 for margin in cloud_required_margins):
        fig.add_trace(
            go.Bar(
                name="Required Cloud Margin",
                x=years,
                y=cloud_required_margins,
                marker_color=VALUE_CHAIN_LAYER_COLORS["cloud_margin"],
                text=[format_number_to_string(margin, is_currency=True, escape_markdown=False) for margin in cloud_required_margins],
                textposition="inside",
            )
        )

    # Only show model margin if it has meaningful values
    if any(margin > 0 for margin in model_required_margins):
        fig.add_trace(
            go.Bar(
                name="Required Model Provider Margin",
                x=years,
                y=model_required_margins,
                marker_color=VALUE_CHAIN_LAYER_COLORS["model_margin"],
                text=[format_number_to_string(margin, is_currency=True, escape_markdown=False) for margin in model_required_margins],
                textposition="inside",
            )
        )

    # Only show app margin if it has meaningful values
    if any(margin > 0 for margin in app_required_margins):
        fig.add_trace(
            go.Bar(
                name="Required App Margin",
                x=years,
                y=app_required_margins,
                marker_color=VALUE_CHAIN_LAYER_COLORS["app_margin"],
                text=[format_number_to_string(margin, is_currency=True, escape_markdown=False) for margin in app_required_margins],
                textposition="inside",
            )
        )

    total_required_revenues = display_timeline["required_app_revenue_for_margins"].tolist()

    # Create 5-10 clean tick intervals at multiples of 5 billion
    max_value = max(total_required_revenues) if total_required_revenues else 1
    tick_values = calculate_chart_tick_intervals(max_value)
    tick_texts = [format_number_to_string(val, is_currency=True, escape_markdown=False) for val in tick_values]

    apply_value_chain_theme(
        fig,
        title="Annual Required Revenue: Infrastructure Cost + Value Chain Margins",
        height=600,
        barmode="stack",
        xaxis=dict(tickformat="d", tickfont={"color": "#000000"}),
        xaxis_title=dict(text="", font=dict(color="#000000")),
        yaxis=dict(tickmode="array", tickvals=tick_values, ticktext=tick_texts, tickfont={"color": "#000000"}),
        yaxis_title=dict(text="Amount ($)", font=dict(color="#000000")),
        margin={"l": 60, "r": 40, "t": 80, "b": 100},
        legend={"orientation": "h", "yanchor": "top", "y": -0.15, "xanchor": "center", "x": 0.5}
    )

    # Extend y-axis range to prevent text cutoff (15% margin)
    fig.update_yaxes(range=[0, max_value * 1.15])

    fig.add_trace(
        go.Scatter(
            x=years,
            y=[rev * 1.05 for rev in total_required_revenues],
            mode="text",
            text=[format_number_to_string(rev, is_currency=True, escape_markdown=False) for rev in total_required_revenues],
            textposition="top center",
            showlegend=False,
            hoverinfo="skip",
            textfont=dict(size=14, color="black"),
        )
    )

    return fig


def _render_required_revenue_analysis(timeline: pd.DataFrame, config: ModelConfigurationDict, max_display_year: int) -> None:
    """Render required revenue analysis at the bottom of the depreciation tab."""
    if timeline.empty:
        st.warning("No timeline data available for required revenue analysis")
        return

    # Use a year with actual depreciation activity for proper normalization
    active_timeline = timeline[timeline["depreciation"] > 0]
    if active_timeline.empty:
        st.warning("No depreciation data available for required revenue analysis")
        return

    # Use the latest year with actual depreciation activity
    latest_active_year = active_timeline["year"].max()
    latest_data = active_timeline[active_timeline["year"] == latest_active_year].iloc[0]

    st.markdown("---")
    st.markdown(
        "Based on the depreciation timeline above, the following charts show the revenue requirements "
        "across the value chain to support the infrastructure investments at the specified margin assumptions."
    )

    required_margin_percentages = _calculate_required_margin_percentages(config)
    normalized_fig = _create_required_revenue_per_dollar_chart(config, latest_data, required_margin_percentages)
    st.plotly_chart(normalized_fig, width="stretch")

    multi_year_fig = _create_required_revenue_timeline_chart(timeline, max_display_year)
    if multi_year_fig:
        st.plotly_chart(multi_year_fig, width="stretch")


def render_depreciation_tab(
    depreciation_schedule: CapexDepreciationSchedule,
    scenario_label: str,
    chips_useful_life_years: int,
    datacenter_useful_life_years: int,
    power_useful_life_years: int,
    timeline: Optional[pd.DataFrame] = None,
    config: Optional[ModelConfigurationDict] = None,
    max_display_year: int = 2031,
) -> None:
    """Render the depreciation timeline tab."""

    depreciation_accounting_schedule = depreciation_schedule.depreciation_accounting_schedule

    # Use updated useful life values from sidebar inputs, not baseline schedule
    useful_life_series = pd.Series({
        "chips": chips_useful_life_years,
        "datacenter": datacenter_useful_life_years,
        "power": power_useful_life_years,
    })

    earliest_year = int(depreciation_accounting_schedule.index.min()) if not depreciation_accounting_schedule.empty else 0
    latest_year = int(depreciation_accounting_schedule.index.max()) if not depreciation_accounting_schedule.empty else 0
    investment_years = f"{earliest_year}-{latest_year}" if earliest_year and latest_year else "N/A"

    if depreciation_accounting_schedule.empty:
        st.warning("No depreciation data available for selected scenario")
        return

    # 1. FIRST: Annual capex investment chart (NEW) - above the header
    capex_investment_fig = _create_annual_capex_investment_chart(depreciation_schedule)
    st.plotly_chart(capex_investment_fig, width="stretch")

    st.write(
        f"Depreciation Assumptions: "
        f"**Compute** ({chips_useful_life_years}yr), "
        f"**Datacenter** ({datacenter_useful_life_years}yr), "
        f"**Power** ({power_useful_life_years}yr) useful lives"
    )

    # Find first year with actual investments to avoid showing empty years
    has_investment = (depreciation_accounting_schedule > 0).any(axis=1)
    if has_investment.any():
        first_investment_year = int(depreciation_accounting_schedule[has_investment].index.min())
    else:
        first_investment_year = int(depreciation_accounting_schedule.index.min())

    years_range = list(range(first_investment_year, max_display_year + 1))
    (
        depreciation_matrix,
        text_matrix,
        row_positions,
        axis_labels,
        hover_matrix,
        group_bounds,
    ) = _create_depreciation_matrix(depreciation_accounting_schedule, useful_life_series, years_range)

    if not depreciation_matrix:
        st.warning("No investments found in the selected scenario")
        return

    window_start = earliest_year
    window_end = latest_year

    capex_slice = depreciation_accounting_schedule.loc[window_start:window_end]
    total_capex = float(capex_slice.fillna(0.0).sum().sum())

    # Recalculate depreciation using updated useful life values from sidebar
    depreciation_series = _recalculate_total_depreciation_by_year(
        depreciation_accounting_schedule, useful_life_series
    )
    depreciation_window = (
        depreciation_series[(depreciation_series.index >= window_start) & (depreciation_series.index <= window_end)]
        if not depreciation_series.empty
        else pd.Series(dtype=float)
    )
    cumulative_dep = float(depreciation_window.sum())

    gantt_fig = _create_gantt_chart(
        depreciation_matrix,
        text_matrix,
        row_positions,
        axis_labels,
        hover_matrix,
        years_range,
        group_bounds,
        scenario_label=scenario_label,
        total_capex=total_capex,
        total_depreciation=cumulative_dep,
        start_year=window_start,
        end_year=window_end,
    )
    st.plotly_chart(gantt_fig, width="stretch")

    depreciation_by_asset = _calculate_annual_depreciation_by_asset(
        depreciation_accounting_schedule,
        useful_life_series,
        years_range,
        max_display_year,
    )
    stacked_fig = _create_stacked_depreciation_chart(depreciation_by_asset, years_range)
    st.plotly_chart(stacked_fig, width="stretch")

    # Add required revenue analysis at the bottom if timeline and config are provided
    if timeline is not None and config is not None:
        _render_required_revenue_analysis(timeline, config, max_display_year)


__all__ = ["render_depreciation_tab"]
