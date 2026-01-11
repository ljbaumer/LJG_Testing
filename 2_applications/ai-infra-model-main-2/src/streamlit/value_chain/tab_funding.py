"""Funding analysis tab utilities for the Value Chain app."""

import pandas as pd
import plotly.graph_objects as go

import streamlit as st
from src.streamlit.value_chain import apply_value_chain_theme
from src.utils.streamlit_app_helpers import calculate_chart_tick_intervals, format_number_to_string

# Annual growth rate assumption for hyperscaler non-AI FCF
HYPERSCALER_FCF_GROWTH_RATE = 0.08


def _load_hyperscaler_fcf_data(years: list, growth_rate: float = HYPERSCALER_FCF_GROWTH_RATE) -> dict:
    """Load real hyperscaler FCF data from 2025 and project with annual growth.

    Args:
        years: List of years to project FCF for
        growth_rate: Annual growth rate (default 10%)

    Returns:
        Dict with structure: {company: {year: fcf_value}}
    """
    # Load CSV with format: Company, 2020, 2021, ..., 2025, ...
    fcf_raw = pd.read_csv("data/hyperscaler_fcf.csv")

    # Extract 2025 FCF for each company (values in millions) and project forward
    company_fcf_by_year = {}
    for _, row in fcf_raw.iterrows():
        company = row['Company']
        fcf_2025_millions = row['2025']
        fcf_2025_dollars = fcf_2025_millions * 1_000_000  # Convert to dollars

        # Project FCF for each year with 10% growth
        company_fcf_by_year[company] = {}
        for year in years:
            years_from_2025 = year - 2025
            company_fcf_by_year[company][year] = fcf_2025_dollars * ((1 + growth_rate) ** years_from_2025)

    return company_fcf_by_year


def _create_funding_comparison_chart(shortfall_data: pd.DataFrame, company_fcf_by_year: dict) -> go.Figure:
    """Create chart comparing shortfall vs stacked hyperscaler FCF with annual growth."""

    fig = go.Figure()

    # Add shortfall bars (light red) on the left side
    funding_gap_data = shortfall_data[shortfall_data['shortfall'] > 0].copy()

    if not funding_gap_data.empty:
        fig.add_trace(go.Bar(
            name='Total Capex minus Revenue (Projected)',
            x=funding_gap_data['year'],
            y=funding_gap_data['shortfall'],
            marker_color='#ffb3b3',  # Light red
            hovertemplate=(
                "Funding Gap: %{customdata}<br>"
                "<extra></extra>"
            ),
            customdata=[format_number_to_string(val, is_currency=True) for val in funding_gap_data['shortfall']],
            offsetgroup=0,
            legendgroup='capex'
        ))

    # Add stacked hyperscaler FCF bars (one trace per company) on the right side
    company_colors = {
        'Oracle': '#F80000',
        'Meta': '#0668E1',
        'Microsoft': '#00A4EF',
        'Amazon': '#FF9900',
        'Google': '#4285F4'
    }

    years = shortfall_data['year'].unique()

    # Calculate total FCF per company across all years for sorting
    company_totals = {company: sum(fcf_by_year.values())
                      for company, fcf_by_year in company_fcf_by_year.items()}

    # Sort companies by total FCF (largest first) for bottom-to-top stacking
    sorted_companies = sorted(company_totals.items(), key=lambda x: x[1], reverse=True)

    for company, _ in sorted_companies:
        fcf_by_year = company_fcf_by_year[company]
        fcf_values = [fcf_by_year[year] for year in years]

        fig.add_trace(go.Bar(
            name=f'{company}',
            x=years,
            y=fcf_values,
            marker_color=company_colors.get(company, '#808080'),
            hovertemplate=(
                f"{company}: %{{customdata}}<br>"
                "<extra></extra>"
            ),
            customdata=[format_number_to_string(val, is_currency=True) for val in fcf_values],
            offsetgroup=1
        ))

    # Calculate max value for y-axis
    max_shortfall = shortfall_data[shortfall_data['shortfall'] > 0]['shortfall'].max() if not funding_gap_data.empty else 0
    total_fcf_by_year = [sum(company_fcf_by_year[company][year] for company in company_fcf_by_year) for year in years]
    max_fcf = max(total_fcf_by_year) if total_fcf_by_year else 0
    max_value = max(max_shortfall, max_fcf)

    # Create tick intervals based on the maximum displayed value
    tick_values = calculate_chart_tick_intervals(max_value, base_interval=10_000_000_000, target_ticks=6)
    tick_texts = [format_number_to_string(val, is_currency=True, escape_markdown=False) for val in tick_values]

    fig.update_layout(
        title="Infrastructure Funding Gap vs Hyperscaler Non-AI Free Cash Flow",
        xaxis_title="",
        yaxis_title="Amount ($)",
        barmode='stack',  # Stack bars with same offsetgroup
        bargap=0.3,  # Gap between groups (years)
        bargroupgap=0.0,  # No gap within groups (touching bars)
        hovermode='closest',
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.40,
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
            dtick=1,
            tickformat='d'
        )
    )

    apply_value_chain_theme(fig)
    return fig


def render_funding_tab(model = None, interpolated_timeline = None, fcf_growth_rate: float = HYPERSCALER_FCF_GROWTH_RATE) -> None:
    """Render the funding analysis tab.

    Args:
        model: ValueChainModel instance
        interpolated_timeline: DataFrame with interpolated market segments over time
        fcf_growth_rate: Annual growth rate for hyperscaler non-AI FCF (default 8%)
    """
    if model is not None:
        # Calculate shortfall using actual market revenue (not projected markup revenue)
        shortfall_data = model.get_funding_shortfall_timeline(interpolated_timeline)

        years = shortfall_data['year'].unique().tolist()
        fcf_data = _load_hyperscaler_fcf_data(years, growth_rate=fcf_growth_rate)
        funding_chart = _create_funding_comparison_chart(shortfall_data, fcf_data)
        st.plotly_chart(funding_chart, use_container_width=True)

        # Add explanatory text
        growth_rate_pct = fcf_growth_rate * 100
        st.markdown(f"""
        **Methodology Note:**

        This analysis uses 2025 free cash flow (FCF) figures as a baseline to represent the non-AI portion of hyperscalers.
        We assume {growth_rate_pct:.0f}% annual FCF growth (from non-AI businesses) from the 2025 baseline.
        """)
    else:
        st.info("Model data not available for funding analysis.")


__all__ = ["render_funding_tab"]
