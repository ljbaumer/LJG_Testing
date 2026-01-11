"""AI Lab Liability Model Streamlit Application.

Visualizes cloud contract liabilities, revenue projections, and external funding requirements
for AI laboratory companies.
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

import streamlit as st
from src.constants.ai_lab_profiles import OPENAI_PROFILE, AILabProfile
from src.constants.cloud_contracts import CloudContract
from src.models.AILabLiabilityModel import AILabLiabilityModel
from src.utils.contract_helpers import load_contract_defaults_from_csv
from src.utils.streamlit_app_helpers import calculate_chart_tick_intervals, format_number_to_string


def calculate_external_funding_needs(analysis_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate external funding requirements from the main analysis DataFrame."""
    funding_df = analysis_df.copy()
    funding_df['external_capital_needed'] = (funding_df['total_liability'] - funding_df['projected_revenue']).clip(lower=0)
    funding_df['revenue_coverage'] = funding_df[['projected_revenue', 'total_liability']].min(axis=1)
    return funding_df


def create_contract_stacked_bar_chart(payment_schedule_df: pd.DataFrame, contracts_df: pd.DataFrame) -> go.Figure:
    """Create stacked bar chart showing cloud contract payments by provider with total labels."""
    # Filter to years up to 2029
    filtered_df = payment_schedule_df[payment_schedule_df['year'] <= 2029].copy()

    # Calculate total payment per provider and sort from largest to smallest
    provider_totals = filtered_df.groupby('provider')['payment'].sum().sort_values(ascending=False)

    # Sort the dataframe by provider size (largest first) for stacking order
    filtered_df['provider_order'] = filtered_df['provider'].map({p: i for i, p in enumerate(provider_totals.index)})
    filtered_df = filtered_df.sort_values('provider_order')

    fig = px.bar(
        filtered_df,
        x='year',
        y='payment',
        color='provider',
        title='Cloud Contract Payments by Provider',
        labels={'payment': 'Annual Payment ($)', 'year': 'Year'},
        category_orders={'provider': list(provider_totals.index)},
        color_discrete_map={
            "Oracle": "#F80000",
            "CoreWeave": "#000000",
            "Nvidia": "#76B900",
            "Broadcom": "#808080", # Grey
            "Microsoft Azure": "#0078D4",
            "Google Cloud": "#4285F4",
            "Nscale": "#87CEEB",
            "AMD": "#ED1C24",
            "G42": "#0066CC",
            "SoftBank": "#E60012",
            "Google": "#4285F4",
            "Microsoft": "#0078D4",
        }
    )

    # Create custom y-axis tick labels using format_number_to_string
    # For stacked bars, we need to sum payments by year to get the max stack height
    total_payment_by_year = filtered_df.groupby('year')['payment'].sum()
    max_value = total_payment_by_year.max()
    tick_values = calculate_chart_tick_intervals(max_value)
    tick_labels = [format_number_to_string(val, is_currency=True) for val in tick_values]

    # Create provider to megawatts mapping
    provider_to_mw = contracts_df.set_index('Company')['Number of Megawatts'].to_dict()

    # Calculate total gigawatts per year based on active providers
    total_gw_by_year = []
    years = sorted(filtered_df['year'].unique())

    for year in years:
        year_providers = filtered_df[filtered_df['year'] == year]['provider'].unique()
        total_mw = sum(provider_to_mw.get(p, 0) for p in year_providers if pd.notna(provider_to_mw.get(p, 0)))
        total_gw_by_year.append(total_mw / 1000)  # Convert MW to GW

    # Add text labels above bars showing total $ and GW
    text_labels = []
    for idx, year in enumerate(years):
        total_payment = total_payment_by_year[year]
        total_gw = total_gw_by_year[idx]

        payment_str = format_number_to_string(total_payment, is_currency=True, escape_markdown=False)
        if total_gw > 0:
            text_labels.append(f"{payment_str}<br>{total_gw:.1f} GW")
        else:
            text_labels.append(payment_str)

    fig.add_trace(
        go.Scatter(
            x=years,
            y=total_payment_by_year.values * 1.05,  # Position 5% above bars
            mode="text",
            text=text_labels,
            textposition="top center",
            textfont={"size": 12, "color": "#37474f"},
            showlegend=False,
            hoverinfo="skip",
        )
    )

    fig.update_layout(
        yaxis=dict(
            tickmode='array',
            tickvals=tick_values,
            ticktext=tick_labels
        ),
        hovermode='x unified',
        legend_title='Cloud Provider',
        legend_traceorder='reversed'
    )
    return fig


# Line configuration constants for reuse across charts
REVENUE_LINE_CONFIG = {
    'y_col': 'projected_revenue',
    'name': 'Revenue',
    'color': '#76B900',  # NVIDIA green
    'border_color': '#5A8C00',  # Darker green
    'metric_name': 'Revenue',
    'growth_col': 'revenue_growth_pct',
    'include_growth': True,
}

LIABILITIES_LINE_CONFIG = {
    'y_col': 'total_liability',
    'name': 'Total Cloud Liabilities',
    'color': '#F80000',  # Oracle red
    'border_color': '#B00000',  # Darker red
    'metric_name': 'Cloud Liabilities',
    'growth_col': 'liability_growth_pct',
    'include_growth': True,
}

EXTERNAL_CAPITAL_LINE_CONFIG = {
    'y_col': 'external_capital_needed',
    'name': 'Incremental External Funding',
    'color': '#808080',  # Grey
    'border_color': '#505050',  # Darker grey
    'metric_name': 'External Capital Needed',
    'growth_col': None,
    'include_growth': False,
    'dash': 'dash',
}


def _create_metric_labels(filtered_df: pd.DataFrame, metric_col: str, growth_col: str, metric_name: str, include_growth: bool = True) -> tuple[list[str], list[str]]:
    """
    Create labels and hover text for a metric.

    Returns:
        Tuple of (labels, hover_text)
    """
    from src.utils.streamlit_app_helpers import format_number_to_string

    labels = []
    hover_text = []

    for idx, row in filtered_df.iterrows():
        value = row[metric_col]
        growth_pct = row.get(growth_col)

        formatted_value = format_number_to_string(value, is_currency=True)

        if include_growth and pd.notna(growth_pct) and growth_pct != 0:
            label = f'<b>{formatted_value} (+{growth_pct:.0f}% YoY)</b>'
            hover = f'{metric_name}: {formatted_value}<br>Growth: +{growth_pct:.1f}% YoY<extra></extra>'
        else:
            label = f'<b>{formatted_value}</b>'
            if include_growth and metric_col == 'projected_revenue' and pd.isna(growth_pct):
                hover = f'{metric_name}: {formatted_value}<br>(Base Year)<extra></extra>'
            else:
                hover = f'{metric_name}: {formatted_value}<extra></extra>'

        labels.append(label)
        hover_text.append(hover)

    return labels, hover_text


def _add_line_trace(fig: go.Figure, filtered_df: pd.DataFrame, y_col: str, name: str, color: str, hover_text: list[str], dash: str = None):
    """Add a line trace to the figure."""
    fig.add_trace(go.Scatter(
        x=filtered_df['year'],
        y=filtered_df[y_col],
        name=name,
        mode='lines+markers',
        line=dict(color=color, width=3, dash=dash) if dash else dict(color=color, width=3),
        marker=dict(size=8),
        hovertemplate=hover_text
    ))


def _add_annotations(filtered_df: pd.DataFrame, y_col: str, labels: list[str], color: str, border_color: str) -> list[dict]:
    """Create annotation dictionaries for a metric."""
    annotations = []
    for idx, row in filtered_df.iterrows():
        annotations.append(dict(
            x=row['year'],
            y=row[y_col],
            text=labels[idx],
            showarrow=False,
            yshift=15,
            font=dict(size=10, color=color),
            bgcolor='rgba(255, 255, 255, 0.9)',
            bordercolor=border_color,
            borderwidth=1,
            borderpad=4
        ))
    return annotations


def _create_annotated_line_chart(
    funding_df: pd.DataFrame,
    title: str,
    lines_config: list[dict]
) -> go.Figure:
    """
    Create a line chart with configurable metrics and optional labels/annotations.

    Args:
        funding_df: DataFrame with funding data
        title: Chart title
        lines_config: List of dicts with keys:
            - y_col: Column name for y-axis
            - name: Display name
            - color: Line color
            - border_color: Annotation border color (darker version)
            - metric_name: Name for hover text
            - growth_col: Column name for growth rate (optional)
            - include_growth: Whether to show growth in labels (default True)
            - show_labels: Whether to show annotations (default True)
            - dash: Line dash style (optional)
    """
    from src.utils.streamlit_app_helpers import (
        calculate_chart_tick_intervals,
        format_number_to_string,
    )

    # Filter to years up to 2029
    filtered_df = funding_df[funding_df['year'] <= 2029].copy()

    # Calculate YoY growth rates for all metrics
    for line_cfg in lines_config:
        if 'growth_col' in line_cfg and line_cfg['growth_col'] not in filtered_df.columns:
            base_col = line_cfg['y_col']
            filtered_df[line_cfg['growth_col']] = filtered_df[base_col].pct_change() * 100

    fig = go.Figure()
    annotations = []

    # Add all lines
    for line_cfg in lines_config:
        growth_col = line_cfg.get('growth_col', None)
        include_growth = line_cfg.get('include_growth', True)

        labels, hover_text = _create_metric_labels(
            filtered_df,
            line_cfg['y_col'],
            growth_col,
            line_cfg['metric_name'],
            include_growth
        )

        _add_line_trace(
            fig,
            filtered_df,
            line_cfg['y_col'],
            line_cfg['name'],
            line_cfg['color'],
            hover_text,
            line_cfg.get('dash')
        )

        if line_cfg.get('show_labels', True):
            annotations.extend(_add_annotations(
                filtered_df,
                line_cfg['y_col'],
                labels,
                line_cfg.get('border_color', line_cfg['color']),
                line_cfg['color']
            ))

    # Create custom y-axis tick labels
    max_value = max(filtered_df[cfg['y_col']].max() for cfg in lines_config)
    tick_values = calculate_chart_tick_intervals(max_value)
    tick_labels = [format_number_to_string(val, is_currency=True) for val in tick_values]

    fig.update_layout(
        title=title,
        xaxis_title='Year',
        yaxis_title='Amount ($)',
        hovermode='x unified',
        xaxis=dict(
            dtick=1,
            tick0=filtered_df['year'].min()
        ),
        yaxis=dict(
            tickmode='array',
            tickvals=tick_values,
            ticktext=tick_labels
        ),
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1
        ),
        annotations=annotations
    )
    return fig


def create_revenue_projection_chart(funding_df: pd.DataFrame) -> go.Figure:
    """Create line chart showing only revenue projection with labels."""
    return _create_annotated_line_chart(
        funding_df,
        'Revenue Projection',
        [
            {**REVENUE_LINE_CONFIG, 'show_labels': True}
        ]
    )


def create_revenue_and_liabilities_chart(funding_df: pd.DataFrame) -> go.Figure:
    """Create line chart showing revenue and cloud liabilities, only liability labels shown."""
    return _create_annotated_line_chart(
        funding_df,
        'Revenue vs Cloud Liabilities',
        [
            {**LIABILITIES_LINE_CONFIG, 'show_labels': True},
            {**REVENUE_LINE_CONFIG, 'show_labels': False}
        ]
    )


def create_funding_stacked_bar_chart(funding_df: pd.DataFrame) -> go.Figure:
    """Create line chart showing revenue, liabilities, and external capital - only external capital labels shown."""
    return _create_annotated_line_chart(
        funding_df,
        'Revenue vs Incremental External Funding Requirements',
        [
            {**LIABILITIES_LINE_CONFIG, 'show_labels': False},
            {**REVENUE_LINE_CONFIG, 'show_labels': False},
            {**EXTERNAL_CAPITAL_LINE_CONFIG, 'show_labels': True}
        ]
    )


def create_coverage_ratio_chart(analysis_df: pd.DataFrame) -> go.Figure:
    """Create line chart showing coverage ratio over time."""
    # Filter to years up to 2029
    filtered_df = analysis_df[analysis_df['year'] <= 2029].copy()

    fig = px.line(
        filtered_df,
        x='year',
        y='coverage_ratio',
        title='Revenue Coverage Ratio Over Time',
        labels={'coverage_ratio': 'Coverage Ratio (Revenue/Liabilities)', 'year': 'Year'},
        markers=True
    )
    fig.add_hline(y=1.0, line_dash="dash", line_color="red", annotation_text="Break-even (Coverage = 1.0)")
    fig.update_layout(yaxis_title='Coverage Ratio', hovermode='x unified')
    return fig


def display_key_metrics(analysis_df: pd.DataFrame, payment_schedule_df: pd.DataFrame, funding_df: pd.DataFrame):
    """Display key summary metrics."""
    total_contract_value = payment_schedule_df['payment'].sum()
    peak_liability = analysis_df['total_liability'].max() if not analysis_df.empty else 0
    min_coverage_ratio = analysis_df['coverage_ratio'].min() if not analysis_df.empty else 0
    total_external_funding = funding_df['external_capital_needed'].sum()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Contract Value", format_number_to_string(total_contract_value, is_currency=True))
    with col2:
        st.metric("Peak Annual Liability", format_number_to_string(peak_liability, is_currency=True))
    with col3:
        st.metric("Minimum Coverage Ratio", f"{min_coverage_ratio:.2f}x")
    with col4:
        st.metric("Total External Funding Needed", format_number_to_string(total_external_funding, is_currency=True))


def create_sidebar() -> dict:
    """Create sidebar controls for the model's tunable parameters."""
    st.sidebar.header("Model Configuration")

    st.sidebar.subheader("Revenue Growth")
    st.sidebar.write("Base Revenue for 2025 ($B)")
    base_revenue = st.sidebar.number_input(
        "base_revenue_input",
        value=10.0,
        min_value=0.0,
        step=0.1,
        label_visibility="collapsed",
        help="Base year revenue in billions of dollars for 2025."
    ) * 1e9

    st.sidebar.write("Initial Annual Growth Rate (%)")
    initial_growth = st.sidebar.number_input(
        "initial_growth_rate",
        value=150.0,
        min_value=0.0,
        step=5.0,
        label_visibility="collapsed",
        help="Initial growth rate for the year after the base year."
    ) / 100

    st.sidebar.write("Growth Decay Factor (%)")
    decay_factor = st.sidebar.number_input(
        "decay_factor_input",
        value=75.0,
        min_value=0.0,
        max_value=100.0,
        step=5.0,
        label_visibility="collapsed",
        help="Factor by which the growth rate decays each year (e.g., 75% means 25% slowdown)."
    ) / 100

    st.sidebar.subheader("Contract Liabilities")
    st.sidebar.write("Chip Purchase Growth Rate (%)")
    chip_purchase_growth_rate = st.sidebar.number_input(
        "chip_purchase_growth_rate_input",
        value=30.0,
        min_value=0.0,
        max_value=100.0,
        step=5.0,
        label_visibility="collapsed",
        help="Year-over-year growth rate of chip purchases."
    ) / 100

    st.sidebar.write("Default Contract Duration (Years)")
    duration_years = st.sidebar.number_input(
        "duration_years_input",
        value=5,
        min_value=1,
        max_value=15,
        step=1,
        label_visibility="collapsed",
        help="Default duration for contracts loaded from CSV and manual entries."
    )

    st.sidebar.subheader("Contract Modifiers")

    # Load contract defaults from CSV
    contracts_df = load_contract_defaults_from_csv()

    # Dynamically create inputs for each provider found in CSV
    provider_configs = {}

    for _, row in contracts_df.iterrows():
        company = row['Company']
        st.sidebar.write(f"**{company}**")
        col1, col2 = st.sidebar.columns(2)

        # Use larger step size for values >= 100B, smaller for others
        step_size = 10.0 if row['value_b'] >= 100 else 1.0

        with col1:
            value = st.number_input(
                "Value ($B)",
                value=float(row['value_b']),
                min_value=0.0,
                step=step_size,
                key=f"{company.lower().replace(' ', '_')}_value"
            )

        with col2:
            start_year = st.number_input(
                "Start Year",
                value=int(row['Contract Start Year']),
                min_value=2025,
                max_value=2035,
                step=1,
                key=f"{company.lower().replace(' ', '_')}_start"
            )

        provider_configs[company] = {
            'value_b': value,
            'start_year': int(start_year)
        }

    return {
        'base_revenue': base_revenue,
        'initial_annual_growth_rate': initial_growth,
        'growth_decay_factor': decay_factor,
        'chip_purchase_growth_rate': chip_purchase_growth_rate,
        'duration_years': int(duration_years),
        'providers': provider_configs  # All providers in one dict
    }


def main():
    """Main Streamlit application."""
    st.set_page_config(page_title="AI Lab Liability Model", layout="wide")
    st.title("AI Lab Liability Model")

    # Display OpenAI cloud contract announcements table
    announcements_df = pd.read_csv('data/cloud_contracts/openai_announcements.csv')

    # Format the display dataframe including Links column
    display_df = announcements_df[['Company', 'Project Name', 'Contract Start Year', 'Contract Value (USD billions)', 'Number of Megawatts', 'Comments', 'Links']].copy()
    display_df.columns = ['Company', 'Project', 'Start Year', 'Value ($B)', 'Capacity (MW)', 'Status & Notes', 'Links']

    # Create a sort key: contracts with values first (by descending value), then others alphabetically
    display_df['_sort_value'] = display_df['Value ($B)'].apply(lambda x: x if pd.notna(x) else 0)
    display_df['_has_value'] = display_df['_sort_value'] > 0

    # Sort: has value (descending), then by value (descending), then by company name
    display_df = display_df.sort_values(['_has_value', '_sort_value', 'Company'], ascending=[False, False, True])

    # Format numeric columns to remove decimals
    display_df['Value ($B)'] = display_df['Value ($B)'].apply(lambda x: f"{int(x)}" if pd.notna(x) else "-")
    display_df['Capacity (MW)'] = display_df['Capacity (MW)'].apply(lambda x: f"{int(x)}" if pd.notna(x) else "-")

    # Drop sort helper columns
    display_df = display_df.drop(columns=['_sort_value', '_has_value'])

    # Display with column configuration to make links clickable
    st.dataframe(
        display_df,
        column_config={
            "Links": st.column_config.LinkColumn("Links")
        },
        hide_index=True,
        use_container_width=True
    )

    st.markdown("---")

    config = create_sidebar()

    # Create contracts dynamically from sidebar provider configs
    final_contracts = []
    for company, config_values in config['providers'].items():
        if config_values['value_b'] > 0:
            final_contracts.append(CloudContract(
                provider_name=company,
                total_value=config_values['value_b'] * 1e9,
                duration_years=config['duration_years'],
                start_year=config_values['start_year'],
                chip_purchase_growth_rate=0
            ))

    # Create a temporary profile for this run with the combined list of contracts
    run_profile = AILabProfile(
        company_name=OPENAI_PROFILE.company_name,
        base_revenue=config['base_revenue'],
        revenue_base_year=OPENAI_PROFILE.revenue_base_year,
        contracts=final_contracts,
        last_updated=OPENAI_PROFILE.last_updated
    )

    model = AILabLiabilityModel(
        profile=run_profile,
        initial_annual_growth_rate=config['initial_annual_growth_rate'],
        growth_decay_factor=config['growth_decay_factor'],
        chip_purchase_growth_rate=config['chip_purchase_growth_rate']
    )
    
    analysis_df = model.analysis_df
    payment_schedule_df = model.payment_schedule_df
    funding_df = calculate_external_funding_needs(analysis_df)

    display_key_metrics(analysis_df, payment_schedule_df, funding_df)
    st.markdown("---")

    # Chart 1: Revenue projection only
    revenue_chart = create_revenue_projection_chart(funding_df)
    st.plotly_chart(revenue_chart, use_container_width=True)

    # Chart 2: Revenue and cloud liabilities
    revenue_liabilities_chart = create_revenue_and_liabilities_chart(funding_df)
    st.plotly_chart(revenue_liabilities_chart, use_container_width=True)

    # Chart 3: Complete funding analysis (revenue, liabilities, external capital)
    funding_chart = create_funding_stacked_bar_chart(funding_df)
    st.plotly_chart(funding_chart, use_container_width=True)

    # Load contracts with megawatts data for chart annotations
    contracts_with_mw = load_contract_defaults_from_csv()
    contract_chart = create_contract_stacked_bar_chart(payment_schedule_df, contracts_with_mw)
    st.plotly_chart(contract_chart, use_container_width=True)


if __name__ == "__main__":
    main()
