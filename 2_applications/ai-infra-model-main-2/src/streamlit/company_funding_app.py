import pandas as pd
import plotly.graph_objects as go

import streamlit as st
from src.constants.cloud_contracts import CloudContract
from src.constants.company_financing_profiles import (
    ORACLE_GPU_CONTRACTS,
    ORACLE_PROFILE,
)
from src.constants.value_chain_depreciation_schedules import CapexDepreciationSchedule
from src.models.CompanyFundingModel import CompanyFundingModel
from src.streamlit.value_chain.tab_depreciation import (
    _create_depreciation_matrix,
    _create_gantt_chart,
)
from src.utils.streamlit_app_helpers import (
    calculate_chart_tick_intervals,
    create_styled_dataframe,
    format_number_to_string,
)


def create_depreciation_gantt_chart(
    capex_schedule: pd.DataFrame,
    chips_life: int = 5,
    datacenter_life: int = 20,
    power_life: int = 25,
    start_year: int = 2025,
    end_year: int = 2029
) -> go.Figure:
    """Create depreciation gantt chart from capex schedule with Oracle-appropriate scale."""
    # Convert capex schedule to depreciation schedule format
    capex_df = capex_schedule[['year', 'chip_capex', 'datacenter_capex', 'power_capex']].copy()
    capex_df['year'] = capex_df['year'].astype(int)
    capex_df = capex_df.set_index('year')
    capex_df.columns = ['chips', 'datacenter', 'power']

    useful_life_series = pd.Series({
        'chips': chips_life,
        'datacenter': datacenter_life,
        'power': power_life
    })

    depreciation_schedule = CapexDepreciationSchedule(
        depreciation_accounting_schedule=capex_df,
        useful_life_series=useful_life_series
    )

    # Calculate depreciation matrix for gantt chart
    years_range = list(range(start_year, end_year + 1))
    (
        depreciation_matrix,
        text_matrix,
        row_positions,
        axis_labels,
        hover_matrix,
        group_bounds,
    ) = _create_depreciation_matrix(
        depreciation_schedule.depreciation_accounting_schedule,
        useful_life_series,
        years_range
    )

    # Calculate totals for title
    total_capex = float(capex_df.sum().sum())
    total_depreciation_series = depreciation_schedule.get_total_depreciation_by_year()
    total_depreciation = float(total_depreciation_series[
        (total_depreciation_series.index >= start_year) &
        (total_depreciation_series.index <= end_year)
    ].sum())

    # Create gantt chart with base settings
    fig = _create_gantt_chart(
        depreciation_matrix,
        text_matrix,
        row_positions,
        axis_labels,
        hover_matrix,
        years_range,
        group_bounds,
        scenario_label="Zero Growth",
        total_capex=total_capex,
        total_depreciation=total_depreciation,
        start_year=start_year,
        end_year=end_year,
    )

    # Find max depreciation value to rescale colorbar appropriately for Oracle
    max_depreciation = max([max(row) for row in depreciation_matrix]) if depreciation_matrix else 1

    # Create Oracle-appropriate colorbar scale (typically in billions, not tens of billions)
    import math
    max_tick = math.ceil(max_depreciation / 1_000_000_000) * 1_000_000_000  # Round up to nearest billion
    num_ticks = min(5, max(2, int(max_tick / 1_000_000_000)))  # 2-5 ticks
    tick_values = [i * (max_tick / (num_ticks - 1)) for i in range(num_ticks)]
    tick_labels = [format_number_to_string(val, is_currency=True) for val in tick_values]

    # Update the heatmap colorbar with Oracle-appropriate scale
    fig.data[0].update(
        zmax=max_tick,
        colorbar=dict(
            title=dict(text="Annual Depreciation", side="top", font={"size": 12}),
            tickmode="array",
            tickvals=tick_values,
            ticktext=tick_labels,
            orientation="h",
            len=0.9,
            thickness=12,
            x=0.5,
            xanchor="center",
            y=-0.08,
            yanchor="bottom",
            tickfont={"size": 10},
            outlinewidth=0,
        )
    )

    return fig


def main():
    st.set_page_config(page_title="Company Funding Model", layout="wide")
    st.title("Company Funding Model: Oracle AI Infrastructure")

    # Sidebar configuration
    st.sidebar.header("Configuration")

    st.sidebar.write("Non-AI FCF Growth Rate (%)")
    non_ai_fcf_growth_rate = st.sidebar.number_input(
        "non_ai_fcf_growth_rate_input",
        value=2.0,
        min_value=-50.0,
        max_value=50.0,
        step=1.0,
        label_visibility="collapsed",
        help="Annual growth rate of non-AI business free cash flow"
    ) / 100  # Convert to decimal

    st.sidebar.write("Chip Purchase Growth Rate (%)")
    chip_purchase_growth_rate = st.sidebar.number_input(
        "chip_purchase_growth_rate_input",
        value=30.0,
        min_value=-100.0,
        max_value=100.0,
        step=5.0,
        label_visibility="collapsed",
        help="Year-over-year growth rate of chip purchases (negative means higher purchases in early years)"
    ) / 100  # Convert to decimal

    # Update contracts with user input
    updated_contracts = [
        CloudContract(
            provider_name=contract.provider_name,
            total_value=contract.total_value,
            duration_years=contract.duration_years,
            start_year=contract.start_year,
            chip_purchase_growth_rate=chip_purchase_growth_rate,
        )
        for contract in ORACLE_GPU_CONTRACTS
    ]

    # Update profile with user input
    profile = ORACLE_PROFILE
    profile.non_ai_fcf_growth_rate = non_ai_fcf_growth_rate

    # Initialize model
    model = CompanyFundingModel(profile=profile, contracts=updated_contracts)

    # Generate revenue breakdown for chart
    revenue_breakdown = pd.concat([
        contract.calculate_payment_schedule()
        .rename(columns={"payment": "ai_revenue"})
        .assign(customer=contract.provider_name)
        for contract in updated_contracts
    ], ignore_index=True)[['year', 'customer', 'ai_revenue']].sort_values(['year', 'customer']).reset_index(drop=True)

    # Calculate summary metrics
    total_ai_revenue = model.shortfall['ai_revenue'].sum()
    total_capex = model.shortfall['total_capex'].sum()
    total_shortfall = model.shortfall['shortfall'].sum()
    total_debt_required = model.debt_schedule['debt_required'].sum()

    # Overview metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Total AI Revenue (5Y)",
            format_number_to_string(total_ai_revenue, is_currency=True)
        )

    with col2:
        st.metric(
            "Total CapEx Required",
            format_number_to_string(total_capex, is_currency=True)
        )

    with col3:
        st.metric(
            "Total Funding Shortfall",
            format_number_to_string(total_shortfall, is_currency=True)
        )

    with col4:
        st.metric(
            "Total Debt Required",
            format_number_to_string(total_debt_required, is_currency=True)
        )

    # Chart 1: CapEx Breakdown (moved to top)
    fig_capex = create_capex_breakdown_chart(model.shortfall)
    st.plotly_chart(fig_capex, use_container_width=True)

    # Depreciation gantt chart
    st.write(
        "Depreciation Assumptions: "
        "**Compute** (5yr), "
        "**Datacenter** (20yr), "
        "**Power** (25yr) useful lives"
    )
    fig_gantt = create_depreciation_gantt_chart(model.capex_schedule)
    st.plotly_chart(fig_gantt, use_container_width=True)

    # Show net PP&E after depreciation
    net_ppe = model.calculate_net_ppe_at_year(2029)
    st.markdown(f"*{format_number_to_string(net_ppe, is_currency=True)} net PP&E on the books entering 2030*")

    # Chart 2: Cloud Revenue by Customer
    fig_revenue = create_revenue_by_customer_chart(revenue_breakdown)
    st.plotly_chart(fig_revenue, use_container_width=True)

    # Chart 3: Funding Waterfall
    fig_funding = create_funding_waterfall_chart(model.debt_schedule)
    st.plotly_chart(fig_funding, use_container_width=True)

    # Chart 4: Cumulative Debt Accumulation
    fig_debt = create_debt_accumulation_chart(model.debt_schedule)
    st.plotly_chart(fig_debt, use_container_width=True)

    # Data Table
    display_shortfall_table(model.debt_schedule)


def create_revenue_by_customer_chart(revenue_breakdown_df: pd.DataFrame) -> go.Figure:
    """Create stacked bar chart of cloud revenue by customer."""
    customers = revenue_breakdown_df['customer'].unique()

    fig = go.Figure()

    for customer in customers:
        customer_data = revenue_breakdown_df[revenue_breakdown_df['customer'] == customer]
        fig.add_trace(go.Bar(
            x=customer_data['year'],
            y=customer_data['ai_revenue'],
            name=customer,
            text=[format_number_to_string(v, is_currency=True) for v in customer_data['ai_revenue']],
            textposition='inside',
        ))

    # Create custom y-axis tick labels using format_number_to_string
    # For stacked bars, sum revenue by year to get max stack height
    max_value = revenue_breakdown_df.groupby('year')['ai_revenue'].sum().max()
    tick_values = calculate_chart_tick_intervals(max_value)
    tick_labels = [format_number_to_string(val, is_currency=True) for val in tick_values]

    fig.update_layout(
        title='Cloud Revenue by Customer',
        barmode='stack',
        xaxis_title="Year",
        yaxis_title="Cloud Revenue (USD)",
        hovermode='x unified',
        yaxis=dict(
            tickmode='array',
            tickvals=tick_values,
            ticktext=tick_labels
        ),
    )

    return fig


def create_capex_breakdown_chart(shortfall_df: pd.DataFrame) -> go.Figure:
    """Create stacked bar chart of CapEx breakdown."""
    fig = go.Figure()

    # Add each capex component
    fig.add_trace(go.Bar(
        x=shortfall_df['year'],
        y=shortfall_df['chip_capex'],
        name='Chip CapEx',
        text=[format_number_to_string(v, is_currency=True) for v in shortfall_df['chip_capex']],
        textposition='inside',
    ))

    fig.add_trace(go.Bar(
        x=shortfall_df['year'],
        y=shortfall_df['datacenter_capex'],
        name='Datacenter CapEx',
        text=[format_number_to_string(v, is_currency=True) for v in shortfall_df['datacenter_capex']],
        textposition='inside',
    ))

    fig.add_trace(go.Bar(
        x=shortfall_df['year'],
        y=shortfall_df['power_capex'],
        name='Power CapEx',
        text=[format_number_to_string(v, is_currency=True) for v in shortfall_df['power_capex']],
        textposition='inside',
    ))

    # Create custom y-axis tick labels using format_number_to_string
    # For stacked bars, sum all capex components by year to get max stack height
    max_value = shortfall_df['total_capex'].max()
    tick_values = calculate_chart_tick_intervals(max_value)
    tick_labels = [format_number_to_string(val, is_currency=True) for val in tick_values]

    fig.update_layout(
        title='CapEx Breakdown',
        barmode='stack',
        xaxis_title="Year",
        yaxis_title="CapEx (USD)",
        hovermode='x unified',
        yaxis=dict(
            tickmode='array',
            tickvals=tick_values,
            ticktext=tick_labels
        ),
    )

    return fig


def create_funding_waterfall_chart(debt_schedule_df: pd.DataFrame) -> go.Figure:
    """Create stacked bar chart showing funding sources."""
    fig = go.Figure()

    # AI Revenue (green, as income)
    fig.add_trace(go.Bar(
        x=debt_schedule_df['year'],
        y=debt_schedule_df['ai_revenue'],
        name='AI Revenue',
        marker_color='green',
        text=[format_number_to_string(v, is_currency=True) for v in debt_schedule_df['ai_revenue']],
        textposition='inside',
    ))

    # Non-AI FCF (blue, as funding source)
    fig.add_trace(go.Bar(
        x=debt_schedule_df['year'],
        y=debt_schedule_df['non_ai_fcf'],
        name='Non-AI FCF',
        marker_color='blue',
        text=[format_number_to_string(v, is_currency=True) for v in debt_schedule_df['non_ai_fcf']],
        textposition='inside',
    ))

    # Debt Required (orange, as funding gap)
    fig.add_trace(go.Bar(
        x=debt_schedule_df['year'],
        y=debt_schedule_df['debt_required'],
        name='Debt Required',
        marker_color='orange',
        text=[format_number_to_string(v, is_currency=True) for v in debt_schedule_df['debt_required']],
        textposition='inside',
    ))

    # Create custom y-axis tick labels using format_number_to_string
    max_value = (debt_schedule_df['ai_revenue'] + debt_schedule_df['non_ai_fcf'] + debt_schedule_df['debt_required']).max()
    tick_values = calculate_chart_tick_intervals(max_value)
    tick_labels = [format_number_to_string(val, is_currency=True) for val in tick_values]

    fig.update_layout(
        title='Funding Analysis',
        barmode='relative',
        xaxis_title="Year",
        yaxis_title="USD",
        hovermode='x unified',
        yaxis=dict(
            tickmode='array',
            tickvals=tick_values,
            ticktext=tick_labels
        ),
    )

    return fig


def create_debt_accumulation_chart(debt_schedule_df: pd.DataFrame) -> go.Figure:
    """Create line chart showing cumulative debt accumulation."""
    df = debt_schedule_df.copy()
    df['cumulative_debt'] = df['debt_required'].cumsum()

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df['year'],
        y=df['cumulative_debt'],
        mode='lines+markers',
        name='Cumulative Debt',
        line=dict(color='orange', width=3),
        marker=dict(size=10),
        text=[format_number_to_string(v, is_currency=True) for v in df['cumulative_debt']],
        textposition='top center',
    ))

    # Create custom y-axis tick labels using format_number_to_string
    max_value = df['cumulative_debt'].max()
    tick_values = calculate_chart_tick_intervals(max_value)
    tick_labels = [format_number_to_string(val, is_currency=True) for val in tick_values]

    fig.update_layout(
        title='Cumulative Debt Accumulation',
        xaxis_title="Year",
        yaxis_title="Cumulative Debt (USD)",
        hovermode='x unified',
        yaxis=dict(
            tickmode='array',
            tickvals=tick_values,
            ticktext=tick_labels
        ),
    )

    return fig


def display_shortfall_table(debt_schedule_df: pd.DataFrame):
    """Display shortfall analysis table with formatted values."""
    display_df = debt_schedule_df[['year', 'ai_revenue', 'total_capex', 'shortfall', 'non_ai_fcf', 'debt_required']].copy()

    # Format currency columns
    currency_columns = ['ai_revenue', 'total_capex', 'shortfall', 'non_ai_fcf', 'debt_required']
    for col in currency_columns:
        display_df[col] = display_df[col].apply(lambda x: format_number_to_string(x, is_currency=True))

    # Rename columns for display
    display_df.columns = ['Year', 'AI Revenue', 'Total CapEx', 'Shortfall', 'Non-AI FCF', 'Debt Required']

    create_styled_dataframe(
        display_df,
        highlight_keys=['Shortfall', 'Debt Required'],
        highlight_column='Year'
    )


if __name__ == "__main__":
    main()
