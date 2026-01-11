import pandas as pd
import plotly.express as px

import streamlit as st
from src.constants.gpu_cloud_scenarios_dataclass import (
    SCENARIOS,
    create_gpu_cloud_model_from_scenario,
)
from src.utils.gpu_cloud_helpers import (
    calculate_curtailment_equivalency,
    calculate_total_power_per_gpu,
)
from src.utils.streamlit_app_helpers import (
    create_styled_dataframe,
    format_number_to_string,
    generate_sidebar_from_dataclass,
)


def display_metrics_table(title, metrics_data, columns=["Metric", "Value"], is_subheader=True):
    """
    Display a metrics table with the given title and data
    
    Args:
        title: The title to display above the table
        metrics_data: List of [metric_name, metric_value] pairs
        columns: Column names for the table
        is_subheader: If True, use subheader for title, otherwise use markdown
    """
    df = pd.DataFrame(metrics_data, columns=columns)
    create_styled_dataframe(
        df,
        highlight_keys=["Total", "EBITDA", "EBIT", "Income"],
        title=title if is_subheader else None
    )
    if not is_subheader:
        st.markdown(f"**{title}**")

def display_income_statement(summary_metrics, scenario):
    """
    Display income statement as one table with colored backgrounds for key totals
    """
    # Colored header for the section
    st.markdown("### :blue[Income Statement Summary]")

    total_revenue = summary_metrics["total_revenue"]


    # Use the input tax rate from scenario instead of calculated effective rate
    tax_rate_percent = scenario.tax_rate * 100

    # Get years from scenario for dynamic labeling
    years = int(scenario.total_deal_years)

    # Calculate annual net income and implied valuation
    annual_net_income = summary_metrics["total_net_income"] / years
    valuation_multiple = 20.0
    implied_valuation = annual_net_income * valuation_multiple

    # Helper functions
    def format_pct(value):
        if total_revenue == 0:
            return "0.0%"
        pct = (value / total_revenue) * 100
        return f"{pct:.1f}%"

    def format_amount(value, is_negative=False):
        formatted = format_number_to_string(abs(value), is_currency=True)
        return f"({formatted})" if is_negative else formatted

    # Single comprehensive income statement data
    income_statement_data = [
        [f"Revenue (Over {years} Year Lease)", format_amount(total_revenue), "—"],
        ["OpEx (Datacenter Rent and Power)", format_amount(summary_metrics["total_opex"], True), format_pct(-summary_metrics["total_opex"])],
        [f"EBITDA (Over {years} Years)", format_amount(summary_metrics["total_ebitda"]), format_pct(summary_metrics["total_ebitda"])],
        ["Depreciation (GPU Servers)", format_amount(summary_metrics["total_depreciation"], True), format_pct(-summary_metrics["total_depreciation"])],
        ["Interest Expense (Financing the Servers)", format_amount(summary_metrics["total_interest_paid"], True), format_pct(-summary_metrics["total_interest_paid"])],
        [f"Pre-tax Income (Over {years} Years)", format_amount(summary_metrics["total_pre_tax_income"]), format_pct(summary_metrics["total_pre_tax_income"])],
        [f"Tax Expense ({tax_rate_percent:.1f}% Rate)", format_amount(summary_metrics["total_tax_expense"], True), format_pct(-summary_metrics["total_tax_expense"])],
        [f"Total Net Income (Over {years} Years)", format_amount(summary_metrics["total_net_income"]), format_pct(summary_metrics["total_net_income"])],
        ["Annual Net Income", format_amount(annual_net_income), "—"],
        [f"Implied Valuation Uplift ({valuation_multiple:.0f}x Multiple)", format_amount(implied_valuation), "—"]
    ]

    # Create DataFrame
    df = pd.DataFrame(income_statement_data, columns=["Line Item", "Amount", "% of Revenue"])

    # Display styled table using helper
    create_styled_dataframe(
        df,
        highlight_keys=["Revenue", "EBITDA", "Pre-tax Income", "Total Net Income", "Implied Valuation Uplift"],
        highlight_column="Line Item"
    )

def create_expense_breakdown_pie_chart(summary_metrics):
    nvidia_revenue = summary_metrics["gpu_hardware_cost"]
    interest_expense = summary_metrics["total_interest_paid"]
    datacenter_rent = summary_metrics["total_datacenter_rent"]
    electricity_cost = summary_metrics["total_electricity_cost"]
    neocloud_share = summary_metrics["final_cumulative_cash_flow"]

    # Create data for pie chart
    labels = ['GPU Server Costs', 'Interest Expense', 'Neocloud EBIT', 'Datacenter Rents', 'Electricity']
    sizes = [nvidia_revenue, interest_expense, neocloud_share, datacenter_rent, electricity_cost]

    # Create DataFrame for Plotly
    df = pd.DataFrame({'labels': labels, 'values': sizes})

    # Create pie chart with Plotly
    fig = px.pie(df, values='values', names='labels',
                 color_discrete_sequence=['#2e6930', '#3030c0', '#c03030', '#606060', '#e0e030'])

    fig.update_traces(
        textposition="inside",
        textinfo="percent+label",
        hovertemplate="<b>%{label}</b><br>Amount: %{customdata}<br>Percentage: %{percent}<extra></extra>",
        customdata=[format_number_to_string(val, is_currency=True) for val in sizes],
    )

    fig.update_layout(
        height=400,
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="middle",
            y=0.5,
            xanchor="left",
            x=1.05
        ),
        margin=dict(l=40, r=140, t=60, b=40)
    )

    return fig

def display_curtailment_economics(calculator, summary_metrics):
    """Display the curtailment economics section with user inputs and calculations."""
    server_cost = (
        calculator.CHIP_PRICE +
        calculator.GPU_MODEL_USED.other_compute_costs +
        calculator.INSTALLATION_COST_PER_GPU
    )
    server_power_kw = calculate_total_power_per_gpu(
        calculator.GPU_MODEL_USED.wattage,
        calculator.GPU_UTILIZATION,
        calculator.PUE,
    ) / 1000
    base_electricity_cost_mwh = calculator.ELECTRICITY_COST_PER_KWH_IN_DOLLARS * 1000

    st.subheader("Curtailment Economics (per Chip)")
    col_hours, col_pct = st.columns(2)
    with col_hours:
        downtime_hours_input = st.number_input(
            "Curtailment (hours per day)",
            min_value=0.0,
            max_value=24.0,
            value=1.0,
            step=0.25,
        )
    with col_pct:
        downtime_percentage_display = (downtime_hours_input / 24) * 100
        st.metric("Downtime (%)", f"{downtime_percentage_display:.2f}%")

    curtailment_results = calculate_curtailment_equivalency(
        server_cost_total=server_cost,
        server_power_kw=server_power_kw,
        electricity_base_cost_mwh=base_electricity_cost_mwh,
        downtime_hours_per_day=downtime_hours_input,
    )
    eq_price_increase = curtailment_results["equivalent_price_increase_per_mwh"]

    def fmt_power(value: float) -> str:
        return f"{value:.1f} kW"

    # Calculate power consumption per chip and equivalent kWh pricing
    power_per_chip_kw = calculator.GPU_MODEL_USED.wattage / 1000  # Convert watts to kW
    equivalent_price_per_kwh = curtailment_results["curtailment_price_threshold_per_mwh"] / 1000  # Convert $/MWh to $/kWh

    curtailment_rows = [
        ["Depreciation cost ($/hour)", format_number_to_string(curtailment_results["depreciation_cost_per_hour"], is_currency=True)],
        ["Lost chip time ($/day)", format_number_to_string(curtailment_results["lost_depreciation_dollars_per_day"], is_currency=True)],
        ["Power consumption per chip", fmt_power(power_per_chip_kw)],
        ["Curtailment-hour break-even price ($/kWh)", format_number_to_string(equivalent_price_per_kwh, is_currency=True)],
        ["Average Equivalent Price Increase ($/MWh)", format_number_to_string(eq_price_increase, is_currency=True)],
    ]

    curtailment_data = pd.DataFrame(curtailment_rows, columns=["Metric", "Value"])
    create_styled_dataframe(
        curtailment_data,
        title=None,
        highlight_keys=["Lost chip time"],
        highlight_column="Metric",
    )

    if eq_price_increase is None:
        st.caption("Curtailment equals or exceeds daily runtime, so no electricity equivalency can be calculated.")
    else:
        st.caption(
            "Equivalent electricity price increase is the average $/MWh savings needed across the remaining uptime hours to offset chip downtime."
        )
        st.caption(
            "Curtailment-hour break-even price is the spot $/MWh threshold during the curtailed period that makes you indifferent between running and shutting down."
        )

def main():
    st.title("GPU Cloud Model Calculator")

    # Default scenario selector
    st.sidebar.title("Configuration")
    selected_scenario_name = st.sidebar.selectbox(
        "Select Scenario",
        options=list(SCENARIOS.keys()),
        index=list(SCENARIOS.keys()).index("Oracle-OpenAI $300B")
    )
    selected_scenario = SCENARIOS[selected_scenario_name]

    # Sidebar + calculator generations
    updated_scenario = generate_sidebar_from_dataclass(selected_scenario)
    calculator = create_gpu_cloud_model_from_scenario(updated_scenario)
    irr, annual_irr = calculator.run_model()

    # Display results
    summary_metrics = calculator.get_summary_metrics()

    # Income Statement Summary (replaces old Topline Metrics)
    display_income_statement(summary_metrics, updated_scenario)

    # OpEx Metrics
    opex_metrics_data = [
        ["Total OpEx", format_number_to_string(summary_metrics["total_opex"], is_currency=True)],
        ["Datacenter Rent", format_number_to_string(summary_metrics["total_datacenter_rent"], is_currency=True)],
        ["Electricity Cost", format_number_to_string(summary_metrics["total_electricity_cost"], is_currency=True)],
        ["Personnel Cost", format_number_to_string(summary_metrics["total_personnel_cost"], is_currency=True)]
    ]
    display_metrics_table("OpEx Breakdown", opex_metrics_data, is_subheader=True)

    st.text(f"This cluster requires {calculator.POWER_REQUIRED_KW / 1000} MW of power.")

    # Profitability Metrics
    profitability_metrics_data = [
        ["Total EBITDA", format_number_to_string(summary_metrics["total_ebitda"], is_currency=True)],
        ["Total EBIT", format_number_to_string(summary_metrics["total_ebit"], is_currency=True)],
        ["Total Pre-tax Income", format_number_to_string(summary_metrics["total_pre_tax_income"], is_currency=True)],
        ["Total Tax Expense", format_number_to_string(summary_metrics["total_tax_expense"], is_currency=True)],
        ["Total Net Income", format_number_to_string(summary_metrics["total_net_income"], is_currency=True)],
        ["Avg EBITDA Margin", f"{summary_metrics['avg_ebitda_margin']:.2f}%"],
        ["Avg EBIT Margin", f"{summary_metrics['avg_ebit_margin']:.2f}%"]
    ]
    # Use styled dataframe directly without highlighting since all rows would be highlighted
    df = pd.DataFrame(profitability_metrics_data, columns=["Metric", "Value"])
    create_styled_dataframe(df, title="Profitability Metrics")

    # Add expense breakdown pie chart
    st.subheader("Expense Breakdown")
    fig = create_expense_breakdown_pie_chart(summary_metrics)
    st.plotly_chart(fig, use_container_width=True)

    # Curtailment equivalency (placed after expense breakdown for narrative flow)
    display_curtailment_economics(calculator, summary_metrics)

    # Detailed data
    with st.expander("View Detailed Data"):
        st.dataframe(calculator.df)

if __name__ == "__main__":
    main()
