"""
Main module for the DCF application.
This is the entry point for the Streamlit application.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import os

# Import custom modules
import data
import dcf_calculator
import visualizations
import utils

# Set page configuration
st.set_page_config(
    page_title="DCF Valuation Tool",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom CSS
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #4CAF50;
        color: white;
    }
    .stMarkdown h1 {
        color: #2E7D32;
    }
    .stMarkdown h2 {
        color: #388E3C;
    }
    .stMarkdown h3 {
        color: #43A047;
    }
    .stSidebar {
        background-color: #E8F5E9;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
    }
    .stButton>button:hover {
        background-color: #388E3C;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Display emoji header
utils.display_emoji_header()

# Sidebar
st.sidebar.image("https://img.icons8.com/color/96/000000/financial-growth.png", width=100)
st.sidebar.title("DCF Settings")

# Company selection
company_keys = data.get_all_company_keys()
selected_company = st.sidebar.selectbox(
    "Select a company",
    company_keys,
    index=0
)

company_data = data.get_company_data(selected_company)
industry_averages = data.get_industry_averages(company_data["sector"])

# Navigation
st.sidebar.markdown("## 🧭 Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Company Overview", "Historical Analysis", "DCF Valuation", "Sensitivity Analysis", "Audit Trail"]
)

# DCF Parameters
st.sidebar.markdown("## ⚙️ DCF Parameters")

# Base year selection
historical_df = data.get_historical_financials_as_df(selected_company)
base_year = st.sidebar.selectbox(
    "Base Year",
    historical_df.index.tolist(),
    index=len(historical_df.index) - 1
)

# Forecast period
forecast_years = st.sidebar.slider(
    "Forecast Period (Years)",
    min_value=3,
    max_value=10,
    value=5,
    step=1
)

# Terminal value method
terminal_value_method = st.sidebar.radio(
    "Terminal Value Method",
    ["Perpetuity Growth", "Exit Multiple"]
)

# Advanced settings toggle
show_advanced = st.sidebar.checkbox("Show Advanced Settings", value=False)

if show_advanced:
    st.sidebar.markdown("### Growth Rates")
    
    # Growth rate settings
    use_custom_growth = st.sidebar.checkbox("Use Custom Growth Rates", value=False)
    
    if use_custom_growth:
        growth_rates = []
        for i in range(forecast_years):
            growth_rate = st.sidebar.slider(
                f"Growth Rate Year {i+1} (%)",
                min_value=0.0,
                max_value=50.0,
                value=float(industry_averages["revenue_growth"] * 100),
                step=0.5
            ) / 100
            growth_rates.append(growth_rate)
    else:
        growth_rate = st.sidebar.slider(
            "Growth Rate (All Years) (%)",
            min_value=0.0,
            max_value=50.0,
            value=float(industry_averages["revenue_growth"] * 100),
            step=0.5
        ) / 100
        growth_rates = growth_rate
    
    st.sidebar.markdown("### Margins")
    
    # EBITDA margin settings
    use_custom_margins = st.sidebar.checkbox("Use Custom EBITDA Margins", value=False)
    
    if use_custom_margins:
        ebitda_margins = []
        for i in range(forecast_years):
            margin = st.sidebar.slider(
                f"EBITDA Margin Year {i+1} (%)",
                min_value=0.0,
                max_value=50.0,
                value=float(industry_averages["ebitda_margin"] * 100),
                step=0.5
            ) / 100
            ebitda_margins.append(margin)
    else:
        ebitda_margin = st.sidebar.slider(
            "EBITDA Margin (All Years) (%)",
            min_value=0.0,
            max_value=50.0,
            value=float(industry_averages["ebitda_margin"] * 100),
            step=0.5
        ) / 100
        ebitda_margins = ebitda_margin
    
    st.sidebar.markdown("### Tax Rates")
    
    # Tax rate settings
    tax_rate = st.sidebar.slider(
        "Tax Rate (%)",
        min_value=0.0,
        max_value=50.0,
        value=float(industry_averages["tax_rate"] * 100),
        step=0.5
    ) / 100
    
    st.sidebar.markdown("### Capital Expenditures")
    
    # CapEx settings
    capex_percent = st.sidebar.slider(
        "CapEx (% of Revenue)",
        min_value=0.0,
        max_value=30.0,
        value=float(industry_averages["capex_percent"] * 100),
        step=0.5
    ) / 100
    
    st.sidebar.markdown("### Depreciation")
    
    # Depreciation settings
    depreciation_percent = st.sidebar.slider(
        "Depreciation (% of Revenue)",
        min_value=0.0,
        max_value=20.0,
        value=5.0,
        step=0.5
    ) / 100
    
    st.sidebar.markdown("### Net Working Capital")
    
    # NWC settings
    nwc_percent = st.sidebar.slider(
        "NWC Change (% of Revenue)",
        min_value=0.0,
        max_value=20.0,
        value=float(industry_averages["nwc_percent"] * 100),
        step=0.5
    ) / 100
    
    st.sidebar.markdown("### WACC")
    
    # WACC settings
    use_wacc_calculator = st.sidebar.checkbox("Use WACC Calculator", value=False)
    
    if use_wacc_calculator:
        with st.sidebar.expander("WACC Calculator"):
            wacc = utils.create_wacc_calculator()
    else:
        wacc = st.sidebar.slider(
            "WACC (%)",
            min_value=1.0,
            max_value=20.0,
            value=float(industry_averages["wacc"] * 100),
            step=0.1
        ) / 100
    
    st.sidebar.markdown("### Terminal Value")
    
    if terminal_value_method == "Perpetuity Growth":
        terminal_growth = st.sidebar.slider(
            "Terminal Growth Rate (%)",
            min_value=0.0,
            max_value=5.0,
            value=float(industry_averages["terminal_growth"] * 100),
            step=0.1
        ) / 100
        ebitda_multiple = industry_averages["ebitda_multiple"]
    else:  # Exit Multiple
        ebitda_multiple = st.sidebar.slider(
            "EBITDA Multiple",
            min_value=1.0,
            max_value=30.0,
            value=float(industry_averages["ebitda_multiple"]),
            step=0.5
        )
        terminal_growth = industry_averages["terminal_growth"]
else:
    # Use industry averages for all parameters
    growth_rates = industry_averages["revenue_growth"]
    ebitda_margins = industry_averages["ebitda_margin"]
    tax_rate = industry_averages["tax_rate"]
    capex_percent = industry_averages["capex_percent"]
    depreciation_percent = 0.05  # Assume 5% of revenue
    nwc_percent = industry_averages["nwc_percent"]
    wacc = industry_averages["wacc"]
    terminal_growth = industry_averages["terminal_growth"]
    ebitda_multiple = industry_averages["ebitda_multiple"]

# Create DCF parameters dictionary
dcf_params = {
    "base_revenue": historical_df.loc[base_year, "Revenue"],
    "forecast_years": forecast_years,
    "growth_rates": growth_rates,
    "ebitda_margins": ebitda_margins,
    "tax_rates": tax_rate,
    "capex_percents": capex_percent,
    "depreciation_percents": depreciation_percent,
    "nwc_percents": nwc_percent,
    "wacc": wacc,
    "terminal_value_method": "perpetuity" if terminal_value_method == "Perpetuity Growth" else "multiple",
    "terminal_growth": terminal_growth,
    "ebitda_multiple": ebitda_multiple,
    "debt": company_data["debt"],
    "cash": company_data["cash"]
}

# Calculate projections
projections_df = dcf_calculator.project_financials(
    dcf_params["base_revenue"],
    dcf_params["forecast_years"],
    dcf_params["growth_rates"],
    dcf_params["ebitda_margins"],
    dcf_params["tax_rates"],
    dcf_params["capex_percents"],
    dcf_params["depreciation_percents"],
    dcf_params["nwc_percents"]
)

# Calculate terminal value
if dcf_params["terminal_value_method"] == "perpetuity":
    terminal_value = dcf_calculator.calculate_terminal_value_perpetuity(
        projections_df["Free Cash Flow"].iloc[-1],
        dcf_params["wacc"],
        dcf_params["terminal_growth"]
    )
else:  # multiple
    terminal_value = dcf_calculator.calculate_terminal_value_multiple(
        projections_df["EBITDA"].iloc[-1],
        dcf_params["ebitda_multiple"]
    )

# Calculate DCF valuation
valuation_results = dcf_calculator.calculate_dcf_valuation(
    projections_df,
    terminal_value,
    dcf_params["wacc"],
    dcf_params["debt"],
    dcf_params["cash"]
)

# Calculate implied share price
share_price = dcf_calculator.calculate_implied_share_price(
    valuation_results["Equity Value"],
    company_data["shares_outstanding"]
)

# Add share price to valuation results
valuation_results["Share Price"] = share_price

# Main content
if page == "Company Overview":
    utils.display_section_header("Company Overview", "🏢")
    
    # Display company information
    utils.display_company_info(company_data)
    
    # Display company overview visualization
    st.plotly_chart(visualizations.plot_company_overview(company_data), use_container_width=True)
    
    # Display historical financials
    st.markdown("### 📈 Historical Financial Performance")
    
    # Display historical financials table
    st.dataframe(historical_df.style.format("${:,.2f}"))
    
    # Display historical financials chart
    st.plotly_chart(visualizations.plot_historical_financials(historical_df), use_container_width=True)
    
    # Display historical metrics
    metrics_df = data.calculate_historical_metrics(selected_company)
    
    st.markdown("### 📊 Historical Financial Metrics")
    
    # Format the metrics DataFrame for display
    formatted_metrics = metrics_df.copy()
    for col in formatted_metrics.columns:
        formatted_metrics[col] = formatted_metrics[col].apply(lambda x: f"{x:.2%}")
    
    st.dataframe(formatted_metrics)
    
    # Display historical metrics chart
    st.plotly_chart(visualizations.plot_historical_metrics(metrics_df), use_container_width=True)

elif page == "Historical Analysis":
    utils.display_section_header("Historical Analysis", "📊")
    
    # Display historical financials
    st.markdown("### 📈 Historical Financial Performance")
    
    # Select metrics to display
    metrics_options = ["Revenue", "EBITDA", "EBIT", "Net Income", "Capital Expenditures", "Depreciation", "Change in NWC"]
    selected_metrics = st.multiselect(
        "Select metrics to display",
        metrics_options,
        default=["Revenue", "EBITDA", "EBIT", "Net Income"]
    )
    
    if selected_metrics:
        # Display historical financials chart
        st.plotly_chart(visualizations.plot_historical_financials(historical_df, selected_metrics), use_container_width=True)
    
    # Display historical metrics
    st.markdown("### 📊 Historical Financial Metrics")
    
    metrics_df = data.calculate_historical_metrics(selected_company)
    
    # Select metrics to display
    metrics_options = ["Revenue Growth", "EBITDA Margin", "EBIT Margin", "Net Income Margin", 
                      "Capex as % of Revenue", "Depreciation as % of Revenue", "NWC Change as % of Revenue"]
    selected_metrics = st.multiselect(
        "Select metrics to display",
        metrics_options,
        default=["EBITDA Margin", "EBIT Margin", "Net Income Margin"]
    )
    
    if selected_metrics:
        # Display historical metrics chart
        st.plotly_chart(visualizations.plot_historical_metrics(metrics_df, selected_metrics), use_container_width=True)
    
    # Year-over-year analysis
    st.markdown("### 📅 Year-over-Year Analysis")
    
    # Calculate year-over-year changes
    yoy_df = historical_df.pct_change()
    
    # Format the YoY DataFrame for display
    formatted_yoy = yoy_df.copy()
    for col in formatted_yoy.columns:
        formatted_yoy[col] = formatted_yoy[col].apply(lambda x: f"{x:.2%}" if pd.notnull(x) else "")
    
    st.dataframe(formatted_yoy)
    
    # Industry comparison
    st.markdown("### 🏭 Industry Comparison")
    
    # Create industry comparison table
    industry_comp_data = {
        "Metric": [
            "Revenue Growth",
            "EBITDA Margin",
            "Tax Rate",
            "CapEx (% of Revenue)",
            "NWC Change (% of Revenue)"
        ],
        f"{company_data['name']} (Latest Year)": [
            f"{metrics_df['Revenue Growth'].iloc[-1]:.2%}",
            f"{metrics_df['EBITDA Margin'].iloc[-1]:.2%}",
            f"{tax_rate:.2%}",
            f"{metrics_df['Capex as % of Revenue'].iloc[-1]:.2%}",
            f"{metrics_df['NWC Change as % of Revenue'].iloc[-1]:.2%}"
        ],
        f"{company_data['sector']} Industry Average": [
            f"{industry_averages['revenue_growth']:.2%}",
            f"{industry_averages['ebitda_margin']:.2%}",
            f"{industry_averages['tax_rate']:.2%}",
            f"{industry_averages['capex_percent']:.2%}",
            f"{industry_averages['nwc_percent']:.2%}"
        ]
    }
    
    industry_comp_df = pd.DataFrame(industry_comp_data)
    st.table(industry_comp_df)

elif page == "DCF Valuation":
    utils.display_section_header("DCF Valuation", "💰")
    
    # Display DCF summary
    st.markdown("### 📝 DCF Summary")
    
    # Create columns for summary metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="Enterprise Value", 
            value=visualizations.format_currency(valuation_results["Enterprise Value"])
        )
    
    with col2:
        st.metric(
            label="Equity Value", 
            value=visualizations.format_currency(valuation_results["Equity Value"])
        )
    
    with col3:
        delta = None
        if company_data["current_price"] > 0:
            delta_pct = (share_price / company_data["current_price"] - 1) * 100
            delta = f"{delta_pct:.1f}%"
        
        st.metric(
            label="Implied Share Price", 
            value=f"${share_price:.2f}",
            delta=delta,
            delta_color="normal"
        )
    
    # Display DCF summary table
    st.table(visualizations.create_dcf_summary_table(valuation_results))
    
    # Display DCF valuation bridge
    st.markdown("### 🌉 DCF Valuation Bridge")
    st.plotly_chart(visualizations.plot_dcf_valuation_bridge(valuation_results), use_container_width=True)
    
    # Display projected financials
    st.markdown("### 📊 Projected Financials")
    
    # Add year labels to projections DataFrame
    projection_years = [base_year + i + 1 for i in range(forecast_years)]
    projections_df.index = projection_years
    
    # Display projected financials table
    st.dataframe(visualizations.create_projected_financials_table(projections_df))
    
    # Display projected financials chart
    st.plotly_chart(visualizations.plot_projected_financials(projections_df), use_container_width=True)
    
    # Display FCF breakdown
    st.markdown("### 💵 Free Cash Flow Breakdown")
    st.plotly_chart(visualizations.plot_fcf_breakdown(projections_df), use_container_width=True)
    
    # Download options
    st.markdown("### 📥 Download Results")
    
    # Create download links
    dcf_summary_df = visualizations.create_dcf_summary_table(valuation_results)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(utils.create_download_link(
            projections_df, 
            f"{selected_company}_projections.csv", 
            "Download Projections as CSV"
        ), unsafe_allow_html=True)
    
    with col2:
        st.markdown(utils.create_excel_download_link(
            {
                "Summary": dcf_summary_df,
                "Projections": projections_df
            },
            f"{selected_company}_dcf_valuation.xlsx",
            "Download Full DCF Model as Excel"
        ), unsafe_allow_html=True)

elif page == "Sensitivity Analysis":
    utils.display_section_header("Sensitivity Analysis", "🔍")
    
    # Single-factor sensitivity analysis
    st.markdown("### 📊 Single-Factor Sensitivity Analysis")
    
    # Select parameter for sensitivity analysis
    param_options = {
        "wacc": "WACC",
        "terminal_growth": "Terminal Growth Rate",
        "ebitda_multiple": "EBITDA Multiple",
        "growth_rates": "Revenue Growth Rate",
        "ebitda_margins": "EBITDA Margin"
    }
    
    selected_param = st.selectbox(
        "Select parameter for sensitivity analysis",
        list(param_options.keys()),
        format_func=lambda x: param_options[x]
    )
    
    # Define parameter ranges for sensitivity analysis
    sensitivity_ranges = {
        "wacc": np.linspace(wacc * 0.7, wacc * 1.3, 11),
        "terminal_growth": np.linspace(max(0.01, terminal_growth * 0.5), terminal_growth * 1.5, 11),
        "ebitda_multiple": np.linspace(max(1, ebitda_multiple * 0.7), ebitda_multiple * 1.3, 11),
        "growth_rates": np.linspace(max(0.01, growth_rates * 0.7 if not isinstance(growth_rates, list) else growth_rates[0] * 0.7), 
                                  growth_rates * 1.3 if not isinstance(growth_rates, list) else growth_rates[0] * 1.3, 11),
        "ebitda_margins": np.linspace(max(0.01, ebitda_margins * 0.7 if not isinstance(ebitda_margins, list) else ebitda_margins[0] * 0.7), 
                                    ebitda_margins * 1.3 if not isinstance(ebitda_margins, list) else ebitda_margins[0] * 1.3, 11)
    }
    
    # Perform sensitivity analysis
    sensitivity_results = dcf_calculator.perform_sensitivity_analysis(
        dcf_params,
        {selected_param: sensitivity_ranges[selected_param]},
        company_data["shares_outstanding"]
    )
    
    # Display sensitivity analysis chart
    st.plotly_chart(visualizations.plot_sensitivity_analysis(sensitivity_results, selected_param), use_container_width=True)
    
    # Display sensitivity analysis table
    sensitivity_data = []
    for result in sensitivity_results[selected_param]:
        if selected_param in ["wacc", "terminal_growth", "ebitda_margins", "growth_rates"]:
            param_value = f"{result['Value']:.2%}"
        else:
            param_value = f"{result['Value']:.2f}x"
        
        sensitivity_data.append([
            param_value,
            visualizations.format_currency(result["Enterprise Value"]),
            visualizations.format_currency(result["Equity Value"]),
            f"${result['Share Price']:.2f}"
        ])
    
    sensitivity_df = pd.DataFrame(
        sensitivity_data,
        columns=[param_options[selected_param], "Enterprise Value", "Equity Value", "Share Price"]
    )
    
    st.table(sensitivity_df)
    
    # Two-factor sensitivity analysis
    st.markdown("### 🔄 Two-Factor Sensitivity Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        factor1 = st.selectbox(
            "Select first factor",
            list(param_options.keys()),
            index=0,
            format_func=lambda x: param_options[x],
            key="factor1"
        )
    
    with col2:
        # Filter out the first factor from options
        factor2_options = {k: v for k, v in param_options.items() if k != factor1}
        factor2 = st.selectbox(
            "Select second factor",
            list(factor2_options.keys()),
            index=0,
            format_func=lambda x: param_options[x],
            key="factor2"
        )
    
    # Create two-factor sensitivity table
    sensitivity_table = dcf_calculator.create_two_factor_sensitivity_table(
        dcf_params,
        {"name": factor1, "values": sensitivity_ranges[factor1]},
        {"name": factor2, "values": sensitivity_ranges[factor2]},
        company_data["shares_outstanding"]
    )
    
    # Format index and columns for display
    if factor1 in ["wacc", "terminal_growth", "ebitda_margins", "growth_rates"]:
        sensitivity_table.index = [f"{val:.2%}" for val in sensitivity_table.index]
    else:
        sensitivity_table.index = [f"{val:.2f}x" for val in sensitivity_table.index]
    
    if factor2 in ["wacc", "terminal_growth", "ebitda_margins", "growth_rates"]:
        sensitivity_table.columns = [f"{val:.2%}" for val in sensitivity_table.columns]
    else:
        sensitivity_table.columns = [f"{val:.2f}x" for val in sensitivity_table.columns]
    
    # Set index and column names
    sensitivity_table.index.name = param_options[factor1]
    sensitivity_table.columns.name = param_options[factor2]
    
    # Display sensitivity table heatmap
    st.plotly_chart(visualizations.plot_sensitivity_table(sensitivity_table), use_container_width=True)
    
    # Display sensitivity table
    st.dataframe(sensitivity_table.style.format("${:.2f}"))
    
    # Download sensitivity analysis
    st.markdown("### 📥 Download Sensitivity Analysis")
    
    st.markdown(utils.create_excel_download_link(
        {
            "Single-Factor": sensitivity_df,
            "Two-Factor": sensitivity_table
        },
        f"{selected_company}_sensitivity_analysis.xlsx",
        "Download Sensitivity Analysis as Excel"
    ), unsafe_allow_html=True)

elif page == "Audit Trail":
    utils.display_section_header("Audit Trail", "🔍")
    
    # Display audit trail
    st.markdown("### 📝 DCF Model Assumptions")
    
    # Create audit trail
    audit_df = utils.create_audit_trail(dcf_params)
    
    # Display audit trail
    st.table(audit_df)
    
    # Display calculation methodology
    st.markdown("### 🧮 Calculation Methodology")
    
    with st.expander("Free Cash Flow Calculation"):
        st.markdown("""
        Free Cash Flow (FCF) is calculated as follows:
        
        1. Revenue × EBITDA Margin = EBITDA
        2. EBITDA - Depreciation = EBIT
        3. EBIT × (1 - Tax Rate) = NOPAT (Net Operating Profit After Tax)
        4. NOPAT + Depreciation + Capital Expenditures + Change in NWC = Free Cash Flow
        """)
    
    with st.expander("Terminal Value Calculation"):
        if terminal_value_method == "Perpetuity Growth":
            st.markdown(f"""
            Terminal Value is calculated using the Perpetuity Growth method:
            
            Terminal Value = FCF_final × (1 + g) / (WACC - g)
            
            Where:
            - FCF_final = Free Cash Flow in the final forecast year (${valuation_results['Free Cash Flows'][-1]:.2f})
            - g = Terminal Growth Rate ({terminal_growth:.2%})
            - WACC = Weighted Average Cost of Capital ({wacc:.2%})
            
            Terminal Value = ${valuation_results['Free Cash Flows'][-1]:.2f} × (1 + {terminal_growth:.2%}) / ({wacc:.2%} - {terminal_growth:.2%}) = ${terminal_value:.2f}
            """)
        else:
            st.markdown(f"""
            Terminal Value is calculated using the Exit Multiple method:
            
            Terminal Value = EBITDA_final × EBITDA Multiple
            
            Where:
            - EBITDA_final = EBITDA in the final forecast year (${projections_df['EBITDA'].iloc[-1]:.2f})
            - EBITDA Multiple = {ebitda_multiple:.2f}x
            
            Terminal Value = ${projections_df['EBITDA'].iloc[-1]:.2f} × {ebitda_multiple:.2f} = ${terminal_value:.2f}
            """)
    
    with st.expander("Present Value Calculation"):
        st.markdown(f"""
        The Present Value (PV) of each cash flow is calculated as:
        
        PV = CF / (1 + WACC)^t
        
        Where:
        - CF = Cash Flow in year t
        - WACC = Weighted Average Cost of Capital ({wacc:.2%})
        - t = Year (1, 2, 3, etc.)
        
        For example, the PV of the first year's FCF is:
        ${valuation_results['Free Cash Flows'][0]:.2f} / (1 + {wacc:.2%})^1 = ${valuation_results['PV of FCFs'][0]:.2f}
        """)
    
    with st.expander("Enterprise Value Calculation"):
        st.markdown(f"""
        Enterprise Value is calculated as:
        
        Enterprise Value = Sum of PV of FCFs + PV of Terminal Value
        
        Where:
        - Sum of PV of FCFs = ${valuation_results['Sum of PV of FCFs']:.2f}
        - PV of Terminal Value = ${valuation_results['PV of Terminal Value']:.2f}
        
        Enterprise Value = ${valuation_results['Sum of PV of FCFs']:.2f} + ${valuation_results['PV of Terminal Value']:.2f} = ${valuation_results['Enterprise Value']:.2f}
        """)
    
    with st.expander("Equity Value Calculation"):
        st.markdown(f"""
        Equity Value is calculated as:
        
        Equity Value = Enterprise Value - Debt + Cash
        
        Where:
        - Enterprise Value = ${valuation_results['Enterprise Value']:.2f}
        - Debt = ${valuation_results['Debt']:.2f}
        - Cash = ${valuation_results['Cash']:.2f}
        
        Equity Value = ${valuation_results['Enterprise Value']:.2f} - ${valuation_results['Debt']:.2f} + ${valuation_results['Cash']:.2f} = ${valuation_results['Equity Value']:.2f}
        """)
    
    with st.expander("Share Price Calculation"):
        st.markdown(f"""
        Implied Share Price is calculated as:
        
        Share Price = Equity Value / Shares Outstanding
        
        Where:
        - Equity Value = ${valuation_results['Equity Value']:.2f}
        - Shares Outstanding = {company_data['shares_outstanding']:,}
        
        Share Price = ${valuation_results['Equity Value']:.2f} / {company_data['shares_outstanding']:,} = ${share_price:.2f}
        """)
    
    # Display model limitations
    st.markdown("### ⚠️ Model Limitations")
    
    st.markdown("""
    This DCF model has the following limitations:
    
    1. **Forecast Uncertainty**: Future cash flows are based on projections that may not materialize.
    2. **Terminal Value Sensitivity**: The terminal value often represents a large portion of the total valuation and is highly sensitive to assumptions.
    3. **Discount Rate Subjectivity**: The selection of an appropriate discount rate involves subjective judgment.
    4. **Simplification**: The model simplifies complex business dynamics and may not capture all relevant factors.
    5. **Market Conditions**: The model does not fully account for changing market conditions or competitive dynamics.
    
    Always use DCF valuation alongside other valuation methods and qualitative analysis.
    """)

# Display emoji footer
utils.display_emoji_footer()
