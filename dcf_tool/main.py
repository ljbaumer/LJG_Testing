import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import plotly.graph_objects as go
from typing import Dict, List, Any, Optional

# Add the current directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dcf_tool.models.dcf_model import DCFModel
from dcf_tool.models.growth_methods import get_available_growth_methods
from dcf_tool.utils.visualization import (
    create_historical_vs_forecast_chart,
    create_dcf_waterfall_chart,
    create_sensitivity_analysis_heatmap,
    create_cash_flow_chart
)

# Set page config
st.set_page_config(
    page_title="DCF Valuation Model",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'model' not in st.session_state:
    # Get the absolute path to the data directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, "data")
    st.session_state.model = DCFModel(data_dir=data_dir)
    st.session_state.model.load_data()

if 'data' not in st.session_state:
    st.session_state.data = st.session_state.model.data.copy()

if 'results' not in st.session_state:
    st.session_state.results = None

if 'show_audit_trail' not in st.session_state:
    st.session_state.show_audit_trail = False

if 'edited_data' not in st.session_state:
    st.session_state.edited_data = False

# Initialize model parameters in session state
model_params = st.session_state.model.get_parameters()
for param, value in model_params.items():
    if param not in st.session_state:
        if param in ["revenue_growth", "ebitda_margin", "tax_rate", "da_to_revenue", 
                    "capex_to_revenue", "wc_to_revenue", "terminal_growth_rate", 
                    "risk_free_rate", "market_risk_premium", "cost_of_debt", "wacc"]:
            # Convert decimal to percentage for display
            st.session_state[param] = value * 100
        else:
            st.session_state[param] = value

# Initialize other required session state variables
if 'multiple_type' not in st.session_state:
    st.session_state.multiple_type = "EV/EBITDA"

# Function to update model parameters
def update_parameters():
    parameters = {
        "forecast_years": st.session_state.forecast_years,
        "revenue_growth": st.session_state.revenue_growth / 100,  # Convert from percentage
        "ebitda_margin": st.session_state.ebitda_margin / 100,  # Convert from percentage
        "tax_rate": st.session_state.tax_rate / 100,  # Convert from percentage
        "da_to_revenue": st.session_state.da_to_revenue / 100,  # Convert from percentage
        "capex_to_revenue": st.session_state.capex_to_revenue / 100,  # Convert from percentage
        "wc_to_revenue": st.session_state.wc_to_revenue / 100,  # Convert from percentage
        "terminal_growth_rate": st.session_state.terminal_growth_rate / 100,  # Convert from percentage
        "terminal_method": st.session_state.terminal_method,
        "exit_multiple": st.session_state.exit_multiple,
        "multiple_type": st.session_state.multiple_type,
        "debt_value": st.session_state.debt_value,
        "cash_equivalents": st.session_state.cash_equivalents,
        "shares_outstanding": st.session_state.shares_outstanding,
        "calculate_wacc": st.session_state.calculate_wacc
    }
    
    # Add WACC parameters
    if st.session_state.calculate_wacc:
        parameters.update({
            "risk_free_rate": st.session_state.risk_free_rate / 100,  # Convert from percentage
            "market_risk_premium": st.session_state.market_risk_premium / 100,  # Convert from percentage
            "beta": st.session_state.beta,
            "cost_of_debt": st.session_state.cost_of_debt / 100  # Convert from percentage
        })
    else:
        parameters["wacc"] = st.session_state.wacc / 100  # Convert from percentage
    
    st.session_state.model.set_parameters(parameters)

# Function to run the DCF valuation
def run_valuation():
    update_parameters()
    
    # If data has been edited, save it to the model
    if st.session_state.edited_data:
        st.session_state.model.data = st.session_state.data.copy()
        st.session_state.edited_data = False
    
    # Run the valuation
    st.session_state.results = st.session_state.model.run_dcf_valuation()
    
    # Update the forecast data in the session state
    st.session_state.forecast_data = st.session_state.model.forecast_data.copy()

# Function to save edited data
def save_edited_data():
    st.session_state.model.save_data(st.session_state.data, as_user_data=True)
    st.success("Data saved successfully!")

# Function to handle data editor changes
def on_data_change(data):
    st.session_state.data = data
    st.session_state.edited_data = True

# Function to format currency values
def format_currency(value):
    return f"${value:,.2f}"

# Function to format percentage values
def format_percentage(value):
    return f"{value:.2f}%"

# Main app title
st.title("DCF Valuation Model")

# Sidebar for model parameters
with st.sidebar:
    st.header("Model Parameters")
    
    # Forecast parameters
    st.subheader("Forecast Parameters")
    st.session_state.forecast_years = st.slider(
        "Forecast Years", 
        min_value=1, 
        max_value=10, 
        value=st.session_state.model.parameters["forecast_years"],
        step=1,
        help="Number of years to forecast"
    )
    
    st.session_state.revenue_growth = st.slider(
        "Revenue Growth (%)", 
        min_value=-20.0, 
        max_value=50.0, 
        value=st.session_state.model.parameters["revenue_growth"] * 100,
        step=0.5,
        help="Annual revenue growth rate"
    )
    
    st.session_state.ebitda_margin = st.slider(
        "EBITDA Margin (%)", 
        min_value=0.0, 
        max_value=50.0, 
        value=st.session_state.model.parameters["ebitda_margin"] * 100,
        step=0.5,
        help="EBITDA as a percentage of revenue"
    )
    
    st.session_state.tax_rate = st.slider(
        "Tax Rate (%)", 
        min_value=0.0, 
        max_value=50.0, 
        value=st.session_state.model.parameters["tax_rate"] * 100,
        step=0.5,
        help="Corporate tax rate"
    )
    
    st.session_state.da_to_revenue = st.slider(
        "Depreciation & Amortization to Revenue (%)", 
        min_value=0.0, 
        max_value=20.0, 
        value=st.session_state.model.parameters["da_to_revenue"] * 100,
        step=0.5,
        help="Depreciation & Amortization as a percentage of revenue"
    )
    
    st.session_state.capex_to_revenue = st.slider(
        "Capital Expenditures to Revenue (%)", 
        min_value=0.0, 
        max_value=20.0, 
        value=st.session_state.model.parameters["capex_to_revenue"] * 100,
        step=0.5,
        help="Capital Expenditures as a percentage of revenue"
    )
    
    st.session_state.wc_to_revenue = st.slider(
        "Working Capital Change to Revenue (%)", 
        min_value=-5.0, 
        max_value=10.0, 
        value=st.session_state.model.parameters["wc_to_revenue"] * 100,
        step=0.5,
        help="Change in Working Capital as a percentage of revenue"
    )
    
    # Terminal value parameters
    st.subheader("Terminal Value Parameters")
    
    # Get available growth methods
    growth_methods = get_available_growth_methods()
    method_options = {method.name: key for key, method in growth_methods.items()}
    
    st.session_state.terminal_method = st.selectbox(
        "Terminal Value Method",
        options=list(method_options.keys()),
        index=0 if st.session_state.model.parameters["terminal_method"] == "perpetuity_growth" else 1,
        format_func=lambda x: x,
        help="Method to calculate terminal value"
    )
    
    # Convert display name back to method key
    st.session_state.terminal_method = method_options[st.session_state.terminal_method]
    
    # Show parameters based on selected method
    if st.session_state.terminal_method == "perpetuity_growth":
        st.session_state.terminal_growth_rate = st.slider(
            "Terminal Growth Rate (%)", 
            min_value=0.0, 
            max_value=5.0, 
            value=st.session_state.model.parameters["terminal_growth_rate"] * 100,
            step=0.1,
            help="Long-term growth rate for perpetuity calculation"
        )
    else:  # exit_multiple
        st.session_state.multiple_type = st.selectbox(
            "Multiple Type",
            options=["EV/EBITDA", "EV/EBIT"],
            index=0,
            help="Type of multiple to use for terminal value calculation"
        )
        
        st.session_state.exit_multiple = st.slider(
            f"{st.session_state.multiple_type} Multiple", 
            min_value=1.0, 
            max_value=20.0, 
            value=st.session_state.model.parameters["exit_multiple"],
            step=0.5,
            help=f"{st.session_state.multiple_type} multiple for terminal value calculation"
        )
    
    # Discount rate parameters
    st.subheader("Discount Rate Parameters")
    
    st.session_state.calculate_wacc = st.checkbox(
        "Calculate WACC",
        value=st.session_state.model.parameters["calculate_wacc"],
        help="If checked, WACC will be calculated using CAPM. Otherwise, a direct WACC input will be used."
    )
    
    if st.session_state.calculate_wacc:
        st.session_state.risk_free_rate = st.slider(
            "Risk-Free Rate (%)", 
            min_value=0.0, 
            max_value=10.0, 
            value=st.session_state.model.parameters["risk_free_rate"] * 100,
            step=0.1,
            help="Risk-free rate for CAPM calculation"
        )
        
        st.session_state.market_risk_premium = st.slider(
            "Market Risk Premium (%)", 
            min_value=0.0, 
            max_value=10.0, 
            value=st.session_state.model.parameters["market_risk_premium"] * 100,
            step=0.1,
            help="Market risk premium for CAPM calculation"
        )
        
        st.session_state.beta = st.slider(
            "Beta", 
            min_value=0.1, 
            max_value=3.0, 
            value=st.session_state.model.parameters["beta"],
            step=0.1,
            help="Company's beta for CAPM calculation"
        )
        
        st.session_state.cost_of_debt = st.slider(
            "Cost of Debt (%)", 
            min_value=0.0, 
            max_value=15.0, 
            value=st.session_state.model.parameters.get("cost_of_debt", 5.0),
            step=0.1,
            help="Pre-tax cost of debt"
        )
    else:
        st.session_state.wacc = st.slider(
            "WACC (%)", 
            min_value=1.0, 
            max_value=20.0, 
            value=st.session_state.model.parameters["wacc"] * 100,
            step=0.1,
            help="Weighted Average Cost of Capital"
        )
    
    # Balance sheet parameters
    st.subheader("Balance Sheet Parameters")
    
    st.session_state.debt_value = st.number_input(
        "Debt Value",
        min_value=0.0,
        value=float(st.session_state.model.parameters["debt_value"]),
        step=10.0,
        help="Market value of debt"
    )
    
    st.session_state.cash_equivalents = st.number_input(
        "Cash & Equivalents",
        min_value=0.0,
        value=float(st.session_state.model.parameters["cash_equivalents"]),
        step=10.0,
        help="Cash and cash equivalents"
    )
    
    st.session_state.shares_outstanding = st.number_input(
        "Shares Outstanding (millions)",
        min_value=0.1,
        value=float(st.session_state.model.parameters["shares_outstanding"]),
        step=1.0,
        help="Number of shares outstanding in millions"
    )
    
    # Run valuation button
    st.button("Run Valuation", on_click=run_valuation, type="primary")

# Main content area with tabs
tab1, tab2, tab3, tab4 = st.tabs(["Financial Data", "Forecast", "Valuation Results", "Audit Trail"])

# Tab 1: Financial Data
with tab1:
    st.header("Financial Data")
    st.write("View and edit the historical financial data. Changes will be used in the next valuation run.")
    
    # Data editor
    edited_data = st.data_editor(
        st.session_state.data,
        use_container_width=True,
        num_rows="dynamic",
        key="financial_data_editor",
        on_change=lambda: on_data_change(st.session_state.financial_data_editor),
        column_config={
            "Year": st.column_config.NumberColumn(
                "Year",
                help="Fiscal year",
                step=1,
                format="%d"
            ),
            "Revenue": st.column_config.NumberColumn(
                "Revenue",
                help="Annual revenue",
                step=1.0,
                format="%.2f"
            ),
            "Revenue_Growth": st.column_config.NumberColumn(
                "Revenue Growth",
                help="Year-over-year revenue growth rate",
                step=0.01,
                format="%.2f"
            ),
            "EBITDA": st.column_config.NumberColumn(
                "EBITDA",
                help="Earnings Before Interest, Taxes, Depreciation & Amortization",
                step=1.0,
                format="%.2f"
            ),
            "EBITDA_Margin": st.column_config.NumberColumn(
                "EBITDA Margin",
                help="EBITDA as a percentage of revenue",
                step=0.01,
                format="%.2f"
            ),
            "Depreciation_Amortization": st.column_config.NumberColumn(
                "Depreciation & Amortization",
                help="Annual depreciation and amortization expense",
                step=1.0,
                format="%.2f"
            ),
            "EBIT": st.column_config.NumberColumn(
                "EBIT",
                help="Earnings Before Interest and Taxes",
                step=1.0,
                format="%.2f"
            ),
            "Tax_Rate": st.column_config.NumberColumn(
                "Tax Rate",
                help="Effective tax rate",
                step=0.01,
                format="%.2f"
            ),
            "NOPAT": st.column_config.NumberColumn(
                "NOPAT",
                help="Net Operating Profit After Tax",
                step=1.0,
                format="%.2f"
            ),
            "Capital_Expenditures": st.column_config.NumberColumn(
                "Capital Expenditures",
                help="Annual capital expenditures",
                step=1.0,
                format="%.2f"
            ),
            "Change_in_Working_Capital": st.column_config.NumberColumn(
                "Change in Working Capital",
                help="Annual change in working capital",
                step=1.0,
                format="%.2f"
            ),
            "Free_Cash_Flow": st.column_config.NumberColumn(
                "Free Cash Flow",
                help="Annual free cash flow",
                step=1.0,
                format="%.2f"
            )
        }
    )
    
    # Save button
    if st.session_state.edited_data:
        st.button("Save Data", on_click=save_edited_data)

# Tab 2: Forecast
with tab2:
    st.header("Forecast")
    
    if 'forecast_data' in st.session_state:
        # Display forecast data
        st.subheader("Forecast Data")
        st.dataframe(st.session_state.forecast_data, use_container_width=True)
        
        # Create charts
        st.subheader("Forecast Charts")
        
        # Get forecast years
        forecast_years = st.session_state.model.data_manager.get_forecast_years(st.session_state.forecast_data)
        
        # Create metrics selection
        metrics = ["Revenue", "EBITDA", "EBIT", "NOPAT", "Free_Cash_Flow"]
        selected_metric = st.selectbox("Select Metric", options=metrics)
        
        # Create chart
        fig = create_historical_vs_forecast_chart(
            data=st.session_state.forecast_data,
            metric=selected_metric,
            forecast_years=forecast_years,
            use_plotly=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Run the valuation to see forecast data and charts.")

# Tab 3: Valuation Results
with tab3:
    st.header("Valuation Results")
    
    if st.session_state.results:
        # Create columns for key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Enterprise Value", 
                format_currency(st.session_state.results["enterprise_value"])
            )
        
        with col2:
            st.metric(
                "Equity Value", 
                format_currency(st.session_state.results["equity_value"])
            )
        
        with col3:
            st.metric(
                "Share Price", 
                format_currency(st.session_state.results["share_price"])
            )
        
        with col4:
            st.metric(
                "WACC", 
                format_percentage(st.session_state.results["wacc"] * 100)
            )
        
        # Create waterfall chart
        st.subheader("DCF Valuation Components")
        
        # Extract present values
        pv_forecast = sum(st.session_state.results["present_values"][:-1])
        pv_terminal = st.session_state.results["present_values"][-1]
        
        # Create waterfall chart components
        components = {
            "PV of Forecast": pv_forecast,
            "PV of Terminal Value": pv_terminal,
            "Enterprise Value": st.session_state.results["enterprise_value"],
            "Debt": -st.session_state.model.parameters["debt_value"],
            "Cash": st.session_state.model.parameters["cash_equivalents"],
            "Equity Value": st.session_state.results["equity_value"]
        }
        
        fig = create_dcf_waterfall_chart(components, use_plotly=True)
        st.plotly_chart(fig, use_container_width=True)
        
        # Create cash flow chart
        st.subheader("Cash Flows and Present Values")
        
        # Get forecast years
        forecast_years = st.session_state.model.data_manager.get_forecast_years(st.session_state.forecast_data)
        
        # Get cash flows
        cash_flows = []
        for year in forecast_years:
            year_data = st.session_state.forecast_data[st.session_state.forecast_data['Year'] == year].iloc[0]
            cash_flows.append(year_data['Free_Cash_Flow'])
        
        # Create chart
        fig = create_cash_flow_chart(
            years=forecast_years,
            cash_flows=cash_flows,
            present_values=st.session_state.results["present_values"][:-1],  # Exclude terminal value
            use_plotly=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Create sensitivity analysis
        st.subheader("Sensitivity Analysis")
        
        # Create columns for sensitivity parameters
        col1, col2 = st.columns(2)
        
        with col1:
            x_param = st.selectbox(
                "X-Axis Parameter",
                options=["WACC", "Terminal Growth Rate", "EBITDA Margin", "Revenue Growth"],
                index=0
            )
        
        with col2:
            y_param = st.selectbox(
                "Y-Axis Parameter",
                options=["Terminal Growth Rate", "WACC", "EBITDA Margin", "Revenue Growth"],
                index=0
            )
        
        # Ensure different parameters are selected
        if x_param == y_param:
            st.warning("Please select different parameters for X and Y axes.")
        else:
            # Define parameter ranges
            param_ranges = {
                "WACC": [max(0.05, st.session_state.results["wacc"] - 0.03), 
                         st.session_state.results["wacc"] - 0.02,
                         st.session_state.results["wacc"] - 0.01,
                         st.session_state.results["wacc"],
                         st.session_state.results["wacc"] + 0.01,
                         st.session_state.results["wacc"] + 0.02,
                         st.session_state.results["wacc"] + 0.03],
                "Terminal Growth Rate": [max(0.005, st.session_state.model.parameters["terminal_growth_rate"] - 0.015),
                                        st.session_state.model.parameters["terminal_growth_rate"] - 0.01,
                                        st.session_state.model.parameters["terminal_growth_rate"] - 0.005,
                                        st.session_state.model.parameters["terminal_growth_rate"],
                                        st.session_state.model.parameters["terminal_growth_rate"] + 0.005,
                                        st.session_state.model.parameters["terminal_growth_rate"] + 0.01,
                                        st.session_state.model.parameters["terminal_growth_rate"] + 0.015],
                "EBITDA Margin": [max(0.05, st.session_state.model.parameters["ebitda_margin"] - 0.06),
                                 st.session_state.model.parameters["ebitda_margin"] - 0.04,
                                 st.session_state.model.parameters["ebitda_margin"] - 0.02,
                                 st.session_state.model.parameters["ebitda_margin"],
                                 st.session_state.model.parameters["ebitda_margin"] + 0.02,
                                 st.session_state.model.parameters["ebitda_margin"] + 0.04,
                                 st.session_state.model.parameters["ebitda_margin"] + 0.06],
                "Revenue Growth": [max(0.01, st.session_state.model.parameters["revenue_growth"] - 0.06),
                                  st.session_state.model.parameters["revenue_growth"] - 0.04,
                                  st.session_state.model.parameters["revenue_growth"] - 0.02,
                                  st.session_state.model.parameters["revenue_growth"],
                                  st.session_state.model.parameters["revenue_growth"] + 0.02,
                                  st.session_state.model.parameters["revenue_growth"] + 0.04,
                                  st.session_state.model.parameters["revenue_growth"] + 0.06]
            }
            
            # Format labels
            x_labels = [f"{x*100:.1f}%" for x in param_ranges[x_param]]
            y_labels = [f"{y*100:.1f}%" for y in param_ranges[y_param]]
            
            # Create a dummy sensitivity matrix (in a real app, this would be calculated)
            # For demonstration, we'll create a simple matrix that shows higher values when both parameters are favorable
            z_values = []
            
            # Create a copy of the model to avoid modifying the original
            current_dir = os.path.dirname(os.path.abspath(__file__))
            data_dir = os.path.join(current_dir, "data")
            temp_model = DCFModel(data_dir=data_dir)
            temp_model.load_data()
            temp_model.set_parameters(st.session_state.model.get_parameters())
            
            # Calculate sensitivity values
            for y_val in param_ranges[y_param]:
                row = []
                for x_val in param_ranges[x_param]:
                    # Set parameters
                    params = {}
                    
                    if x_param == "WACC":
                        params["wacc"] = x_val
                        params["calculate_wacc"] = False
                    elif x_param == "Terminal Growth Rate":
                        params["terminal_growth_rate"] = x_val
                    elif x_param == "EBITDA Margin":
                        params["ebitda_margin"] = x_val
                    elif x_param == "Revenue Growth":
                        params["revenue_growth"] = x_val
                    
                    if y_param == "WACC":
                        params["wacc"] = y_val
                        params["calculate_wacc"] = False
                    elif y_param == "Terminal Growth Rate":
                        params["terminal_growth_rate"] = y_val
                    elif y_param == "EBITDA Margin":
                        params["ebitda_margin"] = y_val
                    elif y_param == "Revenue Growth":
                        params["revenue_growth"] = y_val
                    
                    temp_model.set_parameters(params)
                    
                    try:
                        # Run valuation
                        results = temp_model.run_dcf_valuation()
                        row.append(results["share_price"])
                    except Exception as e:
                        # Handle errors (e.g., when WACC <= growth rate)
                        row.append(None)
                
                z_values.append(row)
            
            # Create heatmap
            fig = create_sensitivity_analysis_heatmap(
                x_values=x_labels,
                y_values=y_labels,
                z_values=z_values,
                x_label=x_param,
                y_label=y_param,
                title=f"Share Price Sensitivity: {x_param} vs {y_param}",
                use_plotly=True
            )
            
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Run the valuation to see results.")

# Tab 4: Audit Trail
with tab4:
    st.header("Audit Trail")
    
    if st.session_state.results:
        # Toggle for showing calculation steps
        st.session_state.show_audit_trail = st.checkbox(
            "Show Detailed Calculation Steps",
            value=st.session_state.show_audit_trail
        )
        
        # Display audit trail
        if st.session_state.show_audit_trail:
            st.subheader("Detailed Calculation Steps")
            
            for step in st.session_state.results["calculation_steps"]:
                with st.expander(step["step"]):
                    st.write(f"**Formula:** {step['formula']}")
                    st.write("**Values:**")
                    for key, value in step["values"].items():
                        if isinstance(value, list):
                            st.write(f"- {key}: {', '.join([str(v) for v in value])}")
                        else:
                            st.write(f"- {key}: {value}")
                    st.write(f"**Result:** {step['result']}")
        else:
            st.subheader("Valuation Steps")
            
            for step in st.session_state.results["audit_trail"]:
                with st.expander(step["step"]):
                    for key, value in step["details"].items():
                        if isinstance(value, dict):
                            st.write(f"**{key}:**")
                            for k, v in value.items():
                                st.write(f"- {k}: {v}")
                        else:
                            st.write(f"**{key}:** {value}")
    else:
        st.info("Run the valuation to see the audit trail.")

# Footer
st.markdown("---")
st.caption("DCF Valuation Model - A Streamlit Application")
