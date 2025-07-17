import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import plotly.graph_objects as go
from typing import Dict, List, Any, Optional, Tuple

# Add the current directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dcf_tool.models.dcf_model import DCFModel
from dcf_tool.models.growth_methods import get_available_growth_methods
from dcf_tool.models.scenario_manager import ScenarioManager
from dcf_tool.utils.visualization import (
    create_historical_vs_forecast_chart,
    create_dcf_waterfall_chart,
    create_sensitivity_analysis_heatmap,
    create_cash_flow_chart
)
from dcf_tool.utils.advanced_visualization import (
    create_football_field_chart,
    create_tornado_chart,
    create_scenario_comparison_chart
)
from dcf_tool.utils.tooltips import (
    get_tooltip,
    get_industry_benchmark,
    get_formula_explanation,
    get_growth_stage_template,
    get_industry_template,
    get_all_industries,
    get_all_growth_stages
)

# Set page config
st.set_page_config(
    page_title="DCF Valuation Model",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Helper function to display tooltips
def display_tooltip(label, tooltip_key):
    tooltip = get_tooltip(tooltip_key)
    if tooltip:
        return f"{label} {st.info(tooltip)}"
    return label

# Helper function to display formula explanations
def display_formula(formula_key):
    explanation = get_formula_explanation(formula_key)
    if explanation:
        st.markdown(f"**{explanation.get('formula', '')}**")
        st.write("**Variables:**")
        for var, desc in explanation.get('variables', {}).items():
            st.write(f"- **{var}**: {desc}")
        st.write(f"**Explanation:** {explanation.get('explanation', '')}")

# Initialize session state
if 'model' not in st.session_state:
    # Get the absolute path to the data directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, "data")
    st.session_state.model = DCFModel(data_dir=data_dir)
    st.session_state.model.load_data()

if 'scenario_manager' not in st.session_state:
    # Initialize scenario manager
    current_dir = os.path.dirname(os.path.abspath(__file__))
    st.session_state.scenario_manager = ScenarioManager()

if 'data' not in st.session_state:
    st.session_state.data = st.session_state.model.data.copy()

if 'results' not in st.session_state:
    st.session_state.results = None

if 'show_audit_trail' not in st.session_state:
    st.session_state.show_audit_trail = False

if 'edited_data' not in st.session_state:
    st.session_state.edited_data = False

if 'active_scenario' not in st.session_state:
    st.session_state.active_scenario = None

if 'selected_scenarios' not in st.session_state:
    st.session_state.selected_scenarios = []

if 'show_tooltips' not in st.session_state:
    st.session_state.show_tooltips = True

if 'selected_industry' not in st.session_state:
    st.session_state.selected_industry = None

if 'selected_growth_stage' not in st.session_state:
    st.session_state.selected_growth_stage = None

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

# Function to save a scenario
def save_scenario():
    if not st.session_state.scenario_name:
        st.error("Please enter a scenario name")
        return
    
    # Update parameters before saving
    update_parameters()
    
    # Save scenario
    scenario_path = st.session_state.scenario_manager.save_scenario(
        name=st.session_state.scenario_name,
        description=st.session_state.scenario_description,
        parameters=st.session_state.model.get_parameters(),
        results=st.session_state.results,
        data=st.session_state.data
    )
    
    st.success(f"Scenario '{st.session_state.scenario_name}' saved successfully!")
    
    # Reset inputs
    st.session_state.scenario_name = ""
    st.session_state.scenario_description = ""
    
    # Refresh available scenarios
    st.session_state.available_scenarios = st.session_state.scenario_manager.get_available_scenarios()

# Function to load a scenario
def load_scenario(scenario_path):
    try:
        # Load scenario
        scenario = st.session_state.scenario_manager.load_scenario(scenario_path)
        
        # Update model parameters
        if "parameters" in scenario:
            st.session_state.model.set_parameters(scenario["parameters"])
            
            # Update session state with parameters
            for param, value in scenario["parameters"].items():
                if param in ["revenue_growth", "ebitda_margin", "tax_rate", "da_to_revenue", 
                           "capex_to_revenue", "wc_to_revenue", "terminal_growth_rate", 
                           "risk_free_rate", "market_risk_premium", "cost_of_debt", "wacc"]:
                    # Convert decimal to percentage for display
                    st.session_state[param] = value * 100
                else:
                    st.session_state[param] = value
        
        # Update data if available
        if "data" in scenario:
            st.session_state.data = scenario["data"]
            st.session_state.model.data = scenario["data"].copy()
        
        # Update results if available
        if "results" in scenario:
            st.session_state.results = scenario["results"]
            
            # If forecast data is in results, update it
            if "forecast_data" in scenario["results"]:
                forecast_data = pd.DataFrame(scenario["results"]["forecast_data"])
                st.session_state.forecast_data = forecast_data
                st.session_state.model.forecast_data = forecast_data.copy()
        
        st.session_state.active_scenario = scenario["name"]
        st.success(f"Scenario '{scenario['name']}' loaded successfully!")
        
    except Exception as e:
        st.error(f"Error loading scenario: {str(e)}")

# Function to apply a template
def apply_template(template_type, template_name):
    if template_type == "industry":
        template = get_industry_template(template_name)
    else:  # growth_stage
        template = get_growth_stage_template(template_name)
    
    if not template:
        st.error(f"Template not found: {template_name}")
        return
    
    # Update session state with template parameters
    for param, value in template.items():
        if param in ["revenue_growth", "ebitda_margin", "tax_rate", "da_to_revenue", 
                   "capex_to_revenue", "wc_to_revenue", "terminal_growth_rate", 
                   "risk_free_rate", "market_risk_premium", "cost_of_debt", "wacc"]:
            # Convert decimal to percentage for display
            if param in st.session_state:
                st.session_state[param] = value * 100
        elif param in st.session_state:
            st.session_state[param] = value
    
    # Update model parameters
    update_parameters()
    
    st.success(f"{template_type.title()} template '{template_name}' applied successfully!")

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

# Sidebar with tabs for different parameter groups
with st.sidebar:
    sidebar_tabs = st.tabs(["Parameters", "Scenarios", "Templates", "Help"])
    
    # Parameters tab
    with sidebar_tabs[0]:
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
    
    # Scenarios tab
    with sidebar_tabs[1]:
        st.header("Scenario Management")
        
        # Create new scenario
        with st.expander("Create New Scenario", expanded=True):
            st.text_input("Scenario Name", key="scenario_name")
            st.text_area("Description", key="scenario_description")
            st.button("Save Current Scenario", on_click=save_scenario)
        
        # Load existing scenarios
        with st.expander("Load Scenario", expanded=True):
            # Get available scenarios
            if 'available_scenarios' not in st.session_state:
                st.session_state.available_scenarios = st.session_state.scenario_manager.get_available_scenarios()
            
            if st.session_state.available_scenarios:
                for scenario in st.session_state.available_scenarios:
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.write(f"**{scenario['name']}**")
                        st.caption(f"{scenario['description']}")
                    with col2:
                        if st.button("Load", key=f"load_{scenario['file_path']}"):
                            load_scenario(scenario['file_path'])
            else:
                st.info("No saved scenarios found.")
        
        # Compare scenarios
        with st.expander("Compare Scenarios", expanded=False):
            if st.session_state.available_scenarios:
                # Multi-select for scenarios to compare
                scenario_options = {s['name']: s['file_path'] for s in st.session_state.available_scenarios}
                selected_names = st.multiselect(
                    "Select scenarios to compare",
                    options=list(scenario_options.keys()),
                    key="compare_scenarios"
                )
                
                if selected_names:
                    st.session_state.selected_scenarios = [scenario_options[name] for name in selected_names]
                    st.info(f"Selected {len(selected_names)} scenarios for comparison. View comparison in the 'Scenario Comparison' tab.")
                else:
                    st.session_state.selected_scenarios = []
            else:
                st.info("No saved scenarios found.")
    
    # Templates tab
    with sidebar_tabs[2]:
        st.header("Templates")
        
        # Industry templates
        with st.expander("Industry Templates", expanded=True):
            industries = get_all_industries()
            selected_industry = st.selectbox(
                "Select Industry",
                options=industries,
                key="selected_industry"
            )
            
            if selected_industry:
                template = get_industry_template(selected_industry)
                if template:
                    st.write(f"**Description:** {template.get('description', '')}")
                    st.write("**Key Parameters:**")
                    st.write(f"- Revenue Growth: {template.get('revenue_growth', 0)*100:.1f}%")
                    st.write(f"- EBITDA Margin: {template.get('ebitda_margin', 0)*100:.1f}%")
                    st.write(f"- Terminal Growth: {template.get('terminal_growth_rate', 0)*100:.1f}%")
                    
                    if st.button("Apply Industry Template"):
                        apply_template("industry", selected_industry)
        
        # Growth stage templates
        with st.expander("Growth Stage Templates", expanded=True):
            growth_stages = get_all_growth_stages()
            selected_stage = st.selectbox(
                "Select Growth Stage",
                options=growth_stages,
                key="selected_growth_stage"
            )
            
            if selected_stage:
                template = get_growth_stage_template(selected_stage)
                if template:
                    st.write(f"**Description:** {template.get('description', '')}")
                    st.write("**Key Parameters:**")
                    st.write(f"- Revenue Growth: {template.get('revenue_growth', 0)*100:.1f}%")
                    st.write(f"- EBITDA Margin: {template.get('ebitda_margin', 0)*100:.1f}%")
                    st.write(f"- Terminal Growth: {template.get('terminal_growth_rate', 0)*100:.1f}%")
                    
                    if st.button("Apply Growth Stage Template"):
                        apply_template("growth_stage", selected_stage)
    
    # Help tab
    with sidebar_tabs[3]:
        st.header("Help & Documentation")
        
        st.checkbox("Show Tooltips", value=True, key="show_tooltips")
        
        with st.expander("DCF Valuation Guide", expanded=False):
            st.write("""
            **Discounted Cash Flow (DCF) Valuation** is a method used to estimate the value of an investment based on its expected future cash flows.
            
            **Key Steps:**
            1. Forecast future cash flows
            2. Determine the terminal value
            3. Apply a discount rate (WACC)
            4. Calculate the present value
            5. Adjust for debt and cash to get equity value
            """)
        
        with st.expander("Key Formulas", expanded=False):
            display_formula("wacc")
            display_formula("terminal_value_perpetuity")
            display_formula("terminal_value_exit_multiple")
            display_formula("free_cash_flow")

# Main content area with tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Financial Data", 
    "Forecast", 
    "Valuation Results", 
    "Advanced Visualization", 
    "Scenario Comparison", 
    "Audit Trail"
])

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
    
    if st.session_state.active_scenario:
        st.info(f"Active Scenario: {st.session_state.active_scenario}")
    
    if st.session_state.results:
        # Create columns for key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            metric_label = "Enterprise Value"
            if st.session_state.show_tooltips:
                st.markdown(f"**{metric_label}** ℹ️")
                st.info(get_tooltip("enterprise_value"))
            else:
                st.markdown(f"**{metric_label}**")
            st.markdown(f"### {format_currency(st.session_state.results['enterprise_value'])}")
        
        with col2:
            metric_label = "Equity Value"
            if st.session_state.show_tooltips:
                st.markdown(f"**{metric_label}** ℹ️")
                st.info(get_tooltip("equity_value"))
            else:
                st.markdown(f"**{metric_label}**")
            st.markdown(f"### {format_currency(st.session_state.results['equity_value'])}")
        
        with col3:
            metric_label = "Share Price"
            if st.session_state.show_tooltips:
                st.markdown(f"**{metric_label}** ℹ️")
                st.info(get_tooltip("share_price"))
            else:
                st.markdown(f"**{metric_label}**")
            st.markdown(f"### {format_currency(st.session_state.results['share_price'])}")
        
        with col4:
            metric_label = "WACC"
            if st.session_state.show_tooltips:
                st.markdown(f"**{metric_label}** ℹ️")
                st.info(get_tooltip("wacc"))
            else:
                st.markdown(f"**{metric_label}**")
            st.markdown(f"### {format_percentage(st.session_state.results['wacc'] * 100)}")
        
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

# Tab 4: Advanced Visualization
with tab4:
    st.header("Advanced Visualization")
    
    if st.session_state.results:
        # Football field chart
        st.subheader("Football Field Valuation")
        
        # Create valuation methods dictionary
        valuation_methods = {}
        
        # DCF valuation
        dcf_value = st.session_state.results["share_price"]
        dcf_range = (dcf_value * 0.9, dcf_value * 1.1)  # +/- 10%
        valuation_methods["DCF"] = dcf_range
        
        # EV/EBITDA multiple
        if 'forecast_data' in st.session_state:
            # Get the last forecast year
            forecast_years = st.session_state.model.data_manager.get_forecast_years(st.session_state.forecast_data)
            if forecast_years:
                last_year = max(forecast_years)
                last_year_data = st.session_state.forecast_data[st.session_state.forecast_data['Year'] == last_year].iloc[0]
                
                # Calculate EV/EBITDA valuation
                ebitda = last_year_data['EBITDA']
                
                # Use a range of multiples
                low_multiple = 8.0
                high_multiple = 12.0
                
                # Calculate enterprise values
                low_ev = ebitda * low_multiple
                high_ev = ebitda * high_multiple
                
                # Calculate equity values
                low_equity = low_ev - st.session_state.model.parameters["debt_value"] + st.session_state.model.parameters["cash_equivalents"]
                high_equity = high_ev - st.session_state.model.parameters["debt_value"] + st.session_state.model.parameters["cash_equivalents"]
                
                # Calculate share prices
                low_share_price = low_equity / st.session_state.model.parameters["shares_outstanding"]
                high_share_price = high_equity / st.session_state.model.parameters["shares_outstanding"]
                
                valuation_methods["EV/EBITDA"] = (low_share_price, high_share_price)
                
                # Calculate EV/EBIT valuation
                ebit = last_year_data['EBIT']
                
                # Use a range of multiples
                low_multiple = 10.0
                high_multiple = 15.0
                
                # Calculate enterprise values
                low_ev = ebit * low_multiple
                high_ev = ebit * high_multiple
                
                # Calculate equity values
                low_equity = low_ev - st.session_state.model.parameters["debt_value"] + st.session_state.model.parameters["cash_equivalents"]
                high_equity = high_ev - st.session_state.model.parameters["debt_value"] + st.session_state.model.parameters["cash_equivalents"]
                
                # Calculate share prices
                low_share_price = low_equity / st.session_state.model.parameters["shares_outstanding"]
                high_share_price = high_equity / st.session_state.model.parameters["shares_outstanding"]
                
                valuation_methods["EV/EBIT"] = (low_share_price, high_share_price)
        
        # Create football field chart
        fig = create_football_field_chart(
            valuation_methods=valuation_methods,
            current_value=dcf_value,
            title="Valuation Range by Method",
            use_plotly=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Tornado chart
        st.subheader("Tornado Chart - Sensitivity Analysis")
        
        # Define parameters to test
        parameters = {
            "WACC": (
                st.session_state.results["wacc"] - 0.02,
                st.session_state.results["wacc"],
                st.session_state.results["wacc"] + 0.02
            ),
            "Terminal Growth": (
                st.session_state.model.parameters["terminal_growth_rate"] - 0.01,
                st.session_state.model.parameters["terminal_growth_rate"],
                st.session_state.model.parameters["terminal_growth_rate"] + 0.01
            ),
            "Revenue Growth": (
                st.session_state.model.parameters["revenue_growth"] - 0.03,
                st.session_state.model.parameters["revenue_growth"],
                st.session_state.model.parameters["revenue_growth"] + 0.03
            ),
            "EBITDA Margin": (
                st.session_state.model.parameters["ebitda_margin"] - 0.03,
                st.session_state.model.parameters["ebitda_margin"],
                st.session_state.model.parameters["ebitda_margin"] + 0.03
            )
        }
        
        # Calculate values for each parameter
        values = {}
        base_value = st.session_state.results["share_price"]
        
        # Create a copy of the model to avoid modifying the original
        current_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(current_dir, "data")
        temp_model = DCFModel(data_dir=data_dir)
        temp_model.load_data()
        
        for param, (low, base, high) in parameters.items():
            # Calculate low value
            temp_model.set_parameters(st.session_state.model.get_parameters())
            
            if param == "WACC":
                temp_model.set_parameters({"wacc": low, "calculate_wacc": False})
            elif param == "Terminal Growth":
                temp_model.set_parameters({"terminal_growth_rate": low})
            elif param == "Revenue Growth":
                temp_model.set_parameters({"revenue_growth": low})
            elif param == "EBITDA Margin":
                temp_model.set_parameters({"ebitda_margin": low})
            
            try:
                low_result = temp_model.run_dcf_valuation()["share_price"]
            except:
                low_result = base_value * 0.8  # Fallback if calculation fails
            
            # Calculate high value
            temp_model.set_parameters(st.session_state.model.get_parameters())
            
            if param == "WACC":
                temp_model.set_parameters({"wacc": high, "calculate_wacc": False})
            elif param == "Terminal Growth":
                temp_model.set_parameters({"terminal_growth_rate": high})
            elif param == "Revenue Growth":
                temp_model.set_parameters({"revenue_growth": high})
            elif param == "EBITDA Margin":
                temp_model.set_parameters({"ebitda_margin": high})
            
            try:
                high_result = temp_model.run_dcf_valuation()["share_price"]
            except:
                high_result = base_value * 1.2  # Fallback if calculation fails
            
            values[param] = (low_result, high_result)
        
        # Create tornado chart
        fig = create_tornado_chart(
            parameters=parameters,
            base_value=base_value,
            values=values,
            title="Share Price Sensitivity",
            use_plotly=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Run the valuation to see advanced visualizations.")

# Tab 5: Scenario Comparison
with tab5:
    st.header("Scenario Comparison")
    
    if st.session_state.selected_scenarios:
        # Load selected scenarios
        scenarios = []
        scenario_names = []
        
        for path in st.session_state.selected_scenarios:
            try:
                scenario = st.session_state.scenario_manager.load_scenario(path)
                scenarios.append(scenario)
                scenario_names.append(scenario.get("name", "Unnamed"))
            except:
                st.warning(f"Failed to load scenario from {path}")
        
        if scenarios:
            # Compare key metrics
            st.subheader("Key Metrics Comparison")
            
            # Extract key metrics
            metrics = {
                "Share Price": [],
                "Enterprise Value": [],
                "Equity Value": [],
                "WACC": []
            }
            
            for scenario in scenarios:
                if "results" in scenario:
                    metrics["Share Price"].append(scenario["results"].get("share_price", 0))
                    metrics["Enterprise Value"].append(scenario["results"].get("enterprise_value", 0))
                    metrics["Equity Value"].append(scenario["results"].get("equity_value", 0))
                    metrics["WACC"].append(scenario["results"].get("wacc", 0) * 100)  # Convert to percentage
                else:
                    # Add placeholder values if results not available
                    for key in metrics:
                        metrics[key].append(0)
            
            # Create comparison chart
            fig = create_scenario_comparison_chart(
                scenarios=scenario_names,
                values=metrics,
                title="Scenario Comparison - Key Metrics",
                use_plotly=True
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Compare parameters
            st.subheader("Parameter Comparison")
            
            # Extract key parameters
            parameters = {
                "Revenue Growth (%)": [],
                "EBITDA Margin (%)": [],
                "Terminal Growth Rate (%)": [],
                "Exit Multiple": []
            }
            
            for scenario in scenarios:
                if "parameters" in scenario:
                    parameters["Revenue Growth (%)"].append(scenario["parameters"].get("revenue_growth", 0) * 100)
                    parameters["EBITDA Margin (%)"].append(scenario["parameters"].get("ebitda_margin", 0) * 100)
                    parameters["Terminal Growth Rate (%)"].append(scenario["parameters"].get("terminal_growth_rate", 0) * 100)
                    parameters["Exit Multiple"].append(scenario["parameters"].get("exit_multiple", 0))
                else:
                    # Add placeholder values if parameters not available
                    for key in parameters:
                        parameters[key].append(0)
            
            # Create comparison chart
            fig = create_scenario_comparison_chart(
                scenarios=scenario_names,
                values=parameters,
                title="Scenario Comparison - Key Parameters",
                use_plotly=True
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Display detailed comparison table
            st.subheader("Detailed Comparison")
            
            # Create comparison dataframe
            comparison_data = {}
            
            # Add scenario names
            comparison_data["Scenario"] = scenario_names
            
            # Add key metrics
            for metric, values in metrics.items():
                if metric == "WACC":
                    comparison_data[metric + " (%)"] = [f"{v:.2f}%" for v in values]
                else:
                    comparison_data[metric] = [f"${v:,.2f}" for v in values]
            
            # Add key parameters
            for param, values in parameters.items():
                if "(%)" in param:
                    comparison_data[param] = [f"{v:.2f}%" for v in values]
                else:
                    comparison_data[param] = [f"{v:.2f}" for v in values]
            
            # Create dataframe
            comparison_df = pd.DataFrame(comparison_data)
            
            # Display table
            st.dataframe(comparison_df, use_container_width=True)
        else:
            st.info("No valid scenarios to compare.")
    else:
        st.info("Select scenarios to compare in the Scenarios tab.")

# Tab 6: Audit Trail
with tab6:
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
