import os
import sys
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from typing import Dict, List, Any, Optional

# Add the current directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.financial_model import FinancialModel
from models.scenario_manager import ScenarioManager
from utils.visualization import (
    create_line_chart,
    create_bar_chart,
    create_historical_vs_forecast_chart,
    create_scenario_comparison_chart,
    create_statement_structure_chart,
    create_financial_metrics_chart
)

# Set page config
st.set_page_config(
    page_title="3-Statement Financial Model",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'model' not in st.session_state:
    # Get the absolute path to the data directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, "data")
    st.session_state.model = FinancialModel(data_dir=data_dir)

if 'excel_file_path' not in st.session_state:
    st.session_state.excel_file_path = None

if 'forecast_data' not in st.session_state:
    st.session_state.forecast_data = None

if 'current_scenario' not in st.session_state:
    st.session_state.current_scenario = "base"

if 'show_audit_trail' not in st.session_state:
    st.session_state.show_audit_trail = False

# Initialize model parameters in session state
if 'parameters' not in st.session_state:
    st.session_state.parameters = {
        "forecast_years": 5,
        "revenue_growth": 5.0,  # Percentage
        "ebitda_margin": 20.0,  # Percentage
        "net_margin": 10.0,     # Percentage
        "tax_rate": 25.0,       # Percentage
        "da_to_revenue": 5.0,   # Percentage
        "capex_to_revenue": 6.0, # Percentage
        "wc_to_revenue": 1.0,   # Percentage
        "ar_days": 45,
        "inventory_days": 30,
        "ap_days": 30,
        "cash_to_revenue": 10.0, # Percentage
        "debt_to_ebitda": 2.0,
        "dividend_payout_ratio": 0.0 # Percentage
    }

# Function to update model parameters
def update_parameters():
    parameters = {
        "forecast_years": st.session_state.forecast_years,
        "revenue_growth": st.session_state.revenue_growth / 100,  # Convert from percentage
        "ebitda_margin": st.session_state.ebitda_margin / 100,    # Convert from percentage
        "net_margin": st.session_state.net_margin / 100,          # Convert from percentage
        "tax_rate": st.session_state.tax_rate / 100,              # Convert from percentage
        "da_to_revenue": st.session_state.da_to_revenue / 100,    # Convert from percentage
        "capex_to_revenue": st.session_state.capex_to_revenue / 100, # Convert from percentage
        "wc_to_revenue": st.session_state.wc_to_revenue / 100,    # Convert from percentage
        "ar_days": st.session_state.ar_days,
        "inventory_days": st.session_state.inventory_days,
        "ap_days": st.session_state.ap_days,
        "cash_to_revenue": st.session_state.cash_to_revenue / 100, # Convert from percentage
        "debt_to_ebitda": st.session_state.debt_to_ebitda,
        "dividend_payout_ratio": st.session_state.dividend_payout_ratio / 100 # Convert from percentage
    }
    
    st.session_state.model.set_parameters(parameters)
    st.session_state.parameters = parameters

# Function to run the forecast
def run_forecast():
    update_parameters()
    
    # Set base parameters for scenario manager
    st.session_state.model.scenario_manager.set_base_parameters(st.session_state.parameters)
    
    # Set upside and downside parameters
    upside_percentage = st.session_state.upside_percentage / 100
    downside_percentage = st.session_state.downside_percentage / 100
    
    st.session_state.model.scenario_manager.set_upside_parameters(percentage_change=upside_percentage)
    st.session_state.model.scenario_manager.set_downside_parameters(percentage_change=downside_percentage)
    
    # Generate forecast for the current scenario
    st.session_state.forecast_data = st.session_state.model.generate_forecast(st.session_state.current_scenario)

# Function to handle file upload
def process_uploaded_file():
    if st.session_state.uploaded_file is not None:
        # Save the uploaded file to a temporary location
        file_path = os.path.join(st.session_state.model.data_manager.data_dir, st.session_state.uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(st.session_state.uploaded_file.getbuffer())
        
        # Load the data
        st.session_state.model.load_data(file_path)
        st.session_state.excel_file_path = file_path
        
        # Update parameters from historical data
        for key, value in st.session_state.model.parameters.items():
            if key in st.session_state.parameters:
                # Convert decimal values to percentages for display
                if key in ["revenue_growth", "ebitda_margin", "net_margin", "tax_rate", 
                          "da_to_revenue", "capex_to_revenue", "wc_to_revenue", 
                          "cash_to_revenue", "dividend_payout_ratio"]:
                    st.session_state[key] = value * 100
                else:
                    st.session_state[key] = value
        
        # Run the forecast
        run_forecast()

# Function to format currency values
def format_currency(value):
    if pd.isna(value):
        return ""
    return f"${value:,.2f}"

# Function to format percentage values
def format_percentage(value):
    if pd.isna(value):
        return ""
    return f"{value:.2f}%"

# Main app title
st.title("3-Statement Financial Model")

# Sidebar for file upload and model parameters
with st.sidebar:
    st.header("Data Input")
    
    # File upload
    st.file_uploader(
        "Upload Excel Financial Data",
        type=["xlsx", "xls"],
        key="uploaded_file",
        on_change=process_uploaded_file,
        help="Upload an Excel file containing financial statements"
    )
    
    if st.session_state.excel_file_path:
        st.success(f"Loaded: {os.path.basename(st.session_state.excel_file_path)}")
    
    st.header("Model Parameters")
    
    # Forecast parameters
    st.subheader("Forecast Parameters")
    st.slider(
        "Forecast Years", 
        min_value=1, 
        max_value=10, 
        value=st.session_state.parameters["forecast_years"],
        step=1,
        key="forecast_years",
        help="Number of years to forecast"
    )
    
    st.slider(
        "Revenue Growth (%)", 
        min_value=-20.0, 
        max_value=50.0, 
        value=st.session_state.parameters["revenue_growth"] * 100,
        step=0.5,
        key="revenue_growth",
        help="Annual revenue growth rate"
    )
    
    st.slider(
        "EBITDA Margin (%)", 
        min_value=0.0, 
        max_value=50.0, 
        value=st.session_state.parameters["ebitda_margin"] * 100,
        step=0.5,
        key="ebitda_margin",
        help="EBITDA as a percentage of revenue"
    )
    
    st.slider(
        "Net Margin (%)", 
        min_value=0.0, 
        max_value=30.0, 
        value=st.session_state.parameters["net_margin"] * 100,
        step=0.5,
        key="net_margin",
        help="Net Income as a percentage of revenue"
    )
    
    st.slider(
        "Tax Rate (%)", 
        min_value=0.0, 
        max_value=50.0, 
        value=st.session_state.parameters["tax_rate"] * 100,
        step=0.5,
        key="tax_rate",
        help="Corporate tax rate"
    )
    
    # Balance Sheet parameters
    st.subheader("Balance Sheet Parameters")
    
    st.slider(
        "Depreciation & Amortization to Revenue (%)", 
        min_value=0.0, 
        max_value=20.0, 
        value=st.session_state.parameters["da_to_revenue"] * 100,
        step=0.5,
        key="da_to_revenue",
        help="Depreciation & Amortization as a percentage of revenue"
    )
    
    st.slider(
        "Capital Expenditures to Revenue (%)", 
        min_value=0.0, 
        max_value=20.0, 
        value=st.session_state.parameters["capex_to_revenue"] * 100,
        step=0.5,
        key="capex_to_revenue",
        help="Capital Expenditures as a percentage of revenue"
    )
    
    st.slider(
        "Working Capital Change to Revenue (%)", 
        min_value=-5.0, 
        max_value=10.0, 
        value=st.session_state.parameters["wc_to_revenue"] * 100,
        step=0.5,
        key="wc_to_revenue",
        help="Change in Working Capital as a percentage of revenue"
    )
    
    st.slider(
        "Cash to Revenue (%)", 
        min_value=0.0, 
        max_value=30.0, 
        value=st.session_state.parameters["cash_to_revenue"] * 100,
        step=0.5,
        key="cash_to_revenue",
        help="Cash as a percentage of revenue"
    )
    
    st.slider(
        "Debt to EBITDA Ratio", 
        min_value=0.0, 
        max_value=5.0, 
        value=st.session_state.parameters["debt_to_ebitda"],
        step=0.1,
        key="debt_to_ebitda",
        help="Total Debt to EBITDA ratio"
    )
    
    # Working Capital parameters
    st.subheader("Working Capital Parameters")
    
    st.slider(
        "Days Sales Outstanding (DSO)", 
        min_value=0, 
        max_value=90, 
        value=st.session_state.parameters["ar_days"],
        step=1,
        key="ar_days",
        help="Average number of days to collect payment"
    )
    
    st.slider(
        "Days Inventory Outstanding (DIO)", 
        min_value=0, 
        max_value=90, 
        value=st.session_state.parameters["inventory_days"],
        step=1,
        key="inventory_days",
        help="Average number of days inventory is held"
    )
    
    st.slider(
        "Days Payable Outstanding (DPO)", 
        min_value=0, 
        max_value=90, 
        value=st.session_state.parameters["ap_days"],
        step=1,
        key="ap_days",
        help="Average number of days to pay suppliers"
    )
    
    # Dividend parameters
    st.subheader("Dividend Parameters")
    
    st.slider(
        "Dividend Payout Ratio (%)", 
        min_value=0.0, 
        max_value=100.0, 
        value=st.session_state.parameters["dividend_payout_ratio"] * 100,
        step=1.0,
        key="dividend_payout_ratio",
        help="Percentage of net income paid as dividends"
    )
    
    # Scenario parameters
    st.subheader("Scenario Parameters")
    
    if 'upside_percentage' not in st.session_state:
        st.session_state.upside_percentage = 10.0
    
    if 'downside_percentage' not in st.session_state:
        st.session_state.downside_percentage = 10.0
    
    st.slider(
        "Upside Case Percentage Change (%)", 
        min_value=1.0, 
        max_value=50.0, 
        value=st.session_state.upside_percentage,
        step=1.0,
        key="upside_percentage",
        help="Percentage change for upside case parameters"
    )
    
    st.slider(
        "Downside Case Percentage Change (%)", 
        min_value=1.0, 
        max_value=50.0, 
        value=st.session_state.downside_percentage,
        step=1.0,
        key="downside_percentage",
        help="Percentage change for downside case parameters"
    )
    
    # Run forecast button
    st.button("Run Forecast", on_click=run_forecast, type="primary")

# Main content area with tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Data Input", "Income Statement", "Balance Sheet", "Cash Flow", "Scenarios"])

# Tab 1: Data Input
with tab1:
    st.header("Financial Data Input")
    
    if st.session_state.excel_file_path is None:
        st.info("Please upload an Excel file containing financial statements using the sidebar.")
    else:
        st.subheader("Historical Data")
        
        if st.session_state.model.historical_data is not None:
            st.dataframe(st.session_state.model.historical_data, use_container_width=True)
        
        st.subheader("Data Preview")
        
        # Create columns for the three statements
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write("Income Statement")
            if st.session_state.model.historical_data is not None:
                income_cols = ["Year"] + [col for col in st.session_state.model.historical_data.columns if col.startswith("IS_")]
                if len(income_cols) > 1:
                    st.dataframe(st.session_state.model.historical_data[income_cols].head(), use_container_width=True)
                else:
                    st.warning("No Income Statement data found")
        
        with col2:
            st.write("Balance Sheet")
            if st.session_state.model.historical_data is not None:
                balance_cols = ["Year"] + [col for col in st.session_state.model.historical_data.columns if col.startswith("BS_")]
                if len(balance_cols) > 1:
                    st.dataframe(st.session_state.model.historical_data[balance_cols].head(), use_container_width=True)
                else:
                    st.warning("No Balance Sheet data found")
        
        with col3:
            st.write("Cash Flow Statement")
            if st.session_state.model.historical_data is not None:
                cash_flow_cols = ["Year"] + [col for col in st.session_state.model.historical_data.columns if col.startswith("CF_")]
                if len(cash_flow_cols) > 1:
                    st.dataframe(st.session_state.model.historical_data[cash_flow_cols].head(), use_container_width=True)
                else:
                    st.warning("No Cash Flow Statement data found")

# Tab 2: Income Statement
with tab2:
    st.header("Income Statement")
    
    if st.session_state.forecast_data is None:
        if st.session_state.excel_file_path is None:
            st.info("Please upload an Excel file containing financial statements using the sidebar.")
        else:
            st.info("Please run the forecast using the button in the sidebar.")
    else:
        # Scenario selection
        scenario = st.selectbox(
            "Select Scenario",
            options=["base", "upside", "downside"],
            index=0,
            key="income_statement_scenario",
            on_change=lambda: setattr(st.session_state, 'current_scenario', st.session_state.income_statement_scenario)
        )
        
        # Get the income statement for the selected scenario
        income_statement = st.session_state.model.get_income_statement(scenario)
        
        # Display the income statement
        st.dataframe(income_statement, use_container_width=True)
        
        # Create charts
        st.subheader("Income Statement Charts")
        
        # Create metrics selection
        metrics = ["Revenue", "EBITDA", "EBIT", "Net_Income"]
        selected_metric = st.selectbox("Select Metric", options=metrics, key="income_statement_metric")
        
        # Get historical end year
        historical_end_year = st.session_state.model.data_manager.historical_end_year
        
        # Create chart
        fig = create_historical_vs_forecast_chart(
            data=income_statement,
            year_column="Year",
            metric_column=selected_metric,
            historical_end_year=historical_end_year,
            title=f"Historical vs Forecast {selected_metric}"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Create income statement structure chart
        st.subheader("Income Statement Structure")
        
        # Select components to display
        if "Revenue" in income_statement.columns and "Cost_of_Revenue" in income_statement.columns:
            components = ["Cost_of_Revenue"]
            
            if "Operating_Expenses" in income_statement.columns:
                components.append("Operating_Expenses")
            
            if "Depreciation_Amortization" in income_statement.columns:
                components.append("Depreciation_Amortization")
            
            if "Interest_Expense" in income_statement.columns:
                components.append("Interest_Expense")
            
            if "Income_Tax" in income_statement.columns:
                components.append("Income_Tax")
            
            if "Net_Income" in income_statement.columns:
                components.append("Net_Income")
            
            # Create chart
            fig = create_statement_structure_chart(
                data=income_statement,
                year_column="Year",
                component_columns=components,
                title="Income Statement Structure"
            )
            
            st.plotly_chart(fig, use_container_width=True)

# Tab 3: Balance Sheet
with tab3:
    st.header("Balance Sheet")
    
    if st.session_state.forecast_data is None:
        if st.session_state.excel_file_path is None:
            st.info("Please upload an Excel file containing financial statements using the sidebar.")
        else:
            st.info("Please run the forecast using the button in the sidebar.")
    else:
        # Scenario selection
        scenario = st.selectbox(
            "Select Scenario",
            options=["base", "upside", "downside"],
            index=0,
            key="balance_sheet_scenario",
            on_change=lambda: setattr(st.session_state, 'current_scenario', st.session_state.balance_sheet_scenario)
        )
        
        # Get the balance sheet for the selected scenario
        balance_sheet = st.session_state.model.get_balance_sheet(scenario)
        
        # Display the balance sheet
        st.dataframe(balance_sheet, use_container_width=True)
        
        # Create charts
        st.subheader("Balance Sheet Charts")
        
        # Create metrics selection
        metrics = ["Total_Assets", "Total_Liabilities", "Total_Equity"]
        selected_metric = st.selectbox("Select Metric", options=metrics, key="balance_sheet_metric")
        
        # Get historical end year
        historical_end_year = st.session_state.model.data_manager.historical_end_year
        
        # Create chart
        fig = create_historical_vs_forecast_chart(
            data=balance_sheet,
            year_column="Year",
            metric_column=selected_metric,
            historical_end_year=historical_end_year,
            title=f"Historical vs Forecast {selected_metric}"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Create balance sheet structure charts
        st.subheader("Balance Sheet Structure")
        
        # Create columns for assets and liabilities/equity
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("Assets")
            
            # Select asset components to display
            asset_components = []
            
            if "Cash_Equivalents" in balance_sheet.columns:
                asset_components.append("Cash_Equivalents")
            
            if "Accounts_Receivable" in balance_sheet.columns:
                asset_components.append("Accounts_Receivable")
            
            if "Inventory" in balance_sheet.columns:
                asset_components.append("Inventory")
            
            if "Short_Term_Investments" in balance_sheet.columns:
                asset_components.append("Short_Term_Investments")
            
            if "Other_Current_Assets" in balance_sheet.columns:
                asset_components.append("Other_Current_Assets")
            
            if "PPE" in balance_sheet.columns:
                asset_components.append("PPE")
            
            if "Goodwill" in balance_sheet.columns:
                asset_components.append("Goodwill")
            
            if "Intangible_Assets" in balance_sheet.columns:
                asset_components.append("Intangible_Assets")
            
            if "Long_Term_Investments" in balance_sheet.columns:
                asset_components.append("Long_Term_Investments")
            
            if "Other_Noncurrent_Assets" in balance_sheet.columns:
                asset_components.append("Other_Noncurrent_Assets")
            
            # Create chart
            if asset_components:
                fig = create_statement_structure_chart(
                    data=balance_sheet,
                    year_column="Year",
                    component_columns=asset_components,
                    title="Asset Structure"
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.write("Liabilities & Equity")
            
            # Select liability and equity components to display
            liability_equity_components = []
            
            if "Accounts_Payable" in balance_sheet.columns:
                liability_equity_components.append("Accounts_Payable")
            
            if "Short_Term_Debt" in balance_sheet.columns:
                liability_equity_components.append("Short_Term_Debt")
            
            if "Deferred_Revenue" in balance_sheet.columns:
                liability_equity_components.append("Deferred_Revenue")
            
            if "Other_Current_Liabilities" in balance_sheet.columns:
                liability_equity_components.append("Other_Current_Liabilities")
            
            if "Long_Term_Debt" in balance_sheet.columns:
                liability_equity_components.append("Long_Term_Debt")
            
            if "Deferred_Tax_Liabilities" in balance_sheet.columns:
                liability_equity_components.append("Deferred_Tax_Liabilities")
            
            if "Other_Noncurrent_Liabilities" in balance_sheet.columns:
                liability_equity_components.append("Other_Noncurrent_Liabilities")
            
            if "Common_Stock" in balance_sheet.columns:
                liability_equity_components.append("Common_Stock")
            
            if "Retained_Earnings" in balance_sheet.columns:
                liability_equity_components.append("Retained_Earnings")
            
            if "Treasury_Stock" in balance_sheet.columns:
                liability_equity_components.append("Treasury_Stock")
            
            if "Other_Equity" in balance_sheet.columns:
                liability_equity_components.append("Other_Equity")
            
            # Create chart
            if liability_equity_components:
                fig = create_statement_structure_chart(
                    data=balance_sheet,
                    year_column="Year",
                    component_columns=liability_equity_components,
                    title="Liability & Equity Structure"
                )
                
                st.plotly_chart(fig, use_container_width=True)

# Tab 4: Cash Flow
with tab4:
    st.header("Cash Flow Statement")
    
    if st.session_state.forecast_data is None:
        if st.session_state.excel_file_path is None:
            st.info("Please upload an Excel file containing financial statements using the sidebar.")
        else:
            st.info("Please run the forecast using the button in the sidebar.")
    else:
        # Scenario selection
        scenario = st.selectbox(
            "Select Scenario",
            options=["base", "upside", "downside"],
            index=0,
            key="cash_flow_scenario",
            on_change=lambda: setattr(st.session_state, 'current_scenario', st.session_state.cash_flow_scenario)
        )
        
        # Get the cash flow statement for the selected scenario
        cash_flow = st.session_state.model.get_cash_flow(scenario)
        
        # Display the cash flow statement
        st.dataframe(cash_flow, use_container_width=True)
        
        # Create charts
        st.subheader("Cash Flow Charts")
        
        # Create metrics selection
        metrics = ["Net_Cash_From_Operating", "Net_Cash_From_Investing", "Net_Cash_From_Financing", "Free_Cash_Flow"]
        selected_metric = st.selectbox("Select Metric", options=metrics, key="cash_flow_metric")
        
        # Get historical end year
        historical_end_year = st.session_state.model.data_manager.historical_end_year
        
        # Create chart
        fig = create_historical_vs_forecast_chart(
            data=cash_flow,
            year_column="Year",
            metric_column=selected_metric,
            historical_end_year=historical_end_year,
            title=f"Historical vs Forecast {selected_metric}"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Create cash flow components chart
        st.subheader("Cash Flow Components")
        
        # Select components to display
        components = []
        
        if "Net_Income" in cash_flow.columns:
            components.append("Net_Income")
        
        if "Depreciation_Amortization" in cash_flow.columns:
            components.append("Depreciation_Amortization")
        
        if "Change_In_Working_Capital" in cash_flow.columns:
            components.append("Change_In_Working_Capital")
        
        if "Capital_Expenditures" in cash_flow.columns:
            components.append("Capital_Expenditures")
        
        if "Debt_Issuance" in cash_flow.columns:
            components.append("Debt_Issuance")
        
        if "Debt_Repayment" in cash_flow.columns:
            components.append("Debt_Repayment")
        
        if "Dividends_Paid" in cash_flow.columns:
            components.append("Dividends_Paid")
        
        # Create chart
        if components:
            # Create a line chart for each component
            fig = create_line_chart(
                data=cash_flow,
                x_column="Year",
                y_columns=components,
                title="Cash Flow Components"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Create free cash flow chart
        if "Free_Cash_Flow" in cash_flow.columns:
            st.subheader("Free Cash Flow")
            
            fig = create_bar_chart(
                data=cash_flow,
                x_column="Year",
                y_columns=["Free_Cash_Flow"],
                title="Free Cash Flow",
                colors=["green"]
            )
            
            st.plotly_chart(fig, use_container_width=True)

# Tab 5: Scenarios
with tab5:
    st.header("Scenario Analysis")
    
    if st.session_state.forecast_data is None:
        if st.session_state.excel_file_path is None:
            st.info("Please upload an Excel file containing financial statements using the sidebar.")
        else:
            st.info("Please run the forecast using the button in the sidebar.")
    else:
        # Create metrics selection
        st.subheader("Compare Scenarios")
        
        # Create columns for statement and metric selection
        col1, col2 = st.columns(2)
        
        with col1:
            statement_type = st.selectbox(
                "Select Statement",
                options=["Income Statement", "Balance Sheet", "Cash Flow", "Metrics"],
                index=0,
                key="scenario_statement_type"
            )
        
        with col2:
            # Get available metrics based on statement type
            if statement_type == "Income Statement":
                income_statement = st.session_state.model.get_income_statement("base")
                available_metrics = income_statement.columns.tolist()
                available_metrics.remove("Year")
                metric_prefix = "IS_"
            elif statement_type == "Balance Sheet":
                balance_sheet = st.session_state.model.get_balance_sheet("base")
                available_metrics = balance_sheet.columns.tolist()
                available_metrics.remove("Year")
                metric_prefix = "BS_"
            elif statement_type == "Cash Flow":
                cash_flow = st.session_state.model.get_cash_flow("base")
                available_metrics = cash_flow.columns.tolist()
                available_metrics.remove("Year")
                metric_prefix = "CF_"
            else:  # Metrics
                metrics = st.session_state.model.get_financial_metrics("base")
                available_metrics = metrics.columns.tolist()
                available_metrics.remove("Year")
                metric_prefix = ""
            
            selected_metric = st.selectbox(
                "Select Metric",
                options=available_metrics,
                index=0,
                key="scenario_metric"
            )
        
        # Add the prefix to the metric if needed
        if metric_prefix:
            metric_with_prefix = f"{metric_prefix}{selected_metric}"
        else:
            metric_with_prefix = selected_metric
        
        # Compare scenarios
        comparison_df = st.session_state.model.compare_scenarios(metric_with_prefix)
        
        # Display the comparison data
        st.dataframe(comparison_df, use_container_width=True)
        
        # Create scenario comparison chart
        if metric_prefix:
            # Extract the column from each scenario's data
            base_data = st.session_state.model.generate_forecast("base")
            upside_data = st.session_state.model.generate_forecast("upside")
            downside_data = st.session_state.model.generate_forecast("downside")
            
            # Create a DataFrame for the chart
            chart_data = pd.DataFrame({
                "Year": base_data["Year"],
                "Base": base_data[metric_with_prefix],
                "Upside": upside_data[metric_with_prefix],
                "Downside": downside_data[metric_with_prefix]
            })
            
            # Create the chart
            fig = go.Figure()
            
            # Add base case
            fig.add_trace(
                go.Scatter(
                    x=chart_data["Year"],
                    y=chart_data["Base"],
                    mode='lines+markers',
                    name='Base Case',
                    line=dict(color="blue")
                )
            )
            
            # Add upside case
            fig.add_trace(
                go.Scatter(
                    x=chart_data["Year"],
                    y=chart_data["Upside"],
                    mode='lines+markers',
                    name='Upside Case',
                    line=dict(color="green")
                )
            )
            
            # Add downside case
            fig.add_trace(
                go.Scatter(
                    x=chart_data["Year"],
                    y=chart_data["Downside"],
                    mode='lines+markers',
                    name='Downside Case',
                    line=dict(color="red")
                )
            )
            
            fig.update_layout(
                title=f"Scenario Comparison: {selected_metric}",
                xaxis_title="Year",
                yaxis_title=selected_metric,
                showlegend=True,
                hovermode="x unified"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Software-specific metrics
        if statement_type == "Metrics":
            st.subheader("Software-Specific Metrics")
            
            # Get metrics for each scenario
            base_metrics = st.session_state.model.get_financial_metrics("base")
            upside_metrics = st.session_state.model.get_financial_metrics("upside")
            downside_metrics = st.session_state.model.get_financial_metrics("downside")
            
            # Check if we have software-specific metrics
            software_metrics = []
            
            if "Sales_Marketing_to_Revenue" in base_metrics.columns:
                software_metrics.append("Sales_Marketing_to_Revenue")
            
            if "R_D_to_Revenue" in base_metrics.columns:
                software_metrics.append("R_D_to_Revenue")
            
            if "Rule_of_40" in base_metrics.columns:
                software_metrics.append("Rule_of_40")
            
            if software_metrics:
                # Create a DataFrame for the metrics
                metrics_data = pd.DataFrame({"Year": base_metrics["Year"]})
                
                for metric in software_metrics:
                    metrics_data[f"Base_{metric}"] = base_metrics[metric]
                    metrics_data[f"Upside_{metric}"] = upside_metrics[metric]
                    metrics_data[f"Downside_{metric}"] = downside_metrics[metric]
                
                # Display the metrics
                st.dataframe(metrics_data, use_container_width=True)
                
                # Create charts for each metric
                for metric in software_metrics:
                    # Create a DataFrame for the chart
                    chart_data = pd.DataFrame({
                        "Year": base_metrics["Year"],
                        "Base": base_metrics[metric],
                        "Upside": upside_metrics[metric],
                        "Downside": downside_metrics[metric]
                    })
                    
                    # Create the chart
                    fig = go.Figure()
                    
                    # Add base case
                    fig.add_trace(
                        go.Scatter(
                            x=chart_data["Year"],
                            y=chart_data["Base"],
                            mode='lines+markers',
                            name='Base Case',
                            line=dict(color="blue")
                        )
                    )
                    
                    # Add upside case
                    fig.add_trace(
                        go.Scatter(
                            x=chart_data["Year"],
                            y=chart_data["Upside"],
                            mode='lines+markers',
                            name='Upside Case',
                            line=dict(color="green")
                        )
                    )
                    
                    # Add downside case
                    fig.add_trace(
                        go.Scatter(
                            x=chart_data["Year"],
                            y=chart_data["Downside"],
                            mode='lines+markers',
                            name='Downside Case',
                            line=dict(color="red")
                        )
                    )
                    
                    fig.update_layout(
                        title=f"Scenario Comparison: {metric}",
                        xaxis_title="Year",
                        yaxis_title=metric,
                        showlegend=True,
                        hovermode="x unified"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No software-specific metrics available.")

# Footer
st.markdown("---")
st.caption("3-Statement Financial Model - A Streamlit Application")
