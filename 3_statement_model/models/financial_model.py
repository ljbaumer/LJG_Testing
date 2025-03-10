import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any

import os
import sys
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.data_manager import DataManager
from models.scenario_manager import ScenarioManager
from utils.financial_utils import (
    calculate_growth_rate,
    calculate_margin,
    calculate_percentage_of_revenue,
    calculate_working_capital,
    calculate_change_in_working_capital,
    calculate_free_cash_flow
)


class FinancialModel:
    """
    Core 3-statement financial model implementation.
    """
    
    def __init__(self, data_dir: str = "../data"):
        """
        Initialize the financial model.
        
        Args:
            data_dir: Directory path where data files are stored
        """
        self.data_manager = DataManager(data_dir)
        self.scenario_manager = ScenarioManager()
        
        # Initialize data containers
        self.historical_data = None
        self.forecast_data = None
        self.combined_data = None
        
        # Initialize parameters
        self.parameters = {
            # Default parameters
            "forecast_years": 5,
            "revenue_growth": 0.05,
            "ebitda_margin": 0.20,
            "net_margin": 0.10,
            "tax_rate": 0.25,
            "da_to_revenue": 0.05,
            "capex_to_revenue": 0.06,
            "wc_to_revenue": 0.01,
            "ar_days": 45,
            "inventory_days": 30,
            "ap_days": 30,
            "cash_to_revenue": 0.10,
            "debt_to_ebitda": 2.0,
            "dividend_payout_ratio": 0.0
        }
        
        # Initialize audit trail
        self.audit_trail = []
    
    def load_data(self, excel_path: str) -> None:
        """
        Load financial data from an Excel file.
        
        Args:
            excel_path: Path to the Excel file
        """
        # Load the data
        self.data_manager.load_excel_file(excel_path)
        
        # Standardize and combine the data
        self.data_manager.standardize_data()
        self.historical_data = self.data_manager.combine_statements()
        
        # Calculate historical metrics
        historical_metrics = self.data_manager.calculate_historical_metrics()
        
        # Update parameters with historical metrics
        for metric, value in historical_metrics.items():
            if metric == "avg_revenue_growth":
                self.parameters["revenue_growth"] = value
            elif metric == "avg_ebitda_margin":
                self.parameters["ebitda_margin"] = value
            elif metric == "avg_net_margin":
                self.parameters["net_margin"] = value
            elif metric == "avg_tax_rate":
                self.parameters["tax_rate"] = value
            elif metric == "avg_da_to_revenue":
                self.parameters["da_to_revenue"] = value
            elif metric == "avg_capex_to_revenue":
                self.parameters["capex_to_revenue"] = value
            elif metric == "avg_wc_to_revenue":
                self.parameters["wc_to_revenue"] = value
            elif metric == "avg_dso":
                self.parameters["ar_days"] = value
            elif metric == "avg_dio":
                self.parameters["inventory_days"] = value
            elif metric == "avg_dpo":
                self.parameters["ap_days"] = value
        
        # Set base parameters for scenario manager
        self.scenario_manager.set_base_parameters(self.parameters)
        
        # Set upside and downside parameters
        self.scenario_manager.set_upside_parameters()
        self.scenario_manager.set_downside_parameters()
    
    def set_parameters(self, parameters: Dict[str, Any]) -> None:
        """
        Set model parameters.
        
        Args:
            parameters: Dictionary of parameter names and values
        """
        # Update only the parameters that are provided
        for key, value in parameters.items():
            if key in self.parameters:
                self.parameters[key] = value
    
    def get_parameters(self) -> Dict[str, Any]:
        """
        Get current model parameters.
        
        Returns:
            Dictionary of parameter names and values
        """
        return self.parameters.copy()
    
    def _add_audit_step(self, step: str, details: Dict[str, Any]) -> None:
        """
        Add a step to the audit trail.
        
        Args:
            step: Name of the step
            details: Dictionary of details about the step
        """
        self.audit_trail.append({
            "step": step,
            "details": details
        })
    
    def generate_forecast(self, scenario_name: str = "base") -> pd.DataFrame:
        """
        Generate forecast data based on the specified scenario.
        
        Args:
            scenario_name: Name of the scenario to use
            
        Returns:
            DataFrame containing the forecast data
        """
        # Get the scenario parameters
        if scenario_name != "base":
            parameters = self.scenario_manager.get_scenario_parameters(scenario_name)
            self.set_parameters(parameters)
        
        # Extend the historical data with forecast years
        extended_data = self.data_manager.extend_forecast_years(self.parameters["forecast_years"])
        
        # Get the forecast years
        forecast_years = self.data_manager.get_forecast_years(extended_data)
        
        # Get the last historical year's data
        historical_end_year = self.data_manager.historical_end_year
        last_historical_data = extended_data[extended_data["Year"] == historical_end_year].iloc[0]
        
        # Forecast data for future years
        for i, year in enumerate(forecast_years):
            forecast_index = extended_data[extended_data["Year"] == year].index[0]
            
            # Income Statement Forecasting
            
            # Revenue forecast with growth rate
            if i == 0:
                previous_revenue = last_historical_data["IS_Revenue"]
            else:
                previous_revenue = extended_data.loc[forecast_index - 1, "IS_Revenue"]
            
            revenue = previous_revenue * (1 + self.parameters["revenue_growth"])
            extended_data.loc[forecast_index, "IS_Revenue"] = revenue
            extended_data.loc[forecast_index, "IS_Revenue_Growth"] = self.parameters["revenue_growth"]
            
            # EBITDA forecast based on margin
            ebitda = revenue * self.parameters["ebitda_margin"]
            extended_data.loc[forecast_index, "IS_EBITDA"] = ebitda
            extended_data.loc[forecast_index, "IS_EBITDA_Margin"] = self.parameters["ebitda_margin"]
            
            # Depreciation & Amortization forecast
            da = revenue * self.parameters["da_to_revenue"]
            extended_data.loc[forecast_index, "IS_Depreciation_Amortization"] = da
            
            # EBIT forecast
            ebit = ebitda - da
            extended_data.loc[forecast_index, "IS_EBIT"] = ebit
            
            # Interest Expense forecast (based on debt)
            if "BS_Long_Term_Debt" in extended_data.columns and "BS_Short_Term_Debt" in extended_data.columns:
                if i == 0:
                    previous_debt = last_historical_data["BS_Long_Term_Debt"] + last_historical_data["BS_Short_Term_Debt"]
                else:
                    previous_debt = extended_data.loc[forecast_index - 1, "BS_Long_Term_Debt"] + extended_data.loc[forecast_index - 1, "BS_Short_Term_Debt"]
                
                # Assume interest rate of 5%
                interest_expense = previous_debt * 0.05
                extended_data.loc[forecast_index, "IS_Interest_Expense"] = interest_expense
            else:
                # If debt data is not available, assume interest expense is a percentage of EBIT
                interest_expense = ebit * 0.1
                extended_data.loc[forecast_index, "IS_Interest_Expense"] = interest_expense
            
            # Pretax Income forecast
            pretax_income = ebit - interest_expense
            extended_data.loc[forecast_index, "IS_Pretax_Income"] = pretax_income
            
            # Income Tax forecast
            income_tax = pretax_income * self.parameters["tax_rate"]
            extended_data.loc[forecast_index, "IS_Income_Tax"] = income_tax
            extended_data.loc[forecast_index, "IS_Tax_Rate"] = self.parameters["tax_rate"]
            
            # Net Income forecast
            net_income = pretax_income - income_tax
            extended_data.loc[forecast_index, "IS_Net_Income"] = net_income
            extended_data.loc[forecast_index, "IS_Net_Margin"] = net_income / revenue
            
            # Balance Sheet Forecasting
            
            # Cash forecast
            cash = revenue * self.parameters["cash_to_revenue"]
            extended_data.loc[forecast_index, "BS_Cash_Equivalents"] = cash
            
            # Accounts Receivable forecast
            ar = (revenue / 365) * self.parameters["ar_days"]
            extended_data.loc[forecast_index, "BS_Accounts_Receivable"] = ar
            
            # Inventory forecast
            if "IS_Cost_of_Revenue" in extended_data.columns:
                cogs = revenue * (1 - self.parameters["ebitda_margin"])  # Simplified
                extended_data.loc[forecast_index, "IS_Cost_of_Revenue"] = cogs
                inventory = (cogs / 365) * self.parameters["inventory_days"]
                extended_data.loc[forecast_index, "BS_Inventory"] = inventory
            
            # Total Current Assets forecast
            if "BS_Short_Term_Investments" in extended_data.columns and "BS_Other_Current_Assets" in extended_data.columns:
                # Assume short-term investments and other current assets grow with revenue
                st_investments = revenue * 0.05  # Simplified
                other_ca = revenue * 0.03  # Simplified
                extended_data.loc[forecast_index, "BS_Short_Term_Investments"] = st_investments
                extended_data.loc[forecast_index, "BS_Other_Current_Assets"] = other_ca
                
                total_ca = cash + ar + inventory + st_investments + other_ca
                extended_data.loc[forecast_index, "BS_Total_Current_Assets"] = total_ca
            
            # PPE forecast
            if "BS_PPE" in extended_data.columns:
                if i == 0:
                    previous_ppe = last_historical_data["BS_PPE"]
                else:
                    previous_ppe = extended_data.loc[forecast_index - 1, "BS_PPE"]
                
                capex = revenue * self.parameters["capex_to_revenue"]
                ppe = previous_ppe + capex - da
                extended_data.loc[forecast_index, "BS_PPE"] = ppe
            
            # Goodwill and Intangible Assets forecast (assume no change)
            if "BS_Goodwill" in extended_data.columns:
                if i == 0:
                    goodwill = last_historical_data["BS_Goodwill"]
                else:
                    goodwill = extended_data.loc[forecast_index - 1, "BS_Goodwill"]
                
                extended_data.loc[forecast_index, "BS_Goodwill"] = goodwill
            
            if "BS_Intangible_Assets" in extended_data.columns:
                if i == 0:
                    intangibles = last_historical_data["BS_Intangible_Assets"]
                else:
                    intangibles = extended_data.loc[forecast_index - 1, "BS_Intangible_Assets"]
                
                extended_data.loc[forecast_index, "BS_Intangible_Assets"] = intangibles
            
            # Other Non-current Assets forecast
            if "BS_Long_Term_Investments" in extended_data.columns and "BS_Other_Noncurrent_Assets" in extended_data.columns:
                # Assume long-term investments and other non-current assets grow with revenue
                lt_investments = revenue * 0.08  # Simplified
                other_nca = revenue * 0.04  # Simplified
                extended_data.loc[forecast_index, "BS_Long_Term_Investments"] = lt_investments
                extended_data.loc[forecast_index, "BS_Other_Noncurrent_Assets"] = other_nca
                
                # Total Non-current Assets forecast
                total_nca = ppe + goodwill + intangibles + lt_investments + other_nca
                extended_data.loc[forecast_index, "BS_Total_Noncurrent_Assets"] = total_nca
            
            # Total Assets forecast
            if "BS_Total_Current_Assets" in extended_data.columns and "BS_Total_Noncurrent_Assets" in extended_data.columns:
                total_assets = extended_data.loc[forecast_index, "BS_Total_Current_Assets"] + extended_data.loc[forecast_index, "BS_Total_Noncurrent_Assets"]
                extended_data.loc[forecast_index, "BS_Total_Assets"] = total_assets
            
            # Accounts Payable forecast
            if "IS_Cost_of_Revenue" in extended_data.columns:
                ap = (cogs / 365) * self.parameters["ap_days"]
                extended_data.loc[forecast_index, "BS_Accounts_Payable"] = ap
            
            # Short-term Debt forecast
            if "BS_Short_Term_Debt" in extended_data.columns:
                # Assume short-term debt is a percentage of total debt
                st_debt = ebitda * self.parameters["debt_to_ebitda"] * 0.2  # Assume 20% of total debt is short-term
                extended_data.loc[forecast_index, "BS_Short_Term_Debt"] = st_debt
            
            # Deferred Revenue forecast
            if "BS_Deferred_Revenue" in extended_data.columns:
                # Assume deferred revenue grows with revenue
                deferred_revenue = revenue * 0.1  # Simplified
                extended_data.loc[forecast_index, "BS_Deferred_Revenue"] = deferred_revenue
            
            # Other Current Liabilities forecast
            if "BS_Other_Current_Liabilities" in extended_data.columns:
                # Assume other current liabilities grow with revenue
                other_cl = revenue * 0.05  # Simplified
                extended_data.loc[forecast_index, "BS_Other_Current_Liabilities"] = other_cl
            
            # Total Current Liabilities forecast
            if "BS_Accounts_Payable" in extended_data.columns and "BS_Short_Term_Debt" in extended_data.columns and "BS_Deferred_Revenue" in extended_data.columns and "BS_Other_Current_Liabilities" in extended_data.columns:
                total_cl = ap + st_debt + deferred_revenue + other_cl
                extended_data.loc[forecast_index, "BS_Total_Current_Liabilities"] = total_cl
            
            # Long-term Debt forecast
            if "BS_Long_Term_Debt" in extended_data.columns:
                # Assume long-term debt is a percentage of total debt
                lt_debt = ebitda * self.parameters["debt_to_ebitda"] * 0.8  # Assume 80% of total debt is long-term
                extended_data.loc[forecast_index, "BS_Long_Term_Debt"] = lt_debt
            
            # Other Non-current Liabilities forecast
            if "BS_Deferred_Tax_Liabilities" in extended_data.columns and "BS_Other_Noncurrent_Liabilities" in extended_data.columns:
                # Assume deferred tax liabilities and other non-current liabilities grow with revenue
                deferred_tax = revenue * 0.02  # Simplified
                other_ncl = revenue * 0.03  # Simplified
                extended_data.loc[forecast_index, "BS_Deferred_Tax_Liabilities"] = deferred_tax
                extended_data.loc[forecast_index, "BS_Other_Noncurrent_Liabilities"] = other_ncl
            
            # Total Non-current Liabilities forecast
            if "BS_Long_Term_Debt" in extended_data.columns and "BS_Deferred_Tax_Liabilities" in extended_data.columns and "BS_Other_Noncurrent_Liabilities" in extended_data.columns:
                total_ncl = lt_debt + deferred_tax + other_ncl
                extended_data.loc[forecast_index, "BS_Total_Noncurrent_Liabilities"] = total_ncl
            
            # Total Liabilities forecast
            if "BS_Total_Current_Liabilities" in extended_data.columns and "BS_Total_Noncurrent_Liabilities" in extended_data.columns:
                total_liabilities = extended_data.loc[forecast_index, "BS_Total_Current_Liabilities"] + extended_data.loc[forecast_index, "BS_Total_Noncurrent_Liabilities"]
                extended_data.loc[forecast_index, "BS_Total_Liabilities"] = total_liabilities
            
            # Retained Earnings forecast
            if "BS_Retained_Earnings" in extended_data.columns:
                if i == 0:
                    previous_re = last_historical_data["BS_Retained_Earnings"]
                else:
                    previous_re = extended_data.loc[forecast_index - 1, "BS_Retained_Earnings"]
                
                # Calculate dividends
                dividends = net_income * self.parameters["dividend_payout_ratio"]
                
                # Update retained earnings
                retained_earnings = previous_re + net_income - dividends
                extended_data.loc[forecast_index, "BS_Retained_Earnings"] = retained_earnings
            
            # Common Stock, Treasury Stock, and Other Equity forecast (assume no change)
            if "BS_Common_Stock" in extended_data.columns:
                if i == 0:
                    common_stock = last_historical_data["BS_Common_Stock"]
                else:
                    common_stock = extended_data.loc[forecast_index - 1, "BS_Common_Stock"]
                
                extended_data.loc[forecast_index, "BS_Common_Stock"] = common_stock
            
            if "BS_Treasury_Stock" in extended_data.columns:
                if i == 0:
                    treasury_stock = last_historical_data["BS_Treasury_Stock"]
                else:
                    treasury_stock = extended_data.loc[forecast_index - 1, "BS_Treasury_Stock"]
                
                extended_data.loc[forecast_index, "BS_Treasury_Stock"] = treasury_stock
            
            if "BS_Other_Equity" in extended_data.columns:
                if i == 0:
                    other_equity = last_historical_data["BS_Other_Equity"]
                else:
                    other_equity = extended_data.loc[forecast_index - 1, "BS_Other_Equity"]
                
                extended_data.loc[forecast_index, "BS_Other_Equity"] = other_equity
            
            # Total Equity forecast
            if "BS_Common_Stock" in extended_data.columns and "BS_Retained_Earnings" in extended_data.columns and "BS_Treasury_Stock" in extended_data.columns and "BS_Other_Equity" in extended_data.columns:
                total_equity = common_stock + retained_earnings + treasury_stock + other_equity
                extended_data.loc[forecast_index, "BS_Total_Equity"] = total_equity
            
            # Cash Flow Statement Forecasting
            
            # Net Income
            extended_data.loc[forecast_index, "CF_Net_Income"] = net_income
            
            # Depreciation & Amortization
            extended_data.loc[forecast_index, "CF_Depreciation_Amortization"] = da
            
            # Change in Working Capital
            if i == 0:
                previous_ar = last_historical_data["BS_Accounts_Receivable"] if "BS_Accounts_Receivable" in last_historical_data else 0
                previous_inventory = last_historical_data["BS_Inventory"] if "BS_Inventory" in last_historical_data else 0
                previous_ap = last_historical_data["BS_Accounts_Payable"] if "BS_Accounts_Payable" in last_historical_data else 0
                previous_deferred_revenue = last_historical_data["BS_Deferred_Revenue"] if "BS_Deferred_Revenue" in last_historical_data else 0
            else:
                previous_ar = extended_data.loc[forecast_index - 1, "BS_Accounts_Receivable"] if "BS_Accounts_Receivable" in extended_data.columns else 0
                previous_inventory = extended_data.loc[forecast_index - 1, "BS_Inventory"] if "BS_Inventory" in extended_data.columns else 0
                previous_ap = extended_data.loc[forecast_index - 1, "BS_Accounts_Payable"] if "BS_Accounts_Payable" in extended_data.columns else 0
                previous_deferred_revenue = extended_data.loc[forecast_index - 1, "BS_Deferred_Revenue"] if "BS_Deferred_Revenue" in extended_data.columns else 0
            
            current_ar = extended_data.loc[forecast_index, "BS_Accounts_Receivable"] if "BS_Accounts_Receivable" in extended_data.columns else 0
            current_inventory = extended_data.loc[forecast_index, "BS_Inventory"] if "BS_Inventory" in extended_data.columns else 0
            current_ap = extended_data.loc[forecast_index, "BS_Accounts_Payable"] if "BS_Accounts_Payable" in extended_data.columns else 0
            current_deferred_revenue = extended_data.loc[forecast_index, "BS_Deferred_Revenue"] if "BS_Deferred_Revenue" in extended_data.columns else 0
            
            change_in_ar = current_ar - previous_ar
            change_in_inventory = current_inventory - previous_inventory
            change_in_ap = current_ap - previous_ap
            change_in_deferred_revenue = current_deferred_revenue - previous_deferred_revenue
            
            # Negative changes in AR and Inventory are cash inflows
            # Positive changes in AP and Deferred Revenue are cash inflows
            change_in_wc = (change_in_ar + change_in_inventory) - (change_in_ap + change_in_deferred_revenue)
            
            extended_data.loc[forecast_index, "CF_Change_In_Accounts_Receivable"] = change_in_ar
            extended_data.loc[forecast_index, "CF_Change_In_Inventory"] = change_in_inventory
            extended_data.loc[forecast_index, "CF_Change_In_Accounts_Payable"] = change_in_ap
            extended_data.loc[forecast_index, "CF_Change_In_Deferred_Revenue"] = change_in_deferred_revenue
            extended_data.loc[forecast_index, "CF_Change_In_Working_Capital"] = change_in_wc
            
            # Net Cash from Operating Activities
            net_cash_from_operating = net_income + da - change_in_wc
            extended_data.loc[forecast_index, "CF_Net_Cash_From_Operating"] = net_cash_from_operating
            
            # Capital Expenditures
            capex = revenue * self.parameters["capex_to_revenue"]
            extended_data.loc[forecast_index, "CF_Capital_Expenditures"] = capex
            
            # Net Cash from Investing Activities
            net_cash_from_investing = -capex  # Simplified
            extended_data.loc[forecast_index, "CF_Net_Cash_From_Investing"] = net_cash_from_investing
            
            # Debt Issuance/Repayment
            if "BS_Long_Term_Debt" in extended_data.columns and "BS_Short_Term_Debt" in extended_data.columns:
                if i == 0:
                    previous_total_debt = last_historical_data["BS_Long_Term_Debt"] + last_historical_data["BS_Short_Term_Debt"]
                else:
                    previous_total_debt = extended_data.loc[forecast_index - 1, "BS_Long_Term_Debt"] + extended_data.loc[forecast_index - 1, "BS_Short_Term_Debt"]
                
                current_total_debt = lt_debt + st_debt
                debt_change = current_total_debt - previous_total_debt
                
                extended_data.loc[forecast_index, "CF_Debt_Issuance"] = max(0, debt_change)
                extended_data.loc[forecast_index, "CF_Debt_Repayment"] = max(0, -debt_change)
            
            # Dividends Paid
            dividends = net_income * self.parameters["dividend_payout_ratio"]
            extended_data.loc[forecast_index, "CF_Dividends_Paid"] = dividends
            
            # Net Cash from Financing Activities
            if "CF_Debt_Issuance" in extended_data.columns and "CF_Debt_Repayment" in extended_data.columns:
                debt_issuance = extended_data.loc[forecast_index, "CF_Debt_Issuance"]
                debt_repayment = extended_data.loc[forecast_index, "CF_Debt_Repayment"]
                net_cash_from_financing = debt_issuance - debt_repayment - dividends
                extended_data.loc[forecast_index, "CF_Net_Cash_From_Financing"] = net_cash_from_financing
            
            # Net Change in Cash
            net_change_in_cash = net_cash_from_operating + net_cash_from_investing
            if "CF_Net_Cash_From_Financing" in extended_data.columns:
                net_change_in_cash += extended_data.loc[forecast_index, "CF_Net_Cash_From_Financing"]
            
            extended_data.loc[forecast_index, "CF_Net_Change_In_Cash"] = net_change_in_cash
            
            # Beginning Cash Balance
            if i == 0:
                beginning_cash = last_historical_data["BS_Cash_Equivalents"] if "BS_Cash_Equivalents" in last_historical_data else 0
            else:
                beginning_cash = extended_data.loc[forecast_index - 1, "BS_Cash_Equivalents"] if "BS_Cash_Equivalents" in extended_data.columns else 0
            
            extended_data.loc[forecast_index, "CF_Beginning_Cash_Balance"] = beginning_cash
            
            # Ending Cash Balance
            ending_cash = beginning_cash + net_change_in_cash
            extended_data.loc[forecast_index, "CF_Ending_Cash_Balance"] = ending_cash
            
            # Free Cash Flow
            free_cash_flow = net_cash_from_operating - capex
            extended_data.loc[forecast_index, "CF_Free_Cash_Flow"] = free_cash_flow
        
        # Store the forecast data
        self.forecast_data = extended_data
        
        # Add audit step
        self._add_audit_step("Forecast Generation", {
            "scenario": scenario_name,
            "forecast_years": self.parameters["forecast_years"],
            "parameters": self.parameters
        })
        
        return extended_data
    
    def get_income_statement(self, scenario_name: str = "base") -> pd.DataFrame:
        """
        Get the income statement for a specific scenario.
        
        Args:
            scenario_name: Name of the scenario
            
        Returns:
            DataFrame containing the income statement
        """
        if self.forecast_data is None:
            self.generate_forecast(scenario_name)
        
        # Extract income statement columns
        income_cols = ["Year"] + [col for col in self.forecast_data.columns if col.startswith("IS_")]
        income_statement = self.forecast_data[income_cols].copy()
        
        # Rename columns to remove the "IS_" prefix
        income_statement.columns = [col.replace("IS_", "") if col.startswith("IS_") else col for col in income_statement.columns]
        
        return income_statement
    
    def get_balance_sheet(self, scenario_name: str = "base") -> pd.DataFrame:
        """
        Get the balance sheet for a specific scenario.
        
        Args:
            scenario_name: Name of the scenario
            
        Returns:
            DataFrame containing the balance sheet
        """
        if self.forecast_data is None:
            self.generate_forecast(scenario_name)
        
        # Extract balance sheet columns
        balance_cols = ["Year"] + [col for col in self.forecast_data.columns if col.startswith("BS_")]
        balance_sheet = self.forecast_data[balance_cols].copy()
        
        # Rename columns to remove the "BS_" prefix
        balance_sheet.columns = [col.replace("BS_", "") if col.startswith("BS_") else col for col in balance_sheet.columns]
        
        return balance_sheet
    
    def get_cash_flow(self, scenario_name: str = "base") -> pd.DataFrame:
        """
        Get the cash flow statement for a specific scenario.
        
        Args:
            scenario_name: Name of the scenario
            
        Returns:
            DataFrame containing the cash flow statement
        """
        if self.forecast_data is None:
            self.generate_forecast(scenario_name)
        
        # Extract cash flow columns
        cash_flow_cols = ["Year"] + [col for col in self.forecast_data.columns if col.startswith("CF_")]
        cash_flow = self.forecast_data[cash_flow_cols].copy()
        
        # Rename columns to remove the "CF_" prefix
        cash_flow.columns = [col.replace("CF_", "") if col.startswith("CF_") else col for col in cash_flow.columns]
        
        return cash_flow
    
    def get_financial_metrics(self, scenario_name: str = "base") -> pd.DataFrame:
        """
        Calculate and return key financial metrics for a specific scenario.
        
        Args:
            scenario_name: Name of the scenario
            
        Returns:
            DataFrame containing financial metrics
        """
        if self.forecast_data is None:
            self.generate_forecast(scenario_name)
        
        # Create a new DataFrame for metrics
        metrics_df = pd.DataFrame({"Year": self.forecast_data["Year"]})
        
        # Growth Metrics
        if "IS_Revenue" in self.forecast_data.columns:
            metrics_df["Revenue_Growth"] = self.forecast_data["IS_Revenue"].pct_change()
        
        if "IS_EBITDA" in self.forecast_data.columns:
            metrics_df["EBITDA_Growth"] = self.forecast_data["IS_EBITDA"].pct_change()
        
        if "IS_Net_Income" in self.forecast_data.columns:
            metrics_df["Net_Income_Growth"] = self.forecast_data["IS_Net_Income"].pct_change()
        
        # Margin Metrics
        if "IS_EBITDA" in self.forecast_data.columns and "IS_Revenue" in self.forecast_data.columns:
            metrics_df["EBITDA_Margin"] = self.forecast_data["IS_EBITDA"] / self.forecast_data["IS_Revenue"]
        
        if "IS_EBIT" in self.forecast_data.columns and "IS_Revenue" in self.forecast_data.columns:
            metrics_df["EBIT_Margin"] = self.forecast_data["IS_EBIT"] / self.forecast_data["IS_Revenue"]
        
        if "IS_Net_Income" in self.forecast_data.columns and "IS_Revenue" in self.forecast_data.columns:
            metrics_df["Net_Margin"] = self.forecast_data["IS_Net_Income"] / self.forecast_data["IS_Revenue"]
        
        # Leverage Metrics
        if "BS_Total_Liabilities" in self.forecast_data.columns and "BS_Total_Assets" in self.forecast_data.columns:
            metrics_df["Debt_to_Assets"] = self.forecast_data["BS_Total_Liabilities"] / self.forecast_data["BS_Total_Assets"]
        
        if "BS_Total_Liabilities" in self.forecast_data.columns and "BS_Total_Equity" in self.forecast_data.columns:
            metrics_df["Debt_to_Equity"] = self.forecast_data["BS_Total_Liabilities"] / self.forecast_data["BS_Total_Equity"]
        
        if "BS_Long_Term_Debt" in self.forecast_data.columns and "BS_Short_Term_Debt" in self.forecast_data.columns and "IS_EBITDA" in self.forecast_data.columns:
            metrics_df["Debt_to_EBITDA"] = (self.forecast_data["BS_Long_Term_Debt"] + self.forecast_data["BS_Short_Term_Debt"]) / self.forecast_data["IS_EBITDA"]
        
        # Liquidity Metrics
        if "BS_Total_Current_Assets" in self.forecast_data.columns and "BS_Total_Current_Liabilities" in self.forecast_data.columns:
            metrics_df["Current_Ratio"] = self.forecast_data["BS_Total_Current_Assets"] / self.forecast_data["BS_Total_Current_Liabilities"]
        
        if "BS_Cash_Equivalents" in self.forecast_data.columns and "BS_Short_Term_Investments" in self.forecast_data.columns and "BS_Total_Current_Liabilities" in self.forecast_data.columns:
            metrics_df["Quick_Ratio"] = (self.forecast_data["BS_Cash_Equivalents"] + self.forecast_data["BS_Short_Term_Investments"]) / self.forecast_data["BS_Total_Current_Liabilities"]
        
        # Efficiency Metrics
        if "IS_Revenue" in self.forecast_data.columns and "BS_Total_Assets" in self.forecast_data.columns:
            metrics_df["Asset_Turnover"] = self.forecast_data["IS_Revenue"] / self.forecast_data["BS_Total_Assets"]
        
        if "BS_Accounts_Receivable" in self.forecast_data.columns and "IS_Revenue" in self.forecast_data.columns:
            metrics_df["Days_Sales_Outstanding"] = (self.forecast_data["BS_Accounts_Receivable"] / self.forecast_data["IS_Revenue"]) * 365
        
        if "BS_Inventory" in self.forecast_data.columns and "IS_Cost_of_Revenue" in self.forecast_data.columns:
            metrics_df["Days_Inventory_Outstanding"] = (self.forecast_data["BS_Inventory"] / self.forecast_data["IS_Cost_of_Revenue"]) * 365
        
        if "BS_Accounts_Payable" in self.forecast_data.columns and "IS_Cost_of_Revenue" in self.forecast_data.columns:
            metrics_df["Days_Payable_Outstanding"] = (self.forecast_data["BS_Accounts_Payable"] / self.forecast_data["IS_Cost_of_Revenue"]) * 365
        
        # Cash Flow Metrics
        if "CF_Free_Cash_Flow" in self.forecast_data.columns and "IS_Revenue" in self.forecast_data.columns:
            metrics_df["FCF_to_Revenue"] = self.forecast_data["CF_Free_Cash_Flow"] / self.forecast_data["IS_Revenue"]
        
        if "CF_Free_Cash_Flow" in self.forecast_data.columns and "IS_Net_Income" in self.forecast_data.columns:
            metrics_df["FCF_to_Net_Income"] = self.forecast_data["CF_Free_Cash_Flow"] / self.forecast_data["IS_Net_Income"]
        
        # Software-specific metrics
        if "IS_Sales_Marketing" in self.forecast_data.columns and "IS_Revenue" in self.forecast_data.columns:
            metrics_df["Sales_Marketing_to_Revenue"] = self.forecast_data["IS_Sales_Marketing"] / self.forecast_data["IS_Revenue"]
        
        if "IS_Research_Development" in self.forecast_data.columns and "IS_Revenue" in self.forecast_data.columns:
            metrics_df["R_D_to_Revenue"] = self.forecast_data["IS_Research_Development"] / self.forecast_data["IS_Revenue"]
        
        # Rule of 40 (for SaaS companies)
        if "IS_Revenue_Growth" in self.forecast_data.columns and "IS_Net_Margin" in self.forecast_data.columns:
            metrics_df["Rule_of_40"] = self.forecast_data["IS_Revenue_Growth"] + self.forecast_data["IS_Net_Margin"]
        
        return metrics_df
    
    def get_audit_trail(self) -> List[Dict[str, Any]]:
        """
        Get the audit trail of the financial model.
        
        Returns:
            List of audit trail steps
        """
        return self.audit_trail
    
    def compare_scenarios(self, metric: str, scenarios: List[str] = None) -> pd.DataFrame:
        """
        Compare a specific metric across different scenarios.
        
        Args:
            metric: Name of the metric to compare
            scenarios: List of scenario names (if None, use all available scenarios)
            
        Returns:
            DataFrame containing the metric values for each scenario
        """
        if scenarios is None:
            scenarios = ["base", "upside", "downside"]
        
        # Create a DataFrame for comparison
        comparison_df = pd.DataFrame()
        
        for scenario in scenarios:
            # Generate forecast for the scenario if not already generated
            self.generate_forecast(scenario)
            
            # Get the metric values
            if metric.startswith("IS_"):
                statement = self.get_income_statement(scenario)
                metric_name = metric.replace("IS_", "")
                if metric_name in statement.columns:
                    comparison_df[f"{scenario}_{metric_name}"] = statement[metric_name]
            elif metric.startswith("BS_"):
                statement = self.get_balance_sheet(scenario)
                metric_name = metric.replace("BS_", "")
                if metric_name in statement.columns:
                    comparison_df[f"{scenario}_{metric_name}"] = statement[metric_name]
            elif metric.startswith("CF_"):
                statement = self.get_cash_flow(scenario)
                metric_name = metric.replace("CF_", "")
                if metric_name in statement.columns:
                    comparison_df[f"{scenario}_{metric_name}"] = statement[metric_name]
            else:
                # Try to find the metric in the financial metrics
                metrics = self.get_financial_metrics(scenario)
                if metric in metrics.columns:
                    comparison_df[f"{scenario}_{metric}"] = metrics[metric]
        
        # Add the Year column
        comparison_df["Year"] = self.forecast_data["Year"]
        
        # Reorder columns to put Year first
        cols = comparison_df.columns.tolist()
        cols = ["Year"] + [col for col in cols if col != "Year"]
        comparison_df = comparison_df[cols]
        
        return comparison_df
