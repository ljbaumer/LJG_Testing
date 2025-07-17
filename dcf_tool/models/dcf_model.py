import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
import os
import json

from dcf_tool.models.data_manager import DataManager
from dcf_tool.models.growth_methods import get_growth_method, GrowthMethod
from dcf_tool.utils.financial_utils import (
    calculate_present_value,
    calculate_wacc,
    calculate_cost_of_equity
)


class DCFModel:
    """
    Discounted Cash Flow model implementation.
    """
    
    def __init__(self, data_dir: str = "../data"):
        """
        Initialize the DCF model.
        
        Args:
            data_dir: Directory path where data files are stored
        """
        self.data_manager = DataManager(data_dir)
        self.data = None
        self.forecast_data = None
        self.forecast_years = 5
        self.terminal_value = 0
        self.present_values = []
        self.enterprise_value = 0
        self.equity_value = 0
        self.audit_trail = []
        self.parameters = {
            # Default parameters
            "forecast_years": 5,
            "revenue_growth": 0.05,
            "ebitda_margin": 0.20,
            "tax_rate": 0.25,
            "da_to_revenue": 0.05,
            "capex_to_revenue": 0.06,
            "wc_to_revenue": 0.01,
            "terminal_growth_rate": 0.02,
            "wacc": 0.10,
            "terminal_method": "perpetuity_growth",
            "exit_multiple": 8.0,
            "debt_value": 500,
            "cash_equivalents": 200,
            "shares_outstanding": 100,
            "risk_free_rate": 0.03,
            "market_risk_premium": 0.05,
            "beta": 1.2,
            "calculate_wacc": True
        }
        self.calculation_steps = []
        
    def load_data(self, use_user_data: bool = False) -> pd.DataFrame:
        """
        Load financial data.
        
        Args:
            use_user_data: If True, load from user_data.csv if it exists
            
        Returns:
            DataFrame containing the financial data
        """
        self.data = self.data_manager.load_data(use_user_data)
        return self.data
    
    def save_data(self, data: pd.DataFrame, as_user_data: bool = True) -> str:
        """
        Save financial data.
        
        Args:
            data: DataFrame containing the financial data to save
            as_user_data: If True, save to user_data.csv
            
        Returns:
            Path to the saved file
        """
        return self.data_manager.save_data(data, as_user_data)
    
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
        
        # Update forecast years if provided
        if "forecast_years" in parameters:
            self.forecast_years = parameters["forecast_years"]
    
    def get_parameters(self) -> Dict[str, Any]:
        """
        Get current model parameters.
        
        Returns:
            Dictionary of parameter names and values
        """
        return self.parameters
    
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
    
    def _add_calculation_step(self, step: str, formula: str, values: Dict[str, Any], result: Any) -> None:
        """
        Add a calculation step for auditability.
        
        Args:
            step: Name of the calculation step
            formula: Formula used in the calculation
            values: Dictionary of values used in the calculation
            result: Result of the calculation
        """
        self.calculation_steps.append({
            "step": step,
            "formula": formula,
            "values": values,
            "result": result
        })
    
    def prepare_forecast(self) -> pd.DataFrame:
        """
        Prepare the forecast data based on historical data and parameters.
        
        Returns:
            DataFrame containing historical and forecast data
        """
        if self.data is None:
            self.load_data()
        
        # Get historical metrics
        historical_metrics = self.data_manager.calculate_historical_metrics(self.data)
        
        # Use historical metrics as defaults if not specified in parameters
        for metric, value in historical_metrics.items():
            if metric == "avg_revenue_growth" and "revenue_growth" not in self.parameters:
                self.parameters["revenue_growth"] = value
            elif metric == "avg_ebitda_margin" and "ebitda_margin" not in self.parameters:
                self.parameters["ebitda_margin"] = value
            elif metric == "avg_da_to_revenue" and "da_to_revenue" not in self.parameters:
                self.parameters["da_to_revenue"] = value
            elif metric == "avg_capex_to_revenue" and "capex_to_revenue" not in self.parameters:
                self.parameters["capex_to_revenue"] = value
            elif metric == "avg_wcchange_to_revenue" and "wc_to_revenue" not in self.parameters:
                self.parameters["wc_to_revenue"] = value
            elif metric == "avg_tax_rate" and "tax_rate" not in self.parameters:
                self.parameters["tax_rate"] = value
        
        # Extend data with forecast years
        extended_data = self.data_manager.extend_forecast_years(self.data, self.forecast_years)
        
        # Get the last historical year
        last_historical_year = self.data['Year'].max()
        
        # Get the last historical year's data
        last_historical_data = self.data[self.data['Year'] == last_historical_year].iloc[0]
        
        # Forecast data for future years
        for i in range(self.forecast_years):
            forecast_year = last_historical_year + i + 1
            forecast_index = extended_data[extended_data['Year'] == forecast_year].index[0]
            
            # Revenue forecast with growth rate
            if i == 0:
                previous_revenue = last_historical_data['Revenue']
            else:
                previous_revenue = extended_data.loc[forecast_index - 1, 'Revenue']
            
            revenue = previous_revenue * (1 + self.parameters["revenue_growth"])
            extended_data.loc[forecast_index, 'Revenue'] = revenue
            extended_data.loc[forecast_index, 'Revenue_Growth'] = self.parameters["revenue_growth"]
            
            # EBITDA forecast based on margin
            ebitda = revenue * self.parameters["ebitda_margin"]
            extended_data.loc[forecast_index, 'EBITDA'] = ebitda
            extended_data.loc[forecast_index, 'EBITDA_Margin'] = self.parameters["ebitda_margin"]
            
            # Depreciation & Amortization forecast
            da = revenue * self.parameters["da_to_revenue"]
            extended_data.loc[forecast_index, 'Depreciation_Amortization'] = da
            
            # EBIT forecast
            ebit = ebitda - da
            extended_data.loc[forecast_index, 'EBIT'] = ebit
            
            # Tax rate
            extended_data.loc[forecast_index, 'Tax_Rate'] = self.parameters["tax_rate"]
            
            # NOPAT forecast
            nopat = ebit * (1 - self.parameters["tax_rate"])
            extended_data.loc[forecast_index, 'NOPAT'] = nopat
            
            # Capital Expenditures forecast
            capex = revenue * self.parameters["capex_to_revenue"]
            extended_data.loc[forecast_index, 'Capital_Expenditures'] = capex
            
            # Change in Working Capital forecast
            wc_change = revenue * self.parameters["wc_to_revenue"]
            extended_data.loc[forecast_index, 'Change_in_Working_Capital'] = wc_change
            
            # Free Cash Flow forecast
            fcf = nopat + da - capex - wc_change
            extended_data.loc[forecast_index, 'Free_Cash_Flow'] = fcf
        
        self.forecast_data = extended_data
        
        # Add audit step
        self._add_audit_step("Forecast Preparation", {
            "forecast_years": self.forecast_years,
            "parameters": {
                "revenue_growth": self.parameters["revenue_growth"],
                "ebitda_margin": self.parameters["ebitda_margin"],
                "tax_rate": self.parameters["tax_rate"],
                "da_to_revenue": self.parameters["da_to_revenue"],
                "capex_to_revenue": self.parameters["capex_to_revenue"],
                "wc_to_revenue": self.parameters["wc_to_revenue"]
            }
        })
        
        return extended_data
    
    def calculate_wacc(self) -> float:
        """
        Calculate the Weighted Average Cost of Capital (WACC).
        
        Returns:
            WACC as a decimal
        """
        if not self.parameters["calculate_wacc"]:
            # Use the provided WACC directly
            wacc = self.parameters["wacc"]
            
            self._add_calculation_step(
                "WACC Calculation",
                "Using provided WACC directly",
                {"wacc": wacc},
                wacc
            )
            
            return wacc
        
        # Calculate cost of equity using CAPM
        cost_of_equity = calculate_cost_of_equity(
            risk_free_rate=self.parameters["risk_free_rate"],
            market_risk_premium=self.parameters["market_risk_premium"],
            beta=self.parameters["beta"]
        )
        
        self._add_calculation_step(
            "Cost of Equity Calculation",
            "Re = Rf + β(Rm - Rf)",
            {
                "risk_free_rate": self.parameters["risk_free_rate"],
                "market_risk_premium": self.parameters["market_risk_premium"],
                "beta": self.parameters["beta"]
            },
            cost_of_equity
        )
        
        # Assume cost of debt is risk-free rate plus a spread
        cost_of_debt = self.parameters.get("cost_of_debt", self.parameters["risk_free_rate"] + 0.02)
        
        # Calculate WACC
        equity_value = self.parameters.get("equity_value", 1000)  # Default if not available yet
        debt_value = self.parameters["debt_value"]
        tax_rate = self.parameters["tax_rate"]
        
        wacc = calculate_wacc(
            equity_value=equity_value,
            debt_value=debt_value,
            cost_of_equity=cost_of_equity,
            cost_of_debt=cost_of_debt,
            tax_rate=tax_rate
        )
        
        self._add_calculation_step(
            "WACC Calculation",
            "WACC = (E/V * Re) + (D/V * Rd * (1 - T))",
            {
                "equity_value": equity_value,
                "debt_value": debt_value,
                "cost_of_equity": cost_of_equity,
                "cost_of_debt": cost_of_debt,
                "tax_rate": tax_rate
            },
            wacc
        )
        
        # Update the WACC parameter
        self.parameters["wacc"] = wacc
        
        return wacc
    
    def calculate_terminal_value(self) -> float:
        """
        Calculate the terminal value using the selected method.
        
        Returns:
            Terminal value
        """
        if self.forecast_data is None:
            self.prepare_forecast()
        
        # Get the terminal year data
        forecast_years = self.data_manager.get_forecast_years(self.forecast_data)
        
        # If forecast_years is empty, use the last rows of forecast_data
        if not forecast_years:
            # Get the last historical year
            last_historical_year = self.data['Year'].max()
            # Get all years greater than the last historical year
            forecast_years = self.forecast_data[self.forecast_data['Year'] > last_historical_year]['Year'].tolist()
        
        terminal_year = max(forecast_years)
        terminal_data = self.forecast_data[self.forecast_data['Year'] == terminal_year].iloc[0]
        
        # Get the growth method
        method_name = self.parameters["terminal_method"]
        growth_method = get_growth_method(method_name)
        
        if growth_method is None:
            raise ValueError(f"Unknown terminal value method: {method_name}")
        
        # Calculate terminal value based on the selected method
        if method_name == "perpetuity_growth":
            terminal_value = growth_method.calculate_terminal_value(
                final_cash_flow=terminal_data['Free_Cash_Flow'],
                growth_rate=self.parameters["terminal_growth_rate"],
                discount_rate=self.parameters["wacc"]
            )
            
            self._add_calculation_step(
                "Terminal Value Calculation (Perpetuity Growth)",
                "TV = FCF_t * (1 + g) / (WACC - g)",
                {
                    "final_cash_flow": terminal_data['Free_Cash_Flow'],
                    "growth_rate": self.parameters["terminal_growth_rate"],
                    "discount_rate": self.parameters["wacc"]
                },
                terminal_value
            )
        elif method_name == "exit_multiple":
            # Determine which metric to use based on the multiple type
            multiple_type = self.parameters.get("multiple_type", "EV/EBITDA")
            
            if multiple_type == "EV/EBITDA":
                final_metric = terminal_data['EBITDA']
            elif multiple_type == "EV/EBIT":
                final_metric = terminal_data['EBIT']
            else:
                raise ValueError(f"Unsupported multiple type: {multiple_type}")
            
            terminal_value = growth_method.calculate_terminal_value(
                final_metric=final_metric,
                multiple=self.parameters["exit_multiple"]
            )
            
            self._add_calculation_step(
                f"Terminal Value Calculation (Exit Multiple - {multiple_type})",
                f"TV = {multiple_type.split('/')[1]} * Multiple",
                {
                    "final_metric": final_metric,
                    "multiple": self.parameters["exit_multiple"]
                },
                terminal_value
            )
        else:
            raise ValueError(f"Unsupported terminal value method: {method_name}")
        
        self.terminal_value = terminal_value
        
        # Add audit step
        self._add_audit_step("Terminal Value Calculation", {
            "method": method_name,
            "parameters": {
                "terminal_growth_rate": self.parameters.get("terminal_growth_rate"),
                "exit_multiple": self.parameters.get("exit_multiple"),
                "multiple_type": self.parameters.get("multiple_type")
            },
            "terminal_value": terminal_value
        })
        
        return terminal_value
    
    def calculate_present_values(self) -> List[float]:
        """
        Calculate the present value of forecast cash flows and terminal value.
        
        Returns:
            List of present values
        """
        if self.forecast_data is None:
            self.prepare_forecast()
        
        if self.terminal_value == 0:
            self.calculate_terminal_value()
        
        # Get forecast years
        forecast_years = self.data_manager.get_forecast_years(self.forecast_data)
        
        # If forecast_years is empty, use the last rows of forecast_data
        if not forecast_years:
            # Get the last historical year
            last_historical_year = self.data['Year'].max()
            # Get all years greater than the last historical year
            forecast_years = self.forecast_data[self.forecast_data['Year'] > last_historical_year]['Year'].tolist()
        
        # Calculate present value for each forecast year's cash flow
        present_values = []
        discount_rate = self.parameters["wacc"]
        
        for i, year in enumerate(forecast_years):
            year_data = self.forecast_data[self.forecast_data['Year'] == year].iloc[0]
            cash_flow = year_data['Free_Cash_Flow']
            
            # Calculate present value
            pv = calculate_present_value(
                future_value=cash_flow,
                discount_rate=discount_rate,
                periods=i + 1
            )
            
            present_values.append(pv)
            
            self._add_calculation_step(
                f"Present Value Calculation (Year {year})",
                "PV = CF / (1 + WACC)^n",
                {
                    "cash_flow": cash_flow,
                    "discount_rate": discount_rate,
                    "periods": i + 1
                },
                pv
            )
        
        # Calculate present value of terminal value
        terminal_pv = calculate_present_value(
            future_value=self.terminal_value,
            discount_rate=discount_rate,
            periods=len(forecast_years)
        )
        
        self._add_calculation_step(
            "Present Value of Terminal Value",
            "PV_TV = TV / (1 + WACC)^n",
            {
                "terminal_value": self.terminal_value,
                "discount_rate": discount_rate,
                "periods": len(forecast_years)
            },
            terminal_pv
        )
        
        # Add terminal value PV to the list
        present_values.append(terminal_pv)
        
        self.present_values = present_values
        
        # Add audit step
        self._add_audit_step("Present Value Calculation", {
            "discount_rate": discount_rate,
            "forecast_present_values": present_values[:-1],
            "terminal_present_value": present_values[-1]
        })
        
        return present_values
    
    def calculate_enterprise_value(self) -> float:
        """
        Calculate the enterprise value.
        
        Returns:
            Enterprise value
        """
        if not self.present_values:
            self.calculate_present_values()
        
        # Sum all present values
        enterprise_value = sum(self.present_values)
        
        self._add_calculation_step(
            "Enterprise Value Calculation",
            "EV = Sum of Present Values",
            {
                "present_values": self.present_values
            },
            enterprise_value
        )
        
        self.enterprise_value = enterprise_value
        
        # Add audit step
        self._add_audit_step("Enterprise Value Calculation", {
            "enterprise_value": enterprise_value
        })
        
        return enterprise_value
    
    def calculate_equity_value(self) -> float:
        """
        Calculate the equity value.
        
        Returns:
            Equity value
        """
        if self.enterprise_value == 0:
            self.calculate_enterprise_value()
        
        # Equity Value = Enterprise Value - Debt + Cash
        debt_value = self.parameters["debt_value"]
        cash_equivalents = self.parameters["cash_equivalents"]
        
        equity_value = self.enterprise_value - debt_value + cash_equivalents
        
        self._add_calculation_step(
            "Equity Value Calculation",
            "Equity Value = Enterprise Value - Debt + Cash",
            {
                "enterprise_value": self.enterprise_value,
                "debt_value": debt_value,
                "cash_equivalents": cash_equivalents
            },
            equity_value
        )
        
        self.equity_value = equity_value
        
        # Add audit step
        self._add_audit_step("Equity Value Calculation", {
            "debt_value": debt_value,
            "cash_equivalents": cash_equivalents,
            "equity_value": equity_value
        })
        
        return equity_value
    
    def calculate_share_price(self) -> float:
        """
        Calculate the share price.
        
        Returns:
            Share price
        """
        if self.equity_value == 0:
            self.calculate_equity_value()
        
        # Share Price = Equity Value / Shares Outstanding
        shares_outstanding = self.parameters["shares_outstanding"]
        
        if shares_outstanding <= 0:
            raise ValueError("Shares outstanding must be greater than zero")
        
        share_price = self.equity_value / shares_outstanding
        
        self._add_calculation_step(
            "Share Price Calculation",
            "Share Price = Equity Value / Shares Outstanding",
            {
                "equity_value": self.equity_value,
                "shares_outstanding": shares_outstanding
            },
            share_price
        )
        
        # Add audit step
        self._add_audit_step("Share Price Calculation", {
            "shares_outstanding": shares_outstanding,
            "share_price": share_price
        })
        
        return share_price
    
    def run_dcf_valuation(self) -> Dict[str, Any]:
        """
        Run the complete DCF valuation process.
        
        Returns:
            Dictionary containing the valuation results
        """
        # Reset audit trail and calculation steps
        self.audit_trail = []
        self.calculation_steps = []
        
        # Load data if not already loaded
        if self.data is None:
            self.load_data()
        
        # Calculate WACC
        wacc = self.calculate_wacc()
        
        # Prepare forecast
        self.prepare_forecast()
        
        # Calculate terminal value
        terminal_value = self.calculate_terminal_value()
        
        # Calculate present values
        present_values = self.calculate_present_values()
        
        # Calculate enterprise value
        enterprise_value = self.calculate_enterprise_value()
        
        # Calculate equity value
        equity_value = self.calculate_equity_value()
        
        # Calculate share price
        share_price = self.calculate_share_price()
        
        # Prepare results
        results = {
            "wacc": wacc,
            "terminal_value": terminal_value,
            "present_values": present_values,
            "enterprise_value": enterprise_value,
            "equity_value": equity_value,
            "share_price": share_price,
            "forecast_data": self.forecast_data.to_dict(orient="records"),
            "audit_trail": self.audit_trail,
            "calculation_steps": self.calculation_steps
        }
        
        return results
    
    def get_audit_trail(self) -> List[Dict[str, Any]]:
        """
        Get the audit trail of the DCF valuation process.
        
        Returns:
            List of audit trail steps
        """
        return self.audit_trail
    
    def get_calculation_steps(self) -> List[Dict[str, Any]]:
        """
        Get the detailed calculation steps of the DCF valuation process.
        
        Returns:
            List of calculation steps
        """
        return self.calculation_steps
    
    def save_results(self, file_path: str) -> None:
        """
        Save the DCF valuation results to a JSON file.
        
        Args:
            file_path: Path to save the results
        """
        if not self.audit_trail:
            raise ValueError("No valuation results to save. Run run_dcf_valuation() first.")
        
        # Convert DataFrame to list of records
        forecast_data = self.forecast_data.to_dict(orient="records") if self.forecast_data is not None else None
        
        # Prepare results
        results = {
            "parameters": self.parameters,
            "wacc": self.parameters["wacc"],
            "terminal_value": self.terminal_value,
            "present_values": self.present_values,
            "enterprise_value": self.enterprise_value,
            "equity_value": self.equity_value,
            "forecast_data": forecast_data,
            "audit_trail": self.audit_trail,
            "calculation_steps": self.calculation_steps
        }
        
        # Save to file
        with open(file_path, 'w') as f:
            json.dump(results, f, indent=2)
