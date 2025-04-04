"""
DCF Calculator module for the DCF application.
Contains functions for DCF calculations using different methods.
"""

import numpy as np
import pandas as pd

def calculate_free_cash_flow(revenue, ebitda_margin, tax_rate, capex_percent, 
                             depreciation_percent, nwc_percent):
    """
    Calculate free cash flow based on revenue and other financial metrics.
    
    Args:
        revenue (float): Revenue
        ebitda_margin (float): EBITDA margin as a decimal
        tax_rate (float): Tax rate as a decimal
        capex_percent (float): Capital expenditure as a percentage of revenue
        depreciation_percent (float): Depreciation as a percentage of revenue
        nwc_percent (float): Net working capital change as a percentage of revenue
        
    Returns:
        dict: Dictionary with calculated financial metrics
    """
    ebitda = revenue * ebitda_margin
    depreciation = revenue * depreciation_percent
    ebit = ebitda - depreciation
    taxes = ebit * tax_rate
    nopat = ebit - taxes
    capex = -revenue * capex_percent
    nwc_change = -revenue * nwc_percent
    
    fcf = nopat + depreciation + capex + nwc_change
    
    return {
        "Revenue": revenue,
        "EBITDA": ebitda,
        "EBITDA Margin": ebitda_margin,
        "Depreciation": depreciation,
        "EBIT": ebit,
        "Taxes": taxes,
        "NOPAT": nopat,
        "Capital Expenditures": capex,
        "Change in NWC": nwc_change,
        "Free Cash Flow": fcf
    }

def project_financials(base_revenue, forecast_years, growth_rates, ebitda_margins, 
                       tax_rates, capex_percents, depreciation_percents, nwc_percents):
    """
    Project financial metrics for a given forecast period.
    
    Args:
        base_revenue (float): Base revenue to start projections
        forecast_years (int): Number of years to forecast
        growth_rates (list or float): Revenue growth rates for each year
        ebitda_margins (list or float): EBITDA margins for each year
        tax_rates (list or float): Tax rates for each year
        capex_percents (list or float): Capital expenditure percentages for each year
        depreciation_percents (list or float): Depreciation percentages for each year
        nwc_percents (list or float): Net working capital change percentages for each year
        
    Returns:
        pandas.DataFrame: Projected financial metrics
    """
    # Convert single values to lists if needed
    if not isinstance(growth_rates, list):
        growth_rates = [growth_rates] * forecast_years
    if not isinstance(ebitda_margins, list):
        ebitda_margins = [ebitda_margins] * forecast_years
    if not isinstance(tax_rates, list):
        tax_rates = [tax_rates] * forecast_years
    if not isinstance(capex_percents, list):
        capex_percents = [capex_percents] * forecast_years
    if not isinstance(depreciation_percents, list):
        depreciation_percents = [depreciation_percents] * forecast_years
    if not isinstance(nwc_percents, list):
        nwc_percents = [nwc_percents] * forecast_years
    
    # Ensure all lists have the correct length
    growth_rates = growth_rates[:forecast_years]
    ebitda_margins = ebitda_margins[:forecast_years]
    tax_rates = tax_rates[:forecast_years]
    capex_percents = capex_percents[:forecast_years]
    depreciation_percents = depreciation_percents[:forecast_years]
    nwc_percents = nwc_percents[:forecast_years]
    
    # Pad lists if they're too short
    while len(growth_rates) < forecast_years:
        growth_rates.append(growth_rates[-1])
    while len(ebitda_margins) < forecast_years:
        ebitda_margins.append(ebitda_margins[-1])
    while len(tax_rates) < forecast_years:
        tax_rates.append(tax_rates[-1])
    while len(capex_percents) < forecast_years:
        capex_percents.append(capex_percents[-1])
    while len(depreciation_percents) < forecast_years:
        depreciation_percents.append(depreciation_percents[-1])
    while len(nwc_percents) < forecast_years:
        nwc_percents.append(nwc_percents[-1])
    
    # Project revenue
    revenues = [base_revenue]
    for i in range(forecast_years):
        revenues.append(revenues[-1] * (1 + growth_rates[i]))
    
    # Remove the base revenue
    revenues = revenues[1:]
    
    # Calculate financial metrics for each year
    projections = []
    for i in range(forecast_years):
        projection = calculate_free_cash_flow(
            revenues[i], 
            ebitda_margins[i], 
            tax_rates[i], 
            capex_percents[i], 
            depreciation_percents[i], 
            nwc_percents[i]
        )
        projections.append(projection)
    
    # Create DataFrame
    df = pd.DataFrame(projections)
    
    return df

def calculate_terminal_value_perpetuity(final_fcf, wacc, terminal_growth):
    """
    Calculate terminal value using the perpetuity growth method.
    
    Args:
        final_fcf (float): Final year free cash flow
        wacc (float): Weighted average cost of capital
        terminal_growth (float): Terminal growth rate
        
    Returns:
        float: Terminal value
    """
    return final_fcf * (1 + terminal_growth) / (wacc - terminal_growth)

def calculate_terminal_value_multiple(final_ebitda, ebitda_multiple):
    """
    Calculate terminal value using the exit multiple method.
    
    Args:
        final_ebitda (float): Final year EBITDA
        ebitda_multiple (float): EBITDA multiple
        
    Returns:
        float: Terminal value
    """
    return final_ebitda * ebitda_multiple

def calculate_present_value(cash_flows, wacc):
    """
    Calculate the present value of cash flows.
    
    Args:
        cash_flows (list): List of cash flows
        wacc (float): Weighted average cost of capital
        
    Returns:
        list: Present values of cash flows
    """
    present_values = []
    for i, cf in enumerate(cash_flows):
        present_values.append(cf / ((1 + wacc) ** (i + 1)))
    
    return present_values

def calculate_dcf_valuation(projections, terminal_value, wacc, debt, cash):
    """
    Calculate DCF valuation.
    
    Args:
        projections (pandas.DataFrame): Projected financial metrics
        terminal_value (float): Terminal value
        wacc (float): Weighted average cost of capital
        debt (float): Total debt
        cash (float): Cash and cash equivalents
        
    Returns:
        dict: DCF valuation results
    """
    # Extract free cash flows
    fcfs = projections["Free Cash Flow"].tolist()
    
    # Calculate present value of free cash flows
    pv_fcfs = calculate_present_value(fcfs, wacc)
    
    # Calculate present value of terminal value
    pv_terminal_value = terminal_value / ((1 + wacc) ** len(fcfs))
    
    # Calculate enterprise value
    enterprise_value = sum(pv_fcfs) + pv_terminal_value
    
    # Calculate equity value
    equity_value = enterprise_value - debt + cash
    
    return {
        "Free Cash Flows": fcfs,
        "PV of FCFs": pv_fcfs,
        "Sum of PV of FCFs": sum(pv_fcfs),
        "Terminal Value": terminal_value,
        "PV of Terminal Value": pv_terminal_value,
        "Enterprise Value": enterprise_value,
        "Debt": debt,
        "Cash": cash,
        "Equity Value": equity_value
    }

def calculate_implied_share_price(equity_value, shares_outstanding):
    """
    Calculate implied share price.
    
    Args:
        equity_value (float): Equity value
        shares_outstanding (float): Number of shares outstanding
        
    Returns:
        float: Implied share price
    """
    return equity_value / shares_outstanding

def perform_sensitivity_analysis(base_params, sensitivity_params, shares_outstanding):
    """
    Perform sensitivity analysis on DCF valuation.
    
    Args:
        base_params (dict): Base parameters for DCF valuation
        sensitivity_params (dict): Parameters to vary for sensitivity analysis
        shares_outstanding (float): Number of shares outstanding
        
    Returns:
        dict: Sensitivity analysis results
    """
    results = {}
    
    for param_name, param_values in sensitivity_params.items():
        param_results = []
        
        for value in param_values:
            # Create a copy of base parameters
            params = base_params.copy()
            
            # Update the parameter value
            params[param_name] = value
            
            # Project financials
            projections = project_financials(
                params["base_revenue"],
                params["forecast_years"],
                params["growth_rates"],
                params["ebitda_margins"],
                params["tax_rates"],
                params["capex_percents"],
                params["depreciation_percents"],
                params["nwc_percents"]
            )
            
            # Calculate terminal value
            if params["terminal_value_method"] == "perpetuity":
                terminal_value = calculate_terminal_value_perpetuity(
                    projections["Free Cash Flow"].iloc[-1],
                    params["wacc"],
                    params["terminal_growth"]
                )
            else:  # multiple
                terminal_value = calculate_terminal_value_multiple(
                    projections["EBITDA"].iloc[-1],
                    params["ebitda_multiple"]
                )
            
            # Calculate DCF valuation
            valuation = calculate_dcf_valuation(
                projections,
                terminal_value,
                params["wacc"],
                params["debt"],
                params["cash"]
            )
            
            # Calculate implied share price
            share_price = calculate_implied_share_price(
                valuation["Equity Value"],
                shares_outstanding
            )
            
            param_results.append({
                "Value": value,
                "Enterprise Value": valuation["Enterprise Value"],
                "Equity Value": valuation["Equity Value"],
                "Share Price": share_price
            })
        
        results[param_name] = param_results
    
    return results

def create_two_factor_sensitivity_table(base_params, factor1, factor2, shares_outstanding):
    """
    Create a two-factor sensitivity table.
    
    Args:
        base_params (dict): Base parameters for DCF valuation
        factor1 (dict): First factor with name and values
        factor2 (dict): Second factor with name and values
        shares_outstanding (float): Number of shares outstanding
        
    Returns:
        pandas.DataFrame: Sensitivity table
    """
    # Create empty DataFrame
    table = pd.DataFrame(index=factor1["values"], columns=factor2["values"])
    
    # Fill the table
    for val1 in factor1["values"]:
        for val2 in factor2["values"]:
            # Create a copy of base parameters
            params = base_params.copy()
            
            # Update parameter values
            params[factor1["name"]] = val1
            params[factor2["name"]] = val2
            
            # Project financials
            projections = project_financials(
                params["base_revenue"],
                params["forecast_years"],
                params["growth_rates"],
                params["ebitda_margins"],
                params["tax_rates"],
                params["capex_percents"],
                params["depreciation_percents"],
                params["nwc_percents"]
            )
            
            # Calculate terminal value
            if params["terminal_value_method"] == "perpetuity":
                terminal_value = calculate_terminal_value_perpetuity(
                    projections["Free Cash Flow"].iloc[-1],
                    params["wacc"],
                    params["terminal_growth"]
                )
            else:  # multiple
                terminal_value = calculate_terminal_value_multiple(
                    projections["EBITDA"].iloc[-1],
                    params["ebitda_multiple"]
                )
            
            # Calculate DCF valuation
            valuation = calculate_dcf_valuation(
                projections,
                terminal_value,
                params["wacc"],
                params["debt"],
                params["cash"]
            )
            
            # Calculate implied share price
            share_price = calculate_implied_share_price(
                valuation["Equity Value"],
                shares_outstanding
            )
            
            # Add to table
            table.loc[val1, val2] = share_price
    
    return table
