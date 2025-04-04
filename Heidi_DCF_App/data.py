"""
Data module for the DCF application.
Contains placeholder financial data for sample companies.
"""

import pandas as pd
import numpy as np

# Sample companies with historical financial data
SAMPLE_COMPANIES = {
    "TechCorp": {
        "name": "TechCorp Inc.",
        "sector": "Technology",
        "description": "A leading technology company specializing in software and cloud services.",
        "current_price": 150.75,
        "shares_outstanding": 1000000000,
        "debt": 5000000000,
        "cash": 3000000000,
        "historical_financials": {
            "revenue": [10000000000, 12000000000, 15000000000, 18000000000, 22000000000],
            "ebitda": [2000000000, 2500000000, 3200000000, 4000000000, 5000000000],
            "ebit": [1500000000, 2000000000, 2700000000, 3500000000, 4500000000],
            "net_income": [1000000000, 1300000000, 1800000000, 2300000000, 3000000000],
            "capex": [-800000000, -900000000, -1100000000, -1300000000, -1500000000],
            "depreciation": [500000000, 500000000, 500000000, 500000000, 500000000],
            "nwc_change": [-200000000, -250000000, -300000000, -350000000, -400000000],
            "years": [2020, 2021, 2022, 2023, 2024]
        }
    },
    "ConsumerGoods": {
        "name": "Consumer Goods Co.",
        "sector": "Consumer Staples",
        "description": "A global consumer goods company with a diverse product portfolio.",
        "current_price": 75.25,
        "shares_outstanding": 2000000000,
        "debt": 10000000000,
        "cash": 5000000000,
        "historical_financials": {
            "revenue": [25000000000, 26500000000, 28000000000, 29500000000, 31000000000],
            "ebitda": [5000000000, 5300000000, 5600000000, 5900000000, 6200000000],
            "ebit": [4000000000, 4300000000, 4600000000, 4900000000, 5200000000],
            "net_income": [2800000000, 3000000000, 3200000000, 3400000000, 3600000000],
            "capex": [-1200000000, -1300000000, -1400000000, -1500000000, -1600000000],
            "depreciation": [1000000000, 1000000000, 1000000000, 1000000000, 1000000000],
            "nwc_change": [-500000000, -550000000, -600000000, -650000000, -700000000],
            "years": [2020, 2021, 2022, 2023, 2024]
        }
    },
    "HealthInnovate": {
        "name": "Health Innovate Ltd.",
        "sector": "Healthcare",
        "description": "An innovative healthcare company focused on medical devices and pharmaceuticals.",
        "current_price": 200.50,
        "shares_outstanding": 500000000,
        "debt": 3000000000,
        "cash": 2000000000,
        "historical_financials": {
            "revenue": [8000000000, 9500000000, 11000000000, 13000000000, 15000000000],
            "ebitda": [2400000000, 2900000000, 3400000000, 4000000000, 4700000000],
            "ebit": [2000000000, 2500000000, 3000000000, 3600000000, 4300000000],
            "net_income": [1500000000, 1900000000, 2300000000, 2800000000, 3300000000],
            "capex": [-1000000000, -1200000000, -1400000000, -1600000000, -1800000000],
            "depreciation": [400000000, 400000000, 400000000, 400000000, 400000000],
            "nwc_change": [-300000000, -350000000, -400000000, -450000000, -500000000],
            "years": [2020, 2021, 2022, 2023, 2024]
        }
    }
}

# Industry averages for key metrics
INDUSTRY_AVERAGES = {
    "Technology": {
        "revenue_growth": 0.15,
        "ebitda_margin": 0.25,
        "tax_rate": 0.21,
        "wacc": 0.09,
        "terminal_growth": 0.03,
        "ebitda_multiple": 15,
        "capex_percent": 0.07,
        "nwc_percent": 0.02
    },
    "Consumer Staples": {
        "revenue_growth": 0.05,
        "ebitda_margin": 0.20,
        "tax_rate": 0.25,
        "wacc": 0.07,
        "terminal_growth": 0.02,
        "ebitda_multiple": 12,
        "capex_percent": 0.05,
        "nwc_percent": 0.03
    },
    "Healthcare": {
        "revenue_growth": 0.10,
        "ebitda_margin": 0.30,
        "tax_rate": 0.23,
        "wacc": 0.08,
        "terminal_growth": 0.025,
        "ebitda_multiple": 14,
        "capex_percent": 0.08,
        "nwc_percent": 0.025
    }
}

def get_company_data(company_key):
    """
    Retrieve data for a specific company.
    
    Args:
        company_key (str): Key for the company in the SAMPLE_COMPANIES dictionary
        
    Returns:
        dict: Company data
    """
    return SAMPLE_COMPANIES.get(company_key, None)

def get_all_company_keys():
    """
    Get a list of all available company keys.
    
    Returns:
        list: List of company keys
    """
    return list(SAMPLE_COMPANIES.keys())

def get_industry_averages(industry):
    """
    Get industry average metrics.
    
    Args:
        industry (str): Industry name
        
    Returns:
        dict: Industry average metrics
    """
    return INDUSTRY_AVERAGES.get(industry, None)

def get_historical_financials_as_df(company_key):
    """
    Get historical financials for a company as a pandas DataFrame.
    
    Args:
        company_key (str): Key for the company
        
    Returns:
        pandas.DataFrame: Historical financials
    """
    if company_key not in SAMPLE_COMPANIES:
        return None
    
    company = SAMPLE_COMPANIES[company_key]
    financials = company["historical_financials"]
    
    # Create a DataFrame with years as index
    df = pd.DataFrame({
        "Revenue": financials["revenue"],
        "EBITDA": financials["ebitda"],
        "EBIT": financials["ebit"],
        "Net Income": financials["net_income"],
        "Capital Expenditures": financials["capex"],
        "Depreciation": financials["depreciation"],
        "Change in NWC": financials["nwc_change"]
    }, index=financials["years"])
    
    return df

def calculate_historical_metrics(company_key):
    """
    Calculate historical financial metrics for a company.
    
    Args:
        company_key (str): Key for the company
        
    Returns:
        pandas.DataFrame: Historical metrics
    """
    if company_key not in SAMPLE_COMPANIES:
        return None
    
    df = get_historical_financials_as_df(company_key)
    
    # Calculate metrics
    metrics = pd.DataFrame(index=df.index)
    metrics["Revenue Growth"] = df["Revenue"].pct_change().fillna(0)
    metrics["EBITDA Margin"] = df["EBITDA"] / df["Revenue"]
    metrics["EBIT Margin"] = df["EBIT"] / df["Revenue"]
    metrics["Net Income Margin"] = df["Net Income"] / df["Revenue"]
    metrics["Capex as % of Revenue"] = -df["Capital Expenditures"] / df["Revenue"]
    metrics["Depreciation as % of Revenue"] = df["Depreciation"] / df["Revenue"]
    metrics["NWC Change as % of Revenue"] = -df["Change in NWC"] / df["Revenue"]
    
    return metrics
