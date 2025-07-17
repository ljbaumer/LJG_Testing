from typing import Dict, List, Any, Tuple

# Dictionary of tooltips for financial concepts
FINANCIAL_TOOLTIPS = {
    # General DCF concepts
    "dcf": "Discounted Cash Flow (DCF) is a valuation method that estimates the value of an investment based on its expected future cash flows, adjusted for the time value of money.",
    "terminal_value": "Terminal Value represents the value of all future cash flows beyond the explicit forecast period, typically calculated using either the perpetuity growth method or exit multiple method.",
    "wacc": "Weighted Average Cost of Capital (WACC) is the average rate of return a company is expected to pay to all its security holders to finance its assets. It represents the minimum return that a company must earn on its existing asset base to satisfy its creditors, owners, and other providers of capital.",
    
    # Cash flow components
    "revenue": "The total amount of money generated from sales of goods or services related to the company's primary operations.",
    "revenue_growth": "The year-over-year percentage increase in revenue, a key indicator of a company's growth trajectory.",
    "ebitda": "Earnings Before Interest, Taxes, Depreciation, and Amortization (EBITDA) is a measure of a company's overall financial performance and is used as an alternative to net income in some circumstances.",
    "ebitda_margin": "EBITDA Margin is EBITDA as a percentage of revenue, indicating the company's operating profitability.",
    "ebit": "Earnings Before Interest and Taxes (EBIT) represents a company's operating profit before interest and taxes are considered.",
    "nopat": "Net Operating Profit After Tax (NOPAT) represents a company's potential cash earnings if it had no debt (i.e., no interest payments). It's calculated as EBIT × (1 - Tax Rate).",
    "fcf": "Free Cash Flow (FCF) represents the cash a company generates after accounting for cash outflows to support operations and maintain capital assets. It's calculated as NOPAT + Depreciation & Amortization - Capital Expenditures - Change in Working Capital.",
    "capex": "Capital Expenditures (CapEx) are funds used by a company to acquire, upgrade, and maintain physical assets such as property, plants, buildings, technology, or equipment.",
    "working_capital": "Working Capital represents the difference between a company's current assets and current liabilities. Changes in working capital impact cash flow.",
    "depreciation": "Depreciation is an accounting method of allocating the cost of a tangible or physical asset over its useful life or life expectancy.",
    
    # Valuation methods
    "perpetuity_growth": "The Perpetuity Growth Method calculates terminal value by assuming that cash flows will grow at a constant rate forever after the forecast period. The formula is: TV = FCF_t × (1 + g) / (WACC - g), where g is the long-term growth rate.",
    "exit_multiple": "The Exit Multiple Method calculates terminal value by applying a multiple to a financial metric (typically EBITDA or EBIT) in the final forecast year. The formula is: TV = Metric_t × Multiple.",
    "football_field": "A Football Field chart displays valuation ranges across different methodologies, providing a visual representation of how different approaches value the company.",
    "sensitivity_analysis": "Sensitivity Analysis examines how changes in key input variables affect the valuation outcome, helping identify which factors have the greatest impact on value.",
    "scenario_analysis": "Scenario Analysis evaluates the impact of different sets of assumptions (scenarios) on the valuation outcome, typically including base case, upside case, and downside case.",
    
    # WACC components
    "cost_of_equity": "Cost of Equity is the return a company requires to decide if an investment meets capital return requirements. It's often calculated using the Capital Asset Pricing Model (CAPM): Re = Rf + β(Rm - Rf).",
    "cost_of_debt": "Cost of Debt is the effective interest rate a company pays on its debt obligations. The after-tax cost of debt is used in WACC calculations: Rd × (1 - Tax Rate).",
    "risk_free_rate": "Risk-Free Rate is the theoretical return of an investment with zero risk, typically based on government bond yields.",
    "market_risk_premium": "Market Risk Premium is the additional return an investor expects to receive for taking on the extra risk of investing in the stock market over risk-free assets.",
    "beta": "Beta is a measure of a stock's volatility in relation to the overall market. A beta greater than 1 indicates higher volatility than the market, while a beta less than 1 indicates lower volatility.",
    
    # Financial ratios
    "ev_ebitda": "Enterprise Value to EBITDA (EV/EBITDA) is a valuation multiple used to determine the fair market value of a company. Lower multiples may indicate undervaluation.",
    "ev_ebit": "Enterprise Value to EBIT (EV/EBIT) is a valuation multiple that takes into account a company's depreciation and amortization expenses.",
    "pe_ratio": "Price to Earnings (P/E) Ratio is the ratio of a company's share price to its earnings per share. It indicates how much investors are willing to pay for each dollar of earnings.",
    "pb_ratio": "Price to Book (P/B) Ratio is the ratio of a company's market value to its book value. It indicates how much investors are willing to pay for each dollar of assets.",
    
    # Other concepts
    "enterprise_value": "Enterprise Value (EV) is a measure of a company's total value, calculated as market capitalization plus debt, minority interest, and preferred shares, minus cash and cash equivalents.",
    "equity_value": "Equity Value represents the value of a company available to shareholders. It's calculated as Enterprise Value - Debt + Cash.",
    "share_price": "Share Price is the Equity Value divided by the number of outstanding shares.",
    "discount_rate": "Discount Rate is the rate used to convert future cash flows to their present value, reflecting the time value of money and the risk of the investment.",
    "present_value": "Present Value is the current worth of a future sum of money or stream of cash flows given a specified rate of return.",
}

# Dictionary of industry benchmarks
INDUSTRY_BENCHMARKS = {
    "Technology": {
        "revenue_growth": (0.10, 0.20),  # (median, top quartile)
        "ebitda_margin": (0.15, 0.25),
        "capex_to_revenue": (0.05, 0.08),
        "ev_ebitda": (12.0, 18.0),
        "wacc": (0.09, 0.12)
    },
    "Healthcare": {
        "revenue_growth": (0.06, 0.12),
        "ebitda_margin": (0.18, 0.28),
        "capex_to_revenue": (0.04, 0.07),
        "ev_ebitda": (10.0, 15.0),
        "wacc": (0.08, 0.11)
    },
    "Consumer Goods": {
        "revenue_growth": (0.03, 0.08),
        "ebitda_margin": (0.12, 0.20),
        "capex_to_revenue": (0.03, 0.06),
        "ev_ebitda": (8.0, 12.0),
        "wacc": (0.07, 0.10)
    },
    "Financial Services": {
        "revenue_growth": (0.04, 0.10),
        "ebitda_margin": (0.25, 0.40),
        "capex_to_revenue": (0.02, 0.04),
        "ev_ebitda": (8.0, 12.0),
        "wacc": (0.08, 0.11)
    },
    "Energy": {
        "revenue_growth": (0.02, 0.07),
        "ebitda_margin": (0.20, 0.35),
        "capex_to_revenue": (0.10, 0.20),
        "ev_ebitda": (6.0, 9.0),
        "wacc": (0.09, 0.12)
    },
    "Industrials": {
        "revenue_growth": (0.03, 0.08),
        "ebitda_margin": (0.12, 0.18),
        "capex_to_revenue": (0.04, 0.08),
        "ev_ebitda": (7.0, 11.0),
        "wacc": (0.08, 0.11)
    },
    "Telecommunications": {
        "revenue_growth": (0.02, 0.06),
        "ebitda_margin": (0.30, 0.40),
        "capex_to_revenue": (0.12, 0.18),
        "ev_ebitda": (6.0, 8.0),
        "wacc": (0.07, 0.09)
    },
    "Utilities": {
        "revenue_growth": (0.01, 0.04),
        "ebitda_margin": (0.25, 0.35),
        "capex_to_revenue": (0.15, 0.25),
        "ev_ebitda": (7.0, 9.0),
        "wacc": (0.06, 0.08)
    },
    "Real Estate": {
        "revenue_growth": (0.02, 0.06),
        "ebitda_margin": (0.50, 0.65),
        "capex_to_revenue": (0.10, 0.20),
        "ev_ebitda": (12.0, 18.0),
        "wacc": (0.07, 0.09)
    }
}

# Dictionary of formula explanations
FORMULA_EXPLANATIONS = {
    "wacc": {
        "formula": "WACC = (E/V × Re) + (D/V × Rd × (1-T))",
        "variables": {
            "E": "Market value of equity",
            "D": "Market value of debt",
            "V": "Total market value (E + D)",
            "Re": "Cost of equity",
            "Rd": "Cost of debt",
            "T": "Corporate tax rate"
        },
        "explanation": "WACC represents the average rate of return a company must pay to its investors. It weights the cost of equity and debt by their relative proportion in the company's capital structure."
    },
    "cost_of_equity": {
        "formula": "Re = Rf + β(Rm - Rf)",
        "variables": {
            "Rf": "Risk-free rate",
            "β": "Beta (measure of stock volatility)",
            "Rm": "Expected market return",
            "Rm - Rf": "Market risk premium"
        },
        "explanation": "The Capital Asset Pricing Model (CAPM) calculates the cost of equity by adding the risk-free rate to the product of the stock's beta and the market risk premium."
    },
    "terminal_value_perpetuity": {
        "formula": "TV = FCF_t × (1 + g) / (WACC - g)",
        "variables": {
            "FCF_t": "Free cash flow in the final forecast year",
            "g": "Long-term growth rate",
            "WACC": "Weighted average cost of capital"
        },
        "explanation": "The perpetuity growth formula calculates the present value of all future cash flows beyond the forecast period, assuming a constant growth rate forever."
    },
    "terminal_value_exit_multiple": {
        "formula": "TV = Metric_t × Multiple",
        "variables": {
            "Metric_t": "Financial metric (e.g., EBITDA, EBIT) in the final forecast year",
            "Multiple": "Valuation multiple (e.g., EV/EBITDA, EV/EBIT)"
        },
        "explanation": "The exit multiple approach calculates terminal value by applying a market-based multiple to a financial metric in the final forecast year."
    },
    "present_value": {
        "formula": "PV = FV / (1 + r)^n",
        "variables": {
            "FV": "Future value",
            "r": "Discount rate",
            "n": "Number of periods"
        },
        "explanation": "Present value calculates the current worth of a future sum of money, given a specified rate of return."
    },
    "free_cash_flow": {
        "formula": "FCF = NOPAT + D&A - CapEx - ΔWC",
        "variables": {
            "NOPAT": "Net operating profit after tax",
            "D&A": "Depreciation and amortization",
            "CapEx": "Capital expenditures",
            "ΔWC": "Change in working capital"
        },
        "explanation": "Free cash flow represents the cash a company generates after accounting for cash outflows to support operations and maintain capital assets."
    },
    "nopat": {
        "formula": "NOPAT = EBIT × (1 - T)",
        "variables": {
            "EBIT": "Earnings before interest and taxes",
            "T": "Corporate tax rate"
        },
        "explanation": "Net operating profit after tax represents a company's potential cash earnings if it had no debt."
    },
    "enterprise_value": {
        "formula": "EV = Equity Value + Debt - Cash",
        "variables": {
            "Equity Value": "Market capitalization (share price × shares outstanding)",
            "Debt": "Total debt",
            "Cash": "Cash and cash equivalents"
        },
        "explanation": "Enterprise value represents the total value of a company, including both equity and debt, minus cash which could be used to pay off debt."
    },
    "equity_value": {
        "formula": "Equity Value = EV - Debt + Cash",
        "variables": {
            "EV": "Enterprise value",
            "Debt": "Total debt",
            "Cash": "Cash and cash equivalents"
        },
        "explanation": "Equity value represents the value of a company available to shareholders."
    }
}

# Dictionary of growth stage templates
GROWTH_STAGE_TEMPLATES = {
    "Startup": {
        "revenue_growth": 0.30,
        "ebitda_margin": 0.05,
        "tax_rate": 0.15,
        "da_to_revenue": 0.03,
        "capex_to_revenue": 0.10,
        "wc_to_revenue": 0.05,
        "terminal_growth_rate": 0.04,
        "beta": 1.8,
        "description": "High growth, low margins, high reinvestment"
    },
    "Growth": {
        "revenue_growth": 0.15,
        "ebitda_margin": 0.15,
        "tax_rate": 0.25,
        "da_to_revenue": 0.04,
        "capex_to_revenue": 0.08,
        "wc_to_revenue": 0.03,
        "terminal_growth_rate": 0.03,
        "beta": 1.4,
        "description": "Strong growth, improving margins, significant reinvestment"
    },
    "Mature": {
        "revenue_growth": 0.05,
        "ebitda_margin": 0.25,
        "tax_rate": 0.25,
        "da_to_revenue": 0.05,
        "capex_to_revenue": 0.05,
        "wc_to_revenue": 0.01,
        "terminal_growth_rate": 0.02,
        "beta": 1.0,
        "description": "Moderate growth, stable margins, maintenance capital expenditure"
    },
    "Decline": {
        "revenue_growth": 0.00,
        "ebitda_margin": 0.20,
        "tax_rate": 0.25,
        "da_to_revenue": 0.06,
        "capex_to_revenue": 0.03,
        "wc_to_revenue": 0.00,
        "terminal_growth_rate": 0.01,
        "beta": 0.8,
        "description": "Flat or declining growth, pressure on margins, minimal reinvestment"
    }
}

# Dictionary of industry templates
INDUSTRY_TEMPLATES = {
    "Technology": {
        "revenue_growth": 0.12,
        "ebitda_margin": 0.20,
        "tax_rate": 0.20,
        "da_to_revenue": 0.04,
        "capex_to_revenue": 0.06,
        "wc_to_revenue": 0.02,
        "terminal_growth_rate": 0.03,
        "beta": 1.2,
        "exit_multiple": 12.0,
        "description": "High growth, high margins, moderate capital intensity"
    },
    "Healthcare": {
        "revenue_growth": 0.08,
        "ebitda_margin": 0.22,
        "tax_rate": 0.22,
        "da_to_revenue": 0.05,
        "capex_to_revenue": 0.05,
        "wc_to_revenue": 0.02,
        "terminal_growth_rate": 0.03,
        "beta": 0.9,
        "exit_multiple": 10.0,
        "description": "Stable growth, high margins, moderate capital intensity"
    },
    "Consumer Goods": {
        "revenue_growth": 0.05,
        "ebitda_margin": 0.15,
        "tax_rate": 0.25,
        "da_to_revenue": 0.03,
        "capex_to_revenue": 0.04,
        "wc_to_revenue": 0.03,
        "terminal_growth_rate": 0.02,
        "beta": 0.8,
        "exit_multiple": 9.0,
        "description": "Moderate growth, moderate margins, low capital intensity"
    },
    "Financial Services": {
        "revenue_growth": 0.06,
        "ebitda_margin": 0.30,
        "tax_rate": 0.25,
        "da_to_revenue": 0.02,
        "capex_to_revenue": 0.03,
        "wc_to_revenue": 0.01,
        "terminal_growth_rate": 0.02,
        "beta": 1.3,
        "exit_multiple": 8.0,
        "description": "Moderate growth, high margins, low capital intensity"
    },
    "Energy": {
        "revenue_growth": 0.04,
        "ebitda_margin": 0.25,
        "tax_rate": 0.30,
        "da_to_revenue": 0.10,
        "capex_to_revenue": 0.15,
        "wc_to_revenue": 0.02,
        "terminal_growth_rate": 0.02,
        "beta": 1.2,
        "exit_multiple": 6.0,
        "description": "Low growth, high margins, high capital intensity"
    },
    "Industrials": {
        "revenue_growth": 0.05,
        "ebitda_margin": 0.15,
        "tax_rate": 0.25,
        "da_to_revenue": 0.06,
        "capex_to_revenue": 0.06,
        "wc_to_revenue": 0.03,
        "terminal_growth_rate": 0.02,
        "beta": 1.1,
        "exit_multiple": 8.0,
        "description": "Moderate growth, moderate margins, moderate capital intensity"
    }
}

def get_tooltip(key: str) -> str:
    """
    Get a tooltip for a financial concept.
    
    Args:
        key: Key for the tooltip
        
    Returns:
        Tooltip text or empty string if not found
    """
    return FINANCIAL_TOOLTIPS.get(key.lower(), "")

def get_industry_benchmark(industry: str, metric: str) -> Tuple[float, float]:
    """
    Get industry benchmark values for a specific metric.
    
    Args:
        industry: Industry name
        metric: Metric name
        
    Returns:
        Tuple of (median, top quartile) or (0, 0) if not found
    """
    if industry in INDUSTRY_BENCHMARKS and metric in INDUSTRY_BENCHMARKS[industry]:
        return INDUSTRY_BENCHMARKS[industry][metric]
    return (0, 0)

def get_formula_explanation(formula_key: str) -> Dict[str, Any]:
    """
    Get explanation for a financial formula.
    
    Args:
        formula_key: Key for the formula
        
    Returns:
        Dictionary with formula explanation or empty dict if not found
    """
    return FORMULA_EXPLANATIONS.get(formula_key.lower(), {})

def get_growth_stage_template(stage: str) -> Dict[str, Any]:
    """
    Get template parameters for a growth stage.
    
    Args:
        stage: Growth stage name
        
    Returns:
        Dictionary with template parameters or empty dict if not found
    """
    return GROWTH_STAGE_TEMPLATES.get(stage, {})

def get_industry_template(industry: str) -> Dict[str, Any]:
    """
    Get template parameters for an industry.
    
    Args:
        industry: Industry name
        
    Returns:
        Dictionary with template parameters or empty dict if not found
    """
    return INDUSTRY_TEMPLATES.get(industry, {})

def get_all_industries() -> List[str]:
    """
    Get a list of all available industry templates.
    
    Returns:
        List of industry names
    """
    return list(INDUSTRY_TEMPLATES.keys())

def get_all_growth_stages() -> List[str]:
    """
    Get a list of all available growth stage templates.
    
    Returns:
        List of growth stage names
    """
    return list(GROWTH_STAGE_TEMPLATES.keys())
