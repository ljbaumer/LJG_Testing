import numpy as np
from typing import Dict, List, Optional, Union


def calculate_wacc(
    equity_value: float,
    debt_value: float,
    cost_of_equity: float,
    cost_of_debt: float,
    tax_rate: float
) -> float:
    """
    Calculate the Weighted Average Cost of Capital (WACC).
    
    Args:
        equity_value: Market value of equity
        debt_value: Market value of debt
        cost_of_equity: Cost of equity (as a decimal, e.g., 0.10 for 10%)
        cost_of_debt: Cost of debt before tax (as a decimal)
        tax_rate: Corporate tax rate (as a decimal)
        
    Returns:
        WACC as a decimal
    """
    total_value = equity_value + debt_value
    
    if total_value == 0:
        raise ValueError("Total value (equity + debt) cannot be zero")
    
    equity_weight = equity_value / total_value
    debt_weight = debt_value / total_value
    
    # WACC = (E/V * Re) + (D/V * Rd * (1 - T))
    wacc = (equity_weight * cost_of_equity) + (debt_weight * cost_of_debt * (1 - tax_rate))
    
    return wacc


def calculate_cost_of_equity(
    risk_free_rate: float,
    market_risk_premium: float,
    beta: float,
    size_premium: float = 0,
    company_specific_premium: float = 0
) -> float:
    """
    Calculate the cost of equity using the Capital Asset Pricing Model (CAPM)
    with optional size and company-specific risk premiums.
    
    Args:
        risk_free_rate: Risk-free rate (as a decimal)
        market_risk_premium: Market risk premium (as a decimal)
        beta: Company's beta
        size_premium: Size premium for small companies (as a decimal)
        company_specific_premium: Company-specific risk premium (as a decimal)
        
    Returns:
        Cost of equity as a decimal
    """
    # Re = Rf + β(Rm - Rf) + size premium + company-specific premium
    cost_of_equity = risk_free_rate + (beta * market_risk_premium) + size_premium + company_specific_premium
    
    return cost_of_equity


def calculate_terminal_value_perpetuity(
    final_cash_flow: float,
    growth_rate: float,
    discount_rate: float
) -> float:
    """
    Calculate terminal value using the perpetuity growth method.
    
    Args:
        final_cash_flow: Cash flow in the final forecast year
        growth_rate: Long-term growth rate (as a decimal)
        discount_rate: Discount rate (WACC, as a decimal)
        
    Returns:
        Terminal value
    """
    if discount_rate <= growth_rate:
        raise ValueError("Discount rate must be greater than growth rate for perpetuity calculation")
    
    # TV = FCF_t+1 / (WACC - g) = FCF_t * (1 + g) / (WACC - g)
    terminal_value = final_cash_flow * (1 + growth_rate) / (discount_rate - growth_rate)
    
    return terminal_value


def calculate_terminal_value_exit_multiple(
    final_metric: float,
    multiple: float
) -> float:
    """
    Calculate terminal value using the exit multiple method.
    
    Args:
        final_metric: Financial metric in the final forecast year (e.g., EBITDA)
        multiple: Exit multiple to apply
        
    Returns:
        Terminal value
    """
    # TV = Final Metric * Multiple
    terminal_value = final_metric * multiple
    
    return terminal_value


def calculate_present_value(
    future_value: float,
    discount_rate: float,
    periods: int
) -> float:
    """
    Calculate the present value of a future cash flow.
    
    Args:
        future_value: Future cash flow value
        discount_rate: Discount rate (as a decimal)
        periods: Number of periods (years)
        
    Returns:
        Present value
    """
    # PV = FV / (1 + r)^n
    present_value = future_value / ((1 + discount_rate) ** periods)
    
    return present_value


def calculate_npv(
    cash_flows: List[float],
    discount_rate: float,
    initial_investment: float = 0
) -> float:
    """
    Calculate Net Present Value (NPV) of a series of cash flows.
    
    Args:
        cash_flows: List of future cash flows
        discount_rate: Discount rate (as a decimal)
        initial_investment: Initial investment (negative cash flow at time 0)
        
    Returns:
        Net Present Value
    """
    npv = -initial_investment
    
    for i, cf in enumerate(cash_flows):
        npv += cf / ((1 + discount_rate) ** (i + 1))
    
    return npv


def calculate_irr(
    cash_flows: List[float],
    initial_investment: float,
    guess: float = 0.1
) -> Optional[float]:
    """
    Calculate Internal Rate of Return (IRR) using numerical methods.
    
    Args:
        cash_flows: List of future cash flows
        initial_investment: Initial investment (positive value)
        guess: Initial guess for IRR
        
    Returns:
        IRR as a decimal, or None if calculation fails to converge
    """
    # Prepare cash flows with initial investment as negative
    full_cash_flows = [-initial_investment] + cash_flows
    
    try:
        # Use numpy's IRR function
        irr = np.irr(full_cash_flows)
        return irr
    except:
        return None


def calculate_enterprise_value(
    equity_value: float,
    debt_value: float,
    cash_equivalents: float
) -> float:
    """
    Calculate Enterprise Value.
    
    Args:
        equity_value: Market value of equity
        debt_value: Market value of debt
        cash_equivalents: Cash and cash equivalents
        
    Returns:
        Enterprise Value
    """
    # EV = Equity Value + Debt - Cash
    enterprise_value = equity_value + debt_value - cash_equivalents
    
    return enterprise_value


def calculate_equity_value(
    enterprise_value: float,
    debt_value: float,
    cash_equivalents: float
) -> float:
    """
    Calculate Equity Value from Enterprise Value.
    
    Args:
        enterprise_value: Enterprise Value
        debt_value: Market value of debt
        cash_equivalents: Cash and cash equivalents
        
    Returns:
        Equity Value
    """
    # Equity Value = EV - Debt + Cash
    equity_value = enterprise_value - debt_value + cash_equivalents
    
    return equity_value


def calculate_perpetuity_value(
    cash_flow: float,
    discount_rate: float,
    growth_rate: float = 0
) -> float:
    """
    Calculate the value of a perpetuity.
    
    Args:
        cash_flow: Annual cash flow
        discount_rate: Discount rate (as a decimal)
        growth_rate: Growth rate (as a decimal)
        
    Returns:
        Perpetuity value
    """
    if discount_rate <= growth_rate:
        raise ValueError("Discount rate must be greater than growth rate for perpetuity calculation")
    
    # Value = CF / (r - g)
    value = cash_flow / (discount_rate - growth_rate)
    
    return value
