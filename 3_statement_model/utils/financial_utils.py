import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any


def calculate_growth_rate(current_value: float, previous_value: float) -> float:
    """
    Calculate the growth rate between two values.
    
    Args:
        current_value: Current period value
        previous_value: Previous period value
        
    Returns:
        Growth rate as a decimal
    """
    if previous_value == 0:
        return 0
    
    return (current_value - previous_value) / previous_value


def calculate_margin(numerator: float, denominator: float) -> float:
    """
    Calculate a margin ratio.
    
    Args:
        numerator: Numerator value (e.g., Net Income)
        denominator: Denominator value (e.g., Revenue)
        
    Returns:
        Margin as a decimal
    """
    if denominator == 0:
        return 0
    
    return numerator / denominator


def calculate_percentage_of_revenue(value: float, revenue: float) -> float:
    """
    Calculate a value as a percentage of revenue.
    
    Args:
        value: Value to calculate as a percentage of revenue
        revenue: Revenue value
        
    Returns:
        Percentage of revenue as a decimal
    """
    if revenue == 0:
        return 0
    
    return value / revenue


def calculate_working_capital(current_assets: float, current_liabilities: float) -> float:
    """
    Calculate working capital.
    
    Args:
        current_assets: Current assets value
        current_liabilities: Current liabilities value
        
    Returns:
        Working capital value
    """
    return current_assets - current_liabilities


def calculate_change_in_working_capital(current_working_capital: float, previous_working_capital: float) -> float:
    """
    Calculate the change in working capital.
    
    Args:
        current_working_capital: Current period working capital
        previous_working_capital: Previous period working capital
        
    Returns:
        Change in working capital value
    """
    return current_working_capital - previous_working_capital


def calculate_free_cash_flow(operating_cash_flow: float, capital_expenditures: float) -> float:
    """
    Calculate free cash flow.
    
    Args:
        operating_cash_flow: Operating cash flow value
        capital_expenditures: Capital expenditures value
        
    Returns:
        Free cash flow value
    """
    return operating_cash_flow - capital_expenditures


def calculate_days_sales_outstanding(accounts_receivable: float, revenue: float) -> float:
    """
    Calculate days sales outstanding (DSO).
    
    Args:
        accounts_receivable: Accounts receivable value
        revenue: Annual revenue value
        
    Returns:
        Days sales outstanding value
    """
    if revenue == 0:
        return 0
    
    return (accounts_receivable / revenue) * 365


def calculate_days_inventory_outstanding(inventory: float, cost_of_goods_sold: float) -> float:
    """
    Calculate days inventory outstanding (DIO).
    
    Args:
        inventory: Inventory value
        cost_of_goods_sold: Annual cost of goods sold value
        
    Returns:
        Days inventory outstanding value
    """
    if cost_of_goods_sold == 0:
        return 0
    
    return (inventory / cost_of_goods_sold) * 365


def calculate_days_payable_outstanding(accounts_payable: float, cost_of_goods_sold: float) -> float:
    """
    Calculate days payable outstanding (DPO).
    
    Args:
        accounts_payable: Accounts payable value
        cost_of_goods_sold: Annual cost of goods sold value
        
    Returns:
        Days payable outstanding value
    """
    if cost_of_goods_sold == 0:
        return 0
    
    return (accounts_payable / cost_of_goods_sold) * 365


def calculate_cash_conversion_cycle(dso: float, dio: float, dpo: float) -> float:
    """
    Calculate cash conversion cycle (CCC).
    
    Args:
        dso: Days sales outstanding
        dio: Days inventory outstanding
        dpo: Days payable outstanding
        
    Returns:
        Cash conversion cycle value
    """
    return dso + dio - dpo


def calculate_return_on_assets(net_income: float, total_assets: float) -> float:
    """
    Calculate return on assets (ROA).
    
    Args:
        net_income: Net income value
        total_assets: Total assets value
        
    Returns:
        Return on assets as a decimal
    """
    if total_assets == 0:
        return 0
    
    return net_income / total_assets


def calculate_return_on_equity(net_income: float, total_equity: float) -> float:
    """
    Calculate return on equity (ROE).
    
    Args:
        net_income: Net income value
        total_equity: Total equity value
        
    Returns:
        Return on equity as a decimal
    """
    if total_equity == 0:
        return 0
    
    return net_income / total_equity


def calculate_debt_to_equity(total_debt: float, total_equity: float) -> float:
    """
    Calculate debt to equity ratio.
    
    Args:
        total_debt: Total debt value
        total_equity: Total equity value
        
    Returns:
        Debt to equity ratio
    """
    if total_equity == 0:
        return 0
    
    return total_debt / total_equity


def calculate_debt_to_ebitda(total_debt: float, ebitda: float) -> float:
    """
    Calculate debt to EBITDA ratio.
    
    Args:
        total_debt: Total debt value
        ebitda: EBITDA value
        
    Returns:
        Debt to EBITDA ratio
    """
    if ebitda == 0:
        return 0
    
    return total_debt / ebitda


def calculate_interest_coverage_ratio(ebit: float, interest_expense: float) -> float:
    """
    Calculate interest coverage ratio.
    
    Args:
        ebit: EBIT value
        interest_expense: Interest expense value
        
    Returns:
        Interest coverage ratio
    """
    if interest_expense == 0:
        return float('inf')
    
    return ebit / interest_expense


def calculate_current_ratio(current_assets: float, current_liabilities: float) -> float:
    """
    Calculate current ratio.
    
    Args:
        current_assets: Current assets value
        current_liabilities: Current liabilities value
        
    Returns:
        Current ratio
    """
    if current_liabilities == 0:
        return float('inf')
    
    return current_assets / current_liabilities


def calculate_quick_ratio(current_assets: float, inventory: float, current_liabilities: float) -> float:
    """
    Calculate quick ratio.
    
    Args:
        current_assets: Current assets value
        inventory: Inventory value
        current_liabilities: Current liabilities value
        
    Returns:
        Quick ratio
    """
    if current_liabilities == 0:
        return float('inf')
    
    return (current_assets - inventory) / current_liabilities


def calculate_enterprise_value(market_cap: float, total_debt: float, cash: float) -> float:
    """
    Calculate enterprise value.
    
    Args:
        market_cap: Market capitalization value
        total_debt: Total debt value
        cash: Cash and cash equivalents value
        
    Returns:
        Enterprise value
    """
    return market_cap + total_debt - cash


def calculate_ev_to_ebitda(enterprise_value: float, ebitda: float) -> float:
    """
    Calculate EV to EBITDA ratio.
    
    Args:
        enterprise_value: Enterprise value
        ebitda: EBITDA value
        
    Returns:
        EV to EBITDA ratio
    """
    if ebitda == 0:
        return 0
    
    return enterprise_value / ebitda


def calculate_ev_to_revenue(enterprise_value: float, revenue: float) -> float:
    """
    Calculate EV to Revenue ratio.
    
    Args:
        enterprise_value: Enterprise value
        revenue: Revenue value
        
    Returns:
        EV to Revenue ratio
    """
    if revenue == 0:
        return 0
    
    return enterprise_value / revenue


def calculate_price_to_earnings(price_per_share: float, earnings_per_share: float) -> float:
    """
    Calculate price to earnings (P/E) ratio.
    
    Args:
        price_per_share: Price per share value
        earnings_per_share: Earnings per share value
        
    Returns:
        Price to earnings ratio
    """
    if earnings_per_share == 0:
        return 0
    
    return price_per_share / earnings_per_share


def calculate_price_to_sales(price_per_share: float, sales_per_share: float) -> float:
    """
    Calculate price to sales (P/S) ratio.
    
    Args:
        price_per_share: Price per share value
        sales_per_share: Sales per share value
        
    Returns:
        Price to sales ratio
    """
    if sales_per_share == 0:
        return 0
    
    return price_per_share / sales_per_share


def calculate_price_to_book(price_per_share: float, book_value_per_share: float) -> float:
    """
    Calculate price to book (P/B) ratio.
    
    Args:
        price_per_share: Price per share value
        book_value_per_share: Book value per share value
        
    Returns:
        Price to book ratio
    """
    if book_value_per_share == 0:
        return 0
    
    return price_per_share / book_value_per_share


def calculate_dividend_yield(dividend_per_share: float, price_per_share: float) -> float:
    """
    Calculate dividend yield.
    
    Args:
        dividend_per_share: Dividend per share value
        price_per_share: Price per share value
        
    Returns:
        Dividend yield as a decimal
    """
    if price_per_share == 0:
        return 0
    
    return dividend_per_share / price_per_share


def calculate_dividend_payout_ratio(dividend_per_share: float, earnings_per_share: float) -> float:
    """
    Calculate dividend payout ratio.
    
    Args:
        dividend_per_share: Dividend per share value
        earnings_per_share: Earnings per share value
        
    Returns:
        Dividend payout ratio as a decimal
    """
    if earnings_per_share == 0:
        return 0
    
    return dividend_per_share / earnings_per_share


def calculate_rule_of_40(revenue_growth: float, profit_margin: float) -> float:
    """
    Calculate Rule of 40 for SaaS companies.
    
    Args:
        revenue_growth: Revenue growth rate as a decimal
        profit_margin: Profit margin as a decimal
        
    Returns:
        Rule of 40 value
    """
    return (revenue_growth * 100) + (profit_margin * 100)
