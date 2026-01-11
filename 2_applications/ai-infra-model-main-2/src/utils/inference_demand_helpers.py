from typing import Dict

from src.utils.streamlit_app_helpers import format_number_to_string


def format_token_consumption_table(daily_data: Dict, daily_all_users_data: Dict, yearly_data: Dict) -> str:
    """
    Create a nicely formatted markdown table for token consumption metrics,
    showing both per-user and aggregate values.
    
    Args:
        daily_data: Dictionary containing daily token consumption metrics per user
        daily_all_users_data: Dictionary containing daily token consumption for all users
        yearly_data: Dictionary containing yearly token consumption metrics for all users
        
    Returns:
        str: Formatted markdown table string
    """
    # Format the per-user values
    daily_user_total = format_number_to_string(daily_data['total_tokens'], is_currency=False)
    daily_user_input = format_number_to_string(daily_data['total_input_tokens'], is_currency=False)
    daily_user_output = format_number_to_string(daily_data['total_output_tokens'], is_currency=False)

    # Format the daily aggregate values
    daily_all_total = format_number_to_string(daily_all_users_data['total_tokens'], is_currency=False)
    daily_all_input = format_number_to_string(daily_all_users_data['total_input_tokens'], is_currency=False)
    daily_all_output = format_number_to_string(daily_all_users_data['total_output_tokens'], is_currency=False)

    # Format the yearly aggregate values
    yearly_total = format_number_to_string(yearly_data['total_tokens'], is_currency=False)
    yearly_input = format_number_to_string(yearly_data['total_input_tokens'], is_currency=False)
    yearly_output = format_number_to_string(yearly_data['total_output_tokens'], is_currency=False)

    # Create the markdown table
    table = f"""
    | Metric        | Daily Per User | Daily All Users | Annual All Users |
    |:--------------|:---------------|:----------------|:-----------------|
    | Total Tokens  | {daily_user_total} | {daily_all_total} | {yearly_total} |
    | Input Tokens  | {daily_user_input} | {daily_all_input} | {yearly_input} |
    | Output Tokens | {daily_user_output} | {daily_all_output} | {yearly_output} |
    """

    return table

def format_token_economics_table(daily_data: Dict, daily_all_users_data: Dict,
                               yearly_data: Dict, num_users: int) -> str:
    """
    Format a comprehensive token economics table with per-user and aggregate metrics.
    
    Args:
        daily_data: Dictionary with daily per-user metrics
        daily_all_users_data: Dictionary with daily aggregate metrics
        yearly_data: Dictionary with yearly aggregate metrics
        num_users: Number of users
        
    Returns:
        str: Formatted markdown table
    """
    # Format per-user token values
    daily_user_total = format_number_to_string(daily_data['total_tokens'], is_currency=False)
    daily_user_input = format_number_to_string(daily_data['total_input_tokens'], is_currency=False)
    daily_user_output = format_number_to_string(daily_data['total_output_tokens'], is_currency=False)
    daily_user_cost = f"${daily_data['total_cost']:.3f}"

    # Format aggregate token values
    daily_all_total = format_number_to_string(daily_all_users_data['total_tokens'], is_currency=False)
    daily_all_input = format_number_to_string(daily_all_users_data['total_input_tokens'], is_currency=False)
    daily_all_output = format_number_to_string(daily_all_users_data['total_output_tokens'], is_currency=False)
    daily_all_cost = format_number_to_string(daily_all_users_data['total_cost'], is_currency=True)

    # Format yearly values
    yearly_total = format_number_to_string(yearly_data['total_tokens'], is_currency=False)
    yearly_input = format_number_to_string(yearly_data['total_input_tokens'], is_currency=False)
    yearly_output = format_number_to_string(yearly_data['total_output_tokens'], is_currency=False)
    yearly_cost = format_number_to_string(yearly_data['total_cost'], is_currency=True)

    table = f"""
    ## Daily Token Economics (Per User)
    
    | Metric           | Value         |
    |:-----------------|:--------------|
    | Total Tokens     | {daily_user_total} |
    | Input Tokens     | {daily_user_input} |
    | Output Tokens    | {daily_user_output} |
    | Cost             | {daily_user_cost} |
    
    ## Daily Token Economics (All {num_users:,} Users)
    
    | Metric           | Value         |
    |:-----------------|:--------------|
    | Total Tokens     | {daily_all_total} |
    | Input Tokens     | {daily_all_input} |
    | Output Tokens    | {daily_all_output} |
    | Cost             | {daily_all_cost} |
    
    ## Annual Token Economics (All {num_users:,} Users)
    
    | Metric           | Value         |
    |:-----------------|:--------------|
    | Total Tokens     | {yearly_total} |
    | Input Tokens     | {yearly_input} |
    | Output Tokens    | {yearly_output} |
    | Cost             | {yearly_cost} |
    """

    return table

def align_token_cost_revenue_table(daily_data: Dict, daily_all_users_data: Dict,
                                 yearly_data: Dict, monthly_revenue: float,
                                 yearly_revenue: float, yearly_profit: float,
                                 profit_margin: float, num_users: int,
                                 days_per_year: int) -> str:
    """
    Create a comprehensive table aligning token consumption with cost, revenue, and profit.
    
    Args:
        daily_data: Dictionary containing daily token consumption metrics
        daily_all_users_data: Dictionary containing daily aggregate metrics
        yearly_data: Dictionary containing yearly token consumption metrics
        monthly_revenue: Monthly revenue amount
        yearly_revenue: Yearly revenue amount
        yearly_profit: Yearly profit amount
        profit_margin: Profit margin percentage
        num_users: Number of users
        days_per_year: Number of days per year (workdays or calendar days)
        
    Returns:
        str: Formatted markdown table string
    """
    # Format the financial values
    daily_user_cost = f"${daily_data['total_cost']:.3f}"
    daily_all_cost = format_number_to_string(daily_all_users_data['total_cost'], is_currency=True)
    yearly_cost = format_number_to_string(yearly_data['total_cost'], is_currency=True)

    # Per user revenue and profit
    daily_user_rev = f"${monthly_revenue/(num_users * 30):.3f}"

    # Calculate daily profit per user (monthly revenue / 30 days - daily cost)
    daily_user_profit = monthly_revenue/(num_users * 30) - daily_data['total_cost']
    daily_user_profit_str = f"${daily_user_profit:.3f}"

    # All users revenue and profit
    daily_all_rev = format_number_to_string(monthly_revenue/30, is_currency=True)
    yearly_rev = format_number_to_string(yearly_revenue, is_currency=True)

    # Calculate daily profit for all users
    daily_all_profit = monthly_revenue/30 - daily_all_users_data['total_cost']
    daily_all_profit_str = format_number_to_string(daily_all_profit, is_currency=True)

    # Annual profit
    yearly_prof = format_number_to_string(yearly_profit, is_currency=True)
    margin = f"{profit_margin:.1f}%"

    # Format token values
    daily_user_total = format_number_to_string(daily_data['total_tokens'], is_currency=False)
    daily_user_input = format_number_to_string(daily_data['total_input_tokens'], is_currency=False)
    daily_user_output = format_number_to_string(daily_data['total_output_tokens'], is_currency=False)

    daily_all_total = format_number_to_string(daily_all_users_data['total_tokens'], is_currency=False)
    daily_all_input = format_number_to_string(daily_all_users_data['total_input_tokens'], is_currency=False)
    daily_all_output = format_number_to_string(daily_all_users_data['total_output_tokens'], is_currency=False)

    yearly_total = format_number_to_string(yearly_data['total_tokens'], is_currency=False)
    yearly_input = format_number_to_string(yearly_data['total_input_tokens'], is_currency=False)
    yearly_output = format_number_to_string(yearly_data['total_output_tokens'], is_currency=False)

    # Create a comprehensive table
    table = f"""
    ## Token Economics Dashboard
    
    | Metric            | Per User (Daily)   | All Users (Daily)    | All Users (Annual)      |
    |:------------------|:-------------------|:---------------------|:------------------------|
    | **Total Tokens**  | {daily_user_total} | {daily_all_total}    | {yearly_total}          |
    | **Input Tokens**  | {daily_user_input} | {daily_all_input}    | {yearly_input}          |
    | **Output Tokens** | {daily_user_output}| {daily_all_output}   | {yearly_output}         |
    | **Cost**          | {daily_user_cost}  | {daily_all_cost}     | {yearly_cost}           |
    | **Revenue**       | {daily_user_rev}   | {daily_all_rev}      | {yearly_rev}            |
    | **Profit/Loss**   | {daily_user_profit_str} | {daily_all_profit_str} | {yearly_prof} ({margin})|
    
    *Note: Annual figures based on {days_per_year} workdays per year*
    """

    return table
