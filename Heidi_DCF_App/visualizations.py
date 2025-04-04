"""
Visualizations module for the DCF application.
Contains functions for creating charts and tables.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

def format_currency(value, precision=0):
    """
    Format a value as currency.
    
    Args:
        value (float): Value to format
        precision (int): Number of decimal places
        
    Returns:
        str: Formatted currency string
    """
    if abs(value) >= 1e9:
        return f"${value / 1e9:.{precision}f}B"
    elif abs(value) >= 1e6:
        return f"${value / 1e6:.{precision}f}M"
    elif abs(value) >= 1e3:
        return f"${value / 1e3:.{precision}f}K"
    else:
        return f"${value:.{precision}f}"

def format_percent(value, precision=1):
    """
    Format a value as a percentage.
    
    Args:
        value (float): Value to format
        precision (int): Number of decimal places
        
    Returns:
        str: Formatted percentage string
    """
    return f"{value * 100:.{precision}f}%"

def plot_historical_financials(historical_df, metrics=None):
    """
    Plot historical financial metrics.
    
    Args:
        historical_df (pandas.DataFrame): Historical financial data
        metrics (list): List of metrics to plot
        
    Returns:
        plotly.graph_objects.Figure: Plotly figure
    """
    if metrics is None:
        metrics = ["Revenue", "EBITDA", "EBIT", "Net Income"]
    
    # Create figure
    fig = go.Figure()
    
    # Add traces
    for metric in metrics:
        if metric in historical_df.columns:
            fig.add_trace(go.Bar(
                x=historical_df.index,
                y=historical_df[metric],
                name=metric,
                hovertemplate="%{y:$.2f}",
                text=[format_currency(val) for val in historical_df[metric]],
                textposition="auto"
            ))
    
    # Update layout
    fig.update_layout(
        title="Historical Financial Performance",
        xaxis_title="Year",
        yaxis_title="Amount",
        legend_title="Metric",
        hovermode="x unified",
        template="plotly_white"
    )
    
    return fig

def plot_historical_metrics(metrics_df, metrics=None):
    """
    Plot historical financial metrics as percentages.
    
    Args:
        metrics_df (pandas.DataFrame): Historical metrics data
        metrics (list): List of metrics to plot
        
    Returns:
        plotly.graph_objects.Figure: Plotly figure
    """
    if metrics is None:
        metrics = ["EBITDA Margin", "EBIT Margin", "Net Income Margin"]
    
    # Create figure
    fig = go.Figure()
    
    # Add traces
    for metric in metrics:
        if metric in metrics_df.columns:
            fig.add_trace(go.Scatter(
                x=metrics_df.index,
                y=metrics_df[metric],
                name=metric,
                mode="lines+markers",
                hovertemplate="%{y:.2%}",
                text=[format_percent(val) for val in metrics_df[metric]],
                textposition="top center"
            ))
    
    # Update layout
    fig.update_layout(
        title="Historical Financial Metrics",
        xaxis_title="Year",
        yaxis_title="Percentage",
        yaxis_tickformat=".0%",
        legend_title="Metric",
        hovermode="x unified",
        template="plotly_white"
    )
    
    return fig

def plot_projected_financials(projections_df, metrics=None):
    """
    Plot projected financial metrics.
    
    Args:
        projections_df (pandas.DataFrame): Projected financial data
        metrics (list): List of metrics to plot
        
    Returns:
        plotly.graph_objects.Figure: Plotly figure
    """
    if metrics is None:
        metrics = ["Revenue", "EBITDA", "EBIT", "Free Cash Flow"]
    
    # Create figure
    fig = go.Figure()
    
    # Add traces
    for metric in metrics:
        if metric in projections_df.columns:
            fig.add_trace(go.Bar(
                x=projections_df.index,
                y=projections_df[metric],
                name=metric,
                hovertemplate="%{y:$.2f}",
                text=[format_currency(val) for val in projections_df[metric]],
                textposition="auto"
            ))
    
    # Update layout
    fig.update_layout(
        title="Projected Financial Performance",
        xaxis_title="Year",
        yaxis_title="Amount",
        legend_title="Metric",
        hovermode="x unified",
        template="plotly_white"
    )
    
    return fig

def plot_dcf_valuation_bridge(valuation_results):
    """
    Create a waterfall chart for DCF valuation bridge.
    
    Args:
        valuation_results (dict): DCF valuation results
        
    Returns:
        plotly.graph_objects.Figure: Plotly figure
    """
    # Extract values
    pv_fcfs = valuation_results["Sum of PV of FCFs"]
    pv_terminal = valuation_results["PV of Terminal Value"]
    enterprise_value = valuation_results["Enterprise Value"]
    debt = valuation_results["Debt"]
    cash = valuation_results["Cash"]
    equity_value = valuation_results["Equity Value"]
    
    # Create figure
    fig = go.Figure(go.Waterfall(
        name="DCF Valuation Bridge",
        orientation="v",
        measure=["relative", "relative", "total", "relative", "relative", "total"],
        x=["PV of FCFs", "PV of Terminal Value", "Enterprise Value", "Debt", "Cash", "Equity Value"],
        textposition="outside",
        text=[
            format_currency(pv_fcfs),
            format_currency(pv_terminal),
            format_currency(enterprise_value),
            format_currency(-debt),
            format_currency(cash),
            format_currency(equity_value)
        ],
        y=[pv_fcfs, pv_terminal, 0, -debt, cash, 0],
        connector={"line": {"color": "rgb(63, 63, 63)"}},
    ))
    
    # Update layout
    fig.update_layout(
        title="DCF Valuation Bridge",
        showlegend=False,
        template="plotly_white"
    )
    
    return fig

def plot_fcf_breakdown(projections_df):
    """
    Create a stacked bar chart for FCF breakdown.
    
    Args:
        projections_df (pandas.DataFrame): Projected financial data
        
    Returns:
        plotly.graph_objects.Figure: Plotly figure
    """
    # Create figure
    fig = go.Figure()
    
    # Add traces
    fig.add_trace(go.Bar(
        x=projections_df.index,
        y=projections_df["NOPAT"],
        name="NOPAT",
        hovertemplate="%{y:$.2f}"
    ))
    
    fig.add_trace(go.Bar(
        x=projections_df.index,
        y=projections_df["Depreciation"],
        name="Depreciation",
        hovertemplate="%{y:$.2f}"
    ))
    
    fig.add_trace(go.Bar(
        x=projections_df.index,
        y=projections_df["Capital Expenditures"],
        name="Capital Expenditures",
        hovertemplate="%{y:$.2f}"
    ))
    
    fig.add_trace(go.Bar(
        x=projections_df.index,
        y=projections_df["Change in NWC"],
        name="Change in NWC",
        hovertemplate="%{y:$.2f}"
    ))
    
    fig.add_trace(go.Scatter(
        x=projections_df.index,
        y=projections_df["Free Cash Flow"],
        name="Free Cash Flow",
        mode="lines+markers",
        line=dict(color="black", width=2),
        hovertemplate="%{y:$.2f}"
    ))
    
    # Update layout
    fig.update_layout(
        title="Free Cash Flow Breakdown",
        xaxis_title="Year",
        yaxis_title="Amount",
        legend_title="Component",
        barmode="relative",
        hovermode="x unified",
        template="plotly_white"
    )
    
    return fig

def plot_sensitivity_analysis(sensitivity_results, param_name, metric="Share Price"):
    """
    Create a line chart for sensitivity analysis.
    
    Args:
        sensitivity_results (dict): Sensitivity analysis results
        param_name (str): Parameter name
        metric (str): Metric to plot
        
    Returns:
        plotly.graph_objects.Figure: Plotly figure
    """
    if param_name not in sensitivity_results:
        return None
    
    # Extract data
    param_results = sensitivity_results[param_name]
    values = [result["Value"] for result in param_results]
    metrics = [result[metric] for result in param_results]
    
    # Create figure
    fig = go.Figure()
    
    # Add trace
    fig.add_trace(go.Scatter(
        x=values,
        y=metrics,
        mode="lines+markers",
        name=metric,
        hovertemplate="%{x}, %{y:$.2f}",
        line=dict(width=2)
    ))
    
    # Update layout
    fig.update_layout(
        title=f"Sensitivity Analysis: {metric} vs {param_name}",
        xaxis_title=param_name,
        yaxis_title=metric,
        template="plotly_white"
    )
    
    return fig

def plot_sensitivity_table(sensitivity_table):
    """
    Create a heatmap for sensitivity table.
    
    Args:
        sensitivity_table (pandas.DataFrame): Sensitivity table
        
    Returns:
        plotly.graph_objects.Figure: Plotly figure
    """
    # Format the values
    text = [[f"${val:.2f}" for val in row] for row in sensitivity_table.values]
    
    # Create figure
    fig = go.Figure(data=go.Heatmap(
        z=sensitivity_table.values,
        x=sensitivity_table.columns,
        y=sensitivity_table.index,
        colorscale="RdBu",
        text=text,
        texttemplate="%{text}",
        colorbar=dict(title="Share Price")
    ))
    
    # Update layout
    fig.update_layout(
        title="Two-Factor Sensitivity Analysis",
        xaxis_title=sensitivity_table.columns.name,
        yaxis_title=sensitivity_table.index.name,
        template="plotly_white"
    )
    
    return fig

def create_summary_metrics_table(valuation_results, current_price=None):
    """
    Create a summary metrics table.
    
    Args:
        valuation_results (dict): DCF valuation results
        current_price (float): Current stock price
        
    Returns:
        pandas.DataFrame: Summary metrics table
    """
    # Extract values
    enterprise_value = valuation_results["Enterprise Value"]
    equity_value = valuation_results["Equity Value"]
    share_price = valuation_results.get("Share Price", None)
    
    # Create metrics dictionary
    metrics = {
        "Enterprise Value": format_currency(enterprise_value),
        "Equity Value": format_currency(equity_value)
    }
    
    if share_price is not None:
        metrics["Implied Share Price"] = f"${share_price:.2f}"
        
        if current_price is not None:
            upside = (share_price / current_price - 1) * 100
            metrics["Current Price"] = f"${current_price:.2f}"
            metrics["Upside/Downside"] = f"{upside:.1f}%"
    
    # Create DataFrame
    df = pd.DataFrame({"Metric": list(metrics.keys()), "Value": list(metrics.values())})
    
    return df

def create_dcf_summary_table(valuation_results):
    """
    Create a DCF summary table.
    
    Args:
        valuation_results (dict): DCF valuation results
        
    Returns:
        pandas.DataFrame: DCF summary table
    """
    # Extract values
    pv_fcfs = valuation_results["Sum of PV of FCFs"]
    pv_terminal = valuation_results["PV of Terminal Value"]
    enterprise_value = valuation_results["Enterprise Value"]
    debt = valuation_results["Debt"]
    cash = valuation_results["Cash"]
    equity_value = valuation_results["Equity Value"]
    
    # Calculate percentages
    pv_fcfs_pct = pv_fcfs / enterprise_value
    pv_terminal_pct = pv_terminal / enterprise_value
    
    # Create data dictionary
    data = {
        "Component": [
            "PV of Forecast Period FCFs",
            "PV of Terminal Value",
            "Enterprise Value",
            "Less: Debt",
            "Plus: Cash",
            "Equity Value"
        ],
        "Value": [
            format_currency(pv_fcfs),
            format_currency(pv_terminal),
            format_currency(enterprise_value),
            format_currency(debt),
            format_currency(cash),
            format_currency(equity_value)
        ],
        "% of EV": [
            format_percent(pv_fcfs_pct),
            format_percent(pv_terminal_pct),
            "100.0%",
            "",
            "",
            ""
        ]
    }
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    return df

def create_projected_financials_table(projections_df):
    """
    Create a projected financials table.
    
    Args:
        projections_df (pandas.DataFrame): Projected financial data
        
    Returns:
        pandas.DataFrame: Formatted projected financials table
    """
    # Create a copy of the DataFrame
    df = projections_df.copy()
    
    # Format currency columns
    currency_columns = [
        "Revenue", "EBITDA", "Depreciation", "EBIT", 
        "Taxes", "NOPAT", "Capital Expenditures", 
        "Change in NWC", "Free Cash Flow"
    ]
    
    for col in currency_columns:
        if col in df.columns:
            df[col] = df[col].apply(format_currency)
    
    # Format percentage columns
    pct_columns = ["EBITDA Margin"]
    
    for col in pct_columns:
        if col in df.columns:
            df[col] = df[col].apply(format_percent)
    
    return df

def plot_company_overview(company_data):
    """
    Create a company overview visualization.
    
    Args:
        company_data (dict): Company data
        
    Returns:
        plotly.graph_objects.Figure: Plotly figure
    """
    # Extract data
    name = company_data["name"]
    sector = company_data["sector"]
    description = company_data["description"]
    current_price = company_data["current_price"]
    shares_outstanding = company_data["shares_outstanding"]
    debt = company_data["debt"]
    cash = company_data["cash"]
    
    # Calculate market cap
    market_cap = current_price * shares_outstanding
    
    # Create figure with subplots
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "domain"}, {"type": "domain"}]],
        subplot_titles=("Capital Structure", "Enterprise Value Components")
    )
    
    # Add pie chart for capital structure
    fig.add_trace(go.Pie(
        labels=["Equity", "Debt", "Cash"],
        values=[market_cap, debt, cash],
        name="Capital Structure",
        hole=0.4,
        textinfo="label+percent",
        marker_colors=["#4CAF50", "#F44336", "#2196F3"]
    ), 1, 1)
    
    # Add pie chart for enterprise value components
    fig.add_trace(go.Pie(
        labels=["Market Cap", "Debt", "Cash"],
        values=[market_cap, debt, -cash],
        name="Enterprise Value Components",
        hole=0.4,
        textinfo="label+percent",
        marker_colors=["#4CAF50", "#F44336", "#2196F3"]
    ), 1, 2)
    
    # Update layout
    fig.update_layout(
        title_text=f"{name} ({sector}) - Overview",
        annotations=[
            dict(
                text=f"Market Cap: {format_currency(market_cap)}<br>Share Price: ${current_price:.2f}",
                x=0.5, y=0.5,
                font_size=10,
                showarrow=False
            ),
            dict(
                text=f"EV: {format_currency(market_cap + debt - cash)}",
                x=1.5, y=0.5,
                font_size=10,
                showarrow=False
            )
        ],
        template="plotly_white"
    )
    
    return fig
