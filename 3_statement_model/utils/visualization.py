import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Union, Any


def create_line_chart(data: pd.DataFrame, x_column: str, y_columns: List[str], title: str = None, colors: List[str] = None) -> go.Figure:
    """
    Create a line chart with multiple lines.
    
    Args:
        data: DataFrame containing the data
        x_column: Column name for the x-axis
        y_columns: List of column names for the y-axis
        title: Chart title
        colors: List of colors for the lines
        
    Returns:
        Plotly figure object
    """
    fig = go.Figure()
    
    # Use default colors if not provided
    if colors is None:
        colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown', 'pink', 'gray', 'olive', 'cyan']
    
    # Add a line for each y column
    for i, y_column in enumerate(y_columns):
        color = colors[i % len(colors)]
        fig.add_trace(
            go.Scatter(
                x=data[x_column],
                y=data[y_column],
                mode='lines+markers',
                name=y_column,
                line=dict(color=color)
            )
        )
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title=x_column,
        yaxis_title="Value",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        hovermode="x unified"
    )
    
    return fig


def create_bar_chart(data: pd.DataFrame, x_column: str, y_columns: List[str], title: str = None, colors: List[str] = None) -> go.Figure:
    """
    Create a bar chart with multiple bars.
    
    Args:
        data: DataFrame containing the data
        x_column: Column name for the x-axis
        y_columns: List of column names for the y-axis
        title: Chart title
        colors: List of colors for the bars
        
    Returns:
        Plotly figure object
    """
    fig = go.Figure()
    
    # Use default colors if not provided
    if colors is None:
        colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown', 'pink', 'gray', 'olive', 'cyan']
    
    # Add a bar for each y column
    for i, y_column in enumerate(y_columns):
        color = colors[i % len(colors)]
        fig.add_trace(
            go.Bar(
                x=data[x_column],
                y=data[y_column],
                name=y_column,
                marker_color=color
            )
        )
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title=x_column,
        yaxis_title="Value",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        hovermode="x unified"
    )
    
    return fig


def create_stacked_bar_chart(data: pd.DataFrame, x_column: str, y_columns: List[str], title: str = None, colors: List[str] = None) -> go.Figure:
    """
    Create a stacked bar chart.
    
    Args:
        data: DataFrame containing the data
        x_column: Column name for the x-axis
        y_columns: List of column names for the y-axis
        title: Chart title
        colors: List of colors for the bars
        
    Returns:
        Plotly figure object
    """
    fig = go.Figure()
    
    # Use default colors if not provided
    if colors is None:
        colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown', 'pink', 'gray', 'olive', 'cyan']
    
    # Add a bar for each y column
    for i, y_column in enumerate(y_columns):
        color = colors[i % len(colors)]
        fig.add_trace(
            go.Bar(
                x=data[x_column],
                y=data[y_column],
                name=y_column,
                marker_color=color
            )
        )
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title=x_column,
        yaxis_title="Value",
        barmode='stack',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        hovermode="x unified"
    )
    
    return fig


def create_area_chart(data: pd.DataFrame, x_column: str, y_columns: List[str], title: str = None, colors: List[str] = None) -> go.Figure:
    """
    Create an area chart with multiple areas.
    
    Args:
        data: DataFrame containing the data
        x_column: Column name for the x-axis
        y_columns: List of column names for the y-axis
        title: Chart title
        colors: List of colors for the areas
        
    Returns:
        Plotly figure object
    """
    fig = go.Figure()
    
    # Use default colors if not provided
    if colors is None:
        colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown', 'pink', 'gray', 'olive', 'cyan']
    
    # Add an area for each y column
    for i, y_column in enumerate(y_columns):
        color = colors[i % len(colors)]
        fig.add_trace(
            go.Scatter(
                x=data[x_column],
                y=data[y_column],
                mode='lines',
                name=y_column,
                line=dict(color=color),
                fill='tozeroy'
            )
        )
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title=x_column,
        yaxis_title="Value",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        hovermode="x unified"
    )
    
    return fig


def create_pie_chart(data: pd.DataFrame, labels_column: str, values_column: str, title: str = None) -> go.Figure:
    """
    Create a pie chart.
    
    Args:
        data: DataFrame containing the data
        labels_column: Column name for the labels
        values_column: Column name for the values
        title: Chart title
        
    Returns:
        Plotly figure object
    """
    fig = go.Figure(
        data=[
            go.Pie(
                labels=data[labels_column],
                values=data[values_column],
                textinfo='label+percent',
                insidetextorientation='radial'
            )
        ]
    )
    
    # Update layout
    fig.update_layout(
        title=title
    )
    
    return fig


def create_historical_vs_forecast_chart(data: pd.DataFrame, year_column: str, metric_column: str, historical_end_year: int, title: str = None) -> go.Figure:
    """
    Create a chart showing historical vs forecast data.
    
    Args:
        data: DataFrame containing the data
        year_column: Column name for the year
        metric_column: Column name for the metric
        historical_end_year: Last year of historical data
        title: Chart title
        
    Returns:
        Plotly figure object
    """
    fig = go.Figure()
    
    # Split the data into historical and forecast
    historical_data = data[data[year_column] <= historical_end_year]
    forecast_data = data[data[year_column] > historical_end_year]
    
    # Add historical data
    fig.add_trace(
        go.Scatter(
            x=historical_data[year_column],
            y=historical_data[metric_column],
            mode='lines+markers',
            name='Historical',
            line=dict(color='blue')
        )
    )
    
    # Add forecast data
    fig.add_trace(
        go.Scatter(
            x=forecast_data[year_column],
            y=forecast_data[metric_column],
            mode='lines+markers',
            name='Forecast',
            line=dict(color='red', dash='dash')
        )
    )
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title=year_column,
        yaxis_title=metric_column,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        hovermode="x unified"
    )
    
    # Add a vertical line at the historical end year
    fig.add_vline(
        x=historical_end_year,
        line_width=1,
        line_dash="dash",
        line_color="gray"
    )
    
    return fig


def create_scenario_comparison_chart(base_data: pd.DataFrame, upside_data: pd.DataFrame, downside_data: pd.DataFrame, year_column: str = "Year", metric_column: str = None, title: str = None) -> go.Figure:
    """
    Create a chart comparing different scenarios.
    
    Args:
        base_data: DataFrame containing the base case data
        upside_data: DataFrame containing the upside case data
        downside_data: DataFrame containing the downside case data
        year_column: Column name for the year
        metric_column: Column name for the metric
        title: Chart title
        
    Returns:
        Plotly figure object
    """
    fig = go.Figure()
    
    # Add base case data
    fig.add_trace(
        go.Scatter(
            x=base_data[year_column],
            y=base_data[metric_column] if metric_column else base_data.iloc[:, 1],
            mode='lines+markers',
            name='Base Case',
            line=dict(color='blue')
        )
    )
    
    # Add upside case data
    fig.add_trace(
        go.Scatter(
            x=upside_data[year_column],
            y=upside_data[metric_column] if metric_column else upside_data.iloc[:, 1],
            mode='lines+markers',
            name='Upside Case',
            line=dict(color='green')
        )
    )
    
    # Add downside case data
    fig.add_trace(
        go.Scatter(
            x=downside_data[year_column],
            y=downside_data[metric_column] if metric_column else downside_data.iloc[:, 1],
            mode='lines+markers',
            name='Downside Case',
            line=dict(color='red')
        )
    )
    
    # Update layout
    fig.update_layout(
        title=title if title else f"Scenario Comparison: {metric_column if metric_column else 'Metric'}",
        xaxis_title=year_column,
        yaxis_title=metric_column if metric_column else "Value",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        hovermode="x unified"
    )
    
    return fig


def create_statement_structure_chart(data: pd.DataFrame, year_column: str, component_columns: List[str], title: str = None) -> go.Figure:
    """
    Create a chart showing the structure of a financial statement.
    
    Args:
        data: DataFrame containing the data
        year_column: Column name for the year
        component_columns: List of column names for the components
        title: Chart title
        
    Returns:
        Plotly figure object
    """
    # Create a stacked bar chart
    fig = create_stacked_bar_chart(
        data=data,
        x_column=year_column,
        y_columns=component_columns,
        title=title
    )
    
    return fig


def create_financial_metrics_chart(data: pd.DataFrame, year_column: str, metric_columns: List[str], title: str = None) -> go.Figure:
    """
    Create a chart showing multiple financial metrics.
    
    Args:
        data: DataFrame containing the data
        year_column: Column name for the year
        metric_columns: List of column names for the metrics
        title: Chart title
        
    Returns:
        Plotly figure object
    """
    # Create a line chart
    fig = create_line_chart(
        data=data,
        x_column=year_column,
        y_columns=metric_columns,
        title=title
    )
    
    return fig


def create_waterfall_chart(data: pd.DataFrame, x_column: str, y_column: str, title: str = None) -> go.Figure:
    """
    Create a waterfall chart.
    
    Args:
        data: DataFrame containing the data
        x_column: Column name for the x-axis
        y_column: Column name for the y-axis
        title: Chart title
        
    Returns:
        Plotly figure object
    """
    # Calculate the cumulative sum
    data['cumulative'] = data[y_column].cumsum()
    
    # Calculate the previous cumulative sum
    data['previous'] = data['cumulative'].shift(1).fillna(0)
    
    # Create the waterfall chart
    fig = go.Figure(
        go.Waterfall(
            name=y_column,
            orientation="v",
            measure=["absolute"] + ["relative"] * (len(data) - 2) + ["total"],
            x=data[x_column],
            textposition="outside",
            text=data[y_column].round(2),
            y=data[y_column],
            connector={"line": {"color": "rgb(63, 63, 63)"}},
        )
    )
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title=x_column,
        yaxis_title=y_column,
        showlegend=False
    )
    
    return fig


def create_heatmap(data: pd.DataFrame, x_column: str, y_column: str, z_column: str, title: str = None) -> go.Figure:
    """
    Create a heatmap.
    
    Args:
        data: DataFrame containing the data
        x_column: Column name for the x-axis
        y_column: Column name for the y-axis
        z_column: Column name for the z-axis (color)
        title: Chart title
        
    Returns:
        Plotly figure object
    """
    # Pivot the data
    pivot_data = data.pivot(index=y_column, columns=x_column, values=z_column)
    
    # Create the heatmap
    fig = go.Figure(
        data=go.Heatmap(
            z=pivot_data.values,
            x=pivot_data.columns,
            y=pivot_data.index,
            colorscale='Viridis',
            colorbar=dict(title=z_column)
        )
    )
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title=x_column,
        yaxis_title=y_column
    )
    
    return fig


def create_radar_chart(data: pd.DataFrame, category_column: str, value_column: str, title: str = None) -> go.Figure:
    """
    Create a radar chart.
    
    Args:
        data: DataFrame containing the data
        category_column: Column name for the categories
        value_column: Column name for the values
        title: Chart title
        
    Returns:
        Plotly figure object
    """
    # Create the radar chart
    fig = go.Figure(
        data=go.Scatterpolar(
            r=data[value_column],
            theta=data[category_column],
            fill='toself'
        )
    )
    
    # Update layout
    fig.update_layout(
        title=title,
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, data[value_column].max() * 1.1]
            )
        )
    )
    
    return fig


def create_bubble_chart(data: pd.DataFrame, x_column: str, y_column: str, size_column: str, color_column: str = None, title: str = None) -> go.Figure:
    """
    Create a bubble chart.
    
    Args:
        data: DataFrame containing the data
        x_column: Column name for the x-axis
        y_column: Column name for the y-axis
        size_column: Column name for the bubble size
        color_column: Column name for the bubble color
        title: Chart title
        
    Returns:
        Plotly figure object
    """
    # Create the bubble chart
    fig = go.Figure(
        data=go.Scatter(
            x=data[x_column],
            y=data[y_column],
            mode='markers',
            marker=dict(
                size=data[size_column],
                color=data[color_column] if color_column else None,
                colorscale='Viridis',
                showscale=True if color_column else False,
                colorbar=dict(title=color_column) if color_column else None
            ),
            text=data.index
        )
    )
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title=x_column,
        yaxis_title=y_column
    )
    
    return fig
