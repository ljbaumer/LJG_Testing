import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Optional, Tuple, Union


def create_historical_vs_forecast_chart(
    data: pd.DataFrame,
    metric: str,
    forecast_years: List[int],
    title: Optional[str] = None,
    use_plotly: bool = True
):
    """
    Create a chart showing historical vs forecast values for a specific metric.
    
    Args:
        data: DataFrame containing the financial data
        metric: Column name of the metric to visualize
        forecast_years: List of years that are forecasted
        title: Chart title (optional)
        use_plotly: If True, use Plotly for interactive charts, else use Matplotlib
        
    Returns:
        Plotly figure or Matplotlib axis object
    """
    if metric not in data.columns:
        raise ValueError(f"Metric '{metric}' not found in data")
    
    if 'Year' not in data.columns:
        raise ValueError("Data must contain a 'Year' column")
    
    # Separate historical and forecast data
    historical_data = data[~data['Year'].isin(forecast_years)]
    forecast_data = data[data['Year'].isin(forecast_years)]
    
    if use_plotly:
        fig = go.Figure()
        
        # Add historical data
        fig.add_trace(go.Bar(
            x=historical_data['Year'],
            y=historical_data[metric],
            name='Historical',
            marker_color='blue'
        ))
        
        # Add forecast data
        fig.add_trace(go.Bar(
            x=forecast_data['Year'],
            y=forecast_data[metric],
            name='Forecast',
            marker_color='lightblue'
        ))
        
        # Update layout
        fig.update_layout(
            title=title or f'{metric} - Historical vs Forecast',
            xaxis_title='Year',
            yaxis_title=metric,
            barmode='group',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        return fig
    else:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot historical data
        ax.bar(historical_data['Year'], historical_data[metric], color='blue', label='Historical')
        
        # Plot forecast data
        ax.bar(forecast_data['Year'], forecast_data[metric], color='lightblue', label='Forecast')
        
        # Add labels and title
        ax.set_xlabel('Year')
        ax.set_ylabel(metric)
        ax.set_title(title or f'{metric} - Historical vs Forecast')
        ax.legend()
        
        plt.tight_layout()
        return ax


def create_dcf_waterfall_chart(
    components: Dict[str, float],
    title: str = "DCF Valuation Components",
    use_plotly: bool = True
):
    """
    Create a waterfall chart showing the components of a DCF valuation.
    
    Args:
        components: Dictionary of component names and values
        title: Chart title
        use_plotly: If True, use Plotly for interactive charts, else use Matplotlib
        
    Returns:
        Plotly figure or Matplotlib axis object
    """
    names = list(components.keys())
    values = list(components.values())
    
    if use_plotly:
        # Determine if each component is increasing or decreasing
        measure = ['absolute'] + ['relative'] * (len(names) - 2) + ['total']
        
        fig = go.Figure(go.Waterfall(
            name=title,
            orientation="v",
            measure=measure,
            x=names,
            y=values,
            connector={"line": {"color": "rgb(63, 63, 63)"}},
        ))
        
        fig.update_layout(
            title=title,
            showlegend=False
        )
        
        return fig
    else:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create a waterfall chart using matplotlib
        # This is a simplified version and may need enhancement
        cumulative = 0
        for i, (name, value) in enumerate(components.items()):
            if i == 0 or i == len(components) - 1:
                # First and last items are absolute
                ax.bar(i, value, bottom=0, color='blue' if value >= 0 else 'red')
                if i == 0:
                    cumulative = value
            else:
                # Middle items are relative
                ax.bar(i, value, bottom=cumulative, color='green' if value >= 0 else 'red')
                cumulative += value
        
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha='right')
        ax.set_title(title)
        
        plt.tight_layout()
        return ax


def create_sensitivity_analysis_heatmap(
    x_values: List[float],
    y_values: List[float],
    z_values: List[List[float]],
    x_label: str,
    y_label: str,
    title: str = "Sensitivity Analysis",
    use_plotly: bool = True
):
    """
    Create a heatmap for sensitivity analysis.
    
    Args:
        x_values: List of values for the x-axis
        y_values: List of values for the y-axis
        z_values: 2D list of values for the heatmap
        x_label: Label for the x-axis
        y_label: Label for the y-axis
        title: Chart title
        use_plotly: If True, use Plotly for interactive charts, else use Matplotlib
        
    Returns:
        Plotly figure or Matplotlib axis object
    """
    if use_plotly:
        fig = go.Figure(data=go.Heatmap(
            z=z_values,
            x=x_values,
            y=y_values,
            colorscale='RdBu_r',
            colorbar=dict(title="Value"),
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title=x_label,
            yaxis_title=y_label,
        )
        
        return fig
    else:
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create heatmap
        im = ax.imshow(z_values, cmap='RdBu_r')
        
        # Set ticks and labels
        ax.set_xticks(np.arange(len(x_values)))
        ax.set_yticks(np.arange(len(y_values)))
        ax.set_xticklabels(x_values)
        ax.set_yticklabels(y_values)
        
        # Rotate x tick labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Add colorbar
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel("Value", rotation=-90, va="bottom")
        
        # Add labels and title
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(title)
        
        plt.tight_layout()
        return ax


def create_multiple_comparison_chart(
    multiples: Dict[str, float],
    company_multiple: float,
    title: str = "Comparable Company Multiples",
    use_plotly: bool = True
):
    """
    Create a chart comparing multiples across comparable companies.
    
    Args:
        multiples: Dictionary of company names and their multiples
        company_multiple: The multiple of the company being valued
        title: Chart title
        use_plotly: If True, use Plotly for interactive charts, else use Matplotlib
        
    Returns:
        Plotly figure or Matplotlib axis object
    """
    companies = list(multiples.keys()) + ['Target Company']
    values = list(multiples.values()) + [company_multiple]
    
    # Calculate average multiple
    avg_multiple = sum(multiples.values()) / len(multiples)
    
    if use_plotly:
        # Create colors list with target company highlighted
        colors = ['blue'] * len(multiples) + ['red']
        
        fig = go.Figure()
        
        # Add bar chart
        fig.add_trace(go.Bar(
            x=companies,
            y=values,
            marker_color=colors
        ))
        
        # Add average line
        fig.add_shape(
            type="line",
            x0=-0.5,
            y0=avg_multiple,
            x1=len(companies) - 0.5,
            y1=avg_multiple,
            line=dict(
                color="green",
                width=2,
                dash="dash",
            )
        )
        
        # Add annotation for average
        fig.add_annotation(
            x=len(companies) - 1,
            y=avg_multiple,
            text=f"Average: {avg_multiple:.2f}",
            showarrow=False,
            yshift=10
        )
        
        fig.update_layout(
            title=title,
            xaxis_title="Company",
            yaxis_title="Multiple",
            showlegend=False
        )
        
        return fig
    else:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create colors list with target company highlighted
        colors = ['blue'] * len(multiples) + ['red']
        
        # Create bar chart
        ax.bar(companies, values, color=colors)
        
        # Add average line
        ax.axhline(y=avg_multiple, color='green', linestyle='--', label=f'Average: {avg_multiple:.2f}')
        
        # Add labels and title
        ax.set_xlabel('Company')
        ax.set_ylabel('Multiple')
        ax.set_title(title)
        ax.legend()
        
        # Rotate x tick labels
        plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        return ax


def create_cash_flow_chart(
    years: List[int],
    cash_flows: List[float],
    present_values: List[float],
    title: str = "Cash Flows and Present Values",
    use_plotly: bool = True
):
    """
    Create a chart showing cash flows and their present values.
    
    Args:
        years: List of years
        cash_flows: List of cash flow values
        present_values: List of present value of cash flows
        title: Chart title
        use_plotly: If True, use Plotly for interactive charts, else use Matplotlib
        
    Returns:
        Plotly figure or Matplotlib axis object
    """
    if use_plotly:
        fig = go.Figure()
        
        # Add cash flows
        fig.add_trace(go.Bar(
            x=years,
            y=cash_flows,
            name='Cash Flow',
            marker_color='blue'
        ))
        
        # Add present values
        fig.add_trace(go.Bar(
            x=years,
            y=present_values,
            name='Present Value',
            marker_color='green'
        ))
        
        # Update layout
        fig.update_layout(
            title=title,
            xaxis_title='Year',
            yaxis_title='Value',
            barmode='group',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        return fig
    else:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Set width for bars
        width = 0.35
        x = np.arange(len(years))
        
        # Plot cash flows and present values
        ax.bar(x - width/2, cash_flows, width, label='Cash Flow', color='blue')
        ax.bar(x + width/2, present_values, width, label='Present Value', color='green')
        
        # Add labels and title
        ax.set_xlabel('Year')
        ax.set_ylabel('Value')
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(years)
        ax.legend()
        
        plt.tight_layout()
        return ax
