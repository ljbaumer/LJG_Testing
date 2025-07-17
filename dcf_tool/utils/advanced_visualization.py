import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Optional, Tuple, Union, Any


def create_football_field_chart(
    valuation_methods: Dict[str, Tuple[float, float]],
    current_value: Optional[float] = None,
    title: str = "Valuation Range by Method",
    currency: str = "$",
    use_plotly: bool = True
):
    """
    Create a football field chart showing valuation ranges across different methods.
    
    Args:
        valuation_methods: Dictionary mapping method names to (min, max) value tuples
        current_value: Current value to highlight (e.g., current share price)
        title: Chart title
        currency: Currency symbol to use
        use_plotly: If True, use Plotly for interactive charts
        
    Returns:
        Plotly figure
    """
    # Sort methods by midpoint value
    methods = []
    for method, (min_val, max_val) in valuation_methods.items():
        midpoint = (min_val + max_val) / 2
        methods.append((method, min_val, max_val, midpoint))
    
    methods.sort(key=lambda x: x[3])
    
    # Extract data for plotting
    method_names = [m[0] for m in methods]
    min_values = [m[1] for m in methods]
    max_values = [m[2] for m in methods]
    
    # Calculate overall min and max for axis limits
    overall_min = min(min_values) * 0.9
    overall_max = max(max_values) * 1.1
    
    if use_plotly:
        fig = go.Figure()
        
        # Add range bars for each method
        for i, (method, min_val, max_val, _) in enumerate(methods):
            fig.add_trace(go.Bar(
                x=[max_val - min_val],
                y=[method],
                orientation='h',
                base=min_val,
                marker_color='lightblue',
                name=method,
                hovertemplate=f"{method}: {currency}{min_val:,.2f} - {currency}{max_val:,.2f}<extra></extra>"
            ))
            
            # Add min and max labels
            fig.add_annotation(
                x=min_val,
                y=method,
                text=f"{currency}{min_val:,.2f}",
                showarrow=False,
                xanchor="right",
                yanchor="middle",
                xshift=-5
            )
            
            fig.add_annotation(
                x=max_val,
                y=method,
                text=f"{currency}{max_val:,.2f}",
                showarrow=False,
                xanchor="left",
                yanchor="middle",
                xshift=5
            )
        
        # Add current value line if provided
        if current_value is not None:
            fig.add_shape(
                type="line",
                x0=current_value,
                y0=-0.5,
                x1=current_value,
                y1=len(methods) - 0.5,
                line=dict(
                    color="red",
                    width=2,
                    dash="dash",
                )
            )
            
            fig.add_annotation(
                x=current_value,
                y=len(methods) - 0.5,
                text=f"Current: {currency}{current_value:,.2f}",
                showarrow=False,
                xanchor="center",
                yanchor="bottom",
                yshift=10
            )
        
        # Update layout
        fig.update_layout(
            title=title,
            xaxis_title="Value",
            yaxis=dict(
                title="",
                categoryorder="array",
                categoryarray=method_names
            ),
            showlegend=False,
            barmode='overlay',
            xaxis=dict(range=[overall_min, overall_max])
        )
        
        return fig


def create_tornado_chart(
    parameters: Dict[str, Tuple[float, float, float]],
    base_value: float,
    values: Dict[str, Tuple[float, float]],
    title: str = "Sensitivity Analysis",
    currency: str = "$",
    use_plotly: bool = True
):
    """
    Create a tornado chart for sensitivity analysis.
    
    Args:
        parameters: Dictionary mapping parameter names to (low, base, high) values
        base_value: Base case valuation result
        values: Dictionary mapping parameter names to (low_result, high_result) values
        title: Chart title
        currency: Currency symbol to use
        use_plotly: If True, use Plotly for interactive charts
        
    Returns:
        Plotly figure
    """
    # Calculate changes from base value
    changes = []
    for param, (low_result, high_result) in values.items():
        low_change = low_result - base_value
        high_change = high_result - base_value
        
        # Ensure low_change is always less than high_change for consistent display
        if low_change > high_change:
            low_change, high_change = high_change, low_change
        
        # Calculate total impact (absolute value of the larger change)
        impact = max(abs(low_change), abs(high_change))
        
        changes.append((param, low_change, high_change, impact))
    
    # Sort by impact (largest first)
    changes.sort(key=lambda x: x[3], reverse=True)
    
    # Extract data for plotting
    param_names = [c[0] for c in changes]
    low_changes = [c[1] for c in changes]
    high_changes = [c[2] for c in changes]
    
    # Create parameter labels with values
    param_labels = []
    for param in param_names:
        low, base, high = parameters[param]
        if isinstance(low, float) and abs(low) < 1:
            # Format as percentage if small decimal
            param_labels.append(f"{param} ({low:.1%}, {base:.1%}, {high:.1%})")
        else:
            # Format as regular number
            param_labels.append(f"{param} ({low}, {base}, {high})")
    
    if use_plotly:
        fig = go.Figure()
        
        # Add bars for low changes
        fig.add_trace(go.Bar(
            y=param_labels,
            x=low_changes,
            orientation='h',
            name='Downside',
            marker_color='red',
            hovertemplate="%{y}: %{x:,.2f}<extra></extra>"
        ))
        
        # Add bars for high changes
        fig.add_trace(go.Bar(
            y=param_labels,
            x=high_changes,
            orientation='h',
            name='Upside',
            marker_color='green',
            hovertemplate="%{y}: %{x:,.2f}<extra></extra>"
        ))
        
        # Add zero line
        fig.add_shape(
            type="line",
            x0=0,
            y0=-0.5,
            x1=0,
            y1=len(param_names) - 0.5,
            line=dict(
                color="black",
                width=1
            )
        )
        
        # Update layout
        fig.update_layout(
            title=f"{title}<br><sub>Base Value: {currency}{base_value:,.2f}</sub>",
            xaxis_title="Change in Value",
            yaxis=dict(
                title="",
                categoryorder="array",
                categoryarray=param_labels
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            barmode='overlay'
        )
        
        return fig


def create_scenario_comparison_chart(
    scenarios: List[str],
    values: Dict[str, List[float]],
    title: str = "Scenario Comparison",
    currency: str = "$",
    use_plotly: bool = True
):
    """
    Create a chart comparing key metrics across different scenarios.
    
    Args:
        scenarios: List of scenario names
        values: Dictionary mapping metric names to lists of values (one per scenario)
        title: Chart title
        currency: Currency symbol to use
        use_plotly: If True, use Plotly for interactive charts
        
    Returns:
        Plotly figure
    """
    if use_plotly:
        fig = go.Figure()
        
        # Add a trace for each metric
        for metric, metric_values in values.items():
            fig.add_trace(go.Bar(
                x=scenarios,
                y=metric_values,
                name=metric,
                hovertemplate="%{x}: %{y:,.2f}<extra></extra>"
            ))
        
        # Update layout
        fig.update_layout(
            title=title,
            xaxis_title="Scenario",
            yaxis_title="Value",
            legend_title="Metric",
            barmode='group'
        )
        
        return fig


def create_monte_carlo_histogram(
    values: List[float],
    base_value: float,
    title: str = "Monte Carlo Simulation Results",
    currency: str = "$",
    use_plotly: bool = True
):
    """
    Create a histogram of Monte Carlo simulation results.
    
    Args:
        values: List of simulation result values
        base_value: Base case valuation result
        title: Chart title
        currency: Currency symbol to use
        use_plotly: If True, use Plotly for interactive charts
        
    Returns:
        Plotly figure
    """
    if use_plotly:
        fig = go.Figure()
        
        # Add histogram
        fig.add_trace(go.Histogram(
            x=values,
            nbinsx=30,
            marker_color='lightblue',
            hovertemplate="Value: %{x:,.2f}<br>Count: %{y}<extra></extra>"
        ))
        
        # Add base value line
        fig.add_shape(
            type="line",
            x0=base_value,
            y0=0,
            x1=base_value,
            y1=1,
            yref="paper",
            line=dict(
                color="red",
                width=2,
                dash="dash",
            )
        )
        
        fig.add_annotation(
            x=base_value,
            y=1,
            yref="paper",
            text=f"Base: {currency}{base_value:,.2f}",
            showarrow=False,
            xanchor="center",
            yanchor="bottom",
            yshift=10
        )
        
        # Calculate percentiles
        p10 = np.percentile(values, 10)
        p50 = np.percentile(values, 50)
        p90 = np.percentile(values, 90)
        
        # Add percentile lines
        for percentile, value, color in [
            (10, p10, "orange"),
            (50, p50, "green"),
            (90, p90, "orange")
        ]:
            fig.add_shape(
                type="line",
                x0=value,
                y0=0,
                x1=value,
                y1=0.9,
                yref="paper",
                line=dict(
                    color=color,
                    width=1.5,
                    dash="dot",
                )
            )
            
            fig.add_annotation(
                x=value,
                y=0.9,
                yref="paper",
                text=f"P{percentile}: {currency}{value:,.2f}",
                showarrow=False,
                xanchor="center",
                yanchor="bottom"
            )
        
        # Update layout
        fig.update_layout(
            title=title,
            xaxis_title="Value",
            yaxis_title="Frequency",
            showlegend=False
        )
        
        return fig
