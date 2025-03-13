import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

def create_cost_breakdown_chart(cost_breakdown):
    """
    Create a pie chart showing the breakdown of life cycle costs by category
    
    Parameters:
    -----------
    cost_breakdown : dict
        Dictionary with cost categories as keys and costs as values
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Plotly figure object containing the pie chart
    """
    # Convert dictionary to lists for plotting
    labels = list(cost_breakdown.keys())
    values = list(cost_breakdown.values())
    
    # Calculate percentages for hover text
    total = sum(values)
    percentages = [f"{(value/total)*100:.1f}%" for value in values]
    
    # Create hover text
    hover_text = [f"{label}<br>${value:,.0f}<br>{percentage}" 
                 for label, value, percentage in zip(labels, values, percentages)]
    
    # Create pie chart
    fig = px.pie(
        names=labels,
        values=values,
        title="Life Cycle Cost Breakdown",
        hover_data=[values],
        labels={'value': 'Cost ($)'},
        color_discrete_sequence=px.colors.qualitative.Bold,
    )
    
    # Update hover template
    fig.update_traces(
        hovertemplate='%{label}<br>$%{value:,.0f}<br>%{percent}',
        textinfo='label+percent',
        insidetextorientation='radial'
    )
    
    # Update layout
    fig.update_layout(
        legend_title="Cost Categories",
        margin=dict(t=50, b=0, l=0, r=0),
    )
    
    return fig

def create_costs_over_time_chart(annual_costs, analysis_period):
    """
    Create a line chart showing costs over time
    
    Parameters:
    -----------
    annual_costs : dict
        Dictionary with cost categories and annual values
    analysis_period : int
        Number of years in the analysis period
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Plotly figure object containing the line chart
    """
    # Create DataFrame from annual costs
    df = pd.DataFrame(annual_costs)
    
    # Create line chart
    fig = go.Figure()
    
    # Add traces for each cost category
    fig.add_trace(go.Scatter(
        x=df['year'],
        y=df['Operating Costs'],
        mode='lines',
        name='Operating Costs',
        line=dict(width=3, color='#1f77b4'),
        stackgroup='one'
    ))
    
    fig.add_trace(go.Scatter(
        x=df['year'],
        y=df['Maintenance Costs'],
        mode='lines',
        name='Maintenance Costs',
        line=dict(width=3, color='#ff7f0e'),
        stackgroup='one'
    ))
    
    fig.add_trace(go.Scatter(
        x=df['year'],
        y=df['Replacement Costs'],
        mode='lines',
        name='Replacement Costs',
        line=dict(width=3, color='#2ca02c'),
        stackgroup='one'
    ))
    
    # Add a trace for total annual costs
    total_annual_costs = df['Operating Costs'] + df['Maintenance Costs'] + df['Replacement Costs']
    fig.add_trace(go.Scatter(
        x=df['year'],
        y=total_annual_costs,
        mode='lines',
        name='Total Annual Costs',
        line=dict(width=4, color='#d62728', dash='dash'),
        stackgroup=None
    ))
    
    # Update layout
    fig.update_layout(
        title="Annual Costs Over Time",
        xaxis_title="Year",
        yaxis_title="Cost ($)",
        legend_title="Cost Categories",
        hovermode="x unified",
        margin=dict(t=50, b=50, l=50, r=50),
    )
    
    # Update y-axis to show dollar format
    fig.update_yaxes(tickprefix="$", tickformat=",")
    
    # Update x-axis to show integers only
    fig.update_xaxes(tickmode='linear', tick0=1, dtick=5)
    
    return fig

def create_scenario_comparison_chart(scenarios):
    """
    Create a bar chart comparing different scenarios
    
    Parameters:
    -----------
    scenarios : list
        List of dictionaries containing scenario data
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Plotly figure object containing the bar chart
    """
    # Extract data from scenarios
    scenario_names = [s['name'] for s in scenarios]
    total_costs = [s['total_cost'] for s in scenarios]
    present_values = [s['present_value'] for s in scenarios]
    initial_costs = [s['initial_cost'] for s in scenarios]
    
    # Create bar chart
    fig = go.Figure()
    
    # Add bars for each cost type
    fig.add_trace(go.Bar(
        x=scenario_names,
        y=initial_costs,
        name='Initial Cost',
        marker_color='#1f77b4'
    ))
    
    fig.add_trace(go.Bar(
        x=scenario_names,
        y=[s['operating_cost'] for s in scenarios],
        name='Operating Costs',
        marker_color='#ff7f0e'
    ))
    
    fig.add_trace(go.Bar(
        x=scenario_names,
        y=[s['maintenance_cost'] for s in scenarios],
        name='Maintenance Costs',
        marker_color='#2ca02c'
    ))
    
    fig.add_trace(go.Bar(
        x=scenario_names,
        y=[s['replacement_cost'] for s in scenarios],
        name='Replacement Costs',
        marker_color='#d62728'
    ))
    
    # Add a line for present value
    fig.add_trace(go.Scatter(
        x=scenario_names,
        y=present_values,
        mode='markers+lines',
        name='Present Value',
        marker=dict(size=12, symbol='diamond', color='#9467bd'),
        line=dict(width=3, dash='dot')
    ))
    
    # Update layout
    fig.update_layout(
        title="Scenario Comparison",
        xaxis_title="Scenario",
        yaxis_title="Cost ($)",
        legend_title="Cost Categories",
        barmode='stack',
        margin=dict(t=50, b=50, l=50, r=50),
    )
    
    # Update y-axis to show dollar format
    fig.update_yaxes(tickprefix="$", tickformat=",")
    
    # Add annotations for total costs
    for i, (name, cost) in enumerate(zip(scenario_names, total_costs)):
        fig.add_annotation(
            x=name,
            y=cost,
            text=f"${cost:,.0f}",
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor="#636363",
            ax=0,
            ay=-40,
            font=dict(size=12, color="black"),
            bgcolor="white",
            bordercolor="#636363",
            borderwidth=1,
            borderpad=4,
            opacity=0.8
        )
    
    return fig

def create_sensitivity_analysis_chart(base_value, parameter_name, variations, results):
    """
    Create a tornado chart for sensitivity analysis
    
    Parameters:
    -----------
    base_value : float
        Base case value of the output metric
    parameter_name : str
        Name of the parameter being varied
    variations : list
        List of parameter values
    results : list
        List of corresponding output values
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Plotly figure object containing the tornado chart
    """
    # Calculate percentage changes from base case
    percent_changes = [(result - base_value) / base_value * 100 for result in results]
    
    # Create labels
    labels = [f"{parameter_name}: {variation}" for variation in variations]
    
    # Sort by absolute percentage change
    sorted_indices = sorted(range(len(percent_changes)), key=lambda i: abs(percent_changes[i]), reverse=True)
    sorted_labels = [labels[i] for i in sorted_indices]
    sorted_changes = [percent_changes[i] for i in sorted_indices]
    
    # Create colors based on positive/negative change
    colors = ['#d62728' if change >= 0 else '#1f77b4' for change in sorted_changes]
    
    # Create horizontal bar chart
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=sorted_labels,
        x=sorted_changes,
        orientation='h',
        marker_color=colors,
        text=[f"{change:+.1f}%" for change in sorted_changes],
        textposition='outside',
    ))
    
    # Update layout
    fig.update_layout(
        title=f"Sensitivity Analysis: Impact on Total Life Cycle Cost",
        xaxis_title="Percentage Change in Total Life Cycle Cost (%)",
        yaxis_title="Parameter Variation",
        margin=dict(t=50, b=50, l=200, r=50),
    )
    
    # Add a vertical line at 0%
    fig.add_shape(
        type="line",
        x0=0, y0=-0.5,
        x1=0, y1=len(sorted_labels) - 0.5,
        line=dict(color="black", width=2, dash="dash"),
    )
    
    return fig
