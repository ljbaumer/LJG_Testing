"""
Visualizations for Energy Source Comparison

This module contains functions for creating various visualizations to compare
energy sources for AI data centers.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

def plot_bar_chart(df, metric, title=None, color_map=None, sort=True):
    """
    Create a bar chart comparing energy sources based on a specific metric.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing energy source data
    metric : str
        The metric to plot
    title : str, optional
        Chart title
    color_map : dict, optional
        Dictionary mapping energy sources to colors
    sort : bool, optional
        Whether to sort the bars by value
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The generated figure
    """
    plt.figure(figsize=(12, 6))
    
    # Create a copy of the dataframe to avoid modifying the original
    plot_df = df.copy()
    
    # Sort if requested
    if sort:
        plot_df = plot_df.sort_values(by=metric)
    
    # Create the bar chart
    ax = sns.barplot(x=metric, y='Energy_Source', data=plot_df, palette=color_map)
    
    # Set title and labels
    if title:
        plt.title(title, fontsize=16)
    else:
        plt.title(f'Comparison of {metric} Across Energy Sources', fontsize=16)
    
    plt.xlabel(metric, fontsize=14)
    plt.ylabel('Energy Source', fontsize=14)
    plt.tight_layout()
    
    return plt.gcf()

def plot_radar_chart(df, energy_sources, metrics, title=None):
    """
    Create a radar chart comparing multiple energy sources across multiple metrics.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing energy source data
    energy_sources : list
        List of energy sources to include
    metrics : list
        List of metrics to include
    title : str, optional
        Chart title
    
    Returns:
    --------
    fig : plotly.graph_objects.Figure
        The generated figure
    """
    # Filter the dataframe
    filtered_df = df[df['Energy_Source'].isin(energy_sources)]
    
    # Create the radar chart
    fig = go.Figure()
    
    # Normalize the metrics for better visualization
    normalized_df = filtered_df.copy()
    for metric in metrics:
        max_val = filtered_df[metric].max()
        min_val = filtered_df[metric].min()
        if max_val > min_val:
            normalized_df[f'{metric}_normalized'] = (filtered_df[metric] - min_val) / (max_val - min_val)
        else:
            normalized_df[f'{metric}_normalized'] = 1
    
    # Add traces for each energy source
    for i, source in enumerate(energy_sources):
        source_df = normalized_df[normalized_df['Energy_Source'] == source]
        
        values = [source_df[f'{metric}_normalized'].values[0] for metric in metrics]
        values.append(values[0])  # Close the loop
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=metrics + [metrics[0]],  # Close the loop
            fill='toself',
            name=source
        ))
    
    # Update layout
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        showlegend=True
    )
    
    if title:
        fig.update_layout(title=title)
    else:
        fig.update_layout(title='Multi-dimensional Comparison of Energy Sources')
    
    return fig

def plot_heatmap(df, metrics, title=None):
    """
    Create a heatmap comparing all energy sources across multiple metrics.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing energy source data
    metrics : list
        List of metrics to include
    title : str, optional
        Chart title
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The generated figure
    """
    # Create a pivot table for the heatmap
    heatmap_df = df.copy()
    
    # Normalize the metrics for better visualization
    for metric in metrics:
        max_val = heatmap_df[metric].max()
        min_val = heatmap_df[metric].min()
        if max_val > min_val:
            heatmap_df[metric] = (heatmap_df[metric] - min_val) / (max_val - min_val)
    
    # Create the heatmap
    plt.figure(figsize=(14, 10))
    heatmap_data = heatmap_df.set_index('Energy_Source')[metrics]
    ax = sns.heatmap(heatmap_data, annot=True, cmap='viridis', linewidths=.5, fmt='.2f')
    
    # Set title and labels
    if title:
        plt.title(title, fontsize=16)
    else:
        plt.title('Heatmap of Normalized Metrics Across Energy Sources', fontsize=16)
    
    plt.ylabel('Energy Source', fontsize=14)
    plt.tight_layout()
    
    return plt.gcf()

def plot_scatter(df, x_metric, y_metric, size_metric=None, color_metric=None, title=None):
    """
    Create a scatter plot comparing energy sources based on two metrics.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing energy source data
    x_metric : str
        The metric to plot on the x-axis
    y_metric : str
        The metric to plot on the y-axis
    size_metric : str, optional
        The metric to determine point size
    color_metric : str, optional
        The metric to determine point color
    title : str, optional
        Chart title
    
    Returns:
    --------
    fig : plotly.graph_objects.Figure
        The generated figure
    """
    # Create the scatter plot
    if size_metric and color_metric:
        fig = px.scatter(
            df, x=x_metric, y=y_metric, 
            size=size_metric, color=color_metric,
            hover_name='Energy_Source', 
            size_max=60
        )
    elif size_metric:
        fig = px.scatter(
            df, x=x_metric, y=y_metric, 
            size=size_metric, 
            hover_name='Energy_Source', 
            size_max=60
        )
    elif color_metric:
        fig = px.scatter(
            df, x=x_metric, y=y_metric, 
            color=color_metric,
            hover_name='Energy_Source'
        )
    else:
        fig = px.scatter(
            df, x=x_metric, y=y_metric, 
            hover_name='Energy_Source'
        )
    
    # Update layout
    if title:
        fig.update_layout(title=title)
    else:
        fig.update_layout(
            title=f'Relationship Between {x_metric} and {y_metric}',
            xaxis_title=x_metric,
            yaxis_title=y_metric
        )
    
    return fig

def plot_parallel_coordinates(df, metrics, title=None):
    """
    Create a parallel coordinates plot comparing energy sources across multiple metrics.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing energy source data
    metrics : list
        List of metrics to include
    title : str, optional
        Chart title
    
    Returns:
    --------
    fig : plotly.graph_objects.Figure
        The generated figure
    """
    # Create a copy of the dataframe
    plot_df = df.copy()
    
    # Add a numeric index for coloring
    plot_df['source_index'] = range(len(plot_df))
    
    # Normalize the metrics for better visualization
    for metric in metrics:
        max_val = plot_df[metric].max()
        min_val = plot_df[metric].min()
        if max_val > min_val:
            plot_df[metric] = (plot_df[metric] - min_val) / (max_val - min_val)
    
    # Create the parallel coordinates plot
    fig = px.parallel_coordinates(
        plot_df,
        dimensions=metrics,
        color='source_index',
        color_continuous_scale=px.colors.qualitative.Plotly,
        labels={col: col.replace('_', ' ') for col in metrics}
    )
    
    # Add custom hover text with energy source names
    fig.update_traces(
        line_colorbar_title='Energy Source',
        line_colorbar_tickvals=plot_df['source_index'],
        line_colorbar_ticktext=plot_df['Energy_Source']
    )
    
    # Update layout
    if title:
        fig.update_layout(title=title)
    else:
        fig.update_layout(title='Parallel Coordinates Plot of Energy Sources')
    
    return fig

def create_metric_explanation():
    """
    Create a dictionary with explanations for each metric.
    
    Returns:
    --------
    dict
        Dictionary with metric explanations
    """
    explanations = {
        'LCOE_Avg': """
        **Levelized Cost of Electricity (LCOE)** represents the average cost per unit of electricity 
        generated over the lifetime of the power plant, including capital costs, fuel costs, 
        operation and maintenance, and financing costs. Lower values are better.
        
        Unit: $/MWh
        """,
        
        'Power_Density': """
        **Power Density** measures how much power can be generated per unit of land area. 
        Higher power density means less land is required for the same power output, 
        which is particularly important for data centers in areas with limited space.
        
        Unit: MW/km²
        """,
        
        'Carbon_Intensity': """
        **Carbon Intensity** represents the lifecycle greenhouse gas emissions per unit of 
        electricity generated. This includes emissions from construction, operation, 
        fuel production, and decommissioning. Lower values indicate more climate-friendly options.
        
        Unit: gCO2eq/kWh
        """,
        
        'Capacity_Factor': """
        **Capacity Factor** is the ratio of actual energy output to the maximum possible 
        output over a period of time. Higher capacity factors indicate more reliable 
        and consistent power generation, which is crucial for data centers.
        
        Unit: %
        """,
        
        'Construction_Time': """
        **Construction Time** is the typical time required to build and commission a new 
        power plant. Shorter construction times allow for faster scaling of power capacity 
        to meet growing AI computing demands.
        
        Unit: Years
        """,
        
        'Operational_Lifespan': """
        **Operational Lifespan** is the expected duration a power plant can operate 
        before requiring major refurbishment or decommissioning. Longer lifespans 
        provide better long-term value and stability.
        
        Unit: Years
        """,
        
        'Water_Usage': """
        **Water Usage** measures the amount of water consumed per unit of electricity 
        generated. Lower water usage is preferable, especially in water-stressed regions 
        and for data centers that already have significant cooling water requirements.
        
        Unit: Gallons/MWh
        """,
        
        'Land_Use': """
        **Land Use** represents the land area required per unit of energy produced. 
        Lower land use is generally preferable, especially in areas where land is scarce or expensive.
        
        Unit: m²/MWh
        """,
        
        'Grid_Reliability': """
        **Grid Reliability** is a score representing how reliable and dispatchable the 
        energy source is for grid operations. Higher scores indicate power sources that 
        can be counted on to deliver electricity when needed, which is critical for data centers.
        
        Unit: Scale (1-10)
        """,
        
        'Scalability': """
        **Scalability** is a score representing how easily the energy source can be scaled 
        to meet the growing power demands of AI data centers. Higher scores indicate 
        better suitability for rapid expansion of computing capacity.
        
        Unit: Scale (1-10)
        """
    }
    
    return explanations

def create_energy_source_explanation():
    """
    Create a dictionary with explanations for each energy source.
    
    Returns:
    --------
    dict
        Dictionary with energy source explanations
    """
    explanations = {
        'Natural Gas (Combined Cycle)': """
        **Natural Gas (Combined Cycle)** power plants burn natural gas to generate electricity 
        using a gas turbine, then use waste heat to power a steam turbine for additional electricity. 
        They offer high efficiency, relatively low carbon emissions compared to coal, 
        and excellent reliability and dispatchability.
        """,
        
        'Nuclear (Fission)': """
        **Nuclear Fission** generates electricity by splitting uranium atoms to release energy 
        as heat, which is used to produce steam and drive turbines. Nuclear power offers 
        very high reliability, minimal carbon emissions, and exceptional power density, 
        but has higher upfront costs and longer construction times.
        """,
        
        'Solar PV (Utility Scale)': """
        **Solar Photovoltaic (PV)** systems convert sunlight directly into electricity using 
        semiconductor materials. Utility-scale solar farms offer zero-emission operation, 
        rapidly declining costs, and short construction times, but are limited by 
        intermittency and lower power density.
        """,
        
        'Wind (Onshore)': """
        **Onshore Wind** turbines convert wind energy into electricity. They offer low operating 
        costs, zero-emission generation, and relatively quick deployment, but are limited by 
        intermittency, low power density, and geographic constraints.
        """,
        
        'Wind (Offshore)': """
        **Offshore Wind** turbines are installed in bodies of water, typically the ocean. 
        They benefit from stronger and more consistent winds than onshore installations, 
        resulting in higher capacity factors, but have higher construction and maintenance costs.
        """,
        
        'Hydroelectric': """
        **Hydroelectric** power generates electricity by harnessing the energy of flowing water. 
        It offers reliable, dispatchable power with minimal emissions during operation, 
        but requires specific geographic features and can have significant environmental impacts.
        """,
        
        'Coal': """
        **Coal** power plants burn coal to produce steam that drives turbines. While offering 
        reliable and dispatchable power with high power density, coal has the highest carbon 
        emissions of all major energy sources and faces increasing regulatory and social pressure.
        """,
        
        'Geothermal': """
        **Geothermal** power taps into underground heat to generate electricity. It provides 
        highly reliable, low-emission baseload power, but is geographically limited to areas 
        with accessible geothermal resources.
        """,
        
        'Biomass': """
        **Biomass** generates electricity by burning organic materials like wood, agricultural 
        residues, or dedicated energy crops. It offers dispatchable power with potential carbon 
        neutrality if sustainably managed, but has lower efficiency and potential land use conflicts.
        """,
        
        'Hydrogen Fuel Cells': """
        **Hydrogen Fuel Cells** generate electricity through an electrochemical reaction between 
        hydrogen and oxygen, producing only water as a byproduct. They offer zero-emission 
        operation at the point of use, high efficiency, and good scalability, but face challenges 
        in hydrogen production, storage, and distribution.
        """,
        
        'Small Modular Reactors (SMRs)': """
        **Small Modular Reactors (SMRs)** are advanced nuclear reactors with a power capacity 
        under 300 MWe, designed for factory fabrication and modular construction. They promise 
        enhanced safety, shorter construction times, and better scalability than traditional 
        nuclear plants, though most designs are still in development or early deployment.
        """
    }
    
    return explanations
