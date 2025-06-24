"""
Energy Source Comparison for AI Data Centers

A Streamlit application for comparing different energy sources
for powering advanced AI data centers.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

# Import our modules
from energy_data import get_energy_data, get_data_sources, get_ai_datacenter_context
from visualizations import (
    plot_bar_chart, plot_radar_chart, plot_heatmap, plot_scatter,
    plot_parallel_coordinates, create_metric_explanation, create_energy_source_explanation
)

# Set page configuration
st.set_page_config(
    page_title="Energy Sources for AI Data Centers",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E88E5;
        margin-bottom: 1rem;
    }
    .section-header {
        font-size: 1.8rem;
        font-weight: 600;
        color: #0D47A1;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .subsection-header {
        font-size: 1.4rem;
        font-weight: 500;
        color: #1565C0;
        margin-top: 1.5rem;
        margin-bottom: 0.8rem;
    }
    .metric-name {
        font-weight: 600;
        color: #1976D2;
    }
    .source-name {
        font-weight: 600;
        color: #1976D2;
    }
    .info-box {
        background-color: #E3F2FD;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .warning-box {
        background-color: #FFF8E1;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    return get_energy_data()

df = load_data()
data_sources = get_data_sources()
ai_context = get_ai_datacenter_context()
metric_explanations = create_metric_explanation()
energy_source_explanations = create_energy_source_explanation()

# Define available metrics for visualization
available_metrics = [
    'LCOE_Avg', 'Power_Density', 'Carbon_Intensity', 'Capacity_Factor',
    'Construction_Time', 'Operational_Lifespan', 'Water_Usage', 'Land_Use',
    'Grid_Reliability', 'Scalability'
]

# Define metric display names
metric_display_names = {
    'LCOE_Avg': 'Levelized Cost of Electricity ($/MWh)',
    'Power_Density': 'Power Density (MW/km²)',
    'Carbon_Intensity': 'Carbon Intensity (gCO2eq/kWh)',
    'Capacity_Factor': 'Capacity Factor (%)',
    'Construction_Time': 'Construction Time (Years)',
    'Operational_Lifespan': 'Operational Lifespan (Years)',
    'Water_Usage': 'Water Usage (Gallons/MWh)',
    'Land_Use': 'Land Use (m²/MWh)',
    'Grid_Reliability': 'Grid Reliability (1-10)',
    'Scalability': 'Scalability for Data Centers (1-10)'
}

# Define metrics where lower values are better
lower_is_better = ['LCOE_Avg', 'Carbon_Intensity', 'Construction_Time', 'Water_Usage', 'Land_Use']

# Main application
def main():
    # Header
    st.markdown('<div class="main-header">Energy Sources for AI Data Centers</div>', unsafe_allow_html=True)
    st.markdown("""
    This application provides a comprehensive comparison of different energy sources for powering advanced AI data centers.
    Explore the visualizations to understand the relative merits of various power sources based on key metrics.
    """)
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select a page:",
        ["Introduction", "Single Metric Comparison", "Multi-Metric Comparison", 
         "Energy Source Deep Dive", "AI Data Center Context", "Data Sources"]
    )
    
    # Display the selected page
    if page == "Introduction":
        show_introduction()
    elif page == "Single Metric Comparison":
        show_single_metric_comparison()
    elif page == "Multi-Metric Comparison":
        show_multi_metric_comparison()
    elif page == "Energy Source Deep Dive":
        show_energy_source_deep_dive()
    elif page == "AI Data Center Context":
        show_ai_datacenter_context()
    elif page == "Data Sources":
        show_data_sources()

def show_introduction():
    st.markdown('<div class="section-header">Introduction</div>', unsafe_allow_html=True)
    
    st.markdown("""
    As artificial intelligence continues to advance, the energy requirements for AI data centers are growing exponentially.
    Choosing the right energy sources for these facilities is crucial for sustainability, reliability, and cost-effectiveness.
    
    This tool allows you to compare different energy sources across multiple metrics relevant to AI data center operations.
    """)
    
    st.markdown('<div class="subsection-header">Key Energy Sources</div>', unsafe_allow_html=True)
    
    # Display energy sources in columns
    cols = st.columns(3)
    energy_sources = list(energy_source_explanations.keys())
    
    for i, source in enumerate(energy_sources):
        col_idx = i % 3
        with cols[col_idx]:
            st.markdown(f'<div class="source-name">{source}</div>', unsafe_allow_html=True)
            st.markdown(energy_source_explanations[source])
    
    st.markdown('<div class="subsection-header">Key Comparison Metrics</div>', unsafe_allow_html=True)
    
    # Display metrics in columns
    cols = st.columns(2)
    metrics = list(metric_explanations.keys())
    
    for i, metric in enumerate(metrics):
        col_idx = i % 2
        with cols[col_idx]:
            st.markdown(f'<div class="metric-name">{metric_display_names[metric]}</div>', unsafe_allow_html=True)
            st.markdown(metric_explanations[metric])
    
    # Overview heatmap
    st.markdown('<div class="subsection-header">Overview: Normalized Comparison Across All Metrics</div>', unsafe_allow_html=True)
    
    st.markdown("""
    The heatmap below provides a normalized comparison of all energy sources across all metrics.
    Values are normalized to a 0-1 scale, where 1 (darker color) represents the best performance for that metric.
    
    Note: For metrics where lower values are better (like LCOE, Carbon Intensity, etc.), the normalization is inverted
    so that lower original values result in higher normalized values (darker colors).
    """)
    
    # Create a copy of the dataframe for the heatmap
    heatmap_df = df.copy()
    
    # Normalize the metrics for the heatmap
    for metric in available_metrics:
        max_val = heatmap_df[metric].max()
        min_val = heatmap_df[metric].min()
        
        if max_val > min_val:
            if metric in lower_is_better:
                # For metrics where lower is better, invert the normalization
                heatmap_df[metric] = 1 - ((heatmap_df[metric] - min_val) / (max_val - min_val))
            else:
                # For metrics where higher is better
                heatmap_df[metric] = (heatmap_df[metric] - min_val) / (max_val - min_val)
    
    # Create the heatmap
    fig = plt.figure(figsize=(12, 8))
    heatmap_data = heatmap_df.set_index('Energy_Source')[available_metrics]
    
    # Rename columns for display
    heatmap_data.columns = [col.replace('_', ' ') for col in heatmap_data.columns]
    
    ax = plt.subplot(111)
    cax = ax.matshow(heatmap_data, cmap='viridis')
    fig.colorbar(cax)
    
    # Set tick labels
    ax.set_xticks(np.arange(len(heatmap_data.columns)))
    ax.set_yticks(np.arange(len(heatmap_data.index)))
    ax.set_xticklabels(heatmap_data.columns, rotation=45, ha='left')
    ax.set_yticklabels(heatmap_data.index)
    
    # Add text annotations
    for i in range(len(heatmap_data.index)):
        for j in range(len(heatmap_data.columns)):
            ax.text(j, i, f'{heatmap_data.iloc[i, j]:.2f}', ha='center', va='center', color='white')
    
    plt.tight_layout()
    st.pyplot(fig)
    
    st.markdown("""
    <div class="info-box">
    This overview provides a quick visual comparison across all metrics. Navigate to the other sections
    for more detailed comparisons and interactive visualizations.
    </div>
    """, unsafe_allow_html=True)

def show_single_metric_comparison():
    st.markdown('<div class="section-header">Single Metric Comparison</div>', unsafe_allow_html=True)
    
    st.markdown("""
    Compare energy sources based on a single metric of your choice.
    This view helps identify the best and worst performers for specific criteria.
    """)
    
    # Select metric
    selected_metric = st.selectbox(
        "Select a metric to compare:",
        available_metrics,
        format_func=lambda x: metric_display_names[x]
    )
    
    # Show explanation for the selected metric
    st.markdown(f'<div class="info-box">{metric_explanations[selected_metric]}</div>', unsafe_allow_html=True)
    
    # Note about interpretation
    if selected_metric in lower_is_better:
        st.markdown('<div class="warning-box">Note: For this metric, lower values are better.</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="warning-box">Note: For this metric, higher values are better.</div>', unsafe_allow_html=True)
    
    # Create bar chart
    st.markdown('<div class="subsection-header">Bar Chart Comparison</div>', unsafe_allow_html=True)
    
    # Sort in the appropriate direction
    sort_ascending = selected_metric in lower_is_better
    sorted_df = df.sort_values(by=selected_metric, ascending=sort_ascending)
    
    # Create the bar chart using Plotly for interactivity
    fig = px.bar(
        sorted_df,
        x=selected_metric,
        y='Energy_Source',
        orientation='h',
        title=f'Comparison of {metric_display_names[selected_metric]} Across Energy Sources',
        labels={selected_metric: metric_display_names[selected_metric], 'Energy_Source': 'Energy Source'},
        height=600
    )
    
    # Update layout
    fig.update_layout(
        xaxis_title=metric_display_names[selected_metric],
        yaxis_title='Energy Source',
        yaxis={'categoryorder': 'array', 'categoryarray': sorted_df['Energy_Source'].tolist()}
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Top and bottom performers
    st.markdown('<div class="subsection-header">Top and Bottom Performers</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Top Performers")
        top_df = sorted_df.head(3)
        for _, row in top_df.iterrows():
            st.markdown(f"**{row['Energy_Source']}**: {row[selected_metric]}")
    
    with col2:
        st.markdown("#### Bottom Performers")
        bottom_df = sorted_df.tail(3).iloc[::-1]  # Reverse to show worst first
        for _, row in bottom_df.iterrows():
            st.markdown(f"**{row['Energy_Source']}**: {row[selected_metric]}")
    
    # Implications for AI data centers
    st.markdown('<div class="subsection-header">Implications for AI Data Centers</div>', unsafe_allow_html=True)
    
    implications = {
        'LCOE_Avg': """
        Cost is a critical factor for AI data centers, which can consume enormous amounts of electricity.
        Lower LCOE directly translates to reduced operational expenses, potentially saving millions of dollars annually for large facilities.
        However, cost must be balanced with reliability and other factors, as even brief outages can be extremely costly.
        """,
        
        'Power_Density': """
        AI data centers often need to be located near population centers for latency reasons or in specific regions for regulatory compliance.
        Higher power density means more generation capacity can be placed in a smaller area, which is valuable in land-constrained regions.
        This is especially important as AI computing demands continue to grow exponentially.
        """,
        
        'Carbon_Intensity': """
        Many tech companies have made climate commitments that require reducing carbon emissions.
        Lower carbon intensity helps meet these goals and can provide regulatory advantages and improved public perception.
        As carbon pricing becomes more common globally, lower emissions can also translate to direct cost savings.
        """,
        
        'Capacity_Factor': """
        AI workloads, especially training runs, often require continuous, uninterrupted power for days or weeks.
        Energy sources with higher capacity factors provide more consistent generation, reducing the need for backup systems or grid balancing.
        This reliability is crucial for maintaining the 99.999% uptime typically required for critical AI infrastructure.
        """,
        
        'Construction_Time': """
        The rapid growth of AI computing demand requires energy infrastructure that can be deployed quickly.
        Shorter construction times allow power capacity to scale more closely with computing demand.
        This agility is valuable in a field where computing requirements can double in months rather than years.
        """,
        
        'Operational_Lifespan': """
        Longer operational lifespans provide more stable, predictable energy costs over time.
        This allows for better long-term planning and potentially lower lifetime costs for AI infrastructure.
        It also reduces the frequency of disruptive replacements or major refurbishments.
        """,
        
        'Water_Usage': """
        Data centers already use significant amounts of water for cooling systems.
        Energy sources with lower water usage reduce the total water footprint and make siting more flexible.
        This is particularly important in water-stressed regions where cooling and power generation compete for limited water resources.
        """,
        
        'Land_Use': """
        AI data centers often need to be located near population centers or network hubs, where land is expensive.
        Energy sources with lower land use requirements can be more easily co-located with or near data centers.
        This proximity can reduce transmission losses and infrastructure costs.
        """,
        
        'Grid_Reliability': """
        AI data centers cannot tolerate power interruptions, which can damage hardware and disrupt critical workloads.
        Energy sources with higher grid reliability scores provide more dependable power, reducing the need for expensive backup systems.
        This reliability directly impacts operational continuity and can prevent costly downtime.
        """,
        
        'Scalability': """
        As AI models grow larger and more complex, power requirements increase dramatically.
        Energy sources that can scale quickly and efficiently are better suited to meet this rapidly growing demand.
        Higher scalability scores indicate power sources that can grow alongside AI computing needs without becoming bottlenecks.
        """
    }
    
    st.markdown(implications[selected_metric])

def show_multi_metric_comparison():
    st.markdown('<div class="section-header">Multi-Metric Comparison</div>', unsafe_allow_html=True)
    
    st.markdown("""
    Compare energy sources across multiple metrics simultaneously.
    This view helps identify which sources offer the best overall performance for AI data center needs.
    """)
    
    # Tabs for different visualization types
    tab1, tab2, tab3 = st.tabs(["Radar Chart", "Parallel Coordinates", "Scatter Plot"])
    
    with tab1:
        st.markdown("### Radar Chart Comparison")
        st.markdown("""
        Radar charts allow for comparing multiple energy sources across multiple metrics simultaneously.
        Each axis represents a different metric, and each shape represents a different energy source.
        """)
        
        # Select energy sources
        selected_sources = st.multiselect(
            "Select energy sources to compare (2-5 recommended):",
            df['Energy_Source'].tolist(),
            default=df['Energy_Source'].tolist()[:3]
        )
        
        # Select metrics
        selected_metrics = st.multiselect(
            "Select metrics to compare:",
            available_metrics,
            default=available_metrics[:5],
            format_func=lambda x: metric_display_names[x]
        )
        
        if selected_sources and selected_metrics:
            # Normalize metrics for radar chart
            radar_df = df[df['Energy_Source'].isin(selected_sources)].copy()
            
            # For metrics where lower is better, invert the values for visualization
            for metric in selected_metrics:
                if metric in lower_is_better:
                    max_val = df[metric].max()
                    min_val = df[metric].min()
                    if max_val > min_val:
                        radar_df[metric] = max_val - radar_df[metric] + min_val
            
            # Create radar chart
            fig = plot_radar_chart(
                radar_df,
                selected_sources,
                selected_metrics,
                title="Multi-dimensional Comparison of Energy Sources"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("""
            <div class="info-box">
            Note: For metrics where lower values are better (like LCOE, Carbon Intensity, etc.), 
            the values have been inverted so that better performance always appears further from the center.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.warning("Please select at least one energy source and one metric.")
    
    with tab2:
        st.markdown("### Parallel Coordinates Plot")
        st.markdown("""
        Parallel coordinates plots show how energy sources perform across multiple metrics simultaneously.
        Each vertical axis represents a different metric, and each line represents a different energy source.
        """)
        
        # Select metrics
        pc_metrics = st.multiselect(
            "Select metrics to include:",
            available_metrics,
            default=available_metrics[:6],
            format_func=lambda x: metric_display_names[x],
            key="pc_metrics"
        )
        
        if pc_metrics:
            # Create a copy of the dataframe
            pc_df = df.copy()
            
            # Create a new list for the dimensions to use in the plot
            plot_dimensions = []
            
            # For metrics where lower is better, invert the values for visualization
            for metric in pc_metrics:
                if metric in lower_is_better:
                    max_val = pc_df[metric].max()
                    min_val = pc_df[metric].min()
                    if max_val > min_val:
                        # Create an inverted version of the metric
                        inverted_name = f"{metric}_inverted"
                        pc_df[inverted_name] = max_val - pc_df[metric] + min_val
                        plot_dimensions.append(inverted_name)
                    else:
                        plot_dimensions.append(metric)
                else:
                    plot_dimensions.append(metric)
            
            # Create parallel coordinates plot
            fig = plot_parallel_coordinates(
                pc_df,
                plot_dimensions,
                title="Parallel Coordinates Plot of Energy Sources"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("""
            <div class="info-box">
            Note: For metrics where lower values are better, the values have been inverted so that 
            higher values always represent better performance across all axes.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.warning("Please select at least one metric.")
    
    with tab3:
        st.markdown("### Scatter Plot Comparison")
        st.markdown("""
        Scatter plots allow for examining the relationship between two metrics.
        Each point represents a different energy source.
        """)
        
        # Select metrics for x and y axes
        col1, col2 = st.columns(2)
        
        with col1:
            x_metric = st.selectbox(
                "Select metric for X-axis:",
                available_metrics,
                index=0,
                format_func=lambda x: metric_display_names[x]
            )
        
        with col2:
            y_metric = st.selectbox(
                "Select metric for Y-axis:",
                available_metrics,
                index=1,
                format_func=lambda x: metric_display_names[x]
            )
        
        # Optional: Select metric for point size
        size_metric = st.selectbox(
            "Select metric for point size (optional):",
            ["None"] + available_metrics,
            format_func=lambda x: "None" if x == "None" else metric_display_names[x]
        )
        
        # Create scatter plot
        if x_metric and y_metric:
            if size_metric == "None":
                size_metric = None
            
            fig = plot_scatter(
                df,
                x_metric,
                y_metric,
                size_metric=size_metric,
                title=f"Relationship Between {metric_display_names[x_metric]} and {metric_display_names[y_metric]}"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Add interpretation notes
            x_better = "lower" if x_metric in lower_is_better else "higher"
            y_better = "lower" if y_metric in lower_is_better else "higher"
            
            st.markdown(f"""
            <div class="info-box">
            Interpretation: 
            - For {metric_display_names[x_metric]}, {x_better} values are better.
            - For {metric_display_names[y_metric]}, {y_better} values are better.
            </div>
            """, unsafe_allow_html=True)
            
            # Identify optimal sources
            st.markdown("### Optimal Energy Sources")
            
            if x_metric in lower_is_better and y_metric in lower_is_better:
                optimal_df = df.sort_values(by=[x_metric, y_metric]).head(3)
                st.markdown(f"Energy sources with both low {x_metric} and low {y_metric}:")
            elif x_metric in lower_is_better and y_metric not in lower_is_better:
                optimal_df = df.sort_values(by=[x_metric, y_metric], ascending=[True, False]).head(3)
                st.markdown(f"Energy sources with low {x_metric} and high {y_metric}:")
            elif x_metric not in lower_is_better and y_metric in lower_is_better:
                optimal_df = df.sort_values(by=[x_metric, y_metric], ascending=[False, True]).head(3)
                st.markdown(f"Energy sources with high {x_metric} and low {y_metric}:")
            else:
                optimal_df = df.sort_values(by=[x_metric, y_metric], ascending=[False, False]).head(3)
                st.markdown(f"Energy sources with both high {x_metric} and high {y_metric}:")
            
            for _, row in optimal_df.iterrows():
                st.markdown(f"**{row['Energy_Source']}**: {row[x_metric]} ({x_metric}), {row[y_metric]} ({y_metric})")
        else:
            st.warning("Please select metrics for both axes.")

def show_energy_source_deep_dive():
    st.markdown('<div class="section-header">Energy Source Deep Dive</div>', unsafe_allow_html=True)
    
    st.markdown("""
    Explore detailed information about specific energy sources and how they compare to alternatives.
    This view helps understand the strengths and weaknesses of each energy source for AI data center applications.
    """)
    
    # Select energy source
    selected_source = st.selectbox(
        "Select an energy source to explore:",
        df['Energy_Source'].tolist()
    )
    
    if selected_source:
        # Get data for the selected source
        source_data = df[df['Energy_Source'] == selected_source].iloc[0]
        
        # Display description
        st.markdown('<div class="subsection-header">Description</div>', unsafe_allow_html=True)
        st.markdown(energy_source_explanations[selected_source])
        
        # Display metrics
        st.markdown('<div class="subsection-header">Key Metrics</div>', unsafe_allow_html=True)
        
        # Create columns for metrics
        cols = st.columns(2)
        
        for i, metric in enumerate(available_metrics):
            col_idx = i % 2
            with cols[col_idx]:
                st.markdown(f"**{metric_display_names[metric]}**: {source_data[metric]}")
                
                # Show how this compares to others
                if metric in lower_is_better:
                    better_count = (df[metric] < source_data[metric]).sum()
                    worse_count = (df[metric] > source_data[metric]).sum()
                    rank = df[metric].rank().loc[source_data.name]
                else:
                    better_count = (df[metric] > source_data[metric]).sum()
                    worse_count = (df[metric] < source_data[metric]).sum()
                    rank = df[metric].rank(ascending=False).loc[source_data.name]
                
                st.markdown(f"Rank: {int(rank)} of {len(df)} (Better than {worse_count} sources, worse than {better_count} sources)")
        
        # Strengths and weaknesses
        st.markdown('<div class="subsection-header">Strengths and Weaknesses</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Strengths")
            
            # Identify top 3 metrics where this source performs well
            strengths = []
            
            for metric in available_metrics:
                if metric in lower_is_better:
                    rank = df[metric].rank().loc[source_data.name]
                else:
                    rank = df[metric].rank(ascending=False).loc[source_data.name]
                
                strengths.append((metric, rank))
            
            strengths.sort(key=lambda x: x[1])
            
            for metric, rank in strengths[:3]:
                st.markdown(f"**{metric_display_names[metric]}**: Ranked {int(rank)} of {len(df)}")
        
        with col2:
            st.markdown("#### Weaknesses")
            
            # Identify bottom 3 metrics where this source performs poorly
            weaknesses = []
            
            for metric in available_metrics:
                if metric in lower_is_better:
                    rank = df[metric].rank().loc[source_data.name]
                else:
                    rank = df[metric].rank(ascending=False).loc[source_data.name]
                
                weaknesses.append((metric, rank))
            
            weaknesses.sort(key=lambda x: x[1], reverse=True)
            
            for metric, rank in weaknesses[:3]:
                st.markdown(f"**{metric_display_names[metric]}**: Ranked {int(rank)} of {len(df)}")
        
        # Comparison with alternatives
        st.markdown('<div class="subsection-header">Comparison with Alternatives</div>', unsafe_allow_html=True)
        
        # Select metrics for comparison
        comparison_metrics = st.multiselect(
            "Select metrics for comparison:",
            available_metrics,
            default=available_metrics[:5],
            format_func=lambda x: metric_display_names[x]
        )
        
        if comparison_metrics:
            # Create bar charts comparing this source to others for each selected metric
            for metric in comparison_metrics:
                st.markdown(f"#### {metric_display_names[metric]}")
                
                # Create a bar chart using Plotly
                fig = px.bar(
                    df,
                    x='Energy_Source',
                    y=metric,
                    title=f"Comparison of {metric_display_names[metric]}",
                    labels={metric: metric_display_names[metric], 'Energy_Source': 'Energy Source'},
                    height=400
                )
                
                # Highlight the selected source
                fig.update_traces(
                    marker_color=['#1E88E5' if source == selected_source else '#A5D6A7' for source in df['Energy_Source']]
                )
                
                # Update layout
                fig.update_layout(
                    xaxis_title='Energy Source',
                    yaxis_title=metric_display_names[metric],
                    xaxis={'categoryorder': 'total ascending'}
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Add interpretation note
                if metric in lower_is_better:
                    st.markdown('<div class="info-box">Note: For this metric, lower values are better.</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="info-box">Note: For this metric, higher values are better.</div>', unsafe_allow_html=True)
        
        # Suitability for AI data centers
        st.markdown('<div class="subsection-header">Suitability for AI Data Centers</div>', unsafe_allow_html=True)
        
        # Calculate a suitability score based on key metrics
        reliability_score = source_data['Grid_Reliability'] * 0.3
        cost_score = (1 - (source_data['LCOE_Avg'] - df['LCOE_Avg'].min()) / (df['LCOE_Avg'].max() - df['LCOE_Avg'].min())) * 0.25
        scalability_score = source_data['Scalability'] * 0.2
        capacity_score = source_data['Capacity_Factor'] / 100 * 0.15
        carbon_score = (1 - (source_data['Carbon_Intensity'] - df['Carbon_Intensity'].min()) / (df['Carbon_Intensity'].max() - df['Carbon_Intensity'].min())) * 0.1
        
        total_score = reliability_score + cost_score + scalability_score + capacity_score + carbon_score
        
        # Display the score
        st.markdown(f"**Overall Suitability Score: {total_score:.2f} / 1.00**")
        
        # Create a gauge chart for the score
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=total_score,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Suitability for AI Data Centers"},
            gauge={
                'axis': {'range': [0, 1]},
                'bar': {'color': "#1E88E5"},
                'steps': [
                    {'range': [0, 0.33], 'color': "#EF5350"},
                    {'range': [0.33, 0.66], 'color': "#FFCA28"},
                    {'range': [0.66, 1], 'color': "#66BB6A"}
                ]
            }
        ))
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Explanation of the score
        st.markdown("""
        The suitability score is calculated based on the following weighted factors:
        - Grid Reliability (30%): How reliable and dispatchable the energy source is
        - Cost (25%): Levelized cost of electricity relative to other sources
        - Scalability (20%): How easily the energy source can be scaled for growing demands
        - Capacity Factor (15%): How consistently the energy source produces power
        - Carbon Intensity (10%): Environmental impact relative to other sources
        
        A higher score indicates better overall suitability for AI data center applications.
        """)

def show_ai_datacenter_context():
    st.markdown('<div class="section-header">AI Data Center Context</div>', unsafe_allow_html=True)
    
    st.markdown("""
    Understanding the unique energy requirements of AI data centers is crucial for selecting
    appropriate power sources. This section provides context on the specific needs and challenges
    of powering advanced AI infrastructure.
    """)
    
    # Display AI data center context
    for key, value in ai_context.items():
        st.markdown(f'<div class="subsection-header">{key.replace("_", " ")}</div>', unsafe_allow_html=True)
        st.markdown(value)
    
    # Additional context specific to energy source selection
    st.markdown('<div class="subsection-header">Energy Source Selection Considerations</div>', unsafe_allow_html=True)
    
    st.markdown("""
    When selecting energy sources for AI data centers, several key factors must be considered:
    
    1. **Reliability and Uptime**: AI data centers require extremely reliable power with minimal downtime.
       Even brief interruptions can damage hardware and disrupt critical workloads.
       
    2. **Scalability**: AI computing demands are growing exponentially, requiring energy sources
       that can scale quickly to meet increasing power needs.
       
    3. **Cost Efficiency**: Energy costs represent a significant portion of AI data center operating expenses.
       Cost-effective power sources can substantially impact the economics of AI deployment.
       
    4. **Environmental Impact**: Many tech companies have sustainability commitments that require
       reducing carbon emissions from their operations, including data centers.
       
    5. **Geographic Flexibility**: AI data centers may need to be located in specific regions for
       latency, regulatory, or strategic reasons, requiring energy sources that can be deployed in various locations.
       
    6. **Regulatory Environment**: Energy regulations vary by region and can impact the viability
       of different power sources for data center applications.
    """)
    
    # Case studies
    st.markdown('<div class="subsection-header">Case Studies</div>', unsafe_allow_html=True)
    
    st.markdown("""
    #### Microsoft's Nuclear-Powered AI Data Center
    
    Microsoft has partnered with nuclear technology companies to explore using small modular reactors (SMRs)
    to power AI data centers. This approach aims to provide reliable, carbon-free electricity at scale
    to meet the growing demands of AI computing.
    
    #### Google's Renewable Energy Strategy
    
    Google has committed to operating its data centers on 24/7 carbon-free energy by 2030.
    This involves a combination of on-site renewable generation, power purchase agreements,
    and advanced energy storage solutions to ensure reliable, clean power.
    
    #### Meta's Hybrid Approach
    
    Meta (formerly Facebook) uses a mix of renewable energy and traditional grid power for its data centers,
    with a focus on locating facilities in regions with access to renewable energy sources.
    They've also invested in energy storage technologies to improve reliability.
    """)
    
    # Future trends
    st.markdown('<div class="subsection-header">Future Trends</div>', unsafe_allow_html=True)
    
    st.markdown("""
    Several emerging trends are likely to shape the future of energy for AI data centers:
    
    1. **Hybrid Energy Systems**: Combining multiple energy sources (e.g., solar + storage + natural gas)
       to optimize for reliability, cost, and environmental impact.
       
    2. **Advanced Nuclear**: Small modular reactors and other advanced nuclear technologies designed
       specifically for data center applications.
       
    3. **Hydrogen Integration**: Green hydrogen as both a primary energy source and a long-duration
       storage medium for renewable energy.
       
    4. **Microgrids**: Self-contained energy systems that can operate independently from the main grid,
       providing enhanced reliability and control.
       
    5. **AI-Optimized Energy Management**: Using AI itself to optimize energy use and distribution
       within data centers, reducing waste and improving efficiency.
    """)

def show_data_sources():
    st.markdown('<div class="section-header">Data Sources</div>', unsafe_allow_html=True)
    
    st.markdown("""
    This application uses data compiled from various credible sources to provide accurate
    comparisons of energy sources for AI data centers. Below are the primary sources used
    for each metric.
    """)
    
    # Display data sources for each metric
    for metric, sources in data_sources.items():
        st.markdown(f'<div class="subsection-header">{metric_display_names.get(metric, metric)}</div>', unsafe_allow_html=True)
        
        for source in sources:
            st.markdown(f"- {source}")
    
    # Methodology notes
    st.markdown('<div class="subsection-header">Methodology Notes</div>', unsafe_allow_html=True)
    
    st.markdown("""
    The data presented in this application represents typical or average values for each energy source
    and metric. Actual values may vary based on specific implementations, geographic locations,
    technological advancements, and other factors.
    
    For metrics where ranges are common (such as LCOE), we've provided both minimum and maximum values
    to reflect this variability. For visualization purposes, we use the average of these values.
    
    Some metrics, such as Grid Reliability and Scalability, are based on expert assessments and
    industry standards rather than direct measurements, as these factors involve qualitative judgments.
    
    All data has been normalized where appropriate to facilitate fair comparisons across different
    energy sources and metrics.
    """)
    
    # Limitations
    st.markdown('<div class="subsection-header">Limitations</div>', unsafe_allow_html=True)
    
    st.markdown("""
    While we've made every effort to provide accurate and up-to-date information, there are several
    limitations to consider when interpreting the data:
    
    1. **Temporal Variability**: Energy technologies are evolving rapidly, and costs and performance
       characteristics may change over time.
       
    2. **Geographic Variability**: The performance and viability of energy sources can vary significantly
       by location due to resource availability, climate, regulations, and other factors.
       
    3. **Implementation Specifics**: Actual performance depends on specific implementation details,
       such as the exact technology used, scale of deployment, and operational practices.
       
    4. **Contextual Factors**: Local grid conditions, regulatory environments, and existing infrastructure
       can significantly impact the suitability of different energy sources.
       
    5. **Emerging Technologies**: Some newer technologies, such as small modular reactors and certain
       hydrogen applications, have limited operational data and rely more on projections.
    """)
    
    # Additional resources
    st.markdown('<div class="subsection-header">Additional Resources</div>', unsafe_allow_html=True)
    
    st.markdown("""
    For those interested in exploring this topic further, the following resources provide additional
    information on energy sources for data centers:
    
    - [International Energy Agency (IEA) - Data Centers and Energy](https://www.iea.org/reports/data-centres-and-data-transmission-networks)
    - [U.S. Department of Energy - Data Center Energy Efficiency](https://www.energy.gov/eere/buildings/data-centers-and-servers)
    - [Uptime Institute - Data Center Energy Efficiency](https://uptimeinstitute.com/resources)
    - [National Renewable Energy Laboratory (NREL) - Renewable Energy for Data Centers](https://www.nrel.gov/computational-science/renewable-energy-data-centers.html)
    - [Lawrence Berkeley National Laboratory - Data Center Energy](https://datacenters.lbl.gov/)
    """)

# Run the application
if __name__ == "__main__":
    main()
