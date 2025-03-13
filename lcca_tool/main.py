import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import os
import sys

# Add the current directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lcca_tool.models.lcca_calculator import calculate_lcca
from lcca_tool.utils.visualization import create_cost_breakdown_chart, create_costs_over_time_chart, create_scenario_comparison_chart

# Set page configuration
st.set_page_config(
    page_title="Skyscraper LCCA Tool",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Title and description
st.title("Life Cycle Cost Analysis for Manhattan Skyscraper")
st.markdown("""
This application helps you perform a Life Cycle Cost Analysis (LCCA) for a skyscraper construction project in Manhattan.
Adjust the parameters in the sidebar to see how they affect the total life cycle cost over a 30-year period.
""")

# Sidebar for inputs
st.sidebar.header("Building Parameters")

# Building size parameters
building_height = st.sidebar.slider("Building Height (meters)", 100, 500, 300, 10)
num_floors = st.sidebar.slider("Number of Floors", 20, 100, 60, 1)
floor_area = st.sidebar.slider("Average Floor Area (sq. meters)", 1000, 5000, 2500, 100)
total_area = floor_area * num_floors
st.sidebar.metric("Total Building Area (sq. meters)", f"{total_area:,}")

# Building specifications
st.sidebar.subheader("Building Specifications")
facade_type = st.sidebar.selectbox(
    "Facade Type",
    ["Glass Curtain Wall", "Precast Concrete", "Stone Veneer", "Metal Panel", "Mixed"]
)
structural_system = st.sidebar.selectbox(
    "Structural System",
    ["Steel Frame", "Reinforced Concrete", "Composite", "Tube System"]
)
hvac_system = st.sidebar.selectbox(
    "HVAC System",
    ["Variable Air Volume (VAV)", "Chilled Beams", "VRF System", "Hybrid System"]
)
interior_quality = st.sidebar.selectbox(
    "Interior Finish Quality",
    ["Standard", "Premium", "Luxury", "Ultra Luxury"]
)

# Cost assumptions
st.sidebar.header("Cost Assumptions")
construction_cost_per_sqm = st.sidebar.slider(
    "Construction Cost ($/sq. meter)",
    5000, 15000, 8000, 100
)
annual_inflation_rate = st.sidebar.slider(
    "Annual Inflation Rate (%)",
    1.0, 5.0, 2.5, 0.1
) / 100
discount_rate = st.sidebar.slider(
    "Discount Rate (%)",
    2.0, 10.0, 5.0, 0.1
) / 100
energy_cost_per_sqm = st.sidebar.slider(
    "Annual Energy Cost ($/sq. meter)",
    20, 100, 50, 5
)
maintenance_cost_percentage = st.sidebar.slider(
    "Annual Maintenance Cost (% of Construction)",
    0.5, 3.0, 1.5, 0.1
) / 100

# Analysis period
analysis_period = 30  # years

# Calculate button
if st.sidebar.button("Calculate Life Cycle Cost"):
    # Show a spinner while calculating
    with st.spinner("Calculating Life Cycle Costs..."):
        # Prepare input data for the calculator
        input_data = {
            "building_height": building_height,
            "num_floors": num_floors,
            "floor_area": floor_area,
            "total_area": total_area,
            "facade_type": facade_type,
            "structural_system": structural_system,
            "hvac_system": hvac_system,
            "interior_quality": interior_quality,
            "construction_cost_per_sqm": construction_cost_per_sqm,
            "annual_inflation_rate": annual_inflation_rate,
            "discount_rate": discount_rate,
            "energy_cost_per_sqm": energy_cost_per_sqm,
            "maintenance_cost_percentage": maintenance_cost_percentage,
            "analysis_period": analysis_period
        }
        
        # Calculate LCCA
        results = calculate_lcca(input_data)
        
        # Display results
        st.header("Life Cycle Cost Analysis Results")
        
        # Summary metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Initial Cost", f"${results['initial_cost']:,.0f}")
        with col2:
            st.metric("Total Life Cycle Cost", f"${results['total_life_cycle_cost']:,.0f}")
        with col3:
            st.metric("Present Value of Life Cycle Cost", f"${results['present_value_lcc']:,.0f}")
        
        # Cost breakdown chart
        st.subheader("Cost Breakdown")
        cost_breakdown_chart = create_cost_breakdown_chart(results['cost_breakdown'])
        st.plotly_chart(cost_breakdown_chart, use_container_width=True)
        
        # Costs over time chart
        st.subheader("Costs Over Time")
        costs_over_time_chart = create_costs_over_time_chart(results['annual_costs'], analysis_period)
        st.plotly_chart(costs_over_time_chart, use_container_width=True)
        
        # Scenario comparison (placeholder for now)
        st.subheader("Scenario Comparison")
        st.info("To compare scenarios, adjust parameters and calculate multiple times. The last 3 scenarios will be shown here.")
        
        # Store the current scenario in session state for comparison
        if 'scenarios' not in st.session_state:
            st.session_state.scenarios = []
        
        # Create a name for this scenario
        scenario_name = f"Scenario {len(st.session_state.scenarios) + 1}"
        
        # Add the current scenario to the list
        current_scenario = {
            "name": scenario_name,
            "total_cost": results['total_life_cycle_cost'],
            "present_value": results['present_value_lcc'],
            "initial_cost": results['initial_cost'],
            "operating_cost": results['cost_breakdown']['Operating Costs'],
            "maintenance_cost": results['cost_breakdown']['Maintenance Costs'],
            "replacement_cost": results['cost_breakdown']['Replacement Costs'],
            "timestamp": datetime.now().strftime("%H:%M:%S")
        }
        
        st.session_state.scenarios.append(current_scenario)
        
        # Keep only the last 3 scenarios
        if len(st.session_state.scenarios) > 3:
            st.session_state.scenarios = st.session_state.scenarios[-3:]
        
        # Display scenario comparison if we have more than one scenario
        if len(st.session_state.scenarios) > 1:
            scenario_chart = create_scenario_comparison_chart(st.session_state.scenarios)
            st.plotly_chart(scenario_chart, use_container_width=True)
        else:
            st.write("Calculate at least one more scenario to see a comparison.")
else:
    # Default view before calculation
    st.info("Adjust the parameters in the sidebar and click 'Calculate Life Cycle Cost' to see the results.")
    
    # Placeholder for what the results will look like
    st.header("Sample Results Preview")
    st.write("This is a preview of what the results will look like. Adjust parameters and calculate to see actual results.")
    
    # Sample data for preview
    sample_cost_breakdown = {
        "Initial Construction": 200000000,
        "Operating Costs": 150000000,
        "Maintenance Costs": 75000000,
        "Replacement Costs": 50000000,
        "Financing Costs": 25000000
    }
    
    # Sample charts
    st.subheader("Sample Cost Breakdown")
    sample_pie = create_cost_breakdown_chart(sample_cost_breakdown)
    st.plotly_chart(sample_pie, use_container_width=True)
    
    # Sample costs over time
    sample_annual_costs = {
        "year": list(range(1, 31)),
        "Operating Costs": [5000000 * (1.025 ** i) for i in range(30)],
        "Maintenance Costs": [2500000 * (1.02 ** i) for i in range(30)],
        "Replacement Costs": [0] * 30
    }
    # Add some replacement costs in specific years
    sample_annual_costs["Replacement Costs"][9] = 10000000  # Year 10
    sample_annual_costs["Replacement Costs"][19] = 15000000  # Year 20
    sample_annual_costs["Replacement Costs"][29] = 5000000   # Year 30
    
    st.subheader("Sample Costs Over Time")
    sample_line = create_costs_over_time_chart(sample_annual_costs, 30)
    st.plotly_chart(sample_line, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("© 2025 Skyscraper LCCA Tool | Developed with Streamlit")
