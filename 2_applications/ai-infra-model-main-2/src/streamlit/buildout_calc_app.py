import numpy as np
import pandas as pd
import plotly.graph_objects as go

import streamlit as st
from src.constants.gpu_dataclass import ALL_GPU_LIST
from src.constants.value_chain_depreciation_schedules import (
    NVIDIA_COMPUTE_SHARE,
    NVIDIA_DEFAULT_GROSS_MARGIN,
    CapexDepreciationSchedule,
)

# gpu_cloud_helpers no longer needed - using capex_helpers instead
from src.utils.streamlit_app_helpers import create_styled_dataframe, format_number_to_string

NVIDIA_GIR_DATA_PATH = "data/nvda_shipment_data/nvda_gir.xlsx"

DEFAULT_PUE = 1.2
DEFAULT_UTILIZATION = 0.8
WATTS_TO_KILOWATTS = 1_000
WATTS_TO_MEGAWATTS = 1_000_000

DEFAULT_GPU_MODEL = "GB300"

# Predefined Scenarios
SCENARIOS = {
    "NVIDIA 2025 Shipments": {
        "input_mode": "Total Budget",
        "total_budget_billions": 400.0,
        "power_capacity_gw": None,
        "power_construction_cost": 2500,
        "datacenter_construction_cost_millions": 15,
        "gpu_model": "GB300"
    },
    "Oracle/OpenAI": {
        "input_mode": "Total Budget",
        "total_budget_billions": 200.0,
        "power_capacity_gw": None,
        "power_construction_cost": 2500,
        "datacenter_construction_cost_millions": 15,
        "gpu_model": "GB300"
    },
    "OpenAI-AMD 6 GW": {
        "input_mode": "Power Capacity (GW)",
        "total_budget_billions": None,
        "power_capacity_gw": 6.0,
        "power_construction_cost": 2500,
        "datacenter_construction_cost_millions": 15,
        "gpu_model": "MI450X"
    }
}

### ---- Helper Methods ---- ###

def setup_sidebar_config():
    """Handle all sidebar configuration inputs"""
    st.sidebar.markdown("## Scenario Selection")
    selected_scenario = st.sidebar.selectbox(
        "Choose Scenario",
        options=list(SCENARIOS.keys()),
        index=0
    )

    scenario = SCENARIOS[selected_scenario]

    st.sidebar.markdown("## Input Mode")
    mode_options = ["Total Budget", "Power Capacity (GW)"]
    default_mode_index = mode_options.index(scenario["input_mode"])
    input_mode = st.sidebar.radio(
        "Calculate from:",
        options=mode_options,
        index=default_mode_index
    )

    st.sidebar.markdown("## Budget and Infrastructure Parameters")

    if input_mode == "Total Budget":
        default_budget = scenario["total_budget_billions"] if scenario["total_budget_billions"] is not None else 200.0
        total_budget = st.sidebar.number_input(
            "Total Budget (Billion $)",
            min_value=0.001,
            max_value=10000.0,
            value=default_budget,
            step=0.1
        ) * 1_000_000_000  # Convert to actual dollars
        power_capacity_gw = None
    else:  # Power Capacity mode
        total_budget = None
        default_power = scenario["power_capacity_gw"] if scenario["power_capacity_gw"] is not None else 6.0
        power_capacity_gw = st.sidebar.number_input(
            "Power Capacity (GW)",
            min_value=0.1,
            max_value=100.0,
            value=default_power,
            step=0.1
        )

    power_construction_cost = st.sidebar.number_input(
        "Power Construction Cost ($/kW)",
        min_value=100,
        max_value=10000,
        value=scenario["power_construction_cost"],
        step=100
    )

    datacenter_construction_cost = st.sidebar.number_input(
        "Datacenter Construction Cost (Million $/MW)",
        min_value=1,
        max_value=20,
        value=scenario["datacenter_construction_cost_millions"],
        step=1
    ) * 1_000_000

    # Lease and Equipment Parameters
    st.sidebar.markdown("## Lease and Equipment Parameters")
    lease_term_years = st.sidebar.slider(
        "Datacenter Lease Term (Years)",
        min_value=5,
        max_value=25,
        value=15,
        step=1
    )

    chip_replacement_years = st.sidebar.slider(
        "Chip Replacement Cycle (Years)",
        min_value=2,
        max_value=10,
        value=5,
        step=1
    )

    # GPU Configuration
    st.sidebar.markdown("## GPU Parameters")
    gpu_models = {gpu.name: gpu for gpu in ALL_GPU_LIST}
    default_gpu_name = scenario["gpu_model"] if selected_scenario in SCENARIOS else DEFAULT_GPU_MODEL
    default_index = next((i for i, gpu in enumerate(ALL_GPU_LIST) if gpu.name == default_gpu_name), 0)
    selected_gpu = st.sidebar.selectbox(
        "GPU Model",
        options=list(gpu_models.keys()),
        index=default_index
    )
    gpu_model = gpu_models[selected_gpu]

    # Wattage and price overrides in expander
    base_wattage = gpu_model.wattage
    base_accelerator_price = gpu_model.ai_accelerator_price

    with st.sidebar.expander("Advanced GPU Settings", expanded=False):
        wattage_override = st.number_input(
            "Per-Chip Wattage Override (W)",
            min_value=100,
            max_value=5000,
            value=int(base_wattage),
            step=50,
            help=f"Override the default wattage for {gpu_model.name} ({int(base_wattage)}W). Affects power and infrastructure calculations."
        )

        if wattage_override != base_wattage:
            st.caption(f"⚠️ Using {wattage_override}W instead of default {int(base_wattage)}W")

        st.markdown("---")

        accelerator_price_override = st.number_input(
            "Accelerator Price Override ($)",
            min_value=100,
            max_value=200000,
            value=int(base_accelerator_price),
            step=1000,
            help=f"Override the default accelerator price for {gpu_model.name} (${int(base_accelerator_price):,}). This is independent of chip vendor margin adjustments."
        )

        if accelerator_price_override != base_accelerator_price:
            pct_change = ((accelerator_price_override / base_accelerator_price) - 1) * 100
            st.caption(f"⚠️ Using ${int(accelerator_price_override):,} instead of ${int(base_accelerator_price):,} ({pct_change:+.1f}%)")

    # Create a modified GPU model with overridden wattage and price
    from dataclasses import replace
    if wattage_override != base_wattage or accelerator_price_override != base_accelerator_price:
        gpu_model = replace(gpu_model, wattage=wattage_override, ai_accelerator_price=accelerator_price_override)

    # Calculate total chip cost (accelerator price + other compute costs)
    accelerator_price = gpu_model.ai_accelerator_price if hasattr(gpu_model, 'ai_accelerator_price') else 0
    other_compute_costs = gpu_model.other_compute_costs if hasattr(gpu_model, 'other_compute_costs') else 0
    chip_cost = accelerator_price + other_compute_costs



    # Calculate the ratio of accelerator price to total cost for later use
    accelerator_ratio = accelerator_price / chip_cost if chip_cost > 0 else 0.7  # Default to 70% if no data

    pue = st.sidebar.slider("Power Usage Effectiveness (PUE)", min_value=1.0, max_value=2.0, value=DEFAULT_PUE)

    # Useful Life Assumptions for Depreciation
    st.sidebar.markdown("## Useful Life Assumptions (Years)")
    chip_useful_life = st.sidebar.slider(
        "Chip Useful Life (Years)",
        min_value=2,
        max_value=10,
        value=5,
        step=1
    )

    power_useful_life = st.sidebar.slider(
        "Power Infrastructure Useful Life (Years)",
        min_value=10,
        max_value=40,
        value=25,
        step=1
    )

    datacenter_useful_life = st.sidebar.slider(
        "Datacenter Shell Useful Life (Years)",
        min_value=5,
        max_value=25,
        value=20,
        step=1
    )

    # Chip Vendor Margin Adjustment section
    st.sidebar.markdown("## Chip Vendor Margin Adjustment")
    chip_gross_margin = st.sidebar.number_input(
        "Chip Vendor Gross Margin (%)",
        value=NVIDIA_DEFAULT_GROSS_MARGIN * 100,
        min_value=0.0,
        max_value=100.0,
        step=1.0,
        help="Adjust NVIDIA's gross margin. Default is 75%."
    ) / 100

    # Calculate and display impact
    nvidia_multiplier, margin_adjustment_multiplier = CapexDepreciationSchedule.calculate_chip_price_multiplier(chip_gross_margin)

    # Show feedback (same as Value Chain app)
    if chip_gross_margin != NVIDIA_DEFAULT_GROSS_MARGIN:
        nvidia_change_pct = (nvidia_multiplier - 1.0) * 100
        total_change_pct = (margin_adjustment_multiplier - 1.0) * 100

        if margin_adjustment_multiplier < 1.0:
            st.sidebar.success(f"✅ Chip prices reduced by {abs(total_change_pct):.1f}%")
        else:
            st.sidebar.warning(f"⚠️ Chip prices increased by {total_change_pct:.1f}%")

        st.sidebar.caption(f"NVIDIA portion: {'+' if nvidia_change_pct > 0 else ''}{nvidia_change_pct:.1f}%")

    # Add expander with math explanation
    with st.sidebar.expander("Margin Impact Math", expanded=chip_gross_margin != NVIDIA_DEFAULT_GROSS_MARGIN):
        st.markdown(f"""
        **Price = Cost / (1 - Margin)**

        - Default margin: {NVIDIA_DEFAULT_GROSS_MARGIN*100:.0f}%
        - New margin: {chip_gross_margin*100:.0f}%
        - NVIDIA multiplier: {nvidia_multiplier:.2f}x
        - Blended multiplier: {margin_adjustment_multiplier:.2f}x

        *Note: {NVIDIA_COMPUTE_SHARE*100:.0f}% of compute cost follows NVIDIA margin,
        {(1-NVIDIA_COMPUTE_SHARE)*100:.0f}% is fixed integration cost.*
        """)

    # Display adjusted pricing information
    st.sidebar.markdown("## Chip Pricing Summary")
    adjusted_accelerator_price = accelerator_price * margin_adjustment_multiplier
    adjusted_chip_cost = adjusted_accelerator_price + other_compute_costs

    if chip_gross_margin != NVIDIA_DEFAULT_GROSS_MARGIN:
        st.sidebar.markdown(f"**Base Accelerator Price:** ~~${int(accelerator_price):,}~~")
        st.sidebar.markdown(f"**Adjusted Accelerator Price:** ${int(adjusted_accelerator_price):,}")
    else:
        st.sidebar.markdown(f"**Accelerator Chip Price:** ${int(accelerator_price):,}")

    st.sidebar.markdown(f"**Other Compute Costs:** ${int(other_compute_costs):,}")
    st.sidebar.markdown(f"**Total Chip Cost:** ${int(adjusted_chip_cost):,}")

    return input_mode, total_budget, power_capacity_gw, power_construction_cost, datacenter_construction_cost, gpu_model, chip_cost, pue, accelerator_ratio, lease_term_years, chip_replacement_years, chip_useful_life, power_useful_life, datacenter_useful_life, chip_gross_margin, margin_adjustment_multiplier

def display_summary_metrics(max_gpus, power_required, total_budget, gpu_model, chip_cost):
    """Display the main summary metrics"""
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Number of GPUs", f"{int(max_gpus):,}")
    with col2:
        st.metric("Power Required", f"{power_required:.0f} MW")
    with col3:
        st.metric("Total Budget", f"${format_number_to_string(total_budget)}")

    # Second row of metrics
    col4, col5, col6 = st.columns(3)
    with col4:
        st.metric("Chip Model", gpu_model.name)
    with col5:
        st.metric("All-In Cost per Chip", f"${chip_cost:,}")
    with col6:
        st.metric("All-In Watts per Chip", f"{gpu_model.wattage:.0f}W")

def display_cost_breakdowns(total_compute_cost, datacenter_cost, power_cost, total_cost, accelerator_cost, non_accelerator_compute_cost, lease_term_years, chip_replacement_years):
    """Display cost breakdown tables and charts"""
    # Create main cost breakdown table
    cost_breakdown = pd.DataFrame({
        "Component": ["Compute Hardware", "Datacenter Construction", "Power Generation"],
        "Cost": [
            f"${format_number_to_string(total_compute_cost)}",
            f"${format_number_to_string(datacenter_cost)}",
            f"${format_number_to_string(power_cost)}"
        ],
        "Percentage": [
            f"{(total_compute_cost / total_cost * 100):.1f}%",
            f"{(datacenter_cost / total_cost * 100):.1f}%",
            f"{(power_cost / total_cost * 100):.1f}%"
        ]
    })

    # Create compute cost breakdown table
    compute_breakdown = pd.DataFrame({
        "Component": ["AI Accelerator Hardware", "Non-Accelerator Hardware"],
        "Cost": [
            f"${format_number_to_string(accelerator_cost)}",
            f"${format_number_to_string(non_accelerator_compute_cost)}"
        ],
        "Percentage": [
            f"{(accelerator_cost / total_compute_cost * 100):.1f}%",
            f"{(non_accelerator_compute_cost / total_compute_cost * 100):.1f}%"
        ]
    })

    create_styled_dataframe(
        cost_breakdown,
        title="Initial Capex to Fill Datacenter"
    )

    create_styled_dataframe(
        compute_breakdown,
        highlight_keys=["AI Accelerator Hardware"],
        title="Compute Hardware Breakdown"
    )

    # Calculate multi-generation costs over lease term
    # For 15 years with 5-year cycles: initial (0-5), replacement 1 (5-10), replacement 2 (10-15) = 3 total
    total_chip_generations = (lease_term_years + chip_replacement_years - 1) // chip_replacement_years
    _ = total_chip_generations - 1  # Subtract initial purchase (not used downstream)
    multi_gen_compute_cost = total_compute_cost * total_chip_generations
    multi_gen_total_cost = multi_gen_compute_cost + datacenter_cost + power_cost

    # Create multi-generation breakdown table
    multi_gen_breakdown = pd.DataFrame({
        "Component": [f"Compute Hardware ({total_chip_generations} generations)", "Datacenter Construction", "Power Generation"],
        "Cost": [
            f"${format_number_to_string(multi_gen_compute_cost)}",
            f"${format_number_to_string(datacenter_cost)}",
            f"${format_number_to_string(power_cost)}"
        ],
        "Percentage": [
            f"{(multi_gen_compute_cost / multi_gen_total_cost * 100):.1f}%",
            f"{(datacenter_cost / multi_gen_total_cost * 100):.1f}%",
            f"{(power_cost / multi_gen_total_cost * 100):.1f}%"
        ]
    })

    create_styled_dataframe(
        multi_gen_breakdown,
        highlight_keys=["Compute Hardware"],
        title=f"Capex Over Entire {lease_term_years}-Year Datacenter Lease"
    )
    st.markdown(f"*Assumes **{total_chip_generations} generations of chips** - refresh cycles every {chip_replacement_years} years over {lease_term_years}-year datacenter lease*")

    # First pie chart - Initial Capex
    compute_pct = (total_compute_cost / total_cost * 100)
    datacenter_pct = (datacenter_cost / total_cost * 100)
    power_pct = (power_cost / total_cost * 100)

    compute_text = f"${format_number_to_string(total_compute_cost)} ({compute_pct:.1f}%)<br>Accelerator: ${format_number_to_string(accelerator_cost)}<br>Other IT: ${format_number_to_string(non_accelerator_compute_cost)}"
    datacenter_text = f"${format_number_to_string(datacenter_cost)}<br>({datacenter_pct:.1f}%)"
    power_text = f"${format_number_to_string(power_cost)}<br>({power_pct:.1f}%)"

    fig1 = go.Figure(data=[go.Pie(
        labels=["Compute Hardware", "Datacenter Construction", "Power Generation"],
        values=[total_compute_cost, datacenter_cost, power_cost],
        hole=.3,
        marker=dict(colors=['darkgreen', 'lightgray', 'yellow']),
        textfont=dict(size=[14, 14, 10], color=['white', 'black', 'black']),
        textinfo='text',
        text=[compute_text, datacenter_text, power_text],
        textposition="inside",
        hovertemplate="<b>%{label}</b><br>" +
                     "Cost: $%{value:,.0f}<br>" +
                     "Percentage: %{percent}<br>" +
                     "<extra></extra>"
    )])

    fig1.update_layout(
        height=500,
        showlegend=True,  # Show legend on first chart too
        margin=dict(t=50, b=80, l=50, r=50),
        legend=dict(
            orientation="h",  # Horizontal legend
            x=0.5,
            xanchor='center',
            y=-0.15,  # Position below the chart
            yanchor='top'
        )
    )

    # Create side-by-side columns for pie charts
    col1, col2 = st.columns(2)

    with col1:
        st.plotly_chart(fig1, use_container_width=True, key="initial_capex_pie")
        st.markdown(f"<div style='text-align: center'><b>Initial Capex to Fill Datacenter</b><br>(${format_number_to_string(total_cost)})</div>", unsafe_allow_html=True)

    # Second pie chart - Multi-generation Capex
    multi_gen_accelerator_cost = accelerator_cost * total_chip_generations
    multi_gen_other_it_cost = non_accelerator_compute_cost * total_chip_generations

    # Calculate percentages for multi-gen
    multi_compute_pct = (multi_gen_compute_cost / multi_gen_total_cost * 100)
    multi_datacenter_pct = (datacenter_cost / multi_gen_total_cost * 100)
    multi_power_pct = (power_cost / multi_gen_total_cost * 100)

    multi_compute_text = f"${format_number_to_string(multi_gen_compute_cost)} ({multi_compute_pct:.1f}%)<br>Accelerator: ${format_number_to_string(multi_gen_accelerator_cost)}<br>Other IT: ${format_number_to_string(multi_gen_other_it_cost)}<br>({total_chip_generations}x total)"
    multi_datacenter_text = f"${format_number_to_string(datacenter_cost)}<br>({multi_datacenter_pct:.1f}%)"
    multi_power_text = f"${format_number_to_string(power_cost)}<br>({multi_power_pct:.1f}%)"

    fig2 = go.Figure(data=[go.Pie(
        labels=["Compute Hardware", "Datacenter Construction", "Power Generation"],
        values=[multi_gen_compute_cost, datacenter_cost, power_cost],
        hole=.3,
        marker=dict(colors=['darkgreen', 'lightgray', 'yellow']),
        textfont=dict(size=[14, 14, 10], color=['white', 'black', 'black']),
        textinfo='text',
        text=[multi_compute_text, multi_datacenter_text, multi_power_text],
        textposition="inside",
        hovertemplate="<b>%{label}</b><br>" +
                     "Cost: $%{value:,.0f}<br>" +
                     "Percentage: %{percent}<br>" +
                     "<extra></extra>"
    )])

    fig2.update_layout(
        height=500,
        showlegend=True,  # Show legend on right chart
        margin=dict(t=50, b=80, l=50, r=50),  # Same bottom margin as left chart
        legend=dict(
            orientation="h",  # Horizontal legend
            x=0.5,
            xanchor='center',
            y=-0.15,  # Position further below to use the margin space
            yanchor='top'
        )
    )

    with col2:
        st.plotly_chart(fig2, use_container_width=True, key="multi_gen_capex_pie")
        st.markdown(f"<div style='text-align: center'><b>Capex Over {lease_term_years}-Year Lease</b><br>with {chip_replacement_years}-year server replacements<br>(${format_number_to_string(multi_gen_total_cost)})</div>", unsafe_allow_html=True)

def get_chip_revenue_data():
    """Get the chip model revenue data - returns dict with chip models and their annual revenues"""
    chip_models = {
        "H100": {
            "total_revenue": 800,   # $800B over 5 years
            "start_year": 2023,
            "duration": 5
        },
        "B200": {
            "total_revenue": 1000,  # $1T over 5 years
            "start_year": 2024,
            "duration": 5
        },
        "GB300": {
            "total_revenue": 1200,  # $1.2T over 5 years
            "start_year": 2025,
            "duration": 5
        },
        "VR200": {
            "total_revenue": 1500,  # $1.5T over 5 years
            "start_year": 2026,
            "duration": 5
        },
        "VR300": {
            "total_revenue": 1800,  # $1.8T over 5 years
            "start_year": 2027,
            "duration": 5
        },
        "Feynman": {
            "total_revenue": 2200,  # $2.2T over 5 years
            "start_year": 2028,
            "duration": 5
        },
        "Feynman Ultra": {
            "total_revenue": 2800,  # $2.8T over 5 years
            "start_year": 2029,
            "duration": 5
        },
        "???": {
            "total_revenue": 3500,  # $3.5T over 5 years
            "start_year": 2030,
            "duration": 5
        },
        "??? Ultra": {
            "total_revenue": 4200,  # $4.2T over 5 years
            "start_year": 2031,
            "duration": 5
        }
    }
    return chip_models

def calculate_yearly_chip_costs():
    """Calculate annual chip costs based on Gantt chart data"""
    chip_models = get_chip_revenue_data()
    years = list(range(2023, 2032))  # 2023-2031 coverage

    yearly_totals = {}
    for year in years:
        yearly_totals[year] = 0

        for chip_name, data in chip_models.items():
            start_year = data['start_year']
            duration = data['duration']
            annual_revenue = data['total_revenue'] * 1_000_000_000  # Convert to actual dollars

            if start_year <= year < start_year + duration:
                yearly_totals[year] += annual_revenue / duration  # Annual portion

    return yearly_totals


def display_annual_depreciation(total_compute_cost, datacenter_cost, power_cost, chip_useful_life, power_useful_life, datacenter_useful_life):
    """Display annual depreciation calculations"""

    # Calculate annual depreciation for initial buildout
    chip_annual_depreciation = total_compute_cost / chip_useful_life
    datacenter_annual_depreciation = datacenter_cost / datacenter_useful_life
    power_annual_depreciation = power_cost / power_useful_life
    total_annual_depreciation = chip_annual_depreciation + datacenter_annual_depreciation + power_annual_depreciation

    # Create pie chart for annual depreciation (no table)
    fig_dep = go.Figure(data=[go.Pie(
        labels=["Compute Hardware", "Datacenter Shell", "Power Infrastructure"],
        values=[chip_annual_depreciation, datacenter_annual_depreciation, power_annual_depreciation],
        hole=.3,
        marker=dict(colors=['darkgreen', 'lightgray', 'yellow']),
        textfont=dict(size=14, color=['white', 'black', 'black']),
        textinfo='label+percent',
        textposition="inside",
        hovertemplate="<b>%{label}</b><br>" +
                     "Annual Depreciation: $%{value:,.0f}<br>" +
                     "Percentage: %{percent}<br>" +
                     "<extra></extra>"
    )])

    fig_dep.update_layout(
        height=500,
        showlegend=True,
        margin=dict(t=100, b=100, l=50, r=50),
        title=dict(
            text=f"<b>Annual Depreciation Expense</b><br><span style='font-size:16px'>${format_number_to_string(total_annual_depreciation)}</span>",
            x=0.5,
            y=0.95,
            font=dict(size=20, color="black"),
            xanchor="center"
        )
    )

    st.plotly_chart(fig_dep, use_container_width=True, key="depreciation_pie_chart")



def create_gpu_shipments_chart_and_table(quarter_labels, models, quantities):
    """Create stacked bar chart and summary table for GPU shipments"""
    # Create a bar chart
    fig = go.Figure()

    # Add each GPU model as a separate trace
    for model_name, values in zip(models, quantities):
        # Determine color based on model name
        base_model = model_name.split(" ")[0] if "(" in model_name else model_name
        color = GPU_COLOR_MAP.get(base_model, "#333333")

        fig.add_trace(go.Bar(
            x=quarter_labels,
            y=values,
            name=model_name,
            marker_color=color
        ))

    # Update layout
    fig.update_layout(
        barmode='stack',
        title="NVIDIA GPU Shipments by Quarter",
        xaxis_title="Quarter",
        yaxis_title="Shipments (K units)",
        legend_title="GPU Model",
        height=500
    )

    # Display the chart
    st.plotly_chart(fig, use_container_width=True, key="gpu_shipments_chart")

    # Calculate cumulative total GPU shipments per year
    total_quantities = np.zeros(len(quarter_labels))
    for qty in quantities:
        total_quantities += np.array(qty)

    # Assuming the quarters represent 2023, 2024, 2025, 2026
    years = [2023, 2024, 2025, 2026]

    # Group quarters into years
    cumulative_shipments_by_year = [
        total_quantities[0:4].sum(),   # 2023
        total_quantities[4:8].sum(),   # 2024
        total_quantities[8:12].sum(),  # 2025
        total_quantities[12:].sum()    # 2026
    ]

    # Create a DataFrame for the table
    shipments_df = pd.DataFrame({
        'Total Shipments (K units)': [int(s) for s in cumulative_shipments_by_year]
    }, index=years)

    # Apply formatting to the shipments column using the imported helper function
    shipments_df['Total Shipments'] = shipments_df['Total Shipments (K units)'].apply(
        lambda x: format_number_to_string(x * 1000)
    )

    # Drop the original numeric column
    shipments_df = shipments_df.drop(columns=['Total Shipments (K units)'])


# Update the color map to only include actual GPU models
GPU_COLOR_MAP = {
    gpu.name: color for gpu, color in zip(
        ALL_GPU_LIST,
        ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2"]
    )
}

def read_gpu_shipment_data():
    """
    Read GPU shipment data from Excel file with special handling for Blackwell.
    Split Blackwell shipments 50/50 between B100 and GB200.
    """
    # Read the Excel file
    df = pd.read_excel(NVIDIA_GIR_DATA_PATH)
    quarter_labels = df.columns[1:].tolist()
    gpu_models = df['Unnamed: 0'].tolist()

    # Create a dictionary for quick GPU model lookup
    gpu_model_dict = {gpu.name: gpu for gpu in ALL_GPU_LIST}

    # Initialize result dictionary
    result = {
        'quarter_labels': quarter_labels,
        'models': [],
        'quantities': []
    }

    # Process each GPU model
    for model_name in gpu_models:
        # Get quantities for this model
        model_row = df[df['Unnamed: 0'] == model_name].iloc[0, 1:]
        quantities = model_row.astype(float).tolist()

        # Skip models with no data
        if np.all(np.array(quantities) == 0):
            continue

        # Special case for Blackwell: split between B100 and GB200
        if model_name == "Blackwell":
            # Calculate 50% of quantities for each model
            half_quantities = [q * 0.5 for q in quantities]

            # Add B100
            result['models'].append("B100")
            result['quantities'].append(half_quantities)

            # Add GB200
            result['models'].append("GB200")
            result['quantities'].append(half_quantities)
        else:
            # Normal case: add model and quantities
            if model_name in gpu_model_dict:
                result['models'].append(model_name)
                result['quantities'].append(quantities)
            else:
                st.warning(f"No data available for GPU model: {model_name}")

    return result

def display_gpu_shipments_chart():
    """Display the NVIDIA GPU shipments chart"""
    st.subheader("Quarterly NVIDIA GPU Shipments (K units)")

    # Read shipment data
    shipment_data = read_gpu_shipment_data()

    quarter_labels = shipment_data['quarter_labels']
    models = shipment_data['models']
    quantities = shipment_data['quantities']

    # Create stacked bar chart and summary table using helper
    create_gpu_shipments_chart_and_table(quarter_labels, models, quantities)

def display_gpu_power_chart(shipment_data, power_construction_cost, datacenter_construction_cost):
    """Display the power requirements based on NVIDIA GPU shipments"""
    st.subheader("Quarterly Incremental Power Requirements from NVIDIA GPU Shipments Alone")

    quarter_labels = shipment_data['quarter_labels']
    models = shipment_data['models']
    quantities = shipment_data['quantities']

    # Create a dictionary for quick GPU model lookup
    gpu_model_dict = {gpu.name: gpu for gpu in ALL_GPU_LIST}

    # Calculate power data for each model
    power_data = []
    for model_name, qty in zip(models, quantities):
        if model_name in gpu_model_dict:
            gpu_model = gpu_model_dict[model_name]
            power_per_gpu = gpu_model.wattage * DEFAULT_UTILIZATION * DEFAULT_PUE / WATTS_TO_MEGAWATTS
            power_array = np.array([q * 1000 * power_per_gpu for q in qty])
            power_data.append(power_array)

    # Create figure
    fig = go.Figure()

    # Add bar traces for each model
    for power_array, model_name in zip(power_data, models):
        # Determine color based on model name
        base_model = model_name.split(" ")[0] if "(" in model_name else model_name
        color = GPU_COLOR_MAP.get(base_model, "#333333")

        # Add trace for this model
        fig.add_trace(go.Bar(
            x=quarter_labels,
            y=power_array,
            name=model_name,
            marker_color=color
        ))

    # Update layout
    fig.update_layout(
        barmode='stack',
        title="Incremental Power Requirements by Quarter",
        xaxis_title="Quarter",
        yaxis_title="Power (MW)",
        legend_title="GPU Model",
        height=500
    )

    # Convert hover template to show GW
    for trace in fig.data:
        trace.y = trace.y / 1000  # Convert MW to GW
        trace.hovertemplate = "%{y:.2f} GW<extra></extra>"  # Updated hover format with 2 decimal places

    # Display the chart in Streamlit
    st.plotly_chart(fig, use_container_width=True, key="power_requirements_chart")

    # Calculate cumulative power requirements per year
    # First, sum all power data arrays to get total power per quarter
    total_power_per_quarter = np.zeros(len(quarter_labels))
    for power_array in power_data:
        total_power_per_quarter += power_array

    # Assuming the quarters represent 2023, 2024, 2025, 2026
    years = [2023, 2024, 2025, 2026]

    # Group quarters into years
    power_requirements_by_year = [
        total_power_per_quarter[0:4].sum(),   # 2023
        total_power_per_quarter[4:8].sum(),   # 2024
        total_power_per_quarter[8:12].sum(),  # 2025
        total_power_per_quarter[12:].sum()    # 2026
    ]

    # Create a DataFrame for the table (keep numeric columns)
    power_df = pd.DataFrame({
        'Power Requirements (MW)': power_requirements_by_year
    }, index=years)

    # Convert MW to GW and format nicely
    power_df['Power Requirements (GW)'] = power_df['Power Requirements (MW)'].apply(
        lambda x: f"{x/1000:.2f} GW"
    )

    # Calculate and format capex nicely
    power_df['Implied Power Capex'] = power_df['Power Requirements (MW)'].apply(
        lambda x: f"${format_number_to_string(x * 1000 * power_construction_cost)}"  # MW to kW * $/kW
    )

    power_df['Implied Datacenter Capex'] = power_df['Power Requirements (MW)'].apply(
        lambda x: f"${format_number_to_string(x * datacenter_construction_cost)}"  # MW * $/MW
    )

    # Display formatted table
    st.table(power_df.drop(columns=['Power Requirements (MW)']))


def display_gpu_flops_chart(shipment_data):
    """Display the FLOPS chart based on NVIDIA GPU shipments"""
    st.subheader("Quarterly Incremental FLOPS from NVIDIA GPU Shipments")

    quarter_labels = shipment_data['quarter_labels']
    models = shipment_data['models']
    quantities = shipment_data['quantities']

    # Create a dictionary for quick GPU model lookup
    gpu_model_dict = {gpu.name: gpu for gpu in ALL_GPU_LIST}

    # Calculate FLOPS data for each model
    flops_data = []
    for model_name, qty in zip(models, quantities):
        if model_name in gpu_model_dict:
            gpu_model = gpu_model_dict[model_name]

            if hasattr(gpu_model, 'quantized_performance') and gpu_model.quantized_performance.get('fp16'):
                # Get FLOPS per GPU in petaFLOPS
                flops_per_gpu = gpu_model.quantized_performance['fp16']

                # Convert to petaFLOPS if needed
                if isinstance(flops_per_gpu, (int, float)):
                    flops_per_gpu = flops_per_gpu / 1e15  # Convert to petaFLOPS

                # Calculate FLOPS array
                flops_array = np.array([q * 1000 * flops_per_gpu for q in qty])
                flops_data.append(flops_array)
            else:
                # No FLOPS data available
                flops_data.append(np.zeros(len(quarter_labels)))
                st.warning(f"No FLOPS data available for GPU model: {model_name}")

    # Create figure
    fig = go.Figure()

    # Add bar traces for each model
    for flops_array, model_name in zip(flops_data, models):
        # Skip models with no data
        if np.all(flops_array == 0):
            continue

        # Determine color based on model name
        base_model = model_name.split(" ")[0] if "(" in model_name else model_name
        color = GPU_COLOR_MAP.get(base_model, "#333333")

        # Add trace for this model
        fig.add_trace(go.Bar(
            x=quarter_labels,
            y=flops_array,
            name=model_name,
            marker_color=color
        ))

    # Update layout
    fig.update_layout(
        barmode='stack',
        title="Incremental FLOPS by Quarter",
        xaxis_title="Quarter",
        yaxis_title="FLOPS (PetaFLOPS)",
        legend_title="GPU Model",
        height=500
    )

    # Convert PFLOPS to ExaFLOPS for display
    for trace in fig.data:
        trace.y = trace.y / 1000  # Convert PFLOPS to ExaFLOPS

    # Display the chart in Streamlit
    st.plotly_chart(fig, use_container_width=True, key="flops_chart")

def display_rack_density_chart():
    """Display the rack density chart"""
    st.subheader("The Evolution of GPUs is Driving Massive Power Density Increases per Rack")
    st.markdown("""
    Data centers have historically offered long-term residual value, but AI’s rapidly escalating power densities during this period of change could render facilities obsolete in a fraction of that time -- even newly built ones. 
    This is a core consideration when sensitizing expected useful life for datacenters, and thus their financing and valuation models, in a way that has not been seen before for traditional CPU-based datacenters.
    """)
    rack_density_data = {
        "Average US Household": 1.0,
        "CPU - Average Rack (2016)": 6.1,
        "CPU - Average Rack (2024)": 12,
        "GPU - 4x DGX H100 (2022)": 40.8,
        "GPU - B200 NVL36/72 Oberon Rack (2024)": 132,
        "GPU - B300 NVL72 Oberon Rack (2025)": 163,
        "GPU - Rubin Rack Oberon Rack (2026)": 198,
        "GPU - Rubin Ultra Kyber Rack (2027)": 600
    }

    # Create color gradient based on power density
    values = list(rack_density_data.values())
    min_val, max_val = min(values), max(values)

    # Create a color scale from green (low) to red (high)
    def get_color(value):
        # Normalize value between 0 and 1
        normalized = (value - min_val) / (max_val - min_val)

        # Interpolate between green and red
        r = int(255 * normalized)
        g = int(255 * (1 - normalized))
        b = 0

        return f'rgb({r},{g},{b})'

    # Create figure
    fig = go.Figure()

    # Add horizontal bar with dynamic colors
    fig.add_trace(go.Bar(
        y=list(rack_density_data.keys()),
        x=list(rack_density_data.values()),
        orientation='h',
        marker_color=[get_color(value) for value in rack_density_data.values()],
        text=[f"{x:,.0f} kW" for x in rack_density_data.values()],
        textposition='outside',
    ))

    # Update layout with linear x-axis
    fig.update_layout(
        xaxis_title="Power Density (kW/rack)",
        yaxis_title=None,
        height=600,
        margin=dict(l=20, r=20, t=40, b=20),
        showlegend=False,
    )

    st.plotly_chart(fig, use_container_width=True, key="rack_density_chart")

    # Create a DataFrame for the table
    # Table is rendered directly; intermediate DataFrame variable not used
    pd.DataFrame({
        'Rack Density (kW/rack)': [f"{value:,.1f}" for value in rack_density_data.values()]
    }, index=list(rack_density_data.keys()))

### ---- Main Streamlit App ---- ###

def main():
    """Main function to run the Streamlit app"""
    st.set_page_config(page_title="GPU Datacenter Buildout Calculator", layout="wide")
    st.title("GPU Datacenter Buildout Calculator")

    # Get configuration from sidebar
    input_mode, total_budget, power_capacity_gw, power_construction_cost, datacenter_construction_cost, gpu_model, chip_cost, pue, accelerator_ratio, lease_term_years, chip_replacement_years, chip_useful_life, power_useful_life, datacenter_useful_life, chip_gross_margin, margin_adjustment_multiplier = setup_sidebar_config()

    # Use new capex helpers for calculation
    from src.utils.capex_helpers import CapexStartingPoint, calculate_infrastructure

    # Calculate all infrastructure using new helper - support both input modes
    if input_mode == "Power Capacity (GW)":
        infra = calculate_infrastructure(
            starting_point=CapexStartingPoint.POWER_CAPACITY,
            value=power_capacity_gw * 1000,  # Convert GW to MW
            gpu_model=gpu_model,
            chip_vendor_margin=chip_gross_margin,
            pue=pue,
            utilization=DEFAULT_UTILIZATION,
            power_cost_per_kw=power_construction_cost,
            datacenter_cost_per_mw=datacenter_construction_cost
        )
        total_budget = infra.total_capex  # Set for display
    else:  # Total Budget mode
        infra = calculate_infrastructure(
            starting_point=CapexStartingPoint.TOTAL_CAPEX,
            value=total_budget,
            gpu_model=gpu_model,
            chip_vendor_margin=chip_gross_margin,
            pue=pue,
            utilization=DEFAULT_UTILIZATION,
            power_cost_per_kw=power_construction_cost,
            datacenter_cost_per_mw=datacenter_construction_cost
        )

    # Extract values for display (keeping same variable names for compatibility)
    max_gpus = infra.num_gpus
    power_required = infra.power_requirement_mw
    total_compute_cost = infra.chip_capex
    datacenter_cost = infra.datacenter_capex
    power_cost = infra.power_capex
    total_cost = infra.total_capex

    # Get chip costs directly from GPU dataclass (margin adjustments handled in capex calculation)
    base_accelerator_price = gpu_model.ai_accelerator_price
    other_compute_costs = gpu_model.other_compute_costs
    adjusted_accelerator_price = base_accelerator_price * margin_adjustment_multiplier
    adjusted_chip_cost = adjusted_accelerator_price + other_compute_costs

    # Calculate accelerator vs non-accelerator split for display
    accelerator_cost = max_gpus * adjusted_accelerator_price
    non_accelerator_compute_cost = max_gpus * other_compute_costs

    # Display various sections
    display_summary_metrics(max_gpus, power_required, total_budget, gpu_model, adjusted_chip_cost)
    display_cost_breakdowns(total_compute_cost, datacenter_cost, power_cost, total_cost, accelerator_cost, non_accelerator_compute_cost, lease_term_years, chip_replacement_years)

    # Display depreciation analysis (pie chart only, no table)
    display_annual_depreciation(total_compute_cost, datacenter_cost, power_cost, chip_useful_life, power_useful_life, datacenter_useful_life)

    st.markdown("""
    **Note**: This calculator provides high-level estimates for datacenter buildout costs. 
    Actual costs may vary based on location, regulations, and specific requirements.
    The model assumes new construction for both datacenter and power generation infrastructure.
    """)

    # Display GPU shipments chart (reads its own data)
    display_gpu_shipments_chart()

    # Read shipment data for other charts that still need it
    shipment_data = read_gpu_shipment_data()
    display_gpu_power_chart(shipment_data, power_construction_cost, datacenter_construction_cost)
    display_rack_density_chart()
    display_gpu_flops_chart(shipment_data)


# Run the main app
if __name__ == "__main__":
    main()
