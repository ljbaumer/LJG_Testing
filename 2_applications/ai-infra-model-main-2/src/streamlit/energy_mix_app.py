import numpy as np
import pandas as pd
import plotly.graph_objects as go

import streamlit as st
from src.constants.common import (
    DEFAULT_HEAT_RATE_BTU_PER_KWH,
    DEFAULT_LNG_PRICE_PER_MMBTU,
    HOURS_PER_MONTH,
)
from src.constants.gpu_cloud_scenarios_dataclass import SCENARIOS
from src.models.GreenWattsModel import GreenWattsModel
from src.utils.gpu_cloud_helpers import calculate_total_power_per_gpu
from src.utils.nat_gas_helpers import convert_kwh_to_mmbtu

st.set_page_config(page_title="Energy Mix & Green Watts Calculator", layout="wide")


def main():
    st.title("Energy Mix & Green Watts Calculator")
    st.caption("Break down required energy by source, cost, and emissions. Calculate green watts from renewable sources. Optionally import load from a GPU Cloud scenario.")

    # Sidebar: Load Definition
    st.sidebar.header("Load Definition")
    use_scenario = st.sidebar.checkbox("Import from GPU Cloud Scenario", value=True)

    if use_scenario:
        selected_name = st.sidebar.selectbox("Select Scenario", options=list(SCENARIOS.keys()), index=list(SCENARIOS.keys()).index("Oracle-OpenAI $300B") if "Oracle-OpenAI $300B" in SCENARIOS else 0)
        scenario = SCENARIOS[selected_name]
        # Compute required average power (kW)
        watts_per_gpu = calculate_total_power_per_gpu(
            wattage=scenario.gpu_model.wattage,
            utilization=scenario.gpu_utilization,
            pue=scenario.pue,
        )
        power_kw = (scenario.gpu_count * watts_per_gpu) / 1000.0
        st.sidebar.write(f"Derived Load: {power_kw/1000:.2f} MW")
    else:
        power_mw = st.sidebar.number_input("Average Load (MW)", min_value=0.1, value=100.0, step=1.0)
        power_kw = power_mw * 1000.0

    period = st.sidebar.selectbox("Time Horizon", ["Annual (8760h)", "Custom months"], index=0)
    if period == "Annual (8760h)":
        hours = 365.0 * 24.0
        months = 12.0
    else:
        months = st.sidebar.number_input("Months", min_value=1.0, value=12.0, step=1.0)
        hours = months * HOURS_PER_MONTH

    # Initialize capacity factors with default values
    # These will be updated in the advanced section if user changes them
    cf_solar = 0.25
    cf_wind = 0.40
    cf_other = 0.50

    total_kwh = power_kw * hours

    # Sidebar: Energy Mix
    st.sidebar.header("Energy Mix (%)")
    grid_pct = st.sidebar.slider("Grid", 0, 100, 40)
    gas_pct = st.sidebar.slider("On-site Natural Gas", 0, 100, 40)
    solar_pct = st.sidebar.slider("Solar", 0, 100, 10)
    wind_pct = st.sidebar.slider("Wind", 0, 100, 10)
    other_pct = st.sidebar.slider("Other", 0, 100, 0)
    total_pct = grid_pct + gas_pct + solar_pct + wind_pct + other_pct
    if total_pct != 100:
        st.sidebar.warning(f"Energy mix sums to {total_pct}%. Values will be normalized.")
    # Normalize to 1.0
    mix = np.array([grid_pct, gas_pct, solar_pct, wind_pct, other_pct], dtype=float)
    mix = mix / max(total_pct, 1)

    # Sidebar: Source Parameters
    st.sidebar.header("Source Parameters")
    st.sidebar.subheader("Grid")
    grid_price = st.sidebar.number_input("Grid Price ($/kWh)", min_value=0.0, value=0.10, step=0.01)
    grid_ef = st.sidebar.number_input("Grid Emissions (kg CO₂/kWh)", min_value=0.0, value=0.40, step=0.05)

    st.sidebar.subheader("On-site Natural Gas")
    heat_rate = st.sidebar.number_input("Heat Rate (BTU/kWh)", min_value=1000.0, value=float(DEFAULT_HEAT_RATE_BTU_PER_KWH), step=100.0)
    gas_price = st.sidebar.number_input("Fuel Price ($/MMBtu)", min_value=0.0, value=float(DEFAULT_LNG_PRICE_PER_MMBTU), step=0.5)
    gas_ef_per_mmbtu = st.sidebar.number_input("Emissions (kg CO₂/MMBtu)", min_value=0.0, value=53.06, step=0.5)

    st.sidebar.subheader("Solar")
    solar_price = st.sidebar.number_input("Solar Price ($/kWh)", min_value=0.0, value=0.035, step=0.005)
    solar_ef = st.sidebar.number_input("Solar Emissions (kg CO₂/kWh)", min_value=0.0, value=0.0, step=0.01)

    st.sidebar.subheader("Wind")
    wind_price = st.sidebar.number_input("Wind Price ($/kWh)", min_value=0.0, value=0.030, step=0.005)
    wind_ef = st.sidebar.number_input("Wind Emissions (kg CO₂/kWh)", min_value=0.0, value=0.0, step=0.01)

    st.sidebar.subheader("Other")
    other_price = st.sidebar.number_input("Other Price ($/kWh)", min_value=0.0, value=0.12, step=0.01)
    other_ef = st.sidebar.number_input("Other Emissions (kg CO₂/kWh)", min_value=0.0, value=0.50, step=0.05)

    labels = ["Grid", "Natural Gas", "Solar", "Wind", "Other"]
    prices = np.array([grid_price, None, solar_price, wind_price, other_price], dtype=object)
    efs = np.array([grid_ef, None, solar_ef, wind_ef, other_ef], dtype=object)

    # Compute energy by source
    energy_kwh = total_kwh * mix

    # Costs
    costs = np.zeros_like(energy_kwh)
    # Grid, solar, wind, other (simple $/kWh)
    for idx in [0, 2, 3, 4]:
        costs[idx] = energy_kwh[idx] * float(prices[idx])
    # Natural gas (fuel-based)
    gas_mmbtu = convert_kwh_to_mmbtu(energy_kwh[1], heat_rate) if energy_kwh[1] > 0 else 0.0
    costs[1] = gas_mmbtu * gas_price

    # Emissions (kg CO2)
    emissions_kg = np.zeros_like(energy_kwh)
    for idx in [0, 2, 3, 4]:
        emissions_kg[idx] = energy_kwh[idx] * float(efs[idx])
    emissions_kg[1] = gas_mmbtu * gas_ef_per_mmbtu

    # Summaries
    total_cost = costs.sum()
    total_emissions_tonnes = emissions_kg.sum() / 1000.0
    total_mwh = total_kwh / 1000.0
    blended_cost_per_mwh = (total_cost / total_mwh) if total_mwh > 0 else 0.0

    # Table (numeric, format at render time)
    df = pd.DataFrame({
        'Source': labels,
        'Share (%)': (mix * 100.0),
        'Energy (MWh)': energy_kwh / 1000.0,
        'Cost ($)': costs,
        'Cost Share (%)': (costs / total_cost * 100.0) if total_cost > 0 else 0.0,
        'CO₂ (t)': emissions_kg / 1000.0,
        'CO₂ Share (%)': (emissions_kg / emissions_kg.sum() * 100.0) if emissions_kg.sum() > 0 else 0.0,
    })

    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Energy", f"{total_mwh:,.0f} MWh")
    with col2:
        st.metric("Total Cost", f"${total_cost:,.0f}")
    with col3:
        st.metric("Blended Cost", f"${blended_cost_per_mwh:,.0f} / MWh")
    with col4:
        st.metric("Total CO₂", f"{total_emissions_tonnes:,.0f} t")

    # Table (numeric, format at render time)
    df = pd.DataFrame({
        'Source': labels,
        'Share (%)': (mix * 100.0),
        'Energy (MWh)': energy_kwh / 1000.0,
        'Cost ($)': costs,
        'Cost Share (%)': (costs / total_cost * 100.0) if total_cost > 0 else 0.0,
        'CO₂ (t)': emissions_kg / 1000.0,
        'CO₂ Share (%)': (emissions_kg / emissions_kg.sum() * 100.0) if emissions_kg.sum() > 0 else 0.0,
    })

    st.dataframe(
        df,
        use_container_width=True,
        column_config={
            'Share (%)': st.column_config.NumberColumn(format='%.1f%%'),
            'Energy (MWh)': st.column_config.NumberColumn(format='%,.0f'),
            'Cost ($)': st.column_config.NumberColumn(format='$%,.0f'),
            'Cost Share (%)': st.column_config.NumberColumn(format='%.1f%%'),
            'CO₂ (t)': st.column_config.NumberColumn(format='%,.0f'),
            'CO₂ Share (%)': st.column_config.NumberColumn(format='%.1f%%'),
        }
    )

    # Charts
    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        fig_energy = go.Figure(data=[go.Pie(labels=labels, values=energy_kwh, hole=0.3)])
        fig_energy.update_layout(title_text="Energy Share", height=400)
        st.plotly_chart(fig_energy, use_container_width=True)
    with c2:
        fig_cost = go.Figure(data=[go.Pie(labels=labels, values=costs, hole=0.3)])
        fig_cost.update_layout(title_text="Cost Share", height=400)
        st.plotly_chart(fig_cost, use_container_width=True)
    with c3:
        fig_em = go.Figure(data=[go.Bar(x=labels, y=emissions_kg / 1000.0)])
        fig_em.update_layout(title_text="Emissions (t)", height=400, yaxis_title="tCO₂")
        st.plotly_chart(fig_em, use_container_width=True)

    # Advanced: Installed capacity estimation
    with st.expander("Installed Capacity (Optional)"):
        st.caption("Estimate required installed capacity by source using capacity factors. Applies to energy shares, independent of cost model.")
        cf_solar = st.number_input("Solar Capacity Factor", min_value=0.01, max_value=1.0, value=0.25, step=0.01)
        cf_wind = st.number_input("Wind Capacity Factor", min_value=0.01, max_value=1.0, value=0.40, step=0.01)
        cf_other = st.number_input("Other Capacity Factor", min_value=0.01, max_value=1.0, value=0.50, step=0.01)

        hours_per_year = 365.0 * 24.0
        # Energy (MWh) per source
        energy_mwh = energy_kwh / 1000.0
        installed = {
            'Solar (MW)': (energy_mwh[2] / (cf_solar * hours_per_year)) if energy_mwh[2] > 0 else 0.0,
            'Wind (MW)': (energy_mwh[3] / (cf_wind * hours_per_year)) if energy_mwh[3] > 0 else 0.0,
            'Other (MW)': (energy_mwh[4] / (cf_other * hours_per_year)) if energy_mwh[4] > 0 else 0.0,
        }
        cap_df = pd.DataFrame(installed, index=["Required Capacity"]).T
        st.dataframe(cap_df, use_container_width=True, column_config={"Required Capacity": st.column_config.NumberColumn(format="%,.0f MW")})
        
        # Add Green Watts capacity requirements
        st.markdown("### Green Watts Capacity Requirements")
        gw_capacity = {
            'Solar (MW)': installed_capacity['solar_mw'],
            'Wind (MW)': installed_capacity['wind_mw'],
            'Other Renewable (MW)': installed_capacity['other_renewable_mw'],
        }
        gw_cap_df = pd.DataFrame(gw_capacity, index=["Required Capacity"]).T
        st.dataframe(gw_cap_df, use_container_width=True, column_config={"Required Capacity": st.column_config.NumberColumn(format="%,.0f MW")})

    # Green Watts calculations using updated capacity factors
    energy_mix_dict = {
        'solar': solar_pct / 100.0,
        'wind': wind_pct / 100.0,
        'other': other_pct / 100.0
    }
    
    green_watts_model = GreenWattsModel(
        total_power_kw=power_kw,
        energy_mix=energy_mix_dict,
        solar_capacity_factor=cf_solar,
        wind_capacity_factor=cf_wind
    )
    
    green_watts_metrics = green_watts_model.calculate_green_watts()
    carbon_offset_metrics = green_watts_model.get_carbon_offset(grid_ef)
    installed_capacity = green_watts_model.calculate_installed_capacity()

    # Green Watts Metrics
    st.markdown("### Green Watts Metrics")
    gw_col1, gw_col2, gw_col3, gw_col4 = st.columns(4)
    with gw_col1:
        st.metric("Green Watts", f"{green_watts_metrics['total_green_watts']/1_000_000:,.0f} MW")
    with gw_col2:
        st.metric("Green %", f"{green_watts_metrics['green_percentage']*100:.1f}%")
    with gw_col3:
        st.metric("CO₂ Offset", f"{carbon_offset_metrics['annual_co2_offset_tonnes']:,.0f} t")
    with gw_col4:
        st.metric("Equivalent Cars", f"{carbon_offset_metrics['equivalent_cars']:,.0f}")

    # Green Watts Visualization
    st.markdown("### Green Watts Breakdown")
    green_sources = ['Solar', 'Wind', 'Other Renewable']
    green_values = [
        green_watts_metrics['solar_watts'] / 1_000_000,  # Convert to MW
        green_watts_metrics['wind_watts'] / 1_000_000,   # Convert to MW
        green_watts_metrics['other_renewable_watts'] / 1_000_000  # Convert to MW
    ]
    
    fig_gw = go.Figure(data=[go.Bar(x=green_sources, y=green_values)])
    fig_gw.update_layout(
        title_text="Green Watts by Source (MW)",
        height=400,
        yaxis_title="MW"
    )
    st.plotly_chart(fig_gw, use_container_width=True)


if __name__ == "__main__":
    main()

