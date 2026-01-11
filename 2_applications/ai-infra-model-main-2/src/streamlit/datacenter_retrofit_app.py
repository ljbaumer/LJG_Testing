"""Interactive Streamlit App for Datacenter Retrofit Cost Analysis"""

# TODO REFACTOR THIS TO BE CLEANER AND MORE MODULAR

import os

import pandas as pd
import plotly.graph_objects as go

import streamlit as st
from src.constants.datacenter_retrofit_scenarios_dataclass import (
    COMPONENT_DESCRIPTIONS,
    RETROFIT_SCENARIOS,
)
from src.models.DatacenterRetrofitModel import DatacenterRetrofitModel

st.set_page_config(
    page_title="Datacenter Retrofit Analysis",
    layout="wide"
)

def main():
    st.title("AI Datacenter End-of-Life Retrofit Calculator")
    st.markdown("""
    **Modeling AI infrastructure obsolescence and 10-year end-of-life retrofit costs**  
    *Core Scenario: 2025 AI Datacenter → 2035 Next-Generation AI Retrofit*
    
    *Based on Schneider Electric Capital Cost Calculator methodology*
    """)

    # Sidebar for scenario selection and editing
    st.sidebar.header("Scenario Analysis")
    selected_scenario_name = st.sidebar.selectbox(
        "Choose Base Scenario:",
        [s.scenario_name for s in RETROFIT_SCENARIOS],
        index=1
    )

    # Editable scenario parameters
    st.sidebar.header("Edit Scenario Parameters")

    # Facility parameters
    st.sidebar.subheader("Facility Settings")
    _ = st.sidebar.number_input("Facility Size (MW)", value=100, min_value=1, max_value=500)
    _ = st.sidebar.number_input("Cost per MW ($)", value=12_000_000, min_value=1_000_000, max_value=50_000_000, step=1_000_000)

    # System percentages
    st.sidebar.subheader("System Allocation")
    power_pct = st.sidebar.slider("Power Systems %", 0.0, 1.0, 0.45, 0.05)
    cooling_pct = st.sidebar.slider("Cooling Systems %", 0.0, 1.0, 0.35, 0.05)
    other_pct = st.sidebar.slider("Other Systems %", 0.0, 1.0, 0.20, 0.05)

    # Normalize percentages to sum to 100%
    total_pct = power_pct + cooling_pct + other_pct
    if total_pct != 1.0:
        st.sidebar.warning(f"⚠️ System percentages sum to {total_pct:.1%}, not 100%")

    # Component replacement factors
    st.sidebar.subheader("Power Component Factors")
    _ = st.sidebar.slider("UPS Replacement %", 0.0, 3.0, 1.1, 0.1)
    _ = st.sidebar.slider("Generator Replacement %", 0.0, 2.0, 0.9, 0.1)
    _ = st.sidebar.slider("Switchgear Replacement %", 0.0, 2.0, 1.3, 0.1)
    _ = st.sidebar.slider("Critical Distribution %", 0.0, 3.0, 1.5, 0.1)

    st.sidebar.subheader("❄️ Cooling Component Factors")
    _ = st.sidebar.slider("In-Building Cooling Replacement %", 0.0, 2.0, 0.9, 0.1)
    _ = st.sidebar.slider("Chiller Replacement %", 0.0, 3.0, 1.4, 0.1)
    _ = st.sidebar.slider("Out-of-Building Heat Rejector %", 0.0, 2.0, 1.2, 0.1)
    _ = st.sidebar.slider("CDU & Liquid Loop % (Tenant Equipment)", 0.0, 10.0, 0.0, 0.5)

    st.sidebar.subheader("Other Component Factors")
    _ = st.sidebar.slider("IT Enclosures %", 0.0, 3.0, 1.2, 0.1)
    _ = st.sidebar.slider("Management & Security %", 0.0, 5.0, 2.0, 0.1)

    # Find selected scenario
    scenario = next(s for s in RETROFIT_SCENARIOS if s.scenario_name == selected_scenario_name)
    calculator = DatacenterRetrofitModel(scenario)

    # Main analysis tabs
    tab1, tab2, tab3 = st.tabs(["Cost Overview", "Rationales & Details", "Key Takeaways"])

    with tab1:
        cost_overview_tab(calculator)

    with tab2:
        rationales_and_details_tab(calculator)

    with tab3:
        key_takeaways_tab()

def cost_overview_tab(calculator):
    """Cost overview with charts and metrics"""
    scenario = calculator.scenario

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Facility Size",
            f"{scenario.facility_size_mw} MW"
        )

    with col2:
        construction_total_m = calculator.construction_costs['total'] / 1_000_000
        _ = (calculator.construction_costs['total'] / scenario.facility_size_mw) / 1_000_000
        st.metric(
            "Construction Cost",
            f"${construction_total_m:.0f}M"
        )

    with col3:
        retrofit_total_m = calculator.retrofit_costs['total'] / 1_000_000
        _ = (calculator.retrofit_costs['total'] / scenario.facility_size_mw) / 1_000_000
        st.metric(
            "Retrofit Cost",
            f"${retrofit_total_m:.0f}M"
        )

    with col4:
        retrofit_pct = calculator.retrofit_costs['total'] / calculator.total_facility_cost
        st.metric(
            "Retrofit as % of Facility",
            f"{retrofit_pct:.1%}"
        )

    # System breakdown charts
    col1, col2 = st.columns(2)

    with col1:
        # Construction costs pie chart
        construction_data = {
            system: sum(components.values())
            for system, components in calculator.construction_costs.items()
            if system != 'total'
        }

        fig = go.Figure(data=[go.Pie(
            values=list(construction_data.values()),
            labels=[s.title() for s in construction_data.keys()]
        )])
        fig.update_layout(title="Construction Cost Breakdown")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Retrofit costs pie chart
        retrofit_data = {
            system: sum(components.values())
            for system, components in calculator.retrofit_costs.items()
            if system != 'total'
        }

        fig = go.Figure(data=[go.Pie(
            values=list(retrofit_data.values()),
            labels=[s.title() for s in retrofit_data.keys()]
        )])
        fig.update_layout(title="Retrofit Cost Breakdown")
        st.plotly_chart(fig, use_container_width=True)

def rationales_and_details_tab(calculator):
    """Combined rationales and component details in one comprehensive tab"""
    scenario = calculator.scenario

    st.header("Component Rationales & Details")

    # Detailed rationales by system
    component_groups = {
        "Power Systems": scenario.power_components,
        "Cooling Systems": scenario.cooling_components,
        "Other Systems": scenario.other_components
    }

    st.subheader("Detailed Component Analysis")

    for system_name, components_obj in component_groups.items():
        st.write(f"### {system_name}")

        for comp_name, spec in components_obj.get_components().items():
            component_title = comp_name.replace("_", " ").title()

            with st.expander(f"{component_title}", expanded=True):
                # Create two columns - one for image, one for text
                col1, col2 = st.columns([1, 2])

                # Check if image exists and display it
                with col1:
                    image_path = f"pictures/{comp_name}.jpg"
                    alt_extensions = [".png", ".jpeg", ".webp", ".gif"]

                    # Try different image extensions
                    if os.path.exists(image_path):
                        st.image(image_path, use_container_width=True)
                    else:
                        # Try alternative extensions
                        found_image = False
                        for ext in alt_extensions:
                            alt_path = f"pictures/{comp_name}{ext}"
                            if os.path.exists(alt_path):
                                st.image(alt_path, use_container_width=True)
                                found_image = True
                                break

                        if not found_image:
                            st.write("*Image not available*")

                # Component description in second column
                with col2:
                    # Get component description
                    description = COMPONENT_DESCRIPTIONS.get(comp_name, {
                        "what_it_is": "Component description not available.",
                        "traditional_vs_ai": "Comparison not available.",
                        "why_it_matters": "Significance not available."
                    })

                    # Display component description
                    st.write("**What is this component?**")
                    st.write(description["what_it_is"])

                    st.write("**Traditional vs AI Infrastructure**")
                    st.write(description["traditional_vs_ai"])

                    st.write("**Why it matters**")
                    st.write(description["why_it_matters"])

                st.divider()

                # Get cost data
                # Normalize system key to match calculator dicts ('power', 'cooling', 'other')
                system_key = (
                    'power' if 'power' in system_name.lower()
                    else ('cooling' if 'cool' in system_name.lower() else 'other')
                )
                construction_cost = calculator.construction_costs[system_key][comp_name]
                retrofit_cost = calculator.retrofit_costs[system_key][comp_name]

                # Display costs
                col_cost1, col_cost2 = st.columns(2)
                with col_cost1:
                    st.metric("Construction Cost", f"${construction_cost/1_000_000:.1f}M")
                with col_cost2:
                    st.metric("Retrofit Cost", f"${retrofit_cost/1_000_000:.1f}M")

                st.divider()

                # Display detailed rationales
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.write("**Construction**")
                    # Calculate percentage of total facility cost
                    system_key = (
                        'power' if 'power' in system_name.lower()
                        else ('cooling' if 'cool' in system_name.lower() else 'other')
                    )
                    if system_key == "power":
                        system_pct = scenario.system_percentages.power_pct
                    elif system_key == "cooling":
                        system_pct = scenario.system_percentages.cooling_pct
                    else:  # other
                        system_pct = scenario.system_percentages.other_pct

                    total_facility_pct = spec.construction.percentage * system_pct
                    st.metric("% of Total Facility", f"{total_facility_pct:.1%}")
                    st.write(spec.construction.rationale)

                with col2:
                    st.write("**Replacement**")
                    st.metric("Cost Factor", f"{spec.replacement.percentage:.0%}")
                    st.write(spec.replacement.rationale)

                with col3:
                    st.write("**Vendor Difference**")
                    vendor_pct = spec.vendor_difference.percentage
                    # Risk label not used downstream; compute vendor_pct only

                    st.metric("Difference", f"{vendor_pct:.0%}")
                    st.write(spec.vendor_difference.rationale)

    # Scenario comparison table
    if len(RETROFIT_SCENARIOS) > 1:
        st.subheader("🆚 Scenario Comparison")
        comparison_data = []

        for scenario_item in RETROFIT_SCENARIOS:
            calc = DatacenterRetrofitModel(scenario_item)
            comparison_data.append({
                "Scenario": scenario_item.scenario_name,
                "Facility Size (MW)": scenario_item.facility_size_mw,
                "Construction Cost": f"${calc.construction_costs['total']/1_000_000:.0f}M",
                "Retrofit Cost": f"${calc.retrofit_costs['total']/1_000_000:.0f}M",
                "Retrofit %": f"{calc.retrofit_costs['total'] / calc.total_facility_cost:.1%}",
                "Cost per MW": f"${calc.retrofit_costs['total'] / scenario_item.facility_size_mw / 1_000_000:.1f}M"
            })

        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, use_container_width=True)

def key_takeaways_tab():
    """Educational insights about AI vs traditional datacenter differences"""
    st.header("Key Insights: AI Datacenter Evolution")
    st.markdown("*Understanding the fundamental differences between AI and traditional datacenters.*")

    # General Modeling Approach
    st.subheader("General Modeling Approach")
    st.markdown("""
    **Facility Power Consistency**
    
    Total facility power capacity remains constant - we replace racks with new ones that redistribute power differently.
    
    *Key Insight: AI datacenter retrofits don't require more total power capacity, just different power distribution patterns within the same facility envelope.*
    """)

    # Infrastructure Evolution
    st.subheader("Infrastructure Evolution")
    with st.expander("**Liquid Cooling Impact on Containment**"):
        st.markdown("""
        Containment aisles for liquid cooling represent a **smaller share of infrastructure**. Retrofit involves removal costs as AI datacenters don't need traditional hot/cold aisle containment.
        
        *Key Insight: AI's shift to liquid cooling eliminates the need for complex airflow management systems that traditional datacenters require.*
        """)

    with st.expander("**Power Distribution Architecture**"):
        st.markdown("""
        **Modern AI datacenters use row-as-pod design** with electrical rooms per 1-2 rows (not per hall) allowing MV→LV and UPS cabinets right next to the load.
        
        *Key Insight: AI datacenters require more granular power distribution due to higher rack densities, fundamentally changing electrical room layouts.*
        """)

    # Component-Level Insights
    st.subheader("Component-Level Insights")

    with st.expander("**Equipment That Doesn't Change**"):
        st.markdown("""
        - **Chillers**: Handle different temperature setpoints as AI workloads evolve (2025→2035)
        - **Generators**: Power requirements remain consistent across AI generations
        - **UPS Systems**: Upstream bulk power systems work regardless of rack-level distribution changes
        
        *Key Insight: Core infrastructure components are technology-agnostic and don't require replacement for AI evolution.*
        """)

    with st.expander("**Heat Rejection Systems**"):
        st.markdown("""
        **External heat rejection** (cooling towers/dry coolers) vs **internal cooling** (precision air units) serve different roles:
        - **Cooling Tower (Evaporative)**: High efficiency but high water usage
        - **Dry Cooler (Convective)**: Zero water usage but less efficient in hot climates
        
        *Key Insight: Heat rejection method choice depends more on geographic location and water availability than on AI vs traditional workloads.*
        """)

    # Financial Implications
    st.subheader("Financial Implications")

    with st.expander("**Normal Wear vs AI-Specific Replacement**"):
        st.markdown("""
        Many components follow **normal datacenter lifecycle costs** rather than AI-specific replacement cycles. The model distinguishes between:
        - Equipment obsolescence due to AI technology evolution
        - Standard wear and tear replacement schedules
        
        *Key Insight: Not all retrofit costs are AI-driven - many are standard datacenter maintenance that would occur regardless of workload type.*
        """)

    with st.expander("**Removal vs Replacement Economics**"):
        st.markdown("""
        Some AI datacenter components represent **removal costs** rather than replacement costs:
        - IT enclosures and containment systems become unnecessary
        - Disposal/removal creates negative value rather than positive investment
        
        *Key Insight: AI datacenter retrofits include subtraction of traditional components, not just addition of new ones.*
        """)

    # Technology Timeline Context
    st.subheader("Technology Timeline Context")

    with st.expander("**2025→2035 AI Evolution**"):
        st.markdown("""
        This model focuses on **AI datacenter evolution over time** rather than traditional-to-AI conversion:
        - How AI datacenters change as technology advances
        - Equipment lifecycle costs within AI-focused facilities
        - Component upgrade cycles driven by AI workload evolution
        
        *Key Insight: The model addresses AI datacenter maturation, not traditional datacenter conversion scenarios.*
        """)

    # Core Considerations
    st.subheader("Core Considerations for Financial Modeling")

    with st.expander("**Financing Structure & Market Boundaries**"):
        st.markdown("""
        **Critical financial distinction**: Components that sometimes get rolled into the financing of the shell must be kept separate and **handled in the ABL (Asset-Based Lending) market** instead.
        
        *Core Consideration: Clear separation between shell financing and equipment financing is essential for accurate cost modeling and risk assessment.*
        """)

    with st.expander("**CDU Impact Reality Check**"):
        st.markdown("""
        **The CDU is changing radically**, but honestly **the cost to the shell is small** - probably just some fittings to support and connect the new CDUs.
        
        *Core Consideration: Dramatic technology changes don't always translate to dramatic infrastructure costs - distinguish between equipment costs vs facility modification costs.*
        """)

if __name__ == "__main__":
    main()
