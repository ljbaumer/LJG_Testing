"""
Energy Data for AI Data Center Power Source Comparison

This module contains data on various energy sources and their metrics relevant to powering AI data centers.
Data is compiled from credible sources including:
- International Energy Agency (IEA)
- U.S. Energy Information Administration (EIA)
- National Renewable Energy Laboratory (NREL)
- Lazard's Levelized Cost of Energy Analysis
- Various peer-reviewed academic papers
"""

import pandas as pd
import numpy as np

# Define the energy sources and their characteristics
def get_energy_data():
    """
    Returns a pandas DataFrame containing data on various energy sources and their metrics.
    
    Metrics:
    - LCOE: Levelized Cost of Electricity ($/MWh)
    - Power_Density: Power output per unit area (MW/km²)
    - Carbon_Intensity: Lifecycle emissions (gCO2eq/kWh)
    - Capacity_Factor: Average capacity factor (%)
    - Construction_Time: Typical construction time (Years)
    - Operational_Lifespan: Expected operational life (Years)
    - Water_Usage: Water consumption (Gallons/MWh)
    - Land_Use: Land required per unit energy (m²/MWh)
    - Grid_Reliability: Reliability/dispatchability score (1-10)
    - Scalability: Ease of scaling for data center needs (1-10)
    """
    
    # Data dictionary for energy sources and their metrics
    energy_data = {
        'Energy_Source': [
            'Natural Gas (Combined Cycle)',
            'Nuclear (Fission)',
            'Solar PV (Utility Scale)',
            'Wind (Onshore)',
            'Wind (Offshore)',
            'Hydroelectric',
            'Coal',
            'Geothermal',
            'Biomass',
            'Hydrogen Fuel Cells',
            'Small Modular Reactors (SMRs)'
        ],
        'LCOE_Min': [
            45,  # Natural Gas
            131, # Nuclear
            32,  # Solar PV
            26,  # Wind Onshore
            83,  # Wind Offshore
            30,  # Hydroelectric
            65,  # Coal
            58,  # Geothermal
            96,  # Biomass
            103, # Hydrogen
            58   # SMRs
        ],
        'LCOE_Max': [
            74,   # Natural Gas
            204,  # Nuclear
            42,   # Solar PV
            50,   # Wind Onshore
            175,  # Wind Offshore
            200,  # Hydroelectric
            159,  # Coal
            92,   # Geothermal
            230,  # Biomass
            196,  # Hydrogen
            100   # SMRs
        ],
        'Power_Density': [
            4000,  # Natural Gas
            2000,  # Nuclear
            50,    # Solar PV
            5,     # Wind Onshore
            30,    # Wind Offshore
            40,    # Hydroelectric
            3000,  # Coal
            500,   # Geothermal
            300,   # Biomass
            800,   # Hydrogen
            1500   # SMRs
        ],
        'Carbon_Intensity': [
            490,   # Natural Gas
            12,    # Nuclear
            45,    # Solar PV
            11,    # Wind Onshore
            12,    # Wind Offshore
            24,    # Hydroelectric
            820,   # Coal
            38,    # Geothermal
            230,   # Biomass
            10,    # Hydrogen (assuming green hydrogen)
            12     # SMRs
        ],
        'Capacity_Factor': [
            87,    # Natural Gas
            93,    # Nuclear
            25,    # Solar PV
            35,    # Wind Onshore
            45,    # Wind Offshore
            40,    # Hydroelectric
            85,    # Coal
            90,    # Geothermal
            83,    # Biomass
            95,    # Hydrogen
            95     # SMRs
        ],
        'Construction_Time': [
            2,     # Natural Gas
            7,     # Nuclear
            1,     # Solar PV
            1,     # Wind Onshore
            3,     # Wind Offshore
            5,     # Hydroelectric
            4,     # Coal
            3,     # Geothermal
            2,     # Biomass
            2,     # Hydrogen
            4      # SMRs
        ],
        'Operational_Lifespan': [
            25,    # Natural Gas
            60,    # Nuclear
            30,    # Solar PV
            25,    # Wind Onshore
            25,    # Wind Offshore
            80,    # Hydroelectric
            40,    # Coal
            30,    # Geothermal
            25,    # Biomass
            20,    # Hydrogen
            40     # SMRs
        ],
        'Water_Usage': [
            300,   # Natural Gas
            720,   # Nuclear
            20,    # Solar PV
            1,     # Wind Onshore
            1,     # Wind Offshore
            4500,  # Hydroelectric (varies widely)
            1000,  # Coal
            800,   # Geothermal
            500,   # Biomass
            10,    # Hydrogen
            600    # SMRs
        ],
        'Land_Use': [
            2.5,   # Natural Gas
            1.0,   # Nuclear
            45.0,  # Solar PV
            72.0,  # Wind Onshore
            3.0,   # Wind Offshore
            54.0,  # Hydroelectric
            8.0,   # Coal
            7.5,   # Geothermal
            20.0,  # Biomass
            15.0,  # Hydrogen
            1.5    # SMRs
        ],
        'Grid_Reliability': [
            9,     # Natural Gas
            10,    # Nuclear
            3,     # Solar PV
            4,     # Wind Onshore
            5,     # Wind Offshore
            8,     # Hydroelectric
            9,     # Coal
            9,     # Geothermal
            8,     # Biomass
            7,     # Hydrogen
            10     # SMRs
        ],
        'Scalability': [
            8,     # Natural Gas
            6,     # Nuclear
            7,     # Solar PV
            6,     # Wind Onshore
            5,     # Wind Offshore
            4,     # Hydroelectric
            7,     # Coal
            5,     # Geothermal
            6,     # Biomass
            8,     # Hydrogen
            9      # SMRs
        ]
    }
    
    # Create DataFrame
    df = pd.DataFrame(energy_data)
    
    # Calculate average LCOE
    df['LCOE_Avg'] = (df['LCOE_Min'] + df['LCOE_Max']) / 2
    
    return df

# Data sources and references
data_sources = {
    'LCOE': [
        'Lazard\'s Levelized Cost of Energy Analysis (Version 15.0, 2021)',
        'IEA World Energy Outlook 2022',
        'EIA Annual Energy Outlook 2022'
    ],
    'Power_Density': [
        'van Zalk, J., & Behrens, P. (2018). The spatial extent of renewable and non-renewable power generation: A review and meta-analysis of power densities and their application in the U.S. Energy Policy, 123, 83-91.',
        'Smil, V. (2015). Power Density: A Key to Understanding Energy Sources and Uses. MIT Press.'
    ],
    'Carbon_Intensity': [
        'IPCC Fifth Assessment Report (2014)',
        'National Renewable Energy Laboratory (NREL) Life Cycle Assessment Harmonization Project',
        'IEA (2020), Projected Costs of Generating Electricity'
    ],
    'Capacity_Factor': [
        'EIA Electric Power Monthly (2022)',
        'IEA World Energy Outlook 2022'
    ],
    'Construction_Time': [
        'IEA (2020), Projected Costs of Generating Electricity',
        'EIA Capital Cost and Performance Characteristic Estimates for Utility Scale Electric Power Generating Technologies (2020)'
    ],
    'Operational_Lifespan': [
        'NREL Annual Technology Baseline (2021)',
        'IEA (2020), Projected Costs of Generating Electricity'
    ],
    'Water_Usage': [
        'Macknick, J., Newmark, R., Heath, G., & Hallett, K. C. (2012). Operational water consumption and withdrawal factors for electricity generating technologies: a review of existing literature. Environmental Research Letters, 7(4), 045802.',
        'U.S. Department of Energy (2006). Energy Demands on Water Resources: Report to Congress on the Interdependency of Energy and Water.'
    ],
    'Land_Use': [
        'Stevens, L., Anderson, B., Cowan, C., Colton, K., & Johnson, D. (2017). The Footprint of Energy: Land Use of U.S. Electricity Production. Strata.',
        'McDonald, R. I., Fargione, J., Kiesecker, J., Miller, W. M., & Powell, J. (2009). Energy sprawl or energy efficiency: climate policy impacts on natural habitat for the United States of America. PloS one, 4(8), e6802.'
    ],
    'Grid_Reliability': [
        'North American Electric Reliability Corporation (NERC) State of Reliability Report (2021)',
        'Expert assessments based on dispatchability and flexibility characteristics'
    ],
    'Scalability': [
        'Expert assessments based on deployment speed, modularity, and suitability for data center applications',
        'Uptime Institute Global Data Center Survey (2022)'
    ]
}

def get_data_sources():
    """Returns a dictionary of data sources for each metric."""
    return data_sources

# Additional context for AI data centers
ai_datacenter_context = {
    'Power_Requirements': """
    Modern AI data centers have extremely high power requirements, often in the range of 100-500 MW for large facilities.
    The training of advanced AI models like GPT-4 can consume millions of kWh of electricity.
    """,
    
    'Reliability_Needs': """
    AI data centers require 99.999% uptime (five nines) or better, translating to less than 5.26 minutes of downtime per year.
    Power interruptions can damage hardware and disrupt critical AI training runs that may take weeks or months.
    """,
    
    'Growth_Projections': """
    AI computing demand is doubling approximately every 3-4 months as of 2023-2024.
    By 2030, data centers could consume 3-8% of global electricity, up from about 1-2% in 2022.
    """,
    
    'Location_Constraints': """
    Many AI companies are strategically locating data centers near abundant, reliable, and affordable power sources.
    Cooling requirements and network latency also influence location decisions.
    """
}

def get_ai_datacenter_context():
    """Returns context information about AI data center energy requirements."""
    return ai_datacenter_context
