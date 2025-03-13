import numpy as np
import pandas as pd
from scipy import stats

def calculate_lcca(input_data):
    """
    Calculate Life Cycle Cost Analysis for a skyscraper project
    
    Parameters:
    -----------
    input_data : dict
        Dictionary containing all input parameters
        
    Returns:
    --------
    dict
        Dictionary containing all LCCA results
    """
    # Extract input parameters
    building_height = input_data["building_height"]
    num_floors = input_data["num_floors"]
    floor_area = input_data["floor_area"]
    total_area = input_data["total_area"]
    facade_type = input_data["facade_type"]
    structural_system = input_data["structural_system"]
    hvac_system = input_data["hvac_system"]
    interior_quality = input_data["interior_quality"]
    construction_cost_per_sqm = input_data["construction_cost_per_sqm"]
    annual_inflation_rate = input_data["annual_inflation_rate"]
    discount_rate = input_data["discount_rate"]
    energy_cost_per_sqm = input_data["energy_cost_per_sqm"]
    maintenance_cost_percentage = input_data["maintenance_cost_percentage"]
    analysis_period = input_data["analysis_period"]
    
    # Calculate initial construction cost
    initial_construction_cost = calculate_initial_construction_cost(
        total_area, 
        construction_cost_per_sqm,
        facade_type,
        structural_system,
        interior_quality,
        building_height
    )
    
    # Calculate annual costs
    annual_costs = calculate_annual_costs(
        total_area,
        initial_construction_cost,
        energy_cost_per_sqm,
        maintenance_cost_percentage,
        hvac_system,
        facade_type,
        annual_inflation_rate,
        analysis_period
    )
    
    # Calculate replacement costs
    replacement_costs = calculate_replacement_costs(
        total_area,
        hvac_system,
        facade_type,
        interior_quality,
        annual_inflation_rate,
        analysis_period
    )
    
    # Add replacement costs to annual costs
    for year, cost in replacement_costs.items():
        annual_costs["Replacement Costs"][year-1] += cost
    
    # Calculate financing costs (simplified)
    financing_costs = initial_construction_cost * 0.4  # Assume financing costs are 40% of initial cost
    
    # Calculate present value of all costs
    present_value = calculate_present_value(
        initial_construction_cost,
        annual_costs,
        financing_costs,
        discount_rate,
        analysis_period
    )
    
    # Calculate total life cycle cost (nominal)
    total_life_cycle_cost = initial_construction_cost + sum(annual_costs["Operating Costs"]) + \
                           sum(annual_costs["Maintenance Costs"]) + sum(annual_costs["Replacement Costs"]) + \
                           financing_costs
    
    # Prepare cost breakdown
    cost_breakdown = {
        "Initial Construction": initial_construction_cost,
        "Operating Costs": sum(annual_costs["Operating Costs"]),
        "Maintenance Costs": sum(annual_costs["Maintenance Costs"]),
        "Replacement Costs": sum(annual_costs["Replacement Costs"]),
        "Financing Costs": financing_costs
    }
    
    # Prepare results
    results = {
        "initial_cost": initial_construction_cost,
        "total_life_cycle_cost": total_life_cycle_cost,
        "present_value_lcc": present_value,
        "cost_breakdown": cost_breakdown,
        "annual_costs": annual_costs,
        "analysis_period": analysis_period
    }
    
    return results

def calculate_initial_construction_cost(total_area, base_cost_per_sqm, facade_type, structural_system, interior_quality, building_height):
    """Calculate the initial construction cost based on building parameters"""
    # Base cost
    cost = total_area * base_cost_per_sqm
    
    # Adjust for facade type
    facade_factors = {
        "Glass Curtain Wall": 1.1,
        "Precast Concrete": 0.95,
        "Stone Veneer": 1.2,
        "Metal Panel": 1.0,
        "Mixed": 1.05
    }
    cost *= facade_factors.get(facade_type, 1.0)
    
    # Adjust for structural system
    structural_factors = {
        "Steel Frame": 1.0,
        "Reinforced Concrete": 0.9,
        "Composite": 1.05,
        "Tube System": 1.1
    }
    cost *= structural_factors.get(structural_system, 1.0)
    
    # Adjust for interior quality
    interior_factors = {
        "Standard": 1.0,
        "Premium": 1.2,
        "Luxury": 1.5,
        "Ultra Luxury": 2.0
    }
    cost *= interior_factors.get(interior_quality, 1.0)
    
    # Height premium (taller buildings cost more per sqm)
    if building_height > 300:
        cost *= 1.1
    if building_height > 400:
        cost *= 1.15
    
    # Add land cost (simplified for Manhattan)
    land_cost = total_area * 0.3 * base_cost_per_sqm  # Assume land cost is 30% of construction cost
    
    # Add soft costs (design, permits, etc.)
    soft_costs = cost * 0.15  # Assume soft costs are 15% of construction cost
    
    return cost + land_cost + soft_costs

def calculate_annual_costs(total_area, initial_cost, energy_cost_per_sqm, maintenance_percentage, 
                          hvac_system, facade_type, inflation_rate, analysis_period):
    """Calculate annual operating and maintenance costs over the analysis period"""
    # Initialize annual costs dictionary
    annual_costs = {
        "year": list(range(1, analysis_period + 1)),
        "Operating Costs": [0] * analysis_period,
        "Maintenance Costs": [0] * analysis_period,
        "Replacement Costs": [0] * analysis_period
    }
    
    # Base operating cost (energy, utilities, staff)
    base_operating_cost = total_area * energy_cost_per_sqm
    
    # Adjust for HVAC system
    hvac_factors = {
        "Variable Air Volume (VAV)": 1.0,
        "Chilled Beams": 0.85,
        "VRF System": 0.9,
        "Hybrid System": 0.95
    }
    base_operating_cost *= hvac_factors.get(hvac_system, 1.0)
    
    # Adjust for facade type (energy efficiency)
    facade_energy_factors = {
        "Glass Curtain Wall": 1.1,
        "Precast Concrete": 0.95,
        "Stone Veneer": 1.0,
        "Metal Panel": 1.05,
        "Mixed": 1.0
    }
    base_operating_cost *= facade_energy_factors.get(facade_type, 1.0)
    
    # Base maintenance cost
    base_maintenance_cost = initial_cost * maintenance_percentage
    
    # Calculate costs for each year with inflation
    for i in range(analysis_period):
        inflation_factor = (1 + inflation_rate) ** i
        annual_costs["Operating Costs"][i] = base_operating_cost * inflation_factor
        annual_costs["Maintenance Costs"][i] = base_maintenance_cost * inflation_factor
    
    return annual_costs

def calculate_replacement_costs(total_area, hvac_system, facade_type, interior_quality, inflation_rate, analysis_period):
    """Calculate replacement costs for building systems over the analysis period"""
    replacement_costs = {}
    
    # HVAC system replacement
    hvac_replacement_year = 15  # Typical HVAC lifespan
    if hvac_replacement_year <= analysis_period:
        hvac_base_cost = total_area * 200  # $200 per sqm for HVAC replacement
        
        # Adjust for HVAC system type
        hvac_cost_factors = {
            "Variable Air Volume (VAV)": 1.0,
            "Chilled Beams": 1.2,
            "VRF System": 1.1,
            "Hybrid System": 1.15
        }
        hvac_cost = hvac_base_cost * hvac_cost_factors.get(hvac_system, 1.0)
        
        # Apply inflation
        inflation_factor = (1 + inflation_rate) ** (hvac_replacement_year - 1)
        replacement_costs[hvac_replacement_year] = hvac_cost * inflation_factor
    
    # Facade maintenance/replacement
    if facade_type == "Glass Curtain Wall":
        # Curtain wall gasket replacement at year 10
        gasket_year = 10
        if gasket_year <= analysis_period:
            gasket_cost = total_area * 50  # $50 per sqm
            inflation_factor = (1 + inflation_rate) ** (gasket_year - 1)
            replacement_costs[gasket_year] = replacement_costs.get(gasket_year, 0) + gasket_cost * inflation_factor
    
    # Interior renovation
    interior_renovation_year = 10  # Typical renovation cycle
    if interior_renovation_year <= analysis_period:
        interior_base_cost = total_area * 300  # $300 per sqm for standard renovation
        
        # Adjust for interior quality
        interior_cost_factors = {
            "Standard": 1.0,
            "Premium": 1.3,
            "Luxury": 1.8,
            "Ultra Luxury": 2.5
        }
        interior_cost = interior_base_cost * interior_cost_factors.get(interior_quality, 1.0)
        
        # Apply inflation
        inflation_factor = (1 + inflation_rate) ** (interior_renovation_year - 1)
        replacement_costs[interior_renovation_year] = replacement_costs.get(interior_renovation_year, 0) + interior_cost * inflation_factor
    
    # Second interior renovation if analysis period is long enough
    second_renovation_year = 20
    if second_renovation_year <= analysis_period:
        interior_base_cost = total_area * 300
        interior_cost_factors = {
            "Standard": 1.0,
            "Premium": 1.3,
            "Luxury": 1.8,
            "Ultra Luxury": 2.5
        }
        interior_cost = interior_base_cost * interior_cost_factors.get(interior_quality, 1.0)
        inflation_factor = (1 + inflation_rate) ** (second_renovation_year - 1)
        replacement_costs[second_renovation_year] = replacement_costs.get(second_renovation_year, 0) + interior_cost * inflation_factor
    
    return replacement_costs

def calculate_present_value(initial_cost, annual_costs, financing_costs, discount_rate, analysis_period):
    """Calculate the present value of all life cycle costs"""
    # Initial cost is already in present value
    pv = initial_cost + financing_costs
    
    # Calculate present value of annual costs
    for i in range(analysis_period):
        year_cost = annual_costs["Operating Costs"][i] + annual_costs["Maintenance Costs"][i] + annual_costs["Replacement Costs"][i]
        pv += year_cost / ((1 + discount_rate) ** (i + 1))
    
    return pv
