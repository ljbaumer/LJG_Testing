"""
Sample data for the LCCA tool.
This module provides default values and sample scenarios for demonstration purposes.
"""

# Default building parameters
DEFAULT_BUILDING_PARAMS = {
    "building_height": 300,  # meters
    "num_floors": 60,
    "floor_area": 2500,  # sq. meters
    "total_area": 150000,  # sq. meters
    "facade_type": "Glass Curtain Wall",
    "structural_system": "Steel Frame",
    "hvac_system": "Variable Air Volume (VAV)",
    "interior_quality": "Premium"
}

# Default cost assumptions
DEFAULT_COST_ASSUMPTIONS = {
    "construction_cost_per_sqm": 8000,  # $ per sq. meter
    "annual_inflation_rate": 0.025,  # 2.5%
    "discount_rate": 0.05,  # 5%
    "energy_cost_per_sqm": 50,  # $ per sq. meter per year
    "maintenance_cost_percentage": 0.015,  # 1.5% of construction cost
    "analysis_period": 30  # years
}

# Sample scenarios for comparison
SAMPLE_SCENARIOS = [
    {
        "name": "Base Case",
        "description": "Standard design with glass curtain wall and steel frame",
        "building_params": {
            **DEFAULT_BUILDING_PARAMS
        },
        "cost_assumptions": {
            **DEFAULT_COST_ASSUMPTIONS
        }
    },
    {
        "name": "Eco-Friendly",
        "description": "Energy efficient design with sustainable materials",
        "building_params": {
            **DEFAULT_BUILDING_PARAMS,
            "facade_type": "Precast Concrete",
            "hvac_system": "Chilled Beams"
        },
        "cost_assumptions": {
            **DEFAULT_COST_ASSUMPTIONS,
            "construction_cost_per_sqm": 8500,  # Higher initial cost
            "energy_cost_per_sqm": 35  # Lower energy cost
        }
    },
    {
        "name": "Luxury Design",
        "description": "High-end finishes and premium systems",
        "building_params": {
            **DEFAULT_BUILDING_PARAMS,
            "facade_type": "Stone Veneer",
            "interior_quality": "Ultra Luxury",
            "hvac_system": "Hybrid System"
        },
        "cost_assumptions": {
            **DEFAULT_COST_ASSUMPTIONS,
            "construction_cost_per_sqm": 12000,  # Much higher initial cost
            "maintenance_cost_percentage": 0.02  # Higher maintenance cost
        }
    },
    {
        "name": "Budget Option",
        "description": "Cost-optimized design with standard finishes",
        "building_params": {
            **DEFAULT_BUILDING_PARAMS,
            "facade_type": "Metal Panel",
            "interior_quality": "Standard",
            "structural_system": "Reinforced Concrete"
        },
        "cost_assumptions": {
            **DEFAULT_COST_ASSUMPTIONS,
            "construction_cost_per_sqm": 6500,  # Lower initial cost
            "energy_cost_per_sqm": 60  # Higher energy cost
        }
    }
]

# Sample cost factors for different building components
FACADE_COST_FACTORS = {
    "Glass Curtain Wall": 1.1,
    "Precast Concrete": 0.95,
    "Stone Veneer": 1.2,
    "Metal Panel": 1.0,
    "Mixed": 1.05
}

STRUCTURAL_COST_FACTORS = {
    "Steel Frame": 1.0,
    "Reinforced Concrete": 0.9,
    "Composite": 1.05,
    "Tube System": 1.1
}

HVAC_COST_FACTORS = {
    "Variable Air Volume (VAV)": 1.0,
    "Chilled Beams": 1.2,
    "VRF System": 1.1,
    "Hybrid System": 1.15
}

INTERIOR_COST_FACTORS = {
    "Standard": 1.0,
    "Premium": 1.2,
    "Luxury": 1.5,
    "Ultra Luxury": 2.0
}

# Sample replacement schedules (years)
REPLACEMENT_SCHEDULES = {
    "HVAC": {
        "Variable Air Volume (VAV)": 15,
        "Chilled Beams": 20,
        "VRF System": 15,
        "Hybrid System": 18
    },
    "Facade": {
        "Glass Curtain Wall": {
            "Gaskets": 10,
            "Sealants": 15,
            "Glass Panels": 30
        },
        "Precast Concrete": {
            "Sealants": 15,
            "Panel Repair": 25
        },
        "Stone Veneer": {
            "Sealants": 15,
            "Stone Repair": 20
        },
        "Metal Panel": {
            "Sealants": 15,
            "Panel Replacement": 25
        },
        "Mixed": {
            "Sealants": 15,
            "Component Repair": 20
        }
    },
    "Interior": {
        "Standard": 10,
        "Premium": 12,
        "Luxury": 15,
        "Ultra Luxury": 20
    }
}

# Sample energy efficiency factors (lower is better)
ENERGY_EFFICIENCY_FACTORS = {
    "Facade": {
        "Glass Curtain Wall": 1.1,
        "Precast Concrete": 0.95,
        "Stone Veneer": 1.0,
        "Metal Panel": 1.05,
        "Mixed": 1.0
    },
    "HVAC": {
        "Variable Air Volume (VAV)": 1.0,
        "Chilled Beams": 0.85,
        "VRF System": 0.9,
        "Hybrid System": 0.95
    }
}
