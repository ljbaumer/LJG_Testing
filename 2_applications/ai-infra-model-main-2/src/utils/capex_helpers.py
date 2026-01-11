"""Simple stateless helpers for calculating infrastructure requirements from various starting points.

No classes, no state management - just pure functions that take inputs and return calculated values.
"""

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.constants.value_chain_depreciation_schedules import CapexDepreciationSchedule

from src.constants.gpu_dataclass import GPUDataclass

# Constants
DEFAULT_POWER_COST_PER_KW = 2500  # $/kW
DEFAULT_DATACENTER_COST_PER_MW = 15_000_000  # $/MW
DEFAULT_PUE = 1.2
DEFAULT_UTILIZATION = 0.8


class CapexStartingPoint(Enum):
    """Different ways to start the capex calculation."""
    NUM_CHIPS = "num_chips"
    POWER_CAPACITY = "power_capacity"  # MW
    TOTAL_CAPEX = "total_capex"
    CHIP_CAPEX = "chip_capex"


@dataclass
class InfrastructureRequirements:
    """Container for all calculated infrastructure values."""
    num_gpus: int
    chip_capex: float
    datacenter_capex: float
    power_capex: float
    total_capex: float
    power_requirement_mw: float

    @property
    def power_requirement_gw(self) -> float:
        return self.power_requirement_mw / 1000


def calculate_infrastructure(
    starting_point: CapexStartingPoint,
    value: float,
    gpu_model: GPUDataclass,
    chip_vendor_margin: float = 0.75,
    pue: float = DEFAULT_PUE,
    utilization: float = DEFAULT_UTILIZATION,
    power_cost_per_kw: float = DEFAULT_POWER_COST_PER_KW,
    datacenter_cost_per_mw: float = DEFAULT_DATACENTER_COST_PER_MW,
) -> InfrastructureRequirements:
    """
    Main router function - calculates all infrastructure based on starting point.

    Args:
        starting_point: Which input type we're starting from
        value: The input value (interpretation depends on starting_point)
        gpu_model: GPU model from gpu_dataclass
        chip_vendor_margin: Chip vendor gross margin (affects pricing)
        pue: Power Usage Effectiveness
        utilization: GPU utilization rate
        power_cost_per_kw: Cost to build power generation
        datacenter_cost_per_mw: Cost to build datacenter

    Returns:
        InfrastructureRequirements with all calculated values

    Example:
        from src.constants.gpu_dataclass import ALL_GPU_LIST
        gpu = next(g for g in ALL_GPU_LIST if g.name == "VR200")
        infra = calculate_infrastructure(
            CapexStartingPoint.CHIP_CAPEX,
            50_000_000_000,  # $50B
            gpu
        )
    """
    # Route to appropriate helper
    if starting_point == CapexStartingPoint.NUM_CHIPS:
        return _from_num_chips(
            int(value), gpu_model, chip_vendor_margin,
            pue, utilization, power_cost_per_kw, datacenter_cost_per_mw
        )
    elif starting_point == CapexStartingPoint.CHIP_CAPEX:
        return _from_chip_capex(
            value, gpu_model, chip_vendor_margin,
            pue, utilization, power_cost_per_kw, datacenter_cost_per_mw
        )
    elif starting_point == CapexStartingPoint.POWER_CAPACITY:
        return _from_power_capacity(
            value, gpu_model, chip_vendor_margin,
            pue, utilization, power_cost_per_kw, datacenter_cost_per_mw
        )
    elif starting_point == CapexStartingPoint.TOTAL_CAPEX:
        return _from_total_capex(
            value, gpu_model, chip_vendor_margin,
            pue, utilization, power_cost_per_kw, datacenter_cost_per_mw
        )
    else:
        raise ValueError(f"Unsupported starting point: {starting_point}")


def _calculate_chip_cost(gpu_model: GPUDataclass, chip_vendor_margin: float) -> float:
    """Calculate adjusted chip cost with margin."""
    from src.constants.value_chain_depreciation_schedules import CapexDepreciationSchedule

    _, margin_adjustment_multiplier = CapexDepreciationSchedule.calculate_chip_price_multiplier(
        chip_vendor_margin
    )

    # Get prices from gpu_model
    accelerator_price = getattr(gpu_model, 'ai_accelerator_price', getattr(gpu_model, 'price', 0))
    other_costs = getattr(gpu_model, 'other_compute_costs', 0)

    adjusted_accelerator_price = accelerator_price * margin_adjustment_multiplier
    return adjusted_accelerator_price + other_costs



def _from_num_chips(
    num_chips: int,
    gpu_model: GPUDataclass,
    chip_vendor_margin: float,
    pue: float,
    utilization: float,
    power_cost_per_kw: float,
    datacenter_cost_per_mw: float
) -> InfrastructureRequirements:
    """
    Calculate everything from number of chips.

    Math:
    - chip_capex = num_chips * chip_cost
    - power_mw = num_chips * gpu_wattage * utilization * pue / 1M
    - datacenter_capex = power_mw * datacenter_cost_per_mw
    - power_capex = power_mw * 1000 * power_cost_per_kw
    """
    chip_cost = _calculate_chip_cost(gpu_model, chip_vendor_margin)
    chip_capex = num_chips * chip_cost

    # Power calculations - infrastructure must be sized for peak (100% utilization)
    # Don't use utilization factor here - datacenters are built for max capacity
    power_per_gpu_watts = gpu_model.wattage * pue
    total_power_watts = power_per_gpu_watts * num_chips
    power_requirement_mw = total_power_watts / 1_000_000

    # Infrastructure capex
    datacenter_capex = power_requirement_mw * datacenter_cost_per_mw
    power_capex = power_requirement_mw * 1000 * power_cost_per_kw
    total_capex = chip_capex + datacenter_capex + power_capex

    return InfrastructureRequirements(
        num_gpus=num_chips,
        chip_capex=chip_capex,
        datacenter_capex=datacenter_capex,
        power_capex=power_capex,
        total_capex=total_capex,
        power_requirement_mw=power_requirement_mw
    )


def _from_chip_capex(
    chip_capex: float,
    gpu_model: GPUDataclass,
    chip_vendor_margin: float,
    pue: float,
    utilization: float,
    power_cost_per_kw: float,
    datacenter_cost_per_mw: float
) -> InfrastructureRequirements:
    """
    Calculate everything from chip capex budget.

    Math:
    - num_chips = chip_capex / chip_cost
    - Then same as _from_num_chips
    """
    chip_cost = _calculate_chip_cost(gpu_model, chip_vendor_margin)
    num_chips = int(chip_capex / chip_cost)

    # Reuse num_chips calculation but preserve exact chip_capex
    result = _from_num_chips(
        num_chips, gpu_model, chip_vendor_margin,
        pue, utilization, power_cost_per_kw, datacenter_cost_per_mw
    )

    # Override to use exact chip_capex passed in
    result.chip_capex = chip_capex
    result.total_capex = chip_capex + result.datacenter_capex + result.power_capex

    return result


def _from_power_capacity(
    power_mw: float,
    gpu_model: GPUDataclass,
    chip_vendor_margin: float,
    pue: float,
    utilization: float,
    power_cost_per_kw: float,
    datacenter_cost_per_mw: float
) -> InfrastructureRequirements:
    """
    Calculate everything from power capacity in MW.

    Math:
    - num_chips = power_mw / (gpu_wattage * utilization * pue / 1M)
    - chip_capex = num_chips * chip_cost
    - datacenter_capex = power_mw * datacenter_cost_per_mw
    - power_capex = power_mw * 1000 * power_cost_per_kw
    """
    chip_cost = _calculate_chip_cost(gpu_model, chip_vendor_margin)

    # Calculate num chips from power - infrastructure sized for peak
    power_per_gpu_watts = gpu_model.wattage * pue
    power_per_gpu_mw = power_per_gpu_watts / 1_000_000
    num_chips = int(power_mw / power_per_gpu_mw)

    # Calculate capex
    chip_capex = num_chips * chip_cost
    datacenter_capex = power_mw * datacenter_cost_per_mw
    power_capex = power_mw * 1000 * power_cost_per_kw
    total_capex = chip_capex + datacenter_capex + power_capex

    return InfrastructureRequirements(
        num_gpus=num_chips,
        chip_capex=chip_capex,
        datacenter_capex=datacenter_capex,
        power_capex=power_capex,
        total_capex=total_capex,
        power_requirement_mw=power_mw
    )


def _from_total_capex(
    total_budget: float,
    gpu_model: GPUDataclass,
    chip_vendor_margin: float,
    pue: float,
    utilization: float,
    power_cost_per_kw: float,
    datacenter_cost_per_mw: float
) -> InfrastructureRequirements:
    """
    Calculate optimal infrastructure given total budget constraint.

    Uses binary search to find maximum chips within budget.

    Math:
    - Find n such that: n*chip_cost + n*power_per_chip*(dc_cost + power_cost) <= budget
    - This simplifies to: n <= budget / (chip_cost + power_per_chip * infra_cost_per_mw)
    """
    chip_cost = _calculate_chip_cost(gpu_model, chip_vendor_margin)

    # Calculate cost per chip including infrastructure - sized for peak power
    power_per_gpu_watts = gpu_model.wattage * pue
    power_per_gpu_mw = power_per_gpu_watts / 1_000_000
    infra_cost_per_mw = datacenter_cost_per_mw + (power_cost_per_kw * 1000)
    total_cost_per_chip = chip_cost + (power_per_gpu_mw * infra_cost_per_mw)

    # Direct calculation (no binary search needed)
    num_chips = int(total_budget / total_cost_per_chip)

    # Calculate final values
    return _from_num_chips(
        num_chips, gpu_model, chip_vendor_margin,
        pue, utilization, power_cost_per_kw, datacenter_cost_per_mw
    )


def nvidia_revenue_to_total_chip_capex(
    nvidia_revenue: float,
    nvidia_compute_share: float = None
) -> float:
    """Convert NVIDIA revenue to total chip capex including other compute costs.

    Args:
        nvidia_revenue: NVIDIA's revenue from chip sales
        nvidia_compute_share: NVIDIA's share of total compute (default uses constant)

    Returns:
        Total chip capex including NVIDIA + other compute costs

    Example:
        # $160B NVIDIA revenue with 75% share → $213B total chip capex
        total = nvidia_revenue_to_total_chip_capex(160e9)
    """
    if nvidia_compute_share is None:
        from src.constants.value_chain_depreciation_schedules import NVIDIA_COMPUTE_SHARE
        nvidia_compute_share = NVIDIA_COMPUTE_SHARE

    return nvidia_revenue / nvidia_compute_share


def create_multiyear_capex_schedule(
    starting_point: CapexStartingPoint,
    annual_value: float,
    gpu_model: GPUDataclass,
    start_year: int,
    num_years: int,
    growth_rate: float = 0.0,
    chip_useful_life: int = 5,
    datacenter_useful_life: int = 20,
    power_useful_life: int = 25,
    **kwargs  # Pass through to calculate_infrastructure
) -> 'CapexDepreciationSchedule':
    """
    Generate multi-year depreciation schedule from any starting point.

    Args:
        starting_point: Type of input value
        annual_value: Annual investment (interpretation depends on starting_point)
        gpu_model: GPU model to use
        start_year: First year of investment
        num_years: Number of years to model
        growth_rate: Annual growth rate (0.1 = 10%)
        **kwargs: Additional params for calculate_infrastructure

    Returns:
        CapexDepreciationSchedule for use in ValueChainModel
    """
    import pandas as pd

    from src.constants.value_chain_depreciation_schedules import CapexDepreciationSchedule

    years = list(range(start_year, start_year + num_years))
    chip_investments = []
    datacenter_investments = []
    power_investments = []

    for i in range(num_years):
        # Apply growth rate
        adjusted_value = annual_value * ((1 + growth_rate) ** i)

        # Calculate infrastructure for this year
        infra = calculate_infrastructure(
            starting_point, adjusted_value, gpu_model, **kwargs
        )

        chip_investments.append(infra.chip_capex)
        datacenter_investments.append(infra.datacenter_capex)
        power_investments.append(infra.power_capex)

    # Create DataFrame
    capex_df = pd.DataFrame({
        'chips': chip_investments,
        'datacenter': datacenter_investments,
        'power': power_investments
    }, index=years)

    useful_life_series = pd.Series({
        'chips': chip_useful_life,
        'datacenter': datacenter_useful_life,
        'power': power_useful_life
    })

    return CapexDepreciationSchedule(
        depreciation_accounting_schedule=capex_df,
        useful_life_series=useful_life_series
    )
