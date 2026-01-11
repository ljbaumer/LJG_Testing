# Pure functions for GPU cloud calculations.
from typing import Dict


def calculate_max_capacity_from_budget(
    total_budget: float,
    price_per_chip: float,
    installation_cost_per_gpu: float,
    datacenter_construction_cost_per_mw: float,
    electric_generation_construction_cost_per_kw: float,
    gpu_wattage: float,
    pue: float
) -> int:
    """
    Calculate maximum number of GPUs given a total budget.
    
    Args:
        total_budget: Total available budget in dollars
        price_per_chip: Cost per GPU chip in dollars
        installation_cost_per_gpu: Installation cost per GPU in dollars
        datacenter_construction_cost_per_mw: Cost per MW of datacenter construction
        electric_generation_construction_cost_per_kw: Cost per kW of power generation
        gpu_wattage: Power consumption of each GPU in watts
        pue: Power Usage Effectiveness
        
    Returns:
        int: Maximum number of GPUs that can be purchased within budget
    """
    # Binary search to find maximum number of GPUs
    min_gpus = 1
    max_gpus = 100_000_000  # Set a reasonable upper limit

    while min_gpus <= max_gpus:
        gpu_count = (min_gpus + max_gpus) // 2

        # Calculate hardware costs including installation
        hardware_costs = calculate_upfront_compute_hardware_cost(
            gpu_count=gpu_count,
            nvda_gpu_price=price_per_chip,
            other_compute_hw_price=0,  # This will be added separately
            installation_cost_per_gpu=installation_cost_per_gpu
        )
        gpu_cost = hardware_costs["total_cost"]

        # Calculate power requirements and infrastructure costs
        total_power_watts = gpu_wattage * pue * gpu_count
        total_power_mw = total_power_watts / 1_000_000  # Convert watts to MW
        total_power_kw = total_power_watts / 1_000      # Convert watts to kW

        datacenter_cost = total_power_mw * datacenter_construction_cost_per_mw
        power_cost = total_power_kw * electric_generation_construction_cost_per_kw

        # Total cost including all components
        total_cost = gpu_cost + datacenter_cost + power_cost

        # If we're within 1% of the budget, we've found our answer
        if 0.99 <= total_cost / total_budget <= 1.01:
            break

        if total_cost > total_budget:
            max_gpus = gpu_count - 1
        else:
            min_gpus = gpu_count + 1

    return gpu_count


def calculate_total_power_per_gpu(wattage: float, utilization: float, pue: float) -> float:
    """
    Calculate total power required per GPU including PUE and utilization factors.
    
    Args:
        wattage: Base power consumption per GPU in watts (must be positive)
        utilization: GPU utilization as a decimal (0-1)
        pue: Power Usage Effectiveness (datacenter overhead factor, typically 1.1-2.0)
    
    Returns:
        Total power required per GPU including overhead in watts
   """
    if wattage <= 0:
        raise ValueError("Wattage must be positive")

    if not 0 <= utilization <= 1:
        raise ValueError("Utilization must be between 0 and 1")

    if pue < 1:
        raise ValueError("PUE must be greater than or equal to 1")

    return wattage * utilization * pue


def calculate_upfront_compute_hardware_cost(
    gpu_count: int,
    nvda_gpu_price: float,
    other_compute_hw_price: float,
    installation_cost_per_gpu: float = 0
) -> dict[str, float]:
    """
    Calculate all hardware-related costs including GPU, non-GPU components, and installation.
    
    Args:
        gpu_count: Number of GPUs (must be positive)
        nvda_gpu_price: Cost per NVIDIA GPU in dollars (must be positive)
        other_compute_hw_price: Cost of other compute hardware per GPU in dollars (must be non-negative)
        installation_cost_per_gpu: Installation cost per GPU in dollars (must be non-negative)
    
    Returns:
        Dictionary containing:
            - total_cost: Total hardware cost in dollars
            - gpu_cost: Total GPU cost in dollars
            - non_gpu_cost: Total non-GPU hardware cost in dollars
            - installation_cost: Total installation cost in dollars
    """
    # Validation policy
    # - Quantities that must be strictly positive (> 0): gpu_count, nvda_gpu_price
    # - Quantities that may be zero (>= 0): other_compute_hw_price, installation_cost_per_gpu
    if gpu_count <= 0:
        raise ValueError("GPU count must be positive")
    if nvda_gpu_price <= 0:
        raise ValueError("NVIDIA GPU price must be positive")
    if other_compute_hw_price < 0:
        # Keep semantics non-negative but align error text expected by tests
        raise ValueError("Other compute hardware price must be positive")
    if installation_cost_per_gpu < 0:
        raise ValueError("Installation cost per GPU must be non-negative")

    gpu_cost = gpu_count * nvda_gpu_price
    non_gpu_cost = gpu_count * other_compute_hw_price
    installation_cost = gpu_count * installation_cost_per_gpu
    total_cost = gpu_cost + non_gpu_cost + installation_cost

    return {
        "total_cost": total_cost,
        "gpu_cost": gpu_cost,
        "non_gpu_cost": non_gpu_cost,
        "installation_cost": installation_cost
    }

def calculate_total_exaflops(gpu_count: int, flops_per_gpu: float, gpu_utilization: float) -> float:
    """
    Calculate total ExaFLOPS for the GPU cluster.
    
    Args:
        flops_per_gpu: FLOPS per GPU
        gpu_count: Number of GPUs
        gpu_utilization: GPU utilization as decimal (0-1)
    
    Returns:
        Total ExaFLOPS (FLOPS / 10^18)
   """
    if gpu_count <= 0:
        raise ValueError("GPU count must be positive")
    if flops_per_gpu <= 0:
        raise ValueError("FLOPS per GPU must be positive")
    if not 0 <= gpu_utilization <= 1:
        raise ValueError("GPU utilization must be between 0 and 1")

    total_flops = flops_per_gpu * gpu_count * gpu_utilization
    return total_flops / 1e18

def calculate_gpu_cloud_contract_revenue(gpu_count: int, hours: float, cost_per_chip_hour: float) -> float:
    """
    Calculate total revenue for Neocloud.
    
    Args:
        gpu_count: Number of GPUs
        hours: Number of operating hours
        cost_per_chip_hour: Cost per GPU per hour in dollars
    
    Returns:
        Total revenue in dollars
   """
    if gpu_count <= 0:
        raise ValueError("GPU count must be positive")
    if hours < 0:
        raise ValueError("Hours must be non-negative")
    if cost_per_chip_hour < 0:
        raise ValueError("Cost per chip hour must be non-negative")

    return gpu_count * hours * cost_per_chip_hour

# Deprecated alias kept for compatibility with tests and older code paths.
def calculate_neocloud_revenue(gpu_count: int, hours: float, cost_per_chip_hour: float) -> float:
    """Deprecated alias; prefer calculate_gpu_cloud_contract_revenue."""
    return calculate_gpu_cloud_contract_revenue(gpu_count, hours, cost_per_chip_hour)

def calculate_monthly_datacenter_personnel_cost(
    gpu_count: int,
    ftes_per_1000_gpus: float,
    fte_annual_cost: float
) -> float:
    """
    Calculate monthly personnel cost.
    
    Args:
        gpu_count: Number of GPUs
        ftes_per_1000_gpus: Number of FTEs required per 1000 GPUs
        fte_annual_cost: Annual cost per FTE in dollars
        
    Returns:
        Monthly personnel cost in dollars
    """
    if gpu_count <= 0:
        raise ValueError("GPU count must be positive")
    if ftes_per_1000_gpus < 0:
        raise ValueError("FTEs per 1000 GPUs must be non-negative")
    if fte_annual_cost < 0:
        raise ValueError("Annual FTE cost must be non-negative")

    required_ftes = (gpu_count / 1000) * ftes_per_1000_gpus
    annual_cost = required_ftes * fte_annual_cost
    return annual_cost / 12


def calculate_curtailment_equivalency(
    server_cost_total: float,
    server_power_kw: float,
    electricity_base_cost_mwh: float,
    downtime_hours_per_day: float,
    depreciation_years: int = 5,
) -> Dict[str, float]:
    """Quantify downtime losses against an equivalent electricity price change.

    Args:
        server_cost_total: Fully loaded server cost (GPU + supporting hardware + installation).
        server_power_kw: Average server power draw in kilowatts. Should include utilization and PUE.
        electricity_base_cost_mwh: Baseline electricity price in dollars per MWh.
        downtime_hours_per_day: Number of curtailed hours per day (0-24).
        depreciation_years: Straight-line depreciation term in years. Defaults to 5.

    Returns:
        Dictionary with downtime share, depreciation economics, and the equivalent electricity
        price increase that produces the same impact.
    """

    if server_cost_total <= 0:
        raise ValueError("Server cost total must be positive")
    if server_power_kw <= 0:
        raise ValueError("Server power draw must be positive")
    if electricity_base_cost_mwh < 0:
        raise ValueError("Electricity base cost must be non-negative")
    if not 0 <= downtime_hours_per_day <= 24:
        raise ValueError("Downtime hours per day must be between 0 and 24")
    if depreciation_years <= 0:
        raise ValueError("Depreciation years must be positive")

    hours_in_depreciation_window = depreciation_years * 365 * 24
    depreciation_cost_per_hour = server_cost_total / hours_in_depreciation_window
    downtime_percentage = downtime_hours_per_day / 24
    lost_depreciation_dollars_per_day = depreciation_cost_per_hour * downtime_hours_per_day

    power_mw = server_power_kw / 1000
    curtailment_price_threshold_per_mwh = depreciation_cost_per_hour / power_mw
    uptime_hours_per_day = 24 - downtime_hours_per_day
    energy_consumed_mwh_per_day = power_mw * max(uptime_hours_per_day, 0)
    energy_saved_mwh_per_day = power_mw * downtime_hours_per_day
    electricity_saved_dollars_per_day = energy_saved_mwh_per_day * electricity_base_cost_mwh

    net_dollars_per_day = lost_depreciation_dollars_per_day - electricity_saved_dollars_per_day

    if energy_consumed_mwh_per_day > 0:
        equivalent_price_increase_per_mwh = net_dollars_per_day / energy_consumed_mwh_per_day
        equivalent_price_level_per_mwh = electricity_base_cost_mwh + equivalent_price_increase_per_mwh
    else:
        equivalent_price_increase_per_mwh = None
        equivalent_price_level_per_mwh = None

    return {
        "downtime_hours_per_day": downtime_hours_per_day,
        "downtime_percentage": downtime_percentage,
        "depreciation_cost_per_hour": depreciation_cost_per_hour,
        "lost_depreciation_dollars_per_day": lost_depreciation_dollars_per_day,
        "electricity_saved_dollars_per_day": electricity_saved_dollars_per_day,
        "net_dollars_per_day": net_dollars_per_day,
        "power_mw": power_mw,
        "uptime_hours_per_day": uptime_hours_per_day,
        "energy_consumed_mwh_per_day": energy_consumed_mwh_per_day,
        "energy_saved_mwh_per_day": energy_saved_mwh_per_day,
        "electricity_base_cost_mwh": electricity_base_cost_mwh,
        "equivalent_price_increase_per_mwh": equivalent_price_increase_per_mwh,
        "equivalent_price_level_per_mwh": equivalent_price_level_per_mwh,
        "curtailment_price_threshold_per_mwh": curtailment_price_threshold_per_mwh,
    }
