from dataclasses import dataclass
from typing import Annotated

from src.constants.gpu_dataclass import GB200, GB300, VR200, GPUDataclass
from src.models.GPUCloudModel import GPUCloudModel

# Retained default constants
DEFAULT_INSTALLATION_COST_PER_GPU = 1750.0
DEFAULT_SECOND_LEASE_DISCOUNT_MULTIPLIER = 0.4
DEFAULT_SETUP_TIME_MONTHS = 2.0
DEFAULT_FTES_PER_1000_GPUS = 4.0
DEFAULT_FTE_ANNUAL_COST = 150000.0
DEFAULT_INSURANCE_RATE = 0.016
DEFAULT_SGA_RATE = 0.01





@dataclass
class GPUCloudScenario:
    """Represents a complete scenario configuration for GPU cloud deployment."""

    # Scenario identification
    name: Annotated[str, "Name", "Descriptive name for the scenario"]

    # Hardware Configuration
    gpu_count: Annotated[int, "Hardware", "Number of GPUs"]
    gpu_model: Annotated[GPUDataclass, "Hardware", "GPU Model"]
    quantization_format: Annotated[str, "Hardware", "Quantization Format"]
    installation_cost_per_gpu: Annotated[float, "Cost", "Installation Cost per GPU ($)"]
    chip_price: Annotated[float, "Cost", "Chip Price ($)"]

    # Financial Parameters
    use_debt_financing: Annotated[bool, "Financial", "Use Debt Financing"]
    chip_financing_interest_rate: Annotated[float, "Financial", "Chip Financing Interest Rate"]
    total_deal_years: Annotated[float, "Financial", "Total Deal Years"]
    enable_second_lease: Annotated[bool, "Financial", "Enable Second Lease Period"]
    first_lease_term_months: Annotated[float, "Financial", "First Lease Term (months)"]
    second_lease_term_months: Annotated[float, "Financial", "Second Lease Term (months)"]
    second_lease_discount_multiplier: Annotated[float, "Financial", "Second Lease Discount"]
    tax_rate: Annotated[float, "Financial", "Corporate Tax Rate (%)"]

    # Operational Parameters
    gpu_utilization: Annotated[float, "Operational", "GPU Utilization (%)"]
    sfu: Annotated[float, "Operational", "System FLOPS Utilization (%)"]
    pue: Annotated[float, "Operational", "Power Usage Effectiveness"]
    setup_time_months: Annotated[float, "Operational", "Setup Time (months)"]

    # Cost Parameters
    electricity_cost_per_kwh_in_dollars: Annotated[float, "Cost", "Electricity Cost ($/kWh)"]
    datacenter_rent_per_kw: Annotated[float, "Cost", "Datacenter Rent ($/kW)"]
    neocloud_gpu_cost_per_chip_hour: Annotated[float, "Cost", "GPU Cost per Chip Hour ($)"]

    # Overhead Parameters
    ftes_per_1000_gpus: Annotated[float, "Overhead", "FTEs per 1000 GPUs"]
    fte_annual_cost: Annotated[float, "Overhead", "FTE Annual Cost ($)"]
    insurance_rate: Annotated[float, "Overhead", "Insurance Rate"]
    sga_rate: Annotated[float, "Overhead", "SG&A Rate"]

def create_gpu_cloud_model_from_scenario(scenario: GPUCloudScenario):
    """
    Generate a GPUCloudModel instance from a GPUCloudScenario configuration.
    
    Args:
        scenario (GPUCloudScenario): The scenario configuration to use
    
    Returns:
        GPUCloudModel: A configured GPUCloudModel instance
    """

    return GPUCloudModel(
        # Hardware Configuration
        gpu_count=scenario.gpu_count,
        gpu_model=scenario.gpu_model,
        quantization_format=scenario.quantization_format,
        installation_cost_per_gpu=scenario.installation_cost_per_gpu,
        chip_price=scenario.chip_price,

        # Financial Parameters
        use_debt_financing=scenario.use_debt_financing,
        total_deal_years=scenario.total_deal_years,
        enable_second_lease=scenario.enable_second_lease,
        first_lease_term_months=scenario.first_lease_term_months,
        second_lease_term_months=scenario.second_lease_term_months,
        second_lease_discount_multiplier=scenario.second_lease_discount_multiplier,
        chip_financing_interest_rate=scenario.chip_financing_interest_rate,

        # Operational Parameters
        gpu_utilization=scenario.gpu_utilization,
        sfu=scenario.sfu,
        pue=scenario.pue,
        setup_time_months=scenario.setup_time_months,

        # Cost Parameters
        electricity_cost_per_kwh_in_dollars=scenario.electricity_cost_per_kwh_in_dollars,
        datacenter_rent_per_kw=scenario.datacenter_rent_per_kw,
        neocloud_gpu_cost_per_chip_hour=scenario.neocloud_gpu_cost_per_chip_hour,

        # Overhead
        ftes_per_1000_gpus=scenario.ftes_per_1000_gpus,
        fte_annual_cost=scenario.fte_annual_cost,
        insurance_rate=scenario.insurance_rate,
        sga_rate=scenario.sga_rate,
        tax_rate=scenario.tax_rate
    )

# Define the default scenario based on the provided reference constants
DEFAULT_SCENARIO = GPUCloudScenario(
    name="Default Case",
    gpu_count=10,  # Using a reasonable default
    gpu_model=GB200,  # Most common in scenarios
    quantization_format="fp16",
    installation_cost_per_gpu=DEFAULT_INSTALLATION_COST_PER_GPU,
    chip_price=GB200.ai_accelerator_price,  # Default to GPU model price
    use_debt_financing=True,
    chip_financing_interest_rate=0.08,
    total_deal_years=5.0,  # 5 year deal
    enable_second_lease=False,  # Single lease by default
    first_lease_term_months=60.0,  # Full 5 years
    second_lease_term_months=0.0,
    second_lease_discount_multiplier=0.0,
    gpu_utilization=0.80,
    sfu=0.60,
    pue=1.1,
    setup_time_months=DEFAULT_SETUP_TIME_MONTHS,
    electricity_cost_per_kwh_in_dollars=0.10,
    datacenter_rent_per_kw=135.0,
    neocloud_gpu_cost_per_chip_hour=3.0,
    ftes_per_1000_gpus=DEFAULT_FTES_PER_1000_GPUS,
    fte_annual_cost=DEFAULT_FTE_ANNUAL_COST,
    insurance_rate=DEFAULT_INSURANCE_RATE,
    sga_rate=DEFAULT_SGA_RATE,
    tax_rate=0.25,  # 25% combined federal + state
)

# Define the scenarios based on the provided reference
SCENARIOS = {
    "Oracle-OpenAI $300B": GPUCloudScenario(
        name="Oracle-OpenAI $300B",
        gpu_count=2283106,  # Adjusted count for closest to $300B revenue at exactly $3/chip/hour
        gpu_model=GB300,
        quantization_format="fp16",
        installation_cost_per_gpu=DEFAULT_INSTALLATION_COST_PER_GPU,
        chip_price=GB300.ai_accelerator_price,
        use_debt_financing=True,
        chip_financing_interest_rate=0.08,  # Default rate for mega-deal
        total_deal_years=5.0,  # 5 year contract
        enable_second_lease=False,  # Single 5-year lease
        first_lease_term_months=60.0,  # Full 5 years as single lease
        second_lease_term_months=0.0,  # No second lease
        second_lease_discount_multiplier=0.0,  # N/A for single lease
        electricity_cost_per_kwh_in_dollars=0.08,  # Efficient at scale
        datacenter_rent_per_kw=120.0,  # Lower cost due to scale
        neocloud_gpu_cost_per_chip_hour=3.00,  # Exactly $3.00 per chip hour
        gpu_utilization=0.85,  # High utilization
        sfu=0.70,  # Advanced workloads
        pue=1.15,  # Efficient at massive scale
        setup_time_months=DEFAULT_SETUP_TIME_MONTHS,
        ftes_per_1000_gpus=2.0,  # Efficient at scale
        fte_annual_cost=DEFAULT_FTE_ANNUAL_COST,
        insurance_rate=0.012,  # Lower risk premium at scale
        sga_rate=0.005,  # Lower overhead percentage
        tax_rate=0.20,  # 20% corporate tax rate
    ),

    "Default": DEFAULT_SCENARIO,

    "CoreWeave B200": GPUCloudScenario(
        name="CoreWeave B200",
        gpu_count=50000,
        gpu_model=GB200,
        quantization_format="fp16",
        installation_cost_per_gpu=DEFAULT_INSTALLATION_COST_PER_GPU,
        chip_price=GB200.ai_accelerator_price,
        use_debt_financing=True,
        chip_financing_interest_rate=0.08,
        total_deal_years=5.0,
        enable_second_lease=True,  # Keep dual lease for this scenario
        first_lease_term_months=48.0,
        second_lease_term_months=12.0,
        second_lease_discount_multiplier=DEFAULT_SECOND_LEASE_DISCOUNT_MULTIPLIER,
        electricity_cost_per_kwh_in_dollars=0.115,
        datacenter_rent_per_kw=135.0,
        neocloud_gpu_cost_per_chip_hour=2.93,
        gpu_utilization=0.80,
        sfu=0.60,
        pue=1.25,
        setup_time_months=DEFAULT_SETUP_TIME_MONTHS,
        ftes_per_1000_gpus=DEFAULT_FTES_PER_1000_GPUS,
        fte_annual_cost=DEFAULT_FTE_ANNUAL_COST,
        insurance_rate=DEFAULT_INSURANCE_RATE,
        sga_rate=DEFAULT_SGA_RATE,
        tax_rate=0.25,  # US combined tax rate
    ),


    "Qatar (Domestic)": GPUCloudScenario(
        name="Qatar (Domestic)",
        gpu_count=50000,
        gpu_model=VR200,
        quantization_format="fp16",
        installation_cost_per_gpu=DEFAULT_INSTALLATION_COST_PER_GPU,
        chip_price=VR200.ai_accelerator_price,
        use_debt_financing=True,
        chip_financing_interest_rate=0.07,
        total_deal_years=6.0,
        enable_second_lease=True,  # Keep dual lease
        first_lease_term_months=48.0,
        second_lease_term_months=24.0,
        second_lease_discount_multiplier=DEFAULT_SECOND_LEASE_DISCOUNT_MULTIPLIER,
        electricity_cost_per_kwh_in_dollars=0.06,
        datacenter_rent_per_kw=135.0,
        neocloud_gpu_cost_per_chip_hour=4.50,
        gpu_utilization=0.80,
        sfu=0.60,
        pue=1.25,
        setup_time_months=DEFAULT_SETUP_TIME_MONTHS,
        ftes_per_1000_gpus=DEFAULT_FTES_PER_1000_GPUS,
        fte_annual_cost=DEFAULT_FTE_ANNUAL_COST,
        insurance_rate=DEFAULT_INSURANCE_RATE,
        sga_rate=DEFAULT_SGA_RATE,
        tax_rate=0.10,  # Qatar corporate tax rate
    ),



    "Qatar (International)": GPUCloudScenario(
        name="Qatar (International)",
        gpu_count=100000,
        gpu_model=GB200,
        quantization_format="fp16",
        installation_cost_per_gpu=DEFAULT_INSTALLATION_COST_PER_GPU,
        chip_price=GB200.ai_accelerator_price,
        use_debt_financing=True,
        chip_financing_interest_rate=0.07,
        total_deal_years=6.0,
        enable_second_lease=True,  # Keep dual lease
        first_lease_term_months=48.0,
        second_lease_term_months=24.0,
        second_lease_discount_multiplier=DEFAULT_SECOND_LEASE_DISCOUNT_MULTIPLIER,
        electricity_cost_per_kwh_in_dollars=0.08,
        datacenter_rent_per_kw=150.0,
        neocloud_gpu_cost_per_chip_hour=4.50,
        gpu_utilization=0.80,
        sfu=0.60,
        pue=1.25,
        setup_time_months=DEFAULT_SETUP_TIME_MONTHS,
        ftes_per_1000_gpus=DEFAULT_FTES_PER_1000_GPUS,
        fte_annual_cost=DEFAULT_FTE_ANNUAL_COST,
        insurance_rate=DEFAULT_INSURANCE_RATE,
        sga_rate=DEFAULT_SGA_RATE,
        tax_rate=0.10,  # Qatar corporate tax rate
    ),


    "Hyperscaler Scenario": GPUCloudScenario(
        name="Hyperscaler Scenario",
        gpu_count=100000,
        gpu_model=GB200,
        quantization_format="fp16",
        installation_cost_per_gpu=DEFAULT_INSTALLATION_COST_PER_GPU,
        chip_price=GB200.ai_accelerator_price,
        use_debt_financing=False,
        chip_financing_interest_rate=0.05,
        total_deal_years=5.0,
        enable_second_lease=False,  # Single lease for hyperscalers
        first_lease_term_months=60.0,  # Full 5 years
        second_lease_term_months=0.0,
        second_lease_discount_multiplier=0.0,
        electricity_cost_per_kwh_in_dollars=0.10,
        datacenter_rent_per_kw=120.0,
        neocloud_gpu_cost_per_chip_hour=2.75,
        gpu_utilization=0.80,
        sfu=0.60,
        pue=1.15,
        setup_time_months=DEFAULT_SETUP_TIME_MONTHS,
        ftes_per_1000_gpus=DEFAULT_FTES_PER_1000_GPUS,
        fte_annual_cost=DEFAULT_FTE_ANNUAL_COST,
        insurance_rate=DEFAULT_INSURANCE_RATE,
        sga_rate=DEFAULT_SGA_RATE,
        tax_rate=0.25,  # US combined tax rate
    ),


    "Stargate $100B": GPUCloudScenario(
        name="Stargate $100B",
        gpu_count=500000,
        gpu_model=GB200,
        quantization_format="fp16",
        installation_cost_per_gpu=DEFAULT_INSTALLATION_COST_PER_GPU,
        chip_price=GB200.ai_accelerator_price,
        use_debt_financing=False,
        chip_financing_interest_rate=0.04,
        total_deal_years=5.0,
        enable_second_lease=False,  # Single lease for mega-projects
        first_lease_term_months=60.0,  # Full 5 years
        second_lease_term_months=0.0,
        second_lease_discount_multiplier=0.0,
        electricity_cost_per_kwh_in_dollars=0.10,
        datacenter_rent_per_kw=130.0,
        neocloud_gpu_cost_per_chip_hour=2.50,
        gpu_utilization=0.85,
        sfu=0.65,
        pue=1.10,
        setup_time_months=DEFAULT_SETUP_TIME_MONTHS,
        ftes_per_1000_gpus=DEFAULT_FTES_PER_1000_GPUS,
        fte_annual_cost=DEFAULT_FTE_ANNUAL_COST,
        insurance_rate=DEFAULT_INSURANCE_RATE,
        sga_rate=DEFAULT_SGA_RATE,
        tax_rate=0.25,  # US combined tax rate
    ),
}
