"""Revenue to CapEx calculation pipeline.

Converts geometric revenue growth to infrastructure CapEx requirements using:
- $3/chip-hour pricing model
- Delta calculations (only new chips needed each year)
- Existing capex_helpers for accurate infrastructure costs
"""


import numpy as np
import pandas as pd

from src.constants.gpu_dataclass import GB300, GPUDataclass
from src.constants.value_chain_depreciation_schedules import NVIDIA_DEFAULT_GROSS_MARGIN
from src.utils.capex_helpers import (
    DEFAULT_PUE,
    DEFAULT_UTILIZATION,
    CapexStartingPoint,
    calculate_infrastructure,
)

# Constants
HOURS_PER_YEAR = 8760  # 365 * 24
DEFAULT_PRICE_PER_CHIP_HOUR = 3.0
DEFAULT_GPU_MODEL = GB300  # Use GB300 as the default build assumption to match GPU contract mix
DEFAULT_CHIP_VENDOR_MARGIN = NVIDIA_DEFAULT_GROSS_MARGIN  # 0.75


def revenue_to_chips_needed(
    revenue_df: pd.DataFrame,
    price_per_chip_hour: float = DEFAULT_PRICE_PER_CHIP_HOUR
) -> pd.DataFrame:
    """
    Convert revenue to number of chips needed for a given year.

    Args:
        revenue_df: DataFrame with 'year' and 'ai_revenue' columns
        price_per_chip_hour: Revenue per chip per hour (default $3)

    Returns:
        DataFrame with additional columns:
        - chip_hours_needed: Total chip-hours to generate the revenue
        - chips_needed_total: Total number of chips needed (cumulative)
    """
    df = revenue_df.copy()

    # Calculate chip-hours needed to generate this revenue
    df['chip_hours_needed'] = df['ai_revenue'] / price_per_chip_hour

    # Convert to number of chips (assuming 100% utilization for revenue generation)
    df['chips_needed_total'] = df['chip_hours_needed'] / HOURS_PER_YEAR

    return df


def calculate_chip_delta(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate incremental chips required each year."""
    df = df.copy().sort_values('year').reset_index(drop=True)

    df['chips_needed_new'] = df['chips_needed_total'].diff().fillna(0)
    df.loc[df.index[0], 'chips_needed_new'] = df.loc[df.index[0], 'chips_needed_total']
    df['chips_needed_new'] = df['chips_needed_new'].clip(lower=0)

    return df


def calculate_capex_from_chips(
    df: pd.DataFrame,
    gpu_model: GPUDataclass
) -> pd.DataFrame:
    """
    Calculate total CapEx for chip purchases using existing infrastructure tools.
    Vectorized for performance—no row-by-row loops.

    Args:
        df: DataFrame with 'year' and 'chips_needed_new' columns
        gpu_model: GpuModel dataclass instance (e.g., get_gpu_model_by_name("GB300"))

    Returns:
        DataFrame with CapEx breakdown by year
    """
    def _compute_infra_for_row(row):
        num_new_chips = int(row['chips_needed_new'])

        # Use existing infrastructure calculator (handles zero chips by returning zeros)
        infra = calculate_infrastructure(
            starting_point=CapexStartingPoint.NUM_CHIPS,
            value=num_new_chips,
            gpu_model=gpu_model,
            chip_vendor_margin=DEFAULT_CHIP_VENDOR_MARGIN,
            pue=DEFAULT_PUE,
            utilization=DEFAULT_UTILIZATION,
        )
        
        return {
            'year': row['year'],
            'num_new_chips': num_new_chips,
            'chip_capex': infra.chip_capex,
            'datacenter_capex': infra.datacenter_capex,
            'power_capex': infra.power_capex,
            'total_capex': infra.total_capex,
            'power_requirement_mw': infra.power_requirement_mw
        }

    # Apply vectorized computation to all rows
    capex_data = df.apply(_compute_infra_for_row, axis=1)
    result_df = pd.DataFrame(list(capex_data))
    return result_df


def calculate_shortfall(
    revenue_df: pd.DataFrame,
    capex_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Calculate AI-related funding shortfall (AI CapEx - AI Revenue).

    Args:
        revenue_df: DataFrame with year and ai_revenue columns.
        capex_df: DataFrame with CapEx figures per year.

    Returns:
        DataFrame with 'shortfall' metric.
    """
    # Merge revenue and capex on 'year'
    merged_df = pd.merge(revenue_df[['year', 'ai_revenue']], capex_df, on='year', how='left')

    # Calculate shortfall
    merged_df['shortfall'] = merged_df['total_capex'] - merged_df['ai_revenue']

    return merged_df


def calculate_debt_required(
    shortfall_df: pd.DataFrame,
    non_ai_fcf_schedule: pd.DataFrame
) -> pd.DataFrame:
    """
    Calculate debt required to cover shortfall after using non-AI FCF.

    Debt Required = max(0, Shortfall - Non-AI FCF)

    Args:
        shortfall_df: DataFrame with 'year' and 'shortfall' columns.
        non_ai_fcf_schedule: DataFrame with 'year' and 'non_ai_fcf' columns.

    Returns:
        DataFrame with 'debt_required' column.
    """
    merged_df = pd.merge(shortfall_df, non_ai_fcf_schedule, on='year', how='left')
    merged_df['debt_required'] = np.maximum(0, merged_df['shortfall'] - merged_df['non_ai_fcf'])
    return merged_df


def create_non_ai_fcf_schedule_with_growth(
    df: pd.DataFrame,
    non_ai_fcf: float,
    non_ai_fcf_growth_rate: float = 0.02
) -> pd.DataFrame:
    """
    Create non-AI FCF schedule with growth projection.
    
    Args:
        df: DataFrame with 'year' column
        non_ai_fcf: Base year non-AI free cash flow
        non_ai_fcf_growth_rate: Annual growth rate (default 2%)
    
    Returns:
        DataFrame with 'non_ai_fcf' column added
    """
    base_year = df['year'].min()
    df = df.copy()
    df['years_from_base'] = df['year'] - base_year
    df['non_ai_fcf'] = non_ai_fcf * ((1 + non_ai_fcf_growth_rate) ** df['years_from_base'])
    return df


def revenue_to_debt_pipeline(
    revenue_df: pd.DataFrame,
    non_ai_fcf: float,
    price_per_chip_hour: float = DEFAULT_PRICE_PER_CHIP_HOUR,
    gpu_model: GPUDataclass = DEFAULT_GPU_MODEL,
    non_ai_fcf_growth_rate: float = 0.02
) -> pd.DataFrame:
    """
    Complete pipeline from revenue to debt requirements.
    
    Args:
        revenue_df: DataFrame with 'year' and 'revenue' columns
        non_ai_fcf: Base year non-AI free cash flow
        price_per_chip_hour: Revenue per chip per hour (default $3)
        gpu_model: GPU model to use for calculations
        non_ai_fcf_growth_rate: Annual growth rate of non-AI FCF (default 2%)
    
    Returns:
        DataFrame with complete analysis including:
        - Revenue and chip calculations
        - CapEx breakdown
        - Shortfall calculations
        - Debt requirements
        - Performance metrics
    """
    # Validate input
    required_columns = {'year', 'revenue'}
    if not required_columns.issubset(revenue_df.columns):
        raise ValueError(f"revenue_df must have columns: {required_columns}")
    
    # Step 1: Revenue to chips then get the delta
    chip_requirements_annually = revenue_to_chips_needed(revenue_df, price_per_chip_hour)
    chip_requirements_annually = calculate_chip_delta(chip_requirements_annually)
    
    # Step 2: Calculate CapEx
    capex_schedule = calculate_capex_from_chips(chip_requirements_annually, gpu_model)

    # Step 3: Calculate Shortfall
    shortfall_df = calculate_shortfall(revenue_df, capex_schedule)

    # Step 4: Create FCF schedule
    fcf_df = create_non_ai_fcf_schedule_with_growth(
        shortfall_df[['year']].copy(), non_ai_fcf, non_ai_fcf_growth_rate
    )

    # Step 5: Calculate Debt Required
    debt_df = calculate_debt_required(shortfall_df, fcf_df)

    return debt_df


def create_geometric_revenue_schedule(
    start_year: int,
    num_years: int,
    initial_revenue: float,
    growth_rate: float
) -> pd.DataFrame:
    """
    Create a geometric revenue growth schedule for testing.

    Args:
        start_year: Starting year
        num_years: Number of years to project
        initial_revenue: Revenue in first year
        growth_rate: Annual growth rate (0.45 = 45%)

    Returns:
        DataFrame with year and revenue columns
    """
    years = list(range(start_year, start_year + num_years))
    revenues = []

    for i in range(num_years):
        revenue = initial_revenue * ((1 + growth_rate) ** i)
        revenues.append(revenue)

    return pd.DataFrame({
        'year': years,
        'revenue': revenues
    })


__all__ = [
    "revenue_to_chips_needed",
    "calculate_chip_delta",
    "calculate_capex_from_chips",
    "calculate_shortfall",
    "calculate_debt_required",
    "create_non_ai_fcf_schedule_with_growth",
    "revenue_to_debt_pipeline",
    "create_geometric_revenue_schedule"
]
