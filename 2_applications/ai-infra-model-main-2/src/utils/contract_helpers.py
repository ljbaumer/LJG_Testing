"""Helper functions for loading and processing cloud contract data."""

import pandas as pd

from src.constants.gpu_dataclass import ALL_GPU_LIST, GB300
from src.utils.capex_helpers import CapexStartingPoint, calculate_infrastructure


def load_contract_defaults_from_csv(
    csv_path: str = 'data/cloud_contracts/openai_announcements.csv'
) -> pd.DataFrame:
    """Load contracts from CSV and calculate default values.

    Priority order for calculating contract value:
    1. Use explicit "Contract Value (USD billions)" if available
    2. Calculate from "Number of Megawatts" using infrastructure helpers with GPU model from CSV
    3. Use $10B placeholder if neither available

    For MW-based calculations:
    - GPU model is specified in "GPU Model for Calculation" column
    - Defaults to GB300 if not specified

    Args:
        csv_path: Path to the CSV file containing contract data

    Returns:
        DataFrame with columns from CSV plus calculated 'value_b' column
        containing contract values in billions of dollars
    """
    # Create GPU model lookup by name
    GPU_MODEL_LOOKUP = {gpu.name: gpu for gpu in ALL_GPU_LIST}

    df = pd.read_csv(csv_path)

    # Add calculated value_b column
    values_b = []

    for _, row in df.iterrows():
        contract_value = row['Contract Value (USD billions)']
        megawatts = row['Number of Megawatts']
        gpu_model_name = row.get('GPU Model for Calculation', 'GB300')

        # Determine value with priority
        if pd.notna(contract_value):
            value_b = float(contract_value)
        elif pd.notna(megawatts):
            # Get GPU model from CSV column, default to GB300
            if pd.notna(gpu_model_name) and gpu_model_name in GPU_MODEL_LOOKUP:
                gpu_model = GPU_MODEL_LOOKUP[gpu_model_name]
            else:
                gpu_model = GB300

            # Calculate using capex_helpers
            infra = calculate_infrastructure(
                starting_point=CapexStartingPoint.POWER_CAPACITY,
                value=float(megawatts),  # Already in MW
                gpu_model=gpu_model,
                pue=1.2,
                utilization=0.8,
                power_cost_per_kw=2500,
                datacenter_cost_per_mw=15_000_000
            )
            value_b = infra.total_capex / 1e9  # Convert to billions
        else:
            value_b = 10.0  # Placeholder

        values_b.append(value_b)

    df['value_b'] = values_b

    # Ensure Contract Start Year has defaults
    df['Contract Start Year'] = df['Contract Start Year'].fillna(2025).astype(int)

    return df
