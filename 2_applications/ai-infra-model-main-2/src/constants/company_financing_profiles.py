"""Company financing profiles for AI infrastructure funding analysis."""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import List, Optional

import pandas as pd

from src.constants.cloud_contracts import CloudContract


class CreditRating(Enum):
    """Standard credit ratings from major agencies."""
    AAA = "AAA"
    AA_PLUS = "AA+"
    AA = "AA"
    AA_MINUS = "AA-"
    A_PLUS = "A+"
    A = "A"
    A_MINUS = "A-"
    BBB_PLUS = "BBB+"
    BBB = "BBB"
    BBB_MINUS = "BBB-"
    BB_PLUS = "BB+"
    BB = "BB"
    BB_MINUS = "BB-"
    B_PLUS = "B+"
    B = "B"
    B_MINUS = "B-"


@dataclass
class CompanyFinancingProfile:
    """Financial profile for a public company's AI infrastructure funding analysis.

    Tracks AI vs non-AI economics to determine subsidy requirements.
    """
    # Company identifier
    company_name: str  # e.g., "ORCL"
    last_updated: datetime  # When this data was pulled

    # Current financials (split AI vs non-AI)
    non_ai_fcf: float  # Non-AI business free cash flow (annual)
    current_debt: float  # Total debt outstanding
    current_ebitda: float  # TTM EBITDA
    current_cash: float  # Cash & equivalents
    current_rating: CreditRating  # Current credit rating

    # GPU lease and cloud service commitments
    gpu_lease_rpo_schedule: pd.DataFrame  # columns: year, ai_revenue

    # Credit constraints
    downgrade_threshold: float  # Debt/EBITDA that triggers downgrade
    interest_rate: float  # Current borrowing rate

    # Growth assumptions
    non_ai_fcf_growth_rate: float  # Expected growth of non-AI FCF

    cloud_contracts: Optional[List[CloudContract]] = None


def validate_company_profile(profile: CompanyFinancingProfile) -> None:
    """Validate CompanyFinancingProfile inputs."""
    if profile.non_ai_fcf <= 0:
        raise ValueError("Non-AI FCF must be positive")

    if profile.current_debt < 0:
        raise ValueError("Debt cannot be negative")

    if profile.current_ebitda <= 0:
        raise ValueError("EBITDA must be positive")

    if profile.downgrade_threshold <= 0:
        raise ValueError("Downgrade threshold must be positive")

    if not isinstance(profile.current_rating, CreditRating):
        raise ValueError("Credit rating must be a CreditRating enum value")

    # Validate GPU lease RPO schedule
    required_columns = {'year', 'ai_revenue'}
    if not required_columns.issubset(profile.gpu_lease_rpo_schedule.columns):
        raise ValueError(f"GPU lease RPO schedule must have columns: {required_columns}")

    if profile.gpu_lease_rpo_schedule['ai_revenue'].min() < 0:
        raise ValueError("AI revenue in GPU lease RPO schedule cannot be negative")

    if profile.cloud_contracts is not None:
        if not isinstance(profile.cloud_contracts, list) or not all(isinstance(contract, CloudContract) for contract in profile.cloud_contracts):
            raise ValueError("cloud_contracts must be a list of CloudContract instances when provided")


# Oracle GPU lease contracts modeled with geometric payment ramps
# Source: Oracle Announces Fiscal Year 2026 First Quarter Financial Results
# https://investor.oracle.com/investor-news/news-details/2025/Oracle-Announces-Fiscal-Year-2026-First-Quarter-Financial-Results/default.aspx
# Reported Q1 FY26 Remaining Performance Obligations (RPO): $455B.
# We attribute $300B of that total to OpenAI commitments, leaving $155B for other
# customers ($455B - $300B = $155B).
ORACLE_GPU_CONTRACTS: List[CloudContract] = [
    CloudContract(
        provider_name="OpenAI",
        total_value=300_000_000_000,  # $300B allocation to OpenAI commitments
        duration_years=5,
        start_year=2025,
        chip_purchase_growth_rate=0.30,  # 30% YoY growth in chip purchases for flagship contract
    ),
    CloudContract(
        provider_name="Other Enterprise",
        total_value=155_000_000_000,  # Remaining $155B of Oracle's reported RPO
        duration_years=5,
        start_year=2025,
        chip_purchase_growth_rate=0.20,  # 20% YoY growth in chip purchases for broader enterprise
    ),
]

_ORACLE_GPU_CONTRACT_SCHEDULES = [
    contract.calculate_payment_schedule()
    .rename(columns={"payment": "ai_revenue"})
    .assign(provider=contract.provider_name)
    for contract in ORACLE_GPU_CONTRACTS
]

ORACLE_GPU_RPO_BREAKDOWN = (
    pd.concat(_ORACLE_GPU_CONTRACT_SCHEDULES, ignore_index=True)
    [['year', 'provider', 'ai_revenue']]
    .sort_values(['year', 'provider'])
    .reset_index(drop=True)
)

_ORACLE_GPU_LEASE_RPO = (
    ORACLE_GPU_RPO_BREAKDOWN.groupby('year', as_index=False)['ai_revenue']
    .sum()
)


# Oracle-specific configuration
ORACLE_PROFILE = CompanyFinancingProfile(
    company_name="ORCL",
    last_updated=datetime(2025, 1, 26),

    # Financials (based on realistic estimates)
    non_ai_fcf=12_000_000_000,  # $12B from traditional DB/apps business
    current_debt=90_000_000_000,  # $90B total debt
    current_ebitda=20_000_000_000,  # $20B EBITDA
    current_cash=10_000_000_000,  # $10B cash
    current_rating=CreditRating.BBB_PLUS,

    # GPU lease and cloud service commitments (contracted revenue via ramped contract)
    gpu_lease_rpo_schedule=_ORACLE_GPU_LEASE_RPO.copy(),
    cloud_contracts=ORACLE_GPU_CONTRACTS,

    # Credit constraint (BBB+ downgrade at 4.0x)
    downgrade_threshold=4.0,  # 4x Debt/EBITDA triggers downgrade
    interest_rate=0.045,  # 4.5%

    # Growth rates
    non_ai_fcf_growth_rate=0.02,  # 2% growth in traditional business
)


"""Deprecated company profiles retained for future reference.

# Microsoft profile - Strong balance sheet, high FCF
MICROSOFT_PROFILE = CompanyFinancingProfile(...)

# Google profile - High margins, moderate buildout
GOOGLE_PROFILE = CompanyFinancingProfile(...)

# Amazon profile - Aggressive buildout, tighter margins
AMAZON_PROFILE = CompanyFinancingProfile(...)

# Meta profile - High growth scenario
META_PROFILE = CompanyFinancingProfile(...)
"""

# Oracle-focused company list
COMPANY_PROFILES = [ORACLE_PROFILE]


__all__ = [
    "CreditRating",
    "CompanyFinancingProfile",
    "validate_company_profile",
    "ORACLE_PROFILE",
    "ORACLE_GPU_CONTRACTS",
    "ORACLE_GPU_RPO_BREAKDOWN",
    "COMPANY_PROFILES",
]
