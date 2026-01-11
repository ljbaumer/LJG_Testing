"""AI Lab company profiles for cloud contract liability analysis."""

from dataclasses import dataclass
from datetime import datetime
from typing import List

from .cloud_contracts import CloudContract


@dataclass
class AILabProfile:
    """AI lab company profile for liability analysis."""
    company_name: str
    base_revenue: float         # Current annual revenue
    revenue_base_year: int      # The year for which base_revenue is reported
    contracts: List[CloudContract]
    last_updated: datetime


# OpenAI profile based on known public contracts
OPENAI_PROFILE = AILabProfile(
    company_name="OpenAI",
    base_revenue=10_000_000_000,         # $10B revenue
    revenue_base_year=2025,
    contracts=[],
    last_updated=datetime(2025, 9, 29)
)


__all__ = [
    "AILabProfile",
    "OPENAI_PROFILE",
]
