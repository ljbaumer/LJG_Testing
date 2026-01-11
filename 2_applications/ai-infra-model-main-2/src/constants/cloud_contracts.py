"""Cloud contract definitions and CSV loading helpers."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd

HOURS_PER_YEAR = 8_760
DEFAULT_CONTRACT_DURATION_YEARS = 5
DEFAULT_PRICE_PER_CHIP_HOUR_USD = 3.0
OPENAI_CLOUD_CONTRACTS_CSV = Path("data/cloud_contracts/openai_announcements.csv")


@dataclass
class CloudContract:
    """Cloud infrastructure contract definition."""

    provider_name: str
    total_value: float
    duration_years: int
    start_year: int
    chip_purchase_growth_rate: float = 0.0

    def calculate_payment_schedule(
        self,
        price_per_chip_hour: float = DEFAULT_PRICE_PER_CHIP_HOUR_USD
    ) -> pd.DataFrame:
        """Calculate annual payments based on chip purchase growth and cumulative fleet revenue.

        Args:
            price_per_chip_hour: Revenue per chip per hour (default $3)

        Returns:
            DataFrame with columns: year, chips_purchased, cumulative_chips, payment, provider
        """
        years = list(range(self.start_year, self.start_year + self.duration_years))
        growth_rate = self.chip_purchase_growth_rate

        # Calculate cumulative chip multipliers for each year
        # Year 1: 1x, Year 2: 1 + (1+g), Year 3: 1 + (1+g) + (1+g)^2, etc.
        cumulative_multipliers = []
        for i in range(self.duration_years):
            cumulative = sum((1 + growth_rate) ** j for j in range(i + 1))
            cumulative_multipliers.append(cumulative)

        # Total revenue over all years = sum of (cumulative_chips × hours × price)
        # cumulative_chips_year_i = X × cumulative_multiplier_i
        # Total revenue = X × HOURS_PER_YEAR × price × sum(cumulative_multipliers)
        sum_of_cumulative = sum(cumulative_multipliers)

        # Solve for X (Year 1 chip purchases)
        year_1_chips = self.total_value / (HOURS_PER_YEAR * price_per_chip_hour * sum_of_cumulative)

        # Calculate chip purchases, cumulative fleet, and revenue for each year
        chips_purchased = []
        cumulative_chips = []
        payments = []

        for i in range(self.duration_years):
            # Chips purchased this year
            chips_this_year = year_1_chips * ((1 + growth_rate) ** i)
            chips_purchased.append(chips_this_year)

            # Cumulative fleet size
            cumulative = year_1_chips * cumulative_multipliers[i]
            cumulative_chips.append(cumulative)

            # Revenue from cumulative fleet
            revenue = cumulative * HOURS_PER_YEAR * price_per_chip_hour
            payments.append(revenue)

        # Validate total
        calculated_total = sum(payments)
        if abs(calculated_total - self.total_value) > 1.0:
            raise ValueError(
                f"Payment calculation error for {self.provider_name}: "
                f"calculated total ${calculated_total:,.2f} does not match contract value "
                f"${self.total_value:,.2f}"
            )

        return pd.DataFrame({
            "year": years,
            "chips_purchased": chips_purchased,
            "cumulative_chips": cumulative_chips,
            "payment": payments,
            "provider": self.provider_name,
        })


def _row_to_contract(
    row: Dict[str, str],
    *,
    duration_years: int,
    price_per_chip_hour: float,
) -> Optional[CloudContract]:
    """Parse one CSV row describing a contract."""

    company = row.get("Company", "").strip()
    start_year_text = (row.get("Contract Start Year", "") or "").strip()
    contract_value_text = (row.get("Contract Value (USD billions)", "") or "").strip()

    if not start_year_text or not contract_value_text:
        return None

    start_year = int(start_year_text)
    contract_value_billions = float(contract_value_text)
    total_value_usd = contract_value_billions * 1e9

    return CloudContract(
        provider_name=company,
        total_value=total_value_usd,
        duration_years=duration_years,
        start_year=start_year,
        chip_purchase_growth_rate=0.0,
    )


def load_cloud_contracts_from_csv(
    csv_path: Path | str = OPENAI_CLOUD_CONTRACTS_CSV,
    *,
    duration_years: int = DEFAULT_CONTRACT_DURATION_YEARS,
    price_per_chip_hour: float = DEFAULT_PRICE_PER_CHIP_HOUR_USD,
) -> List[CloudContract]:
    """Generate CloudContract objects from the disclosure CSV."""

    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")

    contracts: List[CloudContract] = []

    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            contract = _row_to_contract(
                row,
                duration_years=duration_years,
                price_per_chip_hour=price_per_chip_hour,
            )
            if contract is not None:
                contracts.append(contract)

    return contracts


def combine_contract_payment_schedules(contracts: Iterable[CloudContract]) -> pd.DataFrame:
    """Return payment schedule rows for all contracts provided."""
    contract_list = list(contracts)
    if not contract_list:
        return pd.DataFrame(columns=['year', 'provider', 'payment'])

    frames = [contract.calculate_payment_schedule().copy() for contract in contract_list]
    combined = pd.concat(frames, ignore_index=True)

    return combined.sort_values(["year", "provider"]).reset_index(drop=True)


__all__ = [
    "CloudContract",
    "load_cloud_contracts_from_csv",
    "OPENAI_CLOUD_CONTRACTS_CSV",
    "combine_contract_payment_schedules",
]
