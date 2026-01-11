"""Company-specific AI infrastructure funding model.

Analyzes how much a company's non-AI business needs to subsidize AI infrastructure buildout
while maintaining credit quality.
"""

from typing import Iterable, List

import pandas as pd

from src.constants.cloud_contracts import CloudContract, combine_contract_payment_schedules
from src.constants.company_financing_profiles import (
    ORACLE_PROFILE,
    CompanyFinancingProfile,
    validate_company_profile,
)
from src.constants.gpu_dataclass import GB300, GPUDataclass
from src.constants.value_chain_depreciation_schedules import CapexDepreciationSchedule
from src.utils.revenue_to_capex import (
    calculate_capex_from_chips,
    calculate_chip_delta,
    calculate_debt_required,
    create_non_ai_fcf_schedule_with_growth,
    revenue_to_chips_needed,
)
from src.utils.revenue_to_capex import (
    calculate_shortfall as _calculate_shortfall,
)

DEFAULT_PRICE_PER_CHIP_HOUR = 3.0
DEFAULT_GPU_MODEL = GB300

class CompanyFundingModel:
    """Model for analyzing company-specific AI infrastructure funding dynamics.

    Core Question: How much does the non-AI business need to subsidize AI buildout?

    Key Analysis:
    - Convert RPO revenue schedule to CapEx requirements
    - Quantify CapEx gap that must be financed
    - Determine subsidy needed from non-AI business
    - Assess credit impact and funding feasibility
    """

    def __init__(self, profile: CompanyFinancingProfile, contracts: Iterable[CloudContract]):
        """Initialize with company financing profile and required GPU contracts."""
        validate_company_profile(profile)
        self.profile = profile

        self.contracts: List[CloudContract] = list(contracts)
        if not self.contracts:
            raise ValueError("CompanyFundingModel requires at least one CloudContract.")

        self.profile.cloud_contracts = self.contracts
        self.profile.gpu_lease_rpo_schedule = self.generate_gpu_lease_revenue_schedule(self.contracts)


        # Precompute primary analysis tables
        self.capex_schedule = self.calculate_capex_schedule()
        self.shortfall = self.calculate_shortfall()

        # Precompute FCF schedule and store as a class attribute
        self.non_ai_fcf_schedule = create_non_ai_fcf_schedule_with_growth(
            self.shortfall[['year']].copy(),
            self.profile.non_ai_fcf,
            self.profile.non_ai_fcf_growth_rate
        )

        # Precompute debt schedule
        self.debt_schedule = calculate_debt_required(self.shortfall, self.non_ai_fcf_schedule)

    def generate_gpu_lease_revenue_schedule(self, contracts: Iterable[CloudContract]) -> pd.DataFrame:
        """Generate GPU lease revenue schedule from a list of CloudContracts."""
        gpu_contract_df = combine_contract_payment_schedules(contracts)

        # Aggregate the detailed payments to get the annual total revenue
        revenue_schedule = gpu_contract_df.groupby('year', as_index=False)['payment'].sum()

        return (
            revenue_schedule
            .rename(columns={"payment": "ai_revenue"})
            .sort_values("year")
            .reset_index(drop=True)
        )

    def calculate_capex_schedule(
        self,
        price_per_chip_hour: float = DEFAULT_PRICE_PER_CHIP_HOUR,
        gpu_model: GPUDataclass = DEFAULT_GPU_MODEL,
    ) -> pd.DataFrame:
        """Calculate annual CapEx schedule broken down by asset type.

        Returns:
            DataFrame with columns: year, chip_capex, datacenter_capex, power_capex, total_capex
        """
        revenue_df = self.profile.gpu_lease_rpo_schedule.copy()

        chips_df = revenue_to_chips_needed(revenue_df, price_per_chip_hour)
        chips_df = calculate_chip_delta(chips_df)

        capex_df = calculate_capex_from_chips(chips_df, gpu_model)

        return capex_df

    def calculate_shortfall(self) -> pd.DataFrame:
        """Calculate annual funding shortfall, defined as AI CapEx minus AI Revenue.

        Uses self.capex_schedule which must be computed first.
        """
        revenue_df = self.profile.gpu_lease_rpo_schedule.copy()
        shortfall_df = _calculate_shortfall(revenue_df, self.capex_schedule)

        return shortfall_df

    def calculate_net_ppe_at_year(
        self,
        year: int,
        chips_life: int = 5,
        datacenter_life: int = 20,
        power_life: int = 25
    ) -> float:
        """Calculate net PP&E (property, plant, equipment) book value at end of specified year.

        Args:
            year: Year to calculate net PP&E (e.g., 2029 for entering 2030)
            chips_life: Useful life of chips in years (default 5)
            datacenter_life: Useful life of datacenter in years (default 20)
            power_life: Useful life of power infrastructure in years (default 25)

        Returns:
            Net book value of PP&E after accumulated depreciation
        """
        # Use self.capex_schedule directly
        capex_df = self.capex_schedule[['year', 'chip_capex', 'datacenter_capex', 'power_capex']].copy()
        capex_df['year'] = capex_df['year'].astype(int)
        capex_df = capex_df.set_index('year')
        capex_df.columns = ['chips', 'datacenter', 'power']

        useful_life_series = pd.Series({
            'chips': chips_life,
            'datacenter': datacenter_life,
            'power': power_life
        })

        depreciation_schedule = CapexDepreciationSchedule(
            depreciation_accounting_schedule=capex_df,
            useful_life_series=useful_life_series
        )

        # Calculate cumulative depreciation up to and including the specified year
        total_depreciation_series = depreciation_schedule.get_total_depreciation_by_year()
        cumulative_depreciation = total_depreciation_series[total_depreciation_series.index <= year].sum()

        # Calculate gross PP&E (total CapEx spent)
        gross_ppe = capex_df.sum().sum()

        # Net PP&E = Gross PP&E - Accumulated Depreciation
        net_ppe = gross_ppe - cumulative_depreciation

        return net_ppe


def main():
    """Run and print the company funding analysis for Oracle."""
    from src.constants.company_financing_profiles import ORACLE_GPU_CONTRACTS

    model = CompanyFundingModel(profile=ORACLE_PROFILE, contracts=ORACLE_GPU_CONTRACTS)

    print("\nDebt Schedule (to cover shortfall):")
    print(model.debt_schedule)


if __name__ == "__main__":
    main()

__all__ = ["CompanyFundingModel", "main"]

