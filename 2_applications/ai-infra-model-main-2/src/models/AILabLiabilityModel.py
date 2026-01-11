"""AI Lab liability model for cloud contract analysis.

Tracks multiple cloud contracts per AI lab and generates payment schedules
to compare against revenue growth projections.
"""

import dataclasses

import pandas as pd

from src.constants.ai_lab_profiles import AILabProfile
from src.constants.cloud_contracts import combine_contract_payment_schedules


class AILabLiabilityModel:
    """Model for analyzing AI lab cloud contract liabilities vs revenue coverage."""

    def __init__(self,
                 profile: AILabProfile,
                 initial_annual_growth_rate: float = 1.25,
                 growth_decay_factor: float = 0.75,
                 chip_purchase_growth_rate: float = 0.30):
        """Initialize with AI lab profile and tunable modeling assumptions."""
        self.profile = profile

        # Create a new list of contract objects with the chip purchase growth rate applied.
        # This is a robust way to avoid side effects and ensure the rate is used.
        ramped_contracts = [
            dataclasses.replace(contract, chip_purchase_growth_rate=chip_purchase_growth_rate)
            for contract in self.profile.contracts
        ]

        # Get the detailed, per-provider payment schedule using the updated ramp rate
        self.payment_schedule_df = combine_contract_payment_schedules(ramped_contracts)

        # Create the annual total liability DataFrame by aggregating the detailed schedule
        liabilities_df = self.payment_schedule_df.groupby('year', as_index=False)['payment'].sum().rename(
            columns={'payment': 'total_liability'}
        )

        # Determine the latest year we need to project to
        if not liabilities_df.empty:
            projection_end_year = liabilities_df['year'].max()
        else:
            projection_end_year = self.profile.revenue_base_year

        # Generate revenue projections from the profile's base year to the end of the contract period
        revenue_projections_df = self.calculate_revenue_projections(
            base_revenue=self.profile.base_revenue,
            revenue_base_year=self.profile.revenue_base_year,
            initial_annual_growth_rate=initial_annual_growth_rate,
            growth_decay_factor=growth_decay_factor,
            projection_end_year=projection_end_year
        )

        # Merge liabilities with revenue projections - only include years with contracts
        analysis_df = liabilities_df.merge(revenue_projections_df, on='year', how='left')

        coverage_metrics_df = self.calculate_coverage_metrics(analysis_df)
        analysis_df = analysis_df.merge(coverage_metrics_df, on='year', how='left')

        self.analysis_df = analysis_df

    def calculate_revenue_projections(
        self,
        base_revenue: float,
        revenue_base_year: int,
        initial_annual_growth_rate: float,
        growth_decay_factor: float,
        projection_end_year: int
    ) -> pd.DataFrame:
        """
        Calculates a continuous block of revenue projections from a base year to an end year.
        """
        if projection_end_year < revenue_base_year:
            return pd.DataFrame(columns=['year', 'projected_revenue', 'applied_growth_rate'])

        years = list(range(revenue_base_year, projection_end_year + 1))
        revenues = [base_revenue]
        growth_rates = [None]

        current_growth = initial_annual_growth_rate

        for i in range(1, len(years)):
            revenues.append(revenues[-1] * (1 + current_growth))
            growth_rates.append(current_growth)
            current_growth *= growth_decay_factor

        return pd.DataFrame({
            'year': years,
            'projected_revenue': revenues,
            'applied_growth_rate': growth_rates
        })

    def calculate_coverage_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate coverage ratios and surplus/deficit from liability and revenue data."""
        if 'total_liability' not in df.columns or 'projected_revenue' not in df.columns:
            raise ValueError("DataFrame must contain 'total_liability' and 'projected_revenue' columns")

        # Get unique year-level data for coverage calculations
        year_level_df = df[['year', 'total_liability', 'projected_revenue']].drop_duplicates()

        if (year_level_df['total_liability'] == 0).any():
            raise ValueError("Total liability for any given year cannot be zero, as contracts must be present.")

        year_level_df['coverage_ratio'] = year_level_df['projected_revenue'] / year_level_df['total_liability']
        year_level_df['surplus_or_deficit'] = year_level_df['projected_revenue'] - year_level_df['total_liability']

        return year_level_df[['year', 'coverage_ratio', 'surplus_or_deficit']]


if __name__ == "__main__":
    from src.constants.ai_lab_profiles import OPENAI_PROFILE

    # --- Run Analysis for OpenAI ---
    # Tunable parameters are now passed directly to the model
    model = AILabLiabilityModel(
        profile=OPENAI_PROFILE,
        initial_annual_growth_rate=1.25, # 125%
        growth_decay_factor=0.66,      # Growth slows each year
        chip_purchase_growth_rate=0.30   # Chip purchases grow 30% annually
    )

    print("--- OpenAI AI Lab Liability Analysis ---")
    print(model.analysis_df)

