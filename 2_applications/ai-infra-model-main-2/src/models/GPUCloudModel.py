import numpy as np
import numpy_financial as npf
import pandas as pd

from src.constants.common import HOURS_PER_MONTH

# Local imports
from src.constants.gpu_dataclass import GPUDataclass
from src.utils.gpu_cloud_helpers import (
    calculate_gpu_cloud_contract_revenue,
    calculate_monthly_datacenter_personnel_cost,
    calculate_total_power_per_gpu,
    calculate_upfront_compute_hardware_cost,
)

 # HOURS_PER_MONTH imported from common constants

class GPUCloudModel:

    #region Column Constants
    # Core business columns
    COST_COLUMNS = [
        "gpu_purchase_cost",
        "installation_cost",
        "electricity_cost",
        "datacenter_rent",
        "personnel_cost",
        "insurance_cost",
        "sga_cost",
        "total_opex",
    ]
    REVENUE_COLUMNS = [
        "contract_revenue"
    ]
    # Financial reporting columns
    PROFITABILITY_COLUMNS = [
        "ebitda",
        "ebit",
        "pre_tax_income",
        "tax_expense",
        "net_income",
        "ebitda_margin",
        "ebit_margin",
        "depreciation"
    ]
    CASH_FLOW_COLUMNS = [
        "operating_cash_flow",
        "financing_cash_flow",  # Only used if USE_DEBT_FINANCING
        "net_cash_flow",
        "cumulative_cash_flow"
    ]
    DEBT_COLUMNS = [
        "interest_expense",
        "loan_payment",
        "principal_paydown",
        "loan_proceeds",
        "remaining_loan_balance"
    ]
    # We also have a "month" and "lease_period" column in the DF
    #endregion

    def __init__(
        self,
        # Hardware Configuration
        gpu_count: int,
        gpu_model: GPUDataclass,
        quantization_format: str,
        installation_cost_per_gpu: float,
        chip_price: float,

        # Financial Parameters
        use_debt_financing: bool,
        total_deal_years: float,
        enable_second_lease: bool,
        first_lease_term_months: float,
        second_lease_term_months: float,
        second_lease_discount_multiplier: float,
        chip_financing_interest_rate: float,

        # Operational Parameters
        gpu_utilization: float,
        sfu: float,
        pue: float,
        setup_time_months: float,

        # Cost Parameters
        electricity_cost_per_kwh_in_dollars: float,
        datacenter_rent_per_kw: float,
        neocloud_gpu_cost_per_chip_hour: float,

        # Overhead
        ftes_per_1000_gpus: float,
        fte_annual_cost: float,
        insurance_rate: float,
        sga_rate: float,
        tax_rate: float,
    ):
        # Hardware Configuration
        self.GPU_COUNT = gpu_count
        self.GPU_MODEL_USED = gpu_model
        self.QUANTIZATION_FORMAT = quantization_format
        self.INSTALLATION_COST_PER_GPU = installation_cost_per_gpu
        self.CHIP_PRICE = chip_price

        # Financial Parameters
        self.USE_DEBT_FINANCING = use_debt_financing
        self.TOTAL_DEAL_YEARS = total_deal_years
        self.ENABLE_SECOND_LEASE = enable_second_lease
        self.FIRST_LEASE_TERM_MONTHS = first_lease_term_months
        self.SECOND_LEASE_TERM_MONTHS = second_lease_term_months if enable_second_lease else 0.0
        self.SECOND_LEASE_DISCOUNT_MULTIPLIER = second_lease_discount_multiplier if enable_second_lease else 0.0
        self.CHIP_FINANCING_INTEREST_RATE = chip_financing_interest_rate

        # Operational Parameters
        self.GPU_UTILIZATION = gpu_utilization
        self.SFU = sfu
        self.PUE = pue
        self.SETUP_TIME_MONTHS = setup_time_months

        # Cost Parameters
        self.ELECTRICITY_COST_PER_KWH_IN_DOLLARS = electricity_cost_per_kwh_in_dollars
        self.DATACENTER_RENT_PER_KW = datacenter_rent_per_kw
        self.NEOCLOUD_GPU_COST_PER_CHIP_HOUR = neocloud_gpu_cost_per_chip_hour

        # Overhead
        self.FTES_PER_1000_GPUS = ftes_per_1000_gpus
        self.FTE_ANNUAL_COST = fte_annual_cost
        self.INSURANCE_RATE = insurance_rate
        self.SGA_RATE = sga_rate
        self.TAX_RATE = tax_rate

        # Dervied

        # Time
        self.TOTAL_MONTHS = self.SETUP_TIME_MONTHS + self.FIRST_LEASE_TERM_MONTHS + self.SECOND_LEASE_TERM_MONTHS
        self.TOTAL_LEASE_MONTHS = self.FIRST_LEASE_TERM_MONTHS + self.SECOND_LEASE_TERM_MONTHS
        self.LEASE_HOURS = self.TOTAL_LEASE_MONTHS * (30.4 * 24)

        # Lease Revenue
        self.FIRST_LEASE_CONTRACT_VALUE = calculate_gpu_cloud_contract_revenue(
            self.GPU_COUNT,
            self.FIRST_LEASE_TERM_MONTHS * HOURS_PER_MONTH,
            self.NEOCLOUD_GPU_COST_PER_CHIP_HOUR
        )

        if self.ENABLE_SECOND_LEASE:
            self.SECOND_LEASE_CONTRACT_VALUE = calculate_gpu_cloud_contract_revenue(
                self.GPU_COUNT,
                self.SECOND_LEASE_TERM_MONTHS * HOURS_PER_MONTH,
                self.NEOCLOUD_GPU_COST_PER_CHIP_HOUR * self.SECOND_LEASE_DISCOUNT_MULTIPLIER
            )
        else:
            self.SECOND_LEASE_CONTRACT_VALUE = 0.0

        self.LIFETIME_CONTRACT_VALUE = self.FIRST_LEASE_CONTRACT_VALUE + self.SECOND_LEASE_CONTRACT_VALUE

        # Costs
        self.COMPUTE_COSTS = calculate_upfront_compute_hardware_cost(
            gpu_count=self.GPU_COUNT,
            nvda_gpu_price=self.CHIP_PRICE,
            other_compute_hw_price=self.GPU_MODEL_USED.other_compute_costs
        )
        self.POWER_REQUIRED_KW = self.GPU_COUNT * calculate_total_power_per_gpu(self.GPU_MODEL_USED.wattage, self.GPU_UTILIZATION, self.PUE) / 1000
        self.DATACENTER_RENT_MONTHLY = self.POWER_REQUIRED_KW * self.DATACENTER_RENT_PER_KW
        self.ELECTRICITY_COST_MONTHLY = self.POWER_REQUIRED_KW * self.ELECTRICITY_COST_PER_KWH_IN_DOLLARS * HOURS_PER_MONTH
        self.PERSONNEL_COST_MONTHLY = calculate_monthly_datacenter_personnel_cost(self.GPU_COUNT, self.FTES_PER_1000_GPUS, self.FTE_ANNUAL_COST)
        self.INSURANCE_COST_MONTHLY = (self.COMPUTE_COSTS["total_cost"] * self.INSURANCE_RATE) / 12


        # Setup the dataframe
        self._check_valid_values()
        self.df = self._setup_dataframe()

    def _check_valid_values(self):
        scalar_ranges = {
            'GPU_UTILIZATION': (0, 1),
            'SFU': (0, 1),
            'PUE': (1, float('inf')),  # Updated to ensure PUE is at least 1
            'GPU_COUNT': (1, float('inf')),
            'CHIP_PRICE': (0, float('inf')),
            'TOTAL_DEAL_YEARS': (1, float('inf')),
            'SECOND_LEASE_DISCOUNT_MULTIPLIER': (0, float('inf')),
            'CHIP_FINANCING_INTEREST_RATE': (0, float('inf')),
            'FIRST_LEASE_CONTRACT_VALUE': (0, float('inf')),
            'SECOND_LEASE_CONTRACT_VALUE': (0, float('inf')),
            'ELECTRICITY_COST_PER_KWH_IN_DOLLARS': (0, float('inf')),
            'SGA_RATE': (0, 1),  # Assuming SGA rate is a percentage between 0 and 1
            'FIRST_LEASE_TERM_MONTHS': (1, float('inf')),
            'SECOND_LEASE_TERM_MONTHS': (0, float('inf')),  # Allow 0 when second lease disabled
            'SETUP_TIME_MONTHS': (0, float('inf')),
        }

        for attr, (min_val, max_val) in scalar_ranges.items():
            value = getattr(self, attr)
            if not (min_val <= value <= max_val):
                raise ValueError(f"{attr} must be between {min_val} and {max_val}. Current value: {value}")

        if self.USE_DEBT_FINANCING and self.CHIP_FINANCING_INTEREST_RATE <= 0:
            raise ValueError(f"When debt financing is enabled, interest rate must be positive. Current value: {self.CHIP_FINANCING_INTEREST_RATE}")

    def _setup_dataframe(self):
        # Create an array of months from 0 to total months
        months = np.arange(int(self.TOTAL_MONTHS) + 1)

        # Helper function to categorize each month into a lease period
        def categorize_lease_period(month):
            if month == 0:
                return "initial_setup"
            if month <= int(self.SETUP_TIME_MONTHS):
                return "setup"

            lease_month = month - int(self.SETUP_TIME_MONTHS) # start tracking from the first lease month
            if lease_month <= int(self.FIRST_LEASE_TERM_MONTHS):
                return "first_lease"
            if lease_month <= int(self.FIRST_LEASE_TERM_MONTHS) + int(self.SECOND_LEASE_TERM_MONTHS):
                return "second_lease"

            return "post_lease" # This is not accounted for right now

        # Create DataFrame with month and lease period columns
        df = pd.DataFrame({
            "month": months,
            "lease_period": [categorize_lease_period(m) for m in months]
        })

        # Initialize all columns with zeros
        all_columns = (
            self.COST_COLUMNS +
            self.REVENUE_COLUMNS +
            self.PROFITABILITY_COLUMNS +
            self.DEBT_COLUMNS +  # Always initialize debt columns
            self.CASH_FLOW_COLUMNS
        )
        for column in all_columns:
            df[column] = 0.0

        return df

    def _process_initial_setup(self) -> None:
        initial_gpu_capex = self.COMPUTE_COSTS["total_cost"]
        installation_cost = self.INSTALLATION_COST_PER_GPU * self.GPU_COUNT

        # Record the costs
        self.df.at[0, "gpu_purchase_cost"] = -initial_gpu_capex
        self.df.at[0, "installation_cost"] = -installation_cost

    def _log_lease_revenue(self, contract_value: float, term_months: float, start_offset: float) -> None:
        """adds revenue for a lease period to the dataframe.
        
        Args:
            contract_value: Total value of the lease contract
            term_months: Duration of the lease in months
            start_offset: When the lease starts
        """
        revenue_per_period = contract_value / term_months
        start_month = int(self.SETUP_TIME_MONTHS) + int(start_offset)
        end_month = start_month + int(term_months)
        # Vectorized slice assignment
        self.df.loc[start_month:end_month - 1, "contract_revenue"] = revenue_per_period

    def _process_opex(self) -> None:
        # Set all setup period values to zero except datacenter rent
        setup_mask = (self.df['month'] > 0) & (self.df['month'] <= self.SETUP_TIME_MONTHS)
        self.df.loc[setup_mask, 'datacenter_rent'] = -self.DATACENTER_RENT_MONTHLY
        self.df.loc[setup_mask, ['electricity_cost', 'personnel_cost', 'insurance_cost', 'sga_cost']] = 0

        # Process operational period
        operational_mask = (self.df['month'] > int(self.SETUP_TIME_MONTHS)) & (self.df['month'] < len(self.df))
        self.df.loc[operational_mask, "datacenter_rent"] = -self.DATACENTER_RENT_MONTHLY
        self.df.loc[operational_mask, "electricity_cost"] = -self.ELECTRICITY_COST_MONTHLY
        self.df.loc[operational_mask, "personnel_cost"] = -self.PERSONNEL_COST_MONTHLY
        self.df.loc[operational_mask, "insurance_cost"] = -self.INSURANCE_COST_MONTHLY
        self.df.loc[operational_mask, "sga_cost"] = -self.SGA_RATE * self.df.loc[operational_mask, "contract_revenue"]

        # Calculate total opex for all lease periods
        total_opex = (
            self.df.loc[:, "electricity_cost"] +
            self.df.loc[:, "datacenter_rent"] +
            self.df.loc[:, "personnel_cost"] +
            self.df.loc[:, "insurance_cost"] +
            self.df.loc[:, "sga_cost"]
        )
        self.df.loc[:, "total_opex"] = total_opex

    def _process_financing(self) -> None:
        if not self.USE_DEBT_FINANCING:
            return

        initial_loan_amount = self.COMPUTE_COSTS["total_cost"] + (self.INSTALLATION_COST_PER_GPU * self.GPU_COUNT)
        monthly_interest_rate = (1 + self.CHIP_FINANCING_INTEREST_RATE) ** (1/12) - 1
        setup_end = int(self.SETUP_TIME_MONTHS)
        loan_term = int(self.FIRST_LEASE_TERM_MONTHS)

        # Initialize month 0 with the initial loan proceeds
        self.df.at[0, "loan_proceeds"] = initial_loan_amount
        self.df.at[0, "financing_cash_flow"] = initial_loan_amount
        self.df.at[0, "remaining_loan_balance"] = initial_loan_amount
        self.df.at[0, "principal_paydown"] = 0
        self.df.at[0, "interest_expense"] = 0
        self.df.at[0, "loan_payment"] = 0

        current_balance = initial_loan_amount

        # During the setup period, accrue interest without making payments
        for month in range(1, setup_end + 1):
            interest = current_balance * monthly_interest_rate
            self.df.at[month, "interest_expense"] = -interest  # Negative for expense
            self.df.at[month, "principal_paydown"] = 0
            self.df.at[month, "loan_payment"] = 0
            current_balance = current_balance + interest  # Add interest to balance during setup
            self.df.at[month, "remaining_loan_balance"] = current_balance
            self.df.at[month, "financing_cash_flow"] = 0

        # Calculate fixed monthly payment based on post-setup balance
        # npf.pmt returns negative payment for positive PV, which is what we want
        monthly_payment = npf.pmt(monthly_interest_rate, loan_term, current_balance)

        # Make regular payments during the payment period
        for month in range(setup_end + 1, setup_end + loan_term + 1):
            # Calculate interest portion (negative for expense)
            interest = current_balance * monthly_interest_rate
            interest_component = -interest

            # Calculate principal portion (negative for paydown)
            # monthly_payment is already negative, so we add the negative interest
            principal_component = monthly_payment - interest_component

            # Record all components
            self.df.at[month, "loan_payment"] = monthly_payment
            self.df.at[month, "interest_expense"] = interest_component
            self.df.at[month, "principal_paydown"] = principal_component

            # Update balance by adding the negative principal component
            current_balance = current_balance + principal_component
            self.df.at[month, "remaining_loan_balance"] = max(0, current_balance)  # Prevent small negative balances
            self.df.at[month, "financing_cash_flow"] = monthly_payment

            # Verify the payment components sum correctly
            payment_sum = interest_component + principal_component
            assert abs(payment_sum - monthly_payment) < 0.01, \
                f"Payment components don't sum to monthly payment at month {month}. Sum: {payment_sum}, Payment: {monthly_payment}"

        # Zero out remaining months
        for month in range(setup_end + loan_term + 1, len(self.df)):
            self.df.at[month, "remaining_loan_balance"] = 0
            self.df.at[month, "loan_payment"] = 0
            self.df.at[month, "interest_expense"] = 0
            self.df.at[month, "principal_paydown"] = 0
            self.df.at[month, "financing_cash_flow"] = 0

    def _process_ddb_depreciation(self) -> None:
        """Calculate depreciation using straight-line method over the total deal years.
        Implemented using vectorized operations for better performance."""

        initial_cost = self.COMPUTE_COSTS["total_cost"]
        setup_end = int(self.SETUP_TIME_MONTHS) + 1

        # Use total deal years for depreciation period
        depreciation_periods = int(self.TOTAL_DEAL_YEARS * 12)  # Convert years to months
        monthly_depreciation = initial_cost / depreciation_periods

        # Create a mask for the depreciation period (after setup, for the full deal duration)
        depreciation_mask = (self.df['month'] >= setup_end) & (self.df['month'] < setup_end + depreciation_periods)
        self.df.loc[depreciation_mask, "depreciation"] = -monthly_depreciation

    def _process_cash_flows(self):
        self._process_initial_setup()
        self._log_lease_revenue(self.FIRST_LEASE_CONTRACT_VALUE, self.FIRST_LEASE_TERM_MONTHS, 1)

        # Only log second lease if enabled
        if self.ENABLE_SECOND_LEASE:
            self._log_lease_revenue(self.SECOND_LEASE_CONTRACT_VALUE, self.SECOND_LEASE_TERM_MONTHS, self.FIRST_LEASE_TERM_MONTHS + 1)

        self._process_opex()

        if self.USE_DEBT_FINANCING:
            self._process_financing()

        self._calculate_cash_flow_columns()

    def _calculate_cash_flow_columns(self) -> None:
        """Calculate all cash flow metrics for each period.
        - Operating cash flow: Revenue + OpEx + CapEx
        - Financing cash flow: Loan proceeds - Loan payments
        - Net cash flow: Operating + Financing
        - Cumulative cash flow: Running total of net cash flows
        """
        # Operating cash flow includes revenue, opex, and initial capex
        self.df["operating_cash_flow"] = (
            self.df["contract_revenue"] +           # Revenue
            self.df["gpu_purchase_cost"] +         # Hardware CapEx
            self.df["installation_cost"] +         # Installation CapEx
            self.df["total_opex"]                  # All OpEx items combined
        )

        # Financing cash flow is loan proceeds minus payments (will be 0 if not using debt)
        self.df["financing_cash_flow"] = (
            self.df["loan_proceeds"] +         # Money received from loans
            self.df["loan_payment"]           # Principal + interest payments
        )

        # Net cash flow is operating plus financing
        self.df["net_cash_flow"] = (
            self.df["operating_cash_flow"] +
            self.df["financing_cash_flow"]
        )

        # Cumulative cash flow is the running total
        self.df["cumulative_cash_flow"] = self.df["net_cash_flow"].cumsum()

    def _calculate_profitability_metrics(self) -> None:
        """
        Calculate key profitability metrics for each month following proper income statement flow:
        1. EBITDA (Earnings Before Interest, Taxes, Depreciation, and Amortization)
        2. EBIT (Earnings Before Interest and Taxes) = EBITDA - Depreciation
        3. Pre-tax Income = EBIT - Interest
        4. Net Income = Pre-tax Income - Taxes
        """
        # Calculate EBITDA (Revenue - OpEx)
        self.df['ebitda'] = self.df['contract_revenue'] + self.df['total_opex'] # adding a negative

        # Calculate TRUE EBIT (EBITDA - Depreciation only, before interest)
        self.df['ebit'] = self.df['ebitda'] + self.df['depreciation']  # depreciation is negative

        # Calculate Pre-tax Income (EBIT - Interest)
        self.df['pre_tax_income'] = self.df['ebit'] + self.df['interest_expense']  # interest_expense is negative

        # Calculate Tax Expense (negative value)
        # Only apply taxes on positive pre-tax income
        self.df['tax_expense'] = np.where(
            self.df['pre_tax_income'] > 0,
            -self.df['pre_tax_income'] * self.TAX_RATE,
            0
        )

        # Calculate Net Income (Pre-tax Income - Taxes)
        self.df['net_income'] = self.df['pre_tax_income'] + self.df['tax_expense']  # tax_expense is negative

        # Calculate margins (will automatically be NaN where revenue is zero)
        self.df['ebitda_margin'] = 100 * self.df['ebitda'] / self.df['contract_revenue']
        self.df['ebit_margin'] = 100 * self.df['ebit'] / self.df['contract_revenue']

    def get_summary_metrics(self) -> dict:
        """
        Returns a dictionary of summary metrics aggregated from the model.
        """
        return {
            # Revenue metrics
            "total_revenue": self.df["contract_revenue"].sum(),
            "first_lease_revenue": self.FIRST_LEASE_CONTRACT_VALUE,
            "second_lease_revenue": self.SECOND_LEASE_CONTRACT_VALUE,

            # Capex metrics
            "gpu_hardware_cost": abs(self.df["gpu_purchase_cost"].sum()),
            "installation_cost": abs(self.df["installation_cost"].sum()),
            "total_capex": abs(self.df["gpu_purchase_cost"].sum()) + abs(self.df["installation_cost"].sum()),
            # OpEx metrics
            "total_opex": abs(self.df["total_opex"].sum()),
            "total_datacenter_rent": abs(self.df["datacenter_rent"].sum()),
            "total_electricity_cost": abs(self.df["electricity_cost"].sum()),
            "total_personnel_cost": abs(self.df["personnel_cost"].sum()),
            "total_insurance_cost": abs(self.df["insurance_cost"].sum()),
            "total_sga_cost": abs(self.df["sga_cost"].sum()),

            # Financing metrics
            "total_interest_paid": abs(self.df["interest_expense"].sum()) if self.USE_DEBT_FINANCING else 0,
            "total_principal_paid": abs(self.df["principal_paydown"].sum()) if self.USE_DEBT_FINANCING else 0,

            # Profitability metrics
            "total_ebitda": self.df["ebitda"].sum(),
            "total_depreciation": abs(self.df["depreciation"].sum()),
            "total_ebit": self.df["ebit"].sum(),
            "total_pre_tax_income": self.df["pre_tax_income"].sum(),
            "total_tax_expense": abs(self.df["tax_expense"].sum()),
            "total_net_income": self.df["net_income"].sum(),
            "avg_ebitda_margin": (self.df["ebitda"].sum() / self.df["contract_revenue"].sum()) * 100,
            "avg_ebit_margin": (self.df["ebit"].sum() / self.df["contract_revenue"].sum()) * 100,

            # Cash flow metrics
            "final_cumulative_cash_flow": self.df["cumulative_cash_flow"].iloc[-1],
            "min_cumulative_cash_flow": self.df["cumulative_cash_flow"].min(),
            "max_monthly_cash_outflow": abs(self.df["net_cash_flow"].min()),
        }

    def run_model(self):
        self._process_cash_flows()
        self._process_ddb_depreciation()
        self._calculate_profitability_metrics()

        irr = npf.irr(self.df["net_cash_flow"])
        annual_irr = (1 + irr) ** 12 - 1
        return irr, annual_irr

    # TODO I'd assume we need some getters and setters for these types of things

    # def get_exaflops(self) -> tuple[float, float]:
    #     total_exaflops = calculate_total_exaflops(
    #         self.GPU_COUNT,
    #         self.FLOPS_PER_GPU,
    #         self.GPU_UTILIZATION
    #     )
    #     total_cost = abs(self.df['total_opex'].sum())
    #     exaflops_per_dollar = total_exaflops / total_cost if total_cost > 0 else 0
    #     return total_exaflops, exaflops_per_dollar





# -------------------------------------------------------------------
# Example Usage
# -------------------------------------------------------------------

if __name__ == "__main__":
    # TODO integrate scenarios
    from src.constants.gpu_cloud_scenarios_dataclass import SCENARIOS
    scenario = SCENARIOS["Default"]
    calculator = GPUCloudModel(
        gpu_count=scenario.gpu_count,
        gpu_model=scenario.gpu_model,
        quantization_format=scenario.quantization_format,
        installation_cost_per_gpu=scenario.installation_cost_per_gpu,
        chip_price=scenario.chip_price,
        use_debt_financing=scenario.use_debt_financing,
        total_deal_years=scenario.total_deal_years,
        enable_second_lease=scenario.enable_second_lease,
        first_lease_term_months=scenario.first_lease_term_months,
        second_lease_term_months=scenario.second_lease_term_months,
        second_lease_discount_multiplier=scenario.second_lease_discount_multiplier,
        chip_financing_interest_rate=scenario.chip_financing_interest_rate,
        gpu_utilization=scenario.gpu_utilization,
        sfu=scenario.sfu,
        pue=scenario.pue,
        setup_time_months=scenario.setup_time_months,
        electricity_cost_per_kwh_in_dollars=scenario.electricity_cost_per_kwh_in_dollars,
        datacenter_rent_per_kw=scenario.datacenter_rent_per_kw,
        neocloud_gpu_cost_per_chip_hour=scenario.neocloud_gpu_cost_per_chip_hour,
        ftes_per_1000_gpus=scenario.ftes_per_1000_gpus,
        fte_annual_cost=scenario.fte_annual_cost,
        insurance_rate=scenario.insurance_rate,
        sga_rate=scenario.sga_rate
    )

    irr, annual_irr = calculator.run_model()

    print(f"Personnel: {(calculator.GPU_COUNT / 1000) * 4:.2f} FTEs")
    print(f"Insurance Rate: {calculator.INSURANCE_RATE * 100:.1f}% of GPU hardware cost")
    print(f"SG&A Overhead Rate: {calculator.SGA_RATE * 100:.1f}% of revenue")

    print("\nCash Flows (in dollars):")
    df = calculator.df[calculator.DEBT_COLUMNS + ['month']]
    print(df)

    if irr is not None:
        print("\nInternal Rate of Return (IRR):")
        print(f"Monthly: {irr * 100:.2f}%")
        print(f"Annual: {annual_irr * 100:.2f}%")

    print("\nCash Flow Columns:")
    print(calculator.df.columns.tolist())

    # Print financing details
    print("\nFinancing Test Results:")
    print("Initial loan amount:", calculator.df.at[0, "loan_proceeds"])
    df = calculator.df[calculator.DEBT_COLUMNS + ['month']]
    print("\nMonth by month breakdown:")
    print("Month | Balance | Payment | Interest | Principal")
    print("-" * 60)

    for month in range(int(calculator.TOTAL_MONTHS + 1)):
        balance = calculator.df.at[month, "remaining_loan_balance"]
        payment = calculator.df.at[month, "loan_payment"]
        interest = calculator.df.at[month, "interest_expense"]
        principal = calculator.df.at[month, "principal_paydown"]

        print(f"{month:5d} | {balance:8.0f} | {payment:8.0f} | {interest:8.0f} | {principal:8.0f}")
