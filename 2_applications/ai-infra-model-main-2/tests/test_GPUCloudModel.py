from __future__ import annotations

import numpy_financial as npf
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from src.constants.common import HOURS_PER_MONTH
from src.constants.gpu_dataclass import GPUDataclass
from src.models.GPUCloudModel import GPUCloudModel

# Deterministic scenario parameters shared across tests
GPU_COUNT = 2
FIRST_LEASE_TERM_MONTHS = 12
SECOND_LEASE_TERM_MONTHS = 0
SETUP_TIME_MONTHS = 1
TOTAL_DEAL_YEARS = 1.0
NEOCLOUD_RATE = 0.50
INSTALLATION_COST_PER_GPU = 100.0
CHIP_PRICE = 1_000.0
FTES_PER_1000_GPUS = 1.0
FTE_ANNUAL_COST = 120_000.0
ELECTRICITY_COST_PER_KWH = 0.10
DATACENTER_RENT_PER_KW = 5.0
GPU_UTILIZATION = 0.5
PUE = 1.1
SGA_RATE = 0.05
INSURANCE_RATE = 0.0
TAX_RATE = 0.21
SFU = 1.0
QUANTIZATION = "fp16"

# The model requires a GPU dataclass instance; we create a minimal synthetic one for the tests.
TEST_GPU = GPUDataclass(
    name="TestGPU",
    wattage=1_000.0,
    ai_accelerator_price=CHIP_PRICE,
    other_compute_costs=0.0,
    hourly_revenue=None,
    quantized_performance=None,
    data_sources=None,
)


EXPECTED_TIMELINE_COLUMNS = [
    "month",
    "contract_revenue",
    "gpu_purchase_cost",
    "installation_cost",
    "datacenter_rent",
    "electricity_cost",
    "personnel_cost",
    "insurance_cost",
    "sga_cost",
    "total_opex",
    "depreciation",
    "ebitda",
    "ebit",
    "pre_tax_income",
    "tax_expense",
    "net_income",
    "loan_proceeds",
    "loan_payment",
    "interest_expense",
    "principal_paydown",
    "remaining_loan_balance",
    "operating_cash_flow",
    "financing_cash_flow",
    "net_cash_flow",
    "cumulative_cash_flow",
]


def build_gpu_cloud_model() -> GPUCloudModel:
    """Instantiate the GPUCloudModel with the deterministic scenario."""
    return GPUCloudModel(
        gpu_count=GPU_COUNT,
        gpu_model=TEST_GPU,
        quantization_format=QUANTIZATION,
        installation_cost_per_gpu=INSTALLATION_COST_PER_GPU,
        chip_price=CHIP_PRICE,
        use_debt_financing=False,
        total_deal_years=TOTAL_DEAL_YEARS,
        enable_second_lease=False,
        first_lease_term_months=FIRST_LEASE_TERM_MONTHS,
        second_lease_term_months=SECOND_LEASE_TERM_MONTHS,
        second_lease_discount_multiplier=0.0,
        chip_financing_interest_rate=0.05,
        gpu_utilization=GPU_UTILIZATION,
        sfu=SFU,
        pue=PUE,
        setup_time_months=SETUP_TIME_MONTHS,
        electricity_cost_per_kwh_in_dollars=ELECTRICITY_COST_PER_KWH,
        datacenter_rent_per_kw=DATACENTER_RENT_PER_KW,
        neocloud_gpu_cost_per_chip_hour=NEOCLOUD_RATE,
        ftes_per_1000_gpus=FTES_PER_1000_GPUS,
        fte_annual_cost=FTE_ANNUAL_COST,
        insurance_rate=INSURANCE_RATE,
        sga_rate=SGA_RATE,
        tax_rate=TAX_RATE,
    )


def build_expected_timeline() -> pd.DataFrame:
    """Construct the expected month-by-month results for the deterministic scenario."""
    total_months = int(SETUP_TIME_MONTHS + FIRST_LEASE_TERM_MONTHS + SECOND_LEASE_TERM_MONTHS)
    months = range(total_months + 1)

    total_contract_value = GPU_COUNT * FIRST_LEASE_TERM_MONTHS * HOURS_PER_MONTH * NEOCLOUD_RATE
    monthly_revenue = total_contract_value / FIRST_LEASE_TERM_MONTHS

    total_compute_cost = GPU_COUNT * (CHIP_PRICE + TEST_GPU.other_compute_costs)
    installation_cost_total = GPU_COUNT * INSTALLATION_COST_PER_GPU

    power_per_gpu_watts = TEST_GPU.wattage * GPU_UTILIZATION * PUE
    power_required_kw = GPU_COUNT * power_per_gpu_watts / 1_000

    datacenter_rent_monthly = -power_required_kw * DATACENTER_RENT_PER_KW
    electricity_cost_monthly = -power_required_kw * ELECTRICITY_COST_PER_KWH * HOURS_PER_MONTH
    personnel_cost_monthly = -((GPU_COUNT / 1_000) * FTES_PER_1000_GPUS * FTE_ANNUAL_COST) / 12
    insurance_cost_monthly = -(total_compute_cost * INSURANCE_RATE) / 12
    sga_cost_monthly = -SGA_RATE * monthly_revenue

    setup_end_month = int(SETUP_TIME_MONTHS) + 1
    depreciation_period_months = int(TOTAL_DEAL_YEARS * 12)
    monthly_depreciation = -(total_compute_cost / depreciation_period_months)

    data: dict[str, list[float]] = {column: [] for column in EXPECTED_TIMELINE_COLUMNS}
    cumulative_cash_flow = 0.0

    for month in months:
        contract_revenue = monthly_revenue if month >= setup_end_month else 0.0
        gpu_purchase_cost = -total_compute_cost if month == 0 else 0.0
        installation_cost = -installation_cost_total if month == 0 else 0.0

        if month == 0:
            datacenter_rent = 0.0
            electricity_cost = 0.0
            personnel_cost = 0.0
            insurance_cost = 0.0
            sga_cost = 0.0
        elif month <= SETUP_TIME_MONTHS:
            datacenter_rent = datacenter_rent_monthly
            electricity_cost = 0.0
            personnel_cost = 0.0
            insurance_cost = 0.0
            sga_cost = 0.0
        else:
            datacenter_rent = datacenter_rent_monthly
            electricity_cost = electricity_cost_monthly
            personnel_cost = personnel_cost_monthly
            insurance_cost = insurance_cost_monthly
            sga_cost = sga_cost_monthly

        total_opex = datacenter_rent + electricity_cost + personnel_cost + insurance_cost + sga_cost

        if setup_end_month <= month < setup_end_month + depreciation_period_months:
            depreciation = monthly_depreciation
        else:
            depreciation = 0.0

        ebitda = contract_revenue + total_opex
        ebit = ebitda + depreciation
        pre_tax_income = ebit
        tax_expense = -pre_tax_income * TAX_RATE if pre_tax_income > 0 else 0.0
        net_income = pre_tax_income + tax_expense

        operating_cash_flow = contract_revenue + gpu_purchase_cost + installation_cost + total_opex
        financing_cash_flow = 0.0
        net_cash_flow = operating_cash_flow + financing_cash_flow
        cumulative_cash_flow += net_cash_flow

        # Populate row values
        data["month"].append(month)
        data["contract_revenue"].append(contract_revenue)
        data["gpu_purchase_cost"].append(gpu_purchase_cost)
        data["installation_cost"].append(installation_cost)
        data["datacenter_rent"].append(datacenter_rent)
        data["electricity_cost"].append(electricity_cost)
        data["personnel_cost"].append(personnel_cost)
        data["insurance_cost"].append(insurance_cost)
        data["sga_cost"].append(sga_cost)
        data["total_opex"].append(total_opex)
        data["depreciation"].append(depreciation)
        data["ebitda"].append(ebitda)
        data["ebit"].append(ebit)
        data["pre_tax_income"].append(pre_tax_income)
        data["tax_expense"].append(tax_expense)
        data["net_income"].append(net_income)
        data["loan_proceeds"].append(0.0)
        data["loan_payment"].append(0.0)
        data["interest_expense"].append(0.0)
        data["principal_paydown"].append(0.0)
        data["remaining_loan_balance"].append(0.0)
        data["operating_cash_flow"].append(operating_cash_flow)
        data["financing_cash_flow"].append(financing_cash_flow)
        data["net_cash_flow"].append(net_cash_flow)
        data["cumulative_cash_flow"].append(cumulative_cash_flow)

    return pd.DataFrame(data)


def build_expected_summary(timeline: pd.DataFrame) -> dict[str, float]:
    """Aggregate expected summary metrics for comparison with the model output."""
    total_revenue = timeline["contract_revenue"].sum()
    total_ebitda = timeline["ebitda"].sum()
    total_ebit = timeline["ebit"].sum()

    return {
        "total_revenue": total_revenue,
        "first_lease_revenue": total_revenue,
        "second_lease_revenue": 0.0,
        "gpu_hardware_cost": abs(timeline["gpu_purchase_cost"].sum()),
        "installation_cost": abs(timeline["installation_cost"].sum()),
        "total_capex": abs(timeline["gpu_purchase_cost"].sum()) + abs(timeline["installation_cost"].sum()),
        "total_opex": abs(timeline["total_opex"].sum()),
        "total_datacenter_rent": abs(timeline["datacenter_rent"].sum()),
        "total_electricity_cost": abs(timeline["electricity_cost"].sum()),
        "total_personnel_cost": abs(timeline["personnel_cost"].sum()),
        "total_insurance_cost": abs(timeline["insurance_cost"].sum()),
        "total_sga_cost": abs(timeline["sga_cost"].sum()),
        "total_interest_paid": abs(timeline["interest_expense"].sum()),
        "total_principal_paid": abs(timeline["principal_paydown"].sum()),
        "total_ebitda": total_ebitda,
        "total_depreciation": abs(timeline["depreciation"].sum()),
        "total_ebit": total_ebit,
        "total_pre_tax_income": timeline["pre_tax_income"].sum(),
        "total_tax_expense": abs(timeline["tax_expense"].sum()),
        "total_net_income": timeline["net_income"].sum(),
        "avg_ebitda_margin": (total_ebitda / total_revenue) * 100,
        "avg_ebit_margin": (total_ebit / total_revenue) * 100,
        "final_cumulative_cash_flow": timeline["cumulative_cash_flow"].iloc[-1],
        "min_cumulative_cash_flow": timeline["cumulative_cash_flow"].min(),
        "max_monthly_cash_outflow": abs(timeline["net_cash_flow"].min()),
    }


@pytest.fixture
def minimal_gpu_cloud_model_result():
    """Run the deterministic scenario and provide both actual and expected outputs."""
    model = build_gpu_cloud_model()
    irr, annual_irr = model.run_model()

    expected_timeline = build_expected_timeline()
    expected_summary = build_expected_summary(expected_timeline)
    expected_irr = npf.irr(expected_timeline["net_cash_flow"].to_numpy())
    expected_annual_irr = (1 + expected_irr) ** 12 - 1 if expected_irr is not None else None

    return {
        "model": model,
        "irr": irr,
        "annual_irr": annual_irr,
        "expected_timeline": expected_timeline,
        "expected_summary": expected_summary,
        "expected_irr": expected_irr,
        "expected_annual_irr": expected_annual_irr,
    }


def test_gpu_cloud_model_minimal_timeline_matches_expected(minimal_gpu_cloud_model_result):
    result = minimal_gpu_cloud_model_result
    model = result["model"]
    actual_timeline = model.df.loc[:, EXPECTED_TIMELINE_COLUMNS].reset_index(drop=True)

    assert_frame_equal(
        actual_timeline,
        result["expected_timeline"],
        atol=1e-9,
        check_like=False,
    )

    assert result["irr"] == pytest.approx(result["expected_irr"], rel=1e-9, abs=1e-9)
    assert result["annual_irr"] == pytest.approx(result["expected_annual_irr"], rel=1e-9, abs=1e-9)


def test_gpu_cloud_model_summary_metrics(minimal_gpu_cloud_model_result):
    result = minimal_gpu_cloud_model_result
    summary_metrics = result["model"].get_summary_metrics()

    for key, expected_value in result["expected_summary"].items():
        assert summary_metrics[key] == pytest.approx(expected_value, rel=1e-9, abs=1e-9)

    # Spot-check invariants that should hold in the deterministic scenario
    assert summary_metrics["total_interest_paid"] == 0.0
    assert summary_metrics["total_principal_paid"] == 0.0
    assert summary_metrics["second_lease_revenue"] == 0.0
