import pytest

from src.utils.gpu_cloud_helpers import calculate_curtailment_equivalency


def test_curtailment_equivalency_basic_case() -> None:
    results = calculate_curtailment_equivalency(
        server_cost_total=43_800.0,
        server_power_kw=2.0,
        electricity_base_cost_mwh=60.0,
        downtime_hours_per_day=2.0,
    )

    assert results["downtime_percentage"] == pytest.approx(2.0 / 24.0)
    assert results["depreciation_cost_per_hour"] == pytest.approx(1.0)
    assert results["lost_depreciation_dollars_per_day"] == pytest.approx(2.0)
    assert results["electricity_saved_dollars_per_day"] == pytest.approx(0.24)
    assert results["net_dollars_per_day"] == pytest.approx(1.76)
    assert results["energy_consumed_mwh_per_day"] == pytest.approx(0.044)
    assert results["equivalent_price_increase_per_mwh"] == pytest.approx(40.0)
    assert results["equivalent_price_level_per_mwh"] == pytest.approx(100.0)
    assert results["curtailment_price_threshold_per_mwh"] == pytest.approx(500.0)


def test_curtailment_equivalency_zero_downtime() -> None:
    results = calculate_curtailment_equivalency(
        server_cost_total=43_800.0,
        server_power_kw=2.0,
        electricity_base_cost_mwh=40.0,
        downtime_hours_per_day=0.0,
    )

    assert results["equivalent_price_increase_per_mwh"] == 0.0
    assert results["equivalent_price_level_per_mwh"] == pytest.approx(40.0)
    assert results["curtailment_price_threshold_per_mwh"] == pytest.approx(500.0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"server_cost_total": -1.0, "server_power_kw": 1.0, "electricity_base_cost_mwh": 0.0, "downtime_hours_per_day": 1.0},
        {"server_cost_total": 1.0, "server_power_kw": 0.0, "electricity_base_cost_mwh": 0.0, "downtime_hours_per_day": 1.0},
        {"server_cost_total": 1.0, "server_power_kw": 1.0, "electricity_base_cost_mwh": -1.0, "downtime_hours_per_day": 1.0},
        {"server_cost_total": 1.0, "server_power_kw": 1.0, "electricity_base_cost_mwh": 0.0, "downtime_hours_per_day": -0.5},
        {"server_cost_total": 1.0, "server_power_kw": 1.0, "electricity_base_cost_mwh": 0.0, "downtime_hours_per_day": 25.0},
        {"server_cost_total": 1.0, "server_power_kw": 1.0, "electricity_base_cost_mwh": 0.0, "downtime_hours_per_day": 1.0, "depreciation_years": 0},
    ],
)
def test_curtailment_breakeven_invalid_inputs(kwargs) -> None:
    with pytest.raises(ValueError):
        calculate_curtailment_equivalency(**kwargs)
