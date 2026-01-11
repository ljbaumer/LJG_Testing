from pathlib import Path

from src.constants.cloud_contracts import load_cloud_contracts_from_csv

DATA_PATH = Path("data/cloud_contracts/openai_announcements.csv")


def test_load_cloud_contracts_yields_expected_providers() -> None:
    contracts = load_cloud_contracts_from_csv(DATA_PATH)
    providers = {contract.provider_name for contract in contracts}
    assert providers == {"CoreWeave", "Oracle", "Nvidia", "Broadcom"}


def test_load_cloud_contracts_contract_values() -> None:
    contracts = {contract.provider_name: contract for contract in load_cloud_contracts_from_csv(DATA_PATH)}

    coreweave = contracts["CoreWeave"]
    assert coreweave.total_value == 22.4e9
    assert coreweave.start_year == 2025
    assert coreweave.duration_years == 5

    oracle = contracts["Oracle"]
    assert oracle.total_value == 300e9
    assert oracle.start_year == 2025
