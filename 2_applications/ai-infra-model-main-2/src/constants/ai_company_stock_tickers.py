"""Company definitions and categories for market cap analysis."""

from datetime import datetime
from pathlib import Path
from typing import Dict, List

# ChatGPT launch date - baseline for market cap analysis
CHATGPT_LAUNCH_DATE = datetime(2022, 11, 30)

# Path to private company valuations CSV
PRIVATE_COMPANIES_CSV_PATH = Path(__file__).parent / "private_company_valuations.csv"

# Categorized public company tickers (MECE - mutually exclusive, collectively exhaustive)
# Each company appears in exactly ONE category
PUBLIC_COMPANY_CATEGORIES: Dict[str, List[str]] = {
    "Hyperscale Clouds": [
        "MSFT",   # Microsoft
        "GOOGL",  # Google/Alphabet
        "AMZN",   # Amazon
        "META",   # Meta
        "ORCL",   # Oracle
    ],
    "Semiconductors": [
        "NVDA",   # NVIDIA
        "AVGO",   # Broadcom
        "AMD",    # AMD (MI300 series)
    ],
    "Infrastructure": [
        "VRT",    # Vertiv (power/cooling)
        "SMCI",   # Super Micro Computer
        "DELL",   # Dell
    ],
    "AI Applications": [
        "TSLA",   # Tesla (FSD, Optimus, EVs)
        "PLTR",   # Palantir (AI-powered analytics)
    ],
}

# Flattened list of all unique tickers across all categories
ALL_PUBLIC_TICKERS = sorted(set(
    ticker
    for tickers in PUBLIC_COMPANY_CATEGORIES.values()
    for ticker in tickers
))


__all__ = [
    "CHATGPT_LAUNCH_DATE",
    "PRIVATE_COMPANIES_CSV_PATH",
    "PUBLIC_COMPANY_CATEGORIES",
    "ALL_PUBLIC_TICKERS",
]
