"""Stateless helper functions for market capitalization analysis."""

from datetime import datetime
from typing import Dict

import pandas as pd
import yfinance as yf

from src.constants.ai_company_stock_tickers import (
    ALL_PUBLIC_TICKERS,
    CHATGPT_LAUNCH_DATE,
    PRIVATE_COMPANIES_CSV_PATH,
    PUBLIC_COMPANY_CATEGORIES,
)


def load_private_company_data() -> pd.DataFrame:
    """
    Load private company valuations from CSV.

    Returns:
        DataFrame with columns: company_name, ticker (None), categories,
        valuation_at_launch, valuation_at_launch_date, valuation_current,
        most_recent_valuation_date, data_source, last_updated
    """
    df = pd.read_csv(PRIVATE_COMPANIES_CSV_PATH)

    # Convert date columns to datetime
    df['date_of_most_recent_valuation_at_chatgpt_launch'] = pd.to_datetime(df['date_of_most_recent_valuation_at_chatgpt_launch'])
    df['most_recent_valuation_date'] = pd.to_datetime(df['most_recent_valuation_date'])

    # Add ticker column (None for private companies)
    df['ticker'] = None

    # Add categories (Private AI Startups)
    df['categories'] = 'Private AI Startups'

    # Add last_updated timestamp
    df['last_updated'] = datetime.now()

    # Reorder columns to match unified schema
    df = df[[
        'company_name',
        'ticker',
        'categories',
        'chatgpt_launch_valuation',
        'date_of_most_recent_valuation_at_chatgpt_launch',
        'valuation_current',
        'most_recent_valuation_date',
        'data_source',
        'last_updated'
    ]]

    return df


def fetch_public_company_row(
    ticker: str,
    chatgpt_launch_date: datetime,
    current_date: datetime
) -> Dict[str, any]:
    """
    Fetch market cap data for a single public company ticker.

    Args:
        ticker: Stock ticker symbol
        chatgpt_launch_date: ChatGPT launch date to measure changes from
        current_date: Current date for most recent valuation

    Returns:
        Dictionary with company data matching unified DataFrame schema

    Raises:
        ValueError: If no historical data available for ticker
        KeyError: If shares outstanding data is missing
    """
    stock = yf.Ticker(ticker)
    info = stock.info

    # Get company name and shares
    company_name = info.get('longName', ticker)
    shares_outstanding = info['sharesOutstanding']  # Fail if missing

    # Get historical data from ChatGPT launch to now (one API call for both dates)
    hist = stock.history(
        start=chatgpt_launch_date - pd.Timedelta(days=5),
        end=current_date
    )

    if hist.empty:
        raise ValueError(f"No historical data for {ticker}")

    # Remove timezone
    hist.index = hist.index.tz_localize(None)

    # Get market cap at ChatGPT launch
    target_chatgpt_launch = pd.Timestamp(chatgpt_launch_date).tz_localize(None)
    chatgpt_launch_idx = hist.index.get_indexer([target_chatgpt_launch], method='nearest')[0]
    chatgpt_launch_valuation = float(shares_outstanding * hist.iloc[chatgpt_launch_idx]['Close'])

    # Get current market cap
    current_market_cap = float(shares_outstanding * hist.iloc[-1]['Close'])

    # Get category
    category = ""
    for cat_name, cat_tickers in PUBLIC_COMPANY_CATEGORIES.items():
        if ticker in cat_tickers:
            category = cat_name
            break

    return {
        'company_name': company_name,
        'ticker': ticker,
        'categories': category,
        'chatgpt_launch_valuation': chatgpt_launch_valuation,
        'date_of_most_recent_valuation_at_chatgpt_launch': chatgpt_launch_date,
        'valuation_current': current_market_cap,
        'most_recent_valuation_date': current_date,
        'data_source': 'Yahoo Finance',
        'last_updated': current_date
    }


def get_unified_market_cap_data(
    chatgpt_launch_date: datetime = CHATGPT_LAUNCH_DATE
) -> pd.DataFrame:
    """
    Get unified dataset combining public and private companies.

    Args:
        chatgpt_launch_date: ChatGPT launch date to measure changes from. Defaults to Nov 30, 2022.

    Returns:
        DataFrame with all companies and their valuations, sorted by absolute change
    """
    print("Loading private companies...")
    private_df = load_private_company_data()

    print("\nFetching public company data from Yahoo Finance...")

    # Fetch data for all public companies
    current_date = datetime.now()
    public_data = []

    for ticker in ALL_PUBLIC_TICKERS:
        print(f"Fetching data for {ticker}...")
        row_data = fetch_public_company_row(ticker, chatgpt_launch_date, current_date)
        public_data.append(row_data)

    public_df = pd.DataFrame(public_data)

    # Combine the dataframes
    unified_df = pd.concat([public_df, private_df], ignore_index=True)

    # Calculate changes since ChatGPT launch
    unified_df['change_absolute'] = (
        unified_df['valuation_current'] - unified_df['chatgpt_launch_valuation']
    )
    unified_df['change_percent'] = (
        unified_df['change_absolute'] / unified_df['chatgpt_launch_valuation'] * 100
    )

    # Sort by absolute change (descending)
    unified_df = unified_df.sort_values('change_absolute', ascending=False)

    return unified_df


__all__ = [
    'load_private_company_data',
    'fetch_public_company_row',
    'get_unified_market_cap_data',
]
