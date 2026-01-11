#!/usr/bin/env python3
"""Test script for market cap analysis helpers."""

import pandas as pd

from src.constants.ai_company_stock_tickers import CHATGPT_LAUNCH_DATE
from src.utils.market_cap_helpers import get_unified_market_cap_data


def main():
    """Test the market cap analysis functions."""
    print("=" * 80)
    print("Testing Market Cap Analysis")
    print("=" * 80)

    print(f"\nAnalyzing market cap changes since: {CHATGPT_LAUNCH_DATE.date()}\n")

    # Get unified data
    df = get_unified_market_cap_data()

    print("\n" + "=" * 80)
    print("UNIFIED DATA SUMMARY")
    print("=" * 80)
    print(f"Total companies tracked: {len(df)}")
    print(f"Public companies: {df['ticker'].notna().sum()}")
    print(f"Private companies: {df['ticker'].isna().sum()}")

    # Display the data
    print("\n" + "=" * 80)
    print("MARKET CAP CHANGES (sorted by absolute change)")
    print("=" * 80)

    # Format the display
    pd_options = {
        'display.max_rows': None,
        'display.max_columns': None,
        'display.width': None,
        'display.max_colwidth': 30
    }

    with pd.option_context(*[item for pair in pd_options.items() for item in pair]):
        # Select and format columns for display
        display_df = df[[
            'company_name',
            'ticker',
            'categories',
            'chatgpt_launch_valuation',
            'valuation_current',
            'change_absolute',
            'change_percent',
            'data_source'
        ]].copy()

        # Format large numbers
        for col in ['chatgpt_launch_valuation', 'valuation_current', 'change_absolute']:
            display_df[col] = display_df[col].apply(
                lambda x: f"${x/1e9:.2f}B" if pd.notna(x) else "N/A"
            )

        display_df['change_percent'] = display_df['change_percent'].apply(
            lambda x: f"{x:.1f}%" if pd.notna(x) else "N/A"
        )

        print(display_df.to_string(index=False))

    # Summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)

    total_value_created = df['change_absolute'].sum()
    print(f"Total value created: ${total_value_created/1e12:.2f}T")

    print("\nValue created by category:")
    # Expand categories and sum
    category_values = {}
    for _, row in df.iterrows():
        categories = row['categories'].split(', ') if pd.notna(row['categories']) else []
        for category in categories:
            if category not in category_values:
                category_values[category] = 0
            category_values[category] += row['change_absolute']

    for category, value in sorted(category_values.items(), key=lambda x: x[1], reverse=True):
        print(f"  {category}: ${value/1e9:.1f}B")


if __name__ == "__main__":
    main()
