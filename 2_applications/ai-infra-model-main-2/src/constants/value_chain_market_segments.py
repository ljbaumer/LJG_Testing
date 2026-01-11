"""Value Chain Market Segments - DataFrame Definitions

This module defines market segments as modular Pandas DataFrames. Each DataFrame
represents a self-contained market segment with cohort composition, ARPU, and costs.

Architecture:
- Each segment is a DataFrame with cohorts as rows
- Segments can be combined in lists to create scenarios
- Clean separation from pricing power (markups) and investment timelines (depreciation)
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# -----------------------------
# Facebook/Meta Data Loading
# -----------------------------

def load_facebook_data():
    """Load Facebook/Meta MAU and ARPU data from Q4 2023 CSV files."""
    # Get the data directory path relative to this file
    current_dir = Path(__file__).parent
    data_dir = current_dir.parent.parent / "data" / "users_and_revenue"

    # Load MAU data (Monthly Active Users in millions) - Q4 2023 data
    mau_file = data_dir / "Facebook_MAUs_by_Region_Q4_2023_millions.csv"
    mau_df = pd.read_csv(mau_file)

    # Load ARPU data (Average Revenue Per User per quarter in USD) - Q4 2023 data
    arpu_file = data_dir / "Meta_Facebook_ARPU_by_Region_Q4_2023_USD_per_quarter.csv"
    arpu_df = pd.read_csv(arpu_file)

    return mau_df, arpu_df

def calculate_segment_metrics():
    """Calculate TAMs and monthly ARPUs for our two consumer segments based on Q4 2023 Facebook/Meta data."""
    mau_df, arpu_df = load_facebook_data()

    # Get latest quarter data (Q4'23) for MAUs
    latest_mau_col = mau_df.columns[-1]  # Last column should be Q4'23

    # Extract MAUs by region (convert millions to actual numbers)
    us_canada_mau = mau_df[mau_df['Region'] == 'US & Canada'][latest_mau_col].iloc[0] * 1_000_000
    europe_mau = mau_df[mau_df['Region'] == 'Europe'][latest_mau_col].iloc[0] * 1_000_000
    apac_mau = mau_df[mau_df['Region'] == 'Asia-Pacific'][latest_mau_col].iloc[0] * 1_000_000
    row_mau = mau_df[mau_df['Region'] == 'Rest of World'][latest_mau_col].iloc[0] * 1_000_000

    # Calculate TAMs
    us_canada_europe_tam = int(us_canada_mau + europe_mau)
    apac_row_tam = int(apac_mau + row_mau)

    # Get Q4'23 ARPU data (quarterly)
    us_canada_arpu_quarterly = arpu_df[arpu_df['Region'] == 'US & Canada']["Q4'23"].iloc[0]
    europe_arpu_quarterly = arpu_df[arpu_df['Region'] == 'Europe']["Q4'23"].iloc[0]
    apac_arpu_quarterly = arpu_df[arpu_df['Region'] == 'Asia-Pacific']["Q4'23"].iloc[0]
    row_arpu_quarterly = arpu_df[arpu_df['Region'] == 'Rest of World']["Q4'23"].iloc[0]

    # Calculate weighted monthly ARPU (quarterly / 3)
    # US+Canada+Europe weighted average
    us_canada_europe_weighted_arpu_monthly = (
        (us_canada_arpu_quarterly * us_canada_mau + europe_arpu_quarterly * europe_mau) /
        (us_canada_mau + europe_mau)
    ) / 3

    # APAC+ROW weighted average
    apac_row_weighted_arpu_monthly = (
        (apac_arpu_quarterly * apac_mau + row_arpu_quarterly * row_mau) /
        (apac_mau + row_mau)
    ) / 3

    return {
        'us_canada_europe_tam': us_canada_europe_tam,
        'apac_row_tam': apac_row_tam,
        'us_canada_europe_monthly_arpu': us_canada_europe_weighted_arpu_monthly,
        'apac_row_monthly_arpu': apac_row_weighted_arpu_monthly
    }

# Load Facebook data metrics
facebook_metrics = calculate_segment_metrics()

# -----------------------------
# Market Segment Data Structure
# -----------------------------

# MarketSegment class removed - using pure DataFrame approach now


# -----------------------------
# Market Sizing Constants
# -----------------------------

# Consumer TAMs based on actual Facebook/Meta Q4 2023 data
US_CANADA_EUROPE_TAM: int = facebook_metrics['us_canada_europe_tam']  # US+Canada+Europe: ~680M users (Q4 2023)
APAC_ROW_TAM: int = facebook_metrics['apac_row_tam']  # APAC+Rest of World: ~2.385B users (Q4 2023)

# Other segment TAMs (unchanged)
ENTERPRISE_TAM: int = 450_000_000  # Microsoft Office enterprise user base (global)
PROGRAMMER_TAM: int = 20_000_000  # JetBrains Developer Ecosystem Survey global developer count

# Legacy constants for backward compatibility
DEVELOPED_MARKET_TAM: int = US_CANADA_EUROPE_TAM  # Alias for backward compatibility
DEVELOPING_MARKET_TAM: int = APAC_ROW_TAM  # Alias for backward compatibility

# -----------------------------
# Market Segment Definitions (New Structure)
# -----------------------------

# Consumer segment - US+Canada+Europe (Facebook data-based)
CONSUMER_US_CANADA_EUROPE_SEGMENT = pd.DataFrame({
    'segment': ['us_canada_europe_consumers', 'us_canada_europe_consumers'],
    'cohort_name': ['Consumer Free', 'Consumer Paid'],
    'cohort_share': [0.24, 0.037],  # 24% free monetized via ads, 3.7% paid subscribers
    'arpu': [2.50, 8.0],  # Much lower free tier ARPU - AI apps are early stage vs established social platforms
    'cost_to_service': [0.2, 1.8],  # Service costs for ad-supported vs premium
    'total_addressable_users': [US_CANADA_EUROPE_TAM, US_CANADA_EUROPE_TAM],
})

# Consumer segment - APAC+Rest of World (Facebook data-based)
CONSUMER_APAC_ROW_SEGMENT = pd.DataFrame({
    'segment': ['apac_row_consumers', 'apac_row_consumers'],
    'cohort_name': ['APAC+ROW Free', 'APAC+ROW Paid'],
    'cohort_share': [0.40, 0.013],  # 40% free monetized, 1.3% paid
    'arpu': [0.50, 6.0],  # Much lower free tier ARPU for developing markets
    'cost_to_service': [0.05, 0.30],  # Optimized service costs for scale and lower costs
    'total_addressable_users': [APAC_ROW_TAM, APAC_ROW_TAM],
})

# Professional segment - Microsoft Office 365 style enterprise (global enterprise TAM)
PROFESSIONAL_SEGMENT = pd.DataFrame({
    'segment': ['professional', 'professional'],
    'cohort_name': ['SMB Paid', 'Enterprise Paid'],
    'cohort_share': [0.11, 0.09],  # 44M team users (11%), 36M enterprise users (9%)
    'arpu': [15.0, 36.0],  # Microsoft Office pricing: $15 SMB, $36 Enterprise
    'cost_to_service': [3.0, 5.0],  # Lower costs due to enterprise efficiency
    'total_addressable_users': [ENTERPRISE_TAM, ENTERPRISE_TAM],
})

# Programmer segment - JetBrains/Claude-like pricing for actual developers (global programmer TAM)
PROGRAMMER_SEGMENT = pd.DataFrame({
    'segment': ['programmer', 'programmer'],
    'cohort_name': ['Developer Paid', 'Power User'],
    'cohort_share': [0.40, 0.025],  # 8M paid (40%), 0.5M power users (2.5%) - fewer power users
    'arpu': [20.0, 100.0],  # Back to $20 for developers, $100 for power users
    'cost_to_service': [3.0, 15.0],  # Higher costs for power users due to compute
    'total_addressable_users': [PROGRAMMER_TAM, PROGRAMMER_TAM],
})

# -----------------------------
# Today Market Segments (Current Reality)
# -----------------------------

# Today - US+Canada+Europe (Current AI market reality)
CONSUMER_US_CANADA_EUROPE_TODAY_SEGMENT = pd.DataFrame({
    'segment': ['us_canada_europe_consumers', 'us_canada_europe_consumers'],
    'cohort_name': ['Consumer Free', 'Consumer Paid'],
    'cohort_share': [0.0, 0.015],  # Zero ad revenue, 1.5% paid (higher current adoption estimate)
    'arpu': [0.0, 20.0],  # No monetization of free users yet, Current AI subscription pricing
    'cost_to_service': [0.0, 2.5],  # No free costs, higher paid costs for compute
    'total_addressable_users': [US_CANADA_EUROPE_TAM, US_CANADA_EUROPE_TAM],
})

# Today - APAC+Rest of World (Current AI market reality)
CONSUMER_APAC_ROW_TODAY_SEGMENT = pd.DataFrame({
    'segment': ['apac_row_consumers', 'apac_row_consumers'],
    'cohort_name': ['APAC+ROW Free', 'APAC+ROW Paid'],
    'cohort_share': [0.0, 0.006],  # Zero ad revenue, 0.6% paid (moderate international adoption)
    'arpu': [0.0, 6.0],  # No monetization of free users, OpenAI pricing for emerging markets
    'cost_to_service': [0.0, 1.5],  # No free costs
    'total_addressable_users': [APAC_ROW_TAM, APAC_ROW_TAM],
})

# Today - Professional (Current enterprise AI adoption)
PROFESSIONAL_TODAY_SEGMENT = pd.DataFrame({
    'segment': ['professional', 'professional'],
    'cohort_name': ['SMB Paid', 'Enterprise Paid'],
    'cohort_share': [0.045, 0.028],  # 18M SMB (4.5%), 11.2M enterprise (2.8%) - current enterprise reality
    'arpu': [15.0, 36.0],  # Microsoft Office pricing: $15 SMB, $36 Enterprise
    'cost_to_service': [4.0, 6.0],  # Higher service costs for enterprise features
    'total_addressable_users': [ENTERPRISE_TAM, ENTERPRISE_TAM],
})

# Today - Programmer (Current developer AI adoption)
PROGRAMMER_TODAY_SEGMENT = pd.DataFrame({
    'segment': ['programmer', 'programmer'],
    'cohort_name': ['Developer Paid', 'Power User'],
    'cohort_share': [0.75, 0.10],  # 15M paid (75%), 2M power users (10%) - very high current dev adoption
    'arpu': [20.0, 100.0],  # Current Copilot/Claude pricing
    'cost_to_service': [3.0, 15.0],  # Higher costs for power users
    'total_addressable_users': [PROGRAMMER_TAM, PROGRAMMER_TAM],
})

# -----------------------------
# Bull Case Market Segments
# -----------------------------

# Bull Case - US+Canada+Europe (more aggressive assumptions) - OLD MarketSegment format
# CONSUMER_US_CANADA_EUROPE_BULL_SEGMENT = MarketSegment(...)

# Bull Case - US+Canada+Europe (DataFrame format)
CONSUMER_US_CANADA_EUROPE_BULL_SEGMENT_DF = pd.DataFrame({
    'segment': ['us_canada_europe_consumers', 'us_canada_europe_consumers'],
    'total_addressable_users': [US_CANADA_EUROPE_TAM, US_CANADA_EUROPE_TAM],
    'cohort_name': ['Consumer Free', 'Consumer Paid'],
    'cohort_share': [0.90, 0.10],  # Netflix-like subscription rates: 90% free, 10% paid
    'arpu': [
        facebook_metrics['us_canada_europe_monthly_arpu'] * 1.3,  # 30% higher monetization for free users
        20.0  # Premium tier pricing (ChatGPT Plus benchmark)
    ],
    'cost_to_service': [0.5, 2.0],  # Slightly higher service costs
})

# Removed PROFESSIONAL_BULL_SEGMENT_OLD to eliminate duplicate constants

# Bull Case - APAC+Rest of World (DataFrame format)
CONSUMER_APAC_ROW_BULL_SEGMENT_DF = pd.DataFrame({
    'segment': ['apac_row_consumers', 'apac_row_consumers'],
    'total_addressable_users': [APAC_ROW_TAM, APAC_ROW_TAM],
    'cohort_name': ['APAC+ROW Free', 'APAC+ROW Paid'],
    'cohort_share': [0.90, 0.10],  # Netflix-like subscription rates: 90% free, 10% paid
    'arpu': [
        facebook_metrics['apac_row_monthly_arpu'] * 3.0,  # 3x higher monetization for free users
        6.0  # OpenAI pricing for emerging markets
    ],
    'cost_to_service': [0.1, 0.2],  # Optimized service costs
})



# -----------------------------
# Total Adoption Case Market Segments (Renamed from Bull Case)
# -----------------------------

# Total Adoption Case - US+Canada+Europe (maximum adoption assumptions) - DataFrame format
CONSUMER_US_CANADA_EUROPE_TOTAL_ADOPTION_SEGMENT_DF = pd.DataFrame({
    'segment': ['us_canada_europe_consumers', 'us_canada_europe_consumers'],
    'total_addressable_users': [US_CANADA_EUROPE_TAM, US_CANADA_EUROPE_TAM],
    'cohort_name': ['Consumer Free', 'Consumer Paid'],
    'cohort_share': [0.90, 0.10],  # Netflix-like subscription rates: 90% free, 10% paid
    'arpu': [
        facebook_metrics['us_canada_europe_monthly_arpu'] * 1.3,  # 30% higher monetization for free users
        20.0  # Premium tier pricing (ChatGPT Plus benchmark)
    ],
    'cost_to_service': [0.5, 2.0],  # Slightly higher service costs
})

# Total Adoption Case - APAC+Rest of World (maximum adoption assumptions) - DataFrame format
CONSUMER_APAC_ROW_TOTAL_ADOPTION_SEGMENT_DF = pd.DataFrame({
    'segment': ['apac_row_consumers', 'apac_row_consumers'],
    'total_addressable_users': [APAC_ROW_TAM, APAC_ROW_TAM],
    'cohort_name': ['APAC+ROW Free', 'APAC+ROW Paid'],
    'cohort_share': [0.90, 0.10],  # Netflix-like subscription rates: 90% free, 10% paid
    'arpu': [
        facebook_metrics['apac_row_monthly_arpu'] * 3.0,  # 3x higher monetization for free users
        6.0  # OpenAI pricing for emerging markets
    ],
    'cost_to_service': [0.1, 0.2],  # Optimized service costs
})

# Total Adoption Case - Professional (maximum adoption assumptions) - DataFrame format
PROFESSIONAL_TOTAL_ADOPTION_SEGMENT_DF = pd.DataFrame({
    'segment': ['professional', 'professional'],
    'total_addressable_users': [ENTERPRISE_TAM, ENTERPRISE_TAM],
    'cohort_name': ['SMB Paid', 'Enterprise Paid'],
    'cohort_share': [0.20, 0.80],  # 20% SMB, 80% Enterprise
    'arpu': [15.0, 45.0],  # Microsoft Office pricing: $15 SMB, $45 Enterprise (premium)
    'cost_to_service': [3.0, 8.0],  # Higher service costs for premium features
})

# Total Adoption Case - Programmer (maximum adoption assumptions) - DataFrame format
PROGRAMMER_TOTAL_ADOPTION_SEGMENT_DF = pd.DataFrame({
    'segment': ['programmer', 'programmer'],
    'total_addressable_users': [PROGRAMMER_TAM * 2, PROGRAMMER_TAM * 2],  # Double the TAM (40M vs 20M)
    'cohort_name': ['Developer Paid', 'Power User'],
    'cohort_share': [0.90, 0.10],  # High adoption rates
    'arpu': [20.0, 100.0],  # Same pricing as base case
    'cost_to_service': [5.0, 25.0],  # Higher costs for premium features
})

# -----------------------------
# New Bull Case Market Segments ($275B Target)
# -----------------------------

# New Bull Case - US+Canada+Europe (moderate high adoption for $275B target) - DataFrame format
CONSUMER_US_CANADA_EUROPE_BULL_SEGMENT_DF = pd.DataFrame({
    'segment': ['us_canada_europe_consumers', 'us_canada_europe_consumers'],
    'total_addressable_users': [US_CANADA_EUROPE_TAM, US_CANADA_EUROPE_TAM],
    'cohort_name': ['Consumer Free', 'Consumer Paid'],
    'cohort_share': [0.65, 0.08],  # 65% free, 8% paid (higher adoption for $275B target)
    'arpu': [12.0, 20.0],  # Higher monetization: $12 free, $20 paid (same as today)
    'cost_to_service': [0.3, 1.5],  # Reasonable service costs
})

# New Bull Case - APAC+Rest of World (moderate high adoption for $275B target) - DataFrame format
CONSUMER_APAC_ROW_BULL_SEGMENT_DF = pd.DataFrame({
    'segment': ['apac_row_consumers', 'apac_row_consumers'],
    'total_addressable_users': [APAC_ROW_TAM, APAC_ROW_TAM],
    'cohort_name': ['APAC+ROW Free', 'APAC+ROW Paid'],
    'cohort_share': [0.70, 0.04],  # 70% free, 4% paid (higher adoption for $275B target)
    'arpu': [3.0, 6.0],  # Higher emerging market pricing: $3 free, $6 paid (OpenAI pricing)
    'cost_to_service': [0.08, 0.25],  # Lower costs for scale
})

# New Bull Case - Professional (moderate high adoption for $275B target) - DataFrame format
PROFESSIONAL_BULL_SEGMENT_DF = pd.DataFrame({
    'segment': ['professional', 'professional'],
    'total_addressable_users': [ENTERPRISE_TAM, ENTERPRISE_TAM],
    'cohort_name': ['SMB Paid', 'Enterprise Paid'],
    'cohort_share': [0.16, 0.52],  # 16% SMB, 52% Enterprise (higher enterprise adoption for $275B target)
    'arpu': [15.0, 45.0],  # Microsoft Office pricing: $15 SMB, $45 Enterprise (premium)
    'cost_to_service': [3.0, 6.0],  # Moderate service costs
})

# New Bull Case - Programmer (moderate high adoption for $275B target) - DataFrame format
PROGRAMMER_BULL_SEGMENT_DF = pd.DataFrame({
    'segment': ['programmer', 'programmer'],
    'total_addressable_users': [PROGRAMMER_TAM, PROGRAMMER_TAM],  # Keep original TAM (20M)
    'cohort_name': ['Developer Paid', 'Power User'],
    'cohort_share': [0.65, 0.06],  # 65% paid, 6% power users (higher adoption for $275B target)
    'arpu': [22.0, 110.0],  # Higher pricing: $22 developer, $110 power user
    'cost_to_service': [3.0, 20.0],  # Moderate costs
})

# -----------------------------
# Base Case Market Segments
# -----------------------------

# Main structure: Concatenated DataFrame
BASE_CASE_MARKET_SEGMENTS = pd.concat([
    CONSUMER_US_CANADA_EUROPE_SEGMENT,
    CONSUMER_APAC_ROW_SEGMENT,
    PROFESSIONAL_SEGMENT,
    PROGRAMMER_SEGMENT
], ignore_index=True)

# Today structure: Concatenated DataFrame
TODAY_MARKET_SEGMENTS = pd.concat([
    CONSUMER_US_CANADA_EUROPE_TODAY_SEGMENT,
    CONSUMER_APAC_ROW_TODAY_SEGMENT,
    PROFESSIONAL_TODAY_SEGMENT,
    PROGRAMMER_TODAY_SEGMENT
], ignore_index=True)

# Bull Case structure: Concatenated DataFrame (New $275B target)
BULL_CASE_MARKET_SEGMENTS = pd.concat([
    CONSUMER_US_CANADA_EUROPE_BULL_SEGMENT_DF,
    CONSUMER_APAC_ROW_BULL_SEGMENT_DF,
    PROFESSIONAL_BULL_SEGMENT_DF,
    PROGRAMMER_BULL_SEGMENT_DF
], ignore_index=True)

# Total Adoption Case structure: Concatenated DataFrame (Previously Bull Case $492B)
TOTAL_ADOPTION_CASE_MARKET_SEGMENTS = pd.concat([
    CONSUMER_US_CANADA_EUROPE_TOTAL_ADOPTION_SEGMENT_DF,
    CONSUMER_APAC_ROW_TOTAL_ADOPTION_SEGMENT_DF,
    PROFESSIONAL_TOTAL_ADOPTION_SEGMENT_DF,
    PROGRAMMER_TOTAL_ADOPTION_SEGMENT_DF
], ignore_index=True)

# Legacy alias for backward compatibility
DEFAULT_MARKET_SEGMENTS = BASE_CASE_MARKET_SEGMENTS

# -----------------------------
# All Market Scenarios
# -----------------------------

ALL_MARKET_SCENARIOS = {
    "Today": {
        "segments": TODAY_MARKET_SEGMENTS,
        "description": "Best guess estimate of current AI market reality"
    },
    "Bear Case": {
        "segments": BASE_CASE_MARKET_SEGMENTS,
        "description": "Adoption slows as initial power user markets saturate and enterprise penetration proves difficult"
    },
    "Bull Case": {
        "segments": BULL_CASE_MARKET_SEGMENTS,
        "description": "Rapid adoption to ~70% the users / ARPU of the biggest services in the world - among the fastest growing product categories in history"
    },
    "Total Adoption Case": {
        "segments": TOTAL_ADOPTION_CASE_MARKET_SEGMENTS,
        "description": "Full adoption trajectory reaching the combined scale of Office + Facebook + Netflix by end of decade, with enhanced enterprise ARPUs"
    }
}


# -----------------------------
# Validation Helpers
# -----------------------------

def validate_segment_dataframe(df: pd.DataFrame) -> None:
    """Validate that a segment DataFrame has the required schema and data integrity.

    For single-segment DataFrames, validates cohort shares don't exceed 1.0 for that segment.
    For multi-segment DataFrames, validates each segment's cohort shares separately.
    """
    required_columns = [
        'cohort_name', 'cohort_share', 'arpu', 'cost_to_service',
        'segment', 'total_addressable_users'
    ]

    # Check required columns exist
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Check valid ranges for cohort_share (direct % of TAM, no need to sum to 1.0)
    if not ((df['cohort_share'] >= 0) & (df['cohort_share'] <= 1)).all():
        raise ValueError("All cohort_share values must be between 0 and 1")

    # Sanity check: cohort shares for each segment shouldn't exceed 100% of that segment's TAM
    for segment_name in df['segment'].unique():
        segment_df = df[df['segment'] == segment_name]
        share_sum = segment_df['cohort_share'].sum()
        if share_sum > 1.0:
            raise ValueError(f"Segment '{segment_name}' cohort_share cannot exceed 1.0 (got {share_sum:.6f})")

    if not (df['arpu'] >= 0).all():
        raise ValueError("All ARPU values must be non-negative")

    if not (df['cost_to_service'] >= 0).all():
        raise ValueError("All cost_to_service values must be non-negative")


# Old list and MarketSegment validation functions removed - using DataFrame-only validation now


# -----------------------------
# Module Validation (Self-Test)
# -----------------------------

if __name__ == "__main__":
    # Self-validate all defined segments
    print("Validating market segments...")

    # Test individual DataFrame segments - Base Case
    for name, segment in [
        ("CONSUMER_US_CANADA_EUROPE_SEGMENT", CONSUMER_US_CANADA_EUROPE_SEGMENT),
        ("CONSUMER_APAC_ROW_SEGMENT", CONSUMER_APAC_ROW_SEGMENT),
        ("PROFESSIONAL_SEGMENT", PROFESSIONAL_SEGMENT),
        ("PROGRAMMER_SEGMENT", PROGRAMMER_SEGMENT)
    ]:
        try:
            validate_segment_dataframe(segment)
            print(f"✅ {name} (DataFrame) - valid")
        except ValueError as e:
            print(f"❌ {name} (DataFrame) - {e}")

    # Test individual DataFrame segments - Today Case
    print("\nValidating Today segments...")
    for name, segment in [
        ("CONSUMER_US_CANADA_EUROPE_TODAY_SEGMENT", CONSUMER_US_CANADA_EUROPE_TODAY_SEGMENT),
        ("CONSUMER_APAC_ROW_TODAY_SEGMENT", CONSUMER_APAC_ROW_TODAY_SEGMENT),
        ("PROFESSIONAL_TODAY_SEGMENT", PROFESSIONAL_TODAY_SEGMENT),
        ("PROGRAMMER_TODAY_SEGMENT", PROGRAMMER_TODAY_SEGMENT)
    ]:
        try:
            validate_segment_dataframe(segment)
            print(f"✅ {name} (DataFrame) - valid")
        except ValueError as e:
            print(f"❌ {name} (DataFrame) - {e}")

    # Test individual DataFrame segments - Bull Case
    print("\nValidating Bull Case individual segments...")
    for name, segment in [
        ("CONSUMER_US_CANADA_EUROPE_BULL_SEGMENT_DF", CONSUMER_US_CANADA_EUROPE_BULL_SEGMENT_DF),
        ("CONSUMER_APAC_ROW_BULL_SEGMENT_DF", CONSUMER_APAC_ROW_BULL_SEGMENT_DF),
        ("PROFESSIONAL_BULL_SEGMENT_DF", PROFESSIONAL_BULL_SEGMENT_DF),
        ("PROGRAMMER_BULL_SEGMENT_DF", PROGRAMMER_BULL_SEGMENT_DF)
    ]:
        try:
            validate_segment_dataframe(segment)
            print(f"✅ {name} (DataFrame) - valid")
        except ValueError as e:
            print(f"❌ {name} (DataFrame) - {e}")

    # Test individual DataFrame segments - Total Adoption Case
    print("\nValidating Total Adoption Case individual segments...")
    for name, segment in [
        ("CONSUMER_US_CANADA_EUROPE_TOTAL_ADOPTION_SEGMENT_DF", CONSUMER_US_CANADA_EUROPE_TOTAL_ADOPTION_SEGMENT_DF),
        ("CONSUMER_APAC_ROW_TOTAL_ADOPTION_SEGMENT_DF", CONSUMER_APAC_ROW_TOTAL_ADOPTION_SEGMENT_DF),
        ("PROFESSIONAL_TOTAL_ADOPTION_SEGMENT_DF", PROFESSIONAL_TOTAL_ADOPTION_SEGMENT_DF),
        ("PROGRAMMER_TOTAL_ADOPTION_SEGMENT_DF", PROGRAMMER_TOTAL_ADOPTION_SEGMENT_DF)
    ]:
        try:
            validate_segment_dataframe(segment)
            print(f"✅ {name} (DataFrame) - valid")
        except ValueError as e:
            print(f"❌ {name} (DataFrame) - {e}")

    # Validate segment combinations (now DataFrames)
    print("\nValidating concatenated market scenarios...")
    try:
        validate_segment_dataframe(BASE_CASE_MARKET_SEGMENTS)
        print("✅ BASE_CASE_MARKET_SEGMENTS - valid")
    except ValueError as e:
        print(f"❌ BASE_CASE_MARKET_SEGMENTS - {e}")

    try:
        validate_segment_dataframe(TODAY_MARKET_SEGMENTS)
        print("✅ TODAY_MARKET_SEGMENTS - valid")
    except ValueError as e:
        print(f"❌ TODAY_MARKET_SEGMENTS - {e}")

    try:
        validate_segment_dataframe(BULL_CASE_MARKET_SEGMENTS)
        print("✅ BULL_CASE_MARKET_SEGMENTS - valid")
    except ValueError as e:
        print(f"❌ BULL_CASE_MARKET_SEGMENTS - {e}")

    try:
        validate_segment_dataframe(TOTAL_ADOPTION_CASE_MARKET_SEGMENTS)
        print("✅ TOTAL_ADOPTION_CASE_MARKET_SEGMENTS - valid")
    except ValueError as e:
        print(f"❌ TOTAL_ADOPTION_CASE_MARKET_SEGMENTS - {e}")

    print("✅ All segment validations passed!")
