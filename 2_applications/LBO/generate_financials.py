"""
CloudNova Solutions - Mock SaaS Company Financial Model Generator
=================================================================
Generates a complete set of financial statements (Income Statement, Balance Sheet, 
Cash Flow Statement) for a fictional SaaS company following "Rule of 40" principles.

Company Profile:
- ~$300M revenue with 25% initial growth
- 15% EBITDA margin in Year 0, improving over time
- 80% gross margins
- Growth moderates as profitability improves
"""

import csv
from dataclasses import dataclass
from typing import Dict, List

# =============================================================================
# COMPANY ASSUMPTIONS
# =============================================================================

COMPANY_NAME = "CloudNova Solutions"

# Time horizon: 2 historical years + current year + 5 forecast years
YEARS = [2022, 2023, 2024, 2025, 2026, 2027, 2028, 2029]
CURRENT_YEAR = 2024
HISTORICAL_YEARS = [2022, 2023]
FORECAST_YEARS = [2024, 2025, 2026, 2027, 2028, 2029]

# Revenue assumptions ($ millions)
BASE_REVENUE_2024 = 300.0  # $300M in current year

# Growth rates by year (moderating over time)
GROWTH_RATES = {
    2022: 0.35,   # Historical: 35% growth
    2023: 0.30,   # Historical: 30% growth  
    2024: 0.25,   # Current: 25% growth
    2025: 0.22,   # Forecast: 22% growth
    2026: 0.18,   # Forecast: 18% growth
    2027: 0.15,   # Forecast: 15% growth
    2028: 0.12,   # Forecast: 12% growth
    2029: 0.10,   # Forecast: 10% growth
}

# EBITDA margins (improving over time as growth slows)
EBITDA_MARGINS = {
    2022: 0.10,   # Historical: 10%
    2023: 0.12,   # Historical: 12%
    2024: 0.15,   # Current: 15%
    2025: 0.18,   # Forecast: 18%
    2026: 0.20,   # Forecast: 20%
    2027: 0.22,   # Forecast: 22%
    2028: 0.24,   # Forecast: 24%
    2029: 0.25,   # Forecast: 25%
}

# Other margin assumptions
GROSS_MARGIN = 0.80  # 80% gross margin (consistent)
SBC_PCT_REV = 0.08   # Stock-based comp as % of revenue
DA_PCT_REV = 0.03    # D&A as % of revenue

# Balance sheet assumptions (as % of revenue unless noted)
AR_DAYS = 45         # Days sales outstanding
AP_DAYS = 30         # Days payable outstanding
DEFERRED_REV_DAYS = 60  # Days deferred revenue (typical for annual SaaS contracts)
PREPAID_PCT = 0.02   # Prepaid expenses as % of revenue
ACCRUED_PCT = 0.04   # Accrued expenses as % of revenue

# CapEx as % of revenue
CAPEX_PCT = 0.02     # Low capex (asset-light SaaS model)

# Tax rate
TAX_RATE = 0.25      # 25% effective tax rate

# Opening balance sheet items (2021 ending = 2022 beginning)
OPENING_CASH = 50.0           # Starting cash position
OPENING_PP_E = 15.0           # Opening PP&E
OPENING_INTANGIBLES = 50.0    # Opening intangibles (higher to avoid floor)
OPENING_DEBT = 0.0            # No debt initially
OPENING_COMMON_STOCK = 150.0  # Common Stock & APIC (fixed)
# Note: Opening retained earnings will be calculated to balance the opening BS


# =============================================================================
# FINANCIAL MODEL CALCULATIONS
# =============================================================================

def calculate_revenue_schedule() -> Dict[int, float]:
    """Calculate revenue for each year working backwards and forwards from base."""
    revenues = {}
    
    # Set 2024 as base
    revenues[2024] = BASE_REVENUE_2024
    
    # Work backwards for historical years
    revenues[2023] = revenues[2024] / (1 + GROWTH_RATES[2024])
    revenues[2022] = revenues[2023] / (1 + GROWTH_RATES[2023])
    
    # Work forwards for forecast years
    for year in [2025, 2026, 2027, 2028, 2029]:
        revenues[year] = revenues[year - 1] * (1 + GROWTH_RATES[year])
    
    return revenues


def generate_income_statement(revenues: Dict[int, float]) -> Dict[str, Dict[int, float]]:
    """Generate income statement line items."""
    is_data = {}
    
    # Revenue
    is_data["Revenue"] = revenues.copy()
    
    # Cost of Revenue (inverse of gross margin)
    is_data["Cost of Revenue"] = {y: -rev * (1 - GROSS_MARGIN) for y, rev in revenues.items()}
    
    # Gross Profit
    is_data["Gross Profit"] = {y: revenues[y] + is_data["Cost of Revenue"][y] for y in YEARS}
    
    # Operating Expenses (calculated to hit target EBITDA margin)
    # EBITDA = Revenue * EBITDA_Margin
    # EBITDA = Gross Profit - OpEx (excluding D&A)
    # OpEx = Gross Profit - EBITDA
    is_data["EBITDA Target"] = {y: revenues[y] * EBITDA_MARGINS[y] for y in YEARS}
    
    # Break down OpEx into components
    is_data["Stock-Based Compensation"] = {y: -revenues[y] * SBC_PCT_REV for y in YEARS}
    
    # Calculate remaining cash OpEx needed to hit EBITDA target
    # Cash OpEx = Gross Profit - EBITDA - SBC (SBC is non-cash but above EBITDA line for this model)
    # Actually, let's structure this properly:
    # Gross Profit - S&M - R&D - G&A = EBITDA + D&A
    # We'll calculate total OpEx needed and split it
    
    total_cash_opex = {y: is_data["Gross Profit"][y] - is_data["EBITDA Target"][y] for y in YEARS}
    
    # Split OpEx: S&M (50%), R&D (35%), G&A (15%) - typical SaaS mix
    is_data["Sales & Marketing"] = {y: -total_cash_opex[y] * 0.50 for y in YEARS}
    is_data["Research & Development"] = {y: -total_cash_opex[y] * 0.35 for y in YEARS}
    is_data["General & Administrative"] = {y: -total_cash_opex[y] * 0.15 for y in YEARS}
    
    # Total Operating Expenses
    is_data["Total Operating Expenses"] = {
        y: is_data["Sales & Marketing"][y] + is_data["Research & Development"][y] + is_data["General & Administrative"][y]
        for y in YEARS
    }
    
    # EBITDA (should match target)
    is_data["EBITDA"] = {y: is_data["Gross Profit"][y] + is_data["Total Operating Expenses"][y] for y in YEARS}
    
    # Depreciation & Amortization
    is_data["Depreciation & Amortization"] = {y: -revenues[y] * DA_PCT_REV for y in YEARS}
    
    # EBIT (Operating Income)
    is_data["EBIT (Operating Income)"] = {y: is_data["EBITDA"][y] + is_data["Depreciation & Amortization"][y] for y in YEARS}
    
    # Interest Expense (will be minimal for this company - assume 0 debt initially)
    is_data["Interest Expense"] = {y: 0.0 for y in YEARS}
    
    # Pre-Tax Income
    is_data["Pre-Tax Income (EBT)"] = {y: is_data["EBIT (Operating Income)"][y] + is_data["Interest Expense"][y] for y in YEARS}
    
    # Income Tax
    is_data["Income Tax"] = {y: -max(0, is_data["Pre-Tax Income (EBT)"][y]) * TAX_RATE for y in YEARS}
    
    # Net Income
    is_data["Net Income"] = {y: is_data["Pre-Tax Income (EBT)"][y] + is_data["Income Tax"][y] for y in YEARS}
    
    # Remove helper calculation from output
    del is_data["EBITDA Target"]
    
    return is_data


def generate_balance_sheet(revenues: Dict[int, float], is_data: Dict[str, Dict[int, float]]) -> Dict[str, Dict[int, float]]:
    """Generate balance sheet line items."""
    bs_data = {}
    
    # ASSETS
    # ------
    
    # Cash (will be calculated as plug after CF statement)
    bs_data["Cash & Cash Equivalents"] = {y: 0.0 for y in YEARS}
    
    # Accounts Receivable (based on days sales outstanding)
    bs_data["Accounts Receivable"] = {y: revenues[y] * (AR_DAYS / 365) for y in YEARS}
    
    # Prepaid Expenses
    bs_data["Prepaid Expenses"] = {y: revenues[y] * PREPAID_PCT for y in YEARS}
    
    # Total Current Assets (placeholder - will recalculate after cash)
    bs_data["Total Current Assets"] = {y: 0.0 for y in YEARS}
    
    # Property, Plant & Equipment (grows with capex, shrinks with depreciation)
    bs_data["Property, Plant & Equipment, net"] = {}
    for i, year in enumerate(YEARS):
        if i == 0:
            # Beginning balance + CapEx - Depreciation
            capex = revenues[year] * CAPEX_PCT
            depr = -is_data["Depreciation & Amortization"][year] * 0.5  # Half is depreciation
            bs_data["Property, Plant & Equipment, net"][year] = OPENING_PP_E + capex - depr
        else:
            prev_year = YEARS[i-1]
            capex = revenues[year] * CAPEX_PCT
            depr = -is_data["Depreciation & Amortization"][year] * 0.5
            bs_data["Property, Plant & Equipment, net"][year] = bs_data["Property, Plant & Equipment, net"][prev_year] + capex - depr
    
    # Intangible Assets (amortize slowly)
    bs_data["Intangible Assets, net"] = {}
    for i, year in enumerate(YEARS):
        if i == 0:
            amort = -is_data["Depreciation & Amortization"][year] * 0.5  # Half is amortization
            bs_data["Intangible Assets, net"][year] = max(0, OPENING_INTANGIBLES - amort)
        else:
            prev_year = YEARS[i-1]
            amort = -is_data["Depreciation & Amortization"][year] * 0.5
            bs_data["Intangible Assets, net"][year] = max(0, bs_data["Intangible Assets, net"][prev_year] - amort)
    
    # Goodwill (static - no impairment assumed)
    bs_data["Goodwill"] = {y: 50.0 for y in YEARS}
    
    # Other Long-Term Assets
    bs_data["Other Long-Term Assets"] = {y: revenues[y] * 0.01 for y in YEARS}
    
    # Total Long-Term Assets
    bs_data["Total Long-Term Assets"] = {
        y: bs_data["Property, Plant & Equipment, net"][y] + bs_data["Intangible Assets, net"][y] + 
           bs_data["Goodwill"][y] + bs_data["Other Long-Term Assets"][y]
        for y in YEARS
    }
    
    # Total Assets (placeholder)
    bs_data["Total Assets"] = {y: 0.0 for y in YEARS}
    
    # LIABILITIES
    # -----------
    
    # Accounts Payable (based on COGS)
    cogs = {y: -is_data["Cost of Revenue"][y] for y in YEARS}
    bs_data["Accounts Payable"] = {y: cogs[y] * (AP_DAYS / 365) for y in YEARS}
    
    # Accrued Expenses
    bs_data["Accrued Expenses"] = {y: revenues[y] * ACCRUED_PCT for y in YEARS}
    
    # Deferred Revenue (key SaaS metric)
    bs_data["Deferred Revenue"] = {y: revenues[y] * (DEFERRED_REV_DAYS / 365) for y in YEARS}
    
    # Current Portion of Long-Term Debt
    bs_data["Current Portion of Debt"] = {y: 0.0 for y in YEARS}
    
    # Total Current Liabilities
    bs_data["Total Current Liabilities"] = {
        y: bs_data["Accounts Payable"][y] + bs_data["Accrued Expenses"][y] + 
           bs_data["Deferred Revenue"][y] + bs_data["Current Portion of Debt"][y]
        for y in YEARS
    }
    
    # Long-Term Debt
    bs_data["Long-Term Debt"] = {y: 0.0 for y in YEARS}
    
    # Other Long-Term Liabilities
    bs_data["Other Long-Term Liabilities"] = {y: revenues[y] * 0.02 for y in YEARS}
    
    # Total Long-Term Liabilities
    bs_data["Total Long-Term Liabilities"] = {
        y: bs_data["Long-Term Debt"][y] + bs_data["Other Long-Term Liabilities"][y]
        for y in YEARS
    }
    
    # Total Liabilities
    bs_data["Total Liabilities"] = {
        y: bs_data["Total Current Liabilities"][y] + bs_data["Total Long-Term Liabilities"][y]
        for y in YEARS
    }
    
    # SHAREHOLDERS' EQUITY
    # --------------------
    
    # Common Stock & APIC (grows with SBC)
    # SBC is a non-cash expense that increases equity
    bs_data["Common Stock & APIC"] = {}
    for i, year in enumerate(YEARS):
        sbc_amount = revenues[year] * SBC_PCT_REV
        if i == 0:
            bs_data["Common Stock & APIC"][year] = OPENING_COMMON_STOCK + sbc_amount
        else:
            prev_year = YEARS[i-1]
            bs_data["Common Stock & APIC"][year] = bs_data["Common Stock & APIC"][prev_year] + sbc_amount
    
    # Calculate opening retained earnings as a balancing figure
    # Opening BS: Opening Assets = Opening Liabilities + Opening Equity
    # Opening Equity = Opening Common Stock + Opening RE
    # Opening RE = Opening Assets - Opening Liabilities - Opening Common Stock
    rev_2021 = revenues[2022] / (1 + GROWTH_RATES[2022])
    cogs_2021 = rev_2021 * (1 - GROSS_MARGIN)
    
    opening_assets = (
        OPENING_CASH + 
        rev_2021 * (AR_DAYS / 365) +  # AR
        rev_2021 * PREPAID_PCT +  # Prepaid
        OPENING_PP_E + 
        OPENING_INTANGIBLES + 
        50.0 +  # Goodwill
        rev_2021 * 0.01  # Other LTA
    )
    
    opening_liabilities = (
        cogs_2021 * (AP_DAYS / 365) +  # AP
        rev_2021 * ACCRUED_PCT +  # Accrued
        rev_2021 * (DEFERRED_REV_DAYS / 365) +  # Deferred Revenue
        rev_2021 * 0.02  # Other LT Liab
    )
    
    opening_re = opening_assets - opening_liabilities - OPENING_COMMON_STOCK
    
    # Retained Earnings (accumulates net income)
    bs_data["Retained Earnings"] = {}
    for i, year in enumerate(YEARS):
        if i == 0:
            bs_data["Retained Earnings"][year] = opening_re + is_data["Net Income"][year]
        else:
            prev_year = YEARS[i-1]
            bs_data["Retained Earnings"][year] = bs_data["Retained Earnings"][prev_year] + is_data["Net Income"][year]
    
    # Total Shareholders' Equity
    bs_data["Total Shareholders' Equity"] = {
        y: bs_data["Common Stock & APIC"][y] + bs_data["Retained Earnings"][y]
        for y in YEARS
    }
    
    # Total Liabilities & Equity (placeholder)
    bs_data["Total Liabilities & Equity"] = {y: 0.0 for y in YEARS}
    
    return bs_data


def generate_cash_flow_statement(
    revenues: Dict[int, float], 
    is_data: Dict[str, Dict[int, float]], 
    bs_data: Dict[str, Dict[int, float]]
) -> Dict[str, Dict[int, float]]:
    """Generate cash flow statement and calculate ending cash."""
    cf_data = {}
    
    # OPERATING ACTIVITIES
    # --------------------
    
    cf_data["Net Income"] = is_data["Net Income"].copy()
    
    # Add back non-cash items
    cf_data["Depreciation & Amortization"] = {y: -is_data["Depreciation & Amortization"][y] for y in YEARS}
    cf_data["Stock-Based Compensation"] = {y: revenues[y] * SBC_PCT_REV for y in YEARS}
    
    # Changes in working capital
    cf_data["Change in Accounts Receivable"] = {}
    cf_data["Change in Prepaid Expenses"] = {}
    cf_data["Change in Accounts Payable"] = {}
    cf_data["Change in Accrued Expenses"] = {}
    cf_data["Change in Deferred Revenue"] = {}
    cf_data["Change in Other Assets/Liabilities"] = {}
    
    # Calculate opening balances for 2022 (based on 2021 estimated revenue)
    rev_2021 = revenues[2022] / (1 + GROWTH_RATES[2022])
    ar_2021 = rev_2021 * (AR_DAYS / 365)
    prepaid_2021 = rev_2021 * PREPAID_PCT
    cogs_2021 = rev_2021 * (1 - GROSS_MARGIN)
    ap_2021 = cogs_2021 * (AP_DAYS / 365)
    accrued_2021 = rev_2021 * ACCRUED_PCT
    deferred_2021 = rev_2021 * (DEFERRED_REV_DAYS / 365)
    other_lta_2021 = rev_2021 * 0.01
    other_ltl_2021 = rev_2021 * 0.02
    
    for i, year in enumerate(YEARS):
        if i == 0:
            # First year: compare to estimated 2021 balances
            cf_data["Change in Accounts Receivable"][year] = -(bs_data["Accounts Receivable"][year] - ar_2021)
            cf_data["Change in Prepaid Expenses"][year] = -(bs_data["Prepaid Expenses"][year] - prepaid_2021)
            cf_data["Change in Accounts Payable"][year] = bs_data["Accounts Payable"][year] - ap_2021
            cf_data["Change in Accrued Expenses"][year] = bs_data["Accrued Expenses"][year] - accrued_2021
            cf_data["Change in Deferred Revenue"][year] = bs_data["Deferred Revenue"][year] - deferred_2021
            # Change in other LT assets (negative = use of cash) and other LT liab (positive = source)
            delta_other_lta = -(bs_data["Other Long-Term Assets"][year] - other_lta_2021)
            delta_other_ltl = bs_data["Other Long-Term Liabilities"][year] - other_ltl_2021
            cf_data["Change in Other Assets/Liabilities"][year] = delta_other_lta + delta_other_ltl
        else:
            prev_year = YEARS[i-1]
            cf_data["Change in Accounts Receivable"][year] = -(bs_data["Accounts Receivable"][year] - bs_data["Accounts Receivable"][prev_year])
            cf_data["Change in Prepaid Expenses"][year] = -(bs_data["Prepaid Expenses"][year] - bs_data["Prepaid Expenses"][prev_year])
            cf_data["Change in Accounts Payable"][year] = bs_data["Accounts Payable"][year] - bs_data["Accounts Payable"][prev_year]
            cf_data["Change in Accrued Expenses"][year] = bs_data["Accrued Expenses"][year] - bs_data["Accrued Expenses"][prev_year]
            cf_data["Change in Deferred Revenue"][year] = bs_data["Deferred Revenue"][year] - bs_data["Deferred Revenue"][prev_year]
            delta_other_lta = -(bs_data["Other Long-Term Assets"][year] - bs_data["Other Long-Term Assets"][prev_year])
            delta_other_ltl = bs_data["Other Long-Term Liabilities"][year] - bs_data["Other Long-Term Liabilities"][prev_year]
            cf_data["Change in Other Assets/Liabilities"][year] = delta_other_lta + delta_other_ltl
    
    # Total Change in Working Capital
    cf_data["Total Change in Working Capital"] = {
        y: cf_data["Change in Accounts Receivable"][y] + cf_data["Change in Prepaid Expenses"][y] +
           cf_data["Change in Accounts Payable"][y] + cf_data["Change in Accrued Expenses"][y] +
           cf_data["Change in Deferred Revenue"][y] + cf_data["Change in Other Assets/Liabilities"][y]
        for y in YEARS
    }
    
    # Cash from Operating Activities
    cf_data["Cash from Operating Activities"] = {
        y: cf_data["Net Income"][y] + cf_data["Depreciation & Amortization"][y] + 
           cf_data["Stock-Based Compensation"][y] + cf_data["Total Change in Working Capital"][y]
        for y in YEARS
    }
    
    # INVESTING ACTIVITIES
    # --------------------
    
    cf_data["Capital Expenditures"] = {y: -revenues[y] * CAPEX_PCT for y in YEARS}
    cf_data["Other Investing Activities"] = {y: 0.0 for y in YEARS}
    
    cf_data["Cash from Investing Activities"] = {
        y: cf_data["Capital Expenditures"][y] + cf_data["Other Investing Activities"][y]
        for y in YEARS
    }
    
    # FINANCING ACTIVITIES
    # --------------------
    
    cf_data["Debt Issuance / (Repayment)"] = {y: 0.0 for y in YEARS}
    cf_data["Stock Issuance / (Buybacks)"] = {y: 0.0 for y in YEARS}
    cf_data["Dividends Paid"] = {y: 0.0 for y in YEARS}
    
    cf_data["Cash from Financing Activities"] = {
        y: cf_data["Debt Issuance / (Repayment)"][y] + cf_data["Stock Issuance / (Buybacks)"][y] +
           cf_data["Dividends Paid"][y]
        for y in YEARS
    }
    
    # NET CHANGE IN CASH
    cf_data["Net Change in Cash"] = {
        y: cf_data["Cash from Operating Activities"][y] + cf_data["Cash from Investing Activities"][y] +
           cf_data["Cash from Financing Activities"][y]
        for y in YEARS
    }
    
    # Calculate cash balances and update balance sheet
    for i, year in enumerate(YEARS):
        if i == 0:
            bs_data["Cash & Cash Equivalents"][year] = OPENING_CASH + cf_data["Net Change in Cash"][year]
        else:
            prev_year = YEARS[i-1]
            bs_data["Cash & Cash Equivalents"][year] = bs_data["Cash & Cash Equivalents"][prev_year] + cf_data["Net Change in Cash"][year]
    
    cf_data["Beginning Cash"] = {}
    cf_data["Ending Cash"] = {}
    for i, year in enumerate(YEARS):
        if i == 0:
            cf_data["Beginning Cash"][year] = OPENING_CASH
        else:
            cf_data["Beginning Cash"][year] = bs_data["Cash & Cash Equivalents"][YEARS[i-1]]
        cf_data["Ending Cash"][year] = bs_data["Cash & Cash Equivalents"][year]
    
    # Update balance sheet totals
    for year in YEARS:
        bs_data["Total Current Assets"][year] = (
            bs_data["Cash & Cash Equivalents"][year] + 
            bs_data["Accounts Receivable"][year] + 
            bs_data["Prepaid Expenses"][year]
        )
        bs_data["Total Assets"][year] = bs_data["Total Current Assets"][year] + bs_data["Total Long-Term Assets"][year]
        bs_data["Total Liabilities & Equity"][year] = bs_data["Total Liabilities"][year] + bs_data["Total Shareholders' Equity"][year]
    
    return cf_data


def calculate_metrics(revenues: Dict[int, float], is_data: Dict[str, Dict[int, float]]) -> Dict[str, Dict[int, float]]:
    """Calculate key SaaS and financial metrics."""
    metrics = {}
    
    # Growth rates
    metrics["Revenue Growth %"] = {}
    for i, year in enumerate(YEARS):
        if i == 0:
            metrics["Revenue Growth %"][year] = GROWTH_RATES[year] * 100  # Use assumed rate
        else:
            prev_rev = revenues[YEARS[i-1]]
            metrics["Revenue Growth %"][year] = ((revenues[year] - prev_rev) / prev_rev) * 100
    
    # Margins
    metrics["Gross Margin %"] = {y: (is_data["Gross Profit"][y] / revenues[y]) * 100 for y in YEARS}
    metrics["EBITDA Margin %"] = {y: (is_data["EBITDA"][y] / revenues[y]) * 100 for y in YEARS}
    metrics["EBIT Margin %"] = {y: (is_data["EBIT (Operating Income)"][y] / revenues[y]) * 100 for y in YEARS}
    metrics["Net Income Margin %"] = {y: (is_data["Net Income"][y] / revenues[y]) * 100 for y in YEARS}
    
    # Rule of 40
    metrics["Rule of 40 Score"] = {
        y: metrics["Revenue Growth %"][y] + metrics["EBITDA Margin %"][y]
        for y in YEARS
    }
    
    return metrics


def validate_balance_sheet(bs_data: Dict[str, Dict[int, float]]) -> bool:
    """Validate that balance sheet balances for all years."""
    for year in YEARS:
        assets = bs_data["Total Assets"][year]
        liab_eq = bs_data["Total Liabilities & Equity"][year]
        if abs(assets - liab_eq) > 0.01:  # Allow small rounding difference
            print(f"WARNING: Balance sheet doesn't balance in {year}")
            print(f"  Total Assets: {assets:.2f}")
            print(f"  Total L&E: {liab_eq:.2f}")
            return False
    print("✓ Balance sheet balances for all years")
    return True


def validate_cash_flow(cf_data: Dict[str, Dict[int, float]]) -> bool:
    """Validate that cash flow statement ties to ending cash."""
    for year in YEARS:
        calc_ending = cf_data["Beginning Cash"][year] + cf_data["Net Change in Cash"][year]
        actual_ending = cf_data["Ending Cash"][year]
        if abs(calc_ending - actual_ending) > 0.01:
            print(f"WARNING: Cash flow doesn't tie in {year}")
            return False
    print("✓ Cash flow statement ties for all years")
    return True


def write_to_csv(
    revenues: Dict[int, float],
    is_data: Dict[str, Dict[int, float]],
    bs_data: Dict[str, Dict[int, float]],
    cf_data: Dict[str, Dict[int, float]],
    metrics: Dict[str, Dict[int, float]],
    filename: str
):
    """Write all financial data to a nicely formatted CSV file."""
    
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Header
        writer.writerow([f"{'='*60}"])
        writer.writerow([f"{COMPANY_NAME} - Financial Model"])
        writer.writerow([f"{'='*60}"])
        writer.writerow([])
        writer.writerow(["All figures in $ millions unless otherwise noted"])
        writer.writerow(["Historical years: 2022-2023 | Current year: 2024 | Forecast: 2025-2029"])
        writer.writerow([])
        
        # Year headers
        year_row = ["Line Item"] + [str(y) for y in YEARS]
        
        # Helper function to write a section
        def write_section(title: str, data: Dict[str, Dict[int, float]], items: List[str]):
            writer.writerow([])
            writer.writerow([f"--- {title} ---"])
            writer.writerow(year_row)
            for item in items:
                if item in data:
                    row = [item] + [f"{data[item][y]:.2f}" for y in YEARS]
                    writer.writerow(row)
                elif item == "":
                    writer.writerow([])
                else:
                    writer.writerow([item])
        
        # Income Statement
        is_items = [
            "Revenue",
            "Cost of Revenue",
            "Gross Profit",
            "",
            "Sales & Marketing",
            "Research & Development",
            "General & Administrative",
            "Total Operating Expenses",
            "",
            "EBITDA",
            "Depreciation & Amortization",
            "EBIT (Operating Income)",
            "",
            "Interest Expense",
            "Pre-Tax Income (EBT)",
            "Income Tax",
            "Net Income",
        ]
        write_section("INCOME STATEMENT", is_data, is_items)
        
        # Balance Sheet
        bs_items = [
            "ASSETS",
            "Cash & Cash Equivalents",
            "Accounts Receivable",
            "Prepaid Expenses",
            "Total Current Assets",
            "",
            "Property, Plant & Equipment, net",
            "Intangible Assets, net",
            "Goodwill",
            "Other Long-Term Assets",
            "Total Long-Term Assets",
            "",
            "Total Assets",
            "",
            "LIABILITIES",
            "Accounts Payable",
            "Accrued Expenses",
            "Deferred Revenue",
            "Current Portion of Debt",
            "Total Current Liabilities",
            "",
            "Long-Term Debt",
            "Other Long-Term Liabilities",
            "Total Long-Term Liabilities",
            "",
            "Total Liabilities",
            "",
            "SHAREHOLDERS' EQUITY",
            "Common Stock & APIC",
            "Retained Earnings",
            "Total Shareholders' Equity",
            "",
            "Total Liabilities & Equity",
        ]
        write_section("BALANCE SHEET", bs_data, bs_items)
        
        # Cash Flow Statement
        cf_items = [
            "OPERATING ACTIVITIES",
            "Net Income",
            "Depreciation & Amortization",
            "Stock-Based Compensation",
            "",
            "Change in Accounts Receivable",
            "Change in Prepaid Expenses",
            "Change in Accounts Payable",
            "Change in Accrued Expenses",
            "Change in Deferred Revenue",
            "Total Change in Working Capital",
            "",
            "Cash from Operating Activities",
            "",
            "INVESTING ACTIVITIES",
            "Capital Expenditures",
            "Other Investing Activities",
            "Cash from Investing Activities",
            "",
            "FINANCING ACTIVITIES",
            "Debt Issuance / (Repayment)",
            "Stock Issuance / (Buybacks)",
            "Dividends Paid",
            "Cash from Financing Activities",
            "",
            "Net Change in Cash",
            "Beginning Cash",
            "Ending Cash",
        ]
        write_section("CASH FLOW STATEMENT", cf_data, cf_items)
        
        # Key Metrics
        metric_items = [
            "Revenue Growth %",
            "Gross Margin %",
            "EBITDA Margin %",
            "EBIT Margin %",
            "Net Income Margin %",
            "",
            "Rule of 40 Score",
        ]
        write_section("KEY METRICS", metrics, metric_items)
        
        # Assumptions section
        writer.writerow([])
        writer.writerow([f"--- MODEL ASSUMPTIONS ---"])
        writer.writerow(["Assumption", "Value", "Notes"])
        writer.writerow(["Gross Margin", f"{GROSS_MARGIN*100:.0f}%", "Consistent across all years"])
        writer.writerow(["D&A % of Revenue", f"{DA_PCT_REV*100:.1f}%", "Depreciation & Amortization"])
        writer.writerow(["SBC % of Revenue", f"{SBC_PCT_REV*100:.1f}%", "Stock-Based Compensation"])
        writer.writerow(["CapEx % of Revenue", f"{CAPEX_PCT*100:.1f}%", "Asset-light SaaS model"])
        writer.writerow(["Tax Rate", f"{TAX_RATE*100:.0f}%", "Effective tax rate"])
        writer.writerow(["A/R Days (DSO)", f"{AR_DAYS}", "Days Sales Outstanding"])
        writer.writerow(["A/P Days (DPO)", f"{AP_DAYS}", "Days Payable Outstanding"])
        writer.writerow(["Deferred Rev Days", f"{DEFERRED_REV_DAYS}", "Annual SaaS contracts"])
        
        # Growth and margin schedule
        writer.writerow([])
        writer.writerow(["--- GROWTH & MARGIN SCHEDULE ---"])
        writer.writerow(["Year", "Revenue Growth", "EBITDA Margin", "Rule of 40"])
        for year in YEARS:
            r40 = GROWTH_RATES[year]*100 + EBITDA_MARGINS[year]*100
            writer.writerow([str(year), f"{GROWTH_RATES[year]*100:.0f}%", f"{EBITDA_MARGINS[year]*100:.0f}%", f"{r40:.0f}"])
        
        writer.writerow([])
        writer.writerow(["--- END OF MODEL ---"])
    
    print(f"✓ Financial model written to {filename}")


def main():
    """Main function to generate the financial model."""
    print(f"\n{'='*60}")
    print(f"Generating {COMPANY_NAME} Financial Model")
    print(f"{'='*60}\n")
    
    # Step 1: Calculate revenue schedule
    print("Step 1: Calculating revenue schedule...")
    revenues = calculate_revenue_schedule()
    for year in YEARS:
        print(f"  {year}: ${revenues[year]:.1f}M")
    
    # Step 2: Generate Income Statement
    print("\nStep 2: Generating Income Statement...")
    is_data = generate_income_statement(revenues)
    
    # Step 3: Generate Balance Sheet (initial)
    print("Step 3: Generating Balance Sheet...")
    bs_data = generate_balance_sheet(revenues, is_data)
    
    # Step 4: Generate Cash Flow Statement (this also finalizes BS cash balances)
    print("Step 4: Generating Cash Flow Statement...")
    cf_data = generate_cash_flow_statement(revenues, is_data, bs_data)
    
    # Step 5: Calculate metrics
    print("Step 5: Calculating key metrics...")
    metrics = calculate_metrics(revenues, is_data)
    
    # Step 6: Validate
    print("\nStep 6: Validating model integrity...")
    validate_balance_sheet(bs_data)
    validate_cash_flow(cf_data)
    
    # Step 7: Write to CSV
    print("\nStep 7: Writing to CSV...")
    output_file = "LBO/cloudnova_financials.csv"
    write_to_csv(revenues, is_data, bs_data, cf_data, metrics, output_file)
    
    # Print summary
    print(f"\n{'='*60}")
    print("MODEL SUMMARY")
    print(f"{'='*60}")
    print(f"Company: {COMPANY_NAME}")
    print(f"Current Year (2024) Revenue: ${revenues[2024]:.1f}M")
    print(f"Current Year Growth: {GROWTH_RATES[2024]*100:.0f}%")
    print(f"Current Year EBITDA Margin: {EBITDA_MARGINS[2024]*100:.0f}%")
    print(f"Rule of 40 Score: {GROWTH_RATES[2024]*100 + EBITDA_MARGINS[2024]*100:.0f}")
    print(f"\nOutput file: {output_file}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
