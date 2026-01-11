import pandas as pd
import numpy as np
from dataclasses import dataclass

@dataclass
class LBOAssumptions:
    # Transaction
    entry_multiple: float
    exit_multiple: float
    entry_date: int  # Year
    exit_year: int   # Year
    
    # Financing (Start with simple structure)
    senior_debt_multiple: float
    sub_debt_multiple: float
    senior_interest_rate: float
    sub_interest_rate: float
    min_cash: float
    tax_rate: float

def calculate_lbo(financials_df, assumptions: LBOAssumptions):
    """
    Calculate LBO returns based on company financials and deal assumptions.
    Expects financials_df to be the dataframe loaded from the CSV.
    """
    # 1. Setup Timeline & Base Data
    start_year = assumptions.entry_date
    exit_year = assumptions.exit_year
    projection_years = list(range(start_year, exit_year + 1))
    
    # Filter/Load relevant financial data
    # We need strict mapping of rows. Using the known structure from generate_financials.py
    # Pivot to make years columns if not already, but our load function returns sections
    # We'll assume we get a dictionary of dataframes like the viewer does
    
    # Extract EBITDA for entry/exit calc
    ebitda_row = financials_df["INCOME STATEMENT"][financials_df["INCOME STATEMENT"].iloc[:,0] == "EBITDA"]
    
    # Get entry EBITDA (LTM at entry)
    entry_ebitda = float(ebitda_row[str(start_year)].values[0])
    
    # 2. Sources & Uses
    # Uses
    purchase_price = entry_ebitda * assumptions.entry_multiple
    fees = purchase_price * 0.02 # Est 2% fees
    min_cash = assumptions.min_cash
    total_uses = purchase_price + fees + min_cash
    
    # Sources
    senior_debt = entry_ebitda * assumptions.senior_debt_multiple
    sub_debt = entry_ebitda * assumptions.sub_debt_multiple
    total_debt = senior_debt + sub_debt
    sponsor_equity = total_uses - total_debt
    
    sources_uses = {
        "Sources": {
            "Senior Debt": senior_debt,
            "Subordinated Debt": sub_debt,
            "Sponsor Equity": sponsor_equity
        },
        "Uses": {
            "Purchase Price": purchase_price,
            "Transaction Fees": fees,
            "Minimum Cash": min_cash
        }
    }
    
    # 3. Debt Schedule & Projections
    lbo_table = []
    
    # Initialize balances
    senior_balance = senior_debt
    sub_balance = sub_debt
    cum_cash = min_cash  # This will track cash ON TOP of min_cash
    
    for year in projection_years:
        # Get Flow metrics from input df
        col = str(year)
        
        # Helper to get value safely
        def get_val(section, item):
            row = financials_df[section][financials_df[section].iloc[:,0] == item]
            if row.empty: return 0.0
            return float(row[col].values[0])
            
        ebitda = get_val("INCOME STATEMENT", "EBITDA")
        capex = get_val("CASH FLOW STATEMENT", "Capital Expenditures") # is negative in source
        nwc_change = get_val("CASH FLOW STATEMENT", "Total Change in Working Capital")
        tax = get_val("INCOME STATEMENT", "Income Tax") # is negative
        
        # Debt Service
        senior_interest = senior_balance * assumptions.senior_interest_rate
        sub_interest = sub_balance * assumptions.sub_interest_rate
        total_interest = senior_interest + sub_interest
        
        # Free Cash Flow calculation
        # Adjusted for new capital structure interest (tax shield impact simplified for now)
        # FCF = EBITDA - Capex - NWC - Tax - Interest
        
        # Recalculate tax based on new interest?
        # EBIT is EBITDA - D&A. Interest reduces EBT. 
        # Let's do a mini income statement walk
        depreciation = get_val("INCOME STATEMENT", "Depreciation & Amortization") # negative
        ebit = ebitda + depreciation
        ebt = ebit - total_interest
        new_tax = max(0, ebt) * assumptions.tax_rate
        
        # Cash Flow Available for Debt Repayment (CFADS)
        # CFADS = EBITDA - Capex (abs) +/- NWC - Taxes - Interest
        # Note: In source, Capex/Tax/Depr are negative. keeping signs logic consistent.
        fcf = ebitda + capex + nwc_change - new_tax - total_interest
        
        # Mandatory Amortization (Assume 1% of original senior debt)
        mandatory_amort = min(senior_balance, senior_debt * 0.01)
        
        # Cash Sweep
        cash_avail_for_sweep = max(0, fcf - mandatory_amort)
        cash_sweep = min(senior_balance - mandatory_amort, cash_avail_for_sweep)
        
        # Update Balances
        senior_paydown = mandatory_amort + cash_sweep
        senior_balance -= senior_paydown
        
        # Check if there is any remaining cash flow after max sweep
        # (i.e. if debt is fully paid off but we still have FCF)
        # cash_avail_for_sweep was the FCF available.
        # actual sweep was limited by debt balance.
        excess_cash_flow = cash_avail_for_sweep - cash_sweep
        cum_cash += excess_cash_flow
        
        # Sub debt usually bullet, no paydown until exit
        
        row_data = {
            "Year": year,
            "EBITDA": ebitda,
            "EBIT": ebit,
            "Interest Expense": total_interest,
            "Taxes": new_tax,
            "FCF": fcf,
            "Senior Debt Begin": senior_balance + senior_paydown,
            "Senior Paydown": senior_paydown,
            "Senior Debt End": senior_balance,
            "Sub Debt End": sub_balance,
            "Total Debt End": senior_balance + sub_balance,
            "Ending Cash": cum_cash
        }
        lbo_table.append(row_data)

    lbo_df = pd.DataFrame(lbo_table)
    
    # 4. Returns Analysis
    exit_ebitda = lbo_df.iloc[-1]["EBITDA"]
    exit_valuation = exit_ebitda * assumptions.exit_multiple
    
    # Net Debt = Total Debt - (Ending Cash - Min Cash requirement is handled? No, cum_cash includes min_cash)
    # Wait, cum_cash initialized at min_cash. So it is Total Cash.
    # Equity Value = Enterprise Value - Net Debt
    # Net Debt = Gross Debt - Total Cash
    
    ending_gross_debt = lbo_df.iloc[-1]["Total Debt End"]
    ending_total_cash = lbo_df.iloc[-1]["Ending Cash"]
    net_debt_exit = ending_gross_debt - ending_total_cash
    
    equity_value_exit = exit_valuation - net_debt_exit
    
    moic = equity_value_exit / sponsor_equity
    num_years = exit_year - start_year
    irr = (moic ** (1/num_years)) - 1
    
    returns = {
        "Entry Equity": sponsor_equity,
        "Exit Equity": equity_value_exit,
        "MOIC": moic,
        "IRR": irr,
        "Exit Year": exit_year,
        "Exit EBITDA": exit_ebitda,
        "Exit Enterprise Value": exit_valuation
    }
    
    return sources_uses, lbo_df, returns
