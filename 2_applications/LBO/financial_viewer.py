import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from lbo_engine import calculate_lbo, LBOAssumptions

st.set_page_config(
    page_title="CloudNova LBO Model",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better table sizing and styling
st.markdown("""
<style>
    .dataframe {
        font-size: 14px !important;
        font-family: "Source Sans Pro", sans-serif !important;
    }
    /* Clean up index column */
    thead tr th:first-child { display:none }
    tbody th { display:none }
    
    /* Header styling */
    thead th {
        background-color: #f0f2f6;
        color: #31333F;
        font-weight: 600;
    }
    
    /* Section headers in the table using pandas styling instead */
</style>
""", unsafe_allow_html=True)

def load_financial_data(filepath):
    """Load and parse the financial CSV file."""
    # Read the CSV, skipping header rows to find the actual data
    # Based on the file structure, we know where sections start, but let's parse it robustly
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    # Parse manually to handle sections properly
    sections = {}
    current_section = None
    current_data = []
    headers = []
    
    for line in lines:
        line = line.strip()
        if not line: 
            continue
            
        if line.startswith("---") and line.endswith("---"):
            # Save previous section if exists
            if current_section and current_data:
                df = pd.DataFrame(current_data, columns=headers)
                sections[current_section] = df
            
            # Start new section
            current_section = line.strip("- ")
            current_data = []
            headers = []
        elif current_section:
            parts = [p.strip() for p in line.split(',')]
            
            # Handle lines with fewer columns than headers (e.g. assumption notes)
            # Or lines that are just headers
            if not headers:
                headers = parts
            else:
                # Pad or trim to match header length
                if len(parts) < len(headers):
                    parts += [''] * (len(headers) - len(parts))
                elif len(parts) > len(headers):
                    parts = parts[:len(headers)]
                
                # Skip header repetition if encountered
                if parts != headers:
                    current_data.append(parts)
    
    # Save the last section
    if current_section and current_data:
        df = pd.DataFrame(current_data, columns=headers)
        sections[current_section] = df
        
    return sections

def format_currency(val):
    """Format string value as currency string."""
    try:
        f_val = float(val)
        if abs(f_val) < 0.01:
            return "-"
        return f"${f_val:,.1f}"
    except (ValueError, TypeError):
        return val

def format_percentage(val):
    """Format string value as percentage string."""
    try:
        f_val = float(val)
        return f"{f_val:.1f}%"
    except (ValueError, TypeError):
        return val

def display_section(title, df, format_type="currency", highlight_first_col=True):
    """Display a financial section with proper formatting."""
    st.subheader(title)
    
    # Create a display dataframe
    display_df = df.copy()
    
    # Identify numeric columns (all except first)
    numeric_cols = display_df.columns[1:]
    
    # Apply formatting to cells for display
    for col in numeric_cols:
        if format_type == "currency":
            display_df[col] = display_df[col].apply(format_currency)
        elif format_type == "percentage":
            display_df[col] = display_df[col].apply(format_percentage)
            
    # Configure dataframe display
    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            display_df.columns[0]: st.column_config.TextColumn(
                "Line Item",
                width="medium",
                disabled=True,
            )
        }
    )

def render_financial_viewer(sections):
    st.title("CloudNova Solutions - Financial Viewer")
    
    # Sidebar navigation specific to this tab
    options = list(sections.keys())
    selection = st.radio("Go to section:", ["All"] + options, horizontal=True)
    
    st.markdown("---")
    
    # Display logic
    if selection == "All" or selection == "INCOME STATEMENT":
        if "INCOME STATEMENT" in sections:
            display_section("Income Statement", sections["INCOME STATEMENT"])
            
    if selection == "All" or selection == "BALANCE SHEET":
        if "BALANCE SHEET" in sections:
            display_section("Balance Sheet", sections["BALANCE SHEET"])
            
    if selection == "All" or selection == "CASH FLOW STATEMENT":
        if "CASH FLOW STATEMENT" in sections:
            display_section("Cash Flow Statement", sections["CASH FLOW STATEMENT"])
            
    if selection == "All" or selection == "KEY METRICS":
        if "KEY METRICS" in sections:
            display_section("Key Metrics", sections["KEY METRICS"], format_type="percentage")
            
    if selection == "All" or selection == "MODEL ASSUMPTIONS":
        if "MODEL ASSUMPTIONS" in sections:
            st.subheader("Model Assumptions")
            st.dataframe(sections["MODEL ASSUMPTIONS"], use_container_width=True, hide_index=True)
            
    if selection == "All" or selection == "GROWTH & MARGIN SCHEDULE":
        if "GROWTH & MARGIN SCHEDULE" in sections:
            st.subheader("Growth & Margin Schedule")
            st.dataframe(sections["GROWTH & MARGIN SCHEDULE"], use_container_width=True, hide_index=True)

def render_lbo_model(sections):
    st.title("LBO Modeling Engine")
    st.markdown("Configure LBO assumptions below and verify returns.")
    
    with st.form("lbo_assumptions_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("Transaction")
            entry_mult = st.number_input(
                "Entry Multiple (x EBITDA)", 
                value=18.0, step=0.5,
                help="Purchase Price as a multiple of Entry EBITDA. Software assets often trade at 15x-20x+."
            )
            exit_mult = st.number_input(
                "Exit Multiple (x EBITDA)", 
                value=18.0, step=0.5,
                help="Exit Valuation as a multiple of Exit EBITDA. Typically assumed to be equal or lower than entry."
            )
            entry_year = st.selectbox("Entry Year", [2024], disabled=True) 
            exit_year = st.selectbox(
                "Exit Year", 
                [2027, 2028, 2029], index=2,
                help="Year the company is sold. Longer hold periods generally lower IRR but may increase MOIC."
            )
            
        with col2:
            st.subheader("Financing Structure")
            senior_lev = st.number_input(
                "Senior Debt (x EBITDA)", 
                value=4.0, step=0.25, 
                help="Amount of senior debt borrowed at entry. Higher leverage increases IRR (if growth is positive) but adds risk."
            )
            sub_lev = st.number_input(
                "Sub Debt (x EBITDA)", 
                value=1.0, step=0.25,
                help="Amount of subordinated debt borrowed at entry. Increases leverage further than senior lenders allow."
            )
            min_cash = st.number_input(
                "Min Cash Balance ($M)", 
                value=10.0, step=1.0,
                help="Minimum cash required on balance sheet. Money trapped here cannot be used to pay down debt."
            )
            
        with col3:
            st.subheader("Operating / Rates")
            senior_rate = st.number_input(
                "Senior Interest Rate", 
                value=0.08, step=0.005, format="%.3f",
                help="Annual interest rate on senior debt. Higher rates reduce free cash flow available for debt paydown, lowering equity value."
            )
            sub_rate = st.number_input(
                "Sub Debt Interest Rate", 
                value=0.12, step=0.005, format="%.3f",
                help="Annual interest rate on subordinated debt. Higher rates increase interest expense and reduce returns."
            )
            tax_rate = st.number_input(
                "Tax Rate", 
                value=0.25, step=0.01, format="%.2f",
                help="Corporate tax rate. Higher taxes reduce free cash flow available for debt paydown."
            )
            
        submitted = st.form_submit_button("Calculate LBO", type="primary")
        
    if submitted:
        # Create assumptions object
        assumptions = LBOAssumptions(
            entry_multiple=entry_mult,
            exit_multiple=exit_mult,
            entry_date=entry_year,
            exit_year=exit_year,
            senior_debt_multiple=senior_lev,
            sub_debt_multiple=sub_lev,
            senior_interest_rate=senior_rate,
            sub_interest_rate=sub_rate,
            min_cash=min_cash,
            tax_rate=tax_rate
        )
        
        # Calculate
        sources_uses, lbo_df, returns = calculate_lbo(sections, assumptions)
        
        st.markdown("---")
        
        # 1. Returns Summary KPI Cards
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        kpi1.metric("IRR", f"{returns['IRR']*100:.1f}%")
        kpi2.metric("MOIC", f"{returns['MOIC']:.2f}x")
        kpi3.metric("Sponsor Equity Check", f"${returns['Entry Equity']:.1f}M")
        kpi4.metric("Exit Equity Value", f"${returns['Exit Equity']:.1f}M")
        
        st.markdown("---")
        
        # 2. Sources & Uses
        col_su1, col_su2 = st.columns(2)
        with col_su1:
            st.subheader("Sources")
            s_df = pd.DataFrame(list(sources_uses["Sources"].items()), columns=["Source", "Amount"])
            s_df["Amount"] = s_df["Amount"].apply(lambda x: f"${x:,.1f}M")
            st.dataframe(s_df, use_container_width=True, hide_index=True)
            
        with col_su2:
            st.subheader("Uses")
            u_df = pd.DataFrame(list(sources_uses["Uses"].items()), columns=["Use", "Amount"])
            u_df["Amount"] = u_df["Amount"].apply(lambda x: f"${x:,.1f}M")
            st.dataframe(u_df, use_container_width=True, hide_index=True)
            
        # 3. LBO Charts (Cash Flow & Debt Paydown)
        st.subheader("Projected Returns & Deleveraging")
        
        fig = go.Figure()
        fig.add_trace(go.Bar(x=lbo_df["Year"], y=lbo_df["Senior Debt End"], name="Senior Debt"))
        fig.add_trace(go.Bar(x=lbo_df["Year"], y=lbo_df["Sub Debt End"], name="Sub Debt"))
        fig.add_trace(go.Scatter(x=lbo_df["Year"], y=lbo_df["EBITDA"], name="EBITDA", yaxis="y2", line=dict(color='green', width=3)))
        
        fig.update_layout(
            barmode='stack',
            title="Debt Balance vs EBITDA",
            yaxis=dict(title="Debt ($M)"),
            yaxis2=dict(title="EBITDA ($M)", overlaying='y', side='right'),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # 4. Detailed Schedule
        st.subheader("Detailed Debt Schedule")
        
        # Format the dataframe for display
        display_df = lbo_df.copy()
        cols_to_format = [c for c in display_df.columns if c != "Year"]
        for c in cols_to_format:
            display_df[c] = display_df[c].apply(lambda x: f"${x:,.1f}")
            
        st.dataframe(display_df, use_container_width=True, hide_index=True)

def main():
    try:
        # Load data
        sections = load_financial_data("LBO/cloudnova_financials.csv")
        
        tab1, tab2 = st.tabs(["Start: Company Financials", "Step 2: LBO Model"])
        
        with tab1:
            render_financial_viewer(sections)
            
        with tab2:
            render_lbo_model(sections)
                
    except FileNotFoundError:
        st.error("Financial data file not found. Please run 'python LBO/generate_financials.py' first.")
    except Exception as e:
        st.error(f"Error in application: {str(e)}")

if __name__ == "__main__":
    main()
