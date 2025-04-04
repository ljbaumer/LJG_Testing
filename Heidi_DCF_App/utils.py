"""
Utilities module for the DCF application.
Contains helper functions and utilities.
"""

import pandas as pd
import numpy as np
import streamlit as st
import base64
from io import BytesIO

def create_download_link(df, filename, text):
    """
    Create a download link for a DataFrame.
    
    Args:
        df (pandas.DataFrame): DataFrame to download
        filename (str): Filename for the download
        text (str): Text to display for the download link
        
    Returns:
        str: HTML link for downloading the DataFrame
    """
    csv = df.to_csv(index=True)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{text}</a>'
    return href

def create_excel_download_link(df_dict, filename, text):
    """
    Create a download link for multiple DataFrames as Excel.
    
    Args:
        df_dict (dict): Dictionary of sheet_name: DataFrame pairs
        filename (str): Filename for the download
        text (str): Text to display for the download link
        
    Returns:
        str: HTML link for downloading the Excel file
    """
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        for sheet_name, df in df_dict.items():
            df.to_excel(writer, sheet_name=sheet_name)
    
    excel_data = output.getvalue()
    b64 = base64.b64encode(excel_data).decode()
    href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="{filename}">{text}</a>'
    return href

def display_company_info(company_data):
    """
    Display company information in a formatted way.
    
    Args:
        company_data (dict): Company data
    """
    st.subheader(f"{company_data['name']} ({company_data['sector']})")
    st.write(company_data['description'])
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="Current Price", 
            value=f"${company_data['current_price']:.2f}"
        )
    
    with col2:
        market_cap = company_data['current_price'] * company_data['shares_outstanding']
        if market_cap >= 1e9:
            market_cap_str = f"${market_cap / 1e9:.2f}B"
        else:
            market_cap_str = f"${market_cap / 1e6:.2f}M"
        
        st.metric(
            label="Market Cap", 
            value=market_cap_str
        )
    
    with col3:
        enterprise_value = market_cap + company_data['debt'] - company_data['cash']
        if enterprise_value >= 1e9:
            ev_str = f"${enterprise_value / 1e9:.2f}B"
        else:
            ev_str = f"${enterprise_value / 1e6:.2f}M"
        
        st.metric(
            label="Enterprise Value", 
            value=ev_str
        )

def format_large_numbers(num):
    """
    Format large numbers with K, M, B suffixes.
    
    Args:
        num (float): Number to format
        
    Returns:
        str: Formatted number
    """
    if abs(num) >= 1e9:
        return f"{num / 1e9:.2f}B"
    elif abs(num) >= 1e6:
        return f"{num / 1e6:.2f}M"
    elif abs(num) >= 1e3:
        return f"{num / 1e3:.2f}K"
    else:
        return f"{num:.2f}"

def create_audit_trail(params):
    """
    Create an audit trail of DCF parameters.
    
    Args:
        params (dict): DCF parameters
        
    Returns:
        pandas.DataFrame: Audit trail
    """
    audit_data = []
    
    # Add basic parameters
    audit_data.append(["Base Revenue", format_large_numbers(params["base_revenue"])])
    audit_data.append(["Forecast Years", params["forecast_years"]])
    audit_data.append(["WACC", f"{params['wacc'] * 100:.2f}%"])
    
    # Add growth rates
    if isinstance(params["growth_rates"], list):
        for i, rate in enumerate(params["growth_rates"]):
            audit_data.append([f"Growth Rate Year {i+1}", f"{rate * 100:.2f}%"])
    else:
        audit_data.append(["Growth Rate (All Years)", f"{params['growth_rates'] * 100:.2f}%"])
    
    # Add EBITDA margins
    if isinstance(params["ebitda_margins"], list):
        for i, margin in enumerate(params["ebitda_margins"]):
            audit_data.append([f"EBITDA Margin Year {i+1}", f"{margin * 100:.2f}%"])
    else:
        audit_data.append(["EBITDA Margin (All Years)", f"{params['ebitda_margins'] * 100:.2f}%"])
    
    # Add tax rates
    if isinstance(params["tax_rates"], list):
        for i, rate in enumerate(params["tax_rates"]):
            audit_data.append([f"Tax Rate Year {i+1}", f"{rate * 100:.2f}%"])
    else:
        audit_data.append(["Tax Rate (All Years)", f"{params['tax_rates'] * 100:.2f}%"])
    
    # Add terminal value method
    if params["terminal_value_method"] == "perpetuity":
        audit_data.append(["Terminal Value Method", "Perpetuity Growth"])
        audit_data.append(["Terminal Growth Rate", f"{params['terminal_growth'] * 100:.2f}%"])
    else:
        audit_data.append(["Terminal Value Method", "Exit Multiple"])
        audit_data.append(["EBITDA Multiple", f"{params['ebitda_multiple']:.1f}x"])
    
    # Create DataFrame
    df = pd.DataFrame(audit_data, columns=["Parameter", "Value"])
    
    return df

def display_emoji_header():
    """
    Display a header with emojis for the DCF application.
    """
    st.markdown("""
    # 📊 DCF Valuation Tool 💰
    
    Welcome to the Discounted Cash Flow (DCF) Valuation Tool! This application helps you build 
    and analyze DCF models for company valuation. Let's get started! ✨
    """)

def display_emoji_footer():
    """
    Display a footer with emojis for the DCF application.
    """
    st.markdown("""
    ---
    ### Thanks for using the DCF Valuation Tool! 🚀
    
    We hope this tool helps you make better investment decisions. 
    Remember that DCF is just one valuation method and should be used alongside other analyses. 🧠💡
    
    Happy investing! 📈
    """)

def display_section_header(title, emoji):
    """
    Display a section header with an emoji.
    
    Args:
        title (str): Section title
        emoji (str): Emoji to display
    """
    st.markdown(f"## {emoji} {title}")

def get_color_scale(values, min_value=None, max_value=None, center=None):
    """
    Get a color scale for a range of values.
    
    Args:
        values (list or numpy.ndarray): Values to create color scale for
        min_value (float): Minimum value for the scale
        max_value (float): Maximum value for the scale
        center (float): Center value for the scale
        
    Returns:
        list: List of colors
    """
    if min_value is None:
        min_value = np.min(values)
    
    if max_value is None:
        max_value = np.max(values)
    
    if center is None:
        center = (min_value + max_value) / 2
    
    colors = []
    for value in values:
        if value < center:
            # Red to yellow
            ratio = (value - min_value) / (center - min_value)
            r = 255
            g = int(255 * ratio)
            b = 0
        else:
            # Yellow to green
            ratio = (value - center) / (max_value - center)
            r = int(255 * (1 - ratio))
            g = 255
            b = 0
        
        colors.append(f"rgb({r}, {g}, {b})")
    
    return colors

def create_wacc_calculator():
    """
    Create a WACC calculator widget.
    
    Returns:
        float: Calculated WACC
    """
    st.subheader("WACC Calculator")
    
    col1, col2 = st.columns(2)
    
    with col1:
        risk_free_rate = st.slider("Risk-Free Rate (%)", 0.0, 10.0, 3.0, 0.1) / 100
        market_risk_premium = st.slider("Market Risk Premium (%)", 0.0, 15.0, 5.5, 0.1) / 100
        beta = st.slider("Beta", 0.0, 3.0, 1.0, 0.05)
        
    with col2:
        cost_of_debt = st.slider("Cost of Debt (%)", 0.0, 15.0, 4.0, 0.1) / 100
        tax_rate = st.slider("Tax Rate (%)", 0.0, 50.0, 25.0, 0.5) / 100
        debt_weight = st.slider("Debt Weight (%)", 0.0, 100.0, 30.0, 1.0) / 100
    
    equity_weight = 1 - debt_weight
    
    # Calculate WACC
    cost_of_equity = risk_free_rate + beta * market_risk_premium
    wacc = equity_weight * cost_of_equity + debt_weight * cost_of_debt * (1 - tax_rate)
    
    # Display WACC components
    st.write("### WACC Components")
    
    wacc_data = [
        ["Cost of Equity", f"{cost_of_equity:.2%}", f"{equity_weight:.2%}", f"{cost_of_equity * equity_weight:.2%}"],
        ["Cost of Debt (After-Tax)", f"{cost_of_debt * (1 - tax_rate):.2%}", f"{debt_weight:.2%}", f"{cost_of_debt * (1 - tax_rate) * debt_weight:.2%}"],
        ["WACC", "", "", f"{wacc:.2%}"]
    ]
    
    wacc_df = pd.DataFrame(wacc_data, columns=["Component", "Rate", "Weight", "Contribution"])
    st.table(wacc_df)
    
    return wacc
