# DCF Valuation Tool 📊💰

A Streamlit application for building and analyzing Discounted Cash Flow (DCF) models.

## Features

- **User-Friendly Interface**: Clean, intuitive design with helpful emojis and clear instructions
- **Sample Company Data**: Pre-loaded financial data for sample companies
- **Historical Analysis**: Visualize and analyze historical financial performance
- **DCF Valuation**: Build DCF models with customizable parameters
- **Multiple Terminal Value Methods**: Choose between Perpetuity Growth and Exit Multiple methods
- **Sensitivity Analysis**: Perform single-factor and two-factor sensitivity analyses
- **Audit Trail**: Track all assumptions and calculations for transparency
- **Downloadable Results**: Export projections and valuation results to CSV and Excel

## Installation

1. Clone this repository:
   ```
   git clone <repository-url>
   cd Heidi_DCF_App
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Run the Streamlit application:
   ```
   streamlit run main.py
   ```

2. Open your web browser and navigate to the URL displayed in the terminal (usually http://localhost:8501)

3. Use the sidebar to select a company and adjust DCF parameters

4. Navigate between different sections using the radio buttons in the sidebar:
   - Company Overview
   - Historical Analysis
   - DCF Valuation
   - Sensitivity Analysis
   - Audit Trail

## Customization

### Adding New Companies

To add new companies to the application, edit the `SAMPLE_COMPANIES` dictionary in `data.py`. Each company should include:

- Name
- Sector
- Description
- Current price
- Shares outstanding
- Debt
- Cash
- Historical financials (revenue, EBITDA, EBIT, net income, capex, depreciation, NWC change)

### Modifying Industry Averages

Industry averages can be modified in the `INDUSTRY_AVERAGES` dictionary in `data.py`.

## Advanced Settings

The application provides advanced settings for customizing:

- Growth rates
- EBITDA margins
- Tax rates
- Capital expenditures
- Depreciation
- Net working capital changes
- WACC (with a built-in WACC calculator)
- Terminal value parameters

## Dependencies

- Streamlit
- Pandas
- NumPy
- Plotly
- Matplotlib
- XlsxWriter
- Pillow

## License

This project is licensed under the MIT License - see the LICENSE file for details.
