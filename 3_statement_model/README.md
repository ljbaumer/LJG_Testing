# 3-Statement Financial Model

A Python application using Streamlit to build a fully linked 3-statement financial model from Excel financial data.

## Features

- **Excel Data Import**: Upload Excel files containing historical financial statements
- **CSV Data Store**: Automatically converts Excel data to CSV for easier processing
- **Fully Linked 3-Statement Model**: Income Statement, Balance Sheet, and Cash Flow Statement
- **Scenario Analysis**: Base, Upside, and Downside scenarios
- **Interactive Visualization**: Charts and graphs for financial data analysis
- **Customizable Parameters**: Adjust growth rates, margins, and other financial metrics
- **Audit Trail**: Track changes and assumptions

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/3-statement-model.git
cd 3-statement-model
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the Streamlit application:
```bash
streamlit run main.py
```

2. Upload an Excel file containing financial statements:
   - The Excel file should have separate sheets for Income Statement, Balance Sheet, and Cash Flow Statement
   - The application will attempt to automatically identify the sheets based on common naming conventions
   - If the sheets cannot be identified, the first three sheets will be used

3. Adjust the model parameters in the sidebar:
   - Forecast years
   - Revenue growth rate
   - EBITDA margin
   - Tax rate
   - Working capital assumptions
   - And more...

4. Run the forecast to generate the 3-statement model

5. Explore the results in the different tabs:
   - Data Input: View the imported historical data
   - Income Statement: View and analyze the income statement forecast
   - Balance Sheet: View and analyze the balance sheet forecast
   - Cash Flow: View and analyze the cash flow statement forecast
   - Scenarios: Compare different scenarios (Base, Upside, Downside)

## Project Structure

```
3-statement-model/
├── data/                  # Data directory for storing input and processed data
├── models/                # Financial model implementation
│   ├── data_manager.py    # Handles loading, saving, and validating financial data
│   ├── financial_model.py # Core 3-statement financial model implementation
│   └── scenario_manager.py # Manages different scenarios for the model
├── utils/                 # Utility functions and classes
│   ├── excel_converter.py # Converts Excel files to CSV
│   ├── financial_utils.py # Financial calculation utilities
│   └── visualization.py   # Data visualization utilities
├── main.py                # Main Streamlit application
└── requirements.txt       # Python dependencies
```

## Sample Data

A sample Excel file (`ZM_Financials.xlsx`) is included in the `data` directory. This file contains historical financial data for Zoom Video Communications, Inc. (ZM) that can be used to test the application.

## Customization

The application is designed to be flexible and can be customized in several ways:

- **Adding New Metrics**: Extend the financial model by adding new metrics in the `financial_model.py` file
- **Custom Scenarios**: Create custom scenarios beyond the default Base, Upside, and Downside scenarios
- **Visualization**: Add new charts and visualizations in the `visualization.py` file
- **Data Import**: Modify the Excel import logic in the `excel_converter.py` file to support different Excel formats

## Requirements

- Python 3.8+
- Pandas
- NumPy
- Streamlit
- Plotly
- openpyxl
- xlrd

## License

This project is licensed under the MIT License - see the LICENSE file for details.
