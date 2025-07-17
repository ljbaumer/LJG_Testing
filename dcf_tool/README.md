# DCF Valuation Model

A Discounted Cash Flow (DCF) valuation model with a Streamlit GUI that is logically ordered, highly auditable, and includes toggles for standard features.

## Features

- **Interactive Financial Data Management**: View and edit historical financial data directly in the application
- **Flexible Forecasting**: Configure forecast parameters and extend projections up to 10 years
- **Multiple Terminal Value Methods**: Choose between Perpetuity Growth Method and Exit Multiple Method
- **WACC Calculation Options**: Calculate WACC using CAPM or provide a direct input
- **Detailed Audit Trail**: Track every calculation step for full transparency and auditability
- **Interactive Visualizations**: Explore forecast data, valuation components, and sensitivity analysis through interactive charts
- **Data Persistence**: Save modified financial data for future use

## Project Structure

```
dcf_tool/
├── main.py                 # Main Streamlit application entry point
├── data/
│   └── sample_data.csv     # Sample financial data
├── models/
│   ├── dcf_model.py        # Core DCF model calculations
│   ├── growth_methods.py   # Terminal value calculation methods
│   └── data_manager.py     # Data loading, saving, and validation
├── utils/
│   ├── financial_utils.py  # Financial calculation utilities
│   └── visualization.py    # Visualization utilities
└── requirements.txt        # Project dependencies
```

## Installation

1. Clone the repository
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Run the Streamlit application:

```bash
streamlit run main.py
```

### Using the Application

1. **Financial Data**: View and edit historical financial data in the "Financial Data" tab
2. **Model Parameters**: Configure forecast parameters, terminal value method, and discount rate in the sidebar
3. **Run Valuation**: Click the "Run Valuation" button to execute the DCF model
4. **Explore Results**: Navigate through the tabs to view forecast data, valuation results, and the audit trail
5. **Sensitivity Analysis**: Analyze how changes in key parameters affect the valuation

## Model Components

### Financial Data

The model uses the following financial metrics:

- Revenue and Revenue Growth
- EBITDA and EBITDA Margin
- Depreciation & Amortization
- EBIT (Earnings Before Interest and Taxes)
- Tax Rate
- NOPAT (Net Operating Profit After Tax)
- Capital Expenditures
- Change in Working Capital
- Free Cash Flow

### Forecast Parameters

- Forecast Years (1-10 years)
- Revenue Growth Rate
- EBITDA Margin
- Tax Rate
- Depreciation & Amortization to Revenue Ratio
- Capital Expenditures to Revenue Ratio
- Working Capital Change to Revenue Ratio

### Terminal Value Methods

1. **Perpetuity Growth Method**:
   - TV = FCF_t * (1 + g) / (WACC - g)
   - Where g is the long-term growth rate

2. **Exit Multiple Method**:
   - TV = Metric_t * Multiple
   - Supports EV/EBITDA and EV/EBIT multiples

### Discount Rate (WACC)

- Option to calculate WACC using CAPM:
  - Cost of Equity = Risk-Free Rate + Beta * Market Risk Premium
  - WACC = (E/V * Re) + (D/V * Rd * (1 - T))
- Option to provide WACC directly

## Audit Trail

The model provides two levels of audit trail:

1. **Valuation Steps**: High-level overview of each step in the valuation process
2. **Detailed Calculation Steps**: Detailed breakdown of each calculation, including formulas, input values, and results

## License

This project is open source and available under the MIT License.
