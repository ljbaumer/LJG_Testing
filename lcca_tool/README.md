# Life Cycle Cost Analysis (LCCA) Tool for Manhattan Skyscrapers

A Streamlit application for performing Life Cycle Cost Analysis on skyscraper construction projects in Manhattan.

## Overview

This application helps stakeholders analyze and compare the total cost of ownership for skyscraper projects over a 30-year period. It considers various cost categories including:

- Initial construction costs
- Operating costs (energy, utilities)
- Maintenance costs
- Replacement costs for building systems
- Financing costs

Users can adjust building parameters and cost assumptions to explore different scenarios and make informed decisions.

## Features

- **Interactive Parameter Adjustment**: Modify building specifications and cost assumptions through an intuitive sidebar interface
- **Comprehensive Cost Analysis**: Calculate total life cycle costs and present value over a 30-year period
- **Data Visualization**: View cost breakdowns, annual costs over time, and scenario comparisons through interactive charts
- **Scenario Comparison**: Save and compare up to three different scenarios to evaluate trade-offs

## Building Parameters

The tool allows you to adjust the following building parameters:

- Building height
- Number of floors
- Floor area
- Facade type (Glass Curtain Wall, Precast Concrete, Stone Veneer, Metal Panel, Mixed)
- Structural system (Steel Frame, Reinforced Concrete, Composite, Tube System)
- HVAC system (Variable Air Volume, Chilled Beams, VRF System, Hybrid System)
- Interior finish quality (Standard, Premium, Luxury, Ultra Luxury)

## Cost Assumptions

You can also modify key cost assumptions:

- Construction cost per square meter
- Annual inflation rate
- Discount rate
- Energy cost per square meter
- Maintenance cost as a percentage of construction cost

## Installation

1. Clone this repository
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Run the Streamlit application:

```bash
streamlit run main.py
```

## Screenshots

(Screenshots will be added here once the application is running)

## Dependencies

- streamlit
- pandas
- numpy
- matplotlib
- plotly
- scipy

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- This tool uses placeholder data and simplified models for educational purposes
- Cost factors and assumptions are based on industry averages and should be adjusted for specific projects
