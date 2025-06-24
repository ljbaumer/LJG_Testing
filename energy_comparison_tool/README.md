# Energy Source Comparison Tool for AI Data Centers

A comprehensive Streamlit application for comparing different energy sources for powering advanced AI data centers. This tool provides interactive visualizations and detailed analysis to help understand the relative merits of various power sources based on key metrics.

## Overview

As artificial intelligence continues to advance, the energy requirements for AI data centers are growing exponentially. Choosing the right energy sources for these facilities is crucial for sustainability, reliability, and cost-effectiveness.

This application allows users to:
- Compare 11 different energy sources across 10 key metrics
- Visualize data through interactive charts and graphs
- Explore detailed information about specific energy sources
- Understand the unique energy requirements of AI data centers
- Access credible data sources and methodology information

## Energy Sources Analyzed

The application includes data on the following energy sources:
- Natural Gas (Combined Cycle)
- Nuclear (Fission)
- Solar PV (Utility Scale)
- Wind (Onshore)
- Wind (Offshore)
- Hydroelectric
- Coal
- Geothermal
- Biomass
- Hydrogen Fuel Cells
- Small Modular Reactors (SMRs)

## Comparison Metrics

Energy sources are compared across these key metrics:
- Levelized Cost of Electricity (LCOE) - $/MWh
- Power Density - MW/km²
- Carbon Intensity - gCO2eq/kWh
- Capacity Factor - %
- Construction Time - Years
- Operational Lifespan - Years
- Water Usage - Gallons/MWh
- Land Use - m²/MWh
- Grid Reliability/Dispatchability - Scale (1-10)
- Scalability for Data Centers - Scale (1-10)

## Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package installer)

### Setup

1. Clone this repository or download the source code.

2. Navigate to the project directory:
   ```
   cd energy_comparison_tool
   ```

3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. From the project directory, run the Streamlit application:
   ```
   streamlit run main.py
   ```

2. The application will open in your default web browser. If it doesn't, navigate to the URL shown in the terminal (typically http://localhost:8501).

3. Use the sidebar navigation to explore different sections of the application:
   - **Introduction**: Overview of energy sources and metrics
   - **Single Metric Comparison**: Compare all energy sources based on a single metric
   - **Multi-Metric Comparison**: Compare energy sources across multiple metrics using radar charts, parallel coordinates, and scatter plots
   - **Energy Source Deep Dive**: Detailed information about specific energy sources
   - **AI Data Center Context**: Information about the unique energy requirements of AI data centers
   - **Data Sources**: Details about the data sources and methodology

## Data Sources

The data in this application is compiled from various credible sources, including:
- International Energy Agency (IEA)
- U.S. Energy Information Administration (EIA)
- National Renewable Energy Laboratory (NREL)
- Lazard's Levelized Cost of Energy Analysis
- Various peer-reviewed academic papers

For detailed information about the sources for each metric, see the "Data Sources" section in the application.

## Features

### Interactive Visualizations
- Bar charts for comparing metrics across energy sources
- Radar charts for multi-dimensional comparison
- Heatmaps for overall comparison
- Scatter plots for examining relationships between metrics
- Parallel coordinates plots for multi-metric analysis
- Gauge charts for suitability scores

### Detailed Information
- Comprehensive explanations of each energy source
- Detailed descriptions of each metric and its relevance to AI data centers
- Context about AI data center energy requirements
- Case studies of real-world approaches to powering AI infrastructure
- Future trends in energy for data centers

### User-Friendly Interface
- Clean, intuitive design
- Sidebar navigation for easy access to different sections
- Interactive elements for customizing visualizations
- Informative tooltips and explanations throughout

## License

This project is open source and available under the MIT License.

## Acknowledgments

- Data sources are cited in the application
- Visualization libraries: Plotly, Matplotlib, Seaborn
- Web application framework: Streamlit
