# FL-Plotter Project Instructions

## Overview
FL-Plotter is a lightweight plotting library with a clean class-based interface for chart creation. It uses local mplstyle files and handles all plotting context through the `PlotBuddy` class.

## Setup

### Virtual Environment
The project uses a virtual environment for dependencies:
```bash
# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Key Files
- `src/utils/plot_buddy.py` - Main PlotBuddy class implementation
- `src/utils/styles/` - Directory containing matplotlib style files:
  - `gs.mplstyle` - Goldman Sachs default style
  - `gs_wide.mplstyle` - Wide format GS style
  - `gs_boxy.mplstyle` - Boxy format GS style
- `src/templates/area_charts/ai_power_demand_growth.py` - Example usage with PlotBuddy
- `requirements.txt` - Python dependencies

## Running Examples
```bash
# Make sure venv is activated first
source venv/bin/activate

# Run the AI power demand growth example
python3 src/templates/area_charts/ai_power_demand_growth.py
```

## Development Status
- ✅ Core PlotBuddy class implemented
- ✅ Local style loading without system installation
- ✅ Enhanced title, legend, and logo functionality
- ✅ Backward compatibility functions
- ✅ Example updated to use PlotBuddy

## Key Features
- **Local style loading**: Uses local `.mplstyle` files without system installation
- **All context managed**: PlotBuddy handles all styling and layout context
- **Enhanced functionality**: Better title, legend, and logo placement
- **Backward compatibility**: Old function-based API still works
- **Lightweight**: Minimal dependencies, fast execution

## Power Plant Curtailment Analysis

### Overview
The power plant curtailment analysis project (`src/active_projects/power_plant_curtailment/`) provides comprehensive analysis of power plant generation data for datacenter deployment scenarios. It analyzes 9 power plants across the US to determine optimal datacenter placement strategies using slack capacity, solar generation, battery storage, and backup generators.

### Analyzed Power Plants
1. **Grand River Energy Center** (OK) - 300 MW datacenter target
2. **Marshall Steam Station** (NC) - 650 MW datacenter target  
3. **Wolf Hollow II** (TX) - 100 MW datacenter target
4. **Lackawanna Energy Center** (PA) - 150 MW datacenter target
5. **Guernsey Power Station** (OH) - 150 MW datacenter target
6. **Jim Bridger Power Plant** (WY) - 600 MW datacenter target
7. **Belews Creek Steam Station** (NC) - 900 MW datacenter target (existing)
8. **Oleander Power Project LP** (FL) - 300 MW datacenter target (existing)

### Key Files (Modular Architecture)
- `main_analysis.py` - Main charting and analysis orchestration (recommended)
- `utils.py` - Utility functions (solar modeling, battery simulation, chart creation)
- `constants.yaml` - Configuration constants and parameters
- `power_plant_hourly_analysis.py` - Legacy comprehensive file (still functional but being phased out)
- `plant_*/outputs/` - Generated charts (7 charts per plant)

### Dependencies
- **pvlib-python**: Accurate solar irradiance calculations using astronomical models
- **pandas**: Data processing and time series analysis  
- **matplotlib**: Chart generation via PlotBuddy integration
- **pyyaml**: Constants and configuration loading

### Chart Types Generated
1. **Daily Peaks** - Annual generation patterns
2. **Daily Cycle** - 24-hour generation profile (7AM-7AM)
3. **Slack Capacity** - Available unused generation capacity
4. **Solar Potential** - Location-specific solar generation modeling
5. **Combined Capacity** - Plant slack + solar generation
6. **Battery Storage** - Integrated battery storage system analysis
7. **Backup Generators** - Complete backup system with natural gas peakers

### Analysis Methodology  
- **Dynamic Configuration**: All parameters calculated from actual plant data analysis
- **Conservative Targeting**: Uses 50% of median slack capacity for datacenter targets
- **Accurate Solar Modeling**: Uses pvlib-python with location coordinates for precise solar irradiance calculations
- **Battery Integration**: 4-hour storage capacity with realistic charge/discharge cycling
- **Backup Generation**: Natural gas peaker units sized at 20% of datacenter target

### Data Sources
- **Generation Data**: EIA Hourly Generation Data
- **Solar Calculations**: pvlib-python library with clear-sky GHI modeling using plant coordinates  
- **Plant Information**: Built-in database with verified locations and coordinates for all 8 plants

## Commands to Run

### Power Plant Analysis
```bash
# Run analysis for specific plant (uses dynamic configuration - no config files needed)
source venv/bin/activate && cd src/active_projects/power_plant_curtailment && python main_analysis.py plant_165_GRDA/

# Run analysis without displaying charts (batch mode)
source venv/bin/activate && cd src/active_projects/power_plant_curtailment && python main_analysis.py plant_165_GRDA/ --no-display

# Alternative: Use the original comprehensive file (still works but is being phased out)
source venv/bin/activate && cd src/active_projects/power_plant_curtailment && python power_plant_hourly_analysis.py plant_165_GRDA/ --no-display
```

### General Examples
```bash
# Run the example (with venv activated)
source venv/bin/activate && python3 src/templates/area_charts/ai_power_demand_growth.py

# Install/check requirements
source venv/bin/activate && pip install -r requirements.txt

# Lint/typecheck (if available)
# Note: Add specific linting commands when available
```