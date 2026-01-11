# AI Infrastructure Model - System Prompt

## 🏗️ Architecture Rules

### Data Flow (MUST FOLLOW)
```
src/constants/          →  src/models/           →  src/streamlit/
(dataclasses, configs)     (business logic)         (UI only)
                          ↑
                    src/utils/
                  (shared helpers)
```

### File Placement Rules
1. **Helper functions** → `src/utils/` (NEVER in streamlit apps)
2. **Business logic** → `src/models/` (NEVER in streamlit apps)
3. **UI code only** → `src/streamlit/` (import helpers, don't create them)
4. **Configuration/Constants** → `src/constants/`

## 🔍 Search Before You Code

**Before implementing ANY new functionality, search the codebase:**

```bash
# Search for existing implementations with a bunch of different keywords to try and catch the right block of code
grep -r "def [function_name]" src/
grep -r "[keyword]" src/utils/
grep -r "[functionality]" src/
```

**Check what's already imported in similar files** - if other streamlit apps import something, you probably should too.

## 📁 Codebase Structure & Purpose

### `src/utils/` - Shared Utilities
- **`streamlit_app_helpers.py`** - Number formatting, UI components, table styling, and common Streamlit patterns used across apps
- **`capex_helpers.py`** - Infrastructure calculation functions that work from any starting point (budget, chips, power, etc.) with consistent logic
- **`gpu_cloud_helpers.py`** - Financial calculations specific to GPU cloud rental economics
- **`inference_demand_helpers.py`** - Demand modeling and capacity planning utilities
- **`nat_gas_helpers.py`** - Energy cost conversions and natural gas calculations
- **`revenue_to_capex.py`** - Conversion utilities from revenue schedules to CapEx requirements, including chip calculations, shortfall analysis, and FCF scheduling

### `src/models/` - Business Logic
- **`ValueChainModel.py`** - Core value chain analysis with DataFrame-based architecture for market segments and depreciation
- **`GPUCloudModel.py`** - GPU rental economics with IRR, NPV, and cash flow analysis
- **`InferenceDemandModel.py`** - AI inference capacity planning and demand forecasting
- **`DatacenterRetrofitModel.py`** - End-of-life datacenter transition cost modeling
- **`CompanyFundingModel.py`** - Company-specific AI infrastructure funding analysis, calculates how much non-AI business must subsidize AI buildout while maintaining credit quality
- **`AILabLiabilityModel.py`** - AI laboratory cloud contract liability analysis, tracks multiple contracts and compares payment schedules against revenue growth projections
- **`LaborValueShareModel.py`** - Macro-economic labor value share analysis, computes outcomes for AI-driven labor efficiency gains and value distribution

### `src/constants/` - Configuration & Data
- **`gpu_dataclass.py`** - GPU hardware specifications and performance data
- **`value_chain_depreciation_schedules.py`** - Infrastructure investment timelines and depreciation logic
- **`value_chain_market_segments.py`** - Customer segment definitions and cohort data
- **`*_scenarios_dataclass.py`** - Predefined scenarios for each model type
- **`cloud_contracts.py`** - Cloud contract definitions and payment schedule utilities for multi-provider contract analysis
- **`company_financing_profiles.py`** - Company-specific financing profiles including Oracle profile and GPU contract configurations
- **`ai_lab_profiles.py`** - AI laboratory profiles (e.g., OpenAI) with revenue base years and contract lists
- **`labor_share_global_assumptions.py`** - Global economic assumptions for labor share analysis including GDP, knowledge workers, and value capture distributions

### `src/streamlit/` - UI Applications
- **`buildout_calc_app.py`** - Datacenter construction cost calculator with multi-generation analysis
- **`gpu_cloud_model_app.py`** - GPU cloud rental analysis interface
- **`value_chain_app.py`** - Value chain analysis with pricing power modeling
- **`inference_demand_app.py`** - AI inference demand forecasting interface
- **`datacenter_retrofit_app.py`** - Datacenter retrofit cost analysis interface
- **`company_funding_app.py`** - Company funding model interface for Oracle AI infrastructure, analyzes CapEx requirements, funding waterfalls, and debt accumulation with depreciation visualization
- **`ai_lab_liability_app.py`** - AI lab liability model interface, visualizes cloud contract payments by provider, revenue coverage, and external funding requirements for AI laboratories like OpenAI
- **`labor_value_share_app.py`** - Labor value share dashboard, visualizes AI efficiency gains, value capture distribution, and economic impact on knowledge workers

## 🎨 Code Style & Standards

### Code Style Preferences
- **Use descriptive, verbose variable names** for clarity (e.g., `markup_at_cloud_layer`, `revenue_at_application_layer`, `profit_at_model_layer`)
- **Avoid cryptic abbreviations** (e.g., `kc`, `r_app`, `p_mod`) unless they are widely understood in immediate scope
- **Centralize business math** in small, stateless helpers; keep orchestrator methods short and explicit
- **Fail loudly on invalid inputs**; do not silently clamp unless explicitly requested

### Common Constants & Defaults
**Before hardcoding values, check if constants already exist:**
- **Power & Infrastructure**: `DEFAULT_PUE = 1.2`, `DEFAULT_UTILIZATION = 0.8`, `DEFAULT_POWER_COST_PER_KW = 2500`, `DEFAULT_DATACENTER_COST_PER_MW = 15_000_000`
- **Financial**: `NVIDIA_DEFAULT_GROSS_MARGIN = 0.75`, `NVIDIA_COMPUTE_SHARE = 0.7`
- **GPU Models**: Use `ALL_GPU_LIST` from `gpu_dataclass.py` rather than hardcoding specs

## 🧪 Testing & Quality

### Code Quality:
```bash
# Lint before git pushing anything
uvx ruff check .
```

## 🎯 Development Workflow

### For ANY Code Change:
1. **Search existing codebase** for similar functionality
2. **Check for existing constants** before hardcoding values
3. **Import from utils/** rather than reimplementing
4. **Follow the architecture flow** - constants → models → streamlit
5. **Test the specific modules you changed**
6. **Extend existing helpers** rather than creating duplicates

### Common Import Patterns:
```python
# Streamlit apps should import, not implement
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

# Import existing helpers
from src.utils.streamlit_app_helpers import format_number_to_string, create_styled_dataframe
from src.utils.capex_helpers import calculate_infrastructure, CapexStartingPoint
from src.constants.gpu_dataclass import ALL_GPU_LIST
```


## 🚀 Environment & Setup

### Dependencies: UV Package Manager
This project uses **UV** for dependency management.

**Key Commands:**
```bash
# First-time setup (installs dependencies and creates virtual environment)
uv sync

# Daily workflow after pulling changes
uv sync

# Launch all applications
uv run python launcher.py

# Run specific commands in the project environment
uv run streamlit run src/streamlit/gpu_cloud_model_app.py --server.port 8501
uv run pytest
uv run ruff check .

# Add/remove dependencies
uv add package-name
uv remove package-name
```

**That's it!** UV handles everything - virtual environment, dependencies, and execution automatically.

---

**Core Principle: This codebase values reuse over reimplementation. Search first for existing functions, helpers, and constants, then import.**