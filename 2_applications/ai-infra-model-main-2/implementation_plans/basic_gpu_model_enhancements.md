# Basic GPU Model Enhancements

## Overview
Simple, practical enhancements to GPUCloudModel.py for more realistic long-term GPU economics.

## Core Problem
Current model assumes flat revenue across all lease periods. In reality:
- Hardware ages and loses value
- Customers demand price cuts for older hardware
- Fully depreciated assets can still generate revenue (zombie economics)

## Proposed Enhancements

### 1. Revenue Decay Over Time
**What:** Revenue decreases as hardware ages
**Why:** Competitive pressure + performance gap vs newer GPUs
**Implementation:**
```python
# New parameter
annual_revenue_decay_rate: float = 0.15  # 15% per year

# In _log_lease_revenue():
for month in range(start_month, end_month):
    years_elapsed = month / 12
    decay_factor = (1 - annual_revenue_decay_rate) ** years_elapsed
    revenue = base_revenue * decay_factor
```

**Default:** 0.0 (current behavior, no decay)
**Typical range:** 10-20% annually

---

### 2. Zombie Period (Post-Depreciation Revenue)
**What:** After full depreciation, GPUs continue earning low revenue
**Why:** Incumbents with sunk costs can profitably run at marginal pricing
**Implementation:**
```python
# New parameters
enable_zombie_period: bool = True
zombie_years: float = 2.0
zombie_revenue_multiplier: float = 0.20  # 20% of original pricing

# Extends timeline beyond depreciation end
# Only OPEX, no depreciation or capex burden
```

**Key insight:** This is the incumbent's moat - pure OPEX operation vs new entrants with capex burden

---

### 3. Datacenter as Separate Asset
**What:** Track datacenter infrastructure separately from GPUs
**Why:** Datacenter lasts 20-30 years, GPUs depreciate in 3-5 years
**Implementation:**
```python
# New parameters
datacenter_capex_per_kw: float = 2500  # $/kW
datacenter_depreciation_years: int = 25

# Calculate datacenter cost from GPU power requirements
datacenter_capex = power_required_kw * datacenter_capex_per_kw

# Separate depreciation schedules:
# - GPU: total_deal_years (currently 3-5 years)
# - Datacenter: datacenter_depreciation_years (25 years)
```

**Benefits:**
- More accurate long-term economics
- Enables GPU refresh scenarios (swap GPUs, keep datacenter)
- Models datacenter as reusable infrastructure

---

## Implementation Priority

### Phase 1: Revenue Decay (Recommended Start)
- Simplest change
- Backward compatible (default = 0)
- Immediate business value
- Files: GPUCloudModel.py, gpu_cloud_scenarios_dataclass.py

### Phase 2: Zombie Period
- Extends timeline logic
- Requires depreciation completion detection
- Files: GPUCloudModel.py, _process_cash_flows()

### Phase 3: Datacenter Separation
- Most complex (dual asset tracking)
- Requires depreciation schedule refactor
- Enables future GPU refresh scenarios
- Files: GPUCloudModel.py, gpu_cloud_helpers.py

---

## Example Scenario

**H100 Datacenter - 5 Year View**

**Without enhancements:**
- Year 1-3: $2.50/chip-hour (flat)
- Total revenue: Same every year

**With enhancements:**
- Year 1: $2.50/chip-hour
- Year 2: $2.13/chip-hour (15% decay)
- Year 3: $1.81/chip-hour (15% decay)
- Year 4-5: $0.50/chip-hour (zombie @ 20% of original)

**Result:** More realistic revenue curve, zombie period shows incumbent advantage

---

## Questions to Resolve

1. **Default decay rate:** 10%, 15%, or 20% annually?
2. **Decay mode:** Linear vs exponential?
3. **Zombie auto-activation:** Should zombie period automatically trigger after depreciation ends?
4. **Backward compatibility:** Keep existing two-lease structure or migrate fully to decay model?

---

## Files to Modify

1. `src/models/GPUCloudModel.py` - Core logic
2. `src/constants/gpu_cloud_scenarios_dataclass.py` - Add new parameters
3. `src/streamlit/gpu_cloud_model_app.py` - UI controls (optional)
4. `src/utils/gpu_cloud_helpers.py` - Helper functions (if needed)

---

## Success Metrics

- Model shows realistic revenue curves for aging hardware
- Zombie economics demonstrate incumbent cost advantage
- Datacenter/GPU split enables refresh scenario analysis
- All changes backward compatible (existing scenarios still work)