"""
Constants and configuration package - Updated for DataFrame Architecture.
"""

# Market segments (new MarketSegment approach)
# Labor share (labor value capture defaults)
from .labor_share_global_assumptions import (
    BASELINE_GLOBAL_GDP_USD,
    BASELINE_LABOR_SHARE_PERCENTAGES,
    BASELINE_VALUE_CAPTURE,
    LaborSharePercentages,
    ValueCaptureBreakdown,
)

# Depreciation schedules (moved from value_chain_types.py)
from .value_chain_depreciation_schedules import (
    CapexDepreciationSchedule,
)
from .value_chain_market_segments import (
    ALL_MARKET_SCENARIOS,
    BASE_CASE_MARKET_SEGMENTS,
    BULL_CASE_MARKET_SEGMENTS,
    DEFAULT_MARKET_SEGMENTS,
    TODAY_MARKET_SEGMENTS,
)

# Markups (extracted from old scenarios file)
from .value_chain_markups import (
    STANDARD_MARKUPS,
    make_markups_from_margins,
)

# Toggles - ⚠️ Only type definition imported, specific instances not used in UI
from .value_chain_toggles import (
    ValueChainToggle,  # Type definition used in ValueChainModel
    # Specific toggle instances commented out - not used in UI (see value_chain_app.py:93)
    # ALL_TOGGLES_LIST,
    # NEOCLOUD_MARKET_SHARE_TOGGLE,
    # OPEN_MODEL_PARITY_TOGGLE,
    # SOFTWARE_COMMODITIZATION_TOGGLE,
)
# from .value_chain_toggles import (
#     clone_toggle as clone_value_chain_toggle,
# )

__all__ = [
    # Market segments
    "DEFAULT_MARKET_SEGMENTS",

    # Markups
    "STANDARD_MARKUPS",
    "make_markups_from_margins",

    # Depreciation schedules
    "CapexDepreciationSchedule",

    # Toggles - only type definition exported
    "ValueChainToggle",
    # Specific toggle instances not exported - not used in UI
    # "NEOCLOUD_MARKET_SHARE_TOGGLE",
    # "OPEN_MODEL_PARITY_TOGGLE",
    # "SOFTWARE_COMMODITIZATION_TOGGLE",
    # "ALL_TOGGLES_LIST",
    # "clone_value_chain_toggle",

    # Labor share
    "LaborSharePercentages",
    "ValueCaptureBreakdown",
    "BASELINE_GLOBAL_GDP_USD",
    "BASELINE_LABOR_SHARE_PERCENTAGES",
    "BASELINE_VALUE_CAPTURE",
]
