"""⚠️ TOGGLES NOT CURRENTLY IMPLEMENTED

This module contains fully implemented toggle infrastructure, but toggles are NOT used in the UI.
See src/streamlit/value_chain_app.py:93 where toggles=None is hardcoded.

The ValueChainModel has complete infrastructure to apply toggles, but no code path creates
or passes toggle instances. All specific toggle definitions below are commented out.

To implement toggles:
1. Uncomment desired toggle instances below
2. Create UI controls in value_chain_app.py
3. Pass toggle list to ValueChainModel constructor instead of None
"""

from dataclasses import dataclass, replace
from typing import List, Literal

# -----------------------------
# Dataclass Architecture (Revenue-Only Toggles)
# -----------------------------


@dataclass
class ValueChainToggle:
    """Market force toggle that applies multiplier to value chain markup.

    Adjusts pricing power at specific layers (cloud/model/app) based on market conditions.
    Uses linear interpolation between effect_min and effect_max based on input value.

    Example: NVIDIA market share toggle affects cloud markup
    - High NVIDIA share → lower cloud markup (vendor capture)
    - Low NVIDIA share → higher cloud markup (cloud keeps more margin)
    """

    label: str
    description: str
    value: float  # Current market condition value
    input_min: float  # Minimum possible input value
    input_max: float  # Maximum possible input value
    effect_min: float  # Markup multiplier at input_min
    effect_max: float  # Markup multiplier at input_max
    markup_layer: Literal["cloud", "model", "app"]  # Which markup to modify
    polarity: Literal["positive", "negative"] = "positive"  # Direction of effect

    def __post_init__(self):
        if self.polarity not in ("positive", "negative"):
            raise ValueError("polarity must be 'positive' or 'negative'")
        if self.input_max <= self.input_min:
            raise ValueError(f"input_max ({self.input_max}) must be > input_min ({self.input_min})")
        if not (self.input_min <= self.value <= self.input_max):
            raise ValueError(f"value ({self.value}) must be between input_min ({self.input_min}) and input_max ({self.input_max})")

    @staticmethod
    def _linear_interpolation(a: float, b: float, t: float) -> float:
        return a + (b - a) * t

    @staticmethod
    def _normalize(value: float, lo: float, hi: float) -> float:
        if hi == lo:
            return 0.0
        t = (value - lo) / (hi - lo)
        if t < 0.0:
            return 0.0
        if t > 1.0:
            return 1.0
        return t

    def get_multiplier(self) -> float:
        # 1) Normalize to [0,1]
        t = self._normalize(self.value, self.input_min, self.input_max)
        # 2) Curve (v1: linear)
        # 3) Polarity
        if self.polarity == "negative":
            t = 1.0 - t
        # 4) Interpolate (v1: linear)
        return self._linear_interpolation(self.effect_min, self.effect_max, t)

    # Note: applying multipliers to BaseAssumptions is handled by the calculator layer,
    # not by this dataclass. This keeps constants pure and free of business logic.


# -----------------------------
# Canonical Toggle Definitions (defaults; linear & bounded)
# -----------------------------
# ⚠️ ALL TOGGLES COMMENTED OUT - Not used in UI (see module docstring)

# NEOCLOUD_MARKET_SHARE_TOGGLE = ValueChainToggle(
#     label="NeoCloud Market Share",
#     description="Share captured by specialized/sovereign clouds (higher share → lower cloud markup).",
#     markup_layer="cloud",
#     polarity="negative",
#     value=15.0,  # percent - current specialized cloud share is still low
#     input_min=0.0,
#     input_max=100.0,
#     effect_min=1.0,  # baseline cloud markup
#     effect_max=0.75,  # fragmented market → compressed cloud markup
# )

# NVIDIA_SHARE_TOGGLE = ValueChainToggle(
#     label="NVIDIA Market Share",
#     description="Silicon vendor capture reduces cloud surplus (higher NV share → lower cloud markup).",
#     markup_key="gpu",
#     target_path="cloud_margin",
#     value=80.0,  # percent
#     input_min=0.0,
#     input_max=100.0,
#     effect_min=1.05,  # low NV share → slight lift
#     effect_max=0.85,  # high NV share → compression (bounded)
#     polarity="negative",
# )

# OPEN_MODEL_PARITY_TOGGLE = ValueChainToggle(
#     label="Open Model Market Share",
#     description="Open model market share (higher share → lower model markup due to commoditization).",
#     markup_layer="model",
#     polarity="negative",
#     value=20.0,  # percent - Llama, Mistral, etc gaining traction
#     input_min=0.0,
#     input_max=100.0,
#     effect_min=1.0,  # proprietary models → baseline model markup
#     effect_max=0.7,  # open models dominant → compressed model markup
# )

# SOFTWARE_COMMODITIZATION_TOGGLE = ValueChainToggle(
#     label="Software Commoditization (Build vs Buy)",
#     description="Percentage of software built on-premises vs purchased. Higher on-prem development → lower app layer markup.",
#     markup_layer="app",
#     polarity="negative",
#     value=25.0,  # percent - early but accelerating with GitHub Copilot, ChatGPT coding
#     input_min=0.0,
#     input_max=100.0,
#     effect_min=1.05,  # low automation → premium app markup
#     effect_max=0.75,  # high automation → compressed app markup
# )

# Note: Service cost overlays removed in gross margin mode

# Convenience collections
# ⚠️ ALL_TOGGLES_LIST commented out - toggles not used in UI
# ALL_TOGGLES_LIST: List[ValueChainToggle] = [
#     NEOCLOUD_MARKET_SHARE_TOGGLE,
#     # NVIDIA_SHARE_TOGGLE,  # Commented out - GPU layer removed
#     OPEN_MODEL_PARITY_TOGGLE,
#     SOFTWARE_COMMODITIZATION_TOGGLE,
# ]


__all__ = [
    "ValueChainToggle",  # Keep type definition exported
    # Specific toggle instances commented out - not used in UI
    # "NEOCLOUD_MARKET_SHARE_TOGGLE",
    # "OPEN_MODEL_PARITY_TOGGLE",
    # "SOFTWARE_COMMODITIZATION_TOGGLE",
    # "ALL_TOGGLES_LIST",
]


def clone_toggle(t: ValueChainToggle, **overrides) -> ValueChainToggle:
    """Shallow clone a toggle with overrides (e.g., value=0.4)."""
    return replace(t, **overrides)


# -----------------------------
# Validation Helpers
# -----------------------------

def validate_toggle(toggle: ValueChainToggle) -> None:
    """Validate that a toggle has the required structure and values."""
    if not isinstance(toggle, ValueChainToggle):
        raise ValueError("toggle must be a ValueChainToggle instance")

    # Validate toggle has required attributes
    required_attrs = ['label', 'value', 'input_min', 'input_max', 'markup_layer']
    for attr in required_attrs:
        if not hasattr(toggle, attr):
            raise ValueError(f"Toggle missing required attribute: {attr}")

    # Validate markup layer is valid (3-layer model: cloud/model/app)
    if toggle.markup_layer not in ['cloud', 'model', 'app']:
        raise ValueError(f"Toggle ({toggle.label}) has invalid markup_layer: {toggle.markup_layer}")

    # For effect ranges, allow inversion for negative polarity toggles
    if toggle.polarity == "positive" and toggle.effect_min >= toggle.effect_max:
        raise ValueError(f"Toggle ({toggle.label}) with positive polarity: effect_min must be < effect_max")
    elif toggle.polarity == "negative" and toggle.effect_min <= toggle.effect_max:
        raise ValueError(f"Toggle ({toggle.label}) with negative polarity: effect_min must be > effect_max (inverted range)")


def validate_toggles_list(toggles: list[ValueChainToggle]) -> None:
    """Validate a list of toggles."""
    if not isinstance(toggles, list):
        raise ValueError("toggles must be a list")

    for i, toggle in enumerate(toggles):
        try:
            validate_toggle(toggle)
        except ValueError as e:
            raise ValueError(f"Toggle {i} validation failed: {e}") from e


# -----------------------------
# Module Validation (Self-Test)
# -----------------------------
# ⚠️ Self-test commented out - toggles not used in UI

# if __name__ == "__main__":
#     # Self-validate all defined toggles
#     print("Validating toggles...")
#
#     for toggle in ALL_TOGGLES_LIST:
#         try:
#             validate_toggle(toggle)
#             print(f"✅ {toggle.label} - valid")
#         except ValueError as e:
#             print(f"❌ {toggle.label} - {e}")
#
#     # Validate the full list
#     validate_toggles_list(ALL_TOGGLES_LIST)
#     print("✅ All toggle validations passed!")
