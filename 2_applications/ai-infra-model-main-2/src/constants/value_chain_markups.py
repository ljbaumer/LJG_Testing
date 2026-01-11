"""Value Chain Markups - Pricing Power Configurations

This module defines markup presets as Pandas Series. Markups represent pricing power
at each layer of the value chain (GPU → Cloud → Model → App).

Architecture:
- Each markup preset is a pd.Series with layers as index
- Markups are multipliers (k = 1/(1-margin)) applied to cost base
- Clean separation from market segments and investment timelines
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# -----------------------------
# Markup Calculation Helpers
# -----------------------------

def markup_from_margin(margin: float) -> float:
    """Convert a validated margin m to markup k = 1/(1−m).

    Requires margin < 1.0 for finite markup.
    """
    m = float(margin)
    if m >= 1.0:
        raise ValueError("Margin must be < 1.0 for markup to be finite")
    if m <= -1e9:
        # Defensive guard; negative margins would imply markup > 1 in a non-sensical way here
        raise ValueError("Margin must be >= 0.0 in this model")
    return 1.0 / (1.0 - m)


def make_markups_from_margins(
    cloud_margin: float,
    model_margin: float,
    app_margin: float
) -> pd.Series:
    """Create markups Series from margin inputs using markup_from_margin helper."""
    # Direct Series construction - NO dict literals! 🔥
    return pd.Series(
        data=[
            markup_from_margin(cloud_margin),
            markup_from_margin(model_margin),
            markup_from_margin(app_margin),
        ],
        index=["cloud", "model", "app"]
    )

# -----------------------------
# Default Margin Configuration
# -----------------------------

DEFAULT_CLOUD_MARGIN = 0.30
DEFAULT_MODEL_MARGIN = 0.40
DEFAULT_APP_MARGIN = 0.40


# -----------------------------
# Markup Presets
# -----------------------------

# Standard pricing power across the stack - THE ONLY MARKUP CONFIGURATION WE USE
STANDARD_MARKUPS = make_markups_from_margins(
    cloud_margin=DEFAULT_CLOUD_MARGIN,
    model_margin=DEFAULT_MODEL_MARGIN,
    app_margin=DEFAULT_APP_MARGIN,
)


# -----------------------------
# Validation Helpers
# -----------------------------

def validate_markups_series(markups: pd.Series) -> None:
    """Validate that a markups Series has the required structure and values."""
    if not isinstance(markups, pd.Series):
        raise ValueError("markups must be a pandas Series")

    # Check required layers exist
    required_layers = ['cloud', 'model', 'app']
    missing_layers = [layer for layer in required_layers if layer not in markups.index]
    if missing_layers:
        raise ValueError(f"Missing required layers: {missing_layers}")

    # Check for extra layers (might indicate typos)
    extra_layers = [layer for layer in markups.index if layer not in required_layers]
    if extra_layers:
        raise ValueError(f"Unknown markup layers (typos?): {extra_layers}")

    # Check markup values are valid
    if not (markups >= 1.0).all():
        invalid_layers = markups[markups < 1.0].index.tolist()
        raise ValueError(f"All markups must be >= 1.0 (got negative margins in: {invalid_layers})")

    if not (markups <= 100.0).all():
        excessive_layers = markups[markups > 100.0].index.tolist()
        raise ValueError(f"Markups seem excessive (> 100x) in layers: {excessive_layers}")

    # Check for NaN or infinite values
    if markups.isnull().any():
        null_layers = markups[markups.isnull()].index.tolist()
        raise ValueError(f"Markup values cannot be null/NaN: {null_layers}")

    if not markups.apply(lambda x: pd.notna(x) and not np.isinf(x)).all():
        inf_layers = markups[markups.apply(lambda x: np.isinf(x))].index.tolist()
        raise ValueError(f"Markup values cannot be infinite: {inf_layers}")


# -----------------------------
# Module Validation (Self-Test)
# -----------------------------

if __name__ == "__main__":
    # Self-validate the standard markup preset
    print("Validating markup presets...")

    try:
        validate_markups_series(STANDARD_MARKUPS)
        print("✅ STANDARD markups - valid")
        print(f"   Cloud: {STANDARD_MARKUPS['cloud']:.2f}x, Model: {STANDARD_MARKUPS['model']:.2f}x, App: {STANDARD_MARKUPS['app']:.2f}x")
    except ValueError as e:
        print(f"❌ STANDARD markups - {e}")

    print("✅ All markup validations passed!")
