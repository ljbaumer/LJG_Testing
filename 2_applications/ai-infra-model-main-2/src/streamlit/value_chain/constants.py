"""Shared constants and helpers for the Value Chain Streamlit app."""

from typing import Dict, Optional, Sequence, TypedDict, cast

import pandas as pd
import plotly.graph_objects as go

VALUE_CHAIN_LAYER_COLORS: Dict[str, str] = {
    "infrastructure": "#1565c0",
    "cloud_margin": "#90EE90",
    "model_margin": "#32CD32",
    "app_margin": "#228B22",
    "neutral_text": "#1f2933",
}

VALUE_CHAIN_DEPRECIATION_COLORS: Dict[str, str] = {
    "chips": "#1565c0",
    "datacenter": "#2196f3",
    "power": "#64b5f6",
}

COHORT_COLORS: Dict[str, str] = {
    "Consumer Free": "#636EFA",      # Plotly blue
    "Consumer Paid": "#EF553B",      # Plotly red
    "SMB Paid": "#00CC96",           # Plotly green
    "Enterprise Paid": "#AB63FA",    # Plotly purple
    "Developer Paid": "#FFA15A",     # Plotly orange
    "Power User": "#19D3F3",         # Plotly light blue
}


def apply_value_chain_theme(
    fig: go.Figure,
    *,
    title: Optional[str] = None,
    height: Optional[int] = None,
    legend: Optional[Dict] = None,
    margin: Optional[Dict] = None,
    **layout_overrides,
) -> go.Figure:
    """Apply consistent layout configuration to Plotly figures used in the Value Chain app."""

    default_layout = {
    "title": {"text": title, "font": {"size": 20, "color": "#000000"}}
        if title
        else None,
    "template": "plotly_white",
    "font": {"family": "Inter, sans-serif", "size": 12, "color": "#000000"},
        "legend": legend or {"orientation": "h", "yanchor": "top", "y": -0.15, "xanchor": "center", "x": 0.5},
        "margin": margin or {"l": 60, "r": 40, "t": 80, "b": 100},
        "height": height,
    }

    cleaned_layout = {k: v for k, v in default_layout.items() if v is not None}
    cleaned_layout.update(layout_overrides)
    fig.update_layout(**cleaned_layout)
    return fig


SEGMENT_DISPLAY_NAMES = {
    "us_canada_europe_consumers": "US+Canada+Europe Consumers",
    "apac_row_consumers": "APAC+Rest-of-World Consumers",
    "professional": "Professional",
    "programmer": "Programmer",
}

SEGMENT_COMMENTS = {
    "us_canada_europe_consumers": {
        "title": "Facebook/Netflix as the comparison",
        "details": [
            "TAM sized based on Facebook user base in these regions, (these comments are for the bull case)",
            "Consumer Free ARPU in the bull case is based on Facebook's advertising revenue per user",
            "Netflix has ~300M global subscribers, so we assume 10% of TAM converts to paid",
            "Paid ARPU assumes ChatGPT Plus pricing (\\$20/month) is able to hold",
        ],
    },
    "apac_row_consumers": {
        "title": "Facebook/Netflix as the comparison",
        "details": [
            "Addressable Market & Free ARPU are both based on Facebook data for these regions from the last time they broke this out in Q4 2023",
            "OpenAI India pricing is \\$4.60 today)",
        ],
    },
    "professional": {
        "title": "Microsoft Office 365 Sized Business",
        "details": [
            "TAM is based on Office 365's 450M enterprise users.",
            "Base Case & Today: \\$15 SMB (Microsoft Office's current ARPU), \\$36 Enterprise (highest tier of Office Copilot pricing)",
            "Bull Case & Total Adoption: \\$15 SMB, \\$45 Enterprise (premium pricing)",
            "The ARPU toggles here are perhaps the highest impact on the model.",
            "Salesforce highest tier is \\$330/month so there is room to run here",
        ],
    },
    "programmer": {
        "title": "Every developer is paying for AI",
        "details": [
            "TAM source: see the JetBrains Developer Ecosystem survey.",
            "Claude Code from Pro and Max plans are the pricing benchmarks",
            "Developer tools market shows willingness to pay premium",
            "TAM can be expanded as vibe coding expands the definition of developer to include more people",
            "We should expect the cost of serving AI to be higher than traditional web services; numbers will likely need to climb to reflect that.",
        ],
    },
}


def get_segment_display_name(segment_name: str) -> str:
    """Get consistent display name for a segment."""

    return SEGMENT_DISPLAY_NAMES.get(segment_name, segment_name.replace("_", " ").title())


class ModelConfigurationDict(TypedDict):
    num_segments: int
    num_cohorts: int
    base_markups: Dict[str, float]
    adjusted_markups: Dict[str, float]
    num_toggles_applied: int


class UserEconomicsSummaryDict(TypedDict):
    monthly_user_revenue: float
    annual_user_revenue: float
    monthly_user_cost: float
    annual_user_cost: float
    annual_user_profit: float
    total_active_users: float


TIMELINE_REQUIRED_COLUMNS: Sequence[str] = (
    "year",
    "depreciation",
    "required_cloud_revenue_for_margins",
    "required_model_revenue_for_margins",
    "required_app_revenue_for_margins",
    "total_profit",
)

COHORT_REQUIRED_COLUMNS: Sequence[str] = (
    "segment",
    "cohort_name",
    "users",
    "cohort_share",
    "arpu",
    "total_addressable_users",
    "monthly_revenue",
    "monthly_cost",
    "annual_revenue",
    "annual_cost",
    "annual_profit",
)


def validate_dataframe_columns(df: pd.DataFrame, required_columns: Sequence[str], *, context: str) -> pd.DataFrame:
    """Ensure a DataFrame contains the required columns (returns reindexed copy when empty)."""

    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"{context} must be a pandas DataFrame")

    if df.empty:
        return df.reindex(columns=required_columns)

    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"{context} is missing required columns: {', '.join(missing_columns)}")
    return df


def validate_model_configuration(config: Dict) -> ModelConfigurationDict:
    expected_keys = {
        "num_segments",
        "num_cohorts",
        "base_markups",
        "adjusted_markups",
        "num_toggles_applied",
    }
    missing_keys = expected_keys.difference(config.keys())
    if missing_keys:
        raise ValueError(f"Model configuration missing keys: {', '.join(sorted(missing_keys))}")
    return cast(ModelConfigurationDict, config)


def validate_user_economics_summary(summary: pd.Series | Dict[str, float]) -> UserEconomicsSummaryDict:
    if isinstance(summary, pd.Series):
        summary_dict = summary.to_dict()
    else:
        summary_dict = summary

    expected_keys = {
        "monthly_user_revenue",
        "annual_user_revenue",
        "monthly_user_cost",
        "annual_user_cost",
        "annual_user_profit",
        "total_active_users",
    }
    missing_keys = expected_keys.difference(summary_dict.keys())
    if missing_keys:
        raise ValueError(f"User economics summary missing keys: {', '.join(sorted(missing_keys))}")

    return cast(UserEconomicsSummaryDict, summary_dict)


__all__ = [
    "VALUE_CHAIN_LAYER_COLORS",
    "VALUE_CHAIN_DEPRECIATION_COLORS",
    "apply_value_chain_theme",
    "SEGMENT_DISPLAY_NAMES",
    "SEGMENT_COMMENTS",
    "get_segment_display_name",
    "ModelConfigurationDict",
    "UserEconomicsSummaryDict",
    "TIMELINE_REQUIRED_COLUMNS",
    "COHORT_REQUIRED_COLUMNS",
    "validate_dataframe_columns",
    "validate_model_configuration",
    "validate_user_economics_summary",
]
