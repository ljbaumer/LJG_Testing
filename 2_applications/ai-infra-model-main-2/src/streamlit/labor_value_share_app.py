"""Streamlit entry point for the labor value share model."""

from __future__ import annotations

import streamlit as st
from src.models.LaborValueShareModel import LaborValueShareModel
from src.streamlit.labor_value_share import SidebarConfig, build_sidebar, render_dashboard
from src.utils.streamlit_app_helpers import clear_session_state_on_first_load

TAKEAWAYS_MD = """
- Efficiency pool scales directly with the knowledge work base and AI uplift assumptions.
- Value capture split shows who benefits when the efficiency value materializes.
- Adjust the text inputs to test alternative macro or margin scenarios in seconds.
"""

st.set_page_config(
    page_title="Labor Value Share Model",
    layout="wide",
    initial_sidebar_state="expanded",
)


def main() -> None:
    """Render the labor value share dashboard."""

    clear_session_state_on_first_load(
        "labor_value_share_initialized",
        ("share_pct", "global_gdp_trillions", "knowledge_workers_millions"),
    )

    st.title("AI Labor Value Share Dashboard")

    stored_config = st.session_state.get("labor_value_share_sidebar_config")
    existing_config = stored_config if isinstance(stored_config, SidebarConfig) else None
    config = build_sidebar(existing_config)
    st.session_state["labor_value_share_sidebar_config"] = config

    try:
        model = LaborValueShareModel(
            global_gdp_usd=config.global_gdp_usd,
            total_knowledge_workers=config.total_knowledge_workers,
            labor_shares=config.labor_shares,
            value_capture=config.value_capture,
        )
    except ValueError as exc:
        st.error(str(exc))
        st.stop()

    results = model.run()
    render_dashboard(
        global_gdp=config.global_gdp_usd,
        total_knowledge_workers=config.total_knowledge_workers,
        traditional_software_spend_usd=config.traditional_software_spend_usd,
        results=results,
        value_capture=config.value_capture,
        efficiency_gain_pct=config.labor_shares.ai_efficiency_gain_pct,
        takeaways_md=TAKEAWAYS_MD,
    )


if __name__ == "__main__":  # pragma: no cover - Streamlit entrypoint
    main()
