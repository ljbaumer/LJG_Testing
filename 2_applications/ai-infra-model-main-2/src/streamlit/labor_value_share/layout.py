"""Page layout helpers for the labor value share dashboard."""

from __future__ import annotations

import plotly.graph_objects as go

import streamlit as st
from src.constants.labor_share_global_assumptions import ValueCaptureBreakdown
from src.models.LaborValueShareModel import LaborValueShareModel
from src.utils.streamlit_app_helpers import format_number_to_string


def _build_knowledge_waterfall(
    *,
    global_gdp: float,
    results: LaborValueShareModel.Results,
) -> go.Figure:
    """Waterfall showing how we arrive at knowledge work compensation."""

    labor_comp = results.global_labor_compensation_usd
    knowledge_comp = results.knowledge_work_compensation_usd

    non_labor_gdp = max(global_gdp - labor_comp, 0.0)
    other_labor = max(labor_comp - knowledge_comp, 0.0)

    figure = go.Figure(
        go.Waterfall(
            name="Knowledge Work Build",
            orientation="v",
            measure=["absolute", "relative", "relative", "total"],
            x=[
                "Global GDP",
                "Capital Share of GDP (Non-Labor)",
                "Other Labor Compensation",
                "Knowledge Work Compensation",
            ],
            y=[
                global_gdp,
                -non_labor_gdp,
                -other_labor,
                knowledge_comp,
            ],
            textposition="outside",
            text=[
                format_number_to_string(global_gdp, is_currency=True),
                f"-{format_number_to_string(non_labor_gdp, is_currency=True)}",
                f"-{format_number_to_string(other_labor, is_currency=True)}",
                format_number_to_string(knowledge_comp, is_currency=True),
            ],
            connector={"line": {"color": "#94a3b8", "width": 1.2}},
            decreasing={"marker": {"color": "#9ca3af"}},
            increasing={"marker": {"color": "#60a5fa"}},
            totals={"marker": {"color": "#2563eb"}},
            hovertemplate="%{label}: %{y:$,.0f}<extra></extra>",
        )
    )

    # Calculate clean tick intervals for y-axis
    max_y_value = max(global_gdp, knowledge_comp)
    from src.utils.streamlit_app_helpers import calculate_chart_tick_intervals
    tick_values = calculate_chart_tick_intervals(max_y_value)
    tick_labels = [format_number_to_string(val, is_currency=True, escape_markdown=False) for val in tick_values]

    figure.update_layout(
        title="From Global GDP to Knowledge Work Compensation",
        showlegend=False,
        margin=dict(l=20, r=20, t=60, b=40),
        height=520,
        paper_bgcolor="#f8fafc",
        plot_bgcolor="#f8fafc",
        font=dict(size=13, color="#0f172a"),
        yaxis=dict(
            title="USD",
            tickvals=tick_values,
            ticktext=tick_labels,
            gridcolor="#e2e8f0",
            zerolinecolor="#94a3b8",
        ),
    )
    return figure


def _build_value_capture_pie(
    *,
    results: LaborValueShareModel.Results,
    shares: ValueCaptureBreakdown,
) -> go.Figure:
    """Create a pie chart illustrating the value capture split."""

    labels = [
        "Consumer Surplus",
        "Workers: Reduced Hours",
        "Revenue Uplift",
        "Cost Reduction",
        "Software Capture",
    ]
    values = [
        results.consumer_surplus_value_usd,
        results.worker_surplus_value_usd,
        results.revenue_uplift_value_usd,
        results.cost_reduction_value_usd,
        results.software_value_capture_usd,
    ]
    percentages = [
        shares.consumer_surplus,
        shares.worker_surplus,
        shares.revenue_uplift,
        shares.cost_reduction,
        shares.software_capture,
    ]

    hover_text = [
        f"{label}: {format_number_to_string(value, is_currency=True)} ({pct * 100:.1f}%)"
        for label, value, pct in zip(labels, values, percentages)
    ]

    text_labels = [
        f"{label}<br>{format_number_to_string(value, is_currency=True, escape_markdown=False)}<br>({pct * 100:.1f}%)"
        for label, value, pct in zip(labels, values, percentages)
    ]

    figure = go.Figure(
        go.Pie(
            labels=labels,
            values=values,
            hole=0.45,
            hoverinfo="text",
            hovertext=hover_text,
            textinfo="text",
            text=text_labels,
            marker=dict(line=dict(color="#FFFFFF", width=1)),
        )
    )
    figure.update_layout(
        title=None,
        margin=dict(l=40, r=40, t=40, b=40),
        font=dict(size=11),
    )
    return figure


def _render_efficiency_summary(
    *,
    results: LaborValueShareModel.Results,
    efficiency_gain_pct: float,
) -> None:
    knowledge_comp = results.knowledge_work_compensation_usd
    efficiency_value = results.efficiency_value_base_usd
    headline = (
        f"{format_number_to_string(efficiency_value, is_currency=True, escape_markdown=True)} of AI value "
        "based on "
        f"{format_number_to_string(knowledge_comp, is_currency=True, escape_markdown=True)} in knowledge work and "
        f"{efficiency_gain_pct * 100:.1f}% AI efficiency uplift"
    )
    st.subheader(headline)


def _render_value_capture_table(
    *,
    results: LaborValueShareModel.Results,
    shares: ValueCaptureBreakdown,
) -> None:
    rows = [
        (
            "Consumer Surplus",
            format_number_to_string(results.consumer_surplus_value_usd, is_currency=True),
            shares.consumer_surplus * 100,
            "Consumers enjoy more or better service without paying more (e.g., more detailed work from your lawyer for same fee).",
        ),
        (
            "Workers Reduced Hours",
            format_number_to_string(results.worker_surplus_value_usd, is_currency=True),
            shares.worker_surplus * 100,
            "Workers benefit from reduced hours and more comfortable work conditions. Hard to put a precise dollar figure on this benefit.",
        ),
        (
            "Revenue Uplift",
            format_number_to_string(results.revenue_uplift_value_usd, is_currency=True),
            shares.revenue_uplift * 100,
            "How much of the efficiency gain actually converts to new revenue through expanded sales or AI-enabled offerings.",
        ),
        (
            "Cost Reduction",
            format_number_to_string(results.cost_reduction_value_usd, is_currency=True),
            shares.cost_reduction * 100,
            "Direct cost savings from lowering headcount and operational automation.",
        ),
        (
            "Software Revenue Value Capture",
            format_number_to_string(results.software_value_capture_usd, is_currency=True),
            shares.software_capture * 100,
            "Portion retained by software and infrastructure vendors. Historically estimated at 5-20% of total software revenue, though difficult to determine precisely.",
        ),
    ]

    bullet_lines = [
        f"- **{name}**: {description}"
        for name, value, percentage, description in rows
    ]
    st.markdown("\n".join(bullet_lines))


def _render_summary_metrics(results: LaborValueShareModel.Results, efficiency_gain_pct: float) -> None:
    st.markdown("#### Efficiency Gain Snapshot")
    col1, col2, col3 = st.columns(3)
    col1.metric(
        "Knowledge Work Compensation",
        format_number_to_string(results.knowledge_work_compensation_usd, is_currency=True),
    )
    col2.metric(
        "AI Efficiency Gain",
        f"{efficiency_gain_pct * 100:.1f}%",
    )
    col3.metric(
        "Value of Efficiency Gain",
        format_number_to_string(results.efficiency_value_base_usd, is_currency=True),
    )


def _render_workforce_impacts(*, results: LaborValueShareModel.Results, traditional_software_spend_usd: float) -> None:
    """Display headcount and per-worker economics for the automation scenario."""

    st.markdown("### Workforce Impact Analysis")

    headcount_col1, headcount_col2, headcount_col3 = st.columns(3)
    headcount_col1.metric(
        "Baseline Knowledge Workers",
        f"{results.total_knowledge_workers_count:,.0f}",
    )
    headcount_col2.metric(
        "Jobs Displaced",
        f"{results.jobs_displaced_count:,.0f}",
    )
    headcount_col3.metric(
        "Workers Remaining",
        f"{results.remaining_workers_count:,.0f}",
    )

    traditional_monthly_per_worker = traditional_software_spend_usd / results.total_knowledge_workers_count / 12.0

    per_worker_col1, per_worker_col2, per_worker_col3 = st.columns(3)
    per_worker_col1.metric(
        "Average Knowledge Worker Salary",
        format_number_to_string(results.baseline_average_salary_usd, is_currency=True),
    )

    per_worker_col2.metric(
        "Existing Software Spend per Month",
        format_number_to_string(traditional_monthly_per_worker, is_currency=True),
    )

    per_worker_col3.metric(
        "Incremental AI Spend per Month",
        format_number_to_string(
            results.software_spend_per_worker_month_usd, is_currency=True
        ),
    )



def _render_per_worker_cost_mix(
    *,
    results: LaborValueShareModel.Results,
    traditional_software_spend_usd: float,
) -> None:
    """Show how monthly per-worker costs split across wages, legacy SaaS, and AI seats."""

    st.markdown("### Per-Worker Cost Mix")

    remaining_workers = max(results.remaining_workers_count, 1.0)
    wage_monthly = results.baseline_average_salary_usd / 12.0
    ai_monthly = results.software_spend_per_worker_month_usd
    traditional_monthly = (
        traditional_software_spend_usd / remaining_workers / 12.0
        if remaining_workers > 0.0
        else 0.0
    )

    slices = [
        ("Wages", wage_monthly),
        ("AI Spend", ai_monthly),
        ("SaaS Spend", traditional_monthly),
    ]

    figure = go.Figure(
        go.Pie(
            labels=[name for name, _ in slices],
            values=[value for _, value in slices],
            hole=0.45,
            hovertemplate="%{label}: $%{value:,.0f}/mo per worker<extra></extra>",
            textinfo="label",
        )
    )
    figure.update_layout(
        margin=dict(l=20, r=20, t=20, b=20),
        height=360,
        showlegend=False,
        font=dict(size=10),
    )

    st.plotly_chart(figure, use_container_width=True)


def render_dashboard(
    *,
    global_gdp: float,
    total_knowledge_workers: float,
    traditional_software_spend_usd: float,
    results: LaborValueShareModel.Results,
    value_capture: ValueCaptureBreakdown,
    efficiency_gain_pct: float | None = None,
    takeaways_md: str | None = None,
) -> None:
    """Render the full dashboard layout."""

    st.markdown(
        "#### Assumptions\n"
        "- [IMF World GDP](https://www.imf.org/external/datamapper/NGDPD@WEO/OEMDC/ADVEC/WEOWORLD) for global GDP baseline.\n"
        "- [Our World in Data](https://ourworldindata.org/grapher/labor-share-of-gdp?tab=discrete-bar&time=latest) labor share of GDP (54%).\n"
        "- [WEF Future of Jobs 2023](https://www.weforum.org/publications/the-future-of-jobs-report-2023/digest/) knowledge work share (40%).\n"
        "- [Berg, *Automation hits the knowledge worker*](https://sdgs.un.org/sites/default/files/2023-05/B59%20-%20Berg%20-%20Automation%20hits%20the%20knowledge%20worker%20ChatGPT%20and%20the%20future%20of%20work.pdf?utm_source=chatgpt.com) estimating ~875M knowledge workers out of 3.5B total workforce and motivating future AI spend-per-worker analysis (editable in sidebar, current input:"
        f" {total_knowledge_workers:,.0f} workers)."
    )

    gain_pct = efficiency_gain_pct if efficiency_gain_pct is not None else 0.0
    _render_summary_metrics(results, gain_pct)

    waterfall_chart = _build_knowledge_waterfall(global_gdp=global_gdp, results=results)
    pie_chart = _build_value_capture_pie(results=results, shares=value_capture)

    st.plotly_chart(waterfall_chart, use_container_width=True)

    _render_efficiency_summary(results=results, efficiency_gain_pct=gain_pct)

    _render_value_capture_table(results=results, shares=value_capture)
    pie_chart.update_layout(title="Value Accrual Breakdown")
    st.plotly_chart(pie_chart, use_container_width=True)

    _render_workforce_impacts(results=results, traditional_software_spend_usd=traditional_software_spend_usd)
    _render_per_worker_cost_mix(
        results=results,
        traditional_software_spend_usd=traditional_software_spend_usd,
    )

    if takeaways_md and takeaways_md.strip():
        st.markdown("### Takeaways")
        st.markdown(takeaways_md)
