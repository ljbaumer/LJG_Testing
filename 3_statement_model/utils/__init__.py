"""
Utility functions and classes for the 3-statement financial model.
"""

from .excel_converter import ExcelConverter
from .financial_utils import (
    calculate_growth_rate,
    calculate_margin,
    calculate_percentage_of_revenue,
    calculate_working_capital,
    calculate_change_in_working_capital,
    calculate_free_cash_flow
)
from .visualization import (
    create_line_chart,
    create_bar_chart,
    create_stacked_bar_chart,
    create_area_chart,
    create_pie_chart,
    create_historical_vs_forecast_chart,
    create_scenario_comparison_chart,
    create_statement_structure_chart,
    create_financial_metrics_chart
)

__all__ = [
    "ExcelConverter",
    "calculate_growth_rate",
    "calculate_margin",
    "calculate_percentage_of_revenue",
    "calculate_working_capital",
    "calculate_change_in_working_capital",
    "calculate_free_cash_flow",
    "create_line_chart",
    "create_bar_chart",
    "create_stacked_bar_chart",
    "create_area_chart",
    "create_pie_chart",
    "create_historical_vs_forecast_chart",
    "create_scenario_comparison_chart",
    "create_statement_structure_chart",
    "create_financial_metrics_chart"
]
