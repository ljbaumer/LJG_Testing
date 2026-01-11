"""Baselines for labor share percentages and value capture splits."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ValueCaptureBreakdown:
    consumer_surplus: float
    worker_surplus: float
    revenue_uplift: float
    cost_reduction: float
    software_capture: float

    def __post_init__(self) -> None:
        shares = (
            self.consumer_surplus,
            self.worker_surplus,
            self.revenue_uplift,
            self.cost_reduction,
            self.software_capture,
        )
        if any(not 0.0 <= share <= 1.0 for share in shares):
            raise ValueError("Value capture shares must be between 0 and 1")
        total = sum(shares)
        if not 0.99 <= total <= 1.01:
            raise ValueError(f"Value capture shares must sum to 1.0, got {total:.3f}")


@dataclass(frozen=True)
class LaborSharePercentages:
    labor_share_of_gdp: float
    knowledge_work_share: float
    ai_efficiency_gain_pct: float

    def __post_init__(self) -> None:
        for field_name, value in (
            ("labor_share_of_gdp", self.labor_share_of_gdp),
            ("knowledge_work_share", self.knowledge_work_share),
            ("ai_efficiency_gain_pct", self.ai_efficiency_gain_pct),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{field_name} must be between 0 and 1, received {value}")


BASELINE_GLOBAL_GDP_USD: float = 125_000_000_000_000.0
BASELINE_GLOBAL_KNOWLEDGE_WORKERS: float = 875_000_000.0
BASELINE_TRADITIONAL_SOFTWARE_SPEND_USD: float = 700_000_000_000.0
BASELINE_LABOR_SHARE_PERCENTAGES = LaborSharePercentages(
    labor_share_of_gdp=0.54,
    knowledge_work_share=0.40,
    ai_efficiency_gain_pct=0.50,
)
BASELINE_VALUE_CAPTURE = ValueCaptureBreakdown(
    consumer_surplus=0.20,
    worker_surplus=0.10,
    revenue_uplift=0.30,
    cost_reduction=0.20,
    software_capture=0.20,
)
