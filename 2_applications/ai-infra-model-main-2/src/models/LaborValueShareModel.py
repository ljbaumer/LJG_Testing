"""Labor value share economic impact model (simplified)."""

from __future__ import annotations

from dataclasses import dataclass

from src.constants.labor_share_global_assumptions import (
    BASELINE_GLOBAL_GDP_USD,
    BASELINE_GLOBAL_KNOWLEDGE_WORKERS,
    BASELINE_LABOR_SHARE_PERCENTAGES,
    BASELINE_VALUE_CAPTURE,
    LaborSharePercentages,
    ValueCaptureBreakdown,
)


class LaborValueShareModel:
    """Compute macro outcomes for AI-driven labor efficiency gains."""

    @dataclass
    class Results:
        global_labor_compensation_usd: float
        knowledge_work_compensation_usd: float
        efficiency_value_base_usd: float
        total_value_generated_usd: float
        consumer_surplus_value_usd: float
        worker_surplus_value_usd: float
        revenue_uplift_value_usd: float
        cost_reduction_value_usd: float
        software_value_capture_usd: float
        total_knowledge_workers_count: float
        baseline_average_salary_usd: float
        jobs_displaced_count: float
        remaining_workers_count: float
        post_job_loss_compensation_usd: float
        income_gap_usd: float
        software_spend_per_worker_month_usd: float
        software_spend_share_of_salary_pct: float

    def __init__(
        self,
        global_gdp_usd: float,
        total_knowledge_workers: float,
        labor_shares: LaborSharePercentages,
        value_capture: ValueCaptureBreakdown,
    ) -> None:
        if global_gdp_usd <= 0.0:
            raise ValueError("global_gdp_usd must be positive")
        if total_knowledge_workers <= 0.0:
            raise ValueError("total_knowledge_workers must be positive")

        self.global_gdp_usd = global_gdp_usd
        self.total_knowledge_workers = total_knowledge_workers
        self.labor_shares = labor_shares
        self.value_capture = value_capture
        self.results = self._compute_results()

    @staticmethod
    def _compute_labor_income(
        global_gdp_usd: float,
        labor_share_of_gdp: float,
        knowledge_work_share: float,
    ) -> tuple[float, float]:
        total_labor_income = global_gdp_usd * labor_share_of_gdp
        knowledge_labor_income = total_labor_income * knowledge_work_share
        return total_labor_income, knowledge_labor_income

    @staticmethod
    def _distribute_value(
        total_value_usd: float,
        breakdown: ValueCaptureBreakdown,
    ) -> tuple[float, float, float, float, float]:
        return (
            total_value_usd * breakdown.consumer_surplus,
            total_value_usd * breakdown.worker_surplus,
            total_value_usd * breakdown.revenue_uplift,
            total_value_usd * breakdown.cost_reduction,
            total_value_usd * breakdown.software_capture,
        )

    def _compute_results(self) -> Results:
        global_labor_income, knowledge_labor_income = self._compute_labor_income(
            self.global_gdp_usd,
            self.labor_shares.labor_share_of_gdp,
            self.labor_shares.knowledge_work_share,
        )

        efficiency_value = knowledge_labor_income * self.labor_shares.ai_efficiency_gain_pct

        (
            consumer_surplus,
            worker_surplus,
            revenue_uplift,
            cost_reduction,
            software_capture,
        ) = self._distribute_value(efficiency_value, self.value_capture)

        average_salary = knowledge_labor_income / self.total_knowledge_workers
        # Constrain displaced headcount to the available workforce and wage bill.
        jobs_displaced = (
            cost_reduction / average_salary if average_salary > 0.0 else 0.0
        )
        jobs_displaced = min(jobs_displaced, self.total_knowledge_workers)
        remaining_workers = self.total_knowledge_workers - jobs_displaced
        post_job_loss_compensation = average_salary * remaining_workers
        income_gap = knowledge_labor_income - post_job_loss_compensation

        software_spend_per_worker_month = (
            software_capture / remaining_workers / 12.0
            if remaining_workers > 0.0
            else 0.0
        )
        monthly_salary = average_salary / 12.0
        software_spend_share_of_salary = (
            software_spend_per_worker_month / monthly_salary
            if monthly_salary > 0.0
            else 0.0
        )

        return self.Results(
            global_labor_compensation_usd=global_labor_income,
            knowledge_work_compensation_usd=knowledge_labor_income,
            efficiency_value_base_usd=efficiency_value,
            total_value_generated_usd=efficiency_value,
            consumer_surplus_value_usd=consumer_surplus,
            worker_surplus_value_usd=worker_surplus,
            revenue_uplift_value_usd=revenue_uplift,
            cost_reduction_value_usd=cost_reduction,
            software_value_capture_usd=software_capture,
            total_knowledge_workers_count=self.total_knowledge_workers,
            baseline_average_salary_usd=average_salary,
            jobs_displaced_count=jobs_displaced,
            remaining_workers_count=remaining_workers,
            post_job_loss_compensation_usd=post_job_loss_compensation,
            income_gap_usd=income_gap,
            software_spend_per_worker_month_usd=software_spend_per_worker_month,
            software_spend_share_of_salary_pct=software_spend_share_of_salary,
        )

    def run(self) -> Results:
        return self.results

    def pretty_print(self) -> None:
        """Print results as a formatted table."""
        print("\n" + "="*60)
        print("Labor Value Share Economic Model Results")
        print("="*60)
        print(f"{'Metric':<35} {'Value (USD)':<20}")
        print("-"*60)

        results = self.results
        metrics = [
            ("Global GDP", self.global_gdp_usd, "currency"),
            ("Global Labor Compensation", results.global_labor_compensation_usd, "currency"),
            ("Knowledge Work Compensation", results.knowledge_work_compensation_usd, "currency"),
            ("Efficiency Value Base", results.efficiency_value_base_usd, "currency"),
            ("Total Value Generated", results.total_value_generated_usd, "currency"),
            ("Consumer Surplus Value", results.consumer_surplus_value_usd, "currency"),
            ("Worker Surplus Value", results.worker_surplus_value_usd, "currency"),
            ("Revenue Uplift Value", results.revenue_uplift_value_usd, "currency"),
            ("Cost Reduction Value", results.cost_reduction_value_usd, "currency"),
            ("Software Value Capture", results.software_value_capture_usd, "currency"),
            ("Total Knowledge Workers", results.total_knowledge_workers_count, "count"),
            ("Baseline Average Salary", results.baseline_average_salary_usd, "currency"),
            ("Jobs Displaced", results.jobs_displaced_count, "count"),
            ("Remaining Workers", results.remaining_workers_count, "count"),
            (
                "Post-Job-Loss Compensation",
                results.post_job_loss_compensation_usd,
                "currency",
            ),
            ("Income Gap vs Baseline", results.income_gap_usd, "currency"),
            (
                "Software Spend per Worker (Monthly)",
                results.software_spend_per_worker_month_usd,
                "currency",
            ),
            (
                "Software Spend Share of Salary",
                results.software_spend_share_of_salary_pct,
                "percent",
            ),
        ]

        for name, value, value_type in metrics:
            if value_type == "currency":
                if value >= 1e12:
                    formatted_value = f"${value/1e12:.1f}T"
                elif value >= 1e9:
                    formatted_value = f"${value/1e9:.1f}B"
                elif value >= 1e6:
                    formatted_value = f"${value/1e6:.1f}M"
                else:
                    formatted_value = f"${value:,.0f}"
            elif value_type == "count":
                formatted_value = f"{value:,.0f}"
            elif value_type == "percent":
                formatted_value = f"{value * 100:.1f}%"
            else:
                formatted_value = f"{value:,.2f}"
            print(f"{name:<35} {formatted_value:<20}")

        print("="*60)



def _main() -> None:
    model = LaborValueShareModel(
        BASELINE_GLOBAL_GDP_USD,
        BASELINE_GLOBAL_KNOWLEDGE_WORKERS,
        BASELINE_LABOR_SHARE_PERCENTAGES,
        BASELINE_VALUE_CAPTURE,
    )
    model.pretty_print()


if __name__ == "__main__":  # pragma: no cover
    _main()
