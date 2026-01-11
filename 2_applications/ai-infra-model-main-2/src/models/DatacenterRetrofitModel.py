"""Datacenter Retrofit Cost Calculator"""

from src.constants.datacenter_retrofit_scenarios_dataclass import (
    RETROFIT_SCENARIOS,
    DatacenterRetrofitScenario,
)


class DatacenterRetrofitModel:
    """Calculator for datacenter retrofit costs based on system replacement percentages"""

    def __init__(self, scenario: DatacenterRetrofitScenario):
        """Initialize calculator with scenario and compute all costs"""
        self.scenario = scenario
        self.total_facility_cost = (
            scenario.facility_size_mw * scenario.facility_cost_per_mw
        )

        self.construction_costs = self._calculate_construction_costs()
        self.retrofit_costs = self._calculate_retrofit_costs()

        # Add total costs for convenience
        self.construction_costs["total"] = sum(
            sum(c.values()) for c in self.construction_costs.values()
        )
        self.retrofit_costs["total"] = sum(
            sum(c.values()) for c in self.retrofit_costs.values()
        )

    def _calculate_construction_costs(self) -> dict:
        """Calculate construction costs dynamically."""
        system_budgets = {
            "power": self.total_facility_cost
            * self.scenario.system_percentages.power_pct,
            "cooling": self.total_facility_cost
            * self.scenario.system_percentages.cooling_pct,
            "other": self.total_facility_cost
            * self.scenario.system_percentages.other_pct,
        }

        all_costs = {}
        component_groups = {
            "power": self.scenario.power_components,
            "cooling": self.scenario.cooling_components,
            "other": self.scenario.other_components,
        }

        for group_name, components_obj in component_groups.items():
            budget = system_budgets[group_name]
            group_costs = {
                name: budget * spec.construction.percentage
                for name, spec in components_obj.get_components().items()
            }
            all_costs[group_name] = group_costs

        return all_costs

    def _calculate_retrofit_costs(self) -> dict:
        """Calculate retrofit costs dynamically."""
        all_retrofit_costs = {}
        component_groups = {
            "power": self.scenario.power_components,
            "cooling": self.scenario.cooling_components,
            "other": self.scenario.other_components,
        }

        for group_name, components_obj in component_groups.items():
            group_retrofit_costs = {
                name: self.construction_costs[group_name][name]
                * spec.replacement.percentage
                for name, spec in components_obj.get_components().items()
            }
            all_retrofit_costs[group_name] = group_retrofit_costs

        return all_retrofit_costs

    def print_analysis(self) -> None:
        """Print detailed analysis of this calculator's scenario."""
        print(f"\n{'=' * 60}")
        print(f"{self.scenario.scenario_name}")
        print(f"{'=' * 60}")
        print(f"Facility Size: {self.scenario.facility_size_mw} MW")
        print(f"Cost per MW: ${self.scenario.facility_cost_per_mw:,.0f}")
        print(f"Total Facility Cost: ${self.total_facility_cost:,.0f}")

        print("\nSystem Breakdown (Original Construction - Schneider Electric Model):")

        system_percentages = {
            "power": self.scenario.system_percentages.power_pct,
            "cooling": self.scenario.system_percentages.cooling_pct,
            "other": self.scenario.system_percentages.other_pct,
        }

        component_groups = {
            "power": self.scenario.power_components,
            "cooling": self.scenario.cooling_components,
            "other": self.scenario.other_components,
        }

        for group_name, components in self.construction_costs.items():
            if group_name == "total":
                continue
            group_total = sum(components.values())
            title = group_name.replace("_", " ").title()
            print(
                f"  {title} Systems: ${group_total:,.0f} ({system_percentages[group_name]:.1%})"
            )
            for comp_name, cost in components.items():
                comp_title = comp_name.replace("_", " ").title()
                print(f"    - {comp_title}: ${cost:,.0f}")

        print("\nRetrofit Requirements:")

        for group_name, components in self.retrofit_costs.items():
            if group_name == "total":
                continue
            group_total = sum(components.values())
            title = group_name.replace("_", " ").title()
            print(f"  {title} Systems Retrofit: ${group_total:,.0f}")

            specs = component_groups[group_name].get_components()
            for comp_name, cost in components.items():
                comp_title = comp_name.replace("_", " ").title()
                replacement_pct = specs[comp_name].replacement.percentage
                print(f"    - {comp_title}: {replacement_pct:.0%} → ${cost:,.0f}")

        print("\nRetrofit Summary:")
        print(f"  Total Retrofit Cost: ${self.retrofit_costs['total']:,.0f}")
        print(
            f"  Retrofit as % of Facility: {self.retrofit_costs['total'] / self.total_facility_cost:.1%}"
        )
        print(
            f"  Cost per MW (retrofit): ${self.retrofit_costs['total'] / self.scenario.facility_size_mw:,.0f}"
        )


def run_retrofit_cost_analysis():
    """Demo runner for datacenter retrofit cost analysis"""
    print("DATACENTER RETROFIT COST ANALYSIS")
    print("=" * 60)

    # Store calculators to avoid re-computing for summary
    calculators = [DatacenterRetrofitModel(s) for s in RETROFIT_SCENARIOS]

    for calculator in calculators:
        calculator.print_analysis()

    print(f"\n{'=' * 60}")
    print("SUMMARY COMPARISON")
    print(f"{'=' * 60}")
    print(f"{'Scenario':<40} {'Retrofit Cost':<15} {'% of Facility':<12}")
    print("-" * 67)

    for calculator in calculators:
        scenario_name = calculator.scenario.scenario_name[:38]
        retrofit_cost = f"${calculator.retrofit_costs['total']:,.0f}"
        retrofit_pct = (
            f"{calculator.retrofit_costs['total'] / calculator.total_facility_cost:.1%}"
        )
        print(f"{scenario_name:<40} {retrofit_cost:<15} {retrofit_pct:<12}")


if __name__ == "__main__":
    run_retrofit_cost_analysis()
