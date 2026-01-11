"""Datacenter Retrofit Scenarios with Consolidated Component Architecture"""

from dataclasses import dataclass, fields
from typing import Dict, NamedTuple

# Component descriptions explaining what each component is and how it differs between traditional and AI infrastructure
COMPONENT_DESCRIPTIONS = {
    # Power Systems
    "ups": {
        "what_it_is": "Uninterruptible Power Supply - Battery backup systems that provide clean, continuous power during grid outages and power quality issues.",
        "traditional_vs_ai": "Traditional datacenters use standard AC UPS systems (±400V). AI datacenters increasingly use 800V DC UPS systems for higher efficiency at extreme power densities (40-200kW per rack vs 5-15kW traditional).",
        "why_it_matters": "UPS systems are the backbone of datacenter reliability. AI workloads cannot tolerate power interruptions during training runs that may take weeks."
    },
    "generator": {
        "what_it_is": "Backup diesel generators that provide long-term power during extended grid outages, typically sized for full facility load.",
        "traditional_vs_ai": "Both use similar generator technology, but AI datacenters often have smaller generator-to-load ratios due to improved grid reliability and massive battery backup systems.",
        "why_it_matters": "Generators provide extended runtime during grid failures. Less critical for AI due to cloud-scale redundancy and ability to pause/resume training workloads."
    },
    "switchgear": {
        "what_it_is": "High-voltage electrical distribution panels that route power from utility feeds to facility systems with protection and switching capabilities.",
        "traditional_vs_ai": "AI datacenters require significantly more switchgear capacity for concentrated power loads and may use specialized 800V DC distribution vs traditional AC systems.",
        "why_it_matters": "Switchgear handles the massive power distribution requirements. AI facilities need much higher power density distribution to individual racks."
    },
    "critical_distribution": {
        "what_it_is": "Final-stage power distribution including PDUs (Power Distribution Units), busways, and rack-level power connections.",
        "traditional_vs_ai": "Traditional uses standard AC PDUs. AI requires specialized high-current PDUs, 800V DC busways, and custom power connections for GPU servers drawing 40-200kW per rack.",
        "why_it_matters": "This is where power meets compute. AI's extreme power density requires completely different distribution architecture."
    },

    # Cooling Systems
    "in_building_cooling": {
        "what_it_is": "Precision air conditioning units for individual rooms or zones that maintain temperature and humidity in the datacenter space.",
        "traditional_vs_ai": "Traditional datacenters rely heavily on in-building cooling for temperature control. AI datacenters use less air cooling since liquid cooling handles most of the heat load directly at the chip level.",
        "why_it_matters": "In-building cooling units provide environmental control. AI reduces dependency on air cooling through direct liquid cooling of GPUs."
    },
    "chiller": {
        "what_it_is": "Large refrigeration systems that remove heat from facility cooling water loops and reject it to the outside environment.",
        "traditional_vs_ai": "AI datacenters need significantly larger chiller capacity to handle concentrated heat loads from liquid-cooled GPU clusters (40-200kW per rack vs 5-15kW traditional).",
        "why_it_matters": "Chillers are the primary heat removal mechanism. AI's concentrated heat loads require much larger chiller capacity."
    },
    "out_of_building_heat_rejector": {
        "what_it_is": "External building heat rejection system with two approaches: Cooling Tower (Evaporative/Wet) sprays water over fill material where evaporation removes heat. Very energy-efficient but high water usage (like body sweating). Dry Cooler (Convective/Dry) works like car radiator with finned tubes and fans. Zero water usage but less efficient in hot climates. Essential for water-scarce regions.",
        "traditional_vs_ai": "Similar technology but AI datacenters need larger heat rejection capacity to handle the increased heat from higher-capacity chillers.",
        "why_it_matters": "External heat rejection systems provide the final heat rejection to the environment. AI's higher heat loads require more heat rejection capacity."
    },
    "chw_pumps_piping_valves": {
        "what_it_is": "Chilled Water circulation systems including pumps, piping, and control valves that distribute cooling throughout the facility.",
        "traditional_vs_ai": "AI datacenters have more complex CHW systems due to integration with CDU (Coolant Distribution Unit) systems for direct chip cooling.",
        "why_it_matters": "CHW systems distribute cooling capacity. AI requires more sophisticated distribution due to liquid cooling integration."
    },
    "cw_pumps_piping_valves": {
        "what_it_is": "Condenser Water systems that circulate warm water from chillers to cooling towers for heat rejection.",
        "traditional_vs_ai": "Similar systems but AI datacenters need higher capacity pumps and piping for increased heat rejection loads.",
        "why_it_matters": "CW systems complete the cooling loop. AI's higher heat loads require upgraded capacity throughout the system."
    },
    "cdu_and_rack_liquid_loop": {
        "what_it_is": "Coolant Distribution Units and rack-level liquid cooling loops that directly cool GPU chips and other high-heat components. **TENANT RESPONSIBILITY** - CDUs are provided by the tenant, not the facility shell.",
        "traditional_vs_ai": "Traditional datacenters: No liquid cooling (tenant provides standard servers). AI datacenters: Tenant provides extensive CDU systems for direct GPU cooling at 40-200kW per rack. Facility provides connections to CHW loops.",
        "why_it_matters": "CDUs enable AI's extreme power densities and are the most AI-specific component, but are tenant equipment outside of facility shell scope. Facility provides interface connections only."
    },

    # Other Systems
    "fire_protection": {
        "what_it_is": "Fire detection and suppression systems including smoke detectors, alarms, and clean agent suppression systems.",
        "traditional_vs_ai": "AI datacenters need enhanced fire protection due to liquid cooling systems (leak risks) and extremely high power densities.",
        "why_it_matters": "Protects against fire risks. AI's liquid cooling and power density create new fire risk scenarios requiring upgraded protection."
    },
    "raised_floor_dropped_ceiling": {
        "what_it_is": "Structural systems that create plenums for airflow and cable routing, including raised floors and suspended ceilings.",
        "traditional_vs_ai": "AI datacenters need reinforced raised floors for heavier liquid-cooled racks and additional piping infrastructure.",
        "why_it_matters": "Provides infrastructure pathways. AI's liquid cooling systems require enhanced structural support and routing capacity."
    },
    "it_enclosures_containment": {
        "what_it_is": "Server racks, cabinets, and containment systems that house IT equipment and manage airflow.",
        "traditional_vs_ai": "Traditional datacenter: Required for hot/cold aisle containment. AI datacenters typically do not require IT enclosures & containment due to liquid cooling architecture. Consider this component as removal/disposal cost rather than replacement in AI scenarios.",
        "why_it_matters": "Houses the compute equipment. AI's liquid-cooled GPUs require completely different rack architecture - this is where vendor lock-in is highest."
    },
    "mgmt_security": {
        "what_it_is": "Management systems (DCIM, monitoring, networking) and physical security systems (access control, cameras, sensors).",
        "traditional_vs_ai": "AI datacenters need enhanced monitoring for GPU temperatures, liquid cooling systems, and power consumption, plus networking for distributed training.",
        "why_it_matters": "Manages and secures the facility. AI requires more sophisticated monitoring due to liquid cooling complexity and distributed training networking needs."
    },
    "lighting": {
        "what_it_is": "Interior lighting systems for the datacenter facility including emergency lighting and controls.",
        "traditional_vs_ai": "AI datacenters often have minimal lighting since they're designed for unmanned operation with remote management.",
        "why_it_matters": "Provides illumination for maintenance. AI facilities prioritize equipment over human comfort, reducing lighting requirements."
    },
    "project_mgmt_facility_eng": {
        "what_it_is": "Engineering design, project management, and construction oversight costs for building the datacenter.",
        "traditional_vs_ai": "AI datacenters require specialized engineering for liquid cooling integration, 800V DC systems, and high-density power distribution.",
        "why_it_matters": "Ensures proper design and construction. AI's specialized systems require more engineering expertise during construction."
    },
    "core_shell": {
        "what_it_is": "Building structure, exterior walls, roof, foundation, and basic architectural elements of the datacenter facility.",
        "traditional_vs_ai": "AI datacenters often use utilitarian building designs optimized for equipment density rather than aesthetics or human comfort.",
        "why_it_matters": "Provides the physical building. AI prioritizes function over form, often resulting in more basic architectural approaches."
    }
}


class ComponentMetric(NamedTuple):
    """A component metric with percentage and rationale"""
    percentage: float
    rationale: str


def validate_percentages_sum_to_100(percentages: list[float], context_name: str) -> None:
    """DRY helper to validate that percentages sum to 100%"""
    total = sum(percentages)
    if abs(total - 1.0) > 0.001:
        raise ValueError(f"{context_name} percentages must sum to 100%, got {total:.1%}")


# Facility-level system percentages (must sum to 100%)
@dataclass
class FacilitySystemPercentages:
    power_pct: float    # Power systems percentage
    cooling_pct: float  # Cooling systems percentage
    other_pct: float    # Other systems percentage

    def __post_init__(self):
        validate_percentages_sum_to_100(
            [self.power_pct, self.cooling_pct, self.other_pct],
            "Facility system"
        )


# Base class for system components with consolidated ComponentSpec and automatic validation
@dataclass
class SystemComponents:
    """Base for component systems with automatic validation."""

    class Spec(NamedTuple):
        """Component specification with structured metrics"""
        construction: ComponentMetric       # Construction percentage and rationale
        replacement: ComponentMetric        # Replacement cost percentage and rationale
        vendor_difference: ComponentMetric  # Vendor difference percentage and rationale

    def __post_init__(self):
        """Automatically validate instance data and class structure on instantiation."""
        self._validate_descriptions()

        from dataclasses import fields
        component_specs = [getattr(self, f.name) for f in fields(self) if f.name != 'DEFAULT_DESCRIPTIONS']
        total = sum(spec.construction.percentage for spec in component_specs)

        if abs(total - 1.0) > 0.001:
            class_name = self.__class__.__name__.replace('Components', '')
            raise ValueError(f"{class_name} construction percentages must sum to 100%, got {total:.1%}")

    def get_components(self) -> Dict[str, 'SystemComponents.Spec']:
        """Return dict of component name -> Spec"""
        from dataclasses import fields
        return {f.name: getattr(self, f.name) for f in fields(self) if f.name != 'DEFAULT_DESCRIPTIONS'}

    @classmethod
    def _validate_descriptions(cls):
        """Validates that the keys in DEFAULT_DESCRIPTIONS match the class fields."""
        if not hasattr(cls, 'DEFAULT_DESCRIPTIONS'):
            return

        defined_fields = {f.name for f in fields(cls) if f.name != "DEFAULT_DESCRIPTIONS"}
        described_fields = set(cls.DEFAULT_DESCRIPTIONS.keys())

        if defined_fields != described_fields:
            missing = defined_fields - described_fields
            extra = described_fields - defined_fields
            raise AttributeError(
                f"Mismatch between defined fields and descriptions in {cls.__name__}. "
                f"Missing descriptions for: {missing or 'None'}. "
                f"Extra descriptions for: {extra or 'None'}."
            )


@dataclass
class PowerComponents(SystemComponents):
    """Power system components with validation"""
    ups: 'SystemComponents.Spec'
    generator: 'SystemComponents.Spec'
    switchgear: 'SystemComponents.Spec'
    critical_distribution: 'SystemComponents.Spec'

    DEFAULT_DESCRIPTIONS = {
        "ups": "UPS: AC-to-DC conversion & battery backup. **MAJOR VENDOR DIFFERENCES**: 800V HVDC vs ±400V DC architectures require completely different rectifier/inverter designs, battery string configurations, and output stages.",
        "generator": "Generators: AC power during utility outage. **MINOR VENDOR DIFFERENCES**: Engine/alternator same regardless of downstream DC architecture; only control systems vary for UPS integration.",
        "switchgear": "Switchgear: Power distribution & protection. **SPLIT DESIGN**: AC switchgear (utility/gen side) largely standardized; DC switchgear (UPS output side) completely different between voltage architectures - different breaker ratings, conductor counts, fault handling.",
        "critical_distribution": "Critical Distribution: Final power delivery to racks. **MAJOR VENDOR DIFFERENCES**: Busway systems, PDUs, and rack connectors are entirely architecture-specific - 2-wire 800V vs 3-wire ±400V systems are completely incompatible."
    }


@dataclass
class CoolingComponents(SystemComponents):
    """Cooling system components with validation"""
    in_building_cooling: 'SystemComponents.Spec'
    chiller: 'SystemComponents.Spec'
    out_of_building_heat_rejector: 'SystemComponents.Spec'
    chw_pumps_piping_valves: 'SystemComponents.Spec'
    cw_pumps_piping_valves: 'SystemComponents.Spec'
    cdu_and_rack_liquid_loop: 'SystemComponents.Spec'

    DEFAULT_DESCRIPTIONS = {
        "in_building_cooling": "Precision air conditioning units for individual rooms or zones that maintain temperature and humidity in the datacenter space.",
        "chiller": "Chiller Plant: The 'refrigerator' that produces cold water.",
        "out_of_building_heat_rejector": "External building heat rejection system with two approaches: Cooling Tower (Evaporative/Wet) sprays water over fill material where evaporation removes heat. Very energy-efficient but high water usage. Dry Cooler (Convective/Dry) works like car radiator with finned tubes and fans. Zero water usage but less efficient in hot climates.",
        "chw_pumps_piping_valves": "CHW loop: Indoor circuit that carries cold water to the IT cooling equipment.",
        "cw_pumps_piping_valves": "CW loop: Outdoor circuit that carries warm water to the cooling towers for heat rejection.",
        "cdu_and_rack_liquid_loop": "Liquid Cooling Loop: Specialized infrastructure for direct-to-chip or immersion cooling."
    }


@dataclass
class OtherComponents(SystemComponents):
    """Other system components with validation"""
    fire_protection: 'SystemComponents.Spec'
    raised_floor_dropped_ceiling: 'SystemComponents.Spec'
    it_enclosures_containment: 'SystemComponents.Spec'
    mgmt_security: 'SystemComponents.Spec'
    lighting: 'SystemComponents.Spec'
    project_mgmt_facility_eng: 'SystemComponents.Spec'
    core_shell: 'SystemComponents.Spec'

    DEFAULT_DESCRIPTIONS = {
        "fire_protection": "Fire detection and suppression systems.",
        "raised_floor_dropped_ceiling": "Structural elements of the data hall like raised flooring and ceiling grids.",
        "it_enclosures_containment": "Server racks, cabinets, and aisle containment systems.",
        "mgmt_security": "DCIM, BMS, and physical security systems.",
        "lighting": "Basic facility lighting.",
        "project_mgmt_facility_eng": "Engineering and management costs for the project.",
        "core_shell": "The basic building structure (walls, roof, etc.)."
    }



@dataclass
class DatacenterRetrofitScenario:
    """Complete scenario with consolidated component specifications"""
    scenario_name: str
    facility_size_mw: int
    facility_cost_per_mw: int

    system_percentages: FacilitySystemPercentages

    power_components: PowerComponents
    cooling_components: CoolingComponents
    other_components: OtherComponents


# Traditional Datacenter - Schneider Electric Capital Cost Calculator defaults
TRADITIONAL_DATACENTER_RETROFIT = DatacenterRetrofitScenario(
    scenario_name="Traditional DC",
    facility_size_mw=100,
    facility_cost_per_mw=10_000_000,

    system_percentages=FacilitySystemPercentages(
        power_pct=0.3684,
        cooling_pct=0.2109,
        other_pct=0.4207
    ),

    power_components=PowerComponents(
        ups=SystemComponents.Spec(
            construction=ComponentMetric(0.2788, "Standard UPS sizing for traditional IT loads."),
            replacement=ComponentMetric(0.5, "UPS systems are upstream bulk power systems sized for total load, not rack distribution. Same UPS works across AI datacenter generations (2025→2035) as total facility power remains constant."),
            vendor_difference=ComponentMetric(0.1, "Low differences: traditional AC UPS systems highly standardized. Any cloud tenant can reuse existing UPS infrastructure.")
        ),
        generator=SystemComponents.Spec(
            construction=ComponentMetric(0.3619, "Largest component of traditional power, sized for full facility backup."),
            replacement=ComponentMetric(0.2, "Generators do NOT need replacement due to AI technology evolution (2025→2035). Power generation requirements remain consistent regardless of AI workload changes."),
            vendor_difference=ComponentMetric(0.0, "No differences: any traditional cloud tenant uses the same backup power. Generator is tenant-agnostic.")
        ),
        switchgear=SystemComponents.Spec(
            construction=ComponentMetric(0.2760, "Standard switchgear for lower power densities."),
            replacement=ComponentMetric(0.25, "25% - testing/service of breakers; minimal replacement."),
            vendor_difference=ComponentMetric(0.0, "No differences: AC switchgear is standardized. Any traditional cloud tenant uses identical electrical distribution.")
        ),
        critical_distribution=SystemComponents.Spec(
            construction=ComponentMetric(0.0833, "Standard PDUs and wiring for 8-10kW racks."),
            replacement=ComponentMetric(0.3, "30% - some breakers/monitoring; not entire busway."),
            vendor_difference=ComponentMetric(0.2, "Low differences: 208V/415V PDUs are industry standard. Compatible across traditional cloud tenants.")
        )
    ),

    cooling_components=CoolingComponents(
        in_building_cooling=SystemComponents.Spec(
            construction=ComponentMetric(0.3072, "Primary method of cooling in traditional air-cooled facilities."),
            replacement=ComponentMetric(0.6, "60% - ~half replaced + major service on remainder."),
            vendor_difference=ComponentMetric(0.1, "Low differences: In-building cooling units highly standardized. Any traditional cloud tenant can reuse existing air handling.")
        ),
        chiller=SystemComponents.Spec(
            construction=ComponentMetric(0.2941, "Sized for full facility heat load, a major cost driver."),
            replacement=ComponentMetric(0.4, "Chillers do NOT need replacement due to AI technology evolution (2025→2035). They operate on setpoints like home air conditioners and can handle different temperature requirements as AI workloads evolve."),
            vendor_difference=ComponentMetric(0.2, "Low differences: chiller technology standardized. Traditional tenants have similar cooling requirements.")
        ),
        out_of_building_heat_rejector=SystemComponents.Spec(
            construction=ComponentMetric(0.0763, "Standard external heat rejection systems."),
            replacement=ComponentMetric(0.5, "50% - fill media/motors replacement."),
            vendor_difference=ComponentMetric(0.0, "No differences: external heat rejection systems are commodity equipment. Any tenant reuses existing systems unchanged.")
        ),
        chw_pumps_piping_valves=SystemComponents.Spec(
            construction=ComponentMetric(0.2244, "Extensive piping required for air-cooled facility."),
            replacement=ComponentMetric(0.3, "30% - active components focus: pumps/motors."),
            vendor_difference=ComponentMetric(0.0, "No differences: CHW piping is standard infrastructure. New tenant uses existing system unchanged.")
        ),
        cw_pumps_piping_valves=SystemComponents.Spec(
            construction=ComponentMetric(0.0980, "Standard piping for condenser loop."),
            replacement=ComponentMetric(0.3, "30% - active components focus."),
            vendor_difference=ComponentMetric(0.0, "No differences: CW piping is commodity infrastructure. New tenant uses existing system unchanged.")
        ),
        cdu_and_rack_liquid_loop=SystemComponents.Spec(
            construction=ComponentMetric(0.0000, "CDUs are tenant equipment, not facility shell responsibility. No liquid cooling in traditional facilities."),
            replacement=ComponentMetric(0.0, "0% - CDUs are tenant-provided equipment, not facility infrastructure."),
            vendor_difference=ComponentMetric(0.0, "Not applicable: CDUs are tenant equipment outside of facility shell scope.")
        )
    ),

    other_components=OtherComponents(
        fire_protection=SystemComponents.Spec(
            construction=ComponentMetric(0.0617, "Standard fire protection systems."),
            replacement=ComponentMetric(0.3, "30% - electronics/sensors; not piping system."),
            vendor_difference=ComponentMetric(0.1, "Low differences: fire suppression systems standardized across traditional datacenters. Any tenant can reuse existing systems.")
        ),
        raised_floor_dropped_ceiling=SystemComponents.Spec(
            construction=ComponentMetric(0.0616, "Standard raised floor for underfloor air distribution."),
            replacement=ComponentMetric(0.1, "10% - minor tile/grate replacement."),
            vendor_difference=ComponentMetric(0.0, "No differences: raised floors are standard infrastructure. Any tenant uses existing floor unchanged.")
        ),
        it_enclosures_containment=SystemComponents.Spec(
            construction=ComponentMetric(0.1007, "Standard server racks and cabinets."),
            replacement=ComponentMetric(0.7, "70% - form factors & cabling evolve quickly."),
            vendor_difference=ComponentMetric(0.3, "Medium differences: rack form factors vary but 19\" standard allows reasonable compatibility across cloud tenants.")
        ),
        mgmt_security=SystemComponents.Spec(
            construction=ComponentMetric(0.1676, "Standard DCIM, BMS, and security systems."),
            replacement=ComponentMetric(0.8, "80% - tech ages out quickly; full refresh."),
            vendor_difference=ComponentMetric(0.4, "Medium differences: DCIM/BMS systems vary by vendor but traditional cloud requirements are similar.")
        ),
        lighting=SystemComponents.Spec(
            construction=ComponentMetric(0.0356, "Standard facility lighting."),
            replacement=ComponentMetric(0.7, "70% - full upgrade to efficient LED."),
            vendor_difference=ComponentMetric(0.0, "No differences: facility lighting is standard. New tenant uses existing lighting unchanged.")
        ),
        project_mgmt_facility_eng=SystemComponents.Spec(
            construction=ComponentMetric(0.1804, "Standard engineering and project management costs."),
            replacement=ComponentMetric(0.2, "20% - planning & execution for refresh project."),
            vendor_difference=ComponentMetric(0.0, "No differences: engineering costs same regardless of traditional tenant. Standard scope of work.")
        ),
        core_shell=SystemComponents.Spec(
            construction=ComponentMetric(0.3923, "Represents the bulk of 'other' costs in a new build."),
            replacement=ComponentMetric(0.05, "5% - minor building work, roof repairs."),
            vendor_difference=ComponentMetric(0.0, "No differences: building structure is tenant-agnostic. Core and shell works for any occupant.")
        )
    )
)

# AI Datacenter - High Power Density with Liquid Cooling
AI_DATACENTER_RETROFIT = DatacenterRetrofitScenario(
    scenario_name="AI DC",
    facility_size_mw=100,
    facility_cost_per_mw=12_000_000,

    system_percentages=FacilitySystemPercentages(
        power_pct=0.45,
        cooling_pct=0.35,
        other_pct=0.20
    ),

    power_components=PowerComponents(
        ups=SystemComponents.Spec(
            construction=ComponentMetric(0.30, "30% vs 28% traditional - modest increase for AI reliability needs."),
            replacement=ComponentMetric(0.5, "UPS systems are upstream bulk power systems sized for total load, not rack distribution. Same UPS works across AI datacenter generations (2025→2035) as total facility power remains constant."),
            vendor_difference=ComponentMetric(0.0, "Standardized: UPS systems are commodity power equipment. Google, Amazon, NVIDIA all use similar UPS specifications.")
        ),
        generator=SystemComponents.Spec(
            construction=ComponentMetric(0.25, "25% vs 36% traditional - major reduction due to grid reliability + battery backup."),
            replacement=ComponentMetric(0.2, "Generators do NOT need replacement due to AI technology evolution (2025→2035). Power generation requirements remain consistent regardless of AI workload changes."),
            vendor_difference=ComponentMetric(0.0, "Standardized: Generator systems are commodity equipment. Same specifications across all AI cloud providers.")
        ),
        switchgear=SystemComponents.Spec(
            construction=ComponentMetric(0.35, "35% vs 28% traditional - major increase for high-density power distribution."),
            replacement=ComponentMetric(1.3, "130% - complete replacement for 800V DC architecture and 40-200kW rack densities."),
            vendor_difference=ComponentMetric(0.3, "Medium differences: Some configuration variations between Google's, Amazon's, and NVIDIA's power distribution specifications.")
        ),
        critical_distribution=SystemComponents.Spec(
            construction=ComponentMetric(0.10, "10% vs 8% traditional - increase for specialized AI rack PDUs and busbars."),
            replacement=ComponentMetric(1.5, "150% - full replacement for 800V DC busways and specialized GPU power delivery."),
            vendor_difference=ComponentMetric(0.4, "Medium differences: Rack-level power distribution varies between AI providers. Google vs Amazon vs NVIDIA have different PDU and busway preferences.")
        )
    ),

    cooling_components=CoolingComponents(
        in_building_cooling=SystemComponents.Spec(
            construction=ComponentMetric(0.25, "25% vs 31% traditional - reduced air handling since liquid cooling does more work, but still significant facility responsibility."),
            replacement=ComponentMetric(0.9, "90% - hybrid approach: reduced air handling but upgraded for remaining load."),
            vendor_difference=ComponentMetric(0.2, "Low differences: In-building cooling units are fairly standardized. Google vs Amazon vs NVIDIA use similar air handling specifications.")
        ),
        chiller=SystemComponents.Spec(
            construction=ComponentMetric(0.40, "40% vs 29% traditional - increased chiller capacity for AI heat loads. Facility shell provides chiller capacity."),
            replacement=ComponentMetric(0.4, "Chillers do NOT need replacement due to AI technology evolution (2025→2035). They operate on setpoints like home air conditioners and can handle different temperature requirements as AI workloads evolve."),
            vendor_difference=ComponentMetric(0.1, "Low differences: Chillers are commodity equipment. Same specifications across AI cloud providers.")
        ),
        out_of_building_heat_rejector=SystemComponents.Spec(
            construction=ComponentMetric(0.15, "15% vs 8% traditional - increased heat rejection for higher densities. Facility shell responsibility."),
            replacement=ComponentMetric(0.5, "External heat rejection systems use same retrofit rates as traditional datacenter (normal wear and tear, not AI-specific replacement)."),
            vendor_difference=ComponentMetric(0.0, "Standardized: External heat rejection systems are commodity equipment. Same across all AI providers.")
        ),
        chw_pumps_piping_valves=SystemComponents.Spec(
            construction=ComponentMetric(0.15, "15% vs 22% traditional - reduced since tenant CDUs handle rack-level cooling but facility still provides primary CHW distribution."),
            replacement=ComponentMetric(1.0, "100% - integration with tenant CDU systems and higher flow rate requirements."),
            vendor_difference=ComponentMetric(0.1, "Low differences: Pumps/piping mostly standardized, minor integration differences between AI providers.")
        ),
        cw_pumps_piping_valves=SystemComponents.Spec(
            construction=ComponentMetric(0.05, "5% vs 10% traditional - reduced due to less facility-level pumping, but still facility shell responsibility."),
            replacement=ComponentMetric(1.1, "110% - upgraded for higher heat rejection loads from liquid cooling."),
            vendor_difference=ComponentMetric(0.1, "Low differences: Condenser water systems are standardized across AI providers.")
        ),
        cdu_and_rack_liquid_loop=SystemComponents.Spec(
            construction=ComponentMetric(0.0000, "CDUs are tenant equipment, not facility shell responsibility. Tenant provides liquid cooling infrastructure."),
            replacement=ComponentMetric(0.0, "0% - CDUs are tenant-provided equipment, not facility infrastructure."),
            vendor_difference=ComponentMetric(0.9, "Major differences: CDU systems highly proprietary. NVIDIA/AMD partnerships, custom manifolds, vendor-specific cooling loops. Zero tenant portability - but tenant responsibility.")
        )
    ),

    other_components=OtherComponents(
        fire_protection=SystemComponents.Spec(
            construction=ComponentMetric(0.10, "10% vs 6% traditional - enhanced fire suppression for dense equipment."),
            replacement=ComponentMetric(0.8, "80% - upgraded systems for liquid cooling fire risks and dense equipment."),
            vendor_difference=ComponentMetric(0.0, "Standardized: Fire protection systems are code-mandated and standardized across all AI providers.")
        ),
        raised_floor_dropped_ceiling=SystemComponents.Spec(
            construction=ComponentMetric(0.08, "8% vs 6% traditional - enhanced floor systems for liquid cooling."),
            replacement=ComponentMetric(0.6, "60% - structural upgrades for heavier liquid-cooled racks and piping."),
            vendor_difference=ComponentMetric(0.0, "No differences: raised floors are structural. AI tenant uses existing floor - they don't care about cosmetics.")
        ),
        it_enclosures_containment=SystemComponents.Spec(
            construction=ComponentMetric(0.25, "25% vs 10% traditional - MAJOR increase for liquid-cooled AI racks."),
            replacement=ComponentMetric(1.2, "120% - technological obsolescence: specialized liquid-cooled GPU racks replace air-cooled."),
            vendor_difference=ComponentMetric(1.0, "Complete replacement: liquid-cooled GPU racks are vendor-specific. NVIDIA vs AMD partnerships drive completely different rack designs.")
        ),
        mgmt_security=SystemComponents.Spec(
            construction=ComponentMetric(0.20, "20% vs 17% traditional - increase for GPU telemetry and monitoring systems."),
            replacement=ComponentMetric(2.0, "200% - major investment in GPU sensors, advanced DCIM, and thermal monitoring."),
            vendor_difference=ComponentMetric(0.2, "Low differences: basic management Ethernet, DCIM software, and physical security largely reusable. Just standard IT management stuff.")
        ),
        lighting=SystemComponents.Spec(
            construction=ComponentMetric(0.02, "2% vs 4% traditional - reduced due to utilitarian focus."),
            replacement=ComponentMetric(0.5, "50% - minimal upgrades for unmanned AI facilities."),
            vendor_difference=ComponentMetric(0.0, "No differences: AI facilities don't care about lighting. Existing LED lights work fine.")
        ),
        project_mgmt_facility_eng=SystemComponents.Spec(
            construction=ComponentMetric(0.15, "15% vs 18% traditional - reduced due to less facility complexity."),
            replacement=ComponentMetric(0.3, "30% - design integration for 800V DC and liquid cooling systems."),
            vendor_difference=ComponentMetric(0.0, "No differences: engineering costs are the same regardless of AI tenant. Standard design work.")
        ),
        core_shell=SystemComponents.Spec(
            construction=ComponentMetric(0.20, "20% vs 39% traditional - MAJOR reduction with focus on equipment."),
            replacement=ComponentMetric(0.15, "15% - focus on equipment; minimal building envelope changes."),
            vendor_difference=ComponentMetric(0.0, "No differences: building structure is tenant-agnostic. AI companies use existing shell unchanged.")
        )
    )
)

# List of scenarios
RETROFIT_SCENARIOS = [
    TRADITIONAL_DATACENTER_RETROFIT,
    AI_DATACENTER_RETROFIT,
]
