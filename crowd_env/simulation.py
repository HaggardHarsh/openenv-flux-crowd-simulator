"""
Crowd Simulation Engine
========================
Physics-inspired crowd flow model with density-based risk assessment.
Simulates external arrivals, inter-zone movement, exit flow, and congestion effects.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from crowd_env.models import ZoneConfig, ZoneInfo, RiskLevel
from crowd_env.tasks import TaskConfig, SurgeEvent


# ─── Venue Layout: 6-Zone Stadium ────────────────────────────────────────────

STADIUM_ZONES: Tuple[ZoneConfig, ...] = (
    ZoneConfig(
        zone_id="A", name="Main Entrance", area_sqm=500.0, capacity=1000,
        is_entry=True, is_exit=False, num_gates=3, neighbors=("B", "C"),
    ),
    ZoneConfig(
        zone_id="B", name="North Stand", area_sqm=800.0, capacity=2000,
        is_entry=False, is_exit=False, num_gates=2, neighbors=("A", "D", "E"),
    ),
    ZoneConfig(
        zone_id="C", name="South Stand", area_sqm=800.0, capacity=2000,
        is_entry=False, is_exit=False, num_gates=2, neighbors=("A", "D", "F"),
    ),
    ZoneConfig(
        zone_id="D", name="Central Arena", area_sqm=1200.0, capacity=3000,
        is_entry=False, is_exit=False, num_gates=4, neighbors=("B", "C", "E", "F"),
    ),
    ZoneConfig(
        zone_id="E", name="East Concourse", area_sqm=400.0, capacity=800,
        is_entry=True, is_exit=False, num_gates=2, neighbors=("B", "D"),
    ),
    ZoneConfig(
        zone_id="F", name="West Exit", area_sqm=400.0, capacity=800,
        is_entry=False, is_exit=True, num_gates=3, neighbors=("C", "D"),
    ),
)


# ─── Risk Classification ─────────────────────────────────────────────────────

def classify_risk(density: float) -> RiskLevel:
    """Classify risk level from crowd density (people/m²)."""
    if density < 2.0:
        return RiskLevel.SAFE
    elif density < 3.5:
        return RiskLevel.ELEVATED
    elif density < 5.0:
        return RiskLevel.CRITICAL
    else:
        return RiskLevel.STAMPEDE


# ─── Zone Runtime State ──────────────────────────────────────────────────────

@dataclass
class ZoneState:
    """Mutable runtime state for a single zone during simulation."""
    config: ZoneConfig
    population: int = 0
    gates_open: List[bool] = field(default_factory=list)
    alert_active: bool = False
    last_inflow: float = 0.0
    last_outflow: float = 0.0

    def __post_init__(self):
        if not self.gates_open:
            self.gates_open = [True] * self.config.num_gates

    @property
    def density(self) -> float:
        return self.population / max(self.config.area_sqm, 1.0)

    @property
    def risk_level(self) -> RiskLevel:
        return classify_risk(self.density)

    @property
    def open_gate_count(self) -> int:
        return sum(1 for g in self.gates_open if g)

    @property
    def gate_throughput_factor(self) -> float:
        """Fraction of gates open (affects flow capacity)."""
        if self.config.num_gates == 0:
            return 1.0
        return self.open_gate_count / self.config.num_gates

    def to_zone_info(self) -> ZoneInfo:
        return ZoneInfo(
            zone_id=self.config.zone_id,
            name=self.config.name,
            current_population=self.population,
            capacity=self.config.capacity,
            area_sqm=self.config.area_sqm,
            density=round(self.density, 4),
            inflow_rate=round(self.last_inflow, 2),
            outflow_rate=round(self.last_outflow, 2),
            risk_level=self.risk_level.value,
            gates_open=list(self.gates_open),
            neighbors=list(self.config.neighbors),
            alert_active=self.alert_active,
        )


# ─── Simulation Engine ───────────────────────────────────────────────────────

class CrowdSimulation:
    """
    Core crowd dynamics engine.

    Models:
    - External arrivals (Poisson-distributed) at entry zones
    - Inter-zone flow towards less-dense and attractive zones
    - Exit flow through exit zones
    - Congestion effects at high density
    - Surge events (temporary arrival spikes)
    - Agent interventions (redirections, gate control, alerts)
    """

    def __init__(self, task: TaskConfig, seed: Optional[int] = None):
        self.task = task
        self.rng = np.random.default_rng(seed)

        # Initialize zone states
        self.zones: Dict[str, ZoneState] = {}
        for zc in STADIUM_ZONES:
            self.zones[zc.zone_id] = ZoneState(config=zc)

        self.time_step = 0
        self.total_arrivals = 0
        self.total_departures = 0
        self.event_log: List[str] = []
        self.peak_density = 0.0
        self.peak_density_zone = ""

        # Active redirections: {source_zone_id: target_zone_id}
        self._active_redirects: Dict[str, str] = {}
        # Redirect durations remaining
        self._redirect_ttl: Dict[str, int] = {}

    def get_zone_infos(self) -> List[ZoneInfo]:
        """Get observable info for all zones."""
        return [self.zones[zc.zone_id].to_zone_info() for zc in STADIUM_ZONES]

    def get_global_risk_score(self) -> float:
        """Compute a global risk score 0.0 – 1.0 from all zone densities."""
        max_density = max(z.density for z in self.zones.values())
        # Normalize: 0 at density=0, 1.0 at density=5.0
        return min(max_density / 5.0, 1.0)

    def get_active_alerts(self) -> List[str]:
        """Get list of zone IDs with active alerts."""
        return [zid for zid, zs in self.zones.items() if zs.alert_active]

    def _get_active_surges(self) -> List[SurgeEvent]:
        """Get surges active at the current timestep."""
        return [
            s for s in self.task.surges
            if s.timestep <= self.time_step < s.timestep + s.duration
        ]

    def _compute_arrival_rate(self, zone_id: str) -> float:
        """Compute effective arrival rate for an entry zone."""
        if zone_id not in self.task.entry_zones:
            return 0.0

        zone = self.zones[zone_id]
        rate = self.task.base_arrival_rate / len(self.task.entry_zones)

        # Apply gate throughput
        rate *= zone.gate_throughput_factor

        # Apply surge multiplier
        for surge in self._get_active_surges():
            if surge.zone_id == zone_id:
                rate *= surge.intensity

        # Alert reduces inflow by 40%
        if zone.alert_active:
            rate *= 0.6

        return max(rate, 0.0)

    def _compute_exit_rate(self, zone_id: str) -> float:
        """Compute exit flow rate for an exit zone."""
        if zone_id not in self.task.exit_zones:
            return 0.0

        zone = self.zones[zone_id]
        if zone.population <= 0:
            return 0.0

        # Base exit rate: 10% of population per step, scaled by gate throughput
        base_rate = zone.population * 0.10 * zone.gate_throughput_factor

        # Apply task exit capacity constraint
        base_rate *= self.task.exit_capacity_multiplier

        # Congestion: high density reduces exit throughput
        if zone.density > self.task.congestion_threshold:
            congestion_factor = max(0.3, 1.0 - (zone.density - self.task.congestion_threshold) * 0.2)
            base_rate *= congestion_factor

        return max(base_rate, 0.0)

    def _compute_inter_zone_flow(self) -> Dict[Tuple[str, str], int]:
        """
        Compute natural inter-zone movement.
        People flow toward less-dense neighbors, with bias toward attraction zones.
        Returns dict of (source, target) -> number of people moving.
        """
        flows: Dict[Tuple[str, str], int] = {}

        for zone_id, zone in self.zones.items():
            if zone.population <= 0:
                continue

            neighbors = zone.config.neighbors
            if not neighbors:
                continue

            # Number of people willing to move
            mobile_fraction = self.task.base_flow_rate * zone.gate_throughput_factor
            mobile_count = int(zone.population * mobile_fraction)

            if mobile_count <= 0:
                continue

            # Congestion in current zone increases desire to leave
            if zone.density > self.task.congestion_threshold:
                pressure = 1.0 + self.task.panic_factor * (zone.density - self.task.congestion_threshold)
                mobile_count = int(mobile_count * pressure)

            # Compute attractiveness of each neighbor
            weights = []
            valid_neighbors = []
            for nid in neighbors:
                n = self.zones[nid]
                # Can't flow through closed gates
                if n.gate_throughput_factor <= 0:
                    continue

                # Base weight: inverse density (prefer less crowded)
                weight = max(0.1, 1.0 / (1.0 + n.density))

                # Attraction zone bonus
                if nid in self.task.attraction_zones:
                    weight *= 1.5

                # Redirect override
                if zone_id in self._active_redirects:
                    redirect_target = self._active_redirects[zone_id]
                    if nid == redirect_target:
                        weight *= 3.0  # Strong bias toward redirect target
                    else:
                        weight *= 0.3  # Reduced flow to non-target

                # Alert on neighbor discourages flow toward it
                if n.alert_active:
                    weight *= 0.4

                weights.append(weight)
                valid_neighbors.append(nid)

            if not valid_neighbors:
                continue

            # Distribute mobile people among neighbors
            total_weight = sum(weights)
            for nid, w in zip(valid_neighbors, weights):
                share = int(mobile_count * w / total_weight)
                if share > 0:
                    flows[(zone_id, nid)] = flows.get((zone_id, nid), 0) + share

        return flows

    def step_simulation(self) -> List[str]:
        """
        Advance the simulation by one timestep.
        Returns a list of event strings describing what happened.
        """
        self.time_step += 1
        events: List[str] = []

        # Track inflow/outflow per zone
        inflow: Dict[str, float] = {zid: 0.0 for zid in self.zones}
        outflow: Dict[str, float] = {zid: 0.0 for zid in self.zones}

        # ── 1. External Arrivals ──
        for zone_id in self.task.entry_zones:
            rate = self._compute_arrival_rate(zone_id)
            arrivals = self.rng.poisson(max(rate, 0))
            if arrivals > 0:
                self.zones[zone_id].population += arrivals
                inflow[zone_id] += arrivals
                self.total_arrivals += arrivals

        # Log surge events
        for surge in self._get_active_surges():
            remaining = surge.timestep + surge.duration - self.time_step
            events.append(
                f"⚡ SURGE at {self.zones[surge.zone_id].config.name} "
                f"({surge.intensity:.1f}x intensity, {remaining} steps remaining)"
            )

        # ── 2. Inter-Zone Movement ──
        flows = self._compute_inter_zone_flow()
        for (src, dst), count in flows.items():
            actual = min(count, self.zones[src].population)
            if actual > 0:
                self.zones[src].population -= actual
                self.zones[dst].population += actual
                outflow[src] += actual
                inflow[dst] += actual

        # ── 3. Exit Flow ──
        for zone_id in self.task.exit_zones:
            rate = self._compute_exit_rate(zone_id)
            exits = min(int(rate), self.zones[zone_id].population)
            if exits > 0:
                self.zones[zone_id].population -= exits
                outflow[zone_id] += exits
                self.total_departures += exits

        # ── 4. Update Flow Rates & Track Peaks ──
        for zone_id, zone in self.zones.items():
            zone.last_inflow = inflow[zone_id]
            zone.last_outflow = outflow[zone_id]
            zone.population = max(zone.population, 0)

            if zone.density > self.peak_density:
                self.peak_density = zone.density
                self.peak_density_zone = zone_id

        # ── 5. Decay Redirects ──
        expired = []
        for zid in self._redirect_ttl:
            self._redirect_ttl[zid] -= 1
            if self._redirect_ttl[zid] <= 0:
                expired.append(zid)
        for zid in expired:
            del self._redirect_ttl[zid]
            if zid in self._active_redirects:
                events.append(f"↩ Redirect from {self.zones[zid].config.name} expired")
                del self._active_redirects[zid]

        # ── 6. Risk Assessment Events ──
        for zone_id, zone in self.zones.items():
            risk = zone.risk_level
            if risk == RiskLevel.ELEVATED:
                events.append(f"⚠ {zone.config.name} density ELEVATED ({zone.density:.2f} ppm²)")
            elif risk == RiskLevel.CRITICAL:
                events.append(f"🔴 {zone.config.name} density CRITICAL ({zone.density:.2f} ppm²)")
            elif risk == RiskLevel.STAMPEDE:
                events.append(f"💀 STAMPEDE at {zone.config.name}! ({zone.density:.2f} ppm²)")

        # Store events
        self.event_log = events[-10:]  # Keep last 10

        return events

    def apply_redirect(self, source_zone: str, target_zone: str) -> str:
        """Apply a crowd redirect from source to target zone. Lasts 10 steps."""
        if source_zone not in self.zones:
            return f"❌ Unknown source zone: {source_zone}"
        if target_zone not in self.zones:
            return f"❌ Unknown target zone: {target_zone}"
        if target_zone not in self.zones[source_zone].config.neighbors:
            return f"❌ {target_zone} is not adjacent to {source_zone}"

        self._active_redirects[source_zone] = target_zone
        self._redirect_ttl[source_zone] = 10
        src_name = self.zones[source_zone].config.name
        dst_name = self.zones[target_zone].config.name
        return f"🔀 Redirecting crowd from {src_name} → {dst_name} (10 steps)"

    def apply_gate_control(self, zone_id: str, gate_index: int, gate_open: bool) -> str:
        """Open or close a specific gate in a zone."""
        if zone_id not in self.zones:
            return f"❌ Unknown zone: {zone_id}"
        zone = self.zones[zone_id]
        if gate_index < 0 or gate_index >= zone.config.num_gates:
            return f"❌ Invalid gate index {gate_index} for {zone.config.name} (has {zone.config.num_gates} gates)"
        zone.gates_open[gate_index] = gate_open
        action = "🟢 Opened" if gate_open else "🔴 Closed"
        return f"{action} gate {gate_index} at {zone.config.name}"

    def apply_alert(self, zone_id: str) -> str:
        """Issue or toggle alert on a zone (reduces inflow by 40%)."""
        if zone_id not in self.zones:
            return f"❌ Unknown zone: {zone_id}"
        zone = self.zones[zone_id]
        zone.alert_active = not zone.alert_active
        if zone.alert_active:
            return f"🚨 Alert ACTIVATED at {zone.config.name} — inflow reduced"
        else:
            return f"✅ Alert LIFTED at {zone.config.name}"

    def check_stampede(self) -> Optional[str]:
        """Check if any zone has reached stampede density. Returns zone_id or None."""
        for zone_id, zone in self.zones.items():
            if zone.risk_level == RiskLevel.STAMPEDE:
                return zone_id
        return None

    def get_reward_components(self) -> Dict[str, float]:
        """Compute per-step reward components."""
        components = {}

        # Risk-based reward
        risk_levels = [z.risk_level for z in self.zones.values()]

        if all(r == RiskLevel.SAFE for r in risk_levels):
            components["safety_bonus"] = 2.0
        elif any(r == RiskLevel.CRITICAL for r in risk_levels):
            # Penalty proportional to number of critical zones
            n_critical = sum(1 for r in risk_levels if r == RiskLevel.CRITICAL)
            components["critical_penalty"] = -2.0 * n_critical
        elif any(r == RiskLevel.ELEVATED for r in risk_levels):
            components["elevated_mild"] = 0.5

        # Stampede terminal penalty
        if any(r == RiskLevel.STAMPEDE for r in risk_levels):
            components["stampede_penalty"] = -50.0

        return components
