"""
Data Models for Crowd Management OpenEnv Environment
=====================================================
Type-safe Pydantic definitions following the OpenEnv contract.
"""

from __future__ import annotations

from enum import Enum
from typing import List, Dict, Optional, Tuple, Any

from pydantic import BaseModel, Field


# ─── Enums ────────────────────────────────────────────────────────────────────

class RiskLevel(str, Enum):
    """Risk classification based on crowd density (people per m²)."""
    SAFE = "safe"             # density < 2.0 ppm²
    ELEVATED = "elevated"     # 2.0 ≤ density < 3.5 ppm²
    CRITICAL = "critical"     # 3.5 ≤ density < 5.0 ppm²
    STAMPEDE = "stampede"     # density ≥ 5.0 ppm²


class ActionType(str, Enum):
    """Types of crowd management actions the agent can take."""
    REDIRECT = "redirect"         # Divert crowd flow between zones
    GATE_CONTROL = "gate_control" # Open or close access points
    ALERT = "alert"               # Issue crowd alert/restriction
    NO_OP = "no_op"               # Take no action


# ─── Zone Config (immutable layout) ──────────────────────────────────────────

class ZoneConfig(BaseModel):
    """Static configuration for a venue zone."""
    zone_id: str
    name: str
    area_sqm: float
    capacity: int
    is_entry: bool = False
    is_exit: bool = False
    num_gates: int = 2
    neighbors: Tuple[str, ...] = Field(default_factory=tuple)


# ─── Zone Info (per-step observation) ─────────────────────────────────────────

class ZoneInfo(BaseModel):
    """Observable state of a single zone at a given timestep."""
    zone_id: str
    name: str
    current_population: int
    capacity: int
    area_sqm: float
    density: float                 # people per square meter
    inflow_rate: float             # people/step entering this zone
    outflow_rate: float            # people/step leaving this zone
    risk_level: str                # RiskLevel value
    gates_open: List[bool] = Field(default_factory=list)
    neighbors: List[str] = Field(default_factory=list)
    alert_active: bool = False

    @property
    def occupancy_ratio(self) -> float:
        """Fraction of capacity currently filled (0.0 – 1.0+)."""
        return self.current_population / max(self.capacity, 1)


# ─── Observation ──────────────────────────────────────────────────────────────

class Observation(BaseModel):
    """
    What the agent sees at each timestep.
    Provides structured data about all zones, global metrics, and recent events.
    """
    zones: List[ZoneInfo]
    total_population: int
    global_risk_score: float       # 0.0 (safe) – 1.0 (stampede imminent)
    time_step: int
    max_steps: int
    alerts_active: List[str]       # zone IDs with active alerts
    event_log: List[str]           # recent events

    def get_zone(self, zone_id: str) -> Optional[ZoneInfo]:
        """Get a specific zone's info by ID."""
        for z in self.zones:
            if z.zone_id == zone_id:
                return z
        return None

    def to_dict(self) -> dict:
        return self.model_dump()


# ─── Action ───────────────────────────────────────────────────────────────────

class Action(BaseModel):
    """
    An action the agent can take to manage crowd flow.

    action_type: The type of intervention
    source_zone: Zone to act on (required for all except no_op)
    target_zone: Destination zone (required for redirect)
    gate_index:  Which gate to control (for gate_control)
    gate_open:   Whether to open (True) or close (False) the gate
    """
    action_type: str = ActionType.NO_OP.value
    source_zone: str = ""
    target_zone: str = ""
    gate_index: int = 0
    gate_open: bool = True

    def to_dict(self) -> dict:
        return self.model_dump()

    @classmethod
    def redirect(cls, source: str, target: str) -> "Action":
        """Factory: create a redirect action."""
        return cls(action_type=ActionType.REDIRECT.value, source_zone=source, target_zone=target)

    @classmethod
    def close_gate(cls, zone: str, gate_idx: int = 0) -> "Action":
        """Factory: close a gate in a zone."""
        return cls(action_type=ActionType.GATE_CONTROL.value, source_zone=zone, gate_index=gate_idx, gate_open=False)

    @classmethod
    def open_gate(cls, zone: str, gate_idx: int = 0) -> "Action":
        """Factory: open a gate in a zone."""
        return cls(action_type=ActionType.GATE_CONTROL.value, source_zone=zone, gate_index=gate_idx, gate_open=True)

    @classmethod
    def issue_alert(cls, zone: str) -> "Action":
        """Factory: issue an alert on a zone."""
        return cls(action_type=ActionType.ALERT.value, source_zone=zone)

    @classmethod
    def noop(cls) -> "Action":
        """Factory: take no action."""
        return cls(action_type=ActionType.NO_OP.value)


# ─── Reward (OpenEnv Spec) ────────────────────────────────────────────────────

class Reward(BaseModel):
    """Reward returned by the environment."""
    value: float


# ─── State ────────────────────────────────────────────────────────────────────

class State(BaseModel):
    """
    Full internal state of the environment.
    Returned by state() — includes metadata + complete zone data.
    """
    episode_id: str
    step_count: int
    max_steps: int
    task_id: str
    terminated: bool
    truncated: bool
    cumulative_reward: float
    stampede_occurred: bool
    stampede_zone: Optional[str]
    zones: List[ZoneInfo] = Field(default_factory=list)
    action_history: List[Dict[str, Any]] = Field(default_factory=list)
    total_arrivals: int = 0
    total_departures: int = 0
    peak_density: float = 0.0
    peak_density_zone: str = ""

    def to_dict(self) -> dict:
        return self.model_dump()


# ─── Step Result ──────────────────────────────────────────────────────────────

class StepResult(BaseModel):
    """
    Return value of step(action).
    Follows OpenEnv convention: (observation, reward, terminated, truncated, info).
    """
    observation: Observation
    reward: float
    terminated: bool
    truncated: bool
    info: Dict[str, Any] = Field(default_factory=dict)

    def to_dict(self) -> dict:
        return self.model_dump()
