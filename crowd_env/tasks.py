"""
Task Definitions for Crowd Management Environment
===================================================
Three difficulty tiers: easy, medium, hard.
Each task configures arrival rates, surge events, exit constraints, and panic factors.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass(frozen=True)
class SurgeEvent:
    """A temporary spike in crowd arrivals at a specific zone."""
    timestep: int          # When the surge starts
    duration: int          # How many steps the surge lasts
    zone_id: str           # Which zone receives the surge
    intensity: float       # Multiplier on base arrival rate (e.g., 3.0 = 3x normal)


@dataclass(frozen=True)
class TaskConfig:
    """
    Configuration for a crowd management scenario.
    Controls all simulation parameters that define difficulty.
    """
    task_id: str
    name: str
    description: str

    # Episode length
    max_steps: int

    # Arrival parameters
    base_arrival_rate: float           # average people/step at entry zones
    entry_zones: Tuple[str, ...]       # which zones receive external arrivals
    exit_zones: Tuple[str, ...]        # which zones allow departures

    # Exit constraints
    exit_capacity_multiplier: float    # 1.0 = normal, 0.5 = 50% reduced throughput

    # Surge events
    surges: Tuple[SurgeEvent, ...] = ()

    # Crowd behavior
    panic_factor: float = 0.0         # 0.0–1.0: how much density amplifies pressure
    attraction_zones: Tuple[str, ...] = ("D",)  # zones people naturally gravitate toward

    # Inter-zone flow
    base_flow_rate: float = 0.08      # fraction of population that moves per step
    congestion_threshold: float = 3.5  # density above which flow slows down

    # Grading
    par_score: float = 0.7           # "par" score for this difficulty


# ─── Task Definitions ─────────────────────────────────────────────────────────

TASK_EASY = TaskConfig(
    task_id="easy",
    name="Matchday Warm-Up",
    description=(
        "A calm matchday with steady, low-density crowd flow through a single "
        "entrance. No surge events. Exits operate at full capacity. "
        "Focus: learn the basics of crowd monitoring and safe flow management."
    ),
    max_steps=100,
    base_arrival_rate=15.0,
    entry_zones=("A",),
    exit_zones=("F",),
    exit_capacity_multiplier=1.0,
    surges=(),
    panic_factor=0.0,
    base_flow_rate=0.08,
    congestion_threshold=3.5,
    par_score=0.75,
)


TASK_MEDIUM = TaskConfig(
    task_id="medium",
    name="Derby Day Rush",
    description=(
        "A popular derby match with multi-gate arrivals and two halftime surge events. "
        "Higher crowd volume and periodic spikes require proactive gate management "
        "and timely redirections. Exits have slightly reduced capacity."
    ),
    max_steps=200,
    base_arrival_rate=30.0,
    entry_zones=("A", "E"),
    exit_zones=("F",),
    exit_capacity_multiplier=0.75,
    surges=(
        SurgeEvent(timestep=50, duration=15, zone_id="A", intensity=2.5),
        SurgeEvent(timestep=130, duration=20, zone_id="E", intensity=3.0),
    ),
    panic_factor=0.3,
    base_flow_rate=0.07,
    congestion_threshold=3.5,
    par_score=0.65,
)


TASK_HARD = TaskConfig(
    task_id="hard",
    name="Championship Final Chaos",
    description=(
        "The championship final with massive crowds flooding from three entrances. "
        "Five overlapping surge events simulate halftime rushes, goal celebrations, "
        "and post-match exodus. Exits are severely constrained. High panic factor "
        "means density cascades rapidly. Only expert crowd management prevents disaster."
    ),
    max_steps=300,
    base_arrival_rate=50.0,
    entry_zones=("A", "B", "E"),
    exit_zones=("F",),
    exit_capacity_multiplier=0.50,
    surges=(
        SurgeEvent(timestep=30, duration=20, zone_id="A", intensity=3.0),
        SurgeEvent(timestep=80, duration=15, zone_id="E", intensity=2.5),
        SurgeEvent(timestep=120, duration=25, zone_id="A", intensity=3.5),
        SurgeEvent(timestep=180, duration=20, zone_id="B", intensity=2.8),
        SurgeEvent(timestep=240, duration=30, zone_id="A", intensity=4.0),
    ),
    panic_factor=0.7,
    base_flow_rate=0.06,
    congestion_threshold=3.0,
    par_score=0.55,
)


# ─── Task Registry ────────────────────────────────────────────────────────────

TASKS = {
    "easy": TASK_EASY,
    "medium": TASK_MEDIUM,
    "hard": TASK_HARD,
}


def get_task(task_id: str) -> TaskConfig:
    """Retrieve a task configuration by ID."""
    if task_id not in TASKS:
        raise ValueError(
            f"Unknown task '{task_id}'. Available tasks: {list(TASKS.keys())}"
        )
    return TASKS[task_id]
