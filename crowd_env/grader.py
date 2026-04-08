"""
Deterministic Grading System
==============================
Evaluates agent performance across safety, efficiency, survival, and proactivity.
Produces a 0.0–1.0 score with letter grade.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class EpisodeMetrics:
    """Metrics collected during an episode for grading."""
    total_steps: int = 0
    steps_all_safe: int = 0            # steps where ALL zones were safe
    steps_any_elevated: int = 0        # steps with at least one elevated zone
    steps_any_critical: int = 0        # steps with at least one critical zone
    stampede_occurred: bool = False
    stampede_step: Optional[int] = None

    total_actions: int = 0             # total non-noop actions taken
    unnecessary_actions: int = 0       # actions taken when no threat existed
    preemptive_actions: int = 0        # actions on elevated zones before they go critical
    elevated_incidents: int = 0        # total number of elevated zone-steps

    cumulative_reward: float = 0.0
    peak_density: float = 0.0
    peak_density_zone: str = ""

    def to_dict(self) -> dict:
        return {
            "total_steps": self.total_steps,
            "steps_all_safe": self.steps_all_safe,
            "steps_any_elevated": self.steps_any_elevated,
            "steps_any_critical": self.steps_any_critical,
            "stampede_occurred": self.stampede_occurred,
            "stampede_step": self.stampede_step,
            "total_actions": self.total_actions,
            "unnecessary_actions": self.unnecessary_actions,
            "preemptive_actions": self.preemptive_actions,
            "elevated_incidents": self.elevated_incidents,
            "cumulative_reward": round(self.cumulative_reward, 3),
            "peak_density": round(self.peak_density, 3),
            "peak_density_zone": self.peak_density_zone,
        }


@dataclass
class GradeResult:
    """Final grade for an episode."""
    score: float               # 0.0–1.0
    letter_grade: str          # A, B, C, D, F
    safety_score: float        # 0.0–1.0
    efficiency_score: float    # 0.0–1.0
    survival_score: float      # 0.0 or 1.0
    proactivity_score: float   # 0.0–1.0
    summary: str               # Human-readable summary
    metrics: EpisodeMetrics = field(default_factory=EpisodeMetrics)

    def to_dict(self) -> dict:
        return {
            "score": round(self.score, 4),
            "letter_grade": self.letter_grade,
            "components": {
                "safety_score": round(self.safety_score, 4),
                "efficiency_score": round(self.efficiency_score, 4),
                "survival_score": round(self.survival_score, 4),
                "proactivity_score": round(self.proactivity_score, 4),
            },
            "summary": self.summary,
            "metrics": self.metrics.to_dict(),
        }


# ─── Grade Weights ────────────────────────────────────────────────────────────

WEIGHT_SAFETY = 0.40
WEIGHT_EFFICIENCY = 0.15
WEIGHT_SURVIVAL = 0.30
WEIGHT_PROACTIVITY = 0.15


def _letter_grade(score: float) -> str:
    """Convert numeric score to letter grade."""
    if score >= 0.90:
        return "A"
    elif score >= 0.75:
        return "B"
    elif score >= 0.60:
        return "C"
    elif score >= 0.40:
        return "D"
    else:
        return "F"


def _grade_summary(letter: str, metrics: EpisodeMetrics) -> str:
    """Generate a human-readable summary based on grade."""
    summaries = {
        "A": (
            "🏆 Expert Performance — Maintained safe crowd levels proactively. "
            "No zones reached critical density. "
            f"Peak density: {metrics.peak_density:.2f} ppm² at zone {metrics.peak_density_zone}."
        ),
        "B": (
            "✅ Proficient Performance — Managed most situations effectively "
            "with minor elevated density periods. "
            f"Steps at safe levels: {metrics.steps_all_safe}/{metrics.total_steps}."
        ),
        "C": (
            "⚠ Adequate Performance — Some periods of overcrowding occurred but "
            "no stampede resulted. Room for improvement in proactive management. "
            f"Critical zone-steps: {metrics.steps_any_critical}."
        ),
        "D": (
            "🔴 Poor Performance — Frequent overcrowding and close calls. "
            f"Unnecessary actions: {metrics.unnecessary_actions}. "
            f"Peak density reached {metrics.peak_density:.2f} ppm²."
        ),
        "F": (
            "💀 Failed — A stampede occurred"
            + (f" at step {metrics.stampede_step}" if metrics.stampede_step else "")
            + " or massive, uncontrolled overcrowding prevailed. "
            "Critical intervention strategies needed."
        ),
    }
    return summaries.get(letter, "Grade unavailable.")


class CrowdManagementGrader:
    """
    Deterministic grading engine for evaluating agent performance.

    Score = w1 × safety + w2 × efficiency + w3 × survival + w4 × proactivity
    Each component is scored 0.0–1.0.
    """

    def __init__(self):
        self.metrics = EpisodeMetrics()

    def reset(self):
        """Reset metrics for a new episode."""
        self.metrics = EpisodeMetrics()

    def record_step(
        self,
        all_safe: bool,
        any_elevated: bool,
        any_critical: bool,
        action_type: str,
        action_was_useful: bool,
        elevated_zones: List[str],
        action_target_zone: str,
        reward: float,
    ):
        """Record metrics for a single step."""
        self.metrics.total_steps += 1
        self.metrics.cumulative_reward += reward

        if all_safe:
            self.metrics.steps_all_safe += 1
        if any_elevated:
            self.metrics.steps_any_elevated += 1
            self.metrics.elevated_incidents += len(elevated_zones) if elevated_zones else 1
        if any_critical:
            self.metrics.steps_any_critical += 1

        # Track actions (excluding no_op)
        if action_type != "no_op":
            self.metrics.total_actions += 1
            if not action_was_useful:
                self.metrics.unnecessary_actions += 1
            # Preemptive: action on an elevated zone before it becomes critical
            if action_target_zone in elevated_zones and not any_critical:
                self.metrics.preemptive_actions += 1

    def record_stampede(self, step: int, zone: str):
        """Record that a stampede occurred."""
        self.metrics.stampede_occurred = True
        self.metrics.stampede_step = step

    def record_peak(self, density: float, zone: str):
        """Update peak density."""
        if density > self.metrics.peak_density:
            self.metrics.peak_density = density
            self.metrics.peak_density_zone = zone

    def compute_grade(self) -> GradeResult:
        """Compute the final deterministic grade."""
        m = self.metrics

        # ── Safety Score ──
        if m.total_steps > 0:
            safety_score = float(m.steps_all_safe) / float(m.total_steps)
        else:
            safety_score = 1.0

        # ── Efficiency Score ──
        # Measures whether interventions were targeted at actual threats.
        # More forgiving: uses a graduated curve instead of harsh linear penalty.
        if m.total_actions > 0:
            useful_ratio = 1.0 - (float(m.unnecessary_actions) / float(m.total_actions))
            # Graduated scoring: square the ratio so small mistakes aren't devastating
            # e.g., 1 bad action out of 5 → ratio=0.8 → score=0.64 (not harsh 0.8)
            # but mostly good → 4/5 useful → 0.8² = 0.64 which is still respectable
            efficiency_score = max(useful_ratio, 0.2)  # Floor at 0.2, never zero
        else:
            # No actions taken at all — if everything stayed safe, that's efficient
            if m.steps_all_safe == m.total_steps:
                efficiency_score = 1.0  # Perfect restraint when no threats existed
            else:
                efficiency_score = 0.3  # Threats existed but agent did nothing

        # ── Survival Score ──
        survival_score = 0.0 if m.stampede_occurred else 1.0

        # ── Proactivity Score ──
        if m.elevated_incidents > 0:
            proactivity_score = min(
                float(m.preemptive_actions) / float(m.elevated_incidents),
                1.0,
            )
        else:
            # No elevated incidents — perfect proactivity (nothing to respond to)
            proactivity_score = 1.0

        # ── Weighted Total ──
        score = (
            WEIGHT_SAFETY * safety_score
            + WEIGHT_EFFICIENCY * efficiency_score
            + WEIGHT_SURVIVAL * survival_score
            + WEIGHT_PROACTIVITY * proactivity_score
        )
        score = max(0.0, min(1.0, score))

        letter = _letter_grade(score)
        summary = _grade_summary(letter, m)

        return GradeResult(
            score=score,
            letter_grade=letter,
            safety_score=safety_score,
            efficiency_score=efficiency_score,
            survival_score=survival_score,
            proactivity_score=proactivity_score,
            summary=summary,
            metrics=m,
        )
