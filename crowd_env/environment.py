"""
Crowd Management OpenEnv Environment
======================================
OpenEnv-compliant environment with reset(), step(), and state() API.
Integrates the simulation engine, task system, and grading.
"""

from __future__ import annotations

import uuid
from typing import Optional, Dict, List

from crowd_env.models import (
    Action, ActionType, Observation, State, StepResult, RiskLevel,
)
from crowd_env.simulation import CrowdSimulation
from crowd_env.tasks import TaskConfig, get_task, TASKS
from crowd_env.grader import CrowdManagementGrader
from crowd_env.agent import smart_heuristic


class CrowdManagementEnv:
    """
    AI-Powered Crowd Management Environment.

    Simulates dynamic crowd behavior across multiple zones in a stadium venue.
    The agent observes crowd conditions and takes actions to maintain safe
    density levels and prevent stampede situations.

    Complies with the OpenEnv specification:
        - reset()          → initializes the environment
        - step(action)     → updates the environment, returns (obs, reward, done, info)
        - state()          → provides the full system state

    Usage:
        env = CrowdManagementEnv()
        obs = env.reset(seed=42, options={"task": "medium"})
        while True:
            action = agent.act(obs)
            result = env.step(action)
            obs = result.observation
            if result.terminated or result.truncated:
                break
        grade = env.grade()
    """

    def __init__(self):
        self._sim: Optional[CrowdSimulation] = None
        self._task: Optional[TaskConfig] = None
        self._state: Optional[State] = None
        self._grader = CrowdManagementGrader()
        self._step_count = 0
        self._episode_id = ""
        self._done = False
        self._cumulative_reward = 0.0

    # ─── OpenEnv API ──────────────────────────────────────────────────────

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Observation:
        """
        Initialize (or re-initialize) the environment for a new episode.

        Args:
            seed: Random seed for reproducibility.
            options: Dict with optional keys:
                - "task": str — task ID ("easy", "medium", "hard"). Default: "easy".

        Returns:
            Initial Observation of the environment.
        """
        options = options or {}
        task_id = options.get("task", "easy")
        self._task = get_task(task_id)

        # Create simulation
        self._sim = CrowdSimulation(self._task, seed=seed)
        self._grader.reset()

        # Episode metadata
        self._episode_id = str(uuid.uuid4())[:8]
        self._step_count = 0
        self._done = False
        self._cumulative_reward = 0.0

        # Build initial state
        self._update_state(terminated=False, truncated=False, stampede_zone=None)

        return self._build_observation()

    def step(self, action: Action) -> StepResult:
        """
        Execute an action, advance the simulation by one timestep.

        Args:
            action: The Action to execute.

        Returns:
            StepResult containing (observation, reward, terminated, truncated, info).
        """
        if self._sim is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")
        if self._done:
            raise RuntimeError("Episode is done. Call reset() to start a new episode.")

        self._step_count += 1
        info: Dict = {"action_result": "", "events": []}

        # ── 0. Resolve "auto" actions via smart heuristic ──
        if action.action_type == "auto":
            obs = self._build_observation()
            action = smart_heuristic(obs)

        # ── 1. Apply Agent Action ──
        action_result = self._apply_action(action)
        info["action_result"] = action_result

        # Determine if action was on a zone with actual threat
        action_was_useful = self._is_action_useful(action)

        # ── 2. Advance Simulation ──
        events = self._sim.step_simulation()
        info["events"] = events

        # ── 3. Check Termination ──
        stampede_zone = self._sim.check_stampede()
        terminated = stampede_zone is not None
        truncated = self._step_count >= self._task.max_steps

        # ── 4. Compute Reward ──
        reward = self._compute_reward(action, stampede_zone)
        self._cumulative_reward += reward
        info["reward_breakdown"] = self._sim.get_reward_components()

        # ── 5. Record Grading Metrics ──
        zone_risks = {zid: z.risk_level for zid, z in self._sim.zones.items()}
        all_safe = all(r == RiskLevel.SAFE for r in zone_risks.values())
        any_elevated = any(r == RiskLevel.ELEVATED for r in zone_risks.values())
        any_critical = any(r == RiskLevel.CRITICAL for r in zone_risks.values())
        elevated_zones = [zid for zid, r in zone_risks.items() if r == RiskLevel.ELEVATED]

        self._grader.record_step(
            all_safe=all_safe,
            any_elevated=any_elevated,
            any_critical=any_critical,
            action_type=action.action_type,
            action_was_useful=action_was_useful,
            elevated_zones=elevated_zones,
            action_target_zone=action.source_zone,
            reward=reward,
        )

        # Track peak density
        for zid, zone in self._sim.zones.items():
            self._grader.record_peak(zone.density, zid)

        if stampede_zone:
            self._grader.record_stampede(self._step_count, stampede_zone)
            info["stampede_zone"] = stampede_zone

        # ── 6. End-of-episode bonus ──
        if truncated and not terminated:
            # Survived the entire episode
            reward += 20.0
            self._cumulative_reward += 20.0
            info["survival_bonus"] = 20.0

        self._done = terminated or truncated

        # ── 7. Update State ──
        self._update_state(terminated, truncated, stampede_zone)

        return StepResult(
            observation=self._build_observation(),
            reward=round(reward, 3),
            terminated=terminated,
            truncated=truncated,
            info=info,
        )

    def state(self) -> State:
        """
        Provide the full internal state of the environment.

        Returns:
            State object with episode metadata and complete zone data.
        """
        if self._state is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")
        return self._state

    # ─── Grading ──────────────────────────────────────────────────────────

    def grade(self):
        """
        Compute the deterministic grade for the current episode.
        Should be called after the episode is done.

        Returns:
            GradeResult with score, letter grade, and component breakdown.
        """
        return self._grader.compute_grade()

    # ─── Internal Helpers ─────────────────────────────────────────────────

    def _apply_action(self, action: Action) -> str:
        """Apply the agent's action to the simulation. Returns result string."""
        atype = action.action_type

        if atype == ActionType.REDIRECT or atype == "redirect":
            return self._sim.apply_redirect(action.source_zone, action.target_zone)
        elif atype == ActionType.GATE_CONTROL or atype == "gate_control":
            return self._sim.apply_gate_control(
                action.source_zone, action.gate_index, action.gate_open
            )
        elif atype == ActionType.ALERT or atype == "alert":
            return self._sim.apply_alert(action.source_zone)
        elif atype == ActionType.NO_OP or atype == "no_op":
            return "— No action taken"
        else:
            return f"❌ Unknown action type: {atype}"

    def _is_action_useful(self, action: Action) -> bool:
        """Determine if an action addresses an actual threat or is a valid recovery action."""
        if action.action_type in (ActionType.NO_OP, "no_op"):
            return True  # No-op is always "useful" (neutral)

        zone_id = action.source_zone
        source_at_risk = False
        if zone_id and zone_id in self._sim.zones:
            zone = self._sim.zones[zone_id]
            # Useful if the zone is at elevated or critical risk
            source_at_risk = zone.risk_level in (RiskLevel.ELEVATED, RiskLevel.CRITICAL, RiskLevel.STAMPEDE)

            # Recovery actions on safe zones are also useful:
            # - Re-opening a previously closed gate
            if action.action_type in (ActionType.GATE_CONTROL, "gate_control") and action.gate_open:
                if not zone.gates_open[action.gate_index] if action.gate_index < len(zone.gates_open) else False:
                    return True  # Re-opening a closed gate is useful recovery

            # - Lifting an alert on a zone that is now safe
            if action.action_type in (ActionType.ALERT, "alert") and zone.alert_active and zone.risk_level == RiskLevel.SAFE:
                return True  # Lifting alert on a recovered zone is useful

        # For redirects, also check target zone
        if action.action_type in (ActionType.REDIRECT, "redirect") and action.target_zone:
            target = self._sim.zones.get(action.target_zone)
            if target and target.risk_level in (RiskLevel.ELEVATED, RiskLevel.CRITICAL):
                return True

        return source_at_risk

    def _compute_reward(self, action: Action, stampede_zone: Optional[str]) -> float:
        """Compute the reward for the current step."""
        reward = 0.0
        components = self._sim.get_reward_components()

        for component, value in components.items():
            reward += value

        # Bonus for taking action on a threatened zone
        if action.action_type not in (ActionType.NO_OP, "no_op"):
            if self._is_action_useful(action):
                reward += 1.0
            else:
                reward -= 0.3  # Penalty for unnecessary intervention

        return reward

    def _build_observation(self) -> Observation:
        """Build the observation visible to the agent."""
        return Observation(
            zones=self._sim.get_zone_infos(),
            total_population=sum(z.population for z in self._sim.zones.values()),
            global_risk_score=self._sim.get_global_risk_score(),
            time_step=self._step_count,
            max_steps=self._task.max_steps,
            alerts_active=self._sim.get_active_alerts(),
            event_log=list(self._sim.event_log),
        )

    def _update_state(self, terminated: bool, truncated: bool, stampede_zone: Optional[str]):
        """Update the internal state object."""
        self._state = State(
            episode_id=self._episode_id,
            step_count=self._step_count,
            max_steps=self._task.max_steps,
            task_id=self._task.task_id,
            terminated=terminated,
            truncated=truncated,
            cumulative_reward=self._cumulative_reward,
            stampede_occurred=stampede_zone is not None,
            stampede_zone=stampede_zone,
            zones=self._sim.get_zone_infos(),
            total_arrivals=self._sim.total_arrivals,
            total_departures=self._sim.total_departures,
            peak_density=self._sim.peak_density,
            peak_density_zone=self._sim.peak_density_zone,
        )

    # ─── Convenience ──────────────────────────────────────────────────────

    @property
    def available_actions(self) -> List[str]:
        """List all available action types."""
        return [a.value for a in ActionType]

    @property
    def zone_ids(self) -> List[str]:
        """List all zone IDs in the venue."""
        if self._sim:
            return list(self._sim.zones.keys())
        return []

    def get_task_info(self) -> dict:
        """Get info about the current task."""
        if self._task:
            return {
                "task_id": self._task.task_id,
                "name": self._task.name,
                "description": self._task.description,
                "max_steps": self._task.max_steps,
                "difficulty": self._task.task_id,
            }
        return {}
