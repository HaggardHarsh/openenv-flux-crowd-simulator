"""
Random Agent Demo
==================
Demonstrates the CrowdManagementEnv API by running a random agent
through all three difficulty tasks and printing results.
"""

import sys
import os
import random

# Allow running from project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Handle Windows console encoding (cp1252 can't render emojis)
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ('utf-8', 'utf8'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from crowd_env import CrowdManagementEnv, Action, ActionType, TASKS
from crowd_env.grader import GradeResult
from crowd_env.agent import smart_heuristic


def print_header(text: str, char: str = "═", width: int = 72):
    print(f"\n{char * width}")
    print(f"  {text}")
    print(f"{char * width}")


def print_zone_table(obs):
    """Print a formatted zone status table."""
    print(f"\n  {'Zone':<18} {'Pop':>6} {'Cap':>6} {'Density':>8} {'Risk':>10} {'Gates':>8} {'Alert':>6}")
    print(f"  {'─' * 18} {'─' * 6} {'─' * 6} {'─' * 8} {'─' * 10} {'─' * 8} {'─' * 6}")
    for z in obs.zones:
        risk_icon = {"safe": "🟢", "elevated": "🟡", "critical": "🔴", "stampede": "💀"}.get(z.risk_level, "?")
        gates_str = f"{sum(z.gates_open)}/{len(z.gates_open)}"
        alert_str = "🚨" if z.alert_active else "—"
        print(
            f"  {z.name:<18} {z.current_population:>6} {z.capacity:>6} "
            f"{z.density:>7.2f}  {risk_icon} {z.risk_level:<8} {gates_str:>8} {alert_str:>6}"
        )


def random_action(env) -> Action:
    """Generate a random valid action."""
    zone_ids = env.zone_ids
    if not zone_ids:
        return Action.noop()

    action_type = random.choice([ActionType.REDIRECT, ActionType.GATE_CONTROL, ActionType.ALERT, ActionType.NO_OP])

    if action_type == ActionType.NO_OP:
        return Action.noop()

    source = random.choice(zone_ids)

    if action_type == ActionType.REDIRECT:
        # Pick a random neighbor
        obs = env._build_observation()
        zone_info = obs.get_zone(source)
        if zone_info and zone_info.neighbors:
            target = random.choice(zone_info.neighbors)
            return Action.redirect(source, target)
        return Action.noop()

    elif action_type == ActionType.GATE_CONTROL:
        obs = env._build_observation()
        zone_info = obs.get_zone(source)
        max_gate = len(zone_info.gates_open) - 1 if zone_info and zone_info.gates_open else 1
        gate_idx = random.randint(0, max_gate)
        gate_open = random.choice([True, False])
        if gate_open:
            return Action.open_gate(source, gate_idx)
        else:
            return Action.close_gate(source, gate_idx)

    elif action_type == ActionType.ALERT:
        return Action.issue_alert(source)

    return Action.noop()




def run_episode(env, task_id: str, agent: str = "random", verbose: bool = True, seed: int = 42):
    """Run a single episode and return the grade."""
    task_config = TASKS[task_id]
    print_header(f"Task: {task_config.name} ({task_id.upper()})")
    print(f"  {task_config.description}")
    print(f"  Max steps: {task_config.max_steps} | Seed: {seed}")

    obs = env.reset(seed=seed, options={"task": task_id})

    if verbose:
        print(f"\n  Initial state (Step 0):")
        print_zone_table(obs)

    step = 0
    while True:
        # Choose action
        if agent == "smart":
            action = smart_heuristic(obs)
        else:
            action = random_action(env)

        result = env.step(action)
        obs = result.observation
        step += 1

        # Print periodic updates
        if verbose and (step % 25 == 0 or result.terminated or result.truncated):
            print(f"\n  Step {step} | Reward: {result.reward:+.2f} | "
                  f"Population: {obs.total_population} | Risk: {obs.global_risk_score:.3f}")
            if action.action_type != "no_op":
                print(f"  Action: {result.info.get('action_result', '')}")
            print_zone_table(obs)

            if obs.event_log:
                for event in obs.event_log[-3:]:
                    print(f"    {event}")

        if result.terminated:
            print(f"\n  ❌ EPISODE TERMINATED — Stampede at zone {result.info.get('stampede_zone', '?')}")
            break
        if result.truncated:
            print(f"\n  ✅ EPISODE COMPLETED — Survived all {step} steps!")
            break

    # Grade
    grade = env.grade()
    print(f"\n  ┌──────────────────────────────────────────┐")
    print(f"  │  GRADE: {grade.letter_grade}  ({grade.score:.3f} / 1.000)           │")
    print(f"  ├──────────────────────────────────────────┤")
    print(f"  │  Safety:     {grade.safety_score:>6.3f} (×0.40)          │")
    print(f"  │  Efficiency: {grade.efficiency_score:>6.3f} (×0.15)          │")
    print(f"  │  Survival:   {grade.survival_score:>6.3f} (×0.30)          │")
    print(f"  │  Proactivity:{grade.proactivity_score:>6.3f} (×0.15)          │")
    print(f"  └──────────────────────────────────────────┘")
    print(f"  {grade.summary}")

    state = env.state()
    print(f"\n  Stats: {state.total_arrivals} arrivals | {state.total_departures} departures | "
          f"Peak density: {state.peak_density:.2f} at {state.peak_density_zone}")

    return grade


def main():
    print_header("AI-Powered Crowd Management - OpenEnv Demo", "=", 72)
    print("  Demonstrating the CrowdManagementEnv with random + smart agents")
    print("  across Easy, Medium, and Hard difficulty tasks.\n")

    env = CrowdManagementEnv()

    # Run with random agent
    print_header("PART 1: RANDOM AGENT", "─")
    grades_random = {}
    for task_id in ["easy", "medium", "hard"]:
        grade = run_episode(env, task_id, agent="random", verbose=True, seed=42)
        grades_random[task_id] = grade

    # Run with smart heuristic agent
    print_header("PART 2: SMART HEURISTIC AGENT", "─")
    grades_smart = {}
    for task_id in ["easy", "medium", "hard"]:
        grade = run_episode(env, task_id, agent="smart", verbose=True, seed=42)
        grades_smart[task_id] = grade

    # Summary
    print_header("FINAL COMPARISON", "=")
    print(f"\n  {'Task':<12} {'Random':>12} {'Smart':>12}")
    print(f"  {'─' * 12} {'─' * 12} {'─' * 12}")
    for task_id in ["easy", "medium", "hard"]:
        rg = grades_random[task_id]
        sg = grades_smart[task_id]
        print(f"  {task_id.upper():<12} {rg.letter_grade:>4} ({rg.score:>5.3f}) {sg.letter_grade:>4} ({sg.score:>5.3f})")

    print(f"\n  The smart heuristic agent should consistently outperform the random agent.")
    print(f"  A trained RL agent would achieve even better scores.\n")


if __name__ == "__main__":
    main()
