"""Quick test to verify all OpenEnv API methods work correctly."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from crowd_env import CrowdManagementEnv, Action, ActionType, TASKS
from crowd_env.models import RiskLevel

print("=" * 60)
print("  Testing CrowdManagementEnv API")
print("=" * 60)

env = CrowdManagementEnv()

# Test 1: reset()
print("\n[TEST 1] reset() ...")
obs = env.reset(seed=42, options={"task": "easy"})
assert obs is not None
assert obs.total_population == 0
assert len(obs.zones) == 6
assert obs.time_step == 0
print(f"  OK: {len(obs.zones)} zones, population={obs.total_population}")

# Test 2: step() with no-op
print("\n[TEST 2] step(no_op) ...")
result = env.step(Action.noop())
assert result is not None
assert result.observation is not None
assert isinstance(result.reward, float)
assert isinstance(result.terminated, bool)
print(f"  OK: reward={result.reward}, terminated={result.terminated}, pop={result.observation.total_population}")

# Test 3: state()
print("\n[TEST 3] state() ...")
state = env.state()
assert state is not None
assert state.step_count == 1
assert state.task_id == "easy"
assert not state.terminated
print(f"  OK: step={state.step_count}, task={state.task_id}, episode={state.episode_id}")

# Test 4: Action types
print("\n[TEST 4] All action types ...")
actions = [
    Action.redirect("A", "B"),
    Action.close_gate("A", 0),
    Action.open_gate("A", 0),
    Action.issue_alert("D"),
    Action.noop(),
]
for a in actions:
    r = env.step(a)
    print(f"  {a.action_type:<15} -> reward={r.reward:+.2f}, pop={r.observation.total_population}")

# Test 5: Run full easy episode
print("\n[TEST 5] Full easy episode ...")
obs = env.reset(seed=42, options={"task": "easy"})
steps = 0
while True:
    result = env.step(Action.noop())
    steps += 1
    if result.terminated or result.truncated:
        break
print(f"  OK: {steps} steps completed, terminated={result.terminated}, truncated={result.truncated}")

# Test 6: Grading
print("\n[TEST 6] Grading ...")
grade = env.grade()
assert grade is not None
assert 0.0 <= grade.score <= 1.0
assert grade.letter_grade in ("A", "B", "C", "D", "F")
print(f"  OK: Score={grade.score:.3f}, Grade={grade.letter_grade}")
print(f"  Safety={grade.safety_score:.3f}, Efficiency={grade.efficiency_score:.3f}")
print(f"  Survival={grade.survival_score:.3f}, Proactivity={grade.proactivity_score:.3f}")

# Test 7: All tasks exist
print("\n[TEST 7] Task definitions ...")
for tid, task in TASKS.items():
    print(f"  {tid}: {task.name} (max_steps={task.max_steps}, arrival={task.base_arrival_rate})")
assert len(TASKS) == 3

# Test 8: Medium task with smart actions
print("\n[TEST 8] Medium task with smart agent ...")
obs = env.reset(seed=42, options={"task": "medium"})
steps = 0
while True:
    # Simple heuristic: alert on any elevated/critical zone
    action = Action.noop()
    for z in obs.zones:
        if z.risk_level in ("critical", "elevated") and not z.alert_active:
            action = Action.issue_alert(z.zone_id)
            break
    result = env.step(action)
    obs = result.observation
    steps += 1
    if result.terminated or result.truncated:
        break
grade = env.grade()
print(f"  OK: {steps} steps, Score={grade.score:.3f} ({grade.letter_grade})")

# Test 9: Serialization
print("\n[TEST 9] Serialization (to_dict) ...")
obs_dict = obs.to_dict()
assert isinstance(obs_dict, dict)
assert "zones" in obs_dict
state_dict = env.state().to_dict()
assert isinstance(state_dict, dict)
grade_dict = grade.to_dict()
assert isinstance(grade_dict, dict)
print(f"  OK: obs keys={list(obs_dict.keys())}")

# Test 10: Risk classification
print("\n[TEST 10] Risk classification ...")
from crowd_env.simulation import classify_risk
assert classify_risk(1.0) == RiskLevel.SAFE
assert classify_risk(2.5) == RiskLevel.ELEVATED
assert classify_risk(4.0) == RiskLevel.CRITICAL
assert classify_risk(6.0) == RiskLevel.STAMPEDE
print("  OK: All risk thresholds correct")

print("\n" + "=" * 60)
print("  ALL 10 TESTS PASSED!")
print("=" * 60)
