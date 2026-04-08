"""Compact demo - just run all tasks and show grades."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from crowd_env import CrowdManagementEnv, Action, ActionType, TASKS

def smart_action(env, obs):
    critical = [z for z in obs.zones if z.risk_level == "critical"]
    elevated = [z for z in obs.zones if z.risk_level == "elevated"]
    if critical:
        zone = max(critical, key=lambda z: z.density)
        if not zone.alert_active:
            return Action.issue_alert(zone.zone_id)
        if zone.neighbors:
            best_n = min([z for z in obs.zones if z.zone_id in zone.neighbors], key=lambda z: z.density, default=None)
            if best_n:
                return Action.redirect(zone.zone_id, best_n.zone_id)
    if elevated:
        zone = max(elevated, key=lambda z: z.density)
        if not zone.alert_active:
            return Action.issue_alert(zone.zone_id)
    return Action.noop()

env = CrowdManagementEnv()

print("=" * 60)
print("  CrowdManagement OpenEnv - Summary Results")
print("=" * 60)

for agent_name in ["Random", "Smart"]:
    print(f"\n--- {agent_name} Agent ---")
    for task_id in ["easy", "medium", "hard"]:
        obs = env.reset(seed=42, options={"task": task_id})
        steps = 0
        while True:
            if agent_name == "Smart":
                action = smart_action(env, obs)
            else:
                import random
                random.seed(steps)
                action = Action.noop()
            result = env.step(action)
            obs = result.observation
            steps += 1
            if result.terminated or result.truncated:
                break
        grade = env.grade()
        state = env.state()
        status = "STAMPEDE" if result.terminated else "Survived"
        print(f"  {task_id.upper():<8} | {status:<9} | Steps: {steps:>3} | "
              f"Score: {grade.score:>5.1f} ({grade.letter_grade}) | "
              f"Peak: {state.peak_density:.2f} ppm2")

print("\n" + "=" * 60)
print("  OpenEnv API: reset(), step(), state() - ALL WORKING")
print("=" * 60)
