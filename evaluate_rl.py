"""
RL Agent Evaluation Script
==========================
Loads the trained PPO model and evaluates it using OpenEnv's deterministic grader.
"""

import os
from stable_baselines3 import PPO
from crowd_env.rl_wrapper import CrowdGymWrapper

def main():
    model_path = "models/ppo_crowd_final.zip"
    if not os.path.exists(model_path):
        # Fallback to best_model if final doesn't exist
        model_path = "models/best_model.zip"
        
    if not os.path.exists(model_path):
        print("No trained model found! Please run 'python train_rl.py' first.")
        return

    print(f"Loading model from {model_path}...")
    model = PPO.load(model_path)

    # Evaluate on the medium task
    task = "medium"
    print(f"Evaluating on '{task}' task...")
    
    gym_env = CrowdGymWrapper(task_id=task)
    obs, info = gym_env.reset(seed=1337)
    
    steps = 0
    done = False
    
    while not done:
        # Predict action
        action_idx, _states = model.predict(obs, deterministic=True)
        
        # Step env
        obs, reward, terminated, truncated, info = gym_env.step(action_idx)
        done = terminated or truncated
        steps += 1
        
    # Since we wrapped it, the inner env holds our deterministic grader state
    grade = gym_env.env.grade()
    print("\n" + "=" * 40)
    print(f"RL Evaluation Results ({steps} steps)")
    print("=" * 40)
    print(f"Letter Grade: {grade.letter_grade}")
    print(f"Final Score:  {grade.score:.3f} / 1.000")
    print("\nSummary:")
    print(grade.summary)

if __name__ == "__main__":
    main()
