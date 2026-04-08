"""
Reinforcement Learning Training Script
======================================
Trains a PPO agent on the CrowdManagementEnv using Stable-Baselines3.
"""

import os
import time

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env

from crowd_env.rl_wrapper import CrowdGymWrapper

def main():
    print("Initializing RL Training Environment...")
    
    # Optional: We could wrap it in DummyVecEnv, but make_vec_env handles this cleaner.
    env = make_vec_env(lambda: CrowdGymWrapper(task_id="hard"), n_envs=4)
    eval_env = CrowdGymWrapper(task_id="hard")

    model_dir = "models"
    log_dir = "logs"
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # Callback to periodically evaluate the agent during training
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=model_dir,
        log_path=log_dir,
        eval_freq=5000,
        deterministic=True,
        render=False
    )

    print("Building PPO Model...")
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=log_dir,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=64,
        ent_coef=0.01,
    )

    # 100K timesteps is a reasonable training run (~10-15 min on CPU).
    # For production-quality results, increase to 500K-1M+.
    total_timesteps = 100_000
    
    print(f"Beginning Training for {total_timesteps} timesteps...")
    start_time = time.time()
    
    model.learn(
        total_timesteps=total_timesteps,
        callback=eval_callback,
        progress_bar=True
    )
    
    end_time = time.time()
    print(f"Training completed in {end_time - start_time:.2f} seconds.")

    # Save final model
    model_path = os.path.join(model_dir, "ppo_crowd_final")
    model.save(model_path)
    print(f"Model saved to {model_path}.zip")

if __name__ == "__main__":
    main()
