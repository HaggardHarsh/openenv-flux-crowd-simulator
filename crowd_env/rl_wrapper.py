"""
Gymnasium Wrapper for CrowdManagementEnv
==========================================
Bridges the gap between OpenEnv Pydantic returns and StableBaselines3 NumPy requirements.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from crowd_env.environment import CrowdManagementEnv
from crowd_env.models import Action


class CrowdGymWrapper(gym.Env):
    """
    Wraps the CrowdManagementEnv to conform to gymnasium.Env API.
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self, task_id="easy", seed=None):
        super().__init__()
        self.env = CrowdManagementEnv()
        self.task_id = task_id
        self._seed = seed
        
        # Initialize internal env to gather static info (e.g. zones)
        self.env.reset(seed=self._seed, options={"task": self.task_id})
        self.zone_ids = self.env.zone_ids
        
        # Build deterministic action space mapping
        self._action_table = self._build_action_table()
        self.action_space = spaces.Discrete(len(self._action_table))
        
        # Observation space
        num_features = len(self.zone_ids) * 7 + 2
        
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(num_features,), 
            dtype=np.float32
        )

    def _build_action_table(self):
        """Construct a lookup table of valid actions."""
        table = []
        # 1. No-op
        table.append(Action.noop())
        
        # 2. Zone specific commands
        for z in self.zone_ids:
            # Alert
            table.append(Action.issue_alert(z))
            
            # Gate control (assume max 4 gates)
            for g_idx in range(4):
                table.append(Action.close_gate(z, g_idx))
                table.append(Action.open_gate(z, g_idx))
                
            # Redirects to all other zones
            for target in self.zone_ids:
                if z != target:
                    table.append(Action.redirect(z, target))
                    
        return table

    def _encode_observation(self, obs) -> np.ndarray:
        """Flatten Pydantic Observation to NumPy array."""
        features = []
        risk_map = {"safe": 0.0, "elevated": 1.0, "critical": 2.0, "stampede": 3.0}
        
        for zid in self.zone_ids:
            z_info = obs.get_zone(zid)
            if z_info:
                features.append(z_info.occupancy_ratio)
                features.append(z_info.density)
                # Normalize inflow/outflow roughly to [-1, 1] range to aid neural network 
                features.append(z_info.inflow_rate / 50.0) 
                features.append(z_info.outflow_rate / 50.0)
                features.append(risk_map.get(z_info.risk_level, 0.0))
                features.append(1.0 if z_info.alert_active else 0.0)
                
                open_gates = sum(z_info.gates_open)
                total_gates = len(z_info.gates_open) if z_info.gates_open else 1
                features.append(open_gates / float(total_gates))
            else:
                features.extend([0.0] * 7)
                
        features.append(obs.global_risk_score)
        features.append(obs.time_step / max(1, obs.max_steps))
        
        return np.array(features, dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._seed = seed
        opt = {"task": self.task_id}
        if options:
            opt.update(options)
            
        obs = self.env.reset(seed=self._seed, options=opt)
        return self._encode_observation(obs), {}

    def step(self, action_idx):
        discrete_action = self._action_table[int(action_idx)]
        result = self.env.step(discrete_action)
        obs_array = self._encode_observation(result.observation)
        
        # Penalizing stampede severely when terminated helps PPO learn faster
        reward = result.reward
        if result.terminated:
            reward -= 50.0 
            
        return obs_array, reward, result.terminated, result.truncated, result.info

    def render(self):
        pass
