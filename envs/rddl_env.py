"""
SB3-compatible wrapper around pyRDDLGym environments.

All algos (PPO/A2C/DQN): action_space=Discrete(N+1), observation_space=Box(float32, (obs_dim,)).
Valid joint actions (noop + N singletons) are enumerated at construction time.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces

import pyRDDLGym

from envs.action_space import enumerate_valid_combos, build_discrete_space, decode_action
from envs.shaping import RewardShaper


class RDDLGymWrapper(gym.Env):
    """Gymnasium wrapper for pyRDDLGym compatible with Stable-Baselines3.

    Args:
        domain:   rddlrepository domain name, e.g. 'Elevators_MDP_ippc2011'
        instance: instance string, e.g. '1'
        algo:     one of 'ppo', 'a2c', 'dqn'
        seed:     optional random seed
        shaper:   optional RewardShaper; None means no shaping (phase-1 behaviour)
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        domain: str,
        instance: str,
        algo: str,
        seed: Optional[int] = None,
        shaper: Optional[RewardShaper] = None,
    ) -> None:
        super().__init__()
        self.domain = domain
        self.instance = instance
        self.algo = algo.lower()
        assert self.algo in ("ppo", "a2c", "dqn", "noop", "random", "ppo-cp"), f"unknown algo: {algo}"
        self._shaper: Optional[RewardShaper] = shaper

        self._env = pyRDDLGym.make(domain, instance)

        self.horizon: int = int(self._env.horizon)  # type: ignore[arg-type]
        self.max_allowed_actions: int = int(self._env.max_allowed_actions)  # type: ignore[arg-type]
        self._noop: Dict = dict(self._env.sampler.grounded_noop_actions)

        self._action_keys: List[str] = sorted(self._env.action_space.spaces.keys())  # type: ignore[union-attr]
        self._obs_keys: List[str] = sorted(self._env.observation_space.spaces.keys())  # type: ignore[union-attr]

        self._combos = enumerate_valid_combos(self._action_keys)
        self.action_space = build_discrete_space(self._combos)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(len(self._obs_keys),), dtype=np.float32
        )

        if seed is not None:
            self._env.seed(seed)
            np.random.seed(seed)

    def _flatten_obs(self, obs_dict: Dict) -> np.ndarray:
        return np.array([float(obs_dict[k]) for k in self._obs_keys], dtype=np.float32)

    # ------------------------------------------------------------------
    # gym.Env interface
    # ------------------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._env.seed(seed)
            np.random.seed(seed)
        obs_dict, info = self._env.reset()
        self._last_obs_dict = obs_dict
        self._last_obs = self._flatten_obs(obs_dict)
        if self._shaper is not None:
            self._shaper.reset()
        return self._last_obs, info

    def step(self, action):
        rddl_action = decode_action(int(action), self._combos, self._noop)
        obs_dict, reward, terminated, truncated, info = self._env.step(rddl_action)
        self._last_obs_dict = obs_dict
        next_obs = self._flatten_obs(obs_dict)

        raw_reward = float(reward)
        if self._shaper is not None:
            reward = self._shaper.shape(
                self._last_obs, int(action), raw_reward, next_obs, info
            )
        info["raw_reward"] = raw_reward   # ← toujours présent, shapé ou non

        self._last_obs = next_obs
        return next_obs, float(reward), terminated, truncated, info


    def render(self):
        return self._env.render()

    def close(self):
        self._env.close()
