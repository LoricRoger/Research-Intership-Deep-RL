"""Factory for creating SB3 agents from config dicts."""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
from stable_baselines3 import PPO

import gymnasium as gym

_ALGO_CLASSES = {"ppo": PPO, "ppo-cp": PPO}
_POLICY_MAP   = {"ppo": "MlpPolicy", "ppo-cp": "MlpPolicy"}

_PROJECT_ROOT = Path(__file__).resolve().parents[1]


class NoOpAgent:
    """Always plays action index 0 (the noop action)."""

    def predict(self, obs, deterministic: bool = True) -> Tuple[int, None]:
        return 0, None


class RandomAgent:
    """Samples uniformly from the discrete action space."""

    def __init__(self, n_actions: int, seed: int = 0) -> None:
        self._rng = np.random.default_rng(seed)
        self._n = n_actions

    def predict(self, obs, deterministic: bool = False) -> Tuple[int, None]:
        return int(self._rng.integers(self._n)), None


def make_agent(
    env: gym.Env,
    algo: str,
    algo_kwargs: Dict[str, Any],
    seed: int = 0,
    tensorboard_log: Optional[str] = None,
    verbose: int = 1,
    domain: str = "",
    instance: str = "1",
) -> Any:
    algo = algo.lower()

    if algo == "noop":
        return NoOpAgent()
    if algo == "random":
        return RandomAgent(n_actions=env.action_space.n, seed=seed)

    return _ALGO_CLASSES[algo](
        policy=_POLICY_MAP[algo],
        env=env,
        seed=seed,
        verbose=verbose,
        tensorboard_log=tensorboard_log,
        **algo_kwargs,
    )
