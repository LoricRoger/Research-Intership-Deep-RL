"""SB3 callbacks: CSV metrics logger and early stopping on reward plateau."""

import csv
import os
from collections import deque
from typing import IO, Optional, Tuple

from stable_baselines3.common.callbacks import BaseCallback


# ------------------------------------------------------------------
# Shared CSV helpers (used by both CSVMetricsCallback and the
# manual Q-learning training loop)
# ------------------------------------------------------------------

_CSV_HEADER = ["timestep", "episode", "episode_return", "episode_return_raw", "episode_length"]


def open_metrics_csv(csv_path: str) -> Tuple[IO, "csv.writer"]:
    """Open a metrics CSV, write the header, and return (file, writer)."""
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    f = open(csv_path, "w", newline="")
    w = csv.writer(f)
    w.writerow(_CSV_HEADER)
    f.flush()
    return f, w


def write_episode_row(
    writer,
    f: IO,
    timestep: int,
    episode: int,
    episode_return: float,
    episode_return_raw: float,
    episode_length: int,
) -> None:
    """Write one episode row to an open metrics CSV."""
    writer.writerow([timestep, episode, episode_return, episode_return_raw, episode_length])
    f.flush()


class CSVMetricsCallback(BaseCallback):
    """Write episode return and length to CSV after each episode.

    Columns: timestep, episode, episode_return, episode_return_raw, episode_length

    - episode_return     : reward vu par l'agent (shapé si CPShaper actif)
    - episode_return_raw : reward brut environnement (toujours comparable entre algos)

    Rewards are accumulated manually so no Monitor wrapper is needed.
    """

    def __init__(self, csv_path: str, verbose: int = 0) -> None:
        super().__init__(verbose)
        self.csv_path = csv_path
        self._episode_count = 0
        self._current_rewards: list = []
        self._current_raw_rewards: list = []
        self._file: Optional[IO] = None
        self._writer = None

    def _on_training_start(self) -> None:
        os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)
        self._file = open(self.csv_path, "w", newline="")
        self._writer = csv.writer(self._file)
        self._writer.writerow([
            "timestep", "episode",
            "episode_return", "episode_return_raw",
            "episode_length",
        ])
        self._file.flush()

    def _on_step(self) -> bool:
        # Reward shapé (ce que SB3 optimise)
        reward = float(self.locals.get("rewards", [0.0])[0])
        self._current_rewards.append(reward)

        # Reward brut depuis info (injecté par RDDLGymWrapper.step)
        infos = self.locals.get("infos", [{}])
        info = infos[0] if infos else {}
        # Fallback sur reward shapé si raw_reward absent (env sans shaping)
        raw_reward = float(info.get("raw_reward", reward))
        self._current_raw_rewards.append(raw_reward)

        dones = self.locals.get("dones", [False])
        done = bool(dones[0]) if hasattr(dones, "__getitem__") else bool(dones)

        if done:
            ep_return     = sum(self._current_rewards)
            ep_return_raw = sum(self._current_raw_rewards)
            ep_length     = len(self._current_rewards)
            self._episode_count += 1

            assert self._writer is not None
            self._writer.writerow([
                self.num_timesteps,
                self._episode_count,
                ep_return,
                ep_return_raw,
                ep_length,
            ])
            assert self._file is not None
            self._file.flush()

            self._current_rewards = []
            self._current_raw_rewards = []

            if self.verbose >= 1:
                raw_str = (
                    f" raw={ep_return_raw:.2f}"
                    if ep_return_raw != ep_return else ""
                )
                print(
                    f"[Ep {self._episode_count}] t={self.num_timesteps} "
                    f"return={ep_return:.2f}{raw_str} len={ep_length}"
                )
        return True

    def _on_training_end(self) -> None:
        if self._file is not None:
            self._file.close()


class EarlyStoppingCallback(BaseCallback):
    """Stop training when the mean episode return stops improving.

    Computes a rolling mean over `window` episodes and checks every
    `check_freq` episodes whether the mean improved by at least
    `min_delta` (relative) compared to the best seen so far. If not,
    increments a patience counter and stops when it reaches `patience`.

    Args:
        window: number of episodes in the rolling mean.
        patience: number of consecutive checks without improvement before stopping.
        min_delta: minimum relative improvement required to reset patience.
        check_freq: how many episodes between each plateau check.
        verbose: 0 = silent, 1 = log on stop.
    """

    def __init__(
        self,
        window: int = 50,
        patience: int = 10,
        min_delta: float = 0.01,
        check_freq: int = 50,
        min_timesteps: int = 500_000,
        verbose: int = 1,
    ) -> None:
        super().__init__(verbose)
        self._window = window
        self._patience = patience
        self._min_delta = min_delta
        self._check_freq = check_freq
        self._min_timesteps = min_timesteps
        self._returns: deque = deque(maxlen=window)
        self._current_rewards: list = []
        self._episode_count = 0
        self._checks_without_improvement = 0
        self._best_mean: Optional[float] = None

    def _on_step(self) -> bool:
        reward = float(self.locals.get("rewards", [0.0])[0])
        self._current_rewards.append(reward)

        dones = self.locals.get("dones", [False])
        done = bool(dones[0]) if hasattr(dones, "__getitem__") else bool(dones)

        if not done:
            return True

        ep_return = sum(self._current_rewards)
        self._current_rewards = []
        self._returns.append(ep_return)
        self._episode_count += 1

        if self._episode_count % self._check_freq != 0:
            return True

        if self.num_timesteps < self._min_timesteps:
            return True

        if len(self._returns) < self._window:
            return True

        mean = sum(self._returns) / len(self._returns)

        if self._best_mean is None:
            self._best_mean = mean
            return True

        # relative improvement: positive rewards → higher is better,
        # negative rewards → less negative is better (same formula works)
        ref = abs(self._best_mean) if self._best_mean != 0 else 1.0
        improved = (mean - self._best_mean) / ref >= self._min_delta

        if improved:
            self._best_mean = mean
            self._checks_without_improvement = 0
        else:
            self._checks_without_improvement += 1

        if self._checks_without_improvement >= self._patience:
            if self.verbose >= 1:
                print(
                    f"[EarlyStopping] Plateau detected at t={self.num_timesteps} "
                    f"(mean={mean:.2f} over last {self._window} eps, "
                    f"best={self._best_mean:.2f}, "
                    f"patience={self._patience} checks exhausted)"
                )
            return False

        return True
