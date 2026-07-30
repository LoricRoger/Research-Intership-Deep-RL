"""Reward shaping protocol and built-in implementations."""

from __future__ import annotations

import csv
import os
from typing import IO, Any, Dict, List, Optional, runtime_checkable

from envs.cp_state_encoder import get_encoder as _get_cp_encoder

import numpy as np

try:
    from typing import Protocol
except ImportError:
    from typing_extensions import Protocol  # type: ignore[assignment]


# PPO index (alphabétique) → Java action index
# PPO: 0=noop, 1=move-east, 2=move-north, 3=move-south, 4=move-west
# Java:        2=east,      0=north,       1=south,       3=west,   4=noop
_PPO_TO_JAVA_ACTION = {
    0: 4,  # noop
    1: 2,  # move-east  → java 2
    2: 0,  # move-north → java 0
    3: 1,  # move-south → java 1
    4: 3,  # move-west  → java 3
}

@runtime_checkable
class RewardShaper(Protocol):
    """Interface for reward shaping functions.

    Receives the transition (obs, action, raw_reward, next_obs, info) and
    returns the shaped reward that will be passed to the RL agent.
    """

    def shape(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        info: Dict[str, Any],
    ) -> float: ...


class IdentityShaper:
    """No-op shaper — returns reward unchanged."""

    def shape(self, obs, action, reward, next_obs, info) -> float:
        return float(reward)


class ScaledShaper:
    """Multiplies reward by a constant scale factor. Useful for sanity-testing the pipeline."""

    def __init__(self, scale: float) -> None:
        self.scale = scale

    def shape(self, obs, action, reward, next_obs, info) -> float:
        return float(reward) * self.scale

def _open_debug_csv(path: str) -> tuple:
    """Open a shaping debug CSV and return (file, writer). Creates parent dirs."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    f = open(path, "w", newline="")
    w = csv.writer(f)
    w.writerow(["episode", "step", "raw", "etr_before", "etr_after", "shaping_term", "shaped"])
    f.flush()
    return f, w


class CPShaper:
    """
    CP-guided reward shaping via ETR (Expected Total Reward) from the Java server.

        reward_shaped = env_reward + alpha * (gamma * etr_after - etr_before)

    Parameters
    ----------
    client : CPClient
        Already connected TCP client (connect() + init() already called).
    encoder : callable (obs_dict: dict) -> int
        Encodes a raw RDDL obs dict into the integer state index the CP
        server understands.
    alpha : float
        Shaping coefficient (typically 1.0–10.0).
    gamma : float
        Discount factor — must match the PPO gamma to ensure potential-based shaping.
    obs_dict_fn : callable () -> dict
        Returns the latest raw obs dict from RDDLGymWrapper._last_obs_dict.
        Called inside shape() to get the next-state dict after each step.
    debug_csv : str or None
        If provided, log every shaped step to this CSV file.

    Notes
    -----
    - reset() is called automatically by RDDLGymWrapper.reset().
    - If the CP server becomes inconsistent, _cp_broken is set to True and
      shaping is silently disabled for the rest of the episode.
    - STEP is never sent on terminal transitions to avoid CP inconsistencies.
    """

    def __init__(
            self,
            client,
            encoder,
            alpha: float,
            gamma: float,
            obs_dict_fn,
            dead_state_idx: int,
            goal_state_idx: int,
            debug_csv: Optional[str] = None,
    ) -> None:
        self.client = client
        self.encoder = encoder
        self.alpha = alpha
        self.gamma = gamma
        self.obs_dict_fn = obs_dict_fn
        self._dead_state_idx = dead_state_idx
        self._goal_state_idx = goal_state_idx

        self._etr_before: float = 0.0
        self._step_idx: int = 0
        self._cp_broken: bool = False
        self._episode_count: int = 0

        # Par épisode (réinitialisés au reset)
        self._steps_shaped_ep: int = 0
        self._steps_total_ep: int = 0

        self._debug_file: Optional[IO] = None
        self._debug_writer = None
        if debug_csv:
            self._debug_file, self._debug_writer = _open_debug_csv(debug_csv)

    def reset(self) -> None:
        """Sync the CP server at the start of each episode."""
        self._steps_shaped_ep = 0
        self._steps_total_ep = 0
        self._episode_count += 1

        self._cp_broken = False
        self._step_idx = 0
        try:
            ok = self.client.reset()
            if not ok:
                self._cp_broken = True
                return
            etr = self.client.query_etr()
            self._etr_before = etr if etr is not None else 0.0
        except Exception as exc:
            self._cp_broken = True

    def shape(
            self,
            obs: np.ndarray,
            action: int,
            reward: float,
            next_obs: np.ndarray,
            info: Dict[str, Any],
    ) -> float:
        """Return shaped reward, or raw reward if CP is broken."""
        if self._cp_broken:
            self._steps_total_ep += 1
            return float(reward)

        # Never send STEP on terminal transitions
        terminated = info.get("termination", False) or info.get("truncated", False)
        if terminated:
            return float(reward)

        self._steps_total_ep += 1

        try:
            obs_dict = self.obs_dict_fn()
            state_idx = self.encoder(obs_dict)

            # État absorbant (dead ou goal) → pas de STEP
            if state_idx >= self._dead_state_idx:
                return float(reward)

            java_action = _PPO_TO_JAVA_ACTION.get(int(action))

            ok = self.client.send_step(self._step_idx, java_action, state_idx)
            self._step_idx += 1

            if not ok:
                self._cp_broken = True
                return float(reward)

            etr_after = self.client.query_etr()
            if etr_after is None:
                return float(reward)

            self._steps_shaped_ep += 1
            shaping_term = self.alpha * (self.gamma * etr_after - self._etr_before)
            shaped = float(reward) + shaping_term
            if self._debug_writer is not None:
                self._debug_writer.writerow([
                    self._episode_count, self._step_idx - 1,
                    float(reward), self._etr_before, etr_after, shaping_term, shaped,
                ])
                self._debug_file.flush()
            self._etr_before = etr_after
            return shaped

        except Exception as exc:
            self._cp_broken = True
            return float(reward)


class SysAdminCPShaper:
    """
    CP-guided reward shaping for SysAdmin via ETR from the Java server.

        phi(s, t) = etr / (horizon - t)   [normalised potential]
        reward_shaped = env_reward + (alpha/horizon) * (gamma * etr_after - etr_before)

    Dividing by horizon keeps the shaping term on the same scale as the raw reward.

    The STEP command carries the reboot index and the observed next state:
        STEP <step_idx> <reboot_idx> <alive_bitmask>
    where reboot_idx = action_idx - 1  (0-based), or -1 for noop, and
    alive_bitmask has bit i set iff computer i (0-based) is alive after the step.

    Parameters
    ----------
    client : CPClient
        Already connected TCP client (connect() + init() already called).
    alpha : float
        Shaping coefficient.
    gamma : float
        Discount factor — must match the PPO gamma to ensure potential-based shaping.
    horizon : int
        Episode length (number of decision steps).
    n_computers : int
        Number of computers in this instance (used to build the bitmask).
    obs_dict_fn : callable () -> dict
        Returns the latest raw obs dict from RDDLGymWrapper._last_obs_dict.
    action_keys : list[str]
        Sorted RDDL action keys, e.g. ['reboot___c1','reboot___c10','reboot___c2',...].
        Used to map PPO action_index -> Java reboot_idx (0-based computer number).
    """

    def __init__(self, client, alpha: float, gamma: float, horizon: int,
                 n_computers: int, obs_dict_fn, action_keys: list,
                 debug_csv: Optional[str] = None) -> None:
        self.client       = client
        self.alpha        = alpha
        self.gamma        = gamma
        self._horizon     = horizon
        self._n           = n_computers
        self.obs_dict_fn  = obs_dict_fn

        # PPO action_index i (1-based, 0=noop) -> Java reboot_idx = computer number - 1
        # action_keys[i-1] = 'reboot___c{k}' -> Java index k-1
        self._action_to_java = {0: -1}  # noop
        for i, key in enumerate(action_keys):
            k = int(key.split("___c")[1])   # 'reboot___c10' -> 10
            self._action_to_java[i + 1] = k - 1

        self._etr_before:  float = 0.0
        self._step_idx:    int   = 0
        self._cp_broken:   bool  = False
        self._episode_count: int = 0

        self._debug_file: Optional[IO] = None
        self._debug_writer = None
        if debug_csv:
            self._debug_file, self._debug_writer = _open_debug_csv(debug_csv)

    def _alive_bitmask(self) -> int:
        """Build bitmask from current obs_dict: bit i = running___c{i+1}."""
        obs_dict = self.obs_dict_fn()
        mask = 0
        for i in range(self._n):
            if obs_dict.get(f"running___c{i + 1}", False):
                mask |= (1 << i)
        return mask

    def reset(self) -> None:
        self._cp_broken  = False
        self._step_idx   = 0
        self._episode_count += 1
        try:
            ok = self.client.reset()
            if not ok:
                self._cp_broken = True
                return
            etr = self.client.query_etr()
            self._etr_before = etr if etr is not None else 0.0
        except Exception:
            self._cp_broken = True

    def shape(
            self,
            obs:      np.ndarray,
            action:   int,
            reward:   float,
            next_obs: np.ndarray,
            info:     Dict[str, Any],
    ) -> float:
        if self._cp_broken:
            return float(reward)

        terminated = info.get("termination", False) or info.get("truncated", False)
        if terminated:
            return float(reward)

        reboot_idx  = self._action_to_java[int(action)]
        alive_mask  = self._alive_bitmask()  # next_obs already in _last_obs_dict

        t = self._step_idx  # step index before increment (0-based)
        try:
            ok = self.client.send_step_sysadmin(t, reboot_idx, alive_mask)
            self._step_idx += 1
            if not ok:
                self._cp_broken = True
                return float(reward)

            etr_after = self.client.query_etr()
            if etr_after is None:
                self._cp_broken = True
                return float(reward)

            # normalise by horizon so shaping scale ~ raw reward scale
            shaping_term = (self.alpha / self._horizon) * (self.gamma * etr_after - self._etr_before)
            shaped = float(reward) + shaping_term
            if self._debug_writer is not None:
                self._debug_writer.writerow([
                    self._episode_count, t,
                    float(reward), self._etr_before, etr_after, shaping_term, shaped,
                ])
                self._debug_file.flush()
            self._etr_before = etr_after
            return shaped

        except Exception:
            self._cp_broken = True
            return float(reward)


class NavigationCPShaper:
    """
    CP-guided reward shaping for Navigation MDP via ETR from the Java server.

        reward_shaped = env_reward + alpha * (gamma * etr_after - etr_before)

    The STEP command carries the Java action index and the next state index:
        STEP <step_idx> <java_action> <state_idx>
    where state_idx = xi * nY + yj (column-major, 0-based),
    dead_state = nX*nY, goal_state = nX*nY+1.

    Parameters
    ----------
    client : CPClient
    alpha : float
    nX, nY : int  — grid dimensions
    xpos_sorted, ypos_sorted : list[str]  — sorted position labels (ascending numeric)
    obs_dict_fn : callable () -> dict
    dead_state_idx, goal_state_idx : int
    """

    def __init__(self, client, alpha: float, gamma: float, nX: int, nY: int,
                 xpos_sorted: list, ypos_sorted: list,
                 obs_dict_fn,
                 dead_state_idx: int,
                 goal_state_idx: int,
                 debug_csv: Optional[str] = None) -> None:
        self.client          = client
        self.alpha           = alpha
        self.gamma           = gamma
        self._nX             = nX
        self._nY             = nY
        self._xpos           = xpos_sorted
        self._ypos           = ypos_sorted
        self.obs_dict_fn     = obs_dict_fn
        self._dead_state_idx = dead_state_idx
        self._goal_state_idx = goal_state_idx
        self._encoder        = _get_cp_encoder(
            "navigation",
            nX=nX, nY=nY,
            xpos_sorted=xpos_sorted, ypos_sorted=ypos_sorted,
            dead_state_idx=dead_state_idx,
        )

        self._etr_before:    float = 0.0
        self._step_idx:      int   = 0
        self._cp_broken:     bool  = False
        self._episode_count: int   = 0

        self._debug_file: Optional[IO] = None
        self._debug_writer = None
        if debug_csv:
            self._debug_file, self._debug_writer = _open_debug_csv(debug_csv)

    def _is_terminal(self, state_idx: int) -> bool:
        return state_idx == self._goal_state_idx or state_idx == self._dead_state_idx

    def _encode_state(self) -> int:
        """Encode current obs_dict → state_idx. Delegates to the registry encoder."""
        return self._encoder(self.obs_dict_fn())

    def reset(self) -> None:
        self._cp_broken = False
        self._step_idx  = 0
        self._episode_count += 1
        try:
            ok = self.client.reset()
            if not ok:
                self._cp_broken = True
                return
            etr = self.client.query_etr()
            self._etr_before = etr if etr is not None else 0.0
        except Exception:
            self._cp_broken = True

    def shape(self, obs, action: int, reward: float, next_obs, info) -> float:
        if self._cp_broken:
            return float(reward)

        terminated = info.get("termination", False) or info.get("truncated", False)
        if terminated:
            return float(reward)

        state_idx = self._encode_state()
        if self._is_terminal(state_idx):
            self._etr_before = 0.0
            return float(reward)

        java_action = _PPO_TO_JAVA_ACTION.get(int(action), 4)

        t = self._step_idx
        try:
            ok = self.client.send_step(t, java_action, state_idx)
            self._step_idx += 1
            if not ok:
                self._cp_broken = True
                return float(reward)

            etr_after = self.client.query_etr()
            if etr_after is None:
                self._cp_broken = True
                return float(reward)

            shaping_term = self.alpha * (self.gamma * etr_after - self._etr_before)
            shaped = float(reward) + shaping_term
            if self._debug_writer is not None:
                self._debug_writer.writerow([
                    self._episode_count, t,
                    float(reward), self._etr_before, etr_after, shaping_term, shaped,
                ])
                self._debug_file.flush()
            self._etr_before = etr_after
            return shaped
        except Exception:
            self._cp_broken = True
            return float(reward)


class GameOfLifeCPShaper:
    """
    CP-guided reward shaping for GameOfLife via ETR from the Java server.

        reward_shaped = env_reward + alpha * (gamma * etr_after - etr_before)

    The STEP command carries the set index and the observed next state:
        STEP <step_idx> <set_idx> <alive_bitmask>
    where set_idx = action_idx - 1  (0-based cell index), or -1 for noop, and
    alive_bitmask has bit i set iff cell i (0-based, row-major) is alive after the step.

    Cell indexing (row-major): cell i = (xi-1)*nY + (yj-1)
      for a grid with nX rows and nY columns.

    Parameters
    ----------
    client : CPClient
        Already connected TCP client (connect() + init() already called).
    alpha : float
        Shaping coefficient.
    n_cells : int
        Total number of cells (nX * nY).
    nY : int
        Number of columns (y dimension) — needed for cell indexing.
    obs_dict_fn : callable () -> dict
        Returns the latest raw obs dict from RDDLGymWrapper._last_obs_dict.
    action_keys : list[str]
        Sorted RDDL action keys, e.g. ['set___x1__y1', 'set___x1__y2', ...].
        Used to map PPO action_index -> Java set_idx (0-based cell index).
    """

    def __init__(self, client, alpha: float, gamma: float, n_cells: int, nY: int,
                 obs_dict_fn, action_keys: list,
                 debug_csv: Optional[str] = None) -> None:
        self.client      = client
        self.alpha       = alpha
        self.gamma       = gamma
        self._n          = n_cells
        self._nY         = nY
        self.obs_dict_fn = obs_dict_fn

        # PPO action_index i (1-based, 0=noop) -> Java set_idx (0-based cell index)
        # action_keys[i-1] = 'set___x{xi}__y{yj}' -> cell_idx = (xi-1)*nY + (yj-1)
        self._action_to_java = {0: -1}  # noop
        for i, key in enumerate(action_keys):
            m = key.split("___")[1]   # 'x{xi}__y{yj}'
            xi_str, yj_str = m.split("__")
            xi = int(xi_str[1:]) - 1
            yj = int(yj_str[1:]) - 1
            self._action_to_java[i + 1] = xi * nY + yj

        self._etr_before:    float = 0.0
        self._step_idx:      int   = 0
        self._cp_broken:     bool  = False
        self._episode_count: int   = 0

        self._debug_file: Optional[IO] = None
        self._debug_writer = None
        if debug_csv:
            self._debug_file, self._debug_writer = _open_debug_csv(debug_csv)

    def _alive_bitmask(self) -> int:
        """Build bitmask from current obs_dict: bit i set iff cell i is alive."""
        obs_dict = self.obs_dict_fn()
        mask = 0
        for key, val in obs_dict.items():
            if not val:
                continue
            # key format: 'alive___x{xi}__y{yj}'
            if not key.startswith("alive___"):
                continue
            m = key[len("alive___"):]   # 'x{xi}__y{yj}'
            parts = m.split("__")
            if len(parts) != 2:
                continue
            xi = int(parts[0][1:]) - 1
            yj = int(parts[1][1:]) - 1
            idx = xi * self._nY + yj
            mask |= (1 << idx)
        return mask

    def reset(self) -> None:
        self._cp_broken = False
        self._step_idx  = 0
        self._episode_count += 1
        try:
            ok = self.client.reset()
            if not ok:
                self._cp_broken = True
                return
            etr = self.client.query_etr()
            self._etr_before = etr if etr is not None else 0.0
        except Exception:
            self._cp_broken = True

    def shape(
            self,
            obs:      np.ndarray,
            action:   int,
            reward:   float,
            next_obs: np.ndarray,
            info:     Dict[str, Any],
    ) -> float:
        if self._cp_broken:
            return float(reward)

        terminated = info.get("termination", False) or info.get("truncated", False)
        if terminated:
            return float(reward)

        set_idx    = self._action_to_java[int(action)]
        alive_mask = self._alive_bitmask()

        t = self._step_idx
        try:
            response = self.client.send_receive(
                f"STEP {t} {set_idx} {alive_mask}"
            )
            self._step_idx += 1
            if not response.startswith("OK STEP"):
                self._cp_broken = True
                return float(reward)

            etr_after = self.client.query_etr()
            if etr_after is None:
                self._cp_broken = True
                return float(reward)

            shaping_term = self.alpha * (self.gamma * etr_after - self._etr_before)
            shaped = float(reward) + shaping_term
            if self._debug_writer is not None:
                self._debug_writer.writerow([
                    self._episode_count, t,
                    float(reward), self._etr_before, etr_after, shaping_term, shaped,
                ])
                self._debug_file.flush()
            self._etr_before = etr_after
            return shaped

        except Exception:
            self._cp_broken = True
            return float(reward)


class TriangleTireworldCPShaper:
    """
    CP-guided reward shaping for TriangleTireworld via ETR from the Java server.

        reward_shaped = env_reward + alpha * (gamma * etr_after - etr_before)

    Protocol:
        STEP <step_idx> <cp_action> <state_idx>
    where:
        cp_action  in {0..5}    (0=noop, 1-3=move_i, 4=loadtire, 5=changetire)
        state_idx  = 4*p + 2*tau + h

    PPO Discrete layout (action_keys sorted alpha):
        0                          → noop
        1                          → changetire               → cp 5
        2..(2+N_load-1)            → loadtire___<loc>         → cp 4  (generic in CP)
        (2+N_load)..(2+N_load+N_move-1) → move-car___<src>__<dst> → cp 1/2/3 or 0 (noop if illegal)

    The move→cp mapping is contextual: from current position p,
    successors[p][0]→cp 1, successors[p][1]→cp 2, successors[p][2]→cp 3.
    A move whose destination is not in successors[p] → cp 0 (noop).

    Parameters
    ----------
    client : CPClient
    alpha, gamma : float
    loc_names : list[str]
        Location names sorted alphabetically (same order as JSON index 0..N-1).
    successors : list[list[int]]
        successors[p] = list of successor location indices from p.
    obs_dict_fn : callable () -> dict
    action_keys : list[str]
        Sorted RDDL action keys from env._action_keys (length 43 for instance 1).
    debug_csv : str, optional
    """

    def __init__(self, client, alpha: float, gamma: float,
                 loc_names: list, successors: list,
                 obs_dict_fn,
                 action_keys: list,
                 goal_location: int = 0,
                 initial_location: int = 0,
                 spare_locations: list = None,
                 debug_csv: Optional[str] = None) -> None:
        self.client             = client
        self.alpha              = alpha
        self.gamma              = gamma
        self._loc_names         = loc_names
        self._successors        = successors
        self.obs_dict_fn        = obs_dict_fn
        self._goal_location     = goal_location
        self._initial_location  = initial_location
        self._init_spare_locs   = list(spare_locations) if spare_locations else []

        self._veh_keys = [f"vehicle-at___{loc}" for loc in loc_names]

        # loadtire keys: ppo idx 2..(2+N_load-1)
        self._loadtire_keys = [k for k in action_keys if k.startswith("loadtire___")]
        self._move_keys = [k for k in action_keys if k.startswith("move-car___")]
        self._move_offset = 2 + len(self._loadtire_keys)  # first ppo idx mapping to move-car

        self._etr_before:    float = 0.0
        self._step_idx:      int   = 0
        self._cp_broken:     bool  = False
        self._episode_count: int   = 0
        # Pre-action position, maintained across steps (needed for move→cp mapping)
        self._pos_before:    int   = initial_location
        # Spare tracking: set of location indices with a spare on the ground
        self._spare_at: set  = set(self._init_spare_locs)

        self._debug_file = None
        self._debug_writer = None
        if debug_csv:
            self._debug_file, self._debug_writer = _open_debug_csv(debug_csv)

    def _is_terminal(self, state_idx: int) -> bool:
        p   = state_idx // 4
        tau = (state_idx // 2) % 2
        h   = state_idx % 2
        # Goal reached, or flat + no spare in hand + no spare at current position
        return p == self._goal_location or (tau == 0 and h == 0 and p not in self._spare_at)

    def _pos_from_obs(self, obs_dict: dict) -> int:
        return next(
            (i for i, k in enumerate(self._veh_keys) if obs_dict.get(k)),
            0,
        )

    def _encode_state(self, obs_dict: dict) -> int:
        """obs_dict → s = 4*p + 2*tau + h"""
        p   = self._pos_from_obs(obs_dict)
        tau = 1 if obs_dict.get("not-flattire", False) else 0
        h   = 1 if obs_dict.get("hasspare",     False) else 0
        return 4 * p + 2 * tau + h

    def _ppo_to_cp(self, ppo_action: int, pos_before: int) -> int:
        """Map PPO action index → CP action {0..5}.

        pos_before : location index BEFORE the step (for move successor lookup).
        """
        if ppo_action == 0:
            return 0  # noop
        if ppo_action == 1:
            return 5  # changetire
        if 2 <= ppo_action < self._move_offset:
            # loadtire___<loc>: only picks up spare if vehicle is at <loc>
            load_key = self._loadtire_keys[ppo_action - 2]
            load_loc = load_key[len("loadtire___"):]
            try:
                load_idx = self._loc_names.index(load_loc)
            except ValueError:
                return 0
            if load_idx != pos_before:
                return 0  # RDDL does noop when loc ≠ current position
            return 4  # loadtire at current position → CP action 4

        # move-car___<src>__<dst>: only legal if src == current position
        # and dst in successors[pos_before]. RDDL does noop otherwise.
        move_key = self._move_keys[ppo_action - self._move_offset]
        after    = move_key[len("move-car___"):]  # "la1a1__la2a1"
        src_loc, dst_loc = after.split("__")
        try:
            src_idx = self._loc_names.index(src_loc)
            dst_idx = self._loc_names.index(dst_loc)
        except ValueError:
            return 0
        if src_idx != pos_before:
            return 0  # src ≠ current pos → RDDL noop → CP noop
        try:
            return self._successors[pos_before].index(dst_idx) + 1  # cp 1/2/3
        except ValueError:
            return 0  # dst not in successors → noop

    def reset(self) -> None:
        self._cp_broken  = False
        self._step_idx   = 0
        self._episode_count += 1
        self._pos_before = self._initial_location
        self._spare_at   = set(self._init_spare_locs)
        try:
            ok = self.client.reset()
            if not ok:
                self._cp_broken = True
                return
            etr = self.client.query_etr()
            self._etr_before = etr if etr is not None else 0.0
        except Exception:
            self._cp_broken = True

    def shape(self, obs, action: int, reward: float, next_obs, info) -> float:
        if self._cp_broken:
            return float(reward)

        terminated = info.get("termination", False) or info.get("truncated", False)
        if terminated:
            return float(reward)

        # obs_dict_fn() already holds the next state (rddl_env updates before calling shape)
        next_obs_dict = self.obs_dict_fn()
        cp_action = self._ppo_to_cp(int(action), self._pos_before)
        state_idx = self._encode_state(next_obs_dict)
        # Consume spare at ground if loadtire was effective
        if cp_action == 4 and self._pos_before in self._spare_at:
            self._spare_at.discard(self._pos_before)
        # Update pre-action position for the next step (before any early return)
        self._pos_before = self._pos_from_obs(next_obs_dict)

        if self._is_terminal(state_idx):
            self._etr_before = 0.0
            return float(reward)

        t = self._step_idx
        try:
            ok = self.client.send_step(t, cp_action, state_idx)
            self._step_idx += 1
            if not ok:
                self._cp_broken = True
                return float(reward)

            etr_after = self.client.query_etr()
            if etr_after is None:
                self._cp_broken = True
                return float(reward)

            shaping_term = self.alpha * (self.gamma * etr_after - self._etr_before)
            shaped = float(reward) + shaping_term
            if self._debug_writer is not None:
                self._debug_writer.writerow([
                    self._episode_count, t,
                    float(reward), self._etr_before, etr_after, shaping_term, shaped,
                ])
                self._debug_file.flush()
            self._etr_before = etr_after
            return shaped

        except Exception:
            self._cp_broken = True
            return float(reward)


class TrafficCrossingCPShaper:
    """
    CP-guided reward shaping for CrossingTraffic via ETR from the costRegular
    Java server (TrafficCrossingService, default port 12346).

        reward_shaped = env_reward + alpha * (gamma * etr_after - etr_before)

    Protocol (4 tokens after STEP):
        STEP <step_idx> <move> <robot_idx> <obs_mask>
    where:
        move      ∈ {0..4}  Java action index (NORTH=0, SOUTH=1, EAST=2, WEST=3, NOOP=4)
        robot_idx ∈ [-1, W*H)
                  = idx(x, y) = x + W*y  if robot alive at (x, y)
                  = -1                    if robot has disappeared (DEAD)
        obs_mask  = bitmask of W*H bits, bit i = 1 iff cell idx(x, y) has an obstacle
                    (i.e. obstacle-at___x{x+1}__y{y+1} is True)

    Cell indexing MUST match TrafficCrossingDecomposition.idx(x, y, W) = x + W*y,
    with y growing UPWARD (y=0 bottom, y=H-1 top), matching the RDDL convention
    where NORTH increases yj. Verified factually on IPPC2011 instances 1..10.

    Init/goal conventions (IPPC 2011/2014, verified on instances 1..10):
        robot starts at (W-1, 0)   bottom-right, on the safe bottom row
        goal              (W-1, H-1) top-right,    on the safe top row
    These are set on the Java side; this shaper only encodes live observations.

    TODO: end-to-end bitmask correspondence test — place a single obstacle at an
    asymmetric (x, y), verify the bit Python sets equals what Java reads as O[k][i]
    with i = idx(x, y). Catches row-major/column-major or origin shifts that pass
    silently otherwise.

    Parameters
    ----------
    client : CPClient
        Already connected (connect() + init() done).
    alpha, gamma : float
        Potential-based shaping coefficients.
    nX, nY : int
        Grid dimensions (W, H).
    obs_dict_fn : callable () -> dict
        Returns the latest raw RDDL obs dict.
    debug_csv : str, optional
        Path for per-step shaping log.
    """

    def __init__(self, client, alpha: float, gamma: float, nX: int, nY: int,
                 obs_dict_fn, debug_csv: Optional[str] = None) -> None:
        self.client      = client
        self.alpha       = alpha
        self.gamma       = gamma
        self._W          = nX
        self._H          = nY
        self.obs_dict_fn = obs_dict_fn

        # goal cell in Java indexing: top-right corner (W-1, H-1) → (W-1) + W*(H-1)
        self._goal_robot_idx: int = (nX - 1) + nX * (nY - 1)

        self._etr_before:    float = 0.0
        self._step_idx:      int   = 0
        self._cp_broken:     bool  = False
        self._episode_count: int   = 0

        self._debug_file: Optional[IO] = None
        self._debug_writer = None
        if debug_csv:
            self._debug_file, self._debug_writer = _open_debug_csv(debug_csv)

    # --- Encoding helpers ---------------------------------------------------
    # MUST mirror TrafficCrossingDecomposition.idx(x, y, W) = x + W*y.
    # Same y orientation: y=0 bottom, y grows upward (NORTH = y+1).
    # RDDL keys are 1-based: x{xi+1}__y{yj+1} ↔ Python (xi, yj) ↔ Java (x, y).
    def _cell_bit(self, x: int, y: int) -> int:
        return x + self._W * y

    def _encode_robot_idx(self) -> int:
        """Return idx(x, y) for the live robot, or -1 if it has disappeared."""
        obs_dict = self.obs_dict_fn()
        for xi in range(self._W):
            for yj in range(self._H):
                if obs_dict.get(f"robot-at___x{xi + 1}__y{yj + 1}", False):
                    return self._cell_bit(xi, yj)
        return -1

    def _is_terminal(self, robot_idx: int) -> bool:
        """True if the robot is dead (-1) or has reached the goal cell."""
        return robot_idx == -1 or robot_idx == self._goal_robot_idx

    def _encode_obs_mask(self) -> int:
        """Bitmask over all W*H cells; bit idx(x,y) = obstacle-at(x+1, y+1)."""
        obs_dict = self.obs_dict_fn()
        mask = 0
        for xi in range(self._W):
            for yj in range(self._H):
                if obs_dict.get(f"obstacle-at___x{xi + 1}__y{yj + 1}", False):
                    mask |= (1 << self._cell_bit(xi, yj))
        return mask

    # --- Lifecycle ----------------------------------------------------------
    def reset(self) -> None:
        self._cp_broken = False
        self._step_idx  = 0
        self._episode_count += 1
        try:
            ok = self.client.reset()
            if not ok:
                self._cp_broken = True
                return
            etr = self.client.query_etr()
            self._etr_before = etr if etr is not None else 0.0
        except Exception:
            self._cp_broken = True

    def shape(
            self,
            obs:      np.ndarray,
            action:   int,
            reward:   float,
            next_obs: np.ndarray,
            info:     Dict[str, Any],
    ) -> float:
        if self._cp_broken:
            return float(reward)

        terminated = info.get("termination", False) or info.get("truncated", False)
        if terminated:
            return float(reward)

        java_move = _PPO_TO_JAVA_ACTION.get(int(action), 4)
        robot_idx = self._encode_robot_idx()
        obs_mask  = self._encode_obs_mask()

        if self._is_terminal(robot_idx):
            self._etr_before = 0.0
            return float(reward)

        t = self._step_idx
        try:
            response = self.client.send_receive(
                f"STEP {t} {java_move} {robot_idx} {obs_mask}"
            )
            self._step_idx += 1
            if not response.startswith("OK STEP"):
                self._cp_broken = True
                return float(reward)

            etr_after = self.client.query_etr()
            if etr_after is None:
                self._cp_broken = True
                return float(reward)

            shaping_term = self.alpha * (self.gamma * etr_after - self._etr_before)
            shaped = float(reward) + shaping_term
            if self._debug_writer is not None:
                self._debug_writer.writerow([
                    self._episode_count, t,
                    float(reward), self._etr_before, etr_after, shaping_term, shaped,
                ])
                self._debug_file.flush()
            self._etr_before = etr_after
            return shaped

        except Exception:
            self._cp_broken = True
            return float(reward)
