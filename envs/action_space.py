"""
Action space utilities for converting pyRDDLGym Dict action spaces to SB3-compatible spaces.

All algos (PPO/A2C/DQN): Discrete(N+1) where N+1 = noop + N singletons.
Valid joint actions are enumerated at construction time (max_allowed=1 on all IPPC domains).
"""

from typing import Dict, List

from gymnasium import spaces


def enumerate_valid_combos(action_keys: List[str]) -> List[frozenset]:
    """Return the N+1 valid actions: noop (index 0) + one frozenset per singleton."""
    combos: List[frozenset] = [frozenset()]
    for key in action_keys:
        combos.append(frozenset([key]))
    return combos


def build_discrete_space(combos: List[frozenset]) -> spaces.Discrete:
    return spaces.Discrete(len(combos))


def decode_action(action_index: int, combos: List[frozenset], noop: Dict) -> Dict:
    """Decode an integer action index to an RDDL action dict."""
    active_keys = combos[action_index]
    return {k: (k in active_keys) for k in noop}
