"""Config loading with deep merge: defaults → domain override → explicit override."""

from pathlib import Path
from typing import Any, Dict, Optional

import yaml

_PROJECT_ROOT = Path(__file__).parent.parent


def load_config(
    algo: str,
    domain: Optional[str] = None,
    override_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Load and merge hyperparameter configs for a given algo.

    Priority (highest → lowest):
      1. override_path (explicit --config argument)
      2. configs/domains/<domain>.yaml
      3. configs/defaults.yaml

    Returns a dict with keys:
      algo, total_timesteps, eval_episodes, algo_kwargs (dict of SB3 kwargs)
    """
    defaults_path = _PROJECT_ROOT / "configs" / "defaults.yaml"
    with open(defaults_path) as f:
        defaults = yaml.safe_load(f)

    algo_section: Dict[str, Any] = dict(defaults.get(algo.lower(), {}))
    total_timesteps = int(algo_section.pop("total_timesteps", 500_000))

    early_stop_keys = {k: v for k, v in defaults.items() if k.startswith("early_stop_")}
    result: Dict[str, Any] = {
        "algo": algo.lower(),
        "total_timesteps": total_timesteps,
        "eval_episodes": int(defaults.get("eval_episodes", 30)),
        "algo_kwargs": algo_section,
        **early_stop_keys,
    }

    if domain is not None:
        domain_yaml = _PROJECT_ROOT / "configs" / "domains" / f"{domain}.yaml"
        if domain_yaml.exists():
            with open(domain_yaml) as f:
                domain_cfg = yaml.safe_load(f) or {}
            # Strip algo-specific sub-sections before the global merge so they
            # don't leak into unrelated algos.  Then re-apply the section that
            # matches the current algo (if present) on top.
            algo_section_override = domain_cfg.pop(algo.lower(), {})
            result = _deep_merge(result, domain_cfg)
            if algo_section_override:
                result = _deep_merge(result, algo_section_override)

    if override_path is not None:
        with open(override_path) as f:
            override_cfg = yaml.safe_load(f) or {}
        result = _deep_merge(result, override_cfg)

    return result


def _deep_merge(base: Dict, override: Dict) -> Dict:
    result = dict(base)
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = val
    return result
