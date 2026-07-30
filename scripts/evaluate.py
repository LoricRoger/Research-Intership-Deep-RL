#!/usr/bin/env python3
"""Evaluate a saved model from the results/ tree.

Usage:
    python scripts/evaluate.py --domain Navigation_MDP_ippc2011 --instance 1 --algo ppo --seed 0
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from envs.rddl_env import RDDLGymWrapper
from stable_baselines3 import PPO

_ALGO_CLASSES = {"ppo": PPO, "ppo-cp": PPO}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a saved model")
    p.add_argument("--domain",     required=True)
    p.add_argument("--instance",   required=True)
    p.add_argument("--algo",       required=True, choices=["ppo", "ppo-cp"])
    p.add_argument("--seed",       type=int, default=0)
    p.add_argument("--n-episodes", type=int, default=30)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    run_dir = (
        _PROJECT_ROOT / "results"
        / args.domain / args.instance / args.algo / f"run_{args.seed}"
    )

    eval_algo = "ppo" if args.algo == "ppo-cp" else args.algo

    eval_env = RDDLGymWrapper(
        domain=args.domain,
        instance=args.instance,
        algo=eval_algo,
        seed=args.seed + 100_000,
    )

    model_zip = run_dir / "model.zip"
    if not model_zip.exists():
        print(f"[ERROR] Model not found: {model_zip}")
        sys.exit(1)
    model = _ALGO_CLASSES[args.algo].load(str(run_dir / "model"), env=eval_env)

    episode_returns = []
    for ep in range(args.n_episodes):
        obs, _ = eval_env.reset()
        ep_return = 0.0
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = eval_env.step(action)
            ep_return += reward
            done = terminated or truncated
        episode_returns.append(ep_return)
        if (ep + 1) % 5 == 0:
            print(f"  [{ep+1}/{args.n_episodes}] return={ep_return:.2f}")

    eval_env.close()

    mean_r = float(np.mean(episode_returns))
    std_r  = float(np.std(episode_returns))
    print(f"\n[RESULT] mean={mean_r:.2f} ± {std_r:.2f} (N={args.n_episodes})")

    eval_record = {
        "domain":           args.domain,
        "instance":         args.instance,
        "algo":             args.algo,
        "seed":             args.seed,
        "eval_episodes":    args.n_episodes,
        "mean_return":      mean_r,
        "std_return":       std_r,
        "episode_returns":  episode_returns,
    }
    eval_json = run_dir / "eval.json"
    with open(eval_json, "w") as f:
        json.dump(eval_record, f, indent=2)
    print(f"[INFO] Saved → {eval_json}")


if __name__ == "__main__":
    main()
