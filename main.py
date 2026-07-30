# main.py
import os
import subprocess
import argparse
from pathlib import Path
from multiprocessing import Pool

DOMAINS = [
    "Navigation_MDP_ippc2011",
    "CrossingTraffic_MDP_ippc2011",
    "SysAdmin_MDP_ippc2011",
    "TriangleTireworld_MDP_ippc2014",
    "GameOfLife_MDP_ippc2011",
]
ALGOS = ["noop", "random", "ppo", "ppo-cp"]
INSTANCES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
SEEDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]


def run_job(args):
    domain, instance, algo, seed, force = args
    eval_json = Path(f"results/{domain}/{instance}/{algo}/run_{seed}/eval.json")

    if eval_json.exists() and not force:
        print(f"[SKIP] {domain} i={instance} {algo} s={seed}")
        return

    cmd = [
        "python", "scripts/train.py",
        "--domain", domain,
        "--instance", str(instance),
        "--algo", algo,
        "--seed", str(seed),
    ]
    if force:
        cmd.append("--force")

    env = {**os.environ, "SB3_PROGRESS_BAR": "1"}
    subprocess.run(cmd, check=True, env=env)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    jobs = [
        (domain, instance, algo, seed, args.force)
        for domain in DOMAINS
        for instance in INSTANCES
        for algo in ALGOS
        for seed in SEEDS
    ]

    if args.dry_run:
        for job in jobs:
            print(f"[DRY] {job}")
        return

    with Pool(args.workers) as pool:
        pool.map(run_job, jobs)

if __name__ == "__main__":
    main()
