#!/usr/bin/env python3
"""Plot smoothed learning curves from metrics.csv files.

Usage:
    python scripts/plot_learning_curves.py --domain CrossingTraffic_MDP_ippc2011 --instance 1
    python scripts/plot_learning_curves.py --domain CrossingTraffic_MDP_ippc2011 --instance 1 --window 100
    python scripts/plot_learning_curves.py --domain CrossingTraffic_MDP_ippc2011 --instance 1 --metric raw
"""

import matplotlib
matplotlib.use('Agg')

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

ALGOS = ["ppo", "ppo-cp"]
ALGOS_COMPARE = ["ppo", "a2c", "dqn"]

ALGO_LABELS = {
    "ppo":    "PPO",
    "ppo-cp": "PPO-CP",
    "a2c":    "A2C",
    "dqn":    "DQN",
}

ALGO_COLORS = {
    "ppo":    "#7B1FA2",
    "ppo-cp": "#CE93D8",
    "a2c":    "#F57C00",
    "dqn":    "#00897B",
}

# Colonnes disponibles dans metrics.csv
_METRIC_COL = {
    "shaped": "episode_return",      # reward vu par l'agent (shapé si CPShaper)
    "raw":    "episode_return_raw",  # reward brut environnement
}


def load_runs(results_dir: Path, domain: str, instance: str) -> dict:
    """Load all metrics.csv for a given domain/instance.

    Returns: algo -> list of DataFrames (one per seed/run)
    """
    base = results_dir / domain / str(instance)
    runs: dict = {}
    if not base.exists():
        return runs
    for algo_dir in sorted(base.iterdir()):
        if not algo_dir.is_dir():
            continue
        algo = algo_dir.name
        dfs = []
        for run_dir in sorted(algo_dir.iterdir()):
            csv_path = run_dir / "metrics.csv"
            if csv_path.exists():
                dfs.append(pd.read_csv(csv_path))
        if dfs:
            runs[algo] = dfs
    return runs


def _get_metric_col(df: pd.DataFrame, metric: str) -> np.ndarray:
    """Return the requested metric column, falling back to episode_return."""
    col = _METRIC_COL[metric]
    if col in df.columns:
        return df[col].values
    # Vieux CSV sans episode_return_raw → fallback silencieux
    return df["episode_return"].values


def smooth(series: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return series
    kernel = np.ones(window) / window
    padded = np.pad(series, (window - 1, 0), mode='edge')
    return np.convolve(padded, kernel, mode='valid')


def plot_learning_curves(
        domain: str,
        instance: str,
        runs: dict,
        window: int,
        output_dir: Path,
        metric: str = "raw",
        algos: list = ALGOS,
        max_timesteps: int | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    seed_counts: dict = {}

    for algo, dfs in runs.items():
        if algo not in algos:
            continue
        color = ALGO_COLORS.get(algo, "gray")

        all_timesteps = np.concatenate([df["timestep"].values for df in dfs])
        t_min = all_timesteps.min()
        t_max = all_timesteps.max()
        if max_timesteps is not None:
            t_max = min(t_max, max_timesteps)
        grid = np.linspace(t_min, t_max, 500)

        interp_curves = [
            np.interp(
                grid,
                df["timestep"].values,
                smooth(_get_metric_col(df, metric), window),
            )
            for df in dfs
        ]
        mean_curve = np.mean(interp_curves, axis=0)
        seed_counts[algo] = len(dfs)

        label = ALGO_LABELS.get(algo, algo.upper())
        ax.plot(grid, mean_curve, color=color, linewidth=2, label=label)

    domain_label = domain.split("_")[0]

    ax.set_xlabel("Timestep", fontsize=12)
    ax.set_ylabel("Episode Return", fontsize=12)
    ax.set_title(
        f"Learning Curves — {domain_label} — Instance {instance}",
        fontsize=11, fontweight="bold",
    )
    ax.legend(fontsize=10, loc="lower right")
    ax.yaxis.get_major_formatter().set_useOffset(False)
    ax.grid(alpha=0.3)
    plt.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = "raw" if metric == "raw" else "shaped"
    save_path = output_dir / f"{domain}_inst{instance}_curves_{suffix}.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Saved → {save_path}")


def main() -> None:
    p = argparse.ArgumentParser(description="Plot smoothed learning curves")
    p.add_argument("--domain",   default=None, help="Single domain (default: all)")
    p.add_argument("--instance", default=None, help="Single instance (default: all)")
    p.add_argument("--window",   type=int, default=50, help="Smoothing window (episodes)")
    p.add_argument("--metric",   choices=["raw", "shaped"], default="raw",
                   help="raw = env reward (comparable entre algos), "
                        "shaped = reward vu par l'agent pendant l'entraînement")
    p.add_argument("--compare", action="store_true",
                   help="Compare PPO vs A2C vs DQN (instead of PPO vs PPO-CP)")
    p.add_argument("--max-timesteps", type=int, default=None,
                   help="Clip curves at this timestep (e.g. 1500000)")
    p.add_argument("--results-dir", default=str(_PROJECT_ROOT / "results"))
    p.add_argument("--output",      default=str(_PROJECT_ROOT / "results" / "plots"))
    args = p.parse_args()

    results_dir = Path(args.results_dir)
    base_output = Path(args.output)
    algos       = ALGOS_COMPARE if args.compare else ALGOS

    # Collecter les paires (domain, instance) à plotter
    pairs = []
    if args.domain and args.instance:
        pairs = [(args.domain, args.instance)]
    else:
        for domain_dir in sorted(results_dir.iterdir()):
            if not domain_dir.is_dir() or domain_dir.name == "plots":
                continue
            if args.domain and domain_dir.name != args.domain:
                continue
            for inst_dir in sorted(domain_dir.iterdir()):
                if not inst_dir.is_dir():
                    continue
                if args.instance and inst_dir.name != args.instance:
                    continue
                if any(inst_dir.rglob("metrics.csv")):
                    pairs.append((domain_dir.name, inst_dir.name))

    if not pairs:
        print("[ERROR] No metrics.csv found for the given filters.")
        sys.exit(1)

    for domain, instance in pairs:
        runs = load_runs(results_dir, domain, instance)
        if not runs:
            continue
        output_dir = base_output / domain / "learning_curves"
        for algo, dfs in runs.items():
            print(f"  {domain} inst={instance} {algo}: {len(dfs)} seed(s)")
        plot_learning_curves(domain, instance, runs, args.window, output_dir, args.metric, algos=algos, max_timesteps=args.max_timesteps)


if __name__ == "__main__":
    main()