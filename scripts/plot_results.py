#!/usr/bin/env python3
"""Produce grouped-bar charts (one bar per algo, grouped by instance) for each domain.

Usage:
    python scripts/plot_results.py
    python scripts/plot_results.py --domains CrossingTraffic_MDP_ippc2011
    python scripts/plot_results.py --output results/plots/
    python scripts/plot_results.py --all-algos
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

# Ordered list of algorithms to display (presence is optional — missing ones are skipped).
ALGOS = ["GurobiPlan", "JaxPlan", "ppo", "ppo-cp", "noop", "random"]
ALGOS_ALL = ["ppo", "a2c", "dqn", "noop", "random"]

ALGO_LABELS = {
    "GurobiPlan": "GurobiPlan",
    "JaxPlan":    "JaxPlan",
    "ppo":        "PPO",
    "ppo-cp":     "PPO-CP",
    "a2c":        "A2C",
    "dqn":        "DQN",
    "noop":       "NoOp",
    "random":     "Random",
}

ALGO_COLORS = {
    "GurobiPlan": "#E53935",
    "JaxPlan":    "#1E88E5",
    "ppo":        "#7B1FA2",
    "ppo-cp":     "#CE93D8",
    "a2c":        "#F57C00",
    "dqn":        "#00897B",
    "noop":       "#9E9E9E",
    "random":     "#FDD835",
}


_REFERENCE_JSON = _PROJECT_ROOT / "results" / "reference_values.json"


def load_all_results(results_dir: Path) -> Dict:
    """Walk results/ tree and collect eval.json entries, then merge reference_values.json.

    Returns: domain → instance → algo → list[episode_returns]
    All episode returns from every seed/run are pooled per (domain, instance, algo).
    Reference values (GurobiPlan, JaxPlan) are injected as [mean] when mean is not null.
    """
    data: Dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for eval_json in results_dir.rglob("eval.json"):
        try:
            with open(eval_json) as f:
                rec = json.load(f)
            domain   = rec["domain"]
            instance = str(rec["instance"])
            algo     = rec["algo"]
            episodes = rec.get("episode_returns")
            if episodes:
                data[domain][instance][algo].extend(episodes)
            else:
                data[domain][instance][algo].append(rec["mean_return"])
        except (KeyError, json.JSONDecodeError) as e:
            print(f"[WARN] Could not parse {eval_json}: {e}")

    if _REFERENCE_JSON.exists():
        with open(_REFERENCE_JSON) as f:
            refs = json.load(f)
        for domain, inst_map in refs.items():
            if domain == "_comment":
                continue
            for instance, algo_map in inst_map.items():
                for algo, vals in algo_map.items():
                    mean = vals.get("mean")
                    if mean is not None:
                        data[domain][instance][algo].append(float(mean))
    else:
        print(f"[INFO] No reference file found at {_REFERENCE_JSON} — skipping GurobiPlan/JaxPlan.")

    return data


def _instance_stats(
    data: Dict, domain: str
) -> Dict[str, Dict[str, Optional[tuple]]]:
    """Return {algo: {instance: (mean, std)}} for all available (algo, instance) pairs."""
    stats: Dict[str, Dict[str, Optional[tuple]]] = defaultdict(dict)
    for instance, instance_data in data[domain].items():
        for algo, returns in instance_data.items():
            arr = np.array(returns, dtype=float)
            stats[algo][instance] = (float(arr.mean()), float(arr.std()))
    return dict(stats)


def plot_domain_by_instance(
    domain: str,
    stats: Dict[str, Dict[str, Optional[tuple]]],
    output_dir: Path,
    algos: List[str] = ALGOS,
) -> None:
    # Collect all instances present in the data, sorted numerically
    all_instances: List[str] = sorted(
        {inst for algo_stats in stats.values() for inst in algo_stats},
        key=lambda x: int(x) if x.isdigit() else x,
    )
    if not all_instances:
        print(f"[WARN] No data for {domain}, skipping.")
        return

    if not any(a in stats and stats[a] for a in algos):
        print(f"[WARN] No known algo data for {domain}, skipping.")
        return

    n_instances = len(all_instances)
    n_algos     = len(algos)
    bar_width   = 0.8 / n_algos
    x           = np.arange(n_instances)

    fig, ax = plt.subplots(figsize=(max(8, n_instances * 1.2), 5))
    ax.set_facecolor("#f0f0f0")
    fig.patch.set_facecolor("white")

    # First pass: collect missing positions to draw placeholders after ylim is set
    missing_positions: List[float] = []

    for i, algo in enumerate(algos):
        offset = (i - n_algos / 2 + 0.5) * bar_width
        algo_stats = stats.get(algo, {})
        means, errs = [], []
        for inst in all_instances:
            pair = algo_stats.get(inst)
            if pair is not None:
                means.append(pair[0])
                errs.append(pair[1])
            else:
                means.append(np.nan)
                errs.append(0.0)
                missing_positions.append(x[all_instances.index(inst)] + offset)

        ax.bar(
            x + offset,
            means,
            bar_width,
            yerr=errs,
            label=ALGO_LABELS[algo],
            color=ALGO_COLORS[algo],
            error_kw=dict(ecolor="black", capsize=3, linewidth=1),
            alpha=0.9,
        )

    # Second pass: draw white hatched placeholders now that ylim is stable
    if missing_positions:
        ax.autoscale_view()
        ylo, yhi = ax.get_ylim()
        span = yhi - ylo if yhi != ylo else 1.0
        ph_bottom = ylo
        ph_height = span * 0.06
        for xpos in missing_positions:
            ax.bar(
                xpos,
                ph_height,
                bar_width,
                bottom=ph_bottom,
                color="white",
                edgecolor="#999999",
                linewidth=0.8,
                hatch="//",
                zorder=3,
            )

    ax.set_xlabel("Instance", fontsize=12)
    ax.set_ylabel("Return", fontsize=12)
    ax.set_title(domain.replace("_MDP_ippc", " — IPPC ").replace("_", " "), fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(i):02d}" if i.isdigit() else i for i in all_instances], fontsize=10)
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.grid(axis="y", color="white", linewidth=1.0)
    ax.grid(axis="y", which="minor", color="white", linewidth=0.5, alpha=0.5)
    ax.legend(fontsize=9, framealpha=0.9, loc="best")

    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / f"{domain}.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Saved → {save_path}")


def main() -> None:
    p = argparse.ArgumentParser(description="Plot RL baseline results grouped by instance")
    p.add_argument("--results-dir", default=str(_PROJECT_ROOT / "results"))
    p.add_argument("--output", default=str(_PROJECT_ROOT / "results" / "plots"))
    p.add_argument("--domains", nargs="+", default=None)
    p.add_argument("--all-algos", action="store_true",
                   help="Include A2C and DQN in addition to the default algo set")
    args = p.parse_args()

    results_dir = Path(args.results_dir)
    output_dir  = Path(args.output)
    algos       = ALGOS_ALL if args.all_algos else ALGOS

    data    = load_all_results(results_dir)
    domains = args.domains if args.domains else sorted(data.keys())

    for domain in domains:
        if domain not in data:
            print(f"[WARN] No results for {domain}")
            continue
        stats = _instance_stats(data, domain)
        for algo, inst_stats in stats.items():
            for inst, (m, s) in inst_stats.items():
                print(f"  {domain}  inst={inst}  {algo}: mean={m:.2f} ± {s:.2f}")
        plot_domain_by_instance(domain, stats, output_dir, algos=algos)

    print(f"\n[INFO] All plots saved to {output_dir}")


if __name__ == "__main__":
    main()
