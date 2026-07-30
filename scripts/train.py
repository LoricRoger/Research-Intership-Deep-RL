#!/usr/bin/env python3
"""Train an RL agent on an IPPC domain using Stable-Baselines3.

Usage:
    python scripts/train.py --domain Navigation_MDP_ippc2011 --instance 1 --algo ppo --seed 0
    python scripts/train.py --domain ... --force        # re-train even if eval.json exists
    python scripts/train.py --domain ... --config path/to/overrides.yaml

    # PPO with CP-guided reward shaping (requires Java server via MiniCPBP)
    python scripts/train.py --domain CrossingTraffic_MDP_ippc2011 --instance 1 --algo ppo-cp --seed 0
"""

# Must be set before any matplotlib or pyRDDLGym import to avoid GUI backend
# initialization errors in multiprocessing workers.
import matplotlib

matplotlib.use('Agg')

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import yaml

_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from stable_baselines3.common.callbacks import CallbackList

from agents.factory import make_agent
from envs.rddl_env import RDDLGymWrapper
from utils.config import load_config
from utils.logging import CSVMetricsCallback, EarlyStoppingCallback


# ------------------------------------------------------------------
# Argparse
# ------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train RL agent on IPPC domain")
    p.add_argument("--domain",    required=True)
    p.add_argument("--instance",  required=True)
    p.add_argument("--algo",      required=True,
                   choices=["ppo", "ppo-cp", "noop", "random"])
    p.add_argument("--seed",      type=int, default=0)
    p.add_argument("--force",     action="store_true",
                   help="Re-train even if eval.json exists")
    p.add_argument("--config",    default=None,
                   help="Path to YAML override file")
    p.add_argument("--verbose",   type=int, default=0)
    p.add_argument("--debug-shaping", action="store_true",
                   help="Write per-step ETR shaping values to shaping_debug.csv in the run dir")
    return p.parse_args()


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def get_run_dir(domain: str, instance: str, algo: str, seed: int) -> Path:
    return _PROJECT_ROOT / "results" / domain / instance / algo / f"run_{seed}"


def _evaluate(model, domain: str, instance: str, algo: str,
              seed: int, n_episodes: int) -> list:
    """Run n_episodes evaluation episodes and return list of episode returns.

    Evaluation always uses a plain env (no CP shaping) so results are
    comparable across all algos.
    """
    eval_algo = "ppo" if algo == "ppo-cp" else algo
    eval_env = RDDLGymWrapper(domain, instance, algo=eval_algo, seed=seed + 100_000)

    returns = []
    for _ in range(n_episodes):
        obs, _ = eval_env.reset()
        ep_return = 0.0
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = eval_env.step(action)
            ep_return += reward
            done = terminated or truncated
        returns.append(ep_return)
    eval_env.close()
    return returns


# ------------------------------------------------------------------
# CP server + shaper setup
# ------------------------------------------------------------------

def _start_cp_server(port: int, bp_iter: int, java_project_dir: Path) -> subprocess.Popen:
    """Launch TrafficCrossingLineDecompositionService via mvn exec:java.

    Blocks until the port is open (max 60 s) then returns the Popen handle.
    Caller is responsible for calling proc.terminate() when done.
    """
    cmd = [
        "mvn", "-f", str(java_project_dir / "pom.xml"),
        "--offline",
        "exec:java",
        "-Dexec.cleanupDaemonThreads=false",
        "-Dexec.mainClass=minicpbp.examples.TrafficCrossingLineDecompositionService",
        f"-Dexec.args={port}",
        f"-Dexec.jvmArgs=-Xmx8g",
        f"-Dbp.iterations={bp_iter}",
    ]
    proc = subprocess.Popen(
        cmd,
        cwd=str(java_project_dir),
        stdout=open(_PROJECT_ROOT / "java_cp_stdout.log", "w"),
        stderr=open(_PROJECT_ROOT / "java_cp_stderr.log", "w"),
    )
    print(f"[CP] Starting Java server on port {port} (bp_iter={bp_iter}) ...")

    deadline = time.time() + 60
    while time.time() < deadline:
        try:
            with socket.create_connection(("localhost", port), timeout=1):
                print(f"[CP] Server ready on port {port}")
                return proc
        except OSError:
            time.sleep(0.5)

    proc.terminate()
    raise RuntimeError(
        f"Java CP server did not start on port {port} within 60 s."
    )


def _build_cp_env(domain: str, instance: str, seed: int, cp_cfg: dict) -> tuple:
    """Instantiate RDDLGymWrapper + CPShaper for ppo-cp (CrossingTraffic line decomposition)."""
    debug_csv = cp_cfg.get("debug_csv")
    from cp_client import CPClient
    from envs.cp_state_encoder import get_encoder
    from envs.shaping import CPShaper

    port         = cp_cfg.get("port", 12347)
    bp_iter      = cp_cfg.get("bp_iter", 100)
    alpha        = cp_cfg.get("alpha", 1.0)
    gamma        = cp_cfg.get("gamma", 1.0)
    java_dir     = _PROJECT_ROOT / cp_cfg.get("java_project_dir", "MiniCPBP")

    cp_instances_path = _PROJECT_ROOT / "cp_instances" / f"{domain.lower()}.json"
    if not cp_instances_path.exists():
        raise FileNotFoundError(
            f"CP instance metadata not found: {cp_instances_path}\n"
            f"Create cp_instances/{domain.lower()}.json with grid_x/grid_y per instance."
        )
    with open(cp_instances_path) as f:
        cp_meta = json.load(f)

    inst_key = str(instance)
    if inst_key not in cp_meta:
        raise KeyError(
            f"Instance '{instance}' not found in {cp_instances_path}. "
            f"Available: {list(cp_meta)}"
        )
    inst_meta = cp_meta[inst_key]
    nX = inst_meta["grid_x"]
    nY = inst_meta["grid_y"]

    cp_proc = _start_cp_server(port, bp_iter, java_dir)

    env = RDDLGymWrapper(domain=domain, instance=instance, algo="ppo-cp", seed=seed)

    client = CPClient(host="localhost", port=port)
    if not client.connect():
        cp_proc.terminate()
        raise RuntimeError("CPClient failed to connect to the Java server.")
    if not client.init(inst_key):
        client.close()
        cp_proc.terminate()
        raise RuntimeError(f"CPClient INIT failed for instance {instance}.")

    encoder = get_encoder(domain, nX=nX, nY=nY)
    shaper = CPShaper(
        client=client,
        encoder=encoder,
        alpha=alpha,
        gamma=gamma,
        obs_dict_fn=lambda: env._last_obs_dict,
        dead_state_idx=inst_meta["dead_state_idx"],
        goal_state_idx=inst_meta["goal_state_idx"],
        debug_csv=debug_csv,
    )
    env._shaper = shaper

    return env, client, cp_proc


def _build_cp_env_gameoflife(domain: str, instance: str, seed: int, cp_cfg: dict) -> tuple:
    """Instantiate RDDLGymWrapper + GameOfLifeCPShaper for ppo-cp on GameOfLife."""
    from cp_client import CPClient
    from envs.shaping import GameOfLifeCPShaper

    debug_csv = cp_cfg.get("debug_csv")
    port     = cp_cfg.get("port", 12349)
    bp_iter  = cp_cfg.get("bp_iter", 3)
    alpha    = cp_cfg.get("alpha", 1.0)
    gamma    = cp_cfg.get("gamma", 1.0)
    java_dir = _PROJECT_ROOT / cp_cfg.get("java_project_dir", "MiniCPBP")

    cmd = [
        "mvn", "-f", str(java_dir / "pom.xml"),
        "--offline",
        "exec:java",
        "-Dexec.cleanupDaemonThreads=false",
        "-Dexec.mainClass=minicpbp.examples.GameOfLifeService",
        f"-Dexec.args={port}",
        f"-Dexec.jvmArgs=-Xmx8g",
        f"-Dbp.iterations={bp_iter}",
    ]
    proc = subprocess.Popen(
        cmd,
        cwd=str(java_dir),
        stdout=open(_PROJECT_ROOT / "java_cp_stdout.log", "w"),
        stderr=open(_PROJECT_ROOT / "java_cp_stderr.log", "w"),
    )
    print(f"[CP] Starting GameOfLife CP server on port {port} (bp_iter={bp_iter}) ...")

    deadline = time.time() + 60
    while time.time() < deadline:
        try:
            with socket.create_connection(("localhost", port), timeout=1):
                print(f"[CP] Server ready on port {port}")
                break
        except OSError:
            time.sleep(0.5)
    else:
        proc.terminate()
        raise RuntimeError(f"GameOfLife CP server did not start on port {port} within 60 s.")

    env = RDDLGymWrapper(domain=domain, instance=instance, algo="ppo-cp", seed=seed)

    client = CPClient(host="localhost", port=port)
    if not client.connect():
        proc.terminate()
        raise RuntimeError("CPClient failed to connect to the GameOfLife CP server.")
    if not client.init(str(instance)):
        client.close()
        proc.terminate()
        raise RuntimeError(f"CPClient INIT failed for instance {instance}.")

    nY = max(
        int(k.split("__y")[1])
        for k in env._action_keys
    )
    n_cells = len(env._action_keys)

    shaper = GameOfLifeCPShaper(
        client=client,
        alpha=alpha,
        gamma=gamma,
        n_cells=n_cells,
        nY=nY,
        obs_dict_fn=lambda: env._last_obs_dict,
        action_keys=env._action_keys,
        debug_csv=debug_csv,
    )
    env._shaper = shaper

    return env, client, proc


def _build_cp_env_sysadmin(domain: str, instance: str, seed: int, cp_cfg: dict) -> tuple:
    """Instantiate RDDLGymWrapper + SysAdminCPShaper for ppo-cp on SysAdmin."""
    from cp_client import CPClient
    from envs.shaping import SysAdminCPShaper

    debug_csv = cp_cfg.get("debug_csv")
    port     = cp_cfg.get("port", 12348)
    bp_iter  = cp_cfg.get("bp_iter", 3)
    alpha    = cp_cfg.get("alpha", 1.0)
    gamma    = cp_cfg.get("gamma", 1.0)
    java_dir = _PROJECT_ROOT / cp_cfg.get("java_project_dir", "MiniCPBP")

    cmd = [
        "mvn", "-f", str(java_dir / "pom.xml"),
        "--offline",
        "exec:java",
        "-Dexec.cleanupDaemonThreads=false",
        "-Dexec.mainClass=minicpbp.examples.SysAdminService",
        f"-Dexec.args={port}",
        f"-Dexec.jvmArgs=-Xmx8g",
        f"-Dbp.iterations={bp_iter}",
    ]
    proc = subprocess.Popen(
        cmd,
        cwd=str(java_dir),
        stdout=open(_PROJECT_ROOT / "java_cp_stdout.log", "w"),
        stderr=open(_PROJECT_ROOT / "java_cp_stderr.log", "w"),
    )
    print(f"[CP] Starting SysAdmin CP server on port {port} (bp_iter={bp_iter}) ...")

    deadline = time.time() + 60
    while time.time() < deadline:
        try:
            with socket.create_connection(("localhost", port), timeout=1):
                print(f"[CP] Server ready on port {port}")
                break
        except OSError:
            time.sleep(0.5)
    else:
        proc.terminate()
        raise RuntimeError(f"SysAdmin CP server did not start on port {port} within 60 s.")

    env = RDDLGymWrapper(domain=domain, instance=instance, algo="ppo-cp", seed=seed)

    client = CPClient(host="localhost", port=port)
    if not client.connect():
        proc.terminate()
        raise RuntimeError("CPClient failed to connect to the SysAdmin CP server.")
    if not client.init(str(instance)):
        client.close()
        proc.terminate()
        raise RuntimeError(f"CPClient INIT failed for instance {instance}.")

    n_computers = len(env._obs_keys)
    shaper = SysAdminCPShaper(
        client=client,
        alpha=alpha,
        gamma=gamma,
        horizon=env.horizon,
        n_computers=n_computers,
        obs_dict_fn=lambda: env._last_obs_dict,
        action_keys=env._action_keys,
        debug_csv=debug_csv,
    )
    env._shaper = shaper

    return env, client, proc


def _build_cp_env_navigation(domain: str, instance: str, seed: int, cp_cfg: dict) -> tuple:
    """Instantiate RDDLGymWrapper + NavigationCPShaper for ppo-cp on Navigation."""
    from cp_client import CPClient
    from envs.shaping import NavigationCPShaper

    debug_csv = cp_cfg.get("debug_csv")
    port     = cp_cfg.get("port", 12350)
    bp_iter  = cp_cfg.get("bp_iter", 3)
    alpha    = cp_cfg.get("alpha", 1.0)
    gamma    = cp_cfg.get("gamma", 1.0)
    java_dir = _PROJECT_ROOT / cp_cfg.get("java_project_dir", "MiniCPBP")

    cmd = [
        "mvn", "-f", str(java_dir / "pom.xml"),
        "--offline",
        "exec:java",
        "-Dexec.cleanupDaemonThreads=false",
        "-Dexec.mainClass=minicpbp.examples.NavigationService",
        f"-Dexec.args={port}",
        f"-Dexec.jvmArgs=-Xmx4g",
        f"-Dbp.iterations={bp_iter}",
    ]
    proc = subprocess.Popen(
        cmd,
        cwd=str(java_dir),
        stdout=open(_PROJECT_ROOT / "java_cp_stdout.log", "w"),
        stderr=open(_PROJECT_ROOT / "java_cp_stderr.log", "w"),
    )
    print(f"[CP] Starting Navigation CP server on port {port} (bp_iter={bp_iter}) ...")

    deadline = time.time() + 60
    while time.time() < deadline:
        try:
            with socket.create_connection(("localhost", port), timeout=1):
                print(f"[CP] Server ready on port {port}")
                break
        except OSError:
            time.sleep(0.5)
    else:
        proc.terminate()
        raise RuntimeError(f"Navigation CP server did not start on port {port} within 60 s.")

    env = RDDLGymWrapper(domain=domain, instance=instance, algo="ppo-cp", seed=seed)

    cp_instances_path = _PROJECT_ROOT / "cp_instances" / "navigation_mdp_ippc2011.json"
    with open(cp_instances_path) as f:
        cp_meta = json.load(f)
    inst_meta = cp_meta[str(instance)]
    nX            = inst_meta["grid_x"]
    nY            = inst_meta["grid_y"]
    xpos_sorted   = inst_meta["xpos"]
    ypos_sorted   = inst_meta["ypos"]
    dead_state    = inst_meta["dead_state_idx"]
    goal_state    = inst_meta["goal_state_idx"]

    client = CPClient(host="localhost", port=port)
    if not client.connect():
        proc.terminate()
        raise RuntimeError("CPClient failed to connect to Navigation CP server.")
    if not client.init(str(instance)):
        client.close()
        proc.terminate()
        raise RuntimeError(f"CPClient INIT failed for Navigation instance {instance}.")

    shaper = NavigationCPShaper(
        client=client,
        alpha=alpha,
        gamma=gamma,
        nX=nX,
        nY=nY,
        xpos_sorted=xpos_sorted,
        ypos_sorted=ypos_sorted,
        obs_dict_fn=lambda: env._last_obs_dict,
        dead_state_idx=dead_state,
        goal_state_idx=goal_state,
        debug_csv=debug_csv,
    )
    env._shaper = shaper
    return env, client, proc


def _build_cp_env_crossingtraffic(domain: str, instance: str, seed: int, cp_cfg: dict) -> tuple:
    """Instantiate RDDLGymWrapper + TrafficCrossingCPShaper for ppo-cp on CrossingTraffic."""
    from cp_client import CPClient
    from envs.shaping import TrafficCrossingCPShaper

    debug_csv = cp_cfg.get("debug_csv")
    port     = cp_cfg.get("port", 12346)
    bp_iter  = cp_cfg.get("bp_iter", 3)
    alpha    = cp_cfg.get("alpha", 1.0)
    gamma    = cp_cfg.get("gamma", 1.0)
    java_dir = _PROJECT_ROOT / cp_cfg.get("java_project_dir", "MiniCPBP")

    cmd = [
        "mvn", "-f", str(java_dir / "pom.xml"),
        "--offline",
        "exec:java",
        "-Dexec.cleanupDaemonThreads=false",
        "-Dexec.mainClass=minicpbp.examples.TrafficCrossingService",
        f"-Dexec.args={port}",
        f"-Dexec.jvmArgs=-Xmx8g",
        f"-Dbp.iterations={bp_iter}",
    ]
    proc = subprocess.Popen(
        cmd,
        cwd=str(java_dir),
        stdout=open(_PROJECT_ROOT / "java_cp_stdout.log", "w"),
        stderr=open(_PROJECT_ROOT / "java_cp_stderr.log", "w"),
    )
    print(f"[CP] Starting CrossingTraffic CP server on port {port} (bp_iter={bp_iter}) ...")

    deadline = time.time() + 60
    while time.time() < deadline:
        try:
            with socket.create_connection(("localhost", port), timeout=1):
                print(f"[CP] Server ready on port {port}")
                break
        except OSError:
            time.sleep(0.5)
    else:
        proc.terminate()
        raise RuntimeError(f"CrossingTraffic CP server did not start on port {port} within 60 s.")

    env = RDDLGymWrapper(domain=domain, instance=instance, algo="ppo-cp", seed=seed)

    cp_instances_path = _PROJECT_ROOT / "cp_instances" / "crossingtraffic_mdp_ippc2011.json"
    with open(cp_instances_path) as f:
        cp_meta = json.load(f)
    inst_meta = cp_meta[str(instance)]
    nX = inst_meta["grid_x"]
    nY = inst_meta["grid_y"]

    client = CPClient(host="localhost", port=port)
    if not client.connect():
        proc.terminate()
        raise RuntimeError("CPClient failed to connect to CrossingTraffic CP server.")
    if not client.init(str(instance)):
        client.close()
        proc.terminate()
        raise RuntimeError(f"CPClient INIT failed for CrossingTraffic instance {instance}.")

    shaper = TrafficCrossingCPShaper(
        client=client,
        alpha=alpha,
        gamma=gamma,
        nX=nX,
        nY=nY,
        obs_dict_fn=lambda: env._last_obs_dict,
        debug_csv=debug_csv,
    )
    env._shaper = shaper
    return env, client, proc


def _build_cp_env_triangletireworld(domain: str, instance: str, seed: int, cp_cfg: dict) -> tuple:
    """Instantiate RDDLGymWrapper + TriangleTireworldCPShaper for ppo-cp."""
    from cp_client import CPClient
    from envs.shaping import TriangleTireworldCPShaper

    debug_csv = cp_cfg.get("debug_csv")
    port      = cp_cfg.get("port", 12351)
    bp_iter   = cp_cfg.get("bp_iter", 3)
    alpha     = cp_cfg.get("alpha", 1.0)
    gamma     = cp_cfg.get("gamma", 1.0)
    java_dir  = _PROJECT_ROOT / cp_cfg.get("java_project_dir", "MiniCPBP")

    cmd = [
        "mvn", "-f", str(java_dir / "pom.xml"),
        "--offline",
        "exec:java",
        "-Dexec.cleanupDaemonThreads=false",
        "-Dexec.mainClass=minicpbp.examples.TriangleTireworldService",
        f"-Dexec.args={port}",
        f"-Dexec.jvmArgs=-Xmx4g",
        f"-Dbp.iterations={bp_iter}",
    ]
    proc = subprocess.Popen(
        cmd,
        cwd=str(java_dir),
        stdout=open(_PROJECT_ROOT / "java_cp_stdout.log", "w"),
        stderr=open(_PROJECT_ROOT / "java_cp_stderr.log", "w"),
    )
    print(f"[CP] Starting TriangleTireworld CP server on port {port} (bp_iter={bp_iter}) ...")

    deadline = time.time() + 60
    while time.time() < deadline:
        try:
            with socket.create_connection(("localhost", port), timeout=1):
                print(f"[CP] Server ready on port {port}")
                break
        except OSError:
            time.sleep(0.5)
    else:
        proc.terminate()
        raise RuntimeError(f"TriangleTireworld CP server did not start on port {port} within 60 s.")

    env = RDDLGymWrapper(domain=domain, instance=instance, algo="ppo-cp", seed=seed)

    cp_instances_path = _PROJECT_ROOT / "cp_instances" / "triangletireworld_mdp_ippc2014.json"
    with open(cp_instances_path) as f:
        cp_meta = json.load(f)
    inst_meta        = cp_meta[str(instance)]
    successors       = inst_meta["successors"]
    goal_location    = inst_meta["goal_location"]
    initial_location = inst_meta["initial_location"]
    spare_locations  = inst_meta["spare_locations"]
    loc_names = sorted(
        k[len("vehicle-at___"):]
        for k in env._obs_keys
        if k.startswith("vehicle-at___")
    )

    client = CPClient(host="localhost", port=port)
    if not client.connect():
        proc.terminate()
        raise RuntimeError("CPClient failed to connect to TriangleTireworld CP server.")
    if not client.init(str(instance)):
        client.close()
        proc.terminate()
        raise RuntimeError(f"CPClient INIT failed for TriangleTireworld instance {instance}.")

    shaper = TriangleTireworldCPShaper(
        client=client,
        alpha=alpha,
        gamma=gamma,
        loc_names=loc_names,
        successors=successors,
        obs_dict_fn=lambda: env._last_obs_dict,
        action_keys=env._action_keys,
        goal_location=goal_location,
        initial_location=initial_location,
        spare_locations=spare_locations,
        debug_csv=debug_csv,
    )
    env._shaper = shaper
    return env, client, proc


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    run_dir  = get_run_dir(args.domain, args.instance, args.algo, args.seed)
    eval_json = run_dir / "eval.json"

    if eval_json.exists() and not args.force:
        print(f"[SKIP] {eval_json} already exists. Use --force to retrain.")
        return

    run_dir.mkdir(parents=True, exist_ok=True)

    is_baseline = args.algo in ("noop", "random")
    is_cp       = args.algo == "ppo-cp"

    config = load_config(algo=args.algo, domain=args.domain,
                         override_path=args.config)
    total_timesteps: int = 0 if is_baseline else config.pop("total_timesteps", 500_000)

    config_record = {
        "domain":           args.domain,
        "instance":         args.instance,
        "algo":             args.algo,
        "seed":             args.seed,
        "total_timesteps":  total_timesteps,
        **({} if is_baseline else config),
    }
    with open(run_dir / "config.yaml", "w") as f:
        yaml.dump(config_record, f, default_flow_style=False)

    cp_proc = None
    client  = None

    try:
        if is_cp:
            cp_cfg = config.get("cp", {})
            cp_cfg["gamma"] = config.get("algo_kwargs", {}).get("gamma", 1.0)
            if args.debug_shaping:
                cp_cfg["debug_csv"] = str(run_dir / "shaping_debug.csv")
            print(f"[INFO] Building CP env: {args.domain} instance={args.instance} "
                  f"alpha={cp_cfg.get('alpha', 1.0)} gamma={cp_cfg['gamma']}")
            if "sysadmin" in args.domain.lower():
                builder = _build_cp_env_sysadmin
            elif "gameoflife" in args.domain.lower():
                builder = _build_cp_env_gameoflife
            elif "navigation" in args.domain.lower():
                builder = _build_cp_env_navigation
            elif "crossingtraffic" in args.domain.lower():
                cp_model = cp_cfg.get("cp_model", "costregular")
                if cp_model == "costregular":
                    builder = _build_cp_env_crossingtraffic
                elif cp_model == "linedecomposition":
                    builder = _build_cp_env
                else:
                    raise ValueError(
                        f"Unknown cp_model={cp_model!r} for CrossingTraffic. "
                        f"Expected 'costregular' or 'linedecomposition'."
                    )
            elif "triangletireworld" in args.domain.lower():
                builder = _build_cp_env_triangletireworld
            else:
                builder = _build_cp_env
            env, client, cp_proc = builder(
                domain=args.domain,
                instance=args.instance,
                seed=args.seed,
                cp_cfg=cp_cfg,
            )
        else:
            print(f"[INFO] Building env: {args.domain} instance={args.instance} "
                  f"algo={args.algo}")
            env = RDDLGymWrapper(
                domain=args.domain,
                instance=args.instance,
                algo=args.algo,
                seed=args.seed,
            )

        print(f"       action_space={env.action_space}  "
              f"max_allowed={env.max_allowed_actions}  horizon={env.horizon}")

        agent = make_agent(
            env=env,
            algo=args.algo,
            algo_kwargs={} if is_baseline else dict(config.get("algo_kwargs", {})),
            seed=args.seed,
            verbose=args.verbose,
            domain=args.domain,
            instance=args.instance,
        )

        if not is_baseline:
            csv_path      = str(run_dir / "metrics.csv")
            show_progress = os.environ.get("SB3_PROGRESS_BAR", "1") == "1"
            callbacks = [CSVMetricsCallback(csv_path=csv_path, verbose=0)]
            if config.get("early_stop_enabled", False):
                callbacks.append(EarlyStoppingCallback(
                    window        = config.get("early_stop_window",        200),
                    patience      = config.get("early_stop_patience",       20),
                    min_delta     = config.get("early_stop_min_delta",    0.01),
                    check_freq    = config.get("early_stop_check_freq",    200),
                    min_timesteps = config.get("early_stop_min_timesteps", 500_000),
                    verbose=1,
                ))
                print(f"[INFO] Training up to {total_timesteps} timesteps "
                      f"(early stopping enabled) ...")
            else:
                print(f"[INFO] Training up to {total_timesteps} timesteps "
                      f"(early stopping disabled) ...")
            agent.learn(
                total_timesteps=total_timesteps,
                callback=CallbackList(callbacks),
                progress_bar=show_progress,
            )
            model_path = str(run_dir / "model")
            agent.save(model_path)
            print(f"[INFO] Model saved → {model_path}.zip")

        env.close()

        eval_episodes: int = config.get("eval_episodes", 30)
        print(f"[INFO] Evaluating ({eval_episodes} episodes) ...")
        episode_returns = _evaluate(
            agent, args.domain, args.instance, args.algo,
            args.seed, n_episodes=eval_episodes,
        )

        mean_r = float(np.mean(episode_returns))
        std_r  = float(np.std(episode_returns))

        eval_record = {
            "domain":           args.domain,
            "instance":         args.instance,
            "algo":             args.algo,
            "seed":             args.seed,
            "total_timesteps":  total_timesteps,
            "eval_episodes":    eval_episodes,
            "mean_return":      mean_r,
            "std_return":       std_r,
            "episode_returns":  episode_returns,
        }
        with open(eval_json, "w") as f:
            json.dump(eval_record, f, indent=2)

        print(
            f"[RESULT] {args.algo.upper()} {args.domain} inst={args.instance} "
            f"seed={args.seed}: mean={mean_r:.2f} ± {std_r:.2f}  →  {eval_json}"
        )

    finally:
        if client is not None:
            client.close()
        if cp_proc is not None:
            cp_proc.terminate()
            cp_proc.wait()
            print("[CP] Java server stopped.")


if __name__ == "__main__":
    main()
