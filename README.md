# Constraint Programming-Guided Reinforcement Learning on IPPC Stochastic Planning Domains

## Project Description

This project investigates the effectiveness of using Constraint Programming (CP) and Belief Propagation (BP) to generate reward shaping signals for Proximal Policy Optimization (PPO) agents operating in stochastic planning domains from the International Probabilistic Planning Competition (IPPC 2011/2014).

The central hypothesis is that a CP model of the domain's transition dynamics, solved via Belief Propagation, can produce accurate estimates of the Expected Total Reward (ETR) at each state. These estimates serve as a potential function for potential-based reward shaping, accelerating learning without altering the optimal policy.

The method introduced is:
* **CP-ETR Shaping (PPO-CP):** At each environment step, a Java CP server computes the ETR for the current state via Belief Propagation. The shaped reward is defined as:

  `reward_shaped = reward_env + alpha * (gamma * ETR(s') - ETR(s))`

  where `alpha` is a shaping coefficient and `gamma` matches the PPO discount factor to guarantee potential-based shaping.

## Supported Domains

| Domain | IPPC Year | Instances |
|---|---|---|
| Navigation MDP | IPPC 2011 | 1–10 |
| CrossingTraffic MDP | IPPC 2011 | 1–10 |
| SysAdmin MDP | IPPC 2011 | 1–10 |
| TriangleTireworld MDP | IPPC 2014 | 1–10 |
| GameOfLife MDP | IPPC 2011 | 1–10 |

## Repository Contents

* **`main.py`**: Parallel experiment launcher. Runs all combinations of domains, instances, algorithms, and seeds using a multiprocessing pool.
* **`cp_client.py`**: TCP socket client for communicating with the Java CP server (INIT / RESET / STEP / QUERY\_ETR / QUIT protocol).
* **`scripts/train.py`**: Main training script. Builds the environment and agent, runs training, evaluates the final policy, and saves results.
* **`scripts/evaluate.py`**: Standalone evaluation script. Loads a saved model and runs evaluation episodes.
* **`scripts/plot_learning_curves.py`**: Plots smoothed learning curves from `metrics.csv` files.
* **`scripts/plot_results.py`**: Aggregated performance plots across seeds and instances.
* **`agents/`**: Agent factory (`factory.py`) with `NoOpAgent` and `RandomAgent` baselines; PPO and PPO-CP are built via Stable-Baselines3.
* **`envs/`**: Environment wrappers (`rddl_env.py`), action space utilities (`action_space.py`), CP state encoders (`cp_state_encoder.py`), and reward shapers (`shaping.py`).
* **`utils/`**: Config loader (`config.py`) and SB3 callbacks (`logging.py`): `CSVMetricsCallback` and `EarlyStoppingCallback`.
* **`configs/`**: YAML hyperparameter files. `defaults.yaml` defines base settings; `configs/domains/` contains per-domain overrides.
* **`cp_instances/`**: JSON metadata files (grid dimensions, goal/dead state indices, etc.) used by the Python side to configure CP shapers.
* **`MiniCPBP/`**: Java/Maven CP solver (fork of MiniCPBP). Contains one TCP service per domain (`NavigationService`, `TrafficCrossingService`, `SysAdminService`, `TriangleTireworldService`, `GameOfLifeService`) and their corresponding decomposition models.
* **`slurm/`**: SLURM job submission scripts for running experiments on a compute cluster.

## How to Run

### Prerequisites

* Python 3.11+
* Java 11+ and Maven 3.6+

```bash
pip install -r requirements.txt
```

### Building the CP Solver

The Java CP server must be compiled before running any `ppo-cp` experiment:

```bash
cd MiniCPBP
mvn clean compile
cd ..
```

### Training a Single Agent

```bash
python scripts/train.py --domain <DOMAIN> --instance <INSTANCE> --algo <ALGO> --seed <SEED>
```

**Arguments:**

* `--domain`: RDDL domain name (e.g. `Navigation_MDP_ippc2011`)
* `--instance`: Instance index (integer, e.g. `1`)
* `--algo`: Algorithm — one of `ppo`, `ppo-cp`, `noop`, `random`
* `--seed`: Random seed (default: `0`)
* `--force`: Re-train even if `eval.json` already exists
* `--config`: Path to a YAML override file for hyperparameters
* `--verbose`: SB3 verbosity level (default: `0`)
* `--debug-shaping`: Write per-step ETR shaping values to `shaping_debug.csv` in the run directory

**Example commands:**

```bash
# PPO baseline on Navigation, instance 1
python scripts/train.py --domain Navigation_MDP_ippc2011 --instance 1 --algo ppo --seed 0

# PPO with CP-ETR shaping on CrossingTraffic, instance 1
python scripts/train.py --domain CrossingTraffic_MDP_ippc2011 --instance 1 --algo ppo-cp --seed 0

# Noop baseline (no learning)
python scripts/train.py --domain SysAdmin_MDP_ippc2011 --instance 1 --algo noop --seed 0
```

> **Note:** `ppo-cp` automatically starts the appropriate Java CP server for the given domain and terminates it when training ends. Only one `ppo-cp` run should be active per domain at a time to avoid port conflicts.

### Running All Experiments in Parallel

```bash
python main.py --workers 4
```

Use `--dry-run` to preview all jobs without executing them, and `--force` to overwrite existing results.

### Evaluating a Saved Model

```bash
python scripts/evaluate.py --domain Navigation_MDP_ippc2011 --instance 1 --algo ppo --seed 0 --n-episodes 30
```

### Plotting Learning Curves

```bash
# Single domain/instance
python scripts/plot_learning_curves.py --domain Navigation_MDP_ippc2011 --instance 1

# All available results
python scripts/plot_learning_curves.py

# Compare PPO vs A2C vs DQN
python scripts/plot_learning_curves.py --compare
```

### Configuration

Hyperparameters are loaded from `configs/defaults.yaml` and overridden by `configs/domains/<domain>.yaml`. Any value can be further overridden at runtime via `--config path/to/overrides.yaml`.

Key CP parameters (under the `cp:` key in domain configs):

* `bp_iter`: Number of Belief Propagation iterations (1 for Navigation, 3 for all other domains)
* `alpha`: Shaping coefficient scaling the ETR potential difference
* `port`: TCP port for the Java CP server

### Results Structure

Training results are saved under `results/<domain>/<instance>/<algo>/run_<seed>/`:

* `metrics.csv`: Per-episode timestep, shaped reward, and raw reward
* `eval.json`: Final evaluation statistics (mean and std return over `eval_episodes` episodes)
* `model.zip`: Saved SB3 model weights
* `config.yaml`: Full hyperparameter record for reproducibility
