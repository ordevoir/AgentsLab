# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AgentsLab is a modular reinforcement learning research framework built on TorchRL. It supports single-agent (Gymnasium) and multi-agent (VMAS, PettingZoo) environments with PPO as the primary algorithm.

## Installation & Setup

```bash
pip install -e . -r requirements.txt
```

Requires Python 3.10+.

## Project Structure

```
src/agentslab/
├── core/           # GeneralConfigs, RunPaths, seeding
├── envs/           # Environment configs and make_env() factory
├── networks/       # MLPConfig, MultiAgentMLPConfig, build functions
├── policies/       # Stochastic actor construction
├── runners/        # PPOConfigs, evals, objectives, schedulers
├── storages/       # Experience buffers and collectors
└── utils/          # CSVLogger, TBLogger, CheckpointManager, plotters
```

## Core Modules

### core/configs.py
- `resolve_device(preferred)` — resolves "auto"/"cuda"/"mps"/"cpu" to `torch.device`
- `GeneralConfigs` — experiment setup: `algo_name`, `env_name`, `device`, `seed`, `deterministic`

### core/paths.py
- `RunPaths` — frozen dataclass with paths: `run_dir`, `ckpt_dir`, `csv_train`, `csv_eval`, `fig_dir`, `meta_yaml`
- `generate_paths(algo_name, env_name)` — creates unique run with timestamp, sanitizes env_name (e.g., `mpe/simple_spread_v3` → `mpe-simple_spread_v3`)
- `restore_paths(run_name)` — restores paths from existing run

### core/seeding.py
- `set_global_seed()` — sets Python, NumPy, PyTorch, CUDA seeds for reproducibility

## Environment System (envs/)

### Configs (envs/configs.py)
- `BaseEnvConfig` — abstract base with `device`, `seed`, `transforms`
- `GymEnvConfig` — `env_name`, `batch_size`, `categorical_action_encoding`, `frame_skip`, `render_mode`
- `VMASEnvConfig` — `scenario`, `num_envs`, `continuous_actions`, `max_steps`, `group_map`
- `PettingZooEnvConfig` — `task`, `parallel`, `return_state`, `use_mask`, `group_map`
- `TransformConfig` — `observation_norm`, `double_to_float`, `step_counter`, `max_steps`, `reward_sum`
- `ObservationNormConfig` — `enabled`, `in_keys`, `loc`, `scale`, `num_iter`

### Factory (envs/__init__.py)
- `make_env(config)` — universal factory, dispatches by config type with lazy loading

### Transforms (envs/transforms.py)
- `build_transforms(config)` → `TransformBundle` with transforms list and `observation_norm` reference
- `init_observation_norm(env, config)` — initializes stats via random rollouts (call after TransformedEnv creation)

## Networks (networks/networks.py)

### Single-agent
- `MLPConfig` — `in_dim`, `hidden_sizes`, `out_dim`, `activation`, `layer_norm`
- `build_mlp(cfg)` → `nn.Sequential`

### Multi-agent
- `MultiAgentMLPConfig` — `n_agent_inputs`, `n_agent_outputs`, `n_agents`, `centralized`, `share_params`, `depth`, `num_cells`
- `build_multi_agent_mlp(cfg)` → `MultiAgentMLP` (TorchRL)

## Policies (policies/policy.py)

### MultiCategorical Distribution
Custom distribution for `MultiDiscrete` action spaces. Wraps multiple independent `Categorical` distributions:
- `MultiCategorical(nvec, logits)` — `nvec` is list of category counts per sub-action
- `sample()` → tensor of shape `(..., len(nvec))` with action indices
- `log_prob(value)` → sum of log probs across sub-actions
- `entropy()` → sum of entropies across sub-actions

### Utility Functions
- `get_num_action_logits(action_spec)` — returns logits count for output layer
- `build_stochastic_actor(network, action_spec)` — builds `ProbabilisticActor`:
  - `OneHotDiscreteTensorSpec` → `OneHotCategorical`
  - `DiscreteTensorSpec` → `Categorical`
  - `MultiDiscreteTensorSpec` → `MultiCategorical`
  - `BoundedTensorSpec` → `TanhNormal` with rescaling

## Runners (runners/)

### trainers.py
- `PPOConfigs` — PPO hyperparameters: `gamma`, `lmbda`, `clip_epsilon`, `entropy_eps`, `lr`, `critic_coeff`, `max_grad_norm`, `num_epochs`, `sub_batch_size`

### Other modules (less documented)
- `evals.py` — evaluation utilities
- `objectives.py` — loss functions
- `schedulers.py` — learning rate schedulers

## Utilities (utils/)

### loggers.py
- `CSVLogger(csv_path)` — numeric-only CSV logging with schema validation, `.log(row)` method
- `TBLogger(log_dir)` — TensorBoard wrapper, `.log(row, step)` method

### checkpointers.py
- `CheckpointManager` — saves `last.pt`, `best.pt`, `step_N.pt` with rotation
  - `save(step, metrics)` — atomic write, best tracking by metric
  - `load(which, strict, map_location)` — loads into registered statefuls
  - `register_obs_norms_from_env(env)` — auto-registers ObservationNorm transforms
  - Handles optimizer device mismatch on load (transfers state to param device)

### plotters.py
- `plot_metrics_from_csv(csv_path, ema, save_dir, ...)` — plots each metric from CSV
- `plot_metrics(df, ...)` — plots from DataFrame
- Supports EMA smoothing, include/exclude regex patterns, downsampling

## Key Patterns

- Russian docstrings with examples (sphinx-style)
- Full type annotations throughout
- Lazy imports via `__getattr__` in `__init__.py` files
- Pathlib for all path handling (Windows-compatible)
- Dataclass configs with `__post_init__` validation

## Running Experiments

Interactive notebooks in `notebooks/`:
- `PPO_gymnasium.ipynb` — Single-agent PPO (CartPole, MuJoCo)
- `PPO_vmas.ipynb` — Multi-agent PPO with VMAS scenarios

Output structure:
```
runs/{algo}_{env}_{timestamp}/
├── checkpoints/    # last.pt, best.pt, step_*.pt
├── csv_logs/       # train.csv, eval.csv
├── figures/        # metric plots
└── meta_info.yaml
```

## Knowledge Base (junctions/)

External documentation linked via symbolic junctions. Actual paths:

**AgentsLab Documentation** (`ExternalMental/AgentsLab/`):
- `AgentsLab.md` — overview, library structure
- `General Configuration.md` — `GeneralConfigs` details
- `Envs.md` — environment configs, transforms, factory (comprehensive)
- `Paths.md` — `RunPaths`, `generate_paths()`, `restore_paths()`
- `Checkpoint Manager.md` — full API documentation
- `Plotters.md` — plotting functions documentation

**RL/MARL Theory** (`ExternalMental/20 KnowledgeBase/RL MARL/`):
- `_Base Definitions.md` — MDP, V/Q functions, Bellman equations
- `_Policy Gradient.md`, `_Actor-Critic.md` — policy gradient methods
- `Atomic Notes/PPO Algorithm.md` — step-by-step PPO
- `Atomic Notes/Generalized Advantage Estimation GAE.md` — GAE formula
- `TorchRL/` — TensorDict, Envs, Specs
- `Environment Frameworks/` — Gymnasium, PettingZoo, VMAS, MPE

**Machine Learning** (`ExternalMental/20 KnowledgeBase/Machine Learning/`):
- Gradient optimization, regularization, loss functions

**Neural Networks** (`ExternalMental/20 KnowledgeBase/Neural Networks/`):
- Activations, normalization, CNN, initialization

## Working Rules (from AGENTS.md)

- Show all changes as diffs before applying
- Never commit without explicit request
- Propose a plan first, then implement
- No mass refactoring without discussion
- Don't add dependencies without confirmation
- Ask when uncertain
