
# AgentsLab (RL focus, step 1)

This repo is an initial working version of your RL lab that matches your folder structure.
It includes **vanilla REINFORCE** and **DQN** (no TorchRL / Lightning yet), Hydra configs,
TensorBoard logging, and checkpoints. Gymnasium is used directly (no wrappers).

## Install (editable)
```bash
pip install -e .
```

## Train
**DQN (CartPole):**
```bash
python scripts/rl/train.py rl/agent=dqn rl/env=cartpole rl/training=cartpole_fast
```

**REINFORCE (CartPole):**
```bash
python scripts/rl/train.py rl/agent=reinforce rl/env=cartpole rl/training=cartpole_fast
```

You can customize hyper-parameters by overriding Hydra config keys on the CLI, e.g.:
```bash
python scripts/rl/train.py rl/agent=dqn rl.env.id=CartPole-v1 rl.training.num_episodes=500
```

Outputs go to `outputs/` (Hydra), while additional logs/checkpoints/results are stored
under the repo root using absolute paths (so they're not affected by Hydra's cwd change).

## Notes
- Code follows PEP8.
- Hydra is only used in **entrypoints** (`scripts/rl/train.py`). Library code stays framework-agnostic.
- Gymnasium API is handled (`reset` returns `(obs, info)`, `step` returns `(obs, reward, terminated, truncated, info)`).
