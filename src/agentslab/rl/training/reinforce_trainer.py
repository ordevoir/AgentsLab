# src/agentslab/rl/training/reinforce_trainer.py
from __future__ import annotations

import os
import time
from typing import Dict

import gymnasium as gym
import torch
from torch.utils.tensorboard import SummaryWriter

from agentslab.core.checkpointing import save_checkpoint
from agentslab.core.metrics import MovingAverage
from agentslab.rl.agents.reinforce_agent import ReinforceAgent, ReinforceConfig
from agentslab.utils.seeding import set_seed


def _device_from_cfg(device_cfg: str) -> torch.device:
    if device_cfg.lower().startswith("cuda") or device_cfg.lower().startswith("gpu"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_cfg.lower().startswith("cpu"):
        return torch.device("cpu")
    # "cuda if available"
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(cfg) -> Dict[str, float]:
    # Seed and device (device not strictly used here, policy on CPU by default)
    set_seed(int(cfg.env.seed))
    _ = _device_from_cfg(str(cfg.rl.device))

    env = gym.make(cfg.env.id)
    if cfg.env.max_episode_steps is not None:
        env = gym.wrappers.TimeLimit(env, max_episode_steps=int(cfg.env.max_episode_steps))

    obs, _ = env.reset(seed=int(cfg.env.seed))
    obs_dim = int(env.observation_space.shape[0])
    n_actions = env.action_space.n

    agent = ReinforceAgent(
        obs_dim=obs_dim,
        n_actions=n_actions,
        cfg=ReinforceConfig(
            gamma=float(cfg.rl.gamma),
            lr=float(cfg.rl.lr),
            hidden_sizes=tuple(cfg.rl.network.hidden_sizes),
            normalize_returns=bool(cfg.rl.normalize_returns),
        ),
    )

    # Logging & checkpoints
    run_stamp = time.strftime("%Y%m%d-%H%M%S")
    env_id = cfg.env.id.replace("/", "-")
    root = cfg.project.root_dir
    logs_dir = os.path.join(root, "logs", "rl", "reinforce", env_id, run_stamp)
    ckpt_dir = os.path.join(root, "checkpoints", "rl", "reinforce", env_id, run_stamp)
    results_dir = os.path.join(root, "results", "rl", "reinforce", env_id, run_stamp)
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    writer = SummaryWriter(log_dir=logs_dir)

    best_ma_return = -float("inf")
    ma = MovingAverage(window_size=100)

    total_episodes = int(cfg.rl.total_episodes)
    batch_episodes = int(cfg.rl.batch_episodes)

    ep = 0
    while ep < total_episodes:
        logps = []
        rewards = []
        episode_return = 0.0
        ep_len = 0
        obs, _ = env.reset()

        done = False
        while not done:
            action, logp = agent.policy.act(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            logps.append(logp)
            rewards.append(float(reward))
            episode_return += float(reward)
            ep_len += 1
            obs = next_obs

        loss = agent.update(logps, rewards)
        ma.update(episode_return)

        ep += 1
        writer.add_scalar("train/episode_return", episode_return, ep)
        writer.add_scalar("train/episode_length", ep_len, ep)
        writer.add_scalar("train/policy_loss", loss, ep)

        if ep % int(cfg.logging.print_interval_episodes) == 0:
            print(f"[REINFORCE] Ep {ep}/{total_episodes} | Return {episode_return:.1f} | MA100 {ma.value:.1f}")

        # Periodic 'last' checkpoint
        if int(cfg.checkpoint.save_last_every_episodes) > 0 and ep % int(cfg.checkpoint.save_last_every_episodes) == 0:
            save_checkpoint(
                directory=ckpt_dir,
                filename="last",
                model=agent.policy,
                optimizer=agent.optim,
                step=ep,
                meta={
                    "model_type": "reinforce",
                    "env_id": cfg.env.id,
                    "episode": ep,
                    "moving_avg_return": ma.value,
                    "config": {"rl": dict(cfg.rl), "env": dict(cfg.env)},
                },
            )

        # Best by moving average return
        if cfg.checkpoint.save_best and ma.value > best_ma_return:
            best_ma_return = ma.value
            save_checkpoint(
                directory=ckpt_dir,
                filename="best",
                model=agent.policy,
                optimizer=agent.optim,
                step=ep,
                meta={
                    "model_type": "reinforce",
                    "env_id": cfg.env.id,
                    "episode": ep,
                    "moving_avg_return": ma.value,
                    "config": {"rl": dict(cfg.rl), "env": dict(cfg.env)},
                },
            )

    writer.close()
    env.close()
    return {"best_moving_avg_return": best_ma_return, "moving_avg_return": ma.value}
