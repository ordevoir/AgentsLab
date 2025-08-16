# src/agentslab/rl/training/dqn_trainer.py
from __future__ import annotations

import os
import time
from typing import Dict, Tuple

import gymnasium as gym
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from agentslab.core.checkpointing import save_checkpoint
from agentslab.core.metrics import MovingAverage
from agentslab.rl.agents.dqn_agent import DQNAgent, EpsilonSchedule, ReplayBuffer
from agentslab.utils.seeding import set_seed


def _device_from_cfg(device_cfg: str) -> torch.device:
    if device_cfg.lower().startswith("cuda") or device_cfg.lower().startswith("gpu"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_cfg.lower().startswith("cpu"):
        return torch.device("cpu")
    # "cuda if available"
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _obs_dim(space) -> int:
    if len(space.shape) == 1:
        return int(space.shape[0])
    # Flatten any non-1D observation space (simple handling)
    return int(np.prod(space.shape))


def train(cfg) -> Dict[str, float]:
    # Seeding and device
    set_seed(int(cfg.env.seed))
    device = _device_from_cfg(str(cfg.rl.device))

    # Env
    env = gym.make(cfg.env.id)
    if cfg.env.max_episode_steps is not None:
        env = gym.wrappers.TimeLimit(env, max_episode_steps=int(cfg.env.max_episode_steps))

    obs, _ = env.reset(seed=int(cfg.env.seed))
    obs_dim = _obs_dim(env.observation_space)
    n_actions = env.action_space.n

    # Agent, buffer, epsilon schedule
    agent = DQNAgent(
        obs_dim=obs_dim,
        n_actions=n_actions,
        hidden_sizes=list(cfg.rl.network.hidden_sizes),
        lr=float(cfg.rl.lr),
        gamma=float(cfg.rl.gamma),
        device=device,
    )
    buffer = ReplayBuffer(obs_dim=obs_dim, buffer_size=int(cfg.rl.buffer_size), device=device)
    eps_sched = EpsilonSchedule(
        start=float(cfg.rl.epsilon.start),
        end=float(cfg.rl.epsilon.end),
        decay_steps=int(cfg.rl.epsilon.decay_steps),
    )

    # Logging & checkpoint paths (keep top-level lab structure)
    run_stamp = time.strftime("%Y%m%d-%H%M%S")
    env_id = cfg.env.id.replace("/", "-")
    root = cfg.project.root_dir
    logs_dir = os.path.join(root, "logs", "rl", "dqn", env_id, run_stamp)
    ckpt_dir = os.path.join(root, "checkpoints", "rl", "dqn", env_id, run_stamp)
    results_dir = os.path.join(root, "results", "rl", "dqn", env_id, run_stamp)
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    writer = SummaryWriter(log_dir=logs_dir)

    # Training loop
    episode = 0
    global_step = 0
    ep_return_ma = MovingAverage(window_size=100)
    best_ma_return = -float("inf")
    ep_return = 0.0
    ep_len = 0

    while global_step < int(cfg.rl.total_timesteps):
        epsilon = eps_sched.value(global_step)
        action = agent.act(obs, epsilon)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        buffer.add(obs, action, reward, next_obs, done)
        obs = next_obs
        ep_return += float(reward)
        ep_len += 1
        global_step += 1

        # Learn
        if buffer.size >= int(cfg.rl.batch_size) and global_step > int(cfg.rl.learning_starts):
            loss, avg_q = agent.update(
                buffer=buffer,
                batch_size=int(cfg.rl.batch_size),
                target_update_freq=int(cfg.rl.target_update_freq),
                global_step=global_step,
            )
            writer.add_scalar("train/loss", loss, global_step)
            writer.add_scalar("train/avg_q", avg_q, global_step)
            writer.add_scalar("train/epsilon", epsilon, global_step)

        if done:
            ep_return_ma.update(ep_return)
            writer.add_scalar("train/episode_return", ep_return, global_step)
            writer.add_scalar("train/episode_length", ep_len, global_step)

            episode += 1
            if episode % int(cfg.logging.print_interval_episodes) == 0:
                print(f"[DQN] Ep {episode} | Step {global_step} | Return {ep_return:.1f} | MA100 {ep_return_ma.value:.1f} | eps {epsilon:.3f}")

            # Periodic 'last' checkpoint
            if int(cfg.checkpoint.save_last_every_episodes) > 0 and episode % int(cfg.checkpoint.save_last_every_episodes) == 0:
                save_checkpoint(
                    directory=ckpt_dir,
                    filename="last",
                    model=agent,
                    optimizer=agent.optim,
                    step=global_step,
                    meta={
                        "model_type": "dqn",
                        "env_id": cfg.env.id,
                        "episode": episode,
                        "moving_avg_return": ep_return_ma.value,
                        "config": {"rl": dict(cfg.rl), "env": dict(cfg.env)},
                    },
                )

            # Best checkpoint by moving average return
            if cfg.checkpoint.save_best and ep_return_ma.value > best_ma_return:
                best_ma_return = ep_return_ma.value
                save_checkpoint(
                    directory=ckpt_dir,
                    filename="best",
                    model=agent,
                    optimizer=agent.optim,
                    step=global_step,
                    meta={
                        "model_type": "dqn",
                        "env_id": cfg.env.id,
                        "episode": episode,
                        "moving_avg_return": ep_return_ma.value,
                        "config": {"rl": dict(cfg.rl), "env": dict(cfg.env)},
                    },
                )

            # Reset episode
            obs, _ = env.reset()
            ep_return = 0.0
            ep_len = 0

    # Final 'last' checkpoint
    save_checkpoint(
        directory=ckpt_dir,
        filename="last",
        model=agent,
        optimizer=agent.optim,
        step=global_step,
        meta={
            "model_type": "dqn",
            "env_id": cfg.env.id,
            "episode": episode,
            "moving_avg_return": ep_return_ma.value,
            "config": {"rl": dict(cfg.rl), "env": dict(cfg.env)},
        },
    )

    # Store final results
    with open(os.path.join(results_dir, "summary.json"), "w", encoding="utf-8") as f:
        import json
        json.dump(
            {
                "final_step": global_step,
                "episodes": episode,
                "best_moving_avg_return": best_ma_return,
                "moving_avg_return": ep_return_ma.value,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    writer.close()
    env.close()
    return {"best_moving_avg_return": best_ma_return, "moving_avg_return": ep_return_ma.value}
