
from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence, Dict, Any
import time
import csv
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange

from agentslab.utils.paths import logs_dir, ckpts_dir, results_dir
from agentslab.utils.seed import set_seed
from agentslab.rl.training.common import make_env
from agentslab.rl.agents.dqn.agent import DQNAgent, DQNConfig

@dataclass
class DQNTrainConfig:
    seed: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_episodes: int = 300
    max_steps: int = 500
    eval_every: int = 25
    save_every: int = 100
    hidden_sizes: Sequence[int] = (128, 128)

def train_dqn(env_id: str, cfg_train: DQNTrainConfig, cfg_agent: DQNConfig) -> Dict[str, Any]:
    set_seed(cfg_train.seed)
    device = torch.device(cfg_train.device)
    env, obs_dim, n_actions = make_env(env_id, cfg_train.seed)

    agent = DQNAgent(obs_dim, n_actions, cfg_train.hidden_sizes, cfg_agent, device)

    run_name = f"dqn_{env_id}_{int(time.time())}"
    writer = SummaryWriter(logs_dir("rl", "dqn", run_name))
    ckpt_dir = ckpts_dir("rl", "dqn", run_name)
    res_dir = results_dir("rl", "dqn")
    (res_dir / f"{run_name}.csv").parent.mkdir(parents=True, exist_ok=True)

    best_reward = -float("inf")
    global_step = 0

    with open(res_dir / f"{run_name}.csv", "w", newline="") as f_csv:
        csv_writer = csv.writer(f_csv)
        csv_writer.writerow(["episode", "reward", "epsilon", "loss", "td_abs_mean"])

        for ep in trange(cfg_train.num_episodes, desc="DQN episodes"):
            obs, _ = env.reset(seed=cfg_train.seed + ep)  # different seed per episode
            ep_reward = 0.0
            loss_val = 0.0
            td_abs_mean = 0.0

            for step in range(cfg_train.max_steps):
                action = agent.act(obs)
                next_obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                agent.push(obs, action, next_obs, reward, done)
                loss_val, td_abs_mean = agent.train_step()

                obs = next_obs
                ep_reward += float(reward)
                global_step += 1

                if done:
                    break

            writer.add_scalar("charts/episode_reward", ep_reward, ep)
            writer.add_scalar("charts/epsilon", agent.epsilon(), ep)
            if loss_val:
                writer.add_scalar("loss/q_loss", loss_val, ep)
                writer.add_scalar("stats/td_abs_mean", td_abs_mean, ep)

            csv_writer.writerow([ep, ep_reward, agent.epsilon(), loss_val, td_abs_mean])

            # simple eval (greedy) every eval_every
            if (ep + 1) % cfg_train.eval_every == 0:
                eval_reward = evaluate(env, agent, cfg_train.max_steps)
                writer.add_scalar("eval/greedy_reward", eval_reward, ep)

            # save checkpoints
            if (ep + 1) % cfg_train.save_every == 0 or ep_reward > best_reward:
                best_reward = max(best_reward, ep_reward)
                save_checkpoint(ckpt_dir, agent, ep, best_reward)

    env.close()
    writer.close()
    return {"best_reward": best_reward, "run_name": run_name}

@torch.no_grad()
def evaluate(env, agent: DQNAgent, max_steps: int) -> float:
    obs, _ = env.reset()
    ep_reward = 0.0
    for _ in range(max_steps):
        action = agent.act(obs, exploit=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        ep_reward += float(reward)
        if terminated or truncated:
            break
    return ep_reward

def save_checkpoint(ckpt_dir, agent: DQNAgent, episode: int, best_reward: float) -> None:
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = ckpt_dir / f"ep{episode:04d}_R{best_reward:.1f}.pt"
    torch.save({
        "q": agent.q.state_dict(),
        "q_target": agent.q_target.state_dict(),
        "optim": agent.optim.state_dict(),
        "meta": {"episode": episode, "best_reward": best_reward},
    }, path)
