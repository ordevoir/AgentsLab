
from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence, Dict, Any, List
import time
import csv
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange

from agentslab.utils.paths import logs_dir, ckpts_dir, results_dir
from agentslab.utils.seed import set_seed
from agentslab.rl.training.common import make_env
from agentslab.rl.agents.reinforce.agent import ReinforceAgent, ReinforceConfig

@dataclass
class ReinforceTrainConfig:
    seed: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_episodes: int = 300
    max_steps: int = 500
    hidden_sizes: Sequence[int] = (128, 128)

def train_reinforce(env_id: str, cfg_train: ReinforceTrainConfig, cfg_agent: ReinforceConfig) -> Dict[str, Any]:
    set_seed(cfg_train.seed)
    device = torch.device(cfg_train.device)
    env, obs_dim, n_actions = make_env(env_id, cfg_train.seed)

    agent = ReinforceAgent(obs_dim, n_actions, cfg_train.hidden_sizes, cfg_agent, device)

    run_name = f"reinforce_{env_id}_{int(time.time())}"
    writer = SummaryWriter(logs_dir("rl", "reinforce", run_name))
    ckpt_dir = ckpts_dir("rl", "reinforce", run_name)
    res_dir = results_dir("rl", "reinforce")
    (res_dir / f"{run_name}.csv").parent.mkdir(parents=True, exist_ok=True)

    best_reward = -float("inf")

    with open(res_dir / f"{run_name}.csv", "w", newline="") as f_csv:
        csv_writer = csv.writer(f_csv)
        csv_writer.writerow(["episode", "reward", "loss"])

        for ep in trange(cfg_train.num_episodes, desc="REINFORCE episodes"):
            obs, _ = env.reset(seed=cfg_train.seed + ep)
            log_probs: List[torch.Tensor] = []
            rewards: List[float] = []
            ep_reward = 0.0

            for step in range(cfg_train.max_steps):
                # Act and store log prob for gradient
                action, log_prob_val = agent.act(obs)
                # For exact gradient we need the tensor log_prob, so recompute via forward pass
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                logits = agent.policy(obs_t)
                dist = torch.distributions.Categorical(logits=logits)
                action_t = torch.as_tensor(action, device=device)
                log_probs.append(dist.log_prob(action_t))
                # Step env
                next_obs, reward, terminated, truncated, _ = env.step(action)
                rewards.append(float(reward))
                ep_reward += float(reward)
                obs = next_obs
                if terminated or truncated:
                    break

            loss_val = agent.update(log_probs, rewards)

            writer.add_scalar("charts/episode_reward", ep_reward, ep)
            writer.add_scalar("loss/policy_loss", loss_val, ep)
            csv_writer.writerow([ep, ep_reward, loss_val])

            if ep_reward > best_reward:
                best_reward = ep_reward
                save_checkpoint(ckpt_dir, agent, ep, best_reward)

    env.close()
    writer.close()
    return {"best_reward": best_reward, "run_name": run_name}

def save_checkpoint(ckpt_dir, agent: ReinforceAgent, episode: int, best_reward: float) -> None:
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = ckpt_dir / f"ep{episode:04d}_R{best_reward:.1f}.pt"
    torch.save({
        "policy": agent.policy.state_dict(),
        "optim": agent.optim.state_dict(),
        "meta": {"episode": episode, "best_reward": best_reward},
    }, path)
