from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from agentslab.networks.mlp import MLP

@dataclass
class PPOConfig:
    gamma: float = 0.99
    gae_lambda: float = 0.95
    lr: float = 3e-4
    hidden_sizes: tuple[int, int] = (128, 128)
    clip_coef: float = 0.2
    update_epochs: int = 4
    minibatch_size: int = 64
    rollout_steps: int = 2048
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    max_grad_norm: float = 0.5

class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_sizes: tuple[int, int]) -> None:
        super().__init__()
        self.policy = MLP(obs_dim, act_dim, hidden_sizes=hidden_sizes)
        self.valuef = MLP(obs_dim, 1, hidden_sizes=hidden_sizes)

    def get_action_and_value(self, obs: torch.Tensor, action: torch.Tensor | None = None):
        logits = self.policy(obs)
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs=probs)
        if action is None:
            action = dist.sample()
        logprob = dist.log_prob(action)
        entropy = dist.entropy()
        value = self.valuef(obs).squeeze(-1)
        return action, logprob, entropy, value

class PPOAgent:
    def __init__(self, obs_dim: int, act_dim: int, cfg: PPOConfig, device: torch.device) -> None:
        self.device = device
        self.cfg = cfg
        self.ac = ActorCritic(obs_dim, act_dim, cfg.hidden_sizes).to(device)
        self.optimizer = torch.optim.Adam(self.ac.parameters(), lr=cfg.lr)

    def collect_rollout(self, env, seed: int | None = None) -> dict:
        obs, _ = env.reset(seed=seed)
        obs_list, actions, logprobs, rewards, dones, values = [], [], [], [], [], []
        for _ in range(self.cfg.rollout_steps):
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
            with torch.no_grad():
                action_t, logp_t, _, value_t = self.ac.get_action_and_value(obs_t.unsqueeze(0))
            action = int(action_t.item())
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            obs_list.append(obs)
            actions.append(action)
            logprobs.append(float(logp_t.item()))
            rewards.append(float(reward))
            dones.append(float(done))
            values.append(float(value_t.item()))
            obs = next_obs
            if done:
                obs, _ = env.reset()
        # bootstrap value
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
            _, _, _, last_val = self.ac.get_action_and_value(obs_t.unsqueeze(0))
            last_val = float(last_val.item())
        traj = {
            "obs": np.array(obs_list, dtype=np.float32),
            "actions": np.array(actions, dtype=np.int64),
            "logprobs": np.array(logprobs, dtype=np.float32),
            "rewards": np.array(rewards, dtype=np.float32),
            "dones": np.array(dones, dtype=np.float32),
            "values": np.array(values, dtype=np.float32),
            "last_value": last_val,
        }
        return traj

    def compute_gae(self, rewards, values, dones, last_value, gamma, lam):
        T = len(rewards)
        adv = np.zeros(T, dtype=np.float32)
        last_gae = 0.0
        for t in reversed(range(T)):
            next_nonterminal = 1.0 - (dones[t] if t < T - 1 else 0.0)
            next_value = values[t + 1] if t < T - 1 else last_value
            delta = rewards[t] + gamma * next_value * next_nonterminal - values[t]
            last_gae = delta + gamma * lam * next_nonterminal * last_gae
            adv[t] = last_gae
        returns = adv + values
        return adv, returns

    def update(self, traj: dict) -> dict:
        cfg = self.cfg
        obs = torch.as_tensor(traj["obs"], dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(traj["actions"], dtype=torch.int64, device=self.device)
        old_logprobs = torch.as_tensor(traj["logprobs"], dtype=torch.float32, device=self.device)
        values = torch.as_tensor(traj["values"], dtype=torch.float32, device=self.device)
        adv_np, ret_np = self.compute_gae(traj["rewards"], traj["values"], traj["dones"], traj["last_value"], cfg.gamma, cfg.gae_lambda)
        advantages = torch.as_tensor(adv_np, dtype=torch.float32, device=self.device)
        returns = torch.as_tensor(ret_np, dtype=torch.float32, device=self.device)

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        batch_size = obs.size(0)
        idxs = np.arange(batch_size)
        metrics = {}
        for epoch in range(cfg.update_epochs):
            np.random.shuffle(idxs)
            for start in range(0, batch_size, cfg.minibatch_size):
                end = start + cfg.minibatch_size
                mb_idx = idxs[start:end]
                mb_obs = obs[mb_idx]
                mb_actions = actions[mb_idx]
                mb_old_logp = old_logprobs[mb_idx]
                mb_adv = advantages[mb_idx]
                mb_returns = returns[mb_idx]

                _, logp, entropy, value = self.ac.get_action_and_value(mb_obs, mb_actions)
                ratio = torch.exp(logp - mb_old_logp)
                pg_loss1 = -mb_adv * ratio
                pg_loss2 = -mb_adv * torch.clamp(ratio, 1.0 - cfg.clip_coef, 1.0 + cfg.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                v_loss = 0.5 * (mb_returns - value).pow(2).mean()
                ent_loss = entropy.mean()

                loss = pg_loss + cfg.vf_coef * v_loss - cfg.ent_coef * ent_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.ac.parameters(), cfg.max_grad_norm)
                self.optimizer.step()

                metrics = {
                    "loss/policy": float(pg_loss.item()),
                    "loss/value": float(v_loss.item()),
                    "loss/entropy": float(ent_loss.item()),
                }
        return metrics
