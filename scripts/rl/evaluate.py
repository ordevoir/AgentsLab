# scripts/rl/evaluate.py
from __future__ import annotations

import os
from typing import Any, Tuple

import hydra
import gymnasium as gym
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from agentslab.core.checkpointing import load_checkpoint
from agentslab.networks.mlp import MLP


def _obs_dim(space) -> int:
    if len(space.shape) == 1:
        return int(space.shape[0])
    return int(np.prod(space.shape))


def _device_from_cfg(device_cfg: str) -> torch.device:
    if device_cfg.lower().startswith("cuda") or device_cfg.lower().startswith("gpu"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_cfg.lower().startswith("cpu"):
        return torch.device("cpu")
    # "cuda if available"
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@hydra.main(version_base=None, config_path="../../configs", config_name="eval")
def main(cfg: DictConfig) -> Any:
    if not os.path.isfile(cfg.checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {cfg.checkpoint_path}")

    payload = load_checkpoint(cfg.checkpoint_path, map_location="cpu")
    meta = payload.get("meta", {})
    model_type = str(meta.get("model_type", cfg.model)).lower()

    env = gym.make(cfg.env.id, render_mode="human" if cfg.render else None)
    obs, _ = env.reset(seed=int(cfg.env.seed))
    obs_dim = _obs_dim(env.observation_space)
    n_actions = env.action_space.n

    device = _device_from_cfg("cuda if available")

    if model_type == "dqn":
        # Rebuild Q-network and load weights
        hidden = tuple(meta.get("config", {}).get("rl", {}).get("network", {}).get("hidden_sizes", [128, 128]))
        q_net = MLP(obs_dim, n_actions, hidden).to(device)
        q_net.load_state_dict(payload["model_state_dict"]["q.net" if "q.net" in payload["model_state_dict"] else "net" ] if isinstance(payload["model_state_dict"], dict) else payload["model_state_dict"])  # fallback
        q_net.eval()

        def act_fn(ob):
            with torch.no_grad():
                t = torch.as_tensor(ob, dtype=torch.float32, device=device).unsqueeze(0)
                q = q_net(t)
                return int(torch.argmax(q, dim=1).item())

    elif model_type == "reinforce":
        hidden = tuple(meta.get("config", {}).get("rl", {}).get("network", {}).get("hidden_sizes", [128, 128]))
        policy = MLP(obs_dim, n_actions, hidden).to(device)
        policy.load_state_dict(payload["model_state_dict"])  # REINFORCE saved the policy weights
        policy.eval()

        def act_fn(ob):
            with torch.no_grad():
                t = torch.as_tensor(ob, dtype=torch.float32, device=device).unsqueeze(0)
                logits = policy(t)
                return int(torch.argmax(logits, dim=1).item())

    else:
        raise ValueError(f"Unknown model type: {model_type}")

    returns = []
    for ep in range(int(cfg.episodes)):
        obs, _ = env.reset(seed=int(cfg.env.seed) + ep)
        done = False
        ep_ret = 0.0
        ep_len = 0
        while not done:
            action = act_fn(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ep_ret += float(reward)
            ep_len += 1
        print(f"Episode {ep+1}/{cfg.episodes} — Return: {ep_ret:.2f} | Length: {ep_len}")
        returns.append(ep_ret)

    avg = float(np.mean(returns)) if returns else 0.0
    print(f"\nAvg return over {cfg.episodes} episodes: {avg:.2f}")
    env.close()
    return {"avg_return": avg}


if __name__ == "__main__":
    main()
