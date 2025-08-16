from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from hydra import main
from omegaconf import DictConfig

from agentslab.rl.agents.pg import PGPolicy
from agentslab.rl.agents.dqn import QNet
from agentslab.utils.io import load_checkpoint
from agentslab.utils.seed import select_device, set_seed


def play_env(env: gym.Env, act_fn, episodes: int, max_steps: int) -> dict:
    scores: list[float] = []
    for _ in range(episodes):
        obs, _ = env.reset()
        total = 0.0
        with torch.no_grad():
            for _ in range(max_steps):
                a = act_fn(obs)
                obs, r, term, trunc, _ = env.step(a)
                total += r
                if term or trunc:
                    break
        scores.append(total)
    arr = np.asarray(scores, dtype=float)
    return {
        "episodes": episodes,
        "mean_reward": float(arr.mean()),
        "std_reward": float(arr.std()),
        "min_reward": float(arr.min()),
        "max_reward": float(arr.max()),
    }


@main(version_base=None, config_path="../../configs", config_name="config")
def run(cfg: DictConfig) -> None:
    assert cfg.mode == "eval", "Set mode=eval in the config or CLI"

    set_seed(cfg.seed)
    device = select_device(cfg.device)

    cp_path = cfg.eval.checkpoint_path
    if cp_path is None:
        raise ValueError("eval.checkpoint_path must be provided")

    state = load_checkpoint(cp_path, map_location=str(device))

    env_id = cfg.env.id or state.get("env_id", None)
    if env_id is None:
        raise ValueError("Environment id must be set via cfg.env.id or present in checkpoint")

    env = gym.make(env_id, render_mode="human" if cfg.eval.render else None)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    # Detect model type and rebuild
    algo = state.get("algo", "unknown")
    print(f"Evaluating algo={algo}, env={env_id}, checkpoint={cp_path}")

    if algo == "pg":
        hidden = state["cfg"]["rl"]["algo"]["network"]["hidden_units"]
        net = PGPolicy(obs_dim, act_dim, hidden_units=hidden).to(device)
        net.load_state_dict(state["model_state"])  # type: ignore[arg-type]
        net.eval()

        def act(obs):
            x = torch.tensor(obs, dtype=torch.float32, device=device)
            return int(net(x).argmax(dim=-1).item())

    elif algo == "dqn":
        hidden1 = state["cfg"]["rl"]["algo"]["network"]["hidden1"]
        hidden2 = state["cfg"]["rl"]["algo"]["network"]["hidden2"]
        q = QNet(obs_dim, act_dim, hidden1, hidden2).to(device)
        target_state = state.get("target_state", None) or state.get("online_state")
        q.load_state_dict(target_state)  # type: ignore[arg-type]
        q.eval()

        def act(obs):
            x = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            return int(q(x).argmax(dim=-1).item())
    else:
        raise ValueError(f"Unsupported/unknown algo in checkpoint: {algo}")

    stats = play_env(env, act, episodes=cfg.eval.episodes, max_steps=cfg.train.max_steps)
    print({"env": env_id, "algo": algo, "checkpoint": cp_path, **stats})


if __name__ == "__main__":
    run()
