
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Sequence, Dict, Any
import time
import csv
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from agentslab.rl.training.common import make_env
from agentslab.utils.paths import logs_dir, results_dir
from .metrics import summarize
from .loader import build_agent_from_cfg, load_weights

@dataclass
class EvalConfig:
    seed: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_episodes: int = 20
    max_steps: int = 500
    deterministic: bool = True
    obs_noise_std: float = 0.0   # add Gaussian noise to observations
    action_epsilon: float = 0.0  # with this prob take random action
    hidden_sizes: Sequence[int] = (128, 128)

def evaluate_checkpoint(
    algo: str,
    env_id: str,
    cfg_eval: EvalConfig,
    cfg_agent: Any,
    checkpoint_path: str,
) -> Dict[str, Any]:
    device = torch.device(cfg_eval.device)
    env, obs_dim, n_actions = make_env(env_id, cfg_eval.seed)

    # Build and load agent
    agent = build_agent_from_cfg(algo, obs_dim, n_actions, cfg_eval.hidden_sizes, cfg_agent, device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    load_weights(agent, ckpt, algo)

    run_name = f"eval_{algo}_{env_id}_{int(time.time())}"
    writer = SummaryWriter(logs_dir("rl", "eval", run_name))
    res_dir = results_dir("rl", "eval")
    (res_dir / f"{run_name}.csv").parent.mkdir(parents=True, exist_ok=True)

    returns: List[float] = []
    lengths: List[int] = []

    with open(res_dir / f"{run_name}.csv", "w", newline="") as f_csv:
        csv_writer = csv.writer(f_csv)
        csv_writer.writerow(["episode", "return", "length"])

        for ep in range(cfg_eval.num_episodes):
            obs, _ = env.reset(seed=cfg_eval.seed + ep)
            ep_return = 0.0
            ep_len = 0
            for t in range(cfg_eval.max_steps):
                obs_eval = np.array(obs, copy=True, dtype=np.float32)
                if cfg_eval.obs_noise_std > 0.0:
                    obs_eval = obs_eval + np.random.normal(0.0, cfg_eval.obs_noise_std, size=obs_eval.shape).astype(np.float32)

                # epsilon-random override
                if np.random.rand() < cfg_eval.action_epsilon:
                    action = int(np.random.randint(n_actions))
                else:
                    if algo == "dqn":
                        action = agent.act(obs_eval, exploit=cfg_eval.deterministic)
                    elif algo == "reinforce":
                        # Deterministic: argmax over logits; stochastic: sample
                        obs_t = torch.as_tensor(obs_eval, dtype=torch.float32, device=device).unsqueeze(0)
                        logits = agent.policy(obs_t)
                        if cfg_eval.deterministic:
                            action = int(torch.argmax(logits, dim=-1).item())
                        else:
                            dist = torch.distributions.Categorical(logits=logits)
                            action = int(dist.sample().item())
                    else:
                        raise ValueError(f"Unknown algo: {algo}")

                obs, reward, terminated, truncated, _ = env.step(action)
                ep_return += float(reward)
                ep_len += 1
                if terminated or truncated:
                    break

            returns.append(ep_return)
            lengths.append(ep_len)
            writer.add_scalar("eval/episode_return", ep_return, ep)
            writer.add_scalar("eval/episode_length", ep_len, ep)
            csv_writer.writerow([ep, ep_return, ep_len])

    env.close()
    writer.close()

    summary = summarize(returns, lengths)
    # Also dump JSON summary
    import json
    with open(res_dir / f"{run_name}.json", "w") as f_json:
        json.dump(summary.to_dict(), f_json, indent=2)

    return {"run_name": run_name, "summary": summary.to_dict(), "returns": returns, "lengths": lengths}
