from __future__ import annotations
from tqdm import tqdm
import os
import torch
from torchrl.envs import ExplorationType, set_exploration_type
from torchrl.envs.utils import step_mdp
from agentslab.runner.logger import CSVLogger

def evaluate(*, policy, make_env_fn, env_cfg, eval_cfg, device, paths=None, write_csv: bool = True) -> float:
    if paths is not None and write_csv:
        os.makedirs(paths.logs_path, exist_ok=True)
        eval_log_path = os.path.join(paths.logs_path, "eval.csv")
        logger = CSVLogger(eval_log_path, fieldnames=["episode", "return"])
    else:
        logger = None

    returns = []
    pbar = tqdm(total=eval_cfg.episodes, desc="Evaluation", unit="ep", leave=True)
    try:
        with torch.no_grad(), set_exploration_type(ExplorationType.MODE):
            for ep in range(eval_cfg.episodes):
                env = make_env_fn(env_cfg.env_id, env_cfg.seed + ep, env_cfg.render_mode)
                td = env.reset()
                ep_ret = 0.0
                steps = 0
                while True:
                    td = policy(td.to(device))
                    td = env.step(td)  # returns a single TensorDict with next/...
                    reward = float(td.get(("next", "reward")).item())
                    ep_ret += reward
                    steps += 1
                    done = bool(td.get(("next", "done")).item())
                    td = step_mdp(td)  # shift next->root
                    if done or steps >= eval_cfg.max_steps_per_ep:
                        break
                env.close()
                returns.append(ep_ret)
                if logger:
                    logger.log(episode=ep, ret=ep_ret)
                pbar.update(1)
    finally:
        pbar.close()
        if logger:
            logger.close()

    avg_ret = float(sum(returns) / max(len(returns), 1))
    print(f"Average return over {len(returns)} episodes: {avg_ret:.2f}")
    return avg_ret
