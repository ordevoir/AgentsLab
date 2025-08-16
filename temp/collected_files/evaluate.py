
import os
import hydra
from omegaconf import DictConfig
import torch

from agentslab.rl.eval.eval_loop import EvalConfig, evaluate_checkpoint
from agentslab.rl.agents.dqn.agent import DQNConfig
from agentslab.rl.agents.reinforce.agent import ReinforceConfig

@hydra.main(version_base="1.3", config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    # resolve device
    device = cfg.rl.eval.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Build eval config
    eval_cfg = EvalConfig(
        seed=cfg.rl.eval.seed,
        device=device,
        num_episodes=cfg.rl.eval.num_episodes,
        max_steps=cfg.rl.eval.max_steps,
        deterministic=cfg.rl.eval.deterministic,
        obs_noise_std=cfg.rl.eval.obs_noise_std,
        action_epsilon=cfg.rl.eval.action_epsilon,
        hidden_sizes=tuple(cfg.rl.eval.hidden_sizes),
    )

    # Agent configs (must match the trained model's architecture/hparams)
    algo_name = "dqn" if "dqn" in cfg.rl.agent else ("reinforce" if "reinforce" in cfg.rl.agent else None)
    if algo_name is None:
        raise ValueError("Please select agent in defaults (rl/agent=...)")

    if algo_name == "dqn":
        agent_cfg = DQNConfig(
            gamma=cfg.rl.agent.gamma,
            lr=cfg.rl.agent.lr,
            batch_size=cfg.rl.agent.batch_size,
            buffer_size=cfg.rl.agent.buffer_size,
            start_learning_after=cfg.rl.agent.start_learning_after,
            target_update_every=cfg.rl.agent.target_update_every,
            tau=cfg.rl.agent.tau,
            eps_start=cfg.rl.agent.eps_start,
            eps_end=cfg.rl.agent.eps_end,
            eps_decay_steps=cfg.rl.agent.eps_decay_steps,
            huber_delta=cfg.rl.agent.huber_delta,
            clip_grad_norm=cfg.rl.agent.clip_grad_norm,
        )
    else:
        agent_cfg = ReinforceConfig(
            gamma=cfg.rl.agent.gamma,
            lr=cfg.rl.agent.lr,
            entropy_coef=cfg.rl.agent.entropy_coef,
            normalize_returns=cfg.rl.agent.normalize_returns,
            clip_grad_norm=cfg.rl.agent.clip_grad_norm,
        )

    ckpt_path = cfg.rl.eval.checkpoint_path
    if not ckpt_path or not os.path.exists(ckpt_path):
        raise FileNotFoundError("Provide a valid checkpoint via rl.eval.checkpoint_path=/path/to/ckpt.pt")

    _ = evaluate_checkpoint(
        algo=algo_name,
        env_id=cfg.rl.env.id,
        cfg_eval=eval_cfg,
        cfg_agent=agent_cfg,
        checkpoint_path=ckpt_path,
    )

if __name__ == "__main__":
    main()
