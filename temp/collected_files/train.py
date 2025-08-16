
import hydra
from omegaconf import DictConfig
import torch

from agentslab.rl.training.dqn_train import DQNTrainConfig, train_dqn
from agentslab.rl.agents.dqn.agent import DQNConfig
from agentslab.rl.training.reinforce_train import ReinforceTrainConfig, train_reinforce
from agentslab.rl.agents.reinforce.agent import ReinforceConfig

# Hydra only in entrypoint: best practice
@hydra.main(version_base="1.3", config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    mode = cfg.get("mode", "rl")
    if mode != "rl":
        raise NotImplementedError("Only RL mode is implemented in this phase.")

    # Resolve device
    device = cfg.rl.training.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    agent_name = cfg.rl.agent.get("name", "dqn").lower()

    if agent_name == "dqn":
        train_cfg = DQNTrainConfig(
            seed=cfg.rl.training.seed,
            device=device,
            num_episodes=cfg.rl.training.num_episodes,
            max_steps=cfg.rl.training.max_steps,
            eval_every=cfg.rl.training.eval_every,
            save_every=cfg.rl.training.save_every,
            hidden_sizes=tuple(cfg.rl.network.hidden_sizes),
        )
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
        train_dqn(cfg.rl.env.id, train_cfg, agent_cfg)

    elif agent_name == "reinforce":
        train_cfg = ReinforceTrainConfig(
            seed=cfg.rl.training.seed,
            device=device,
            num_episodes=cfg.rl.training.num_episodes,
            max_steps=cfg.rl.training.max_steps,
            hidden_sizes=tuple(cfg.rl.network.hidden_sizes),
        )
        agent_cfg = ReinforceConfig(
            gamma=cfg.rl.agent.gamma,
            lr=cfg.rl.agent.lr,
            entropy_coef=cfg.rl.agent.entropy_coef,
            normalize_returns=cfg.rl.agent.normalize_returns,
            clip_grad_norm=cfg.rl.agent.clip_grad_norm,
        )
        train_reinforce(cfg.rl.env.id, train_cfg, agent_cfg)
    else:
        raise NotImplementedError(f"Unknown agent name: {agent_name}")

if __name__ == "__main__":
    main()
