from dataclasses import dataclass
from torchrl.objectives import ClipPPOLoss

@dataclass
class PPOLossConfig:
    clip_epsilon: float = 0.2
    entropy_coef: float = 1e-4
    critic_coef: float = 1.0
    entropy_bonus: bool = True
    # Note: TorchRL PPO already uses smooth_l1 for value by default.

def build_ppo_loss(cfg: PPOLossConfig, actor, critic) -> ClipPPOLoss:
    return ClipPPOLoss(
        actor,
        critic,
        clip_epsilon=cfg.clip_epsilon,
        entropy_bonus=cfg.entropy_bonus,
        entropy_coef=cfg.entropy_coef,
        critic_coef=cfg.critic_coef,
    )
