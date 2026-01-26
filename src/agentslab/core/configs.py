from dataclasses import dataclass
from typing import Union

import torch

__all__ = [
    "resolve_device",
    "GeneralConfigs",
]

def resolve_device(preferred: Union[str, torch.device, None]) -> torch.device:
    """
    Резолвит устройство для вычислений.
    
    Args:
        preferred: "cuda", "cpu", "mps", "auto", None, или torch.device
        
    Returns:
        torch.device
    """
    if isinstance(preferred, torch.device):
        return preferred
    
    if preferred is None or preferred == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    
    preferred = preferred.lower()
    
    if preferred == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        print("Warning: CUDA not available, falling back to CPU")
        return torch.device("cpu")
    
    if preferred == "mps":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        print("Warning: MPS not available, falling back to CPU")
        return torch.device("cpu")
    
    if preferred == "cpu":
        return torch.device("cpu")
    
    raise ValueError(f"Unknown device: {preferred}. Use 'cuda', 'mps', 'cpu', or 'auto'")


@dataclass
class GeneralConfigs:
    """
    Глобальные настройки эксперимента.
    
    Определяет что запускаем и через какие вычислительные ресурсы.
    
    Attributes:
        algo_name: Название алгоритма (PPO, SAC, DQN, ...)
        env_name: Универсальный идентификатор среды. Используется как:
            - env_id для Gymnasium ("CartPole-v1", "Ant-v4")
            - scenario для VMAS ("navigation", "transport")
            - task для PettingZoo ("mpe/simple_spread_v3")
        device: Устройство для вычислений
        seed: Random seed для воспроизводимости
        deterministic: Детерминированный режим PyTorch
        
    Example:
        >>> cfg = GeneralConfigs(algo_name="PPO", env_name="CartPole-v1")
        >>> cfg.device
        device(type='cuda')
        
        >>> cfg = GeneralConfigs(
        ...     algo_name="SAC",
        ...     env_name="Pendulum-v1",
        ...     device="cpu",
        ...     seed=123,
        ... )
        
        >>> # VMAS
        >>> cfg = GeneralConfigs(algo_name="MAPPO", env_name="navigation")
        
        >>> # PettingZoo
        >>> cfg = GeneralConfigs(algo_name="IPPO", env_name="mpe/simple_spread_v3")
    """
    algo_name: str
    env_name: str
    device: Union[str, torch.device] = "auto"
    seed: int = 42
    deterministic: bool = False
    
    def __post_init__(self):
        self.device = resolve_device(self.device)

