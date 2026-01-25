"""
Утилиты для воспроизводимости экспериментов.

Основная функция:
    set_global_seed() — устанавливает seed для всех генераторов

Example:
    >>> from agentslab.core import set_global_seed
    >>> set_global_seed(42)
    >>> set_global_seed(42, deterministic=True)

References:
    https://pytorch.org/docs/stable/notes/randomness.html
"""

import os
import random

import numpy as np
import torch

__all__ = ["set_global_seed"]


def set_global_seed(seed: int = 42, deterministic: bool = False) -> None:
    """
    Устанавливает seed для всех генераторов случайных чисел.
    
    Args:
        seed: Значение seed
        deterministic: Включить детерминированный режим PyTorch.
            Гарантирует воспроизводимость, но может снизить
            производительность на 10-20%.
    
    Example:
        >>> set_global_seed(42)
        >>> set_global_seed(42, deterministic=True)
        
        >>> # С конфигом:
        >>> cfg = GeneralConfigs(algo_name="PPO", env_id="CartPole-v1")
        >>> set_global_seed(cfg.seed, cfg.deterministic)
    """
    
    random.seed(seed)       # Python
    np.random.seed(seed)    # NumPy
    
    # PyTorch CPU + CUDA
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Python hash (эффективно только до запуска интерпретатора)
    os.environ["PYTHONHASHSEED"] = str(seed)
    
    # Детерминизм
    if deterministic:
        _enable_deterministic_mode()


def _enable_deterministic_mode() -> None:
    """Включает детерминированный режим PyTorch."""
    # cuBLAS
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    
    # PyTorch 1.8+
    if hasattr(torch, "use_deterministic_algorithms"):
        torch.use_deterministic_algorithms(True, warn_only=True)
    
    # cuDNN
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

