"""
AgentsLab Policies.

Модуль для построения стохастических акторов под различные типы action spaces.
"""

from agentslab.policies.policy import (
    MultiCategorical,
    get_num_action_logits,
    build_stochastic_actor,
)

__all__ = [
    "MultiCategorical",
    "get_num_action_logits",
    "build_stochastic_actor",
]
