from __future__ import annotations
from typing import Optional

from tensordict.nn import TensorDictSequential as Seq
from torchrl.modules import EGreedyModule

def build_dqn_policy(value_net, q_head, action_spec=None, epsilon_steps: int = 100_000, eps_init: float = 1.0, eps_end: float = 0.05):
    """Compose value_net + QValueModule (+ exploration) into two policies:
       - greedy policy (no exploration), used for training/eval
       - explore policy, with epsilon-greedy on top, used for data collection
    """
    if getattr(q_head, "spec", None) is None and action_spec is not None:
        q_head.spec = action_spec
    greedy = Seq(value_net, q_head)
    explore = Seq(
        greedy,
        EGreedyModule(
            spec=action_spec,
            annealing_num_steps=epsilon_steps,
            eps_init=eps_init,
            eps_end=eps_end,
        ),
    )
    return greedy, explore
