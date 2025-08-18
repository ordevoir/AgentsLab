from __future__ import annotations
from tensordict.nn import TensorDictSequential
from torchrl.modules import QValueActor, EGreedyModule

class DQNPolicyFactory:
    def __init__(self, eps_init: float, eps_end: float, anneal_steps: int) -> None:
        self.eps_init = eps_init
        self.eps_end = eps_end
        self.anneal_steps = anneal_steps

    def build(self, q_net, action_spec, in_key: str = "observation"):
        device = next(q_net.parameters()).device
        q_actor = QValueActor(q_net, in_keys=[in_key], spec=action_spec)
        eps_module = EGreedyModule(
            spec=action_spec,
            eps_init=self.eps_init,
            eps_end=self.eps_end,
            annealing_num_steps=self.anneal_steps,
            device=device,
        )
        return TensorDictSequential(q_actor, eps_module)
