from __future__ import annotations
from typing import Sequence

from tensordict.nn import TensorDictModule as Mod
from torchrl.modules import MLP, QValueModule

def build_qvalue_actor(n_actions: int, hidden: Sequence[int]=(256,256)):
    """Return a TensorDictModule computing action-values and a QValueModule head.
    Uses TorchRL's MLP with implicit input-size inference.
    """
    value_mlp = MLP(out_features=n_actions, num_cells=list(hidden))
    value_net = Mod(value_mlp, in_keys=["observation"], out_keys=["action_value"])
    q_head = QValueModule(spec=None)  # spec set later if available
    return value_net, q_head
