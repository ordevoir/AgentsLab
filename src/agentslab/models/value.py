from __future__ import annotations
from tensordict.nn import TensorDictModule
from torchrl.modules import MLP

def make_value_net(in_dim: int, device=None):
    net = MLP(in_features=in_dim, out_features=1, num_cells=[256, 256], device=device)
    return TensorDictModule(net, in_keys=[("agents", "observation")], out_keys=[("agents", "state_value")])
