
from __future__ import annotations
from tensordict.nn import TensorDictSequential, TensorDictModule
from torchrl.modules import QValueActor, EGreedyModule, ProbabilisticActor, TanhNormal
from tensordict.nn.distributions import NormalParamExtractor

def build_dqn_policy(q_net, action_spec, eps_init=1.0, eps_end=0.01, annealing_num_steps=5_000):
    return TensorDictSequential(
        QValueActor(module=q_net, spec=action_spec, in_keys=["observation"]),
        EGreedyModule(spec=action_spec, eps_init=eps_init, eps_end=eps_end, annealing_num_steps=annealing_num_steps),
    )

def build_gaussian_policy_for_marl(policy_net, action_key, action_spec_unbatched):
    # policy_net outputs concatenated parameters for Normal: (..., 2 * act_dim)
    # We first write them under ("agents","param"), then split into ("agents","loc"), ("agents","scale").
    param_module = TensorDictModule(
        policy_net,
        in_keys=[("agents", "observation")],
        out_keys=[("agents", "param")],
    )
    extractor = TensorDictModule(
        NormalParamExtractor(),
        in_keys=[("agents", "param")],
        out_keys=[("agents", "loc"), ("agents", "scale")],
    )
    module = TensorDictSequential(param_module, extractor)

    low = action_spec_unbatched[action_key].space.low
    high = action_spec_unbatched[action_key].space.high
    actor = ProbabilisticActor(
        module=module,
        spec=action_spec_unbatched,
        in_keys=[("agents", "loc"), ("agents", "scale")],
        out_keys=[action_key],
        distribution_class=TanhNormal,
        distribution_kwargs={"low": low, "high": high},
        return_log_prob=True,
    )
    return actor
