from typing import Any
from torchrl.envs.utils import check_env_specs

def print_specs(env: Any):
    print("Observation:", env.observation_spec)
    print("Action:", env.action_spec)
    print("Reward:", env.reward_spec)
    print("Done:", env.done_spec)
    if hasattr(env, "full_observation_spec"):
        print("Full observation:", env.full_observation_spec)
    try:
        check_env_specs(env)
        print("Specs OK.")
    except Exception as e:
        print("Spec check failed:", e)
