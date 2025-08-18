from torchrl.envs.utils import check_env_specs

def print_and_check_specs(env):
    print("observation_spec:", env.observation_spec)
    print("action_spec:", env.action_spec)
    print("reward_spec:", env.reward_spec)
    check_env_specs(env)
