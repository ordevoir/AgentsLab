import gymnasium as gym
def make_env(env_id, render_mode=None):
    return gym.make(env_id, render_mode=render_mode)
