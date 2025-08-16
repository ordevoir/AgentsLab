
from __future__ import annotations
import torch
import hydra
from omegaconf import DictConfig
import numpy as np
from agentslab.rl.environments.factory import make_env
from agentslab.utils.checkpointing import load_checkpoint
from agentslab.rl.agents.reinforce import ReinforceAgent, ReinforceConfig
from agentslab.rl.agents.dqn import QNetwork
from agentslab.rl.agents.ppo import ActorCritic

@hydra.main(config_path='../../configs', config_name='config', version_base=None)
def main(cfg: DictConfig) -> None:
    from hydra.utils import get_original_cwd
    import os, pprint
    env_id = cfg.common.env.env_id
    try:
        episodes = int(cfg.common.eval.episodes)
    except Exception:
        episodes = int(cfg.get('episodes', 10))
    root = get_original_cwd()
    try:
        checkpoint_path = os.path.join(root, cfg.common.eval.checkpoint_path)
    except Exception:
        checkpoint_path = os.path.join(root, cfg.get('checkpoint_path', 'checkpoints/rl/last.pt'))
    algo = cfg.rl.algorithm
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    render_mode = getattr(cfg.common.env, 'render_mode', None)
    try:
        render_mode = cfg.common.eval.render_mode or render_mode
    except Exception:
        pass
    env = make_env(env_id, render_mode=render_mode)

    ckpt = load_checkpoint(checkpoint_path)
    meta = ckpt.meta or {}
    meta_algo = meta.get("algorithm", algo)
    meta_env = meta.get("env_id", env_id)
    model_name = meta.get("model", "unknown")

    # Determine obs/act dims
    obs_shape = env.observation_space.shape
    assert obs_shape is not None
    obs_dim = int(np.prod(obs_shape))
    act_dim = env.action_space.n

    # Prefer hidden sizes from meta, else from cfg
    def get_hidden(default_list):
        hs = meta.get("agent_cfg", {}).get("hidden_sizes", None)
        if isinstance(hs, list):
            return tuple(int(x) for x in hs)
        return tuple(default_list)

    if meta_algo == 'reinforce' or (algo == 'reinforce' and model_name == 'ReinforceAgent'):
        agent = ReinforceAgent(obs_dim, act_dim, ReinforceConfig())
        agent.load_state_dict(ckpt.model_state)
        policy = agent.policy.to(device).eval()
        act_label = "policy argmax"
        def act_fn(obs):
            with torch.no_grad():
                logits = policy(torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0))
                action = int(torch.argmax(logits, dim=-1).item())
            return action
    elif meta_algo == 'dqn' or model_name == 'QNetwork':
        hidden = get_hidden(cfg.rl.dqn.hidden_sizes)
        q = QNetwork(obs_dim, act_dim, hidden).to(device)
        q.load_state_dict(ckpt.model_state)
        q.eval()
        act_label = "Q argmax"
        def act_fn(obs):
            with torch.no_grad():
                qvals = q(torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0))
                return int(torch.argmax(qvals, dim=-1).item())
    elif meta_algo == 'ppo' or model_name == 'ActorCritic':
        hidden = get_hidden(cfg.rl.ppo.hidden_sizes)
        ac = ActorCritic(obs_dim, act_dim, hidden).to(device)
        ac.load_state_dict(ckpt.model_state)
        ac.eval()
        act_label = "policy argmax"
        def act_fn(obs):
            with torch.no_grad():
                logits = ac.policy(torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0))
                action = int(torch.argmax(logits, dim=-1).item())
            return action
    else:
        raise ValueError(f"Unknown/unsupported checkpoint: algo={meta_algo}, model={model_name}")

    # Announce what we are using
    print("=== EVALUATE INFO ===")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Algorithm:  {meta_algo}")
    print(f"Model:      {model_name} ({act_label})")
    print(f"Environment:{meta_env}")
    if meta:
        print("Meta:", {k: v for k, v in meta.items() if k not in ['model_state', 'optimizer_state']})

    # Run episodes
    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        ep_return = 0.0
        while not done:
            action = act_fn(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ep_return += reward
        print(f"Episode {ep+1}: return={ep_return:.2f}")

if __name__ == '__main__':
    main()
