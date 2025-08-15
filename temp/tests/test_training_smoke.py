from agentslab.environments.gym_wrapper import EnvConfig, make_env
from agentslab.networks.mlp import MLP
from agentslab.agents.reinforce_agent import ReinforcePolicy
from agentslab.training.trainer import Trainer, TrainCfg
from agentslab.training.callbacks import EarlySolvedStop
from agentslab.training.evaluator import evaluate as eval_loop
import torch


def test_smoke_runs_one_episode():
    env_cfg = EnvConfig(id="CartPole-v1")
    env = make_env(env_cfg)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    policy = ReinforcePolicy(MLP(obs_dim, act_dim), device=torch.device("cpu"))
    opt = torch.optim.Adam(policy.parameters(), lr=1e-2)
    trainer = Trainer(policy, env, opt, algo_cfg=type("A", (), {"gamma": 0.99}), logger=type("L", (), {"info": print}), callbacks=[], eval_fn=lambda p,n: [])
    trainer.train(TrainCfg(episodes=1, lr=1e-2, log_every=1, solved_threshold=500, solved_window=20, eval_every=999, eval_episodes=1), seed=0)



