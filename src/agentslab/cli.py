from __future__ import annotations
import argparse
from typing import Optional
from agentslab.rl.training.dqn_train import DQNTrainConfig, train_dqn
from agentslab.rl.training.reinforce_train import ReinforceTrainConfig, train_reinforce
from agentslab.rl.agents.dqn import DQNConfig
from agentslab.rl.agents.reinforce import ReinforceConfig

def main() -> None:
    parser = argparse.ArgumentParser(prog="agentslab", description="AgentsLab CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # Train
    p_train = sub.add_parser("train", help="Train an RL agent")
    p_train.add_argument("--algo", choices=["dqn", "reinforce"], required=True)
    p_train.add_argument("--env-id", required=True)
    p_train.add_argument("--episodes", type=int, default=500)
    p_train.add_argument("--seed", type=int, default=1)
    p_train.add_argument("--device", default="auto", help="cpu|cuda|mps|auto")
    p_train.add_argument("--run-name", default=None, help="Optional run name suffix")

    # Eval
    p_eval = sub.add_parser("eval", help="Evaluate a trained agent")
    p_eval.add_argument("--algo", choices=["dqn", "reinforce"], required=True)
    p_eval.add_argument("--env-id", required=True)
    p_eval.add_argument("--checkpoint", required=True)
    p_eval.add_argument("--episodes", type=int, default=10)
    p_eval.add_argument("--device", default="auto")

    args = parser.parse_args()

    if args.cmd == "train":
        if args.algo == "dqn":
            train_cfg = DQNTrainConfig(episodes=args.episodes, seed=args.seed, device=args.device, run_name=args.run_name)
            agent_cfg = DQNConfig()  # relies on defaults; for fine-tuning, use Hydra configs
            train_dqn(args.env_id, train_cfg, agent_cfg)
        elif args.algo == "reinforce":
            train_cfg = ReinforceTrainConfig(episodes=args.episodes, seed=args.seed, device=args.device, run_name=args.run_name)
            agent_cfg = ReinforceConfig()
            train_reinforce(args.env_id, train_cfg, agent_cfg)
    elif args.cmd == "eval":
        # Expect the scripts/rl/evaluate.py to be used for rich eval with Hydra.
        from agentslab.rl.eval.loader import build_agent_from_cfg, load_weights
        from agentslab.rl.eval.eval_loop import eval_agent
        # Build agent with minimal defaults
        if args.algo == "dqn":
            agent_cfg = DQNConfig()
        else:
            agent_cfg = ReinforceConfig()
        agent = build_agent_from_cfg(args.algo, agent_cfg, args.env_id, args.device)
        load_weights(agent, args.checkpoint)
        eval_agent(agent, args.env_id, episodes=args.episodes, device=args.device)
    else:
        parser.error("Unknown command")

if __name__ == "__main__":
    main()
