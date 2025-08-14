"""
Экспортируем обученную политику в TorchScript (inference-only).
"""
from __future__ import annotations
import argparse
import torch
from agentslab.networks.mlp import MLP
from agentslab.agents.reinforce_agent import ReinforcePolicy
from agentslab.utils.checkpoints import load_checkpoint


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--obs-dim", type=int, required=True)
    ap.add_argument("--act-dim", type=int, required=True)
    ap.add_argument("--out", default="policy_scripted.pt")
    args = ap.parse_args()

    device = torch.device("cpu")
    policy = ReinforcePolicy(MLP(args.obs_dim, args.act_dim), device=device)
    load_checkpoint(policy, args.ckpt, map_location=device)
    policy.eval()

    scripted = torch.jit.script(policy)
    scripted.save(args.out)
    print(f"Saved TorchScript to {args.out}")


