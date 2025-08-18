from __future__ import annotations
import os, time, torch

class Checkpointer:
    def __init__(self, base_dir: str):
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)

    def save(self, *, step: int, policy, q_net, optimizer, loss_module, extra: dict | None = None) -> str:
        name = f"ckpt_step{step:08d}.pt"
        path = os.path.join(self.base_dir, name)
        payload = {
            "step": step,
            "q_net": q_net.state_dict(),
            "optimizer": optimizer.state_dict(),
            "loss_module": loss_module.state_dict(),
            "policy_eps": getattr(policy[-1], "eps", None),
            "extra": extra or {},
            "ts": time.time(),
        }
        torch.save(payload, path)
        return path

    def latest(self) -> str | None:
        files = [f for f in os.listdir(self.base_dir) if f.startswith("ckpt_step") and f.endswith(".pt")]
        if not files:
            return None
        files.sort()
        return os.path.join(self.base_dir, files[-1])

    @staticmethod
    def load_into(path: str, *, policy, q_net, optimizer, loss_module, map_location=None):
        data = torch.load(path, map_location=map_location)
        q_net.load_state_dict(data["q_net"])
        optimizer.load_state_dict(data["optimizer"])
        loss_module.load_state_dict(data["loss_module"])
        if data.get("policy_eps") is not None:
            try:
                policy[-1].eps = data["policy_eps"]
            except Exception:
                pass
        return data
