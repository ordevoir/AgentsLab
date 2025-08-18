from __future__ import annotations
import os, time, uuid, torch
from pathlib import Path

def default_run_name(alg: str, env_name: str, seed: int | None = None) -> str:
    ts = time.strftime("%Y%m%d-%H%M%S")
    uid = uuid.uuid4().hex[:6]
    seed_txt = f"s{seed}" if seed is not None else "sNA"
    safe_env = env_name.replace("/", "-")
    return f"{ts}_{alg}_{safe_env}_{seed_txt}_{uid}"

class Checkpointer:
    def __init__(self, base_dir: str | os.PathLike, run_name: str):
        self.base = Path(base_dir) / run_name
        self.base.mkdir(parents=True, exist_ok=True)

    def save(self, **state_objs):
        for name, obj in state_objs.items():
            path = self.base / f"{name}.pt"
            if hasattr(obj, "state_dict"):
                torch.save(obj.state_dict(), path)
            else:
                torch.save(obj, obj)
        return self.base

    def load_into(self, **targets):
        for name, target in targets.items():
            path = self.base / f"{name}.pt"
            if path.exists():
                sd = torch.load(path, map_location="cpu")
                target.load_state_dict(sd)
