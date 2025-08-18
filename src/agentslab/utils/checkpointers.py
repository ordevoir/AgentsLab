import os, time, json, torch, datetime
from dataclasses import dataclass, asdict
from typing import Dict, Any

@dataclass
class CheckpointInfo:
    algo: str
    env_id: str
    run_name: str
    dir_root: str = "checkpoints"

    def make_run_dir(self) -> str:
        ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        name = f"{self.algo}_{self.env_id}_{self.run_name}_{ts}"
        run_dir = os.path.join(self.dir_root, name)
        os.makedirs(run_dir, exist_ok=True)
        return run_dir

def save_checkpoint(run_dir: str, actor, critic, optimizer, scheduler, extra: Dict[str, Any]):
    state = {
        "actor": actor.state_dict(),
        "critic": critic.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer else None,
        "scheduler": scheduler.state_dict() if scheduler else None,
        "extra": extra,
    }
    file = os.path.join(run_dir, "checkpoint.pt")
    torch.save(state, file)
    with open(os.path.join(run_dir, "meta.json"), "w") as f:
        json.dump(extra, f, indent=2)
    return file

def load_checkpoint(path: str, actor, critic, optimizer=None, scheduler=None):
    state = torch.load(path, map_location="cpu")
    actor.load_state_dict(state["actor"])
    critic.load_state_dict(state["critic"])
    if optimizer and state.get("optimizer"):
        optimizer.load_state_dict(state["optimizer"])
    if scheduler and state.get("scheduler"):
        scheduler.load_state_dict(state["scheduler"])
    return state.get("extra", {})
