import torch

def select_device(prefer: str | int | None = None) -> torch.device:
    if isinstance(prefer, int):
        if torch.cuda.is_available():
            return torch.device(prefer)
    if prefer in {"cuda", "gpu"} and torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")

def split_devices(policy_device: torch.device, sim_device: torch.device | None = None):
    return policy_device, (sim_device or policy_device)
