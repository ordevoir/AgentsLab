import torch

def pick_device(preferred: str | torch.device = "cuda"):
    if isinstance(preferred, str) and preferred.lower() == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(preferred)
