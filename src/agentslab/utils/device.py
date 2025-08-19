import torch

def resolve_device(preferred: str | torch.device = "cuda"):

    if preferred.lower() == "cuda":
        if torch.cuda.is_available():
            return torch.device(preferred)
        else:
            print("warning: cuda is not abvailable! cpu is using")
            return torch.device("cpu")
    if preferred.lower() == "cpu":
        return torch.device("cpu")
    else:
        raise ValueError('Select "cuda" or "cpu"!')
