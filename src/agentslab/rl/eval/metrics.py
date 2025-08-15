
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import List, Dict, Any
import numpy as np

@dataclass
class EvalSummary:
    returns_mean: float
    returns_std: float
    returns_median: float
    returns_p05: float
    returns_p95: float
    lengths_mean: float
    lengths_std: float
    num_episodes: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

def summarize(returns: List[float], lengths: List[int]) -> EvalSummary:
    r = np.asarray(returns, dtype=np.float32)
    L = np.asarray(lengths, dtype=np.float32)
    return EvalSummary(
        returns_mean=float(r.mean()) if r.size else 0.0,
        returns_std=float(r.std(ddof=0)) if r.size else 0.0,
        returns_median=float(np.median(r)) if r.size else 0.0,
        returns_p05=float(np.percentile(r, 5)) if r.size else 0.0,
        returns_p95=float(np.percentile(r, 95)) if r.size else 0.0,
        lengths_mean=float(L.mean()) if L.size else 0.0,
        lengths_std=float(L.std(ddof=0)) if L.size else 0.0,
        num_episodes=int(len(returns)),
    )
