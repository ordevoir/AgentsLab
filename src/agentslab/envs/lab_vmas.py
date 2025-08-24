
"""
lab_vmas.py
-----------
Helpers for VMAS with inline playback in Jupyter.

Usage patterns:
- make_env_vmas(scenario="waterfall", num_envs=1, seed=0, continuous_actions=False, render_mode="rgb_array")
- rollout_episode_vmas(env, actor, ...) and play_episode_inline_vmas to display mp4/gif
- play_episode_human_vmas for native window (if VMAS build supports it)

Actor:
- callable(obs_dict)->act_dict for each env step (VMAS is vectorized over num_envs)
- or shared torch policy mapping each agent obs to action
"""

from __future__ import annotations
from typing import Any, Callable, Dict, List, Optional
import importlib
import os
import random
import sys
import logging
import numpy as np

try:
    import torch
except Exception:
    torch = None

logger = logging.getLogger("lab_vmas")
if not logger.handlers:
    _h = logging.StreamHandler(sys.stdout)
    _h.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(_h)
logger.setLevel(logging.INFO)

def set_global_seeds(seed: int, deterministic_torch: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if torch is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic_torch:
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except Exception:
                pass
            if getattr(torch.backends, "cudnn", None) is not None:
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False

def make_env_vmas(**kwargs: Any):
    """
    Common VMAS entry: vmas.make_env(**kwargs)
    Example kwargs: scenario="waterfall", num_envs=1, device="cpu", seed=0, continuous_actions=False, render_mode="rgb_array"
    """
    try:
        vmas = importlib.import_module("vmas")
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError("Install VMAS: pip install vmas") from e

    if hasattr(vmas, "make_env"):
        return vmas.make_env(**kwargs)
    # Fallback older API
    try:
        env_mod = importlib.import_module("vmas.simulator.environment")
        Environment = getattr(env_mod, "Environment", None)
        if Environment is None:
            raise AttributeError
        return Environment(**kwargs)
    except Exception as e:
        raise RuntimeError("Cannot construct VMAS env. Check your vmas version and kwargs.") from e

def _to_numpy(x: Any) -> Any:
    if torch is not None and isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return x

def _resolve_actions_vmas(actor: Any, observations: Dict[str, Any]) -> Dict[str, Any]:
    """
    VMAS observations/actions are dicts per agent with batched dims for num_envs.
    Actor could be callable(obs_dict)->act_dict or per-agent dict of callables.
    """
    if callable(actor):
        return {a: _to_numpy(actor({a: obs}) if _expects_dict(actor) else actor(obs)) for a, obs in observations.items()}
    if isinstance(actor, dict):
        return {a: _to_numpy(pol(observations[a])) for a, pol in actor.items()}
    # Torch nn.Module shared policy
    if torch is not None and hasattr(torch.nn, "Module") and isinstance(actor, torch.nn.Module):
        acts = {}
        with torch.no_grad():
            for a, obs in observations.items():
                dev = next(actor.parameters()).device if hasattr(actor, "parameters") else "cpu"
                x = torch.as_tensor(obs, dtype=torch.float32, device=dev)
                if x.ndim == 1:
                    x = x.unsqueeze(0)
                out = actor(x)
                if hasattr(out, "sample") and callable(out.sample):
                    act = out.sample()
                else:
                    act = out
                acts[a] = _to_numpy(act)
        return acts
    return {a: _to_numpy(actor(obs)) for a, obs in observations.items()}

def _expects_dict(fn: Callable) -> bool:
    import inspect
    try:
        sig = inspect.signature(fn)
        if len(sig.parameters) == 1:
            name = next(iter(sig.parameters.keys()))
            return "dict" in str(sig.parameters[name]).lower() or "obs_dict" in name
    except Exception:
        pass
    return False

def rollout_episode_vmas(
    env: Any,
    actor: Any,
    max_steps: Optional[int] = None,
    capture_frames: bool = True,
) -> Dict[str, Any]:
    # VMAS reset returns (observations, info)
    try:
        observations, infos = env.reset()
    except Exception:
        observations = env.reset()
        infos = {}

    frames: List[np.ndarray] = []
    steps = 0
    done = False
    # VMAS may return dicts of rewards/terminations/truncations per agent
    while not done:
        actions = _resolve_actions_vmas(actor, observations)
        observations, rewards, terminations, truncations, infos = env.step(actions)
        steps += 1

        if capture_frames:
            try:
                fr = env.render()
                if fr is not None:
                    frames.append(fr)
            except Exception:
                pass

        # Consider done if all agents done across all parallel envs
        # terminations/truncations are dicts per agent with shape (num_envs,)
        done = True
        keys = set(terminations.keys()) | set(truncations.keys())
        for a in keys:
            t = np.array(terminations.get(a, False))
            tr = np.array(truncations.get(a, False))
            if not np.all(t | tr):
                done = False
                break

        if max_steps is not None and steps >= max_steps:
            break

    return {"length": steps, "frames": frames, "last_info": infos}

def _display_video(frames: List[np.ndarray], fps: int = 30, prefix: str = "vmas_episode") -> str:
    if len(frames) == 0:
        raise ValueError("No frames captured. Ensure render_mode='rgb_array'.")
    import imageio
    out_mp4 = f"/mnt/data/{prefix}.mp4"
    try:
        with imageio.get_writer(out_mp4, format="ffmpeg", mode="I", fps=fps) as w:
            for fr in frames:
                w.append_data(fr)
        return out_mp4
    except Exception:
        out_gif = f"/mnt/data/{prefix}.gif"
        imageio.mimsave(out_gif, frames, duration=1.0 / fps)
        return out_gif

def play_episode_inline_vmas(env: Any, actor: Any, fps: int = 30, max_steps: Optional[int] = None) -> Dict[str, Any]:
    log = rollout_episode_vmas(env, actor, max_steps=max_steps, capture_frames=True)
    path = _display_video(log["frames"], fps=fps, prefix="vmas_episode")
    return {"video_path": path, "length": log["length"]}

def play_episode_human_vmas(env: Any, actor: Any, max_steps: Optional[int] = None) -> Dict[str, Any]:
    log = rollout_episode_vmas(env, actor, max_steps=max_steps, capture_frames=False)
    return {"length": log["length"]}
