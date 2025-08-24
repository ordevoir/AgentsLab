
"""
lab_gymnasium.py
----------------
Jupyter-friendly helpers for Gymnasium:
- make_env_gym: create (optionally seeded) env with chosen render_mode
- play_episode_inline: run one episode with a trained actor and display inline video (mp4 or gif)
- play_episode_human: run one episode with a trained actor and open a native window (render_mode="human")
- rollout_episode: utility that returns logs and (optionally) captured frames

Actor options:
- A plain Python callable: action = actor(obs)
- A torch.nn.Module: actor(observation_tensor) -> action tensor or distribution
  * If returns a Distribution, we sample()
  * If returns logits/tensor and action space is Discrete, we argmax
  * If Box space, we detach to numpy

Best practices:
- Deterministic seeds via set_global_seeds
- Lazy imports for optional deps
- Robust action sampling
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import importlib
import io
import os
import random
import sys
import logging

import numpy as np

# Optional torch
try:
    import torch
    import torch.nn as nn
except Exception:  # pragma: no cover
    torch = None
    nn = None

logger = logging.getLogger("lab_gymnasium")
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

def _import_gym() -> Any:
    try:
        return importlib.import_module("gymnasium")
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError("Install gymnasium: pip install gymnasium") from e

def make_env_gym(
    env_id: str,
    seed: Optional[int] = None,
    render_mode: Optional[str] = "rgb_array",
    autoreset: bool = True,
    **env_kwargs: Any,
):
    gym = _import_gym()
    env = gym.make(env_id, render_mode=render_mode, **env_kwargs)
    if seed is not None:
        try:
            env.reset(seed=seed)
            if hasattr(env.action_space, "seed"):
                env.action_space.seed(seed)
            if hasattr(env.observation_space, "seed"):
                env.observation_space.seed(seed)
        except TypeError:
            try:
                env.seed(seed)
            except Exception:
                pass
    if autoreset and hasattr(gym.wrappers, "AutoResetWrapper"):
        env = gym.wrappers.AutoResetWrapper(env)
    return env

def _to_numpy_action(action: Any) -> Any:
    if torch is not None and isinstance(action, torch.Tensor):
        return action.detach().cpu().numpy()
    return action

def _act_with_actor(
    actor: Any,
    obs: Any,
    action_space: Any,
    device: Optional[str] = None,
) -> Any:
    """Infer actions from various actor types."""
    # Plain callable
    if callable(actor) and (torch is None or not isinstance(actor, nn.Module)):
        return _to_numpy_action(actor(obs))

    # Torch actor
    if torch is not None and isinstance(actor, nn.Module):
        with torch.no_grad():
            x = torch.as_tensor(obs, dtype=torch.float32, device=device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
            # Add batch dim if needed
            if x.ndim == 1:
                x = x.unsqueeze(0)
            out = actor(x)
            # Distribution?
            if hasattr(out, "sample") and callable(out.sample):
                a = out.sample()
            else:
                # If Discrete -> argmax
                try:
                    gym = _import_gym()
                    if isinstance(action_space, gym.spaces.Discrete):
                        if isinstance(out, (tuple, list)):
                            out = out[0]
                        a = out.argmax(dim=-1)
                    else:
                        a = out
                except Exception:
                    a = out
            a = a.squeeze(0)
            a_np = _to_numpy_action(a)
            # Convert to Python int for Discrete actions
            try:
                gym = _import_gym()
                if isinstance(action_space, gym.spaces.Discrete):
                    if isinstance(a_np, np.ndarray):
                        return int(a_np.item())
                    elif torch is not None and isinstance(a, torch.Tensor):
                        return int(a.item())
            except Exception:
                pass
            return a_np
    # Fallback
    return _to_numpy_action(actor(obs))

def rollout_episode(
    env: Any,
    actor: Any,
    max_steps: Optional[int] = None,
    capture_frames: bool = True,
) -> Dict[str, Any]:
    gym = _import_gym()
    if not (hasattr(env, "reset") and hasattr(env, "step")):
        raise TypeError("Expected a Gymnasium-like env with reset/step.")

    try:
        obs, info = env.reset()
    except Exception:
        obs = env.reset()
        info = {}

    frames: List[np.ndarray] = []
    total_reward = 0.0
    steps = 0
    terminated = False
    truncated = False

    while True:
        action = _act_with_actor(actor, obs, env.action_space)
        out = env.step(action)
        if len(out) == 5:
            obs, reward, terminated, truncated, info = out
        else:
            obs, reward, done, info = out
            terminated, truncated = bool(done), False
        total_reward += float(reward)
        steps += 1

        if capture_frames:
            try:
                frame = env.render()
                if frame is not None:
                    frames.append(frame)
            except Exception:
                pass

        if terminated or truncated or (max_steps is not None and steps >= max_steps):
            break

    return {"return": total_reward, "length": steps, "frames": frames, "last_info": info}

def _display_video(frames: List[np.ndarray], fps: int = 30, prefix: str = "episode") -> str:
    """
    Save frames as MP4 (preferred) or GIF (fallback) and return file path.
    """
    if len(frames) == 0:
        raise ValueError("No frames captured. Ensure render_mode='rgb_array' and capture_frames=True.")
    h, w = frames[0].shape[:2]
    # Try MP4 via imageio-ffmpeg
    import imageio
    out_mp4 = f"/mnt/data/{prefix}.mp4"
    try:
        with imageio.get_writer(out_mp4, format="ffmpeg", mode="I", fps=fps) as writer:
            for fr in frames:
                writer.append_data(fr)
        return out_mp4
    except Exception:
        # Fallback to GIF
        out_gif = f"/mnt/data/{prefix}.gif"
        imageio.mimsave(out_gif, frames, duration=1.0 / fps)
        return out_gif

def play_episode_inline(
    env_id: str,
    actor: Any,
    seed: Optional[int] = None,
    fps: int = 30,
    max_steps: Optional[int] = None,
    **env_kwargs: Any,
) -> Dict[str, Any]:
    """
    Run one episode in rgb_array mode and display inline video in Jupyter.
    Returns the rollout log and the video path.
    """
    env = make_env_gym(env_id, seed=seed, render_mode="rgb_array", autoreset=True, **env_kwargs)
    log = rollout_episode(env, actor, max_steps=max_steps, capture_frames=True)
    env.close()
    path = _display_video(log["frames"], fps=fps, prefix=f"{env_id.replace('/', '_')}_episode")
    return {"video_path": path, "return": log["return"], "length": log["length"]}

def play_episode_human(
    env_id: str,
    actor: Any,
    seed: Optional[int] = None,
    max_steps: Optional[int] = None,
    **env_kwargs: Any,
) -> Dict[str, Any]:
    """
    Run one episode with a native window (render_mode='human').
    Useful when running locally with display server.
    """
    env = make_env_gym(env_id, seed=seed, render_mode="human", autoreset=True, **env_kwargs)
    log = rollout_episode(env, actor, max_steps=max_steps, capture_frames=False)
    env.close()
    return {"return": log["return"], "length": log["length"]}
