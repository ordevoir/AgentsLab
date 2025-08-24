
"""
lab_pz_mpe2.py
--------------
Jupyter helpers for PettingZoo (incl. MPE) and mpe2:
- make_env_pz: create PettingZoo env (parallel if available)
- make_env_mpe2: create mpe2 env (parallel if available)
- play_episode_inline_parallel: run one episode (parallel API) with actor and display inline video
- rollout_episode_parallel: Collect logs and frames for parallel API
- play_episode_human_parallel: run with "human" window (if supported by the env)

Actor pattern:
- actor(observations_dict) -> actions_dict
  OR
- per-agent dict of callables: {agent: policy_fn}

If you have a single torch policy shared across agents, pass a callable that maps each obs to an action.
"""

from __future__ import annotations
from typing import Any, Callable, Dict, List, Optional, Union
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

logger = logging.getLogger("lab_pz_mpe2")
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

def _import_pz() -> Any:
    try:
        return importlib.import_module("pettingzoo")
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError("Install pettingzoo: pip install pettingzoo") from e

def _try_import(module: str) -> Any:
    try:
        return importlib.import_module(module)
    except ModuleNotFoundError:
        return None

def make_env_pz(
    module_id: str,  # e.g. "mpe.simple_spread_v3" or "butterfly.knights_archers_zombies_v10"
    seed: Optional[int] = None,
    render_mode: Optional[str] = "rgb_array",
    **env_kwargs: Any,
):
    """Return parallel_env() if available, else env() (AEC)."""
    pz = _import_pz()
    candidates = [
        f"pettingzoo.{module_id}",
        f"pettingzoo.mpe.{module_id}",
        f"pettingzoo.butterfly.{module_id}",
        f"pettingzoo.sisl.{module_id}",
        f"pettingzoo.mpe2.{module_id}",
    ]
    module = None
    for name in candidates:
        module = _try_import(name)
        if module is not None:
            break
    if module is None:
        raise ModuleNotFoundError(f"Cannot import PettingZoo module for '{module_id}'. Tried: {candidates}")

    kwargs = dict(env_kwargs)
    if render_mode is not None:
        kwargs.setdefault("render_mode", render_mode)

    make_parallel = getattr(module, "parallel_env", None)
    make_aec = getattr(module, "env", None)
    if make_parallel is not None:
        env = make_parallel(**kwargs)
    elif make_aec is not None:
        env = make_aec(**kwargs)
    else:
        raise AttributeError(f"Module '{module.__name__}' lacks parallel_env/env factory.")

    if seed is not None:
        try:
            env.reset(seed=seed)
        except TypeError:
            try:
                env.seed(seed)
            except Exception:
                pass
    return env

def make_env_mpe2(
    module_id: str,   # e.g. "simple_spread_v3"
    seed: Optional[int] = None,
    render_mode: Optional[str] = "rgb_array",
    **env_kwargs: Any,
):
    """mpe2 can be installed separately. Tries mpe2.<id> and pettingzoo.mpe2.<id>."""
    candidates = [
        f"mpe2.{module_id}",
        f"pettingzoo.mpe2.{module_id}",
    ]
    module = None
    for name in candidates:
        module = _try_import(name)
        if module is not None:
            break
    if module is None:
        raise ModuleNotFoundError(f"Cannot import mpe2 module for '{module_id}'. Tried: {candidates}")

    kwargs = dict(env_kwargs)
    if render_mode is not None:
        kwargs.setdefault("render_mode", render_mode)

    make_parallel = getattr(module, "parallel_env", None)
    make_aec = getattr(module, "env", None)
    if make_parallel is not None:
        env = make_parallel(**kwargs)
    elif make_aec is not None:
        env = make_aec(**kwargs)
    else:
        raise AttributeError(f"Module '{module.__name__}' lacks parallel_env/env factory.")

    if seed is not None:
        try:
            env.reset(seed=seed)
        except TypeError:
            try:
                env.seed(seed)
            except Exception:
                pass
    return env

def _to_numpy(x: Any) -> Any:
    if torch is not None and isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return x

def _resolve_actions(
    actor: Any,
    observations: Dict[str, Any],
    env: Any,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """
    actor could be:
      - callable(obs_dict)->act_dict
      - dict(agent->callable(obs)->act)
      - shared torch.nn.Module mapping each obs -> action
    """
    if callable(actor) and not (torch is not None and hasattr(torch.nn, "Module") and isinstance(actor, torch.nn.Module)):
        return {a: _to_numpy(actor({a: obs}) if _expects_dict(actor) else actor(obs)) for a, obs in observations.items()}

    if isinstance(actor, dict):
        acts = {}
        for a, obs in observations.items():
            pol = actor.get(a, None)
            if pol is None:
                raise KeyError(f"No policy provided for agent '{a}'")
            acts[a] = _to_numpy(pol(obs))
        return acts

    # Torch shared policy
    if torch is not None and hasattr(torch.nn, "Module") and isinstance(actor, torch.nn.Module):
        acts = {}
        dev = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        with torch.no_grad():
            for a, obs in observations.items():
                x = torch.as_tensor(obs, dtype=torch.float32, device=dev)
                if x.ndim == 1:
                    x = x.unsqueeze(0)
                out = actor(x)
                if hasattr(out, "sample") and callable(out.sample):
                    act = out.sample()
                else:
                    # Discrete vs Box: try env.action_space(a)
                    try:
                        import gymnasium as gym
                        if isinstance(env.action_space(a), gym.spaces.Discrete):
                            if isinstance(out, (tuple, list)):
                                out = out[0]
                            act = out.argmax(dim=-1)
                        else:
                            act = out
                    except Exception:
                        act = out
                act = act.squeeze(0)
                np_act = _to_numpy(act)
                # Convert to int for Discrete
                try:
                    import gymnasium as gym
                    if isinstance(env.action_space(a), gym.spaces.Discrete):
                        if isinstance(np_act, np.ndarray):
                            np_act = int(np_act.item())
                        elif torch is not None and isinstance(act, torch.Tensor):
                            np_act = int(act.item())
                except Exception:
                    pass
                acts[a] = np_act
        return acts

    # Fallback: assume callable per-obs
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

def rollout_episode_parallel(
    env: Any,
    actor: Any,
    max_steps: Optional[int] = None,
    capture_frames: bool = True,
) -> Dict[str, Any]:
    try:
        observations, infos = env.reset()
    except Exception:
        observations = env.reset()
        infos = {}

    frames: List[np.ndarray] = []
    total_reward_per_agent: Dict[str, float] = {a: 0.0 for a in getattr(env, "agents", [])}
    steps = 0

    terminations = {a: False for a in getattr(env, "agents", [])}
    truncations = {a: False for a in getattr(env, "agents", [])}

    while True:
        actions = _resolve_actions(actor, observations, env)
        observations, rewards, terminations, truncations, infos = env.step(actions)

        for a, r in rewards.items():
            total_reward_per_agent[a] = total_reward_per_agent.get(a, 0.0) + float(r)

        steps += 1

        if capture_frames:
            try:
                fr = env.render()
                if fr is not None:
                    frames.append(fr)
            except Exception:
                pass

        done = all(terminations.get(a, False) or truncations.get(a, False) for a in terminations.keys() | truncations.keys())
        if done or (max_steps is not None and steps >= max_steps):
            break

    return {"return_per_agent": total_reward_per_agent, "length": steps, "frames": frames, "last_info": infos}

def _display_video(frames: List[np.ndarray], fps: int = 30, prefix: str = "marl_episode") -> str:
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

def play_episode_inline_parallel(
    env: Any,
    actor: Any,
    fps: int = 30,
    max_steps: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Use an already-constructed parallel env (render_mode='rgb_array') to run and display a video.
    This avoids re-instantiating envs with heavy scenarios.
    """
    log = rollout_episode_parallel(env, actor, max_steps=max_steps, capture_frames=True)
    path = _display_video(log["frames"], fps=fps, prefix="pz_episode")
    return {"video_path": path, "return_per_agent": log["return_per_agent"], "length": log["length"]}

def play_episode_human_parallel(
    env: Any,
    actor: Any,
    max_steps: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Step with native window rendering ("human") if the env supports it.
    """
    log = rollout_episode_parallel(env, actor, max_steps=max_steps, capture_frames=False)
    return {"return_per_agent": log["return_per_agent"], "length": log["length"]}
