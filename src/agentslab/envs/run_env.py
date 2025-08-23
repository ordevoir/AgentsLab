# run_env.py
import time
import argparse

import torch

from agentslab.utils.device import resolve_device
from agentslab.utils.seeding import set_global_seed
from agentslab.envs.gym_factory import GymEnvConfig, make_gym_env


def _td_try(td, key):
    """Пробует вытащить ключ из td; поддерживает ('next', 'key')."""
    try:
        return td[key]
    except Exception:
        return None


def run_episode(env, fps=60, render_mode="human", video_writer=None):
    td = env.reset()
    ep_return = 0.0
    steps = 0

    while True:
        frame = env.render()
        if render_mode == "rgb_array" and video_writer is not None and frame is not None:
            # frame: HxWx3 uint8
            video_writer.append_data(frame)

        # случайное действие (для демо)
        td = env.rand_step(td)

        # в torchrl после шага награда и done обычно лежат в ("next", ...)
        rew = _td_try(td, ("next", "reward")) or _td_try(td, "reward")
        done = _td_try(td, ("next", "done")) or _td_try(td, "done")

        if rew is not None:
            ep_return += float(rew.item())
        steps += 1

        if done is not None and bool(done.item()):
            break

        if render_mode == "human":
            time.sleep(1.0 / max(fps, 1))

    return ep_return, steps


def main():
    parser = argparse.ArgumentParser(description="Run and visualize a Gymnasium env via TorchRL.")
    parser.add_argument("--env-id", type=str, default="InvertedDoublePendulum-v4")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fps", type=int, default=60, help="Влияет только на human/rgb_array задержку и видео FPS")
    parser.add_argument("--render", choices=["human", "rgb_array"], default="human",
                        help="human — окно; rgb_array — кадры для записи видео")
    parser.add_argument("--video", type=str, default=None,
                        help="Путь к mp4-файлу (используйте вместе с --render rgb_array)")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Жёсткий лимит шагов на эпизод (если нужен)")
    args = parser.parse_args()

    # девайс и сид
    device = resolve_device("cuda")
    print("Device:", device)
    set_global_seed(args.seed, deterministic=True)

    # конфиг среды; важное — render_mode
    env_cfg = GymEnvConfig(
        env_id=args.env_id,
        render_mode=args.render,      # "human" окно, "rgb_array" — массивы кадров
        norm_obs=True,
        init_norm_iter=1000,
        max_steps=args.max_steps,     # если хотите ограничить длину эпизода
        device=device,
        seed=args.seed,
    )
    env = make_gym_env(env_cfg)

    # Опционально — запись видео
    writer = None
    if args.render == "rgb_array" and args.video is not None:
        import imageio.v2 as imageio
        writer = imageio.get_writer(args.video, fps=args.fps)

    try:
        for ep in range(1, args.episodes + 1):
            ret, steps = run_episode(env, fps=args.fps, render_mode=args.render, video_writer=writer)
            print(f"Episode {ep}: return={ret:.3f}, steps={steps}")
    finally:
        if writer is not None:
            writer.close()
        env.close()


if __name__ == "__main__":
    main()
