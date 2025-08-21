from dataclasses import dataclass
from typing import Union, Optional
import torch
from typing import Dict
from torchrl.envs.utils import set_exploration_type, ExplorationType

@torch.no_grad()
def eval_policy(
    env,
    policy,
    steps: int = 1000,
    episodes: int = 1,
    deterministic: bool = True,
    *,
    progress: bool = True,
    desc: Optional[str] = "eval",
    leave: bool = False,
) -> Dict[str, float]:
    """
    Оценивает policy в среде env (TorchRL) через env.rollout.
    - Если среда батчевая, усредняет возврат по batch-осям.
    - progress=True включает tqdm-progressbar по эпизодам.
    ВАЖНО: имя 'eval' скрывает встроенную функцию Python eval() — это осознанно.

    Возвращает словарь метрик:
      eval_reward_mean, eval_reward_sum, eval_step_count, episodes
    """
    # tqdm (необязателен)
    try:
        from tqdm.auto import tqdm
        iterator = tqdm(
            range(episodes),
            desc=desc,
            total=episodes,
            leave=leave,
            disable=not progress,
        )
        is_tqdm = True
    except Exception:
        iterator = range(episodes)
        is_tqdm = False

    was_training = getattr(policy, "training", False)
    if deterministic and hasattr(policy, "eval"):
        policy.eval()

    episode_returns = []
    episode_lengths = []

    expl_type = ExplorationType.DETERMINISTIC if deterministic else ExplorationType.RANDOM

    for ep in iterator:
        with set_exploration_type(expl_type):
            td = env.rollout(steps, policy)

        # Возврат: суммируем по времени и усредняем по batch-осям (если есть)
        rew = td.get(("next", "reward"))  # shape ~ [T, *B, ...]
        ret = rew.sum(0)                  # суммарная награда на *B
        while ret.ndim > 0:
            ret = ret.mean(0)
        ret = float(ret.item())
        episode_returns.append(ret)

        # Длина эпизода (макс. step_count из rollout)
        ep_len = int(td.get("step_count").max().item())
        episode_lengths.append(ep_len)

        # Обновляем прогресс-бар
        if is_tqdm:
            iterator.set_postfix({"ret": f"{ret:.3f}", "len": ep_len})

        del td  # освобождаем память

    if was_training and hasattr(policy, "train"):
        policy.train()

    avg_return = float(sum(episode_returns) / len(episode_returns))
    sum_return = float(sum(episode_returns))
    max_len = int(max(episode_lengths))

    return {
        "return_mean": avg_return,
        "return_sum": sum_return,
        "max_episode_lengh": max_len,
        "num_episodes": episodes,
    }
