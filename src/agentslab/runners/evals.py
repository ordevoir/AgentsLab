from typing import Optional
import torch
from typing import Dict
from torchrl.envs.utils import set_exploration_type, ExplorationType
from agentslab.envs.gym_factory import is_acts_discrete

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
    policy.eval()

    episode_returns = []
    episode_lengths = []
    
    if deterministic:
        expl_type = ExplorationType.MODE if is_acts_discrete(env.action_spec) else ExplorationType.DETERMINISTIC
    else:
        expl_type= ExplorationType.RANDOM


    policy_device = next(policy.parameters()).device
    env_device = getattr(env, "device", torch.device("cpu"))
    
    # создаем обертку, над policy, чтобы td автоматически перегонялась по девайсам
    def policy_wrapper(td):
        td = td.to(policy_device)
        td = policy(td)
        return td.to(env_device)

    for ep in iterator:
        with set_exploration_type(expl_type):
            td = env.rollout(
                steps,
                policy_wrapper,    # <-- вместо policy
                auto_reset=True,
            )

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
        "max_episode_length": max_len,
        "num_episodes": episodes,
    }
