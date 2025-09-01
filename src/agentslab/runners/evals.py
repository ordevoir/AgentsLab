from typing import Optional, Dict
import torch
from torchrl.envs.utils import set_exploration_type, ExplorationType
from agentslab.envs.gym_envs import is_acts_discrete


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


@torch.no_grad()
def eval_policy_marl(
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
    Оценивает MARL policy в среде env (TorchRL) через env.rollout.
    - Усредняет возвраты по агентам и batch-осям.
    - progress=True включает tqdm-progressbar по эпизодам.
    
    Возвращает словарь метрик:
      return_mean, return_std, return_min, return_max, 
      return_sum, max_episode_length, num_episodes
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
        expl_type = ExplorationType.RANDOM

    policy_device = next(policy.parameters()).device
    env_device = getattr(env, "device", torch.device("cpu"))
    
    # создаем обертку над policy, чтобы td автоматически перегонялась по девайсам
    def policy_wrapper(td):
        td = td.to(policy_device)
        td = policy(td)
        return td.to(env_device)

    for ep in iterator:
        with set_exploration_type(expl_type):
            td = env.rollout(
                steps,
                policy_wrapper,
                auto_reset=True,
            )

        # Возврат для MARL: извлекаем награды агентов
        rew = td.get(("next", "agents", "reward"))  # shape ~ [T, *B, n_agents]
        ret = rew.sum(0)  # суммируем по времени -> [*B, n_agents]
        
        # Усредняем по агентам
        ret_per_agent = ret.mean(-1)  # -> [*B]
        
        # Усредняем по batch-осям (если есть)
        while ret_per_agent.ndim > 0:
            ret_per_agent = ret_per_agent.mean(0)
        
        ret_mean = float(ret_per_agent.item())
        episode_returns.append(ret_mean)

        # Длина эпизода - пробуем несколько вариантов
        ep_len = steps  # fallback значение
        
        # Пробуем разные возможные ключи для step_count
        step_count_candidates = [
            "step_count",
            ("agents", "step_count"),
            ("next", "step_count"),
            ("next", "agents", "step_count")
        ]
        
        for key in step_count_candidates:
            step_count = td.get(key)
            if step_count is not None:
                ep_len = int(step_count.max().item())
                break
        
        # Если step_count не найден, вычисляем из структуры tensordict
        if ep_len == steps:
            # Получаем фактическую длину из done flags
            try:
                done = td.get(("next", "agents", "done"))
                if done is not None:
                    # Находим первый done=True по временной оси
                    done_indices = done.any(-1).long()  # any по агентам
                    if done_indices.any():
                        ep_len = int(done_indices.argmax().item()) + 1
                    else:
                        ep_len = done.shape[0]  # если done не найден, берем полную длину
            except:
                # В крайнем случае берем размер временной оси
                ep_len = rew.shape[0]
        
        episode_lengths.append(ep_len)

        # Обновляем прогресс-бар
        if is_tqdm:
            iterator.set_postfix({"ret": f"{ret_mean:.3f}", "len": ep_len})

        del td  # освобождаем память

    if was_training and hasattr(policy, "train"):
        policy.train()

    # Вычисляем статистики
    episode_returns_tensor = torch.tensor(episode_returns)
    avg_return = float(episode_returns_tensor.mean().item())
    std_return = float(episode_returns_tensor.std().item()) if len(episode_returns) > 1 else 0.0
    min_return = float(episode_returns_tensor.min().item())
    max_return = float(episode_returns_tensor.max().item())
    sum_return = float(episode_returns_tensor.sum().item())
    max_len = int(max(episode_lengths))

    return {
        "return_mean": avg_return,
        "return_std": std_return,
        "return_min": min_return,
        "return_max": max_return,
        "return_sum": sum_return,
        "max_episode_length": max_len,
        "num_episodes": episodes,
    }

