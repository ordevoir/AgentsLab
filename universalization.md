классный вопрос. краткий вердикт: сейчас всё работает для DQN/REINFORCE + CartPole, но при добавлении PPO/Actor-Critic и новых сред начнут «шевелиться» сразу тренеры, агенты, сети, коллекция данных и обёртки среды. Это лечится выделением **стабильных интерфейсов** и «плагинной» регистрации. Ниже — где узкие места и как сделать систему по-настоящему универсальной (и для MARL тоже).

---

# Где система станет хрупкой

* **Связность “тренер ↔ агент ↔ сборщик данных”**: сейчас логика обновления и сбора опыта частично «зашита» под конкретный алгоритм (on-policy vs off-policy).
* **Разнородные среды**: Gymnasium vs PettingZoo (MARL) дают разные сигнатуры `step/reset`. Без адаптера придётся трогать тренировочные циклы.
* **Сети и распределения действий**: дискрет/континуум, логполиси/энтропия/критик — если это не абстрагировано, каждый новый алгоритм тащит правки.
* **Логгинг/чекпоинты**: если нет единого менеджера состояния/метрик, новые агенты будут добавлять свой ad-hoc код.

---

# Цели дизайна

1. **Минимальные изменения при добавлении алгоритма**: один файл `ppo.py` с классом агента + конфиг.
2. **Единый интерфейс окружений**: один `EnvAdapter` для SA и один для MARL.
3. **Композиция тренировки**: тренер знает только «как собирать данные» и «как вызвать `agent.update()`».
4. **Регистры и фабрики**: выбор по строковому ключу из Hydra без условных if-лесов.
5. **Стандартизованные логгинг/чекпоинт/оценка**.

---

# Предлагаемая архитектура (слоями)

```
agentslab/
 ├─ core/
 │   ├─ envs/
 │   │   ├─ base_env.py        # интерфейсы + адаптеры: SA, MARL
 │   │   ├─ gym_adapter.py     # Gymnasium → BaseSingleEnv
 │   │   └─ pz_adapter.py      # PettingZoo → BaseMultiEnv
 │   ├─ data/
 │   │   ├─ replay_buffer.py   # уже перенесён
 │   │   ├─ collectors.py      # OnPolicyCollector, OffPolicyCollector, VecCollector
 │   │   └─ rollout.py         # типы переходов, батчи (SA/MARL)
 │   ├─ logging/
 │   │   ├─ logger.py          # MetricsLogger (TB + JSONL)
 │   │   └─ checkpoints.py     # CheckpointManager (state_dict + cfg)
 │   ├─ registry.py            # простые регистры: agents, networks, envs, trainers
 │   └─ utils.py               # сеяние, device, типы
 ├─ networks/
 │   ├─ policy_heads.py        # Categorical, DiagGaussian
 │   ├─ actor_critic.py        # общий A2C-head (π, V)
 │   ├─ mlp.py                 # MLP блоки
 │   └─ factory.py             # build_network(cfg) через registry/Hydra instantiate
 ├─ rl/
 │   ├─ agents/
 │   │   ├─ base.py            # BaseAgent API (SA), BaseAgentMA (MARL)
 │   │   ├─ dqn.py             # только логика алгоритма
 │   │   ├─ reinforce.py
 │   │   ├─ a2c.py             # будущий
 │   │   └─ ppo.py             # будущий
 │   ├─ trainers/
 │   │   ├─ base.py            # BaseTrainer API
 │   │   ├─ on_policy.py       # универсален для REINFORCE/A2C/PPO
 │   │   └─ off_policy.py      # универсален для DQN/DDPG/SAC
 │   └─ evaluate.py            # единый цикл оценки SA/MARL
 └─ marl/
     ├─ agents/                # MARL-алгоритмы (MAPPO и т.п.)
     ├─ trainers/              # MARL-тренеры (централ. критик/парам. sharing)
     └─ coordination/          # аллокация политик на агентов, коммуникация
```

---

## 1) Единые интерфейсы агентов

```python
# src/agentslab/rl/agents/base.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Mapping, Any, Protocol
import torch

@dataclass
class AgentOutput:
    action: torch.Tensor
    logprob: torch.Tensor | None = None
    value: torch.Tensor | None = None  # для actor-critic

class BaseAgent(Protocol):
    def act(self, obs) -> AgentOutput: ...
    def update(self, batch) -> Mapping[str, float]: ...
    def state_dict(self) -> dict[str, Any]: ...
    def load_state_dict(self, state: dict[str, Any]) -> None: ...
    def train_mode(self) -> None: ...
    def eval_mode(self) -> None: ...
```

* Для **MARL** аналогично, но `obs`/`action` — словари по агентам, либо батч с ключами.

**Как это помогает:** любой тренер, видя `BaseAgent`, умеет: (1) собрать трейсы, (2) вызвать `update`, (3) залогировать, (4) сохранить веса — без знания внутренностей.

---

## 2) Тренеры как композиция

```python
# src/agentslab/rl/trainers/base.py
class BaseTrainer(Protocol):
    def run(self) -> None: ...
```

```python
# src/agentslab/rl/trainers/on_policy.py
class OnPolicyTrainer:
    def __init__(self, env, agent, collector, logger, ckpt, cfg):
        self.env, self.agent = env, agent
        self.collector = collector        # GAE/returns могут считаться тут
        self.logger, self.ckpt = logger, ckpt
        self.cfg = cfg

    def run(self):
        for it in range(self.cfg.num_iterations):
            rollout = self.collector.collect(self.env, self.agent, steps=self.cfg.steps_per_iter)
            metrics = self.agent.update(rollout)
            self.logger.log_scalars(metrics, step=it)
            if it % self.cfg.ckpt_every == 0:
                self.ckpt.save(self.agent, step=it, metrics=metrics)
```

```python
# src/agentslab/rl/trainers/off_policy.py
class OffPolicyTrainer:
    def __init__(self, env, agent, collector, replay, logger, ckpt, cfg):
        self.env, self.agent = env, agent
        self.collector, self.replay = collector, replay
        self.logger, self.ckpt, self.cfg = logger, ckpt, cfg

    def run(self):
        self.collector.warmup(self.env, self.agent, self.replay, steps=self.cfg.warmup)
        for it in range(self.cfg.num_iterations):
            self.collector.interact(self.env, self.agent, self.replay, steps=self.cfg.steps_per_iter)
            for _ in range(self.cfg.grad_updates_per_iter):
                batch = self.replay.sample(self.cfg.batch_size)
                metrics = self.agent.update(batch)
                self.logger.log_scalars(metrics, step=it)
            if it % self.cfg.ckpt_every == 0:
                self.ckpt.save(self.agent, step=it, metrics=metrics)
```

**Как это помогает:** добавление PPO сводится к реализации `PPOAgent.update()` и выбору `OnPolicyTrainer` в конфиге — **ни один другой файл трогать не нужно**.

---

## 3) Коллекторы и роллауты

* `OnPolicyCollector`: собирает фиксированное число шагов/трансишнов, умеет считать returns/GAE (по конфигу).
* `OffPolicyCollector`: только взаимодействие и наполнение буфера.
* Варианты: одиночная среда, `gym.vector`/TorchRL-вектор, MARL-последовательности PettingZoo.

Типы батчей в `core/data/rollout.py` стандартизуют поля: `obs`, `action`, `reward`, `done`, `logprob`, `value`, `advantage`, `return`, … (для MARL — словари по агентам или батчи с измерением `actor`).

---

## 4) Адаптеры сред

```python
# src/agentslab/core/envs/base_env.py
class BaseSingleEnv(Protocol):
    def reset(self, seed: int | None = None): ...
    def step(self, action): ...
    @property
    def action_space(self): ...
    @property
    def observation_space(self): ...

class BaseMultiEnv(Protocol):  # PettingZoo API адаптирован до «батча» по агентам
    def reset(self, seed: int | None = None): ...
    def step(self, actions_dict): ...
    def agents(self) -> list[str]: ...
```

Фабрика окружений из конфига:

```python
env = build_env(cfg.rl.env)  # под капотом выбирается gym_adapter или pz_adapter
```

**Как это помогает:** тренеры не знают, Gym это или PettingZoo — они работают с единым интерфейсом.

---

## 5) Регистры и фабрики

Простой реестр на дикте (можно заменить на entry points при желании):

```python
# src/agentslab/core/registry.py
AGENTS: dict[str, type] = {}
TRAINERS: dict[str, type] = {}
NETWORKS: dict[str, type] = {}

def register_agent(name: str):
    def deco(cls):
        AGENTS[name] = cls
        return cls
    return deco
```

Пример регистрации:

```python
# src/agentslab/rl/agents/ppo.py
@register_agent("ppo")
class PPOAgent(BaseAgent):
    ...
```

В скрипте:

```python
agent = AGENTS[cfg.rl.agent.name](cfg.rl.agent, network=build_network(cfg.rl.network))
trainer = TRAINERS[cfg.rl.training.trainer_type](env, agent, ...)
```

**Как это помогает:** добавил `ppo.py` — и он «подхватился» без `if/elif` и без правок в других файлах.

---

## 6) Сети и распределения действий

* **Policy head-ы**: `CategoricalHead` (дискрет), `DiagGaussianHead` (континуум).
* **Actor-Critic общий блок** для A2C/PPO (общая шина фичей + два head-а).
* **Фабрика сетей** принимает `obs_space`, `action_space`, `cfg.network` и собирает нужное.
* Преобразование наблюдений (`np → torch`, нормализация) в одном месте (`core/data/convert.py`).

**Итог:** смена среды/размерности действий не требует переписывать агент — корректно выберется head.

---

## 7) Логи/чекпоинты — стандартизация

* `MetricsLogger`: единый API `log_scalar(s)`, `log_histogram(s)`, `flush()`. Под капотом — TensorBoard + JSONL.
* `CheckpointManager`: `save(agent, step, metrics)` и `load(agent, path)`. Всегда кладёт:

  * `agent_state_dict`, `optimizer_state_dict` (если есть),
  * `rng_state` (torch, numpy, random),
  * **снимок конфига** (yaml) — для полного воспроизведения.

**Итог:** скрипты `train.py`/`evaluate.py` одинаковые для всех алгоритмов.

---

## 8) Конфиги (Hydra) — стабильные группы

* `rl/agent/*` — только гиперы алгоритма (lr, γ, clip, λ и т.п.), `name: ppo`.
* `rl/training/*` — **тип тренера** (`trainer_type: on_policy|off_policy`), `num_iterations`, `steps_per_iter`, `grad_updates_per_iter`, `ckpt_every`, `device`.
* `rl/env/*` — `type: gym|pettingzoo`, `id`, `wrappers: [..]`, `vector: {num_envs: N}`.
* `rl/network/*` — `name: actor_critic_mlp`, `hidden_sizes`, `activation`, `init`.

Добавление нового алгоритма/среды — это **новый файл конфига** и один `.py` с реализацией.

---

## 9) MARL: шов там же

* `BaseAgentMA`: `act(obs_dict) -> action_dict`, `update(batch_ma) -> metrics`.
* **Контроллер политик** для параметр-шеринга: `agent_groups: {policy_A: [id0,id1,...]}`.
* **Централизованный критик**: в батче хранить `global_state` (если доступен) отдельно.
* MARL-тренеры используют тот же `OnPolicyTrainer`, только над `BaseMultiEnv` и `OnPolicyCollectorMA`.

**Результат:** MAPPO — это всего лишь новый класс агента + конфиг; тренер переиспользуется.

---

# Что менять уже сейчас (минимально, но с будущим ростом)

1. Ввести `BaseAgent`, `BaseTrainer`, адаптеры `BaseSingleEnv/BaseMultiEnv`.
2. Перенести «сбор данных» в `core/data/collectors.py` и отвязать от алгоритмов.
3. Сделать простые **регистры** (agents/trainers/networks).
4. Унифицировать `evaluate.py` вокруг `BaseAgent` + адаптеров (без if-ов).
5. В `networks/` выделить policy-heads и фабрику.
6. В конфиге обучения добавить `trainer_type` и `env.type`.

Эти шаги дадут «скелет», после которого добавление PPO — **один файл** + конфиг.

---

# Как будет выглядеть добавление нового агента (пример: PPO)

1. `src/agentslab/rl/agents/ppo.py`:

   * класс `PPOAgent(BaseAgent)` с `update(self, rollout)` (использует advantages/returns из collector), `act()` через policy head.
   * регистрация `@register_agent("ppo")`.
2. `configs/rl/agent/ppo.yaml` (clip\_coef, ent\_coef, vf\_coef, λ, num\_epochs, minibatch\_size…).
3. Запуск:

   ```bash
   python scripts/rl/train.py rl/agent=ppo rl/training=on_policy_vec rl/env=atari_pong_vec
   ```

Ни один другой файл трогать не нужно.

---

# Тестирование и стабильность (коротко)

* **Контракт-тесты**: для любого агента прогон 1–2 итерации on/off-policy тренера на фиктивной среде (CartPole и песочница MARL).
* **Smoke-тесты CLI**: `train.py` и `evaluate.py` с дефолтными конфигами.
* pre-commit: `ruff`, `black`, `mypy` (минимум на публичных интерфейсах).
* Регресс метрик — лёгкие «золотые» пороги (напр., средняя награда > X за N итераций на CartPole-v1).

---

# Заключение

Сейчас проект уже аккуратный, но расширение действительно начнёт трогать несколько мест. Чтобы сделать его **универсальным и «плагинным»**, достаточно:

* зафиксировать **интерфейсы** (агент/тренер/адаптер среды),
* вынести **коллекторы** и **policy-heads**,
* использовать **реестры** + Hydra-группы как единственную точку выбора компонентов,
* стандартизовать **логирование/чекпоинт**.

После этого любое добавление (PPO, A2C, новые среды, MARL-алго) — это **новый файл + конфиг**, без каскада правок.

Если хочешь, могу накидать минимальный PR-пакет с `Base*`, регистрами и on/off-policy тренерами, чтобы ты сразу увидел, как это склеивается поверх текущего кода.
