
# Как задавать конфиги из CLI (Hydra)

Hydra позволяет:

* **Выбирать конфигурации из групп**: `rl/agent=...`, `rl/env=...`, `rl/network=...`, `rl/training=...`
* **Переопределять отдельные поля**: `rl.training.num_episodes=1000 rl.agent.lr=5e-4`
* **Делать мультизапуски (sweeps)**: `-m param1=a,b param2=1,2`

Hydra меняет рабочую директорию на `outputs/YYYY-MM-DD/HH-MM-SS/...`, но все пути логов/чекпоинтов/результатов в коде уже направлены в корень проекта через `utils/paths.py` (стабильно для любых запусков).

---

# Команды запуска: обучение и оценка

> Ниже — полностью рабочие примеры для твоей структуры.
> По умолчанию конфиг `configs/config.yaml` выбирает DQN + CartPole + `mlp_small` + `cartpole_fast`.

## Обучение

### 1) DQN (по умолчанию)

```bash
python scripts/rl/train.py
```

### 2) Явно указать группы

```bash
python scripts/rl/train.py rl/agent=dqn rl/env=cartpole rl/network=mlp_small rl/training=cartpole_fast
```

### 3) С переопределением гиперпараметров

```bash
python scripts/rl/train.py \
  rl/agent=dqn \
  rl.env.id=CartPole-v1 \
  rl.training.num_episodes=500 \
  rl.training.device=auto \
  rl.agent.lr=5e-4 \
  rl.network.hidden_sizes=[256,256]
```

### 4) REINFORCE

```bash
python scripts/rl/train.py rl/agent=reinforce
# при желании:
# rl.agent.entropy_coef=0.01 rl.training.num_episodes=400 ...
```

### 5) Мультизапуск (sweep) по lr и seed

```bash
python scripts/rl/train.py -m \
  rl/agent=dqn \
  rl.agent.lr=3e-4,1e-3 \
  rl.training.seed=0,1,2
```

## Оценка

> В `scripts/rl/evaluate.py` агент выбирается через **Hydra runtime choice** — нужно указать `rl/agent=...`, а также путь до чекпоинта.

### 1) DQN

```bash
python scripts/rl/evaluate.py \
  rl/agent=dqn \
  rl.eval.checkpoint_path=checkpoints/rl/dqn/<имя_запуска>/ep00xx_RYYY.Y.pt \
  rl.eval.num_episodes=50 \
  rl.eval.deterministic=true
```

### 2) REINFORCE

```bash
python scripts/rl/evaluate.py \
  rl/agent=reinforce \
  rl.eval.checkpoint_path=checkpoints/rl/reinforce/<имя_запуска>/ep00xx_RYYY.Y.pt \
  rl.eval.num_episodes=50
```

> Логи TensorBoard и метрики от оценки пишутся в `logs/rl/eval/<run_name>/`.
