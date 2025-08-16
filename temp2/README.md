
# AgentsLab

Исследовательская лаборатория для RL/MARL на базе **Gymnasium**, **PyTorch** и **Hydra**.  
В этом снэпшоте реализованы три single‑agent алгоритма для `CartPole-v1`:

- **REINFORCE** (vanilla policy gradient)  
- **DQN** (с реплеем, таргет-сетью, Huber loss)  
- **PPO** (clipped, дискретные действия, GAE)

> **Hydra** используется **только** на уровне CLI‑скриптов (`scripts/rl/*.py`).  
> Вся библиотека под `src/agentslab/...` — чистый PyTorch, без зависимостей от Hydra.

---

## Требования и установка

- Python >= 3.10
- Зависимости: `torch`, `gymnasium`, `hydra-core`, `tensorboard`, `tqdm`, `numpy` (устанавливаются через `pip` из `pyproject.toml`).

```bash
# из корня проекта
pip install -e .
```

(Опционально) Создайте окружение:
```bash
# conda
conda create -n agentslab python=3.10 -y
conda activate agentslab
pip install -e .
```

---

## Структура проекта (сокращённо)

```
AgentsLab/
├─ configs/                # Hydra-конфиги
│  ├─ config.yaml          # главный конфиг
│  ├─ common/
│  │  ├─ env/cartpole.yaml
│  │  ├─ train/base.yaml   # seed, timesteps, eval_interval, run_name, ckpt_root
│  │  └─ eval/base.yaml    # episodes, checkpoint_path, render_mode
│  └─ rl/
│     ├─ reinforce.yaml
│     ├─ dqn.yaml
│     └─ ppo.yaml
├─ src/agentslab/
│  ├─ core/                # сидинг, TB-логгер
│  ├─ networks/mlp.py
│  ├─ rl/
│  │  ├─ agents/           # REINFORCE, DQN, PPO
│  │  ├─ environments/     # фабрика gym
│  │  └─ training/         # тренеры + прогрессбар
│  └─ utils/checkpointing.py
├─ scripts/rl/
│  ├─ train.py             # @hydra.main (только здесь)
│  └─ evaluate.py          # @hydra.main (только здесь)
├─ logs/tb/                # TensorBoard логи
└─ checkpoints/rl/         # чекпоинты (см. ниже)
```

---

## Конфиги (Hydra)

Главный конфиг: `configs/config.yaml`

**Группы и дефолты:**
```yaml
defaults:
  - common/env: cartpole
  - common/logging: default
  - common/train: base
  - common/eval: base
  - rl: reinforce
  - _self_
```

Можно переопределять через CLI:
- Алгоритм: `rl={reinforce|dqn|ppo}`
- Тренировка: `common.train.*` (например, `total_timesteps`, `eval_interval`, `run_name`, `ckpt_root`, `seed`)
- Оценка: `common.eval.*` (например, `episodes`, `checkpoint_path`, `render_mode`)
- Окружение: `common.env.*` (`env_id`, `render_mode`).

> Если вы переопределяете ключ, которого **нет** в конфиге — используйте `+ключ=значение` (правило Hydra).

---

## Тренировка

Общий вид:
```bash
python scripts/rl/train.py rl=<algo>   common.train.total_timesteps=200000   common.train.eval_interval=10000   common.train.run_name=<ваше_имя_прогона>
```

### Примеры
**REINFORCE:**
```bash
python scripts/rl/train.py rl=reinforce   common.train.total_timesteps=100000   common.train.run_name=reinforce_cartpole_exp1
```

**DQN:**
```bash
python scripts/rl/train.py rl=dqn   common.train.total_timesteps=200000   rl.dqn.batch_size=128 rl.dqn.target_update_interval=2000   common.train.eval_interval=20000   common.train.run_name=dqn_cartpole_exp1
```

**PPO:**
```bash
python scripts/rl/train.py rl=ppo   rl.ppo.rollout_steps=2048 rl.ppo.update_epochs=4 rl.ppo.minibatch_size=64   common.train.total_timesteps=300000   common.train.run_name=ppo_cartpole_exp1
```

**Что выводится в консоль:** информативный `tqdm`‑progressbar.  
- REINFORCE: `R_mean`, `Lpi`  
- DQN: `R_mean`, `loss_q`, `eps`, `buf`  
- PPO: `R_mean`, `Lpi`, `Lv`, `H`

Логи **TensorBoard** пишутся в `logs/tb/` (см. раздел ниже).

---

## Чекпоинты

Во время тренировки чекпоинты сохраняются каждые `common.train.eval_interval` шагов:

```
checkpoints/rl/<algo>/<env_id>/<run_name>/step_<N>.pt
                                          └─ last.pt  # копия последнего
```
- `run_name`: по умолчанию `YYYYmmdd_HHMMSS_seed<seed>` (или задайте явно `common.train.run_name`).
- Внутри `.pt` хранится `meta`: `algorithm`, `env_id`, `model`, `seed`, `step`, `agent_cfg`.
- Рядом пишется `step_<N>.meta.json` — то же самое, но читаемо.

> Параметры путей стабилизированы относительно исходного CWD через `hydra.utils.get_original_cwd()` — чекпоинты и логи не «проваливаются» в `outputs/` Hydra.

---

## Оценка (evaluate)

Рекомендуемый способ: переопределять через группу `common.eval`.

```bash
# минимальный пример
python scripts/rl/evaluate.py rl=ppo

# уточнить чекпоинт и число эпизодов
python scripts/rl/evaluate.py rl=ppo   common.eval.checkpoint_path=checkpoints/rl/ppo/CartPole-v1/<run_name>/last.pt   common.eval.episodes=10

# включить отрисовку (если поддерживается)
python scripts/rl/evaluate.py rl=ppo common.eval.render_mode=human
```

Скрипт выводит информационный блок:
```
=== EVALUATE INFO ===
Checkpoint: <полный путь>
Algorithm:  <algo (из meta или конфигурации)>
Model:      <тип модели> (policy argmax | Q argmax)
Environment:<env_id>
Meta: {...}
```
Далее печатается возврат по каждому эпизоду.

**Совместимость:** если по каким‑то причинам хочется старым стилем — можно (ключей нет в структуре → нужен префикс `+`):
```bash
python scripts/rl/evaluate.py rl=ppo   +checkpoint_path=checkpoints/rl/ppo/CartPole-v1/<run_name>/last.pt   +episodes=10
```

> При наличии `meta.agent_cfg.hidden_sizes` сеть для оценки собирается по архитектуре из чекпоинта; иначе берётся из текущего конфига.

---

## TensorBoard

Логи пишутся в `logs/tb/`. Запуск:

**Bash / WSL / Linux / macOS**
```bash
tensorboard --logdir logs/tb --port 6006
```

**Windows PowerShell**
```powershell
tensorboard --logdir logs/tb --port 6006
```

Откройте в браузере: http://localhost:6006  
Что смотреть:
- `train/return` — эпизодные вознаграждения (REINFORCE/DQN) или средние по rollout (PPO)
- `loss/policy`, `loss/value`, `loss/entropy` (PPO/REINFORCE)
- `loss/q` (DQN)

Если хотите «начать чисто», удалите старую папку логов:
```bash
rm -rf logs/tb/*
```

---

## Подсказки и типичные ошибки

- **Hydra override grammar**: если ключа нет в конфиге, используйте `+ключ=значение`  
  Пример: `+checkpoint_path=...` или добавьте ключ в `common.eval` и переопределяйте без `+`.
- **Пути в Windows**: используйте прямые слэши `C:/...` или берите в кавычки.
- В `CartPole-v1` Gymnasium использует пару флагов `terminated|truncated` — это учтено в коде.
- Прогрессбар `tqdm` всегда включён в CLI‑скриптах и не влияет на TensorBoard.

---

## Дорожная карта

- Вариант на **TorchRL** (Collector, LossModules, Tensordict).
- Ветка **MARL** (PettingZoo/MPE, коммуникация/координация, SuperSuit‑обёртки).
- «Лучший чекпоинт» по метрике, экспорт метрик в `meta.json` и в каталог `results/`.

Приятных экспериментов! 🚀
