# AgentsLab (RL)

Стартовые скрипты обучения и оценки (Hydra + PyTorch, без Lightning/TorchRL, без обёрток).

## Установка
```bash
pip install -e ./src  # editable пакет agentslab
pip install torch gymnasium tensorboard hydra-core omegaconf
```

## Тренировка (DQN)
```bash
python scripts/rl/train_dqn.py rl=dqn env.id=CartPole-v1 rl.total_timesteps=50000
```
Параметры DQN берутся из `configs/rl/dqn.yaml`.

## Тренировка (REINFORCE)
```bash
python scripts/rl/train_reinforce.py rl=reinforce env.id=CartPole-v1 rl.total_episodes=500
```

## Оценка (evaluate)
```bash
python scripts/rl/evaluate.py model=dqn checkpoint_path=checkpoints/rl/dqn/CartPole-v1/<run>/best.pt env.id=CartPole-v1 episodes=10 render=false
```
Для REINFORCE используйте `model=reinforce`.

### Логи и чекпоинты
- Чекпоинты: `checkpoints/rl/<algo>/<env>/<run>/best.pt` и `last.pt` (+ `.meta.json`)
- TensorBoard: `logs/rl/<algo>/<env>/<run>/`
- Результаты: `results/rl/<algo>/<env>/<run>/summary.json`

Конфиг Hydra задаёт структуру `outputs/` и общий root проекта.
