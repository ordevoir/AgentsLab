
# RL Research Lab — CartPole REINFORCE

## Быстрый старт

```bash
# обучение (Hydra)
python scripts/train.py train.episodes=500 net.hidden=64 train.lr=5e-3 env.id=CartPole-v1

# обучение с записью видео (требуется render_mode=rgb_array)
python scripts/train.py env.record_video=true env.render_mode=rgb_array

# мультизапуск (sweep) через Hydra
python scripts/train.py -m train.lr=1e-3,5e-3 net.hidden=32,64 seed=1,2,3

# оценка обученной модели
python scripts/evaluate.py --ckpt checkpoints/best.pt --env CartPole-v1
```