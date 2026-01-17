from torchrl.envs.libs.vmas import VmasEnv
from torchrl.envs import TransformedEnv
from torchrl.envs.transforms import RewardSum

import matplotlib.pyplot as plt
from IPython.display import display, DisplayHandle, clear_output
import time
import torch
import gc

from dataclasses import dataclass, field, asdict, replace
from typing import Callable, Optional, Tuple, Any, Union, Dict


# -----------------------------
# Config dataclass for VMAS env
# -----------------------------
@dataclass(frozen=True)
class VMASEnvConfig:
    """Configuration for building a VMAS multi-agent environment in TorchRL.
    """
    scenario: str = "navigation"
    num_envs: int = 1
    continuous_actions: bool = True
    max_steps: int = 100
    # Use string for portability; resolved to torch.device at build time.
    device: str = "auto"  # "auto" | "cpu" | "cuda" | "mps" | CUDA index like "cuda:0"
    reward_sum_out_key: Tuple[str, str] = ("agents", "episode_reward")
    # Extra kwargs passed to the VMAS scenario (e.g., arena_size, goal_radius, etc.)
    scenario_kwargs: Dict[str, Any] = field(default_factory=dict)
    sum_rewards: bool = True

    # ---- Convenience methods ----
    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "VMASEnvConfig":
        return cls(**d)

    def override(self, **updates: Any) -> "VMASEnvConfig":
        """Return a copy with provided fields updated (handy for sweeps)."""
        return replace(self, **updates)

# ---------------
# Create VMAS env
# ---------------
def make_vmas_env(cfg: VMASEnvConfig):
    base_env = VmasEnv(
    scenario=cfg.scenario,
    num_envs=cfg.num_envs,
    continuous_actions=cfg.continuous_actions,
    max_steps=cfg.max_steps,
    device=cfg.device,
    # Scenario kwargs
    **cfg.scenario_kwargs,
    )

    if cfg.sum_rewards:
        env = TransformedEnv(
            base_env,
            RewardSum(in_keys=[base_env.reward_key], 
                    out_keys=[("agents", "episode_reward")],
                    reset_keys=base_env.reset_keys,  
                    ),
        )
    return env



# ---------
# Rendering
# ---------
class RenderingManager:
    """
    Менеджер для рендеринга VMAS environment с контролем FPS и восстановлением после ошибок
    """
    def __init__(self, fps=30, figsize=(8, 6)):
        """
        Args:
            fps: количество кадров в секунду
            figsize: размер фигуры matplotlib
        """
        self.fps = fps
        self.frame_time = 1.0 / fps  # время между кадрами
        self.figsize = figsize
        self.last_render_time = 0
        self.display_handle = None
        self.step_count = 0
        self.consecutive_errors = 0  # счетчик последовательных ошибок
        self.max_consecutive_errors = 5  # максимум ошибок подряд
        self.is_broken = False  # флаг сломанного состояния
        
    def reset(self):
        """Сброс состояния менеджера"""
        self.display_handle = None
        self.step_count = 0
        self.consecutive_errors = 0
        self.is_broken = False
        self.last_render_time = 0
        # Очищаем matplotlib
        plt.close('all')
        gc.collect()
        
    def rendering_callback(self, env, tensordict):
        """
        Callback функция для рендеринга во время rollout с обработкой ошибок
        
        Args:
            env: environment instance
            tensordict: текущее состояние environment (тензорный словарь)
        """
        # Если менеджер в сломанном состоянии, пропускаем рендеринг
        if self.is_broken:
            return
            
        # Контроль FPS - рендерим только если прошло достаточно времени
        current_time = time.time()
        if current_time - self.last_render_time < self.frame_time:
            return
        
        self.last_render_time = current_time
        self.step_count += 1
        
        try:
            # Сначала сбрасываем viewer если есть ошибки
            if self.consecutive_errors > 0:
                self._reset_environment_renderer(env)
            
            # Получаем RGB array из первого environment
            rgb_array = env.render(env_index=0, mode="rgb_array")
            
            # Проверяем валидность rgb_array
            if rgb_array is None or len(rgb_array.shape) != 3:
                raise ValueError("Invalid RGB array received")
            
            # Создаем фигуру
            fig, ax = plt.subplots(figsize=self.figsize)
            ax.imshow(rgb_array)
            ax.axis('off')  # убираем оси
            
            # Получаем step_count из tensordict если доступен
            step_info = tensordict.get('step_count', self.step_count)
            ax.set_title(f"VMAS Navigation Environment - Step {step_info}")
            plt.tight_layout()
            
            # Используем DisplayHandle для избежания мерцания
            if self.display_handle is None:
                self.display_handle = display(fig, display_id=True)
            else:
                self.display_handle.update(fig)
                
            plt.close(fig)  # закрываем фигуру чтобы избежать накопления
            
            # Сбрасываем счетчик ошибок при успешном рендеринге
            self.consecutive_errors = 0
            
        except Exception as e:
            self.consecutive_errors += 1
            error_msg = str(e)
            
            print(f"Ошибка при рендеринге (#{self.consecutive_errors}): {error_msg}")
            
            # Если слишком много ошибок подряд, отключаем рендеринг
            if self.consecutive_errors >= self.max_consecutive_errors:
                print(f"Слишком много ошибок рендеринга подряд. Отключаем визуализацию.")
                self.is_broken = True
                self._emergency_cleanup()
                return
            
            # Попытка восстановления при stack overflow
            if 'stack overflow' in error_msg.lower():
                print("Обнаружен stack overflow. Попытка восстановления...")
                self._recover_from_stack_overflow(env)
            
            # Очищаем matplotlib при любой ошибке
            plt.close('all')
    
    def _reset_environment_renderer(self, env):
        """Сброс renderer'а environment"""
        try:
            if hasattr(env, 'unwrapped'):
                env_unwrapped = env.unwrapped
                if hasattr(env_unwrapped, 'viewer'):
                    env_unwrapped.viewer = None
                if hasattr(env_unwrapped, '_viewers'):
                    env_unwrapped._viewers = {}
        except Exception as e:
            print(f"Не удалось сбросить renderer: {e}")
    
    def _recover_from_stack_overflow(self, env):
        """Восстановление после stack overflow"""
        try:
            # Полная очистка matplotlib
            plt.close('all')
            plt.clf()
            plt.cla()
            
            # Сброс display handle
            self.display_handle = None
            
            # Принудительный сброс renderer'а environment
            self._reset_environment_renderer(env)
            
            # Принудительная сборка мусора
            gc.collect()
            
            # Небольшая пауза
            time.sleep(0.1)
            
        except Exception as e:
            print(f"Ошибка при восстановлении: {e}")
    
    def _emergency_cleanup(self):
        """Экстренная очистка при критических ошибках"""
        try:
            plt.close('all')
            self.display_handle = None
            clear_output()
            gc.collect()
        except:
            pass

def rendering_callback(env, tensordict, fps=30, figsize=(8, 6)):
    """
    Простая callback функция для рендеринга во время rollout
    
    Args:
        env: environment instance
        tensordict: текущее состояние environment
        fps: количество кадров в секунду (frames per second)
        figsize: размер фигуры matplotlib
    """
    # Используем глобальный менеджер или создаем новый
    if not hasattr(rendering_callback, 'manager'):
        rendering_callback.manager = RenderingManager(fps=fps, figsize=figsize)
    
    rendering_callback.manager.rendering_callback(env, tensordict)

def reset_rendering():
    """
    Функция для сброса состояния рендеринга после ошибок
    """
    # Сбрасываем глобальный менеджер если существует
    if hasattr(rendering_callback, 'manager'):
        rendering_callback.manager.reset()
        delattr(rendering_callback, 'manager')
    
    # Очищаем matplotlib
    plt.close('all')
    clear_output()
    gc.collect()
    
    print("Состояние рендеринга сброшено. Можно запускать заново.")

def play_vmas(env, policy, max_steps=200, fps=30, figsize=(8, 6), verbose=True, reset_env=True):
    """
    Воспроизводит rollout с визуализацией в реальном времени
    
    Args:
        env: VMAS environment instance
        policy: политика для принятия решений
        max_steps: максимальное количество шагов
        fps: количество кадров в секунду
        figsize: размер фигуры для отображения
        verbose: выводить ли дополнительную информацию
        reset_env: сбрасывать ли environment перед запуском
        
    Returns:
        rollout_data: данные rollout'а
    """
    # Сброс рендеринга перед запуском
    reset_rendering()
    
    # Сброс environment если требуется
    if reset_env:
        try:
            env.reset()
            if verbose:
                print("Environment сброшен")
        except Exception as e:
            print(f"Предупреждение: не удалось сбросить environment: {e}")
    
    # Создаем новый менеджер рендеринга
    render_manager = RenderingManager(fps=fps, figsize=figsize)
    
    if verbose:
        print(f"Запуск rollout с визуализацией:")
        print(f"- Максимальное количество шагов: {max_steps}")
        print(f"- FPS: {fps}")
        print(f"- Размер фигуры: {figsize}")
    
    # Определяем callback функцию
    def callback(env, tensordict):
        render_manager.rendering_callback(env, tensordict)
    
    start_time = time.time()
    
    # Запускаем rollout с рендерингом
    with torch.no_grad():
        try:
            rollout_data = env.rollout(
                max_steps=max_steps,
                policy=policy,
                callback=callback,
                auto_cast_to_device=True,
                break_when_any_done=False,
            )
        except KeyboardInterrupt:
            print("\nRollout прерван пользователем")
            print("Выполняется очистка...")
            render_manager.reset()
            return None
        except Exception as e:
            print(f"Ошибка во время rollout: {e}")
            print("Выполняется очистка...")
            render_manager.reset()
            return None
    
    end_time = time.time()
    
    if verbose and not render_manager.is_broken:
        print(f"\nRollout завершен!")
        print(f"Форма данных rollout: {rollout_data.batch_size}")
        print(f"Время выполнения: {end_time - start_time:.2f} секунд")
        if render_manager.step_count > 0:
            print(f"Реальный FPS: {render_manager.step_count / (end_time - start_time):.1f}")
    elif render_manager.is_broken:
        print(f"\nRollout завершен с отключенной визуализацией")
    
    return rollout_data

# Дополнительная функция для принудительной очистки
def force_cleanup_rendering():
    """
    Принудительная очистка всех ресурсов рендеринга
    Используйте если обычный reset не помогает
    """
    import matplotlib
    
    # Сброс matplotlib бэкенда
    matplotlib.pyplot.close('all')
    
    # Очистка всех figure managers
    if hasattr(matplotlib._pylab_helpers, 'Gcf'):
        matplotlib._pylab_helpers.Gcf.figs.clear()
    
    # Сброс состояния рендеринга
    reset_rendering()
    
    print("Принудительная очистка выполнена")

# Пример использования:

# Вариант 1: Использование простой callback функции
# with torch.no_grad():
#     rollout_data = env.rollout(
#         max_steps=max_steps,
#         policy=policy,
#         callback=lambda env, td: rendering_callback(env, td, fps=24, figsize=(10, 8)),
#         auto_cast_to_device=True,
#         break_when_any_done=False,
#     )

# Вариант 2: Использование функции play (рекомендуемый)
# rollout_data = play(
#     env=env,
#     policy=policy,
#     max_steps=200,
#     fps=24,  # 24 кадра в секунду
#     figsize=(10, 8),
#     verbose=True
# )

# Вариант 3: Тонкая настройка с RenderingManager
# render_manager = RenderingManager(fps=60, figsize=(12, 9))
# 
# with torch.no_grad():
#     rollout_data = env.rollout(
#         max_steps=max_steps,
#         policy=policy,
#         callback=render_manager.rendering_callback,
#         auto_cast_to_device=True,
#         break_when_any_done=False,
#     )