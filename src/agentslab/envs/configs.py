"""
Конфигурации сред для AgentsLab.

Этот модуль предоставляет dataclass-конфигурации для различных типов сред
обучения с подкреплением (reinforcement learning environments), совместимые
с TorchRL API.

Поддерживаемые бэкенды:
    - **Gymnasium** — single-agent среды (CartPole, MuJoCo, Atari, ...)
    - **VMAS** — векторизованные multi-agent среды на PyTorch
    - **PettingZoo** — multi-agent среды с AEC/Parallel API

Архитектура:
    Конфигурации разделены на два уровня:
    
    1. **EnvConfig** — параметры самой среды (scenario, num_envs, ...)
    2. **TransformConfig** — параметры трансформов (нормализация, rewards, ...)
    
    Это соответствует архитектуре TorchRL, где трансформы применяются
    к среде через ``TransformedEnv``.

Example:
    Создание конфигурации Gymnasium среды::
    
        from agentslab.configs import GymEnvConfig, TransformConfig
        
        config = GymEnvConfig(
            env_name="HalfCheetah-v4",
            device="cuda",
            transforms=TransformConfig(
                double_to_float=True,
                observation_norm=ObservationNormConfig(enabled=True),
            ),
        )
    
    Создание конфигурации VMAS среды::
    
        from agentslab.configs import VMASEnvConfig
        
        config = VMASEnvConfig(
            scenario="navigation",
            num_envs=64,
            continuous_actions=True,
            group_map="all",  # все агенты в одной группе
        )

See Also:
    - TorchRL Environments: https://pytorch.org/rl/stable/reference/envs.html
    - VMAS Documentation: https://vmas.readthedocs.io/
    - PettingZoo Documentation: https://pettingzoo.farama.org/

Note:
    Все конфигурации используют ``auto`` device по умолчанию, который
    автоматически выбирает CUDA > MPS > CPU в зависимости от доступности.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field, fields, replace
from typing import (
    Any,
    Dict,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
)

import torch
from tensordict.utils import NestedKey

try:
    from torchrl.envs.utils import MarlGroupMapType as TorchRLMarlGroupMapType
except ImportError:
    TorchRLMarlGroupMapType = None  # type: ignore[misc, assignment]

from agentslab.core.configs import resolve_device


__all__ = [
    # Типы
    "MarlGroupMapType",
    "BatchSizeType",
    # Конфигурации трансформов
    "ObservationNormConfig",
    "TransformConfig",
    # Конфигурации сред
    "BaseEnvConfig",
    "GymEnvConfig", 
    "VMASEnvConfig",
    "PettingZooEnvConfig",
    # Алиасы типов
    "EnvConfig",
    "SingleAgentEnvConfig",
    "MultiAgentEnvConfig",
    "VmasEnvConfig",
]


# ============================================================================
# Type Definitions
# ============================================================================

#: TypeVar для методов, возвращающих тот же тип конфигурации
T = TypeVar("T", bound="BaseEnvConfig")

#: Тип для группировки агентов в MARL средах.
#: 
#: Поддерживаемые значения:
#:   - ``"all"`` — все агенты в одной группе (cooperative)
#:   - ``"agent"`` — каждый агент в своей группе (independent)  
#:   - ``Dict[str, List[str]]`` — кастомная группировка
#:   - ``TorchRLMarlGroupMapType`` — нативный TorchRL enum
MarlGroupMapType = Union[
    Literal["all", "agent"],
    Dict[str, List[str]],
    Any,  # TorchRLMarlGroupMapType when available
]

#: Тип для размера батча.
#:
#: TorchRL использует ``torch.Size`` для batch dimensions, но мы принимаем
#: также list/tuple для удобства пользователя.
BatchSizeType = Union[torch.Size, Tuple[int, ...], List[int], None]


# ============================================================================
# Helper Functions  
# ============================================================================

def _to_torch_size(batch_size: BatchSizeType) -> Optional[torch.Size]:
    """
    Конвертирует batch_size в ``torch.Size``.
    
    TorchRL ожидает ``torch.Size`` для batch dimensions, но пользователю
    удобнее передавать list или tuple.
    
    Args:
        batch_size: Размер батча в любом поддерживаемом формате.
        
    Returns:
        ``torch.Size`` или ``None`` если входное значение ``None``.
        
    Example:
        >>> _to_torch_size([32])
        torch.Size([32])
        >>> _to_torch_size((64, 4))
        torch.Size([64, 4])
        >>> _to_torch_size(None)
        None
    """
    if batch_size is None:
        return None
    if isinstance(batch_size, torch.Size):
        return batch_size
    return torch.Size(batch_size)


def _normalize_nested_key(key: Any) -> NestedKey:
    """
    Нормализует ключ в формат TensorDict NestedKey.
    
    TensorDict поддерживает вложенные ключи как строки или кортежи.
    Эта функция приводит различные форматы к каноническому виду.
    
    Args:
        key: Ключ в формате ``str``, ``list``, или ``tuple``.
        
    Returns:
        Нормализованный ключ: ``str`` для простых ключей,
        ``tuple`` для вложенных.
        
    Example:
        >>> _normalize_nested_key("observation")
        'observation'
        >>> _normalize_nested_key(["agents", "obs"])
        ('agents', 'obs')
        >>> _normalize_nested_key(("agents", "action"))
        ('agents', 'action')
    """
    if isinstance(key, str):
        return key
    if isinstance(key, (list, tuple)):
        return tuple(key)
    return key


def _normalize_group_map(group_map: Any) -> Any:
    """
    Конвертирует строковые литералы группировки в TorchRL ``MarlGroupMapType``.
    
    Для удобства пользователя мы поддерживаем строковые литералы
    ``"all"`` и ``"agent"``, которые конвертируются в соответствующие
    значения TorchRL enum.
    
    Args:
        group_map: Конфигурация группировки агентов.
        
    Returns:
        - ``MarlGroupMapType.ALL_IN_ONE_GROUP`` для ``"all"``
        - ``MarlGroupMapType.ONE_GROUP_PER_AGENT`` для ``"agent"``
        - Без изменений для ``None``, ``dict``, или уже нормализованных значений
        
    Example:
        >>> _normalize_group_map("all")
        <MarlGroupMapType.ALL_IN_ONE_GROUP: ...>
        >>> _normalize_group_map({"team_a": ["agent_0", "agent_1"]})
        {'team_a': ['agent_0', 'agent_1']}
        
    Note:
        Если TorchRL не установлен или ``MarlGroupMapType`` недоступен,
        возвращает значение без изменений.
    """
    if TorchRLMarlGroupMapType is None or group_map is None:
        return group_map
    if group_map == "all":
        return TorchRLMarlGroupMapType.ALL_IN_ONE_GROUP
    if group_map == "agent":
        return TorchRLMarlGroupMapType.ONE_GROUP_PER_AGENT
    return group_map


# ============================================================================
# Transform Configurations
# ============================================================================

@dataclass
class ObservationNormConfig:
    """
    Конфигурация нормализации наблюдений (observations).
    
    Соответствует ``torchrl.envs.transforms.ObservationNorm`` — трансформу,
    который нормализует наблюдения среды по формуле:
    
    .. math::
        
        y = \\frac{x - \\mu}{\\sigma}
    
    где :math:`\\mu` (``loc``) и :math:`\\sigma` (``scale``) могут быть:
    
    - Заданы явно через параметры конфигурации
    - Вычислены автоматически через ``init_stats()`` на случайных роллаутах
    
    Attributes:
        enabled: Включить нормализацию. По умолчанию ``False``.
        in_keys: Ключи TensorDict для нормализации. 
            По умолчанию ``None`` — нормализуется ``"observation"``.
        out_keys: Ключи для записи результата. 
            По умолчанию совпадают с ``in_keys``.
        loc: Среднее значение :math:`\\mu`. Если ``None``, вычисляется
            автоматически через ``init_stats()``.
        scale: Стандартное отклонение :math:`\\sigma`. Если ``None``,
            вычисляется автоматически.
        standard_normal: Если ``True``, нормализует к стандартному
            нормальному распределению :math:`\\mathcal{N}(0, 1)`.
        num_iter: Количество итераций для инициализации статистик
            через случайные действия.
        cat_dim: Размерность для конкатенации при сборе статистик.
        reduce_dim: Размерности для редукции при вычислении mean/std.
        
    Example:
        Автоматическая инициализация статистик::
        
            norm_config = ObservationNormConfig(
                enabled=True,
                num_iter=1000,
            )
            
        Явное задание статистик (например, из предыдущего эксперимента)::
        
            norm_config = ObservationNormConfig(
                enabled=True,
                loc=torch.zeros(17),   # для MuJoCo HalfCheetah
                scale=torch.ones(17),
            )
            
        Нормализация нескольких ключей::
        
            norm_config = ObservationNormConfig(
                enabled=True,
                in_keys=[("agents", "observation"), "state"],
            )
    
    See Also:
        - ``torchrl.envs.transforms.ObservationNorm``
        - ``EnvBase.transform.init_stats()``
        
    References:
        https://pytorch.org/rl/stable/reference/generated/torchrl.envs.transforms.ObservationNorm.html
    """
    enabled: bool = False
    in_keys: Optional[Sequence[NestedKey]] = None
    out_keys: Optional[Sequence[NestedKey]] = None
    loc: Optional[Union[float, torch.Tensor]] = None
    scale: Optional[Union[float, torch.Tensor]] = None
    standard_normal: bool = False
    num_iter: int = 1000
    cat_dim: int = 0
    reduce_dim: Union[int, Tuple[int, ...]] = 0
    
    def __post_init__(self) -> None:
        """Валидация и нормализация параметров."""
        if self.num_iter < 0:
            raise ValueError(
                f"num_iter должен быть >= 0, получено {self.num_iter}"
            )
        # Нормализуем ключи
        if self.in_keys is not None:
            self.in_keys = tuple(_normalize_nested_key(k) for k in self.in_keys)
        if self.out_keys is not None:
            self.out_keys = tuple(_normalize_nested_key(k) for k in self.out_keys)


@dataclass
class TransformConfig:
    """
    Конфигурация трансформов среды.
    
    В TorchRL трансформы применяются к среде через ``TransformedEnv``
    и модифицируют данные на входе/выходе. Этот класс группирует
    конфигурации наиболее часто используемых трансформов.
    
    Attributes:
        observation_norm: Конфигурация нормализации наблюдений.
            См. :class:`ObservationNormConfig`.
        double_to_float: Конвертировать ``float64`` в ``float32``.
            Рекомендуется для совместимости с PyTorch моделями.
        step_counter: Добавить счётчик шагов в TensorDict.
            Полезно для отладки и логирования.
        max_steps: Максимальное количество шагов в эпизоде.
            Если задано, добавляется ``StepCounter`` transform с truncation.
        reward_sum: Накапливать суммарную награду за эпизод.
            Используется в MARL для отслеживания episode return.
        reward_sum_key: Ключ для записи суммарной награды.
            По умолчанию ``("agents", "episode_reward")`` для MARL.
            
    Example:
        Базовая конфигурация для MuJoCo::
        
            transforms = TransformConfig(
                double_to_float=True,
                observation_norm=ObservationNormConfig(enabled=True),
                max_steps=1000,
            )
            
        Конфигурация для MARL::
        
            transforms = TransformConfig(
                double_to_float=True,
                reward_sum=True,
                reward_sum_key=("agents", "episode_reward"),
            )
            
    Note:
        Порядок применения трансформов в TorchRL имеет значение.
        Типичный порядок:
        
        1. ``DoubleToFloat`` — конвертация типов
        2. ``ObservationNorm`` — нормализация
        3. ``StepCounter`` — подсчёт шагов
        4. ``RewardSum`` — накопление наград
        
    See Also:
        - ``torchrl.envs.transforms.TransformedEnv``
        - ``torchrl.envs.transforms.Compose``
    """
    observation_norm: Optional[ObservationNormConfig] = None
    double_to_float: bool = True
    step_counter: bool = False
    max_steps: Optional[int] = None
    reward_sum: bool = False
    reward_sum_key: NestedKey = ("agents", "episode_reward")
    
    def __post_init__(self) -> None:
        """Валидация и нормализация параметров."""
        # Конвертируем dict в ObservationNormConfig если нужно
        if isinstance(self.observation_norm, dict):
            self.observation_norm = ObservationNormConfig(**self.observation_norm)
            
        if self.max_steps is not None and self.max_steps <= 0:
            raise ValueError(
                f"max_steps должен быть > 0 или None, получено {self.max_steps}"
            )
            
        self.reward_sum_key = _normalize_nested_key(self.reward_sum_key)


# ============================================================================
# Base Environment Configuration
# ============================================================================

@dataclass
class BaseEnvConfig(ABC):
    """
    Абстрактная базовая конфигурация среды.
    
    Определяет общий интерфейс и параметры для всех типов сред.
    Наследники должны реализовать property ``env_name``.
    
    Attributes:
        device: Устройство для вычислений. Поддерживаемые значения:
            - ``"auto"`` — автоматический выбор (CUDA > MPS > CPU)
            - ``"cuda"`` / ``"cuda:0"`` — NVIDIA GPU
            - ``"mps"`` — Apple Silicon GPU
            - ``"cpu"`` — CPU
            - ``torch.device`` — явный device object
        seed: Random seed для воспроизводимости. Передаётся в ``env.reset(seed=...)``.
        transforms: Конфигурация трансформов. См. :class:`TransformConfig`.
        
    Example:
        Этот класс абстрактный и не может быть инстанциирован напрямую::
        
            # Это вызовет ошибку:
            config = BaseEnvConfig()  # TypeError: Can't instantiate abstract class
            
            # Используйте конкретные классы:
            config = GymEnvConfig(env_name="CartPole-v1")
            
    Note:
        Все наследники должны вызывать ``super().__post_init__()``
        для корректной инициализации device и transforms.
    """
    device: Union[str, torch.device, None] = "auto"
    seed: Optional[int] = None
    transforms: TransformConfig = field(default_factory=TransformConfig)
    
    def __post_init__(self) -> None:
        """
        Инициализация после создания dataclass.
        
        Выполняет:
            1. Резолвинг device через ``resolve_device()``
            2. Конвертацию dict в TransformConfig если необходимо
        """
        self.device = resolve_device(self.device)
        if isinstance(self.transforms, dict):
            self.transforms = TransformConfig(**self.transforms)
    
    @property
    @abstractmethod
    def env_name(self) -> str:
        """
        Унифицированный идентификатор среды.
        
        Возвращает строку, однозначно идентифицирующую среду.
        Формат зависит от типа среды:
        
        - Gymnasium: ``"CartPole-v1"``, ``"HalfCheetah-v4"``
        - VMAS: ``"vmas/navigation"``, ``"vmas/transport"``
        - PettingZoo: ``"pettingzoo/simple_spread_v3"``
        
        Returns:
            Строковый идентификатор среды.
        """
        ...
    
    @property
    def is_multi_agent(self) -> bool:
        """
        Является ли среда мультиагентной.
        
        Returns:
            ``True`` для VMAS и PettingZoo, ``False`` для Gymnasium.
            
        Example:
            >>> GymEnvConfig(env_name="CartPole-v1").is_multi_agent
            False
            >>> VMASEnvConfig(scenario="navigation").is_multi_agent
            True
        """
        return False
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Сериализует конфигурацию в JSON-совместимый словарь.
        
        Конвертирует не-JSON типы:
            - ``torch.device`` → ``str``
            - ``torch.Tensor`` → ``list``
            - ``tuple`` → ``list``
            - Вложенные dataclasses → ``dict``
        
        Returns:
            Словарь с примитивными типами, пригодный для JSON/YAML.
            
        Example:
            >>> config = GymEnvConfig(env_name="CartPole-v1", device="cuda")
            >>> config.to_dict()
            {'env_name': 'CartPole-v1', 'device': 'cuda', ...}
            
        See Also:
            - :meth:`from_dict` — обратная операция
        """
        def jsonable(x: Any) -> Any:
            if isinstance(x, torch.device):
                return str(x)
            if isinstance(x, torch.Tensor):
                return x.detach().cpu().tolist()
            if hasattr(x, "__dataclass_fields__"):
                return asdict(x)
            if isinstance(x, dict):
                return {str(k): jsonable(v) for k, v in x.items()}
            if isinstance(x, (list, tuple)):
                return [jsonable(v) for v in x]
            return x
            
        return {k: jsonable(v) for k, v in asdict(self).items()}
    
    @classmethod
    def from_dict(cls: Type[T], data: Dict[str, Any]) -> T:
        """
        Создаёт конфигурацию из словаря.
        
        Фильтрует неизвестные ключи, что позволяет загружать
        конфигурации с дополнительными полями (forward compatibility).
        
        Args:
            data: Словарь с параметрами конфигурации.
            
        Returns:
            Новый экземпляр конфигурации.
            
        Example:
            >>> data = {"env_name": "Pendulum-v1", "device": "cpu", "unknown": 123}
            >>> config = GymEnvConfig.from_dict(data)
            >>> config.env_name
            'Pendulum-v1'
            
        Warning:
            Неизвестные ключи игнорируются без предупреждения.
            Используйте :meth:`to_dict` для полной сериализации.
        """
        allowed_fields = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in data.items() if k in allowed_fields}
        return cls(**filtered)
    
    def override(self: T, **updates: Any) -> T:
        """
        Создаёт копию конфигурации с изменёнными полями.
        
        Иммутабельный паттерн: оригинал не изменяется.
        
        Args:
            **updates: Поля для переопределения.
            
        Returns:
            Новый экземпляр с обновлёнными полями.
            
        Example:
            >>> base = GymEnvConfig(env_name="CartPole-v1", seed=42)
            >>> modified = base.override(seed=123, device="cpu")
            >>> base.seed
            42
            >>> modified.seed
            123
            
        Raises:
            TypeError: Если передано несуществующее поле.
        """
        return replace(self, **updates)


# ============================================================================
# Gymnasium Environment Configuration
# ============================================================================

@dataclass
class GymEnvConfig(BaseEnvConfig):
    """
    Конфигурация среды Gymnasium (single-agent).
    
    Gymnasium (бывший OpenAI Gym) — стандартный интерфейс для single-agent
    сред обучения с подкреплением. Этот класс соответствует параметрам
    ``torchrl.envs.libs.gym.GymEnv``.
    
    Attributes:
        env_name: Идентификатор среды в реестре Gymnasium.
            Примеры: ``"CartPole-v1"``, ``"HalfCheetah-v4"``, ``"Pong-v5"``.
        batch_size: Размер батча для векторизации.
            - ``None`` — одиночная среда
            - ``[N]`` — N параллельных сред (AsyncVectorEnv)
            Пример: ``batch_size=[32]`` создаст 32 параллельные среды.
        categorical_action_encoding: Использовать one-hot encoding для
            дискретных действий вместо индексов.
        frame_skip: Количество пропускаемых кадров. Действие повторяется
            ``frame_skip`` раз, награды суммируются.
        render_mode: Режим рендеринга: ``"human"``, ``"rgb_array"``, ``None``.
        gym_kwargs: Дополнительные аргументы для ``gymnasium.make()``.
        
    Example:
        Простая среда::
        
            config = GymEnvConfig(env_name="CartPole-v1")
            
        MuJoCo с векторизацией::
        
            config = GymEnvConfig(
                env_name="HalfCheetah-v4",
                batch_size=[64],  # 64 параллельные среды
                device="cuda",
                transforms=TransformConfig(
                    double_to_float=True,
                    observation_norm=ObservationNormConfig(enabled=True),
                ),
            )
            
        Atari с frame skip::
        
            config = GymEnvConfig(
                env_name="PongNoFrameskip-v4",
                frame_skip=4,
                gym_kwargs={"full_action_space": False},
            )
            
    Note:
        В TorchRL параметр называется ``env_name``, а не ``env_id``.
        Мы следуем этому соглашению для совместимости.
        
    See Also:
        - ``torchrl.envs.libs.gym.GymEnv``
        - https://gymnasium.farama.org/
        
    References:
        https://pytorch.org/rl/stable/reference/generated/torchrl.envs.GymEnv.html
    """
    # Переопределяем env_name как поле (не property)
    env_name: str = field(default="CartPole-v1")  # type: ignore[assignment]
    
    # Векторизация
    batch_size: BatchSizeType = None
    
    # GymEnv-специфичные параметры
    categorical_action_encoding: bool = False
    frame_skip: int = 1
    
    # Рендеринг
    render_mode: Optional[str] = None
    
    # Дополнительные kwargs для gymnasium.make()
    gym_kwargs: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self) -> None:
        """
        Валидация параметров Gymnasium среды.
        
        Raises:
            ValueError: Если ``env_name`` пустой или ``frame_skip < 1``.
        """
        super().__post_init__()
        
        # Валидация env_name
        if not self.env_name or not self.env_name.strip():
            raise ValueError("env_name не может быть пустым")
            
        # Валидация frame_skip
        if self.frame_skip < 1:
            raise ValueError(
                f"frame_skip должен быть >= 1, получено {self.frame_skip}"
            )
            
        # Конвертация batch_size в torch.Size
        self.batch_size = _to_torch_size(self.batch_size)
    
    @property
    def is_vectorized(self) -> bool:
        """
        Является ли среда векторизованной.
        
        Returns:
            ``True`` если ``batch_size`` задан и не пустой.
            
        Example:
            >>> GymEnvConfig(env_name="CartPole-v1").is_vectorized
            False
            >>> GymEnvConfig(env_name="CartPole-v1", batch_size=[32]).is_vectorized
            True
        """
        return self.batch_size is not None and len(self.batch_size) > 0


# ============================================================================
# VMAS Environment Configuration  
# ============================================================================

@dataclass
class VMASEnvConfig(BaseEnvConfig):
    """
    Конфигурация среды VMAS (Vectorized Multi-Agent Simulator).
    
    VMAS — это векторизованный симулятор для мультиагентного обучения,
    написанный полностью на PyTorch. Он позволяет запускать тысячи
    параллельных симуляций на GPU.
    
    Attributes:
        scenario: Название сценария VMAS.
            Встроенные: ``"navigation"``, ``"transport"``, ``"balance"``,
            ``"wheel"``, ``"discovery"``, ``"flocking"``, и др.
        num_envs: Количество параллельных сред. VMAS всегда векторизован,
            минимальное значение — 1.
        continuous_actions: Тип действий агентов.
            - ``True`` — непрерывные действия (Box space)
            - ``False`` — дискретные действия (Discrete space)
        max_steps: Максимальное количество шагов в эпизоде.
            VMAS требует явного указания горизонта.
        group_map: Группировка агентов для обучения.
            - ``"all"`` — все агенты в одной группе (parameter sharing)
            - ``"agent"`` — каждый агент независим
            - ``Dict[str, List[str]]`` — кастомные группы
        scenario_kwargs: Параметры конкретного сценария.
            Например, ``{"n_agents": 5, "n_targets": 3}``.
            
    Example:
        Базовая конфигурация::
        
            config = VMASEnvConfig(
                scenario="navigation",
                num_envs=64,
                device="cuda",
            )
            
        Кастомный сценарий с параметрами::
        
            config = VMASEnvConfig(
                scenario="transport",
                num_envs=128,
                continuous_actions=True,
                max_steps=200,
                group_map="all",  # cooperative
                scenario_kwargs={
                    "n_agents": 4,
                    "package_mass": 1.0,
                },
            )
            
        Независимые агенты (IPPO)::
        
            config = VMASEnvConfig(
                scenario="simple_spread",
                num_envs=32,
                group_map="agent",  # independent learning
            )
            
    Note:
        VMAS всегда запускается в батчевом режиме. Даже ``num_envs=1``
        создаст среду с batch dimension.
        
    See Also:
        - ``torchrl.envs.libs.vmas.VmasEnv``
        - https://vmas.readthedocs.io/
        - https://github.com/proroklab/VectorizedMultiAgentSimulator
        
    References:
        https://pytorch.org/rl/stable/reference/generated/torchrl.envs.VmasEnv.html
    """
    scenario: str = "navigation"
    num_envs: int = 32
    continuous_actions: bool = True
    max_steps: Optional[int] = 100
    group_map: Optional[MarlGroupMapType] = None
    scenario_kwargs: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self) -> None:
        """
        Валидация параметров VMAS среды.
        
        Raises:
            ValueError: Если ``scenario`` пустой или ``num_envs < 1``.
        """
        super().__post_init__()
        
        if not self.scenario or not self.scenario.strip():
            raise ValueError("scenario не может быть пустым")
            
        if self.num_envs < 1:
            raise ValueError(
                f"num_envs должен быть >= 1, получено {self.num_envs}"
            )
            
        # Нормализуем group_map к TorchRL формату
        self.group_map = _normalize_group_map(self.group_map)
    
    @property
    def env_name(self) -> str:
        """
        Унифицированный идентификатор среды.
        
        Returns:
            Строка в формате ``"vmas/{scenario}"``.
            
        Example:
            >>> VMASEnvConfig(scenario="navigation").env_name
            'vmas/navigation'
        """
        return f"vmas/{self.scenario}"
    
    @property
    def is_multi_agent(self) -> bool:
        """VMAS всегда мультиагентный."""
        return True
    
    @property
    def kwargs(self) -> Dict[str, Any]:
        """
        Аргументы для конструктора сценария.
        
        TorchRL ``VmasEnv`` передаёт эти kwargs в конструктор сценария.
        
        Returns:
            Словарь ``scenario_kwargs``.
            
        Example:
            >>> config = VMASEnvConfig(scenario_kwargs={"n_agents": 5})
            >>> config.kwargs
            {'n_agents': 5}
        """
        return self.scenario_kwargs


# ============================================================================
# PettingZoo Environment Configuration
# ============================================================================

@dataclass
class PettingZooEnvConfig(BaseEnvConfig):
    """
    Конфигурация среды PettingZoo (multi-agent).
    
    PettingZoo — стандартный API для мультиагентных сред, поддерживающий
    два режима взаимодействия:
    
    - **Parallel API**: все агенты действуют одновременно
    - **AEC (Agent Environment Cycle)**: агенты действуют последовательно
    
    Attributes:
        task: Идентификатор задачи в формате ``"{family}/{env_name}"``.
            Примеры: ``"mpe/simple_spread_v3"``, ``"atari/pong_v3"``.
        parallel: Использовать Parallel API вместо AEC.
            - ``True`` — одновременные действия (рекомендуется)
            - ``False`` — последовательные действия
        return_state: Возвращать глобальное состояние среды.
            Полезно для CTDE (Centralized Training, Decentralized Execution).
        use_mask: Использовать маску допустимых действий.
            Обязательно для AEC сред и сред с переменным action space.
        group_map: Группировка агентов (аналогично VMAS).
        render_mode: Режим рендеринга.
        env_kwargs: Дополнительные аргументы для среды.
        
    Example:
        MPE среда с Parallel API::
        
            config = PettingZooEnvConfig(
                task="mpe/simple_spread_v3",
                parallel=True,
                group_map="all",
            )
            
        Atari мультиплеер::
        
            config = PettingZooEnvConfig(
                task="atari/pong_v3",
                parallel=True,
                use_mask=False,
            )
            
        Карточная игра с AEC (последовательные ходы)::
        
            config = PettingZooEnvConfig(
                task="classic/texas_holdem_v4",
                parallel=False,
                use_mask=True,  # обязательно для AEC
            )
            
    Warning:
        AEC-среды требуют ``use_mask=True`` для корректной обработки
        ситуаций, когда агент не может действовать (terminated/truncated).
        Конфигурация автоматически включает маску для AEC с предупреждением.
        
    Note:
        PettingZoo MPE среды перенесены в отдельный пакет ``mpe2``.
        При использовании ``task="mpe/..."`` может потребоваться
        установка: ``pip install mpe2``.
        
    See Also:
        - ``torchrl.envs.libs.pettingzoo.PettingZooEnv``
        - https://pettingzoo.farama.org/
        
    References:
        https://pytorch.org/rl/stable/reference/generated/torchrl.envs.PettingZooEnv.html
    """
    task: str = "mpe/simple_spread_v3"
    parallel: bool = True
    return_state: bool = False
    use_mask: bool = False
    group_map: Optional[MarlGroupMapType] = None
    render_mode: Optional[str] = None
    env_kwargs: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self) -> None:
        """
        Валидация параметров PettingZoo среды.
        
        Автоматически включает ``use_mask`` для AEC сред.
        
        Raises:
            ValueError: Если ``task`` пустой.
        """
        import warnings
        
        super().__post_init__()
        
        if not self.task or not self.task.strip():
            raise ValueError("task не может быть пустым")
            
        # AEC среды требуют маску
        if not self.parallel and not self.use_mask:
            warnings.warn(
                "AEC-среды (parallel=False) обычно требуют use_mask=True "
                "для корректной обработки недоступных действий. "
                "Автоматически устанавливаю use_mask=True.",
                UserWarning,
                stacklevel=2,
            )
            self.use_mask = True
            
        # Нормализуем group_map
        self.group_map = _normalize_group_map(self.group_map)
    
    @property
    def env_name(self) -> str:
        """
        Унифицированный идентификатор среды.
        
        Returns:
            Строка в формате ``"pettingzoo/{task}"``.
            
        Example:
            >>> PettingZooEnvConfig(task="mpe/simple_spread_v3").env_name
            'pettingzoo/mpe/simple_spread_v3'
        """
        return f"pettingzoo/{self.task}"
    
    @property
    def is_multi_agent(self) -> bool:
        """PettingZoo всегда мультиагентный."""
        return True
    
    @property
    def kwargs(self) -> Dict[str, Any]:
        """
        Дополнительные аргументы для среды.
        
        TorchRL ``PettingZooEnv`` передаёт эти kwargs в конструктор среды.
        
        Returns:
            Словарь ``env_kwargs``.
        """
        return self.env_kwargs


# ============================================================================
# Type Aliases for Convenience
# ============================================================================

#: Объединённый тип всех конфигураций сред
EnvConfig = Union[GymEnvConfig, VMASEnvConfig, PettingZooEnvConfig]

#: Конфигурации single-agent сред
SingleAgentEnvConfig = GymEnvConfig

#: Конфигурации multi-agent сред  
MultiAgentEnvConfig = Union[VMASEnvConfig, PettingZooEnvConfig]

#: TorchRL-style алиас для VMAS конфигурации
VmasEnvConfig = VMASEnvConfig

