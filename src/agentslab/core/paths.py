"""Управление путями и структурой директорий эксперимента."""

import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class RunPaths:
    """
    Структура путей для одного эксперимента (run).
    
    Структура директорий:
        {root}/runs/{run_name}/
        ├── checkpoints/
        ├── csv_logs/
        │   ├── train.csv
        │   └── eval.csv
        ├── figures/
        └── meta_info.yaml
    
    Attributes:
        run_name: Уникальное имя эксперимента
        root: Корневая директория проекта
        run_dir: Директория данного run
        ckpt_dir: Директория для чекпоинтов
        csv_train: Путь к CSV с метриками обучения
        csv_eval: Путь к CSV с метриками оценки
        fig_dir: Директория для графиков
        meta_yaml: Путь к файлу метаданных
    """
    run_name: str
    root: Path
    run_dir: Path
    ckpt_dir: Path
    csv_train: Path
    csv_eval: Path
    fig_dir: Path
    meta_yaml: Path
    
    def ensure_dirs(self) -> "RunPaths":
        """
        Создаёт все необходимые директории.
        
        Returns:
            self для chaining
        """
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.csv_train.parent.mkdir(parents=True, exist_ok=True)
        self.fig_dir.mkdir(parents=True, exist_ok=True)
        return self


def _sanitize_for_path(name: str) -> str:
    """
    Санитизирует строку для безопасного использования в путях.
    
    Преобразования:
        - "/" → "-" (разделитель PettingZoo: "mpe/simple_spread_v3" → "mpe-simple_spread_v3")
        - "\\" → "-"
        - пробелы → "_"
        - убирает небезопасные символы: < > : " | ? *
        - схлопывает множественные "-" и "_"
    
    Args:
        name: Исходная строка (например, env_name)
        
    Returns:
        Безопасная для файловой системы строка
        
    Examples:
        >>> _sanitize_for_path("mpe/simple_spread_v3")
        'mpe-simple_spread_v3'
        >>> _sanitize_for_path("ALE/Pong-v5")
        'ALE-Pong-v5'
        >>> _sanitize_for_path("my env:test")
        'my_env_test'
    """
    if not name:
        return name
    
    # Заменяем слэши на дефис
    result = name.replace("/", "-").replace("\\", "-")
    
    # Пробелы → подчёркивание
    result = result.replace(" ", "_")
    
    # Заменяем двоеточие на подчёркивание (сохраняем разделение)
    result = result.replace(":", "_")
    
    # Убираем остальные небезопасные символы для Windows/Linux
    result = re.sub(r'[<>"|?*]', "", result)
    
    # Схлопываем множественные дефисы и подчёркивания
    result = re.sub(r'-+', '-', result)
    result = re.sub(r'_+', '_', result)
    
    # Убираем дефисы/подчёркивания в начале и конце
    result = result.strip("-_")
    
    return result


def find_project_root(start_path: Optional[Path] = None) -> Path:
    """
    Ищет корень проекта по маркерным файлам.
    
    Поиск идёт вверх от start_path до первой директории,
    содержащей один из маркеров: pyproject.toml, setup.py, .git
    
    Args:
        start_path: Начальная точка поиска (по умолчанию — cwd)
        
    Returns:
        Путь к корню проекта
        
    Raises:
        FileNotFoundError: Если корень проекта не найден
        
    Example:
        >>> root = find_project_root()
        >>> root = find_project_root(Path(__file__).parent)
    """
    markers = ("pyproject.toml", "setup.py", ".git", "setup.cfg")
    
    current = Path(start_path or Path.cwd()).resolve()
    
    for parent in [current, *current.parents]:
        if any((parent / marker).exists() for marker in markers):
            return parent
    
    raise FileNotFoundError(
        f"Project root not found. Searched from {current} for markers: {markers}. "
        f"Please specify 'root' explicitly or create one of the marker files."
    )


def generate_paths(
    algo_name: Optional[str] = None,
    env_name: Optional[str] = None,
    root: Optional[Path] = None,
    prefix: Optional[str] = None,
    suffix: Optional[str] = None,
    run_name: Optional[str] = None,
) -> RunPaths:
    """
    Генерирует пути для нового эксперимента.
    
    Формирует структуру директорий:
        {root}/runs/{run_name}/
    
    Имя run формируется по правилам:
        1. Если run_name задан — используется как есть (после санитизации)
        2. Иначе собирается: {prefix}_{algo_name}_{env_name}_{timestamp}_{suffix}
           (пустые компоненты пропускаются, env_name санитизируется)
    
    Args:
        algo_name: Название алгоритма (например, "PPO")
        env_name: Название среды (например, "CartPole-v1", "mpe/simple_spread_v3")
        root: Корневая директория проекта (автоопределение если None)
        prefix: Префикс к имени run
        suffix: Суффикс к имени run
        run_name: Полное имя run (если задано, остальные компоненты игнорируются)
        
    Returns:
        RunPaths с уникальным именем
        
    Raises:
        ValueError: Если не задан ни run_name, ни algo_name
        FileNotFoundError: Если root=None и корень проекта не найден
        
    Examples:
        >>> # Стандартное использование
        >>> paths = generate_paths("PPO", "CartPole-v1")
        >>> paths.run_name
        'PPO_CartPole-v1_20250125_143022'
        
        >>> # PettingZoo среда (слэш заменяется на дефис)
        >>> paths = generate_paths("MAPPO", "mpe/simple_spread_v3")
        >>> paths.run_name
        'MAPPO_mpe-simple_spread_v3_20250125_143022'
        
        >>> # Atari среда
        >>> paths = generate_paths("DQN", "ALE/Pong-v5")
        >>> paths.run_name
        'DQN_ALE-Pong-v5_20250125_143022'
        
        >>> # С префиксом и суффиксом
        >>> paths = generate_paths("PPO", "CartPole-v1", prefix="exp01", suffix="seed42")
        >>> paths.run_name
        'exp01_PPO_CartPole-v1_20250125_143022_seed42'
        
        >>> # Явное имя
        >>> paths = generate_paths(run_name="my_custom_run")
        >>> paths.run_name
        'my_custom_run'
    """
    # Определяем root
    if root is None:
        root = find_project_root()
    root = Path(root).resolve()
    
    # Формируем run_name
    if run_name is None:
        if algo_name is None:
            raise ValueError(
                "Either 'run_name' or 'algo_name' must be provided. "
                "Cannot generate run name without at least algo_name."
            )
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Санитизируем env_name для безопасного использования в пути
        safe_env_name = _sanitize_for_path(env_name) if env_name else None
        
        # Собираем компоненты имени (пропускаем None и пустые)
        components = [prefix, algo_name, safe_env_name, timestamp, suffix]
        run_name = "_".join(c for c in components if c)
    else:
        # Санитизируем явно заданное имя
        run_name = _sanitize_for_path(run_name)
    
    # Проверяем уникальность и создаём пути
    run_dir = root / "runs" / run_name
    
    # Гарантируем уникальность при совпадении по времени
    if run_dir.exists():
        base_name = run_name
        counter = 2
        while run_dir.exists():
            run_name = f"{base_name}__{counter}"
            run_dir = root / "runs" / run_name
            counter += 1
    
    return _build_run_paths(root, run_name)


def restore_paths(
    run_name: str,
    root: Optional[Path] = None,
) -> RunPaths:
    """
    Восстанавливает пути из существующего эксперимента.
    
    Args:
        run_name: Имя существующего run
        root: Корневая директория проекта (автоопределение если None)
        
    Returns:
        RunPaths для существующего эксперимента
        
    Raises:
        FileNotFoundError: Если директория run не существует
        
    Example:
        >>> paths = restore_paths("PPO_CartPole-v1_20250125_143022")
        >>> paths = restore_paths("PPO_CartPole-v1_20250125_143022", root=Path("/projects/rl"))
    """
    # Определяем root
    if root is None:
        root = find_project_root()
    root = Path(root).resolve()
    
    # Проверяем существование
    run_dir = root / "runs" / run_name
    if not run_dir.exists():
        raise FileNotFoundError(
            f"Run directory not found: {run_dir}\n"
            f"Available runs: {_list_available_runs(root)}"
        )
    
    return _build_run_paths(root, run_name)


def _build_run_paths(root: Path, run_name: str) -> RunPaths:
    """Внутренняя функция построения RunPaths."""
    run_dir = root / "runs" / run_name
    
    return RunPaths(
        run_name=run_name,
        root=root,
        run_dir=run_dir,
        ckpt_dir=run_dir / "checkpoints",
        csv_train=run_dir / "csv_logs" / "train.csv",
        csv_eval=run_dir / "csv_logs" / "eval.csv",
        fig_dir=run_dir / "figures",
        meta_yaml=run_dir / "meta_info.yaml",
    )


def _list_available_runs(root: Path, max_display: int = 10) -> str:
    """Возвращает список доступных run для сообщения об ошибке."""
    runs_dir = root / "runs"
    if not runs_dir.exists():
        return "(no runs directory)"
    
    runs = sorted(runs_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
    run_names = [r.name for r in runs if r.is_dir()]
    
    if not run_names:
        return "(empty)"
    
    if len(run_names) <= max_display:
        return ", ".join(run_names)
    
    return ", ".join(run_names[:max_display]) + f", ... ({len(run_names) - max_display} more)"
