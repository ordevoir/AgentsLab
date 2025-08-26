from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union
import warnings
import re

import pandas as pd
import matplotlib.pyplot as plt
from contextlib import nullcontext

__all__ = [
    "plot_metrics_from_csv",
    "plot_metrics",
]

# Кандидаты для оси X по умолчанию
DEFAULT_X_CANDIDATES: Tuple[str, ...] = ("step", "global_step", "iteration", "epoch")


def _choose_x_col(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    """Возвращает первый найденный столбец из candidates, присутствующий в df."""
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _compile_patterns(patterns: Optional[Sequence[str]]) -> Optional[List[re.Pattern]]:
    if not patterns:
        return None
    return [re.compile(p) for p in patterns]


def _filter_numeric_columns(
    df: pd.DataFrame,
    x_col: Optional[str],
    include: Optional[Sequence[str]] = None,
    exclude: Optional[Sequence[str]] = None,
) -> List[str]:
    """Возвращает список числовых метрик с учётом include/exclude и исключением оси X."""
    cols = df.select_dtypes(include="number").columns.tolist()
    if x_col in cols:
        cols.remove(x_col)

    inc = _compile_patterns(include)
    exc = _compile_patterns(exclude)

    if inc:
        cols = [c for c in cols if any(p.search(c) for p in inc)]
    if exc:
        cols = [c for c in cols if not any(p.search(c) for p in exc)]

    # Убираем полностью NaN-овые
    cols = [c for c in cols if not df[c].isna().all()]
    return cols


def _style_context(style: Optional[str]):
    if style is None:
        return nullcontext()
    if style in plt.style.available:
        return plt.style.context(style)
    warnings.warn(f"Matplotlib style '{style}' не найден; используется стиль по умолчанию.")
    return nullcontext()


def plot_metrics_from_csv(
    csv_path: Union[str, Path],
    *,
    x_candidates: Sequence[str] = DEFAULT_X_CANDIDATES,
    include: Optional[Sequence[str]] = None,
    exclude: Optional[Sequence[str]] = None,
    style: Optional[str] = None,
    figsize: Tuple[float, float] = (9, 4.5),
    grid: bool = True,
    ema: Optional[float] = None,
    downsample: Optional[int] = None,
    save_dir: Optional[Union[str, Path]] = None,
    show: bool = True,
    low_memory: bool = False,
    dpi: int = 120,
) -> Dict[str, object]:
    """
    Читает CSV и строит отдельный график для каждой числовой метрики.

    Параметры:
        csv_path: путь к train.csv.
        x_candidates: приоритетный список названий столбцов для оси X.
        include: список regex для фильтрации метрик (оставить только совпадающие).
        exclude: список regex для исключения метрик.
        style: имя стиля matplotlib (например, "ordevoir-dark"); если нет — игнорируется.
        figsize: размер фигуры для каждого графика.
        grid: рисовать ли сетку.
        ema: сглаживание экспоненциальным средним (alpha в (0,1)); None — без сглаживания.
        downsample: брать каждый N-й элемент (для больших CSV).
        save_dir: если указан — сохраняет PNG для каждого графика в эту папку.
        show: отображать графики через plt.show(). Если False — фигуры закрываются.
        low_memory: проксируется в pandas.read_csv.
        dpi: DPI при сохранении.

    Возвращает:
        dict с ключами {"x_col", "plotted", "saved"}.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"train.csv не найден по пути: {csv_path}")

    df = pd.read_csv(csv_path, low_memory=low_memory)
    return plot_metrics(
        df,
        x_candidates=x_candidates,
        include=include,
        exclude=exclude,
        style=style,
        figsize=figsize,
        grid=grid,
        ema=ema,
        downsample=downsample,
        save_dir=save_dir,
        show=show,
        dpi=dpi,
    )


def plot_metrics(
    df: pd.DataFrame,
    *,
    x_candidates: Sequence[str] = DEFAULT_X_CANDIDATES,
    include: Optional[Sequence[str]] = None,
    exclude: Optional[Sequence[str]] = None,
    style: Optional[str] = None,
    figsize: Tuple[float, float] = (9, 4.5),
    grid: bool = True,
    ema: Optional[float] = None,
    downsample: Optional[int] = None,
    save_dir: Optional[Union[str, Path]] = None,
    show: bool = True,
    dpi: int = 120,
) -> Dict[str, object]:
    """
    Строит графики метрик из уже загруженного DataFrame.

    Логика совпадает с plot_metrics_from_csv, но на вход подаётся df.
    """
    if downsample is not None:
        if downsample <= 0:
            raise ValueError("downsample должен быть положительным целым")
        df = df.iloc[::downsample, :].reset_index(drop=True)

    x_col = _choose_x_col(df, x_candidates)
    num_cols = _filter_numeric_columns(df, x_col, include, exclude)

    if not num_cols:
        raise ValueError("В DataFrame не найдено числовых метрик для построения графиков.")

    if ema is not None and not (0.0 < ema < 1.0):
        raise ValueError("ema должен быть в диапазоне (0, 1)")

    save_dir_path: Optional[Path] = Path(save_dir) if save_dir is not None else None
    if save_dir_path is not None:
        save_dir_path.mkdir(parents=True, exist_ok=True)

    result: Dict[str, object] = {"x_col": x_col, "plotted": [], "saved": []}

    with _style_context(style):
        for col in num_cols:
            y = df[col]
            if ema is not None:
                y = pd.Series(y, copy=False).ewm(alpha=ema, adjust=False).mean()

            fig = plt.figure(figsize=figsize)
            ax = fig.gca()

            if x_col is not None:
                ax.plot(df[x_col].values, y.values)
                ax.set_xlabel(x_col)
            else:
                ax.plot(range(len(df)), y.values)
                ax.set_xlabel("index")

            ax.set_ylabel(col)
            title = f"{col}" + (f" (EMA α={ema})" if ema is not None else "")
            ax.set_title(title)
            if grid:
                ax.grid(True)

            fig.tight_layout()

            if save_dir_path is not None:
                out_path = save_dir_path / f"{col}.png"
                fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
                result["saved"].append(str(out_path))

            if show:
                plt.show()
            else:
                plt.close(fig)

            result["plotted"].append(col)

    return result
