from __future__ import annotations
from pathlib import Path
from typing import Mapping, Union, Optional, List
from numbers import Real
import csv


class CSVLogger:
    """
    Простой и безопасный CSV-логгер.

    Поведение:
      - Заголовок берётся из ключей первой записи (порядок колонок сохраняется).
      - Каждая следующая запись должна иметь тот же набор колонок.
      - Родительские директории создаются автоматически.
      - Если файл существует и непустой, схема читается из его заголовка.

    Ограничения:
      - Значения должны быть числами (int/float). Булевы значения не допускаются.
    """
    def __init__(self, csv_path: Union[str, Path]) -> None:
        self.path = Path(csv_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

        self._fieldnames: Optional[List[str]] = None
        self._header_written: bool = False

        # Если файл уже есть и не пустой — считываем заголовок как схему
        if self.path.exists() and self.path.stat().st_size > 0:
            with self.path.open("r", newline="", encoding="utf-8") as f:
                reader = csv.reader(f)
                try:
                    header = next(reader)
                except StopIteration:
                    header = None
            if header:
                self._fieldnames = list(header)
                self._header_written = True

    @property
    def fieldnames(self) -> Optional[List[str]]:
        """Текущая фиксированная схема колонок (или None до первой записи)."""
        return None if self._fieldnames is None else list(self._fieldnames)

    def log(self, row: Mapping[str, Real]) -> None:
        """
        Записать одну строку в CSV.
        :param row: словарь {имя_колонки: числовое_значение}
        """
        if not row:
            raise ValueError("row пуст: нет данных для записи")

        # Проверка ключей и значений
        for k, v in row.items():
            if not isinstance(k, str):
                raise TypeError(f"Имя колонки должно быть str, получено: {type(k).__name__}")
            # bool является подклассом int — явно запретим, чтобы не путать с метриками
            if isinstance(v, bool) or not isinstance(v, Real):
                raise TypeError(f"Значение в колонке '{k}' должно быть числом (int/float), получено: {type(v).__name__}")

        # Инициализируем схему при первой записи
        if self._fieldnames is None:
            self._fieldnames = list(row.keys())

        # Проверяем согласованность схемы
        row_keys = set(row.keys())
        schema_keys = set(self._fieldnames)
        if row_keys != schema_keys:
            missing = schema_keys - row_keys
            extra = row_keys - schema_keys
            parts = []
            if missing:
                parts.append(f"отсутствуют колонки: {sorted(missing)}")
            if extra:
                parts.append(f"лишние колонки: {sorted(extra)}")
            hint = "; ".join(parts)
            raise ValueError(f"Схема не совпадает с заголовком CSV: {hint}")

        # Готовим упорядоченную запись в соответствии со схемой
        ordered_row = {k: row[k] for k in self._fieldnames}

        # Пишем (append), при необходимости — заголовок
        write_header_now = not self._header_written
        with self.path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self._fieldnames)
            if write_header_now:
                writer.writeheader()
                self._header_written = True
            writer.writerow(ordered_row)

    def __repr__(self) -> str:
        return f"CSVLogger(path={self.path!s}, fieldnames={self._fieldnames})"
    
    def close(self) -> None:
        pass  # placeholder для совместимости API

    def __enter__(self) -> "CSVLogger":
        return self

    def __exit__(self, *args) -> None:
        self.close()



class TBLogger:
    """
    Логгер для TensorBoard.

    Конструктор:
      TBLogger(log_dir, flush_secs=10, tag_prefix="")
        - log_dir: каталог, куда писать события TB (будет создан при необходимости)
        - flush_secs: период авто-сброса буфера SummaryWriter
        - tag_prefix: необязательный префикс для всех тэгов (напр. "train/")

    Методы:
      - log(row: Mapping[str, Real], step: Optional[int] = None)
      - flush()
      - close()
      - контекстный менеджер (with TBLogger(...) as tb: ...)
    """
    def __init__(self, log_dir: Union[str, Path], flush_secs: int = 10, tag_prefix: str = "") -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.tag_prefix = (tag_prefix.rstrip("/") + "/") if tag_prefix else ""
        self._step = 0

        # Пытаемся импортировать лучший доступный SummaryWriter
        writer_cls = None
        try:
            from torch.utils.tensorboard import SummaryWriter as _SW
            writer_cls = _SW
        except Exception:
            try:
                from tensorboardX import SummaryWriter as _SW  # type: ignore
                writer_cls = _SW
            except Exception as e:
                raise ImportError(
                    "TBLogger требует либо PyTorch (torch.utils.tensorboard), либо tensorboardX."
                ) from e

        self._writer = writer_cls(log_dir=str(self.log_dir), flush_secs=flush_secs)

    def log(self, row: Mapping[str, Real], step: Optional[int] = None) -> None:
        """
        Записать набор скалярных метрик.
        :param row: словарь {тег: число}
        :param step: явный global_step; если None — используется внутренний счётчик и авто-инкремент
        """
        if not row:
            raise ValueError("row пуст: нет данных для записи в TensorBoard")

        # Валидация ключей/значений
        for k, v in row.items():
            if not isinstance(k, str):
                raise TypeError(f"Имя тэга должно быть str, получено {type(k).__name__}")
            # bool — подкласс int, но это почти всегда ошибка при логировании метрик
            if isinstance(v, bool) or not isinstance(v, Real):
                raise TypeError(f"Значение для '{k}' должно быть числом (int/float), получено {type(v).__name__}")

        gstep = self._step if step is None else int(step)

        for k, v in row.items():
            tag = f"{self.tag_prefix}{k}"
            self._writer.add_scalar(tag, float(v), global_step=gstep)

        if step is None:
            self._step += 1

    def flush(self) -> None:
        self._writer.flush()

    def close(self) -> None:
        self._writer.close()
    
    def __repr__(self) -> str:
        return f"TBLogger(log_dir={self.log_dir!s}, step={self._step})"

    def __enter__(self) -> "TBLogger":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

