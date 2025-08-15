#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
collect_files_bfs.py — рекурсивный сбор всех файлов в одну папку с обходом в ширину.
Правила:
  • Сканирование начинается ТОЛЬКО из путей, перечисленных в START_ITEMS.
  • Все найденные файлы копируются в одну папку OUTPUT_DIR без подпапок.
  • Если позже встретится файл с тем же именем (basename) — он ПРОПУСКАЕТСЯ.
  • Порядок — BFS (обход в ширину), поэтому «выигрывает» более близкий/ранний по очереди файл.
  • Итог печатается в конце: сколько скопировано, сколько пропущено и какие дубликаты встречались.

Запуск:
  python collect_files_bfs.py
(или просто двойной клик, если ассоциировано с Python)
"""

from __future__ import annotations
import shutil
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Set, Tuple

# ====== НАСТРОЙКИ (меняйте под себя) =========================================
# Корень проекта, относительно которого задаются START_ITEMS
ROOT_DIR = Path(".").resolve()

# Верхнеуровневые элементы для сканирования (папки ИЛИ файлы).
# Пути относительные к ROOT_DIR. Порядок важен: левые идут первыми в BFS.
START_ITEMS: List[str] = [
    "configs",
    "src",
    "scripts",
    "README.md",
]

# Папка, куда складываем все файлы (будет создана/очищена)
OUTPUT_DIR = ROOT_DIR / "collected_files"

# Следовать по симлинкам?
FOLLOW_SYMLINKS = False

# Игнорируемые папки и файлы по ИМЕНИ (без путей)
IGNORE_DIR_NAMES: Set[str] = {".git", "__pycache__", ".mypy_cache", ".pytest_cache",
                              ".idea", ".vscode", "node_modules", "build", "dist", ".venv", "venv"}
IGNORE_FILE_NAMES: Set[str] = {".DS_Store", "__init__.py" }

# Чувствительность к регистру для сравнения имён файлов на дубликаты
CASE_SENSITIVE = True

# Очищать OUTPUT_DIR перед началом
CLEAN_OUTPUT_DIR = True
# ============================================================================


@dataclass
class Stats:
    scanned_files: int = 0
    copied_files: int = 0
    skipped_duplicates: int = 0


def safe_basename(name: str) -> str:
    return name if CASE_SENSITIVE else name.lower()


def prepare_output_dir(out_dir: Path, clean: bool) -> None:
    if out_dir.exists():
        if clean:
            # очищаем содержимое
            for p in out_dir.iterdir():
                try:
                    if p.is_dir():
                        shutil.rmtree(p)
                    else:
                        p.unlink()
                except Exception as e:
                    print(f"[warn] Не удалось удалить {p}: {e}")
    else:
        out_dir.mkdir(parents=True, exist_ok=True)


def bfs_collect(start_paths: List[Path], out_dir: Path) -> Tuple[Stats, Dict[str, Path], Dict[str, List[Path]]]:
    """
    Возвращает:
      stats — счётчики,
      winners — имя файла -> путь-победитель (первое вхождение по BFS),
      duplicates_map — имя файла -> список путей, которые были пропущены как дубликаты.
    """
    q: deque[Path] = deque()
    for p in start_paths:
        if not p.exists():
            print(f"[warn] Стартовый путь не найден: {p}")
            continue
        q.append(p)

    seen_names: Set[str] = set()           # нормализованные имена, которые уже встретились
    winners: Dict[str, Path] = {}          # оригинальное имя -> исходный путь (победитель)
    duplicates_map: Dict[str, List[Path]] = {}

    stats = Stats()

    while q:
        cur = q.popleft()

        # Если это файл — обрабатываем сразу
        try:
            if cur.is_file():
                name_key = safe_basename(cur.name)
                stats.scanned_files += 1

                if name_key in seen_names:
                    stats.skipped_duplicates += 1
                    duplicates_map.setdefault(cur.name, []).append(cur)
                else:
                    # новый файл — копируем
                    seen_names.add(name_key)
                    winners[cur.name] = cur
                    dst = out_dir / cur.name
                    try:
                        shutil.copy2(cur, dst)
                        stats.copied_files += 1
                        print(f"[copy] {cur}  ->  {dst.name}")
                    except Exception as e:
                        print(f"[error] Не удалось копировать {cur}: {e}")

                continue  # файл обработан, к следующему элементу очереди
        except Exception as e:
            print(f"[warn] Не удалось проверить тип {cur}: {e}")
            continue

        # Если это папка — добавляем детей в очередь (BFS)
        try:
            if cur.is_dir():
                # игнор по имени папки
                if cur.name in IGNORE_DIR_NAMES:
                    continue

                with os_scandir(cur) as it:
                    entries = sorted((Path(entry.path) for entry in it),
                                     key=lambda p: p.name.lower())
                for p in entries:
                    try:
                        # игнор по имени файла
                        if p.is_file() and p.name in IGNORE_FILE_NAMES:
                            continue
                        # контроль симлинков
                        if p.is_symlink() and not FOLLOW_SYMLINKS:
                            continue
                        q.append(p)
                    except Exception as e:
                        print(f"[warn] Пропуск {p}: {e}")
                        continue
        except Exception as e:
            print(f"[warn] Не удалось прочитать каталог {cur}: {e}")
            continue

    return stats, winners, duplicates_map


class os_scandir:
    """Контекстный менеджер для безопасного os.scandir с автоматическим закрытием."""
    def __init__(self, path: Path):
        self.path = path
        self._it = None

    def __enter__(self):
        import os
        self._it = os.scandir(self.path)
        return self._it

    def __exit__(self, exc_type, exc, tb):
        if self._it:
            self._it.close()


def main():
    print(f"[info] ROOT_DIR: {ROOT_DIR}")
    start_paths = [ROOT_DIR / s for s in START_ITEMS]
    print("[info] START_ITEMS:")
    for p in start_paths:
        print(f"   - {p}")

    prepare_output_dir(OUTPUT_DIR, CLEAN_OUTPUT_DIR)
    print(f"[info] OUTPUT_DIR: {OUTPUT_DIR} (очистка: {'да' if CLEAN_OUTPUT_DIR else 'нет'})")
    print(f"[info] FOLLOW_SYMLINKS: {'да' if FOLLOW_SYMLINKS else 'нет'}, "
          f"CASE_SENSITIVE: {'да' if CASE_SENSITIVE else 'нет'}")

    stats, winners, duplicates_map = bfs_collect(start_paths, OUTPUT_DIR)

    # ===== Итог =====
    print("\n===== ИТОГ =====")
    print(f"Файлов просмотрено: {stats.scanned_files}")
    print(f"Файлов скопировано: {stats.copied_files}")
    print(f"Пропущено дубликатов: {stats.skipped_duplicates}")

    if duplicates_map:
        print("\nДубликаты по именам (оставлен первый встретившийся файл):")
        for name, dup_list in sorted(duplicates_map.items(), key=lambda x: x[0].lower()):
            kept = winners.get(name)
            print(f"  {name}")
            if kept:
                print(f"    ✓ оставлен: {kept}")
            for p in dup_list:
                print(f"    ✗ пропущен: {p}")

    print("\nГотово.")

if __name__ == "__main__":
    main()
