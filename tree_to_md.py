#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
tree_to_md.py — строит Markdown-дерево файлов/папок.
Пример:
    python tree_to_md.py . -o STRUCTURE.md --max-depth 6 --ignore .git __pycache__ .venv --descriptions tree_descriptions.json
"""

from __future__ import annotations
import argparse
import fnmatch
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

DEFAULT_IGNORES = {
    ".git", ".idea", ".vscode", "__pycache__", ".mypy_cache", ".pytest_cache",
    "node_modules", "dist", "build", ".DS_Store", "venv", ".venv",
    "checkpoints", "logs", "results", "data", "outputs", "temp",
    "tree_to_md.py", "STRUCTURE.md",
}

def load_descriptions(path: Path | None) -> Dict[str, str]:
    """
    Загружает словарь описаний путей (относительных) -> комментарий.
    Формат файла: JSON, ключи — относительные POSIX-пути (например: "configs/", "src/agentslab/networks/mlp.py")
    """
    if not path:
        return {}
    if not path.exists():
        print(f"[warn] Файл описаний не найден: {path}")
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError("descriptions JSON must be an object")
        return {k.strip(): str(v) for k, v in data.items()}
    except Exception as e:
        print(f"[warn] Не удалось прочитать descriptions: {e}")
        return {}

def norm_rel(p: Path, root: Path) -> str:
    """Относительный путь в POSIX-виде, с '/' у директорий."""
    rel = p.relative_to(root).as_posix()
    if p.is_dir() and not rel.endswith("/"):
        rel += "/"
    return rel

def should_ignore(rel_posix: str, name: str, patterns: Sequence[str], include_hidden: bool) -> bool:
    """Проверка игнора по шаблонам и скрытым.*"""
    # Скрытые
    if not include_hidden and name.startswith("."):
        return True
    # Сопоставление по имени и по полному относительному пути
    for pat in patterns:
        pat = pat.strip()
        if not pat:
            continue
        if fnmatch.fnmatch(name, pat) or fnmatch.fnmatch(rel_posix, pat):
            return True
    return False

def list_children(dir_path: Path) -> List[Path]:
    try:
        return sorted([p for p in dir_path.iterdir()], key=lambda p: (not p.is_dir(), p.name.lower()))
    except PermissionError:
        print(f"[warn] Нет доступа к: {dir_path}")
        return []
    except Exception as e:
        print(f"[warn] Ошибка чтения {dir_path}: {e}")
        return []

def annotate(rel_posix: str, descriptions: Dict[str, str]) -> str:
    """Возвращает строку-комментарий ' # ...', если есть описание для пути."""
    desc = descriptions.get(rel_posix)
    return f"  # {desc}" if desc else ""

def build_tree_lines(
    root: Path,
    current: Path,
    prefix: str,
    max_depth: int,
    ignore_patterns: Sequence[str],
    include_hidden: bool,
    descriptions: Dict[str, str],
    follow_symlinks: bool,
    depth: int = 0,
) -> List[str]:
    lines: List[str] = []

    if max_depth >= 0 and depth >= max_depth:
        return lines

    children = list_children(current)
    # Отфильтровать по ignore
    filtered: List[Path] = []
    for p in children:
        rel = norm_rel(p, root)
        if should_ignore(rel, p.name, ignore_patterns, include_hidden):
            continue
        if p.is_symlink() and not follow_symlinks:
            continue
        filtered.append(p)

    for idx, child in enumerate(filtered):
        is_last = idx == len(filtered) - 1
        branch = "└── " if is_last else "├── "
        next_prefix = prefix + ("    " if is_last else "│   ")
        rel = norm_rel(child, root)

        if child.is_dir():
            line = f"{prefix}{branch}📁 {child.name}/" + annotate(rel, descriptions)
            lines.append(line)
            lines.extend(
                build_tree_lines(
                    root, child, next_prefix, max_depth, ignore_patterns,
                    include_hidden, descriptions, follow_symlinks, depth + 1
                )
            )
        else:
            line = f"{prefix}{branch}{child.name}" + annotate(rel, descriptions)
            lines.append(line)

    return lines

def generate_markdown_tree(
    root: Path,
    max_depth: int,
    ignore_patterns: Sequence[str],
    include_hidden: bool,
    descriptions: Dict[str, str],
    follow_symlinks: bool,
    code_fence: bool = True,
) -> str:
    title = f"{root.name}/"
    top_annot = annotate("", descriptions)  # обычно пусто
    header = f"{title}{top_annot}"

    lines = [header]
    lines.extend(
        build_tree_lines(
            root=root,
            current=root,
            prefix="",
            max_depth=max_depth,
            ignore_patterns=ignore_patterns,
            include_hidden=include_hidden,
            descriptions=descriptions,
            follow_symlinks=follow_symlinks,
        )
    )

    body = "\n".join(lines)
    if code_fence:
        return "```\n" + body + "\n```"
    return body

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Сохраняет структуру директории в Markdown.")
    p.add_argument("root", nargs="?", default=".", help="Корневая директория (по умолчанию .)")
    p.add_argument("-o", "--output", default="STRUCTURE.md", help="Путь к MD файлу вывода")
    p.add_argument("--max-depth", type=int, default=-1, help="Глубина (−1 — без ограничений)")
    p.add_argument("--ignore", nargs="*", default=sorted(DEFAULT_IGNORES),
                   help="Шаблоны игнора (glob). Можно указывать много значений.")
    p.add_argument("--include-hidden", action="store_true", help="Не скрывать скрытые файлы/папки (.*)")
    p.add_argument("--descriptions", type=str, default=None,
                   help="JSON-файл с описаниями путей -> комментарий")
    p.add_argument("--no-fence", action="store_true", help="Не оборачивать вывод в ``` код-блок")
    p.add_argument("--follow-symlinks", action="store_true", help="Следовать по symlink-папкам")
    return p.parse_args()

def main():
    args = parse_args()
    root = Path(args.root).resolve()
    if not root.exists() or not root.is_dir():
        raise SystemExit(f"Корень не найден или это не папка: {root}")

    descriptions = load_descriptions(Path(args.descriptions) if args.descriptions else None)

    md = generate_markdown_tree(
        root=root,
        max_depth=args.max_depth,
        ignore_patterns=args.ignore,
        include_hidden=args.include_hidden,
        descriptions=descriptions,
        follow_symlinks=args.follow_symlinks,
        code_fence=not args.no_fence,
    )

    out_path = Path(args.output)
    out_path.write_text(md, encoding="utf-8")
    print(f"[ok] Cтруктура сохранена в: {out_path}")

if __name__ == "__main__":
    main()
