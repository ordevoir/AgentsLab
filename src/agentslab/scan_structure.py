#!/usr/bin/env python3
"""
Скрипт для генерации структуры директории (directory structure) в текстовом виде.
Результат сохраняется в файл project_structure.txt
"""

import os
from pathlib import Path


# Директории и файлы, которые нужно игнорировать
IGNORE = {
    '__pycache__', '.git', '.idea', '.vscode', 
    'node_modules', '.pytest_cache', '.mypy_cache',
    'venv', 'env', '.env', '.venv', '*.pyc', '.DS_Store'
}


def should_ignore(name: str) -> bool:
    """Проверяет, нужно ли игнорировать файл/директорию."""
    return name in IGNORE or name.startswith('.')


def generate_tree(directory: Path, prefix: str = "") -> list[str]:
    """
    Рекурсивно генерирует древовидную структуру (tree structure).
    
    Args:
        directory: Путь к директории
        prefix: Префикс для текущего уровня вложенности
    
    Returns:
        Список строк с древовидной структурой
    """
    lines = []
    
    # Получаем содержимое директории
    try:
        entries = list(directory.iterdir())
    except PermissionError:
        return lines
    
    # Фильтруем игнорируемые элементы
    entries = [e for e in entries if not should_ignore(e.name)]
    
    # Сортируем: сначала директории, потом файлы, алфавитно
    entries.sort(key=lambda e: (not e.is_dir(), e.name.lower()))
    
    for i, entry in enumerate(entries):
        is_last = (i == len(entries) - 1)
        
        # Выбираем символы для текущего элемента
        connector = "└── " if is_last else "├── "
        
        # Добавляем "/" для директорий
        name = entry.name + "/" if entry.is_dir() else entry.name
        lines.append(f"{prefix}{connector}{name}")
        
        # Рекурсивно обрабатываем поддиректории
        if entry.is_dir():
            # Выбираем префикс для вложенных элементов
            extension = "    " if is_last else "│   "
            lines.extend(generate_tree(entry, prefix + extension))
    
    return lines


def main():
    # Получаем директорию, где находится скрипт
    script_dir = Path(__file__).parent.resolve()
    
    # Генерируем структуру
    root_name = script_dir.name + "/"
    tree_lines = [root_name] + generate_tree(script_dir)
    
    # Формируем итоговый текст
    result = "\n".join(tree_lines)
    
    # Сохраняем в файл
    output_file = script_dir / "project_structure.txt"
    output_file.write_text(result, encoding="utf-8")
    
    print(f"Структура директории сохранена в: {output_file}")
    print("\n" + result)


if __name__ == "__main__":
    main()


