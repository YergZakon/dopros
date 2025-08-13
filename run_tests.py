#!/usr/bin/env python3
"""
Скрипт для запуска тестов ДОПРОС MVP 2.0
"""

import sys
import os
import subprocess
import argparse
from pathlib import Path


def run_command(cmd, description=""):
    """Запуск команды с выводом"""
    print(f"\n{'='*60}")
    print(f"🔧 {description}")
    print(f"{'='*60}")
    print(f"Команда: {' '.join(cmd)}")
    print()
    
    result = subprocess.run(cmd, capture_output=False)
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="Запуск тестов ДОПРОС MVP 2.0")
    
    parser.add_argument(
        '--type', 
        choices=['all', 'compatibility', 'unit', 'integration', 'models', 'pipeline', 'performance'],
        default='all',
        help='Тип тестов для запуска'
    )
    
    parser.add_argument(
        '--coverage', 
        action='store_true',
        help='Включить анализ покрытия кода'
    )
    
    parser.add_argument(
        '--slow', 
        action='store_true',
        help='Включить медленные тесты'
    )
    
    parser.add_argument(
        '--gpu', 
        action='store_true',
        help='Включить GPU тесты'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Подробный вывод'
    )
    
    parser.add_argument(
        '--parallel', '-n',
        type=int,
        help='Количество параллельных процессов'
    )
    
    parser.add_argument(
        '--failed', 
        action='store_true',
        help='Запустить только упавшие тесты'
    )
    
    parser.add_argument(
        '--setup-only', 
        action='store_true',
        help='Только проверка окружения'
    )
    
    args = parser.parse_args()
    
    # Переходим в корневую директорию проекта
    project_root = Path(__file__).parent
    os.chdir(project_root)
    
    # Активируем виртуальное окружение
    venv_python = project_root / "venv" / "bin" / "python"
    if not venv_python.exists():
        venv_python = project_root / "venv" / "Scripts" / "python.exe"  # Windows
    
    if not venv_python.exists():
        print("❌ Виртуальное окружение не найдено!")
        print("Запустите: python -m venv venv && source venv/bin/activate && pip install -r requirements.txt")
        return 1
    
    # Проверяем установку pytest
    result = subprocess.run([str(venv_python), "-c", "import pytest"], 
                          capture_output=True)
    if result.returncode != 0:
        print("❌ pytest не установлен!")
        print("Запустите: pip install pytest pytest-cov pytest-mock")
        return 1
    
    # Базовая команда
    cmd = [str(venv_python), "-m", "pytest"]
    
    # Настройка типа тестов
    if args.type == 'compatibility':
        cmd.extend(["tests/test_compatibility.py"])
    elif args.type == 'unit':
        cmd.extend(["-m", "unit"])
    elif args.type == 'integration':
        cmd.extend(["-m", "integration"])
    elif args.type == 'models':
        cmd.extend(["tests/test_models.py"])
    elif args.type == 'pipeline':
        cmd.extend(["tests/test_pipeline.py"])
    elif args.type == 'performance':
        cmd.extend(["-m", "performance"])
    else:  # all
        cmd.extend(["tests/"])
    
    # Настройка покрытия кода
    if args.coverage:
        cmd.extend([
            "--cov=core",
            "--cov=models", 
            "--cov=utils",
            "--cov=integrations",
            "--cov-report=term-missing",
            "--cov-report=html:htmlcov"
        ])
    
    # Дополнительные маркеры
    markers = []
    if not args.slow:
        markers.append("not slow")
    if not args.gpu:
        markers.append("not gpu")
    
    if markers:
        cmd.extend(["-m", " and ".join(markers)])
    
    # Параллельное выполнение
    if args.parallel:
        cmd.extend(["-n", str(args.parallel)])
    
    # Подробный вывод
    if args.verbose:
        cmd.extend(["-v", "-s"])
    
    # Только упавшие тесты
    if args.failed:
        cmd.extend(["--lf"])
    
    # Только проверка окружения
    if args.setup_only:
        cmd = [str(venv_python), "-m", "pytest", "tests/test_compatibility.py", "-v"]
        success = run_command(cmd, "Проверка совместимости окружения")
        return 0 if success else 1
    
    # Запуск тестов
    print(f"🚀 Запуск тестов ДОПРОС MVP 2.0")
    print(f"📁 Рабочая директория: {project_root}")
    print(f"🐍 Python: {venv_python}")
    print(f"📦 Тип тестов: {args.type}")
    
    success = run_command(cmd, f"Выполнение тестов ({args.type})")
    
    if success:
        print("\n✅ Все тесты прошли успешно!")
        
        if args.coverage:
            print("📊 Отчет о покрытии сохранен в htmlcov/index.html")
            
    else:
        print("\n❌ Некоторые тесты завершились с ошибками!")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())