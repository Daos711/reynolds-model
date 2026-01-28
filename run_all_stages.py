#!/usr/bin/env python3
"""
Запуск всех этапов расчёта подшипника.

Использование:
    python run_all_stages.py              # все этапы 1-8
    python run_all_stages.py --stages 1 3 5   # только этапы 1, 3, 5
    python run_all_stages.py --from 3         # этапы 3-8
    python run_all_stages.py --to 5           # этапы 1-5
    python run_all_stages.py --from 2 --to 6  # этапы 2-6

Результаты сохраняются в results/stage{N}/
"""

import subprocess
import sys
import time
import argparse
from pathlib import Path


STAGES = list(range(1, 9))  # 1-8


def run_stage(stage_num: int) -> tuple[bool, float]:
    """
    Запустить один этап.

    Returns:
        (success, elapsed_time)
    """
    script_path = Path(__file__).parent / f"run_stage{stage_num}.py"

    if not script_path.exists():
        print(f"  [!] Скрипт {script_path.name} не найден")
        return False, 0.0

    start = time.time()

    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=False,
            text=True,
            cwd=Path(__file__).parent,
        )
        elapsed = time.time() - start

        if result.returncode == 0:
            return True, elapsed
        else:
            print(f"  [!] Этап {stage_num} завершился с ошибкой (код {result.returncode})")
            return False, elapsed

    except Exception as e:
        elapsed = time.time() - start
        print(f"  [!] Ошибка запуска этапа {stage_num}: {e}")
        return False, elapsed


def main():
    parser = argparse.ArgumentParser(
        description="Запуск всех этапов расчёта подшипника"
    )
    parser.add_argument(
        "--stages", "-s",
        type=int,
        nargs="+",
        help="Список этапов для запуска (например: 1 3 5)"
    )
    parser.add_argument(
        "--from", "-f",
        type=int,
        dest="from_stage",
        help="Начать с этапа N"
    )
    parser.add_argument(
        "--to", "-t",
        type=int,
        dest="to_stage",
        help="Закончить на этапе N"
    )

    args = parser.parse_args()

    # Определяем список этапов
    if args.stages:
        stages = sorted(set(args.stages))
    else:
        from_s = args.from_stage or 1
        to_s = args.to_stage or 8
        stages = list(range(from_s, to_s + 1))

    # Валидация
    for s in stages:
        if s < 1 or s > 8:
            print(f"Ошибка: этап {s} не существует (доступны 1-8)")
            sys.exit(1)

    print("=" * 60)
    print("ЗАПУСК ЭТАПОВ РАСЧЁТА ПОДШИПНИКА")
    print("=" * 60)
    print(f"Этапы: {', '.join(map(str, stages))}")
    print()

    total_start = time.time()
    results = {}

    for stage in stages:
        print(f"\n{'─' * 60}")
        print(f"ЭТАП {stage}")
        print("─" * 60)

        success, elapsed = run_stage(stage)
        results[stage] = (success, elapsed)

        status = "OK" if success else "ОШИБКА"
        print(f"\n  → Этап {stage}: {status} ({elapsed:.1f} сек)")

    total_time = time.time() - total_start

    # Итоговый отчёт
    print("\n" + "=" * 60)
    print("ИТОГИ")
    print("=" * 60)

    success_count = sum(1 for s, _ in results.values() if s)
    fail_count = len(results) - success_count

    for stage in stages:
        success, elapsed = results[stage]
        status = "✓" if success else "✗"
        print(f"  {status} Этап {stage}: {elapsed:.1f} сек")

    print(f"\nВсего: {success_count} успешно, {fail_count} ошибок")
    print(f"Общее время: {total_time:.1f} сек ({total_time/60:.1f} мин)")

    # Код возврата
    sys.exit(0 if fail_count == 0 else 1)


if __name__ == "__main__":
    main()
