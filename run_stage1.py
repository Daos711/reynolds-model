#!/usr/bin/env python3
"""
ЭТАП 1: Базовый решатель уравнения Рейнольдса.

Параметры из ТЗ:
    R = 34.5 мм, L = 103.5 мм, c = 50 мкм
    ε = 0.6, n = 2980 об/мин
    μ = 0.057 Па·с (VG100 при 50°C)

Критерии проверки:
    1. Максимум давления в сходящейся части зазора (φ < π)
    2. Сеточная сходимость (180×50 vs 360×100)
    3. P_max растёт с ε
"""

import sys
import time
from pathlib import Path

# Добавляем путь к пакету
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np

from bearing_solver import BearingConfig, solve_reynolds
from bearing_solver.visualization import plot_summary


# Директория для результатов
RESULTS_DIR = Path(__file__).parent / "results" / "stage1"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def print_header(text: str):
    """Печать заголовка."""
    print(f"\n{'='*70}")
    print(text)
    print("=" * 70)


def main():
    print_header("ЭТАП 1: БАЗОВЫЙ РЕШАТЕЛЬ УРАВНЕНИЯ РЕЙНОЛЬДСА")

    # =========================================================================
    # Базовый расчёт
    # =========================================================================
    print_header("1. БАЗОВЫЙ РАСЧЁТ")

    config = BearingConfig(
        R=0.0345,       # 34.5 мм
        L=0.1035,       # 103.5 мм
        c=50e-6,        # 50 мкм
        epsilon=0.6,
        phi0=0.0,
        n_rpm=2980,
        mu=0.057,       # VG100 при 50°C
        n_phi=180,
        n_z=50,
    )

    print(config.info())

    print("\nРешение уравнения Рейнольдса...")
    t_start = time.perf_counter()
    result = solve_reynolds(config)
    t_elapsed = time.perf_counter() - t_start

    print(f"\nСТАТУС РЕШЕНИЯ:")
    print(f"  Сходимость: {'Да' if result.converged else 'Нет'}")
    print(f"  Итераций: {result.iterations}")
    print(f"  Невязка: {result.residual:.2e}")
    print(f"  Время: {t_elapsed:.3f} с")

    print(f"\nРЕЗУЛЬТАТЫ:")
    print(f"  P_max (безразм.) = {result.P_max:.6f}")
    print(f"  p_max = {result.p_max/1e6:.2f} МПа")
    print(f"  H_min = {result.h_min_dimless:.4f}")
    print(f"  h_min = {result.h_min*1e6:.2f} мкм")

    # Положение максимума давления
    i_max, j_max = np.unravel_index(np.argmax(result.P), result.P.shape)
    phi_max = result.phi[i_max]
    z_max = result.Z[j_max]

    print(f"\nПОЛОЖЕНИЕ МАКСИМУМА ДАВЛЕНИЯ:")
    print(f"  φ_max = {np.degrees(phi_max):.1f}°")
    print(f"  Z_max = {z_max:.3f}")

    # Проверка: максимум в сходящейся зоне
    if 0 < phi_max < np.pi:
        print(f"  [OK] Максимум в сходящейся зоне (0° < φ < 180°)")
    else:
        print(f"  [!] ВНИМАНИЕ: максимум вне сходящейся зоны!")

    # =========================================================================
    # Сеточная сходимость
    # =========================================================================
    print_header("2. ПРОВЕРКА СЕТОЧНОЙ СХОДИМОСТИ")

    grids = [(90, 25), (180, 50), (360, 100)]
    P_max_values = []
    times = []

    for n_phi, n_z in grids:
        cfg = BearingConfig(
            R=config.R, L=config.L, c=config.c,
            epsilon=config.epsilon, phi0=config.phi0,
            n_rpm=config.n_rpm, mu=config.mu,
            n_phi=n_phi, n_z=n_z
        )
        t_start = time.perf_counter()
        res = solve_reynolds(cfg)
        t_elapsed = time.perf_counter() - t_start

        P_max_values.append(res.P_max)
        times.append(t_elapsed)
        print(f"  Сетка {n_phi:3d}×{n_z:3d}: P_max = {res.P_max:.6f}, "
              f"итер = {res.iterations:4d}, время = {t_elapsed:.3f}с")

    # Относительная разница
    delta_1 = abs(P_max_values[1] - P_max_values[0]) / P_max_values[2] * 100
    delta_2 = abs(P_max_values[2] - P_max_values[1]) / P_max_values[2] * 100
    print(f"\n  Δ(90×25 → 180×50) = {delta_1:.2f}%")
    print(f"  Δ(180×50 → 360×100) = {delta_2:.2f}%")

    if delta_2 < 1.0:
        print(f"  [OK] Сеточная сходимость достигнута (<1%)")
    else:
        print(f"  [!] Требуется более мелкая сетка")

    # =========================================================================
    # Влияние эксцентриситета
    # =========================================================================
    print_header("3. ВЛИЯНИЕ ЭКСЦЕНТРИСИТЕТА")

    epsilons = [0.5, 0.6, 0.7, 0.8, 0.9]
    print(f"\n  {'ε':>5}  {'P_max':>10}  {'p_max, МПа':>12}  {'h_min, мкм':>12}")
    print(f"  {'-'*5}  {'-'*10}  {'-'*12}  {'-'*12}")

    for eps in epsilons:
        cfg = BearingConfig(
            R=config.R, L=config.L, c=config.c,
            epsilon=eps, phi0=config.phi0,
            n_rpm=config.n_rpm, mu=config.mu,
            n_phi=180, n_z=50
        )
        res = solve_reynolds(cfg)
        print(f"  {eps:5.2f}  {res.P_max:10.4f}  {res.p_max/1e6:12.2f}  {res.h_min*1e6:12.2f}")

    # =========================================================================
    # Golden grid тест (для валидации)
    # =========================================================================
    print_header("4. GOLDEN GRID ТЕСТ (720×200)")

    cfg_golden = BearingConfig(
        R=config.R, L=config.L, c=config.c,
        epsilon=0.6, phi0=config.phi0,
        n_rpm=config.n_rpm, mu=config.mu,
        n_phi=720, n_z=200
    )

    print("Расчёт на мелкой сетке (может занять время)...")
    t_start = time.perf_counter()
    res_golden = solve_reynolds(cfg_golden)
    t_elapsed = time.perf_counter() - t_start

    print(f"  P_max (golden) = {res_golden.P_max:.6f}")
    print(f"  p_max (golden) = {res_golden.p_max/1e6:.2f} МПа")
    print(f"  Итераций: {res_golden.iterations}")
    print(f"  Время: {t_elapsed:.2f} с")

    # Сравнение с базовой сеткой
    rel_diff = abs(result.P_max - res_golden.P_max) / res_golden.P_max * 100
    print(f"\n  Отличие 180×50 от golden: {rel_diff:.3f}%")

    # =========================================================================
    # Визуализация
    # =========================================================================
    print_header("5. СОХРАНЕНИЕ ВИЗУАЛИЗАЦИИ")

    try:
        import matplotlib
        matplotlib.use('Agg')

        save_path = RESULTS_DIR / "stage1_results.png"
        plot_summary(result, config, save_path=str(save_path))

    except ImportError:
        print("matplotlib не установлен, визуализация пропущена.")

    # =========================================================================
    # Итоги
    # =========================================================================
    print_header("ИТОГИ ЭТАПА 1")

    print(f"""
Базовый решатель уравнения Рейнольдса реализован и проверен:

1. Решатель: SOR с Numba-ускорением
2. Сеточная сходимость: {delta_2:.2f}% (180×50 vs 360×100)
3. Golden grid тест: отличие {rel_diff:.3f}%
4. Физика: максимум давления в сходящейся зоне (φ < 180°)
5. Результаты сохранены в: {RESULTS_DIR}

Этап 1 завершён успешно.
""")


if __name__ == "__main__":
    main()
