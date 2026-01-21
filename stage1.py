#!/usr/bin/env python3
"""
Пример использования пакета bearing_solver.

Этап 1: Базовый решатель уравнения Рейнольдса.

Параметры из ТЗ:
  R = 34.5 мм, L = 103.5 мм, c = 50 мкм
  ε = 0.6, n = 2980 об/мин
  μ = 0.057 Па·с (VG100 при 50°C)
"""

import sys
import numpy as np

# Добавляем путь к пакету
sys.path.insert(0, '/home/user/reynolds-model')

from bearing_solver import BearingConfig, SmoothFilmModel, ReynoldsSolver, solve_reynolds


def main():
    print("=" * 70)
    print("РАСЧЁТ ГИДРОДИНАМИЧЕСКОГО ПОДШИПНИКА СКОЛЬЖЕНИЯ")
    print("Этап 1: Базовый решатель уравнения Рейнольдса")
    print("=" * 70)

    # Конфигурация из ТЗ
    config = BearingConfig(
        R=0.0345,       # 34.5 мм
        L=0.1035,       # 103.5 мм
        c=50e-6,        # 50 мкм
        epsilon=0.6,
        phi0=0.0,
        n_rpm=2980,
        mu=0.057,       # VG100 при 50°C
        n_phi=180,      # точек по окружности
        n_z=50,         # точек по длине
    )

    print(f"\nПАРАМЕТРЫ ПОДШИПНИКА:")
    print(f"  Радиус R = {config.R*1000:.1f} мм")
    print(f"  Диаметр D = {config.D*1000:.1f} мм")
    print(f"  Длина L = {config.L*1000:.1f} мм")
    print(f"  Отношение L/D = {config.L_D_ratio:.2f}")
    print(f"  Зазор c = {config.c*1e6:.1f} мкм")

    print(f"\nРЕЖИМ РАБОТЫ:")
    print(f"  Эксцентриситет ε = {config.epsilon}")
    print(f"  Угол положения φ₀ = {np.degrees(config.phi0):.1f}°")
    print(f"  Скорость вращения n = {config.n_rpm} об/мин")
    print(f"  Угловая скорость ω = {config.omega:.2f} рад/с")
    print(f"  Линейная скорость U = {config.U:.2f} м/с")
    print(f"  Вязкость μ = {config.mu} Па·с")

    print(f"\nСЕТКА:")
    print(f"  Точек по φ: {config.n_phi}")
    print(f"  Точек по Z: {config.n_z}")

    print(f"\nМАСШТАБНЫЕ КОЭФФИЦИЕНТЫ:")
    print(f"  Масштаб давления = {config.pressure_scale:.2e} Па")
    print(f"  Масштаб сил = {config.force_scale:.2e} Н")

    # Решение уравнения Рейнольдса
    print(f"\n{'='*70}")
    print("РЕШЕНИЕ УРАВНЕНИЯ РЕЙНОЛЬДСА")
    print("=" * 70)

    result = solve_reynolds(config)  # SOR по умолчанию

    print(f"\nСТАТУС РЕШЕНИЯ:")
    print(f"  Метод: {result.method.upper()}")
    print(f"  Сходимость: {'Да' if result.converged else 'Нет'}")
    print(f"  Итераций: {result.iterations}")
    print(f"  Невязка (норм.): {result.residual:.2e}")

    print(f"\nРЕЗУЛЬТАТЫ:")
    print(f"  Максимальное безразмерное давление P_max = {result.P_max:.6f}")
    print(f"  Максимальное размерное давление p_max = {result.p_max/1e6:.3f} МПа")
    print(f"  Минимальная безразмерная толщина H_min = {result.h_min_dimless:.4f}")
    print(f"  Минимальная размерная толщина h_min = {result.h_min*1e6:.2f} мкм")

    # Положение максимума давления
    i_max, j_max = np.unravel_index(np.argmax(result.P), result.P.shape)
    phi_max = result.phi[i_max]
    z_max = result.Z[j_max]

    print(f"\nПОЛОЖЕНИЕ МАКСИМУМА ДАВЛЕНИЯ:")
    print(f"  φ_max = {np.degrees(phi_max):.1f}°")
    print(f"  Z_max = {z_max:.3f}")

    # Проверка: максимум должен быть в сходящейся зоне (0 < φ < π при φ₀ = 0)
    if 0 < phi_max < np.pi:
        print(f"  ✓ Максимум в сходящейся зоне (0° < φ < 180°)")
    else:
        print(f"  ✗ ВНИМАНИЕ: максимум вне сходящейся зоны!")

    # Сравнение с разными сетками
    print(f"\n{'='*70}")
    print("ПРОВЕРКА СЕТОЧНОЙ СХОДИМОСТИ")
    print("=" * 70)

    grids = [(90, 25), (180, 50), (270, 75)]  # уменьшил мелкую сетку для скорости
    P_max_values = []

    for n_phi, n_z in grids:
        cfg = BearingConfig(
            R=config.R, L=config.L, c=config.c,
            epsilon=config.epsilon, phi0=config.phi0,
            n_rpm=config.n_rpm, mu=config.mu,
            n_phi=n_phi, n_z=n_z
        )
        res = solve_reynolds(cfg)  # SOR по умолчанию
        P_max_values.append(res.P_max)
        print(f"  Сетка {n_phi:3d}×{n_z:3d}: P_max = {res.P_max:.6f}, итераций: {res.iterations}")

    # Относительная разница
    delta_1 = abs(P_max_values[1] - P_max_values[0]) / P_max_values[2] * 100
    delta_2 = abs(P_max_values[2] - P_max_values[1]) / P_max_values[2] * 100
    print(f"\n  Δ(90×25 → 180×50) = {delta_1:.2f}%")
    print(f"  Δ(180×50 → 360×100) = {delta_2:.2f}%")

    # Влияние эксцентриситета
    print(f"\n{'='*70}")
    print("ВЛИЯНИЕ ЭКСЦЕНТРИСИТЕТА")
    print("=" * 70)

    epsilons = [0.6, 0.7, 0.8, 0.9]
    print(f"\n  {'ε':>5}  {'P_max':>10}  {'p_max, МПа':>12}  {'h_min, мкм':>12}")
    print(f"  {'-'*5}  {'-'*10}  {'-'*12}  {'-'*12}")

    for eps in epsilons:
        cfg = BearingConfig(
            R=config.R, L=config.L, c=config.c,
            epsilon=eps, phi0=config.phi0,
            n_rpm=config.n_rpm, mu=config.mu,
            n_phi=180, n_z=50
        )
        res = solve_reynolds(cfg)  # SOR по умолчанию
        print(f"  {eps:5.2f}  {res.P_max:10.4f}  {res.p_max/1e6:12.3f}  {res.h_min*1e6:12.2f}")

    # Визуализация (если matplotlib доступен)
    try:
        import matplotlib
        matplotlib.use('Agg')  # Для сохранения без дисплея
        import matplotlib.pyplot as plt
        import os
        from bearing_solver.visualization import plot_summary

        # Создаём папку если её нет
        os.makedirs('results/stage_1', exist_ok=True)

        print(f"\n{'='*70}")
        print("СОХРАНЕНИЕ ВИЗУАЛИЗАЦИИ")
        print("=" * 70)

        plot_summary(result, config, save_path='results/stage_1/pressure_field.png')

    except ImportError:
        print("\nМодуль matplotlib не установлен, визуализация пропущена.")

    print(f"\n{'='*70}")
    print("РАСЧЁТ ЗАВЕРШЁН")
    print("=" * 70)


if __name__ == "__main__":
    main()
