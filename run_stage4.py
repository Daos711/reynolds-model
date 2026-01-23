#!/usr/bin/env python3
"""
Этап 4: Тестирование динамических коэффициентов K и C.

Проверки:
1. Расчёт K и C для известного положения равновесия
2. Проверка диапазонов: K ~ 10⁷-10⁹ Н/м, C ~ 10⁴-10⁶ Н·с/м
3. Анализ чувствительности к delta
4. Сравнение с литературными данными (если доступны)
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path

from bearing_solver import (
    BearingConfig,
    find_equilibrium,
    compute_dynamic_coefficients,
    check_delta_sensitivity,
)


def create_test_config():
    """Создать тестовую конфигурацию подшипника."""
    return BearingConfig(
        R=0.050,        # радиус 50 мм
        L=0.050,        # длина 50 мм (L/D = 0.5)
        c=50e-6,        # зазор 50 мкм
        epsilon=0.6,    # будет переопределено
        phi0=0.0,       # будет переопределено
        n_rpm=3000,     # 3000 об/мин
        mu=0.04,        # вязкость 40 мПа·с
        n_phi=180,
        n_z=50,
    )


def check_range(value, name, min_val, max_val):
    """Проверить, что значение в допустимом диапазоне."""
    if min_val <= abs(value) <= max_val:
        return "OK"
    elif abs(value) < min_val:
        return "LOW"
    else:
        return "HIGH"


def run_basic_test(verbose=True):
    """
    Базовый тест: расчёт K и C для одного положения.
    """
    print("=" * 60)
    print("ТЕСТ 1: Базовый расчёт K и C")
    print("=" * 60)

    config = create_test_config()

    # Типичные параметры
    epsilon = 0.6
    phi0 = np.radians(45)  # 45°

    print(f"\nПараметры подшипника:")
    print(f"  R = {config.R*1000:.1f} мм")
    print(f"  L = {config.L*1000:.1f} мм")
    print(f"  c = {config.c*1e6:.1f} мкм")
    print(f"  n = {config.n_rpm} об/мин")
    print(f"  μ = {config.mu*1000:.1f} мПа·с")
    print(f"\nПоложение:")
    print(f"  ε = {epsilon}")
    print(f"  φ₀ = {np.degrees(phi0):.1f}°")

    coeffs = compute_dynamic_coefficients(
        config, epsilon, phi0,
        delta_e=0.01, delta_v_star=0.01,
        n_phi=180, n_z=50,
        verbose=verbose
    )

    # Проверка диапазонов
    # Для данных параметров (c=50мкм, F0≈94кН):
    # K ~ K* * F0/c ~ 5 * 94000 / 50e-6 ~ 1e10 Н/м
    # C ~ C* * F0/(ωc) ~ 5 * 94000 / (314 * 50e-6) ~ 3e7 Н·с/м
    print("\n" + "-" * 40)
    print("Проверка диапазонов:")

    K_min, K_max = 1e8, 1e11  # Н/м (зависит от параметров)
    C_min, C_max = 1e5, 1e8   # Н·с/м

    results = []
    for name, val, vmin, vmax in [
        ("Kxx", coeffs.Kxx, K_min, K_max),
        ("Kxy", coeffs.Kxy, K_min, K_max),
        ("Kyx", coeffs.Kyx, K_min, K_max),
        ("Kyy", coeffs.Kyy, K_min, K_max),
        ("Cxx", coeffs.Cxx, C_min, C_max),
        ("Cxy", coeffs.Cxy, C_min, C_max),
        ("Cyx", coeffs.Cyx, C_min, C_max),
        ("Cyy", coeffs.Cyy, C_min, C_max),
    ]:
        status = check_range(val, name, vmin, vmax)
        results.append(status)
        if "K" in name:
            print(f"  {name} = {val/1e6:+8.2f} МН/м    [{status}]")
        else:
            print(f"  {name} = {val/1e3:+8.2f} кН·с/м  [{status}]")

    # Общий результат
    ok_count = results.count("OK")
    total = len(results)
    print(f"\nРезультат: {ok_count}/{total} в допустимом диапазоне")

    return coeffs, ok_count == total


def run_equilibrium_test(W_ext=50e3, verbose=True):
    """
    Тест 2: Расчёт K и C в положении равновесия.
    """
    print("\n" + "=" * 60)
    print(f"ТЕСТ 2: K и C в положении равновесия (W={W_ext/1000:.0f} кН)")
    print("=" * 60)

    config = create_test_config()

    # Находим равновесие
    print("\nПоиск равновесия...")
    eq_result = find_equilibrium(
        config,
        W_ext=W_ext,
        load_angle=-np.pi/2,
        verbose=False
    )

    print(f"  ε = {eq_result.epsilon:.4f}")
    print(f"  φ₀ = {np.degrees(eq_result.phi0):.1f}°")
    print(f"  W_achieved = {eq_result.W_achieved/1000:.2f} кН")
    print(f"  Невязка: {eq_result.residual_vec*100:.2f}%")

    # Расчёт K и C
    print("\nРасчёт динамических коэффициентов...")
    coeffs = compute_dynamic_coefficients(
        config,
        eq_result.epsilon,
        eq_result.phi0,
        delta_e=0.01,
        delta_v_star=0.01,
        n_phi=180,
        n_z=50,
        verbose=verbose
    )

    # Вывод матриц
    print("\n" + "-" * 40)
    print("Матрица жёсткости K (МН/м):")
    print(f"  [{coeffs.Kxx/1e6:+8.2f}  {coeffs.Kxy/1e6:+8.2f}]")
    print(f"  [{coeffs.Kyx/1e6:+8.2f}  {coeffs.Kyy/1e6:+8.2f}]")

    print("\nМатрица демпфирования C (кН·с/м):")
    print(f"  [{coeffs.Cxx/1e3:+8.2f}  {coeffs.Cxy/1e3:+8.2f}]")
    print(f"  [{coeffs.Cyx/1e3:+8.2f}  {coeffs.Cyy/1e3:+8.2f}]")

    # Безразмерные коэффициенты
    print("\nБезразмерные коэффициенты:")
    print("K*:")
    print(f"  [{coeffs.Kxx_dimless:+8.4f}  {coeffs.Kxy_dimless:+8.4f}]")
    print(f"  [{coeffs.Kyx_dimless:+8.4f}  {coeffs.Kyy_dimless:+8.4f}]")
    print("C*:")
    print(f"  [{coeffs.Cxx_dimless:+8.4f}  {coeffs.Cxy_dimless:+8.4f}]")
    print(f"  [{coeffs.Cyx_dimless:+8.4f}  {coeffs.Cyy_dimless:+8.4f}]")

    return eq_result, coeffs


def run_sensitivity_test(verbose=True):
    """
    Тест 3: Анализ чувствительности к delta.
    """
    print("\n" + "=" * 60)
    print("ТЕСТ 3: Чувствительность к delta")
    print("=" * 60)

    config = create_test_config()
    epsilon = 0.6
    phi0 = np.radians(45)

    print(f"\nПозиция: ε={epsilon}, φ₀={np.degrees(phi0):.0f}°")

    results = check_delta_sensitivity(
        config, epsilon, phi0,
        delta_values=[0.005, 0.01, 0.02, 0.04],
        n_phi=180, n_z=50,
        verbose=verbose
    )

    # Проверка: изменения < 10% при ×2 вариации delta
    print("\n" + "-" * 40)
    print("Критерий: изменения < 10% при ×2 вариации delta")

    ref = results[0.01]
    test = results[0.02]

    passed = True
    for attr in ['Kxx', 'Kyy', 'Cxx', 'Cyy']:
        ref_val = getattr(ref, attr)
        test_val = getattr(test, attr)
        if ref_val != 0:
            change = abs(test_val - ref_val) / abs(ref_val) * 100
            status = "OK" if change < 10 else "FAIL"
            if status == "FAIL":
                passed = False
            print(f"  Δ{attr}: {change:.1f}% [{status}]")

    return results, passed


def run_epsilon_study(plot=True):
    """
    Тест 4: Зависимость K и C от ε.
    """
    print("\n" + "=" * 60)
    print("ТЕСТ 4: Зависимость K и C от ε")
    print("=" * 60)

    config = create_test_config()
    phi0 = np.radians(45)

    epsilons = [0.2, 0.4, 0.6, 0.8]
    results = []

    for eps in epsilons:
        print(f"\nε = {eps}...")
        coeffs = compute_dynamic_coefficients(
            config, eps, phi0,
            delta_e=0.01, delta_v_star=0.01,
            n_phi=120, n_z=30,  # грубая сетка для скорости
            verbose=False
        )
        results.append(coeffs)
        print(f"  Kxx = {coeffs.Kxx/1e6:.2f} МН/м, Kyy = {coeffs.Kyy/1e6:.2f} МН/м")
        print(f"  Cxx = {coeffs.Cxx/1e3:.2f} кН·с/м, Cyy = {coeffs.Cyy/1e3:.2f} кН·с/м")

    if plot:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # K vs ε
        ax1 = axes[0]
        ax1.plot(epsilons, [r.Kxx/1e6 for r in results], 'o-', label='Kxx')
        ax1.plot(epsilons, [r.Kxy/1e6 for r in results], 's-', label='Kxy')
        ax1.plot(epsilons, [r.Kyx/1e6 for r in results], '^-', label='Kyx')
        ax1.plot(epsilons, [r.Kyy/1e6 for r in results], 'd-', label='Kyy')
        ax1.set_xlabel('ε')
        ax1.set_ylabel('K, МН/м')
        ax1.set_title('Жёсткость vs эксцентриситет')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # C vs ε
        ax2 = axes[1]
        ax2.plot(epsilons, [r.Cxx/1e3 for r in results], 'o-', label='Cxx')
        ax2.plot(epsilons, [r.Cxy/1e3 for r in results], 's-', label='Cxy')
        ax2.plot(epsilons, [r.Cyx/1e3 for r in results], '^-', label='Cyx')
        ax2.plot(epsilons, [r.Cyy/1e3 for r in results], 'd-', label='Cyy')
        ax2.set_xlabel('ε')
        ax2.set_ylabel('C, кН·с/м')
        ax2.set_title('Демпфирование vs эксцентриситет')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        out_dir = Path("output")
        out_dir.mkdir(exist_ok=True)
        plt.savefig(out_dir / 'stage4_epsilon_study.png', dpi=150)
        print(f"\nГрафик сохранён: {out_dir / 'stage4_epsilon_study.png'}")
        plt.close()

    return epsilons, results


def main():
    parser = argparse.ArgumentParser(description="Тестирование Этапа 4: K и C")
    parser.add_argument("--test", type=int, default=0,
                       help="Номер теста (0=все, 1-4)")
    parser.add_argument("--plot", action="store_true", default=True,
                       help="Показывать графики")
    parser.add_argument("--no-plot", dest="plot", action="store_false")
    parser.add_argument("-v", "--verbose", action="store_true", default=True)
    parser.add_argument("-q", "--quiet", dest="verbose", action="store_false")
    args = parser.parse_args()

    print("ЭТАП 4: Динамические коэффициенты K и C")
    print("=" * 60)

    all_passed = True

    if args.test == 0 or args.test == 1:
        coeffs, passed = run_basic_test(verbose=args.verbose)
        if not passed:
            all_passed = False

    if args.test == 0 or args.test == 2:
        eq_result, coeffs = run_equilibrium_test(W_ext=50e3, verbose=args.verbose)

    if args.test == 0 or args.test == 3:
        results, passed = run_sensitivity_test(verbose=args.verbose)
        if not passed:
            all_passed = False

    if args.test == 0 or args.test == 4:
        epsilons, results = run_epsilon_study(plot=args.plot)

    # Итог
    print("\n" + "=" * 60)
    if all_passed:
        print("ЭТАП 4: ВСЕ ТЕСТЫ ПРОЙДЕНЫ")
    else:
        print("ЭТАП 4: ЕСТЬ ПРОБЛЕМЫ (см. выше)")
    print("=" * 60)


if __name__ == "__main__":
    main()
