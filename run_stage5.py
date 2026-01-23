#!/usr/bin/env python3
"""
Этап 5: Анализ устойчивости ротора (Jeffcott rotor).

Тесты:
1. Одна точка: расчёт собственных значений для заданных параметров
2. Зависимость от скорости n: max(Re(λ)) vs n
3. Зависимость от нагрузки W: max(Re(λ)) vs W
4. Влияние массы ротора m

Выходы:
- CSV с результатами
- Графики: max_real_vs_n.png, max_real_vs_W.png
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse
from pathlib import Path

from bearing_solver import (
    BearingConfig,
    find_equilibrium,
    compute_dynamic_coefficients,
    analyze_stability,
    StabilityResult,
)


def create_test_config(n_rpm=3000):
    """Создать тестовую конфигурацию подшипника."""
    return BearingConfig(
        R=0.050,        # радиус 50 мм
        L=0.050,        # длина 50 мм (L/D = 0.5)
        c=50e-6,        # зазор 50 мкм
        epsilon=0.6,    # будет переопределено
        phi0=0.0,       # будет переопределено
        n_rpm=n_rpm,
        mu=0.04,        # вязкость 40 мПа·с
        n_phi=180,
        n_z=50,
    )


def run_single_point_test(mass=30.0, W_ext=50e3, n_rpm=3000, verbose=True):
    """
    Тест 1: Расчёт устойчивости в одной точке.
    """
    print("=" * 60)
    print("ТЕСТ 1: Анализ устойчивости в одной точке")
    print("=" * 60)

    config = create_test_config(n_rpm)

    print(f"\nПараметры:")
    print(f"  Масса ротора: m = {mass} кг")
    print(f"  Нагрузка: W = {W_ext/1000:.0f} кН")
    print(f"  Скорость: n = {n_rpm} об/мин")

    # Находим равновесие
    print("\nПоиск равновесия...")
    eq = find_equilibrium(config, W_ext=W_ext, load_angle=-np.pi/2, verbose=False)
    print(f"  ε = {eq.epsilon:.4f}")
    print(f"  φ₀ = {np.degrees(eq.phi0):.1f}°")

    # Вычисляем K, C
    print("\nРасчёт K и C...")
    coeffs = compute_dynamic_coefficients(
        config, eq.epsilon, eq.phi0,
        delta_e=0.01, delta_v_star=0.01,
        n_phi=180, n_z=50, verbose=False
    )

    print(f"\nМатрица K (МН/м):")
    print(f"  [{coeffs.Kxx/1e6:+8.2f}  {coeffs.Kxy/1e6:+8.2f}]")
    print(f"  [{coeffs.Kyx/1e6:+8.2f}  {coeffs.Kyy/1e6:+8.2f}]")

    print(f"\nМатрица C (кН·с/м):")
    print(f"  [{coeffs.Cxx/1e3:+8.2f}  {coeffs.Cxy/1e3:+8.2f}]")
    print(f"  [{coeffs.Cyx/1e3:+8.2f}  {coeffs.Cyy/1e3:+8.2f}]")

    # Анализ устойчивости
    print("\nАнализ устойчивости...")
    stab = analyze_stability(coeffs.K, coeffs.C, mass, n_rpm)

    print(f"\nСобственные значения матрицы A:")
    for i, lam in enumerate(stab.eigenvalues):
        print(f"  λ{i+1} = {lam.real:+.2f} {'+' if lam.imag >= 0 else '-'} {abs(lam.imag):.2f}i")

    print(f"\n" + "-" * 40)
    status = "УСТОЙЧИВО" if stab.is_stable else "НЕУСТОЙЧИВО"
    print(f"Статус: {status}")
    print(f"  Запас устойчивости: {stab.stability_margin:.2f} 1/с")
    print(f"  Доминирующее λ: {stab.dominant_real:.2f} ± {stab.dominant_imag:.2f}i")
    print(f"  Частота вихря: {stab.dominant_freq_hz:.1f} Гц")
    print(f"  Скорость вала: {stab.shaft_speed_hz:.1f} Гц")
    print(f"  Whirl ratio γ: {stab.whirl_ratio:.3f}")

    # Проверка типичного диапазона whirl ratio
    if 0.3 < stab.whirl_ratio < 0.6:
        print(f"    (в типичном диапазоне oil whirl 0.4-0.5)")
    elif stab.whirl_ratio > 0:
        print(f"    (вне типичного диапазона)")

    return stab, coeffs, eq


def run_speed_study(mass=30.0, W_ext=50e3, n_range=(500, 5000), n_points=20, plot=True):
    """
    Тест 2: Зависимость устойчивости от скорости вращения.
    """
    print("\n" + "=" * 60)
    print("ТЕСТ 2: Зависимость от скорости вращения n")
    print("=" * 60)

    print(f"\nПараметры:")
    print(f"  Масса ротора: m = {mass} кг")
    print(f"  Нагрузка: W = {W_ext/1000:.0f} кН")
    print(f"  Диапазон скоростей: {n_range[0]}-{n_range[1]} об/мин")

    n_values = np.linspace(n_range[0], n_range[1], n_points)
    results = []

    for n_rpm in n_values:
        config = create_test_config(n_rpm)

        try:
            eq = find_equilibrium(config, W_ext=W_ext, load_angle=-np.pi/2, verbose=False)
            coeffs = compute_dynamic_coefficients(
                config, eq.epsilon, eq.phi0,
                delta_e=0.01, delta_v_star=0.01,
                n_phi=120, n_z=30, verbose=False  # грубая сетка для скорости
            )
            stab = analyze_stability(coeffs.K, coeffs.C, mass, n_rpm)

            results.append({
                'n_rpm': n_rpm,
                'epsilon': eq.epsilon,
                'phi0_deg': np.degrees(eq.phi0),
                'Kxx': coeffs.Kxx, 'Kxy': coeffs.Kxy,
                'Kyx': coeffs.Kyx, 'Kyy': coeffs.Kyy,
                'Cxx': coeffs.Cxx, 'Cxy': coeffs.Cxy,
                'Cyx': coeffs.Cyx, 'Cyy': coeffs.Cyy,
                'max_real': stab.dominant_real,
                'is_stable': stab.is_stable,
                'margin': stab.stability_margin,
                'whirl_ratio': stab.whirl_ratio,
                'freq_hz': stab.dominant_freq_hz,
                'mass': mass,
                'W_ext': W_ext,
            })

            status = "OK" if stab.is_stable else "UNSTABLE"
            print(f"n={n_rpm:5.0f}: ε={eq.epsilon:.3f}, max_Re={stab.dominant_real:+8.1f}, "
                  f"γ={stab.whirl_ratio:.3f} [{status}]")

        except Exception as e:
            print(f"n={n_rpm:5.0f}: ОШИБКА - {e}")

    # Сохраняем CSV
    df = pd.DataFrame(results)
    out_dir = Path("results/stage5")
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / 'speed_study.csv'
    df.to_csv(csv_path, index=False)
    print(f"\nCSV сохранён: {csv_path}")

    # График
    if plot and len(results) > 0:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        n_arr = [r['n_rpm'] for r in results]
        max_real_arr = [r['max_real'] for r in results]
        whirl_arr = [r['whirl_ratio'] for r in results]
        eps_arr = [r['epsilon'] for r in results]
        margin_arr = [r['margin'] for r in results]

        # max(Re(λ)) vs n
        ax1 = axes[0, 0]
        colors = ['green' if r['is_stable'] else 'red' for r in results]
        ax1.scatter(n_arr, max_real_arr, c=colors, s=50)
        ax1.axhline(y=0, color='k', linestyle='--', linewidth=1)
        ax1.set_xlabel('n, об/мин')
        ax1.set_ylabel('max(Re(λ)), 1/с')
        ax1.set_title('Доминирующее собственное значение')
        ax1.grid(True, alpha=0.3)

        # Whirl ratio vs n
        ax2 = axes[0, 1]
        ax2.plot(n_arr, whirl_arr, 'o-', color='blue')
        ax2.axhline(y=0.5, color='orange', linestyle='--', label='γ = 0.5 (oil whirl)')
        ax2.set_xlabel('n, об/мин')
        ax2.set_ylabel('Whirl ratio γ')
        ax2.set_title('Отношение частоты вихря к скорости вала')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Epsilon vs n
        ax3 = axes[1, 0]
        ax3.plot(n_arr, eps_arr, 's-', color='purple')
        ax3.set_xlabel('n, об/мин')
        ax3.set_ylabel('ε')
        ax3.set_title('Эксцентриситет в равновесии')
        ax3.grid(True, alpha=0.3)

        # Stability margin vs n
        ax4 = axes[1, 1]
        ax4.fill_between(n_arr, margin_arr, 0, where=[m > 0 for m in margin_arr],
                        color='green', alpha=0.3, label='Устойчиво')
        ax4.fill_between(n_arr, margin_arr, 0, where=[m <= 0 for m in margin_arr],
                        color='red', alpha=0.3, label='Неустойчиво')
        ax4.plot(n_arr, margin_arr, 'k-', linewidth=2)
        ax4.axhline(y=0, color='k', linestyle='--')
        ax4.set_xlabel('n, об/мин')
        ax4.set_ylabel('Запас устойчивости, 1/с')
        ax4.set_title('Запас устойчивости')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.suptitle(f'Зависимость от скорости (m={mass} кг, W={W_ext/1000:.0f} кН)', fontsize=14)
        plt.tight_layout()
        plt.savefig(out_dir / 'max_real_vs_n.png', dpi=150)
        print(f"График сохранён: {out_dir / 'max_real_vs_n.png'}")
        plt.close()

    return results


def run_load_study(mass=30.0, n_rpm=3000, W_range=(10e3, 100e3), n_points=10, plot=True):
    """
    Тест 3: Зависимость устойчивости от нагрузки.
    """
    print("\n" + "=" * 60)
    print("ТЕСТ 3: Зависимость от нагрузки W")
    print("=" * 60)

    print(f"\nПараметры:")
    print(f"  Масса ротора: m = {mass} кг")
    print(f"  Скорость: n = {n_rpm} об/мин")
    print(f"  Диапазон нагрузок: {W_range[0]/1000:.0f}-{W_range[1]/1000:.0f} кН")

    W_values = np.linspace(W_range[0], W_range[1], n_points)
    results = []

    config = create_test_config(n_rpm)

    for W_ext in W_values:
        try:
            eq = find_equilibrium(config, W_ext=W_ext, load_angle=-np.pi/2, verbose=False)
            coeffs = compute_dynamic_coefficients(
                config, eq.epsilon, eq.phi0,
                delta_e=0.01, delta_v_star=0.01,
                n_phi=120, n_z=30, verbose=False
            )
            stab = analyze_stability(coeffs.K, coeffs.C, mass, n_rpm)

            results.append({
                'W_ext': W_ext,
                'epsilon': eq.epsilon,
                'phi0_deg': np.degrees(eq.phi0),
                'Kxx': coeffs.Kxx, 'Kxy': coeffs.Kxy,
                'Kyx': coeffs.Kyx, 'Kyy': coeffs.Kyy,
                'Cxx': coeffs.Cxx, 'Cxy': coeffs.Cxy,
                'Cyx': coeffs.Cyx, 'Cyy': coeffs.Cyy,
                'max_real': stab.dominant_real,
                'is_stable': stab.is_stable,
                'margin': stab.stability_margin,
                'whirl_ratio': stab.whirl_ratio,
                'freq_hz': stab.dominant_freq_hz,
                'mass': mass,
                'n_rpm': n_rpm,
            })

            status = "OK" if stab.is_stable else "UNSTABLE"
            print(f"W={W_ext/1000:5.0f} кН: ε={eq.epsilon:.3f}, max_Re={stab.dominant_real:+8.1f}, "
                  f"γ={stab.whirl_ratio:.3f} [{status}]")

        except Exception as e:
            print(f"W={W_ext/1000:5.0f} кН: ОШИБКА - {e}")

    # Сохраняем CSV
    df = pd.DataFrame(results)
    out_dir = Path("results/stage5")
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / 'load_study.csv'
    df.to_csv(csv_path, index=False)
    print(f"\nCSV сохранён: {csv_path}")

    # График
    if plot and len(results) > 0:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        W_arr = [r['W_ext']/1000 for r in results]
        max_real_arr = [r['max_real'] for r in results]
        eps_arr = [r['epsilon'] for r in results]
        margin_arr = [r['margin'] for r in results]
        whirl_arr = [r['whirl_ratio'] for r in results]

        # max(Re(λ)) vs W
        ax1 = axes[0, 0]
        colors = ['green' if r['is_stable'] else 'red' for r in results]
        ax1.scatter(W_arr, max_real_arr, c=colors, s=50)
        ax1.axhline(y=0, color='k', linestyle='--', linewidth=1)
        ax1.set_xlabel('W, кН')
        ax1.set_ylabel('max(Re(λ)), 1/с')
        ax1.set_title('Доминирующее собственное значение')
        ax1.grid(True, alpha=0.3)

        # max(Re(λ)) vs ε
        ax2 = axes[0, 1]
        ax2.scatter(eps_arr, max_real_arr, c=colors, s=50)
        ax2.axhline(y=0, color='k', linestyle='--', linewidth=1)
        ax2.set_xlabel('ε')
        ax2.set_ylabel('max(Re(λ)), 1/с')
        ax2.set_title('Устойчивость vs эксцентриситет')
        ax2.grid(True, alpha=0.3)

        # Epsilon vs W
        ax3 = axes[1, 0]
        ax3.plot(W_arr, eps_arr, 's-', color='purple')
        ax3.set_xlabel('W, кН')
        ax3.set_ylabel('ε')
        ax3.set_title('Эксцентриситет в равновесии')
        ax3.grid(True, alpha=0.3)

        # Whirl ratio vs W
        ax4 = axes[1, 1]
        ax4.plot(W_arr, whirl_arr, 'o-', color='blue')
        ax4.axhline(y=0.5, color='orange', linestyle='--', label='γ = 0.5')
        ax4.set_xlabel('W, кН')
        ax4.set_ylabel('Whirl ratio γ')
        ax4.set_title('Отношение частоты вихря к скорости вала')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.suptitle(f'Зависимость от нагрузки (m={mass} кг, n={n_rpm} об/мин)', fontsize=14)
        plt.tight_layout()
        plt.savefig(out_dir / 'max_real_vs_W.png', dpi=150)
        print(f"График сохранён: {out_dir / 'max_real_vs_W.png'}")
        plt.close()

    return results


def run_mass_study(W_ext=50e3, n_rpm=3000, mass_values=None, plot=True):
    """
    Тест 4: Влияние массы ротора на устойчивость.
    """
    print("\n" + "=" * 60)
    print("ТЕСТ 4: Влияние массы ротора m")
    print("=" * 60)

    if mass_values is None:
        mass_values = [10, 30, 50, 100]

    print(f"\nПараметры:")
    print(f"  Нагрузка: W = {W_ext/1000:.0f} кН")
    print(f"  Скорость: n = {n_rpm} об/мин")
    print(f"  Массы: {mass_values} кг")

    config = create_test_config(n_rpm)

    # Находим равновесие (одинаково для всех масс)
    eq = find_equilibrium(config, W_ext=W_ext, load_angle=-np.pi/2, verbose=False)
    coeffs = compute_dynamic_coefficients(
        config, eq.epsilon, eq.phi0,
        delta_e=0.01, delta_v_star=0.01,
        n_phi=180, n_z=50, verbose=False
    )

    print(f"\nРавновесие: ε = {eq.epsilon:.4f}, φ₀ = {np.degrees(eq.phi0):.1f}°")

    results = []
    for mass in mass_values:
        stab = analyze_stability(coeffs.K, coeffs.C, mass, n_rpm)

        results.append({
            'mass': mass,
            'max_real': stab.dominant_real,
            'is_stable': stab.is_stable,
            'margin': stab.stability_margin,
            'whirl_ratio': stab.whirl_ratio,
            'freq_hz': stab.dominant_freq_hz,
        })

        status = "OK" if stab.is_stable else "UNSTABLE"
        print(f"m={mass:4.0f} кг: max_Re={stab.dominant_real:+8.1f}, "
              f"f={stab.dominant_freq_hz:.1f} Гц, γ={stab.whirl_ratio:.3f} [{status}]")

    # График
    if plot and len(results) > 0:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        mass_arr = [r['mass'] for r in results]
        margin_arr = [r['margin'] for r in results]
        freq_arr = [r['freq_hz'] for r in results]

        ax1 = axes[0]
        colors = ['green' if r['is_stable'] else 'red' for r in results]
        ax1.bar(mass_arr, margin_arr, color=colors, width=8, edgecolor='black')
        ax1.axhline(y=0, color='k', linestyle='--')
        ax1.set_xlabel('Масса m, кг')
        ax1.set_ylabel('Запас устойчивости, 1/с')
        ax1.set_title('Запас устойчивости vs масса')
        ax1.grid(True, alpha=0.3, axis='y')

        ax2 = axes[1]
        ax2.bar(mass_arr, freq_arr, color='blue', width=8, edgecolor='black', alpha=0.7)
        ax2.set_xlabel('Масса m, кг')
        ax2.set_ylabel('Частота вихря, Гц')
        ax2.set_title('Частота вихревого движения vs масса')
        ax2.grid(True, alpha=0.3, axis='y')

        plt.suptitle(f'Влияние массы (W={W_ext/1000:.0f} кН, n={n_rpm} об/мин)', fontsize=14)
        plt.tight_layout()

        out_dir = Path("results/stage5")
        out_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_dir / 'mass_study.png', dpi=150)
        print(f"\nГрафик сохранён: {out_dir / 'mass_study.png'}")
        plt.close()

    return results


def main():
    parser = argparse.ArgumentParser(description="Этап 5: Анализ устойчивости")
    parser.add_argument("--test", type=int, default=0,
                       help="Номер теста (0=все, 1-4)")
    parser.add_argument("--mass", type=float, default=30.0,
                       help="Масса ротора, кг")
    parser.add_argument("--W", type=float, default=50.0,
                       help="Нагрузка, кН")
    parser.add_argument("--n", type=float, default=3000.0,
                       help="Скорость вращения, об/мин")
    parser.add_argument("--plot", action="store_true", default=True)
    parser.add_argument("--no-plot", dest="plot", action="store_false")
    args = parser.parse_args()

    print("ЭТАП 5: Анализ устойчивости ротора")
    print("=" * 60)

    W_ext = args.W * 1000  # кН -> Н

    if args.test == 0 or args.test == 1:
        stab, coeffs, eq = run_single_point_test(
            mass=args.mass, W_ext=W_ext, n_rpm=args.n
        )

    if args.test == 0 or args.test == 2:
        results = run_speed_study(
            mass=args.mass, W_ext=W_ext, n_range=(500, 5000),
            n_points=20, plot=args.plot
        )

    if args.test == 0 or args.test == 3:
        results = run_load_study(
            mass=args.mass, n_rpm=args.n, W_range=(10e3, 100e3),
            n_points=10, plot=args.plot
        )

    if args.test == 0 or args.test == 4:
        results = run_mass_study(
            W_ext=W_ext, n_rpm=args.n, mass_values=[10, 30, 50, 100],
            plot=args.plot
        )

    print("\n" + "=" * 60)
    print("ЭТАП 5: ЗАВЕРШЁН")
    print("=" * 60)


if __name__ == "__main__":
    main()
