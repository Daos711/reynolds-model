#!/usr/bin/env python3
"""
ЭТАП 3: Поиск положения равновесия вала.

Задача: по заданной внешней нагрузке W_ext найти (ε, φ₀),
при которых несущая способность плёнки уравновешивает нагрузку.

Тесты:
    1. W_ext = 10 кН → ε ≈ 0.3-0.5
    2. W_ext = 20 кН → ε ≈ 0.4-0.6
    3. W_ext = 50 кН → ε ≈ 0.5-0.7
    4. W_ext = 100 кН → ε ≈ 0.6-0.8
    5. Невязка |W - W_ext| / W_ext < 1%
    6. Сходимость за < 50 итераций
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd

from bearing_solver import (
    BearingConfig,
    find_equilibrium,
    find_equilibrium_1d,
)


# Директория для результатов
RESULTS_DIR = Path(__file__).parent / "results" / "stage3"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def print_header(text: str):
    """Печать заголовка."""
    print(f"\n{'='*70}")
    print(text)
    print("=" * 70)


def create_base_config() -> BearingConfig:
    """Создать базовую конфигурацию (ε и φ₀ будут найдены)."""
    return BearingConfig(
        R=0.0345,       # 34.5 мм
        L=0.1035,       # 103.5 мм
        c=50e-6,        # 50 мкм
        epsilon=0.5,    # начальное приближение (будет изменено)
        phi0=0.0,       # начальное приближение (будет изменено)
        n_rpm=2980,
        mu=0.057,       # VG100 при 50°C
        n_phi=90,       # уменьшенная сетка для скорости
        n_z=25,
    )


def test_single_load(base_config: BearingConfig, W_ext_kN: float, verbose: bool = False):
    """Тест для одной нагрузки."""
    W_ext = W_ext_kN * 1000  # Н

    print(f"\n--- W_ext = {W_ext_kN:.0f} кН ---")

    t_start = time.perf_counter()
    result = find_equilibrium(
        base_config,
        W_ext=W_ext,
        load_angle=-np.pi/2,  # вертикально вниз
        verbose=verbose
    )
    t_elapsed = time.perf_counter() - t_start

    # Вывод результатов
    print(f"  Найдено: ε = {result.epsilon:.4f}, φ₀ = {np.degrees(result.phi0):.1f}°")
    print(f"  W = {result.W_achieved/1000:.3f} кН (цель: {W_ext_kN:.0f} кН)")
    print(f"  Невязка: {result.residual*100:.4f}%")
    print(f"  Итераций: {result.iterations}, время: {t_elapsed:.2f}с")

    if result.stage2:
        s2 = result.stage2
        print(f"  h_min = {s2.reynolds.h_min*1e6:.2f} мкм")
        print(f"  p_max = {s2.reynolds.p_max/1e6:.2f} МПа")
        print(f"  μ_fr = {s2.friction.mu_friction:.6f}")

    # Проверки
    if result.converged:
        print(f"  [OK] Сходимость достигнута")
    else:
        print(f"  [!] НЕ сошлось: {result.message}")

    if result.residual < 0.01:
        print(f"  [OK] Невязка < 1%")
    else:
        print(f"  [!] Невязка > 1%")

    return result


def run_load_series():
    """Серия тестов для разных нагрузок."""
    print_header("1. СЕРИЯ ТЕСТОВ ДЛЯ РАЗНЫХ НАГРУЗОК")

    base_config = create_base_config()

    print(f"\nПараметры подшипника:")
    print(f"  R = {base_config.R*1000:.1f} мм, L = {base_config.L*1000:.1f} мм")
    print(f"  c = {base_config.c*1e6:.1f} мкм")
    print(f"  n = {base_config.n_rpm} об/мин, μ = {base_config.mu} Па·с")
    print(f"  Сетка: {base_config.n_phi}×{base_config.n_z}")

    loads_kN = [10, 20, 30, 50, 70, 100]
    results = []

    for W_kN in loads_kN:
        result = test_single_load(base_config, W_kN)
        results.append({
            'W_ext_kN': W_kN,
            'epsilon': result.epsilon,
            'phi0_deg': np.degrees(result.phi0),
            'W_achieved_kN': result.W_achieved / 1000,
            'residual_pct': result.residual * 100,
            'iterations': result.iterations,
            'converged': result.converged,
            'h_min_um': result.stage2.reynolds.h_min * 1e6 if result.stage2 else None,
            'p_max_MPa': result.stage2.reynolds.p_max / 1e6 if result.stage2 else None,
            'mu_friction': result.stage2.friction.mu_friction if result.stage2 else None,
        })

    return results


def run_summary_table(results: list):
    """Сводная таблица результатов."""
    print_header("2. СВОДНАЯ ТАБЛИЦА")

    print(f"\n{'W_ext':>8} {'ε':>8} {'φ₀, °':>8} {'W, кН':>8} {'δ, %':>8} "
          f"{'iter':>6} {'h_min':>8} {'p_max':>8} {'μ_fr':>10}")
    print("-" * 95)

    for r in results:
        status = "✓" if r['converged'] and r['residual_pct'] < 1 else "!"
        print(f"{r['W_ext_kN']:>7.0f} {r['epsilon']:>8.4f} {r['phi0_deg']:>8.1f} "
              f"{r['W_achieved_kN']:>8.3f} {r['residual_pct']:>8.4f} "
              f"{r['iterations']:>6d} {r['h_min_um']:>8.2f} {r['p_max_MPa']:>8.2f} "
              f"{r['mu_friction']:>10.6f} {status}")

    # Сохраняем CSV
    df = pd.DataFrame(results)
    df.to_csv(RESULTS_DIR / "equilibrium_results.csv", index=False)
    print(f"\nСохранено: {RESULTS_DIR / 'equilibrium_results.csv'}")


def run_validation():
    """Валидация: проверка обратного расчёта."""
    print_header("3. ВАЛИДАЦИЯ: ОБРАТНЫЙ РАСЧЁТ")

    base_config = create_base_config()

    # Берём известный ε и вычисляем W
    test_epsilon = 0.6
    config_test = BearingConfig(
        R=base_config.R, L=base_config.L, c=base_config.c,
        epsilon=test_epsilon, phi0=np.radians(180),
        n_rpm=base_config.n_rpm, mu=base_config.mu,
        n_phi=90, n_z=25
    )

    from bearing_solver import solve_reynolds, compute_stage2
    reynolds = solve_reynolds(config_test)
    stage2 = compute_stage2(reynolds, config_test)

    W_from_direct = stage2.forces.W
    print(f"\nПрямой расчёт: ε = {test_epsilon}, φ₀ = 180°")
    print(f"  W = {W_from_direct/1000:.3f} кН")

    # Теперь ищем равновесие для этой нагрузки
    print(f"\nОбратный расчёт: ищем ε для W_ext = {W_from_direct/1000:.3f} кН")
    result = find_equilibrium(base_config, W_ext=W_from_direct, verbose=False)

    print(f"  Найдено: ε = {result.epsilon:.4f} (было: {test_epsilon})")
    print(f"  φ₀ = {np.degrees(result.phi0):.1f}° (было: 180°)")
    print(f"  Невязка: {result.residual*100:.4f}%")

    eps_error = abs(result.epsilon - test_epsilon)
    if eps_error < 0.01:
        print(f"  [OK] Ошибка по ε: {eps_error:.4f} < 0.01")
    else:
        print(f"  [!] Ошибка по ε: {eps_error:.4f} >= 0.01")


def run_1d_comparison():
    """Сравнение 2D и 1D методов."""
    print_header("4. СРАВНЕНИЕ 2D И 1D МЕТОДОВ")

    base_config = create_base_config()
    W_ext = 50e3  # 50 кН

    print(f"\nНагрузка: W_ext = 50 кН")

    # 2D метод
    t_start = time.perf_counter()
    result_2d = find_equilibrium(base_config, W_ext=W_ext, verbose=False)
    t_2d = time.perf_counter() - t_start

    print(f"\n2D метод (hybr):")
    print(f"  ε = {result_2d.epsilon:.4f}, φ₀ = {np.degrees(result_2d.phi0):.1f}°")
    print(f"  Невязка: {result_2d.residual*100:.4f}%, итераций: {result_2d.iterations}")
    print(f"  Время: {t_2d:.3f}с")

    # 1D метод
    t_start = time.perf_counter()
    result_1d = find_equilibrium_1d(base_config, W_ext=W_ext, verbose=False)
    t_1d = time.perf_counter() - t_start

    print(f"\n1D метод (Brent):")
    print(f"  ε = {result_1d.epsilon:.4f}, φ₀ = {np.degrees(result_1d.phi0):.1f}°")
    print(f"  Невязка: {result_1d.residual*100:.4f}%, итераций: {result_1d.iterations}")
    print(f"  Время: {t_1d:.3f}с")


def create_plots(results: list):
    """Создать графики."""
    print_header("5. СОЗДАНИЕ ГРАФИКОВ")

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        W_ext = [r['W_ext_kN'] for r in results]
        eps = [r['epsilon'] for r in results]
        h_min = [r['h_min_um'] for r in results]
        p_max = [r['p_max_MPa'] for r in results]
        mu_fr = [r['mu_friction'] for r in results]

        # ε vs W_ext
        axes[0, 0].plot(W_ext, eps, 'bo-', lw=2, markersize=8)
        axes[0, 0].set_xlabel('W_ext, кН')
        axes[0, 0].set_ylabel('ε')
        axes[0, 0].set_title('Эксцентриситет равновесия')
        axes[0, 0].grid(True, alpha=0.3)

        # h_min vs W_ext
        axes[0, 1].plot(W_ext, h_min, 'go-', lw=2, markersize=8)
        axes[0, 1].set_xlabel('W_ext, кН')
        axes[0, 1].set_ylabel('h_min, мкм')
        axes[0, 1].set_title('Минимальная толщина плёнки')
        axes[0, 1].grid(True, alpha=0.3)

        # p_max vs W_ext
        axes[1, 0].plot(W_ext, p_max, 'ro-', lw=2, markersize=8)
        axes[1, 0].set_xlabel('W_ext, кН')
        axes[1, 0].set_ylabel('p_max, МПа')
        axes[1, 0].set_title('Максимальное давление')
        axes[1, 0].grid(True, alpha=0.3)

        # μ_fr vs W_ext
        axes[1, 1].plot(W_ext, mu_fr, 'mo-', lw=2, markersize=8)
        axes[1, 1].set_xlabel('W_ext, кН')
        axes[1, 1].set_ylabel('μ_fr')
        axes[1, 1].set_title('Коэффициент трения')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

        fig.suptitle('Этап 3: Зависимости параметров от внешней нагрузки', fontsize=14, y=1.02)
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / 'equilibrium_study.png', dpi=150, bbox_inches='tight')
        print(f"  Сохранено: equilibrium_study.png")
        plt.close()

    except ImportError as e:
        print(f"  [!] matplotlib не установлен: {e}")


def main():
    print_header("ЭТАП 3: ПОИСК ПОЛОЖЕНИЯ РАВНОВЕСИЯ ВАЛА")

    # 1. Серия тестов
    results = run_load_series()

    # 2. Сводная таблица
    run_summary_table(results)

    # 3. Валидация
    run_validation()

    # 4. Сравнение методов
    run_1d_comparison()

    # 5. Графики
    create_plots(results)

    # Итоги
    print_header("ИТОГИ ЭТАПА 3")

    all_converged = all(r['converged'] for r in results)
    all_accurate = all(r['residual_pct'] < 1 for r in results)

    print(f"""
Этап 3 завершён:

1. Реализован поиск равновесия:
   - 2D метод: scipy.optimize.root (hybr)
   - 1D метод: scipy.optimize.brentq

2. Тесты на нагрузках 10-100 кН:
   - Сходимость: {"✓ все сошлись" if all_converged else "! не все сошлись"}
   - Точность: {"✓ все < 1%" if all_accurate else "! не все < 1%"}

3. Результаты сохранены в: {RESULTS_DIR}

Этап 3 завершён {"успешно" if all_converged and all_accurate else "с замечаниями"}.
""")


if __name__ == "__main__":
    main()
