#!/usr/bin/env python3
"""
ЭТАП 3: Поиск положения равновесия вала.

Задача: по заданной внешней нагрузке W_ext найти (ε, φ₀).

ВЕКТОРНАЯ ПОСТАНОВКА:
    Fx(ε, φ₀) = Wx_ext
    Fy(ε, φ₀) = Wy_ext

Переменные оптимизации: (ex, ey) = (ε·cos(φ₀), ε·sin(φ₀))

Особенности этого подшипника (c=50мкм, L/D=1.5, n=2980):
    - Высокая грузоподъёмность: ~100-150 кН при ε=0.5-0.6
    - При нагрузках 10-30 кН: ε ~ 0.08-0.22 (низкие!)
    - При нагрузках 50-100 кН: ε ~ 0.34-0.55
    - μ_fr ~ 0.003-0.03 (выше при малых нагрузках - нормально)

Тесты:
    1. Векторная невязка < 1%
    2. Сходимость за < 30 итераций
    3. Валидация: прямой → обратный расчёт
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
    """Создать базовую конфигурацию."""
    return BearingConfig(
        R=0.0345,       # 34.5 мм
        L=0.1035,       # 103.5 мм
        c=50e-6,        # 50 мкм
        epsilon=0.5,    # placeholder
        phi0=0.0,       # placeholder
        n_rpm=2980,
        mu=0.057,       # VG100 при 50°C
        n_phi=90,       # для быстрого поиска (уточнение на 180×50)
        n_z=25,
    )


def check_mu_friction(mu_fr: float) -> str:
    """Проверка коэффициента трения."""
    if mu_fr < 0.01:
        return "[OK]"
    elif mu_fr < 0.05:
        return "[WARN]"  # при малых нагрузках это нормально
    else:
        return "[FAIL]"


def test_single_load(base_config: BearingConfig, W_ext_kN: float, verbose: bool = False):
    """Тест для одной нагрузки."""
    W_ext = W_ext_kN * 1000  # Н

    print(f"\n--- W_ext = {W_ext_kN:.0f} кН (вниз) ---")

    t_start = time.perf_counter()
    result = find_equilibrium(
        base_config,
        W_ext=W_ext,
        load_angle=-np.pi/2,  # вертикально вниз
        verbose=verbose
    )
    t_elapsed = time.perf_counter() - t_start

    # Вывод
    print(f"  ε = {result.epsilon:.4f}, φ₀ = {np.degrees(result.phi0):.1f}°")
    print(f"  (ex, ey) = ({result.ex:.4f}, {result.ey:.4f})")
    print(f"  W = {result.W_achieved/1000:.3f} кН (цель: {W_ext_kN:.0f})")
    print(f"  Fx = {result.Fx_achieved/1000:.3f} кН (цель: {result.Wx_ext/1000:.3f})")
    print(f"  Fy = {result.Fy_achieved/1000:.3f} кН (цель: {result.Wy_ext/1000:.3f})")
    print(f"  Невязка: |W|={result.residual_W*100:.3f}%, вект={result.residual_vec*100:.3f}%")
    print(f"  Итераций: {result.iterations}, время: {t_elapsed:.2f}с")

    if result.stage2:
        s2 = result.stage2
        mu_status = check_mu_friction(s2.friction.mu_friction)
        print(f"  h_min = {s2.reynolds.h_min*1e6:.2f} мкм")
        print(f"  p_max = {s2.reynolds.p_max/1e6:.2f} МПа")
        print(f"  μ_fr = {s2.friction.mu_friction:.6f} {mu_status}")

    # Проверки
    if result.converged:
        print(f"  [OK] Сходимость (вект. невязка < 1%)")
    else:
        print(f"  [!] Не сошлось: вект. невязка = {result.residual_vec*100:.2f}%")

    return result


def run_load_series():
    """Серия тестов для разных нагрузок."""
    print_header("1. СЕРИЯ ТЕСТОВ (ВЕКТОРНОЕ РАВНОВЕСИЕ)")

    base_config = create_base_config()

    print(f"\nПараметры подшипника:")
    print(f"  R = {base_config.R*1000:.1f} мм, L = {base_config.L*1000:.1f} мм")
    print(f"  c = {base_config.c*1e6:.1f} мкм")
    print(f"  n = {base_config.n_rpm} об/мин, μ = {base_config.mu} Па·с")
    print(f"  Поиск: грубая сетка 90×25, финал 180×50")

    loads_kN = [10, 20, 30, 50, 70, 100]
    results = []

    for W_kN in loads_kN:
        result = test_single_load(base_config, W_kN)
        results.append({
            'W_ext_kN': W_kN,
            'epsilon': result.epsilon,
            'phi0_deg': np.degrees(result.phi0),
            'ex': result.ex,
            'ey': result.ey,
            'W_achieved_kN': result.W_achieved / 1000,
            'Fx_kN': result.Fx_achieved / 1000,
            'Fy_kN': result.Fy_achieved / 1000,
            'residual_W_pct': result.residual_W * 100,
            'residual_vec_pct': result.residual_vec * 100,
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

    print(f"\n{'W_ext':>6} {'ε':>7} {'φ₀':>7} {'ex':>7} {'ey':>7} "
          f"{'δ_W%':>6} {'δ_vec%':>7} {'iter':>5} {'h_min':>6} {'p_max':>6} {'μ_fr':>8}")
    print("-" * 95)

    for r in results:
        status = "✓" if r['converged'] else "!"
        mu_st = check_mu_friction(r['mu_friction'])[1:-1]  # убираем скобки
        print(f"{r['W_ext_kN']:>6.0f} {r['epsilon']:>7.4f} {r['phi0_deg']:>7.1f} "
              f"{r['ex']:>7.4f} {r['ey']:>7.4f} "
              f"{r['residual_W_pct']:>6.2f} {r['residual_vec_pct']:>7.3f} "
              f"{r['iterations']:>5d} {r['h_min_um']:>6.1f} {r['p_max_MPa']:>6.1f} "
              f"{r['mu_friction']:>8.5f} {status}")

    # Сохраняем CSV
    df = pd.DataFrame(results)
    df.to_csv(RESULTS_DIR / "equilibrium_results.csv", index=False)
    print(f"\nСохранено: {RESULTS_DIR / 'equilibrium_results.csv'}")


def run_validation():
    """Валидация: прямой → обратный расчёт."""
    print_header("3. ВАЛИДАЦИЯ: ПРЯМОЙ → ОБРАТНЫЙ")

    base_config = create_base_config()

    # Прямой расчёт с известными (ε, φ₀)
    test_epsilon = 0.6
    test_phi0 = np.radians(315)  # 315° = -45°

    config_test = BearingConfig(
        R=base_config.R, L=base_config.L, c=base_config.c,
        epsilon=test_epsilon, phi0=test_phi0,
        n_rpm=base_config.n_rpm, mu=base_config.mu,
        n_phi=180, n_z=50
    )

    from bearing_solver import solve_reynolds, compute_stage2
    reynolds = solve_reynolds(config_test)
    stage2 = compute_stage2(reynolds, config_test)

    W_direct = stage2.forces.W
    Fx_direct = stage2.forces.Fx
    Fy_direct = stage2.forces.Fy
    angle_direct = np.degrees(stage2.forces.force_angle)

    print(f"\nПрямой расчёт:")
    print(f"  ε = {test_epsilon}, φ₀ = {np.degrees(test_phi0):.1f}°")
    print(f"  W = {W_direct/1000:.3f} кН")
    print(f"  Fx = {Fx_direct/1000:.3f} кН, Fy = {Fy_direct/1000:.3f} кН")
    print(f"  Угол силы = {angle_direct:.1f}°")

    # Обратный расчёт: ищем (ε, φ₀) для этой нагрузки
    # Направление нагрузки = направление силы (чтобы уравновесить)
    load_angle = stage2.forces.force_angle

    print(f"\nОбратный расчёт: ищем (ε, φ₀) для W={W_direct/1000:.3f} кН")
    result = find_equilibrium(
        base_config,
        W_ext=W_direct,
        load_angle=load_angle,
        verbose=False
    )

    print(f"  Найдено: ε = {result.epsilon:.4f} (было: {test_epsilon})")
    print(f"  Найдено: φ₀ = {np.degrees(result.phi0):.1f}° (было: {np.degrees(test_phi0):.1f}°)")
    print(f"  Невязка |W|: {result.residual_W*100:.4f}%")
    print(f"  Невязка вект: {result.residual_vec*100:.4f}%")

    eps_error = abs(result.epsilon - test_epsilon)
    # φ₀ может отличаться на 360°, нормализуем
    phi0_error = abs(result.phi0 - test_phi0)
    if phi0_error > np.pi:
        phi0_error = 2*np.pi - phi0_error

    if eps_error < 0.02:
        print(f"  [OK] Ошибка по ε: {eps_error:.4f} < 0.02")
    else:
        print(f"  [!] Ошибка по ε: {eps_error:.4f} >= 0.02")

    if phi0_error < np.radians(10):
        print(f"  [OK] Ошибка по φ₀: {np.degrees(phi0_error):.1f}° < 10°")
    else:
        print(f"  [!] Ошибка по φ₀: {np.degrees(phi0_error):.1f}° >= 10°")


def run_method_comparison():
    """Сравнение 2D (векторного) и 1D методов."""
    print_header("4. СРАВНЕНИЕ МЕТОДОВ: ВЕКТОРНЫЙ vs 1D")

    base_config = create_base_config()
    W_ext = 50e3  # 50 кН

    print(f"\nНагрузка: W_ext = 50 кН (вниз)")

    # Векторный 2D метод
    t_start = time.perf_counter()
    result_2d = find_equilibrium(base_config, W_ext=W_ext, verbose=False)
    t_2d = time.perf_counter() - t_start

    print(f"\nВекторный 2D метод:")
    print(f"  ε = {result_2d.epsilon:.4f}, φ₀ = {np.degrees(result_2d.phi0):.1f}°")
    print(f"  Fx = {result_2d.Fx_achieved/1000:.3f} кН (цель: 0)")
    print(f"  Fy = {result_2d.Fy_achieved/1000:.3f} кН (цель: -50)")
    print(f"  Невязка вект: {result_2d.residual_vec*100:.4f}%")
    print(f"  Итераций: {result_2d.iterations}, время: {t_2d:.3f}с")

    # 1D метод (только для гладкого подшипника)
    t_start = time.perf_counter()
    result_1d = find_equilibrium_1d(base_config, W_ext=W_ext, verbose=False)
    t_1d = time.perf_counter() - t_start

    print(f"\n1D метод (Brent):")
    print(f"  ε = {result_1d.epsilon:.4f}, φ₀ = {np.degrees(result_1d.phi0):.1f}°")
    print(f"  Fx = {result_1d.Fx_achieved/1000:.3f} кН (цель: 0)")
    print(f"  Fy = {result_1d.Fy_achieved/1000:.3f} кН (цель: -50)")
    print(f"  Невязка вект: {result_1d.residual_vec*100:.4f}%")
    print(f"  Итераций: {result_1d.iterations}, время: {t_1d:.3f}с")

    # Вывод: какой метод лучше по векторной невязке
    if result_2d.residual_vec < result_1d.residual_vec:
        print(f"\n  → Векторный 2D точнее (невязка {result_2d.residual_vec*100:.3f}% vs {result_1d.residual_vec*100:.3f}%)")
    else:
        print(f"\n  → 1D достаточно для гладкого подшипника")


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
        residual = [r['residual_vec_pct'] for r in results]

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

        # Невязка vs W_ext
        axes[1, 1].semilogy(W_ext, residual, 'mo-', lw=2, markersize=8)
        axes[1, 1].axhline(1.0, color='r', linestyle='--', label='1% порог')
        axes[1, 1].set_xlabel('W_ext, кН')
        axes[1, 1].set_ylabel('Вект. невязка, %')
        axes[1, 1].set_title('Точность сходимости')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].legend()

        fig.suptitle('Этап 3: Равновесие (векторная постановка)', fontsize=14, y=1.02)
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / 'equilibrium_study.png', dpi=150, bbox_inches='tight')
        print(f"  Сохранено: equilibrium_study.png")
        plt.close()

    except ImportError as e:
        print(f"  [!] matplotlib не установлен: {e}")


def main():
    print_header("ЭТАП 3: ПОИСК ПОЛОЖЕНИЯ РАВНОВЕСИЯ (ВЕКТОРНЫЙ)")

    # 1. Серия тестов
    results = run_load_series()

    # 2. Сводная таблица
    run_summary_table(results)

    # 3. Валидация
    run_validation()

    # 4. Сравнение методов
    run_method_comparison()

    # 5. Графики
    create_plots(results)

    # Итоги
    print_header("ИТОГИ ЭТАПА 3")

    all_converged = all(r['converged'] for r in results)
    max_residual = max(r['residual_vec_pct'] for r in results)

    print(f"""
Этап 3 завершён:

1. Векторная постановка равновесия:
   - Переменные: (ex, ey) = (ε·cos(φ₀), ε·sin(φ₀))
   - Уравнения: Fx = Wx_ext, Fy = Wy_ext
   - Готов к асимметричным случаям (этапы 6-7)

2. Двухуровневая точность:
   - Поиск: грубая сетка 90×25
   - Финал: тонкая сетка 180×50

3. Тесты на нагрузках 10-100 кН:
   - Сходимость: {"✓ все сошлись" if all_converged else "! не все"}
   - Макс. вект. невязка: {max_residual:.3f}%

4. Результаты: {RESULTS_DIR}

Этап 3 завершён {"успешно" if all_converged else "с замечаниями"}.
""")


if __name__ == "__main__":
    main()
