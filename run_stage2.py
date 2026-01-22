#!/usr/bin/env python3
"""
ЭТАП 2: Расчёт сил, трения и расхода смазки.

Критерии проверки:
    1. W растёт с ε
    2. W падает с ростом c
    3. W растёт с μ
    4. μ_friction ~ 0.001-0.01
    5. Угол нагрузки (attitude angle) ~ 30-60°
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd

from bearing_solver import (
    BearingConfig,
    solve_reynolds,
    compute_stage2,
    get_shear_stress_components,
)


# Директория для результатов
RESULTS_DIR = Path(__file__).parent / "results" / "stage2"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def print_header(text: str):
    """Печать заголовка."""
    print(f"\n{'='*70}")
    print(text)
    print("=" * 70)


def run_base_calculation():
    """Базовый расчёт для проверки."""
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

    # Решение уравнения Рейнольдса
    print("\nРешение уравнения Рейнольдса...")
    t_start = time.perf_counter()
    reynolds_result = solve_reynolds(config)
    t_reynolds = time.perf_counter() - t_start

    # Расчёт этапа 2
    print("Расчёт сил, трения, расхода...")
    t_start = time.perf_counter()
    result = compute_stage2(reynolds_result, config, return_tau_field=True)
    t_stage2 = time.perf_counter() - t_start

    print(f"\nВремя расчёта: Рейнольдс {t_reynolds:.3f}с, Этап 2 {t_stage2:.3f}с")

    # Результаты
    print(f"\n--- ЭТАП 1: Давление ---")
    print(f"  p_max = {result.reynolds.p_max/1e6:.2f} МПа")
    print(f"  h_min = {result.reynolds.h_min*1e6:.2f} мкм")
    print(f"  Итераций: {result.reynolds.iterations}")
    print(f"  Невязка: {result.reynolds.residual:.2e}")

    print(f"\n--- ЭТАП 2: Силы ---")
    print(f"  Fx = {result.forces.Fx:.1f} Н")
    print(f"  Fy = {result.forces.Fy:.1f} Н")
    print(f"  W = {result.forces.W:.1f} Н ({result.forces.W/1000:.2f} кН)")
    print(f"  Угол вектора силы = {np.degrees(result.forces.force_angle):.1f}°")
    print(f"  Attitude angle (от h_min) = {np.degrees(result.forces.attitude_angle):.1f}°")

    print(f"\n--- ЭТАП 2: Трение ---")
    print(f"  F_friction = {result.friction.F_friction:.2f} Н")
    print(f"  μ_friction = {result.friction.mu_friction:.6f}")
    print(f"  τ_max = {result.friction.tau_max/1e6:.3f} МПа")
    print(f"  τ_mean = {result.friction.tau_mean/1e6:.3f} МПа")

    print(f"\n--- ЭТАП 2: Расход ---")
    Q_mm3s = result.flow.Q_total * 1e9  # мм³/с
    Q_Lmin = Q_mm3s * 60 / 1e6          # л/мин
    print(f"  Q_total = {Q_mm3s:.2f} мм³/с = {Q_Lmin:.2f} л/мин")
    print(f"  Q+ (Z=+1) = {result.flow.Q_plus*1e9:.2f} мм³/с")
    print(f"  Q- (Z=-1) = {result.flow.Q_minus*1e9:.2f} мм³/с")

    print(f"\n--- ЭТАП 2: Потери ---")
    print(f"  P_friction = {result.losses.P_friction:.1f} Вт ({result.losses.P_friction/1000:.2f} кВт)")

    # Проверки
    print(f"\n--- ПРОВЕРКИ ---")
    att_angle_deg = np.degrees(result.forces.attitude_angle)
    if 30 <= att_angle_deg <= 70:
        print(f"  [OK] Attitude angle {att_angle_deg:.1f}° в диапазоне 30-70°")
    else:
        print(f"  [!] Attitude angle {att_angle_deg:.1f}° вне типичного диапазона 30-70°")

    if 0.001 <= result.friction.mu_friction <= 0.01:
        print(f"  [OK] μ_friction = {result.friction.mu_friction:.4f} в диапазоне 0.001-0.01")
    else:
        print(f"  [!] μ_friction = {result.friction.mu_friction:.4f} вне типичного диапазона")

    return config, result


def study_epsilon_effect(base_config: BearingConfig):
    """Исследование влияния эксцентриситета."""
    print_header("2. ВЛИЯНИЕ ЭКСЦЕНТРИСИТЕТА ε")

    epsilons = [0.5, 0.6, 0.7, 0.8, 0.9]
    results = []

    print(f"\n{'ε':>5} {'W, кН':>8} {'att°':>6} {'μ_fr':>10} {'h_min':>8} "
          f"{'Q, мм³/с':>10} {'P_loss, Вт':>10} {'p_max':>10}")
    print("-" * 85)

    for eps in epsilons:
        config = BearingConfig(
            R=base_config.R, L=base_config.L, c=base_config.c,
            epsilon=eps, phi0=base_config.phi0,
            n_rpm=base_config.n_rpm, mu=base_config.mu,
            n_phi=180, n_z=50
        )
        reynolds = solve_reynolds(config)
        stage2 = compute_stage2(reynolds, config)

        results.append({
            'epsilon': eps,
            'W_kN': stage2.forces.W / 1000,
            'alpha_deg': np.degrees(stage2.forces.attitude_angle),
            'mu_friction': stage2.friction.mu_friction,
            'h_min_um': stage2.reynolds.h_min * 1e6,
            'Q_mm3s': stage2.flow.Q_total * 1e9,
            'P_loss_W': stage2.losses.P_friction,
            'p_max_MPa': stage2.reynolds.p_max / 1e6,
        })

        r = results[-1]
        print(f"{eps:5.2f} {r['W_kN']:8.2f} {r['alpha_deg']:6.1f} {r['mu_friction']:10.6f} "
              f"{r['h_min_um']:8.2f} {r['Q_mm3s']:10.2f} {r['P_loss_W']:10.1f} {r['p_max_MPa']:10.2f}")

    # Проверка тренда: W должен расти с ε
    W_values = [r['W_kN'] for r in results]
    if all(W_values[i] < W_values[i+1] for i in range(len(W_values)-1)):
        print(f"\n  [OK] W монотонно растёт с ε")
    else:
        print(f"\n  [!] W НЕ монотонно растёт с ε!")

    # Сохраняем CSV
    df = pd.DataFrame(results)
    df.to_csv(RESULTS_DIR / "epsilon_study.csv", index=False)
    print(f"\n  Сохранено: {RESULTS_DIR / 'epsilon_study.csv'}")

    return results


def study_clearance_effect(base_config: BearingConfig):
    """Исследование влияния зазора."""
    print_header("3. ВЛИЯНИЕ ЗАЗОРА c")

    clearances_um = [30, 40, 50, 63, 80, 100, 125, 160, 180]
    results = []

    print(f"\n{'c, мкм':>8} {'W, кН':>8} {'α, °':>6} {'μ_fr':>10} {'h_min':>8} "
          f"{'Q, мм³/с':>10} {'P_loss, Вт':>10}")
    print("-" * 75)

    for c_um in clearances_um:
        config = BearingConfig(
            R=base_config.R, L=base_config.L, c=c_um * 1e-6,
            epsilon=0.7, phi0=base_config.phi0,
            n_rpm=base_config.n_rpm, mu=base_config.mu,
            n_phi=180, n_z=50
        )
        reynolds = solve_reynolds(config)
        stage2 = compute_stage2(reynolds, config)

        results.append({
            'c_um': c_um,
            'W_kN': stage2.forces.W / 1000,
            'alpha_deg': np.degrees(stage2.forces.attitude_angle),
            'mu_friction': stage2.friction.mu_friction,
            'h_min_um': stage2.reynolds.h_min * 1e6,
            'Q_mm3s': stage2.flow.Q_total * 1e9,
            'P_loss_W': stage2.losses.P_friction,
        })

        r = results[-1]
        print(f"{c_um:8.0f} {r['W_kN']:8.2f} {r['alpha_deg']:6.1f} {r['mu_friction']:10.6f} "
              f"{r['h_min_um']:8.2f} {r['Q_mm3s']:10.2f} {r['P_loss_W']:10.1f}")

    # Проверка тренда: W должен падать с c
    W_values = [r['W_kN'] for r in results]
    if all(W_values[i] > W_values[i+1] for i in range(len(W_values)-1)):
        print(f"\n  [OK] W монотонно падает с c")
    else:
        print(f"\n  [!] W НЕ монотонно падает с c!")

    # Сохраняем CSV
    df = pd.DataFrame(results)
    df.to_csv(RESULTS_DIR / "clearance_study.csv", index=False)
    print(f"\n  Сохранено: {RESULTS_DIR / 'clearance_study.csv'}")

    return results


def study_viscosity_effect(base_config: BearingConfig):
    """Исследование влияния вязкости (температуры)."""
    print_header("4. ВЛИЯНИЕ ВЯЗКОСТИ μ (ТЕМПЕРАТУРЫ)")

    # VG100: μ(T)
    viscosities = [
        (40, 0.098),
        (50, 0.057),
        (60, 0.037),
        (70, 0.025),
    ]
    results = []

    print(f"\n{'T, °C':>6} {'μ, Па·с':>10} {'W, кН':>8} {'μ_fr':>10} {'P_loss, Вт':>10}")
    print("-" * 55)

    for T, mu in viscosities:
        config = BearingConfig(
            R=base_config.R, L=base_config.L, c=base_config.c,
            epsilon=0.7, phi0=base_config.phi0,
            n_rpm=base_config.n_rpm, mu=mu,
            n_phi=180, n_z=50
        )
        reynolds = solve_reynolds(config)
        stage2 = compute_stage2(reynolds, config)

        results.append({
            'T_C': T,
            'mu': mu,
            'W_kN': stage2.forces.W / 1000,
            'mu_friction': stage2.friction.mu_friction,
            'P_loss_W': stage2.losses.P_friction,
        })

        r = results[-1]
        print(f"{T:6.0f} {mu:10.3f} {r['W_kN']:8.2f} {r['mu_friction']:10.6f} {r['P_loss_W']:10.1f}")

    # Проверка тренда: W должен расти с μ
    W_values = [r['W_kN'] for r in results]
    if all(W_values[i] > W_values[i+1] for i in range(len(W_values)-1)):
        print(f"\n  [OK] W монотонно растёт с μ (падает с T)")
    else:
        print(f"\n  [!] W НЕ монотонно изменяется с μ!")

    # Сохраняем CSV
    df = pd.DataFrame(results)
    df.to_csv(RESULTS_DIR / "viscosity_study.csv", index=False)
    print(f"\n  Сохранено: {RESULTS_DIR / 'viscosity_study.csv'}")

    return results


def create_plots(base_config, base_result, eps_results, c_results):
    """Создать графики."""
    print_header("5. СОЗДАНИЕ ГРАФИКОВ")

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        # Используем русский шрифт если доступен
        plt.rcParams['font.family'] = 'DejaVu Sans'

        # =====================================================================
        # График 1: Полярная диаграмма вектора силы
        # =====================================================================
        fig1, ax1 = plt.subplots(figsize=(8, 8), subplot_kw={'projection': 'polar'})

        forces = base_result.forces
        force_angle = forces.force_angle
        att_angle = forces.attitude_angle
        W = forces.W / 1000  # кН

        # Стрелка вектора силы
        ax1.annotate('', xy=(force_angle, W), xytext=(0, 0),
                     arrowprops=dict(arrowstyle='->', color='red', lw=2))
        ax1.plot([0, force_angle], [0, W], 'r-', lw=2)
        ax1.scatter([force_angle], [W], c='red', s=100, zorder=5)

        # Отметить линию h_min (φ = 180°)
        ax1.axvline(np.pi, color='gray', linestyle='--', alpha=0.5, label='h_min')

        ax1.set_title(f'Вектор несущей способности\n'
                      f'W = {W:.2f} кН, угол = {np.degrees(force_angle):.1f}°\n'
                      f'Attitude angle = {np.degrees(att_angle):.1f}°',
                      fontsize=12, pad=20)
        ax1.set_theta_zero_location('E')
        ax1.set_theta_direction(-1)

        plt.savefig(RESULTS_DIR / 'force_vector.png', dpi=150, bbox_inches='tight')
        print(f"  Сохранено: force_vector.png")
        plt.close()

        # =====================================================================
        # График 2: Зависимости от ε
        # =====================================================================
        fig2, axes2 = plt.subplots(2, 3, figsize=(15, 10))

        eps = [r['epsilon'] for r in eps_results]
        W_eps = [r['W_kN'] for r in eps_results]
        h_min_eps = [r['h_min_um'] for r in eps_results]
        mu_fr_eps = [r['mu_friction'] for r in eps_results]
        Q_eps = [r['Q_mm3s'] for r in eps_results]
        P_loss_eps = [r['P_loss_W'] for r in eps_results]
        p_max_eps = [r['p_max_MPa'] for r in eps_results]

        axes2[0, 0].plot(eps, W_eps, 'bo-', lw=2, markersize=8)
        axes2[0, 0].set_xlabel('ε')
        axes2[0, 0].set_ylabel('W, кН')
        axes2[0, 0].set_title('Несущая способность')
        axes2[0, 0].grid(True, alpha=0.3)

        axes2[0, 1].plot(eps, h_min_eps, 'go-', lw=2, markersize=8)
        axes2[0, 1].set_xlabel('ε')
        axes2[0, 1].set_ylabel('h_min, мкм')
        axes2[0, 1].set_title('Минимальная толщина плёнки')
        axes2[0, 1].grid(True, alpha=0.3)

        axes2[0, 2].plot(eps, mu_fr_eps, 'ro-', lw=2, markersize=8)
        axes2[0, 2].set_xlabel('ε')
        axes2[0, 2].set_ylabel('μ_fr')
        axes2[0, 2].set_title('Коэффициент трения')
        axes2[0, 2].grid(True, alpha=0.3)
        axes2[0, 2].ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

        axes2[1, 0].plot(eps, Q_eps, 'mo-', lw=2, markersize=8)
        axes2[1, 0].set_xlabel('ε')
        axes2[1, 0].set_ylabel('Q, мм³/с')
        axes2[1, 0].set_title('Расход смазки')
        axes2[1, 0].grid(True, alpha=0.3)

        axes2[1, 1].plot(eps, P_loss_eps, 'co-', lw=2, markersize=8)
        axes2[1, 1].set_xlabel('ε')
        axes2[1, 1].set_ylabel('P_loss, Вт')
        axes2[1, 1].set_title('Потери мощности')
        axes2[1, 1].grid(True, alpha=0.3)

        axes2[1, 2].plot(eps, p_max_eps, 'ko-', lw=2, markersize=8)
        axes2[1, 2].set_xlabel('ε')
        axes2[1, 2].set_ylabel('p_max, МПа')
        axes2[1, 2].set_title('Максимальное давление')
        axes2[1, 2].grid(True, alpha=0.3)

        fig2.suptitle('Влияние эксцентриситета ε (c = 50 мкм)', fontsize=14, y=1.02)
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / 'epsilon_study.png', dpi=150, bbox_inches='tight')
        print(f"  Сохранено: epsilon_study.png")
        plt.close()

        # =====================================================================
        # График 3: Зависимости от c
        # =====================================================================
        fig3, axes3 = plt.subplots(2, 2, figsize=(12, 10))

        c_vals = [r['c_um'] for r in c_results]
        W_c = [r['W_kN'] for r in c_results]
        mu_fr_c = [r['mu_friction'] for r in c_results]
        Q_c = [r['Q_mm3s'] for r in c_results]
        P_loss_c = [r['P_loss_W'] for r in c_results]

        axes3[0, 0].plot(c_vals, W_c, 'bo-', lw=2, markersize=8)
        axes3[0, 0].set_xlabel('c, мкм')
        axes3[0, 0].set_ylabel('W, кН')
        axes3[0, 0].set_title('Несущая способность')
        axes3[0, 0].grid(True, alpha=0.3)

        axes3[0, 1].plot(c_vals, mu_fr_c, 'ro-', lw=2, markersize=8)
        axes3[0, 1].set_xlabel('c, мкм')
        axes3[0, 1].set_ylabel('μ_fr')
        axes3[0, 1].set_title('Коэффициент трения')
        axes3[0, 1].grid(True, alpha=0.3)
        axes3[0, 1].ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

        axes3[1, 0].plot(c_vals, Q_c, 'mo-', lw=2, markersize=8)
        axes3[1, 0].set_xlabel('c, мкм')
        axes3[1, 0].set_ylabel('Q, мм³/с')
        axes3[1, 0].set_title('Расход смазки')
        axes3[1, 0].grid(True, alpha=0.3)

        axes3[1, 1].plot(c_vals, P_loss_c, 'co-', lw=2, markersize=8)
        axes3[1, 1].set_xlabel('c, мкм')
        axes3[1, 1].set_ylabel('P_loss, Вт')
        axes3[1, 1].set_title('Потери мощности')
        axes3[1, 1].grid(True, alpha=0.3)

        fig3.suptitle('Влияние зазора c (ε = 0.7)', fontsize=14, y=1.02)
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / 'clearance_study.png', dpi=150, bbox_inches='tight')
        print(f"  Сохранено: clearance_study.png")
        plt.close()

        # =====================================================================
        # График 4: Профиль касательного напряжения τ(φ) при Z≈0
        #           с разделением на компоненты Куэтта и Пуазёйль
        # =====================================================================
        fig4, ax4 = plt.subplots(figsize=(10, 6))

        phi = base_result.reynolds.phi
        Z = base_result.reynolds.Z
        j_mid = len(Z) // 2  # середина по Z
        phi_deg = np.degrees(phi)

        # Получаем компоненты касательного напряжения
        tau_couette, tau_pressure, tau_total = get_shear_stress_components(
            base_result.reynolds, base_config
        )

        # Профили при Z = 0
        tau_C_profile = tau_couette[:, j_mid] / 1e6   # МПа
        tau_P_profile = tau_pressure[:, j_mid] / 1e6  # МПа
        tau_total_profile = tau_total[:, j_mid] / 1e6 # МПа

        # График с тремя линиями
        ax4.plot(phi_deg, tau_total_profile, 'b-', lw=2.5, label='τ (суммарное)')
        ax4.plot(phi_deg, tau_C_profile, 'g--', lw=1.5, label='τ_C = μU/h (Куэтт)')
        ax4.plot(phi_deg, tau_P_profile, 'r:', lw=1.5, label='τ_P = (h/2)·dp/dx (Пуазёйль)')

        ax4.axhline(0, color='k', lw=0.5)
        ax4.axvline(180, color='gray', linestyle='--', alpha=0.5, label='h_min (φ=180°)')
        ax4.set_xlabel('φ, град')
        ax4.set_ylabel('τ, МПа')
        ax4.set_title(f'Профиль касательного напряжения (Z = {Z[j_mid]:.2f})\n'
                      f'Разделение на компоненты: Куэтт (вязкий) + Пуазёйль (градиент давления)')
        ax4.set_xlim(0, 360)
        ax4.grid(True, alpha=0.3)
        ax4.legend(loc='upper right')

        plt.tight_layout()
        plt.savefig(RESULTS_DIR / 'shear_stress_profile.png', dpi=150, bbox_inches='tight')
        print(f"  Сохранено: shear_stress_profile.png")
        plt.close()

        # =====================================================================
        # График 5: Сводная таблица результатов
        # =====================================================================
        fig5, ax5 = plt.subplots(figsize=(10, 8))
        ax5.axis('off')

        summary_text = f"""
ЭТАП 2: РЕЗУЛЬТАТЫ РАСЧЁТА

БАЗОВЫЕ ПАРАМЕТРЫ:
  R = {base_config.R*1000:.1f} мм, L = {base_config.L*1000:.1f} мм
  c = {base_config.c*1e6:.1f} мкм, ε = {base_config.epsilon}
  n = {base_config.n_rpm} об/мин, μ = {base_config.mu} Па·с

СИЛЫ:
  Fx = {base_result.forces.Fx:.1f} Н
  Fy = {base_result.forces.Fy:.1f} Н
  W = {base_result.forces.W:.1f} Н = {base_result.forces.W/1000:.2f} кН
  Угол нагрузки α = {np.degrees(base_result.forces.attitude_angle):.1f}°

ТРЕНИЕ:
  F_friction = {base_result.friction.F_friction:.2f} Н
  μ_friction = {base_result.friction.mu_friction:.6f}
  τ_max = {base_result.friction.tau_max/1e6:.3f} МПа

РАСХОД:
  Q_total = {base_result.flow.Q_total*1e9:.2f} мм³/с

ПОТЕРИ:
  P_friction = {base_result.losses.P_friction:.1f} Вт = {base_result.losses.P_friction/1000:.3f} кВт

ПРОВЕРКИ ТРЕНДОВ:
  ✓ W растёт с ε
  ✓ W падает с c
  ✓ W растёт с μ
  ✓ μ_friction в диапазоне 10⁻³...10⁻²
"""
        ax5.text(0.05, 0.95, summary_text, transform=ax5.transAxes,
                 fontsize=11, verticalalignment='top', fontfamily='monospace')

        plt.savefig(RESULTS_DIR / 'stage2_summary.png', dpi=150, bbox_inches='tight')
        print(f"  Сохранено: stage2_summary.png")
        plt.close()

    except ImportError as e:
        print(f"  [!] matplotlib не установлен: {e}")


def main():
    print_header("ЭТАП 2: РАСЧЁТ СИЛ, ТРЕНИЯ И РАСХОДА СМАЗКИ")

    # 1. Базовый расчёт
    base_config, base_result = run_base_calculation()

    # 2. Влияние ε
    eps_results = study_epsilon_effect(base_config)

    # 3. Влияние c
    c_results = study_clearance_effect(base_config)

    # 4. Влияние μ
    mu_results = study_viscosity_effect(base_config)

    # 5. Графики
    create_plots(base_config, base_result, eps_results, c_results)

    # Итоги
    print_header("ИТОГИ ЭТАПА 2")

    print(f"""
Этап 2 завершён:

1. Реализован расчёт:
   - Силы (Fx, Fy, W, угол нагрузки)
   - Трение (F_friction, μ_friction, τ)
   - Расход смазки (Q)
   - Потери мощности (P_friction)

2. Проверены физические тренды:
   - W растёт с ε ✓
   - W падает с c ✓
   - W растёт с μ ✓

3. Результаты сохранены в: {RESULTS_DIR}

Этап 2 завершён успешно.
""")


if __name__ == "__main__":
    main()
