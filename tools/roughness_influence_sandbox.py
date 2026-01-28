#!/usr/bin/env python3
"""
T3: Модуль тестирования влияния шероховатости Patir-Cheng.

Цель: Доказать на серии режимов, что Patir-Cheng заметно влияет
при малом lambda (1-3) и почти не влияет при lambda >> 1.

Серия A: Sweep по c и Ra с решателем
Серия B: Контрольный тест модуля roughness без решателя

Выходные файлы:
- results/roughness_sandbox/roughness_influence.csv
- results/roughness_sandbox/roughness_control.csv
- results/roughness_sandbox/dP_loss_vs_lambda_min.png
- results/roughness_sandbox/dp_max_vs_lambda_min.png
- results/roughness_sandbox/phi_x_min_vs_lambda_min.png
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bearing_solver import (
    BearingConfig,
    SmoothFilmModel,
    RoughnessParams,
    solve_reynolds,
    compute_forces,
    compute_stage2,
    compute_roughness_fields,
    find_equilibrium,
    compute_dynamic_coefficients,
    analyze_stability,
)


# ============================================================================
# КОНФИГУРАЦИЯ
# ============================================================================

# Базовые параметры как в проекте
BASE_R = 0.050          # радиус, м
BASE_L = 0.050          # длина, м
BASE_N_RPM = 3000       # об/мин
BASE_W_EXT = 50e3       # Н
BASE_MASS = 30.0        # кг
BASE_MU = 0.057         # вязкость при 50°C

# Sweep по зазору (мкм → м)
C_VALUES_UM = [10, 20, 30, 50]  # микрометры
C_VALUES = [c * 1e-6 for c in C_VALUES_UM]

# Уровни шероховатости
RA_LEVELS = {
    'Low': {'Ra_out': 0.5e-6, 'Ra_shaft': 0.3e-6},
    'Mid': {'Ra_out': 2.0e-6, 'Ra_shaft': 1.0e-6},
    'High': {'Ra_out': 8.0e-6, 'Ra_shaft': 4.0e-6},
}

# Порог эффекта шероховатости
EFFECT_THRESHOLD_PCT = 1.0  # %

OUT_DIR = Path("results/roughness_sandbox")


# ============================================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ============================================================================

def create_config(c: float, mu: float = BASE_MU, n_phi: int = 180, n_z: int = 50) -> BearingConfig:
    """Создать конфигурацию подшипника."""
    return BearingConfig(
        R=BASE_R,
        L=BASE_L,
        c=c,
        epsilon=0.5,  # начальное значение, будет найдено через find_equilibrium
        phi0=np.radians(45),
        n_rpm=BASE_N_RPM,
        mu=mu,
        n_phi=n_phi,
        n_z=n_z,
    )


def solve_smooth(config: BearingConfig) -> Tuple:
    """
    Решить для гладкого подшипника (без Patir-Cheng).

    Returns:
        (eps_eq, h_min_um, p_max_MPa, P_loss_W, Q_cm3_s, f)
    """
    # Находим равновесие
    eq = find_equilibrium(
        config, W_ext=BASE_W_EXT, load_angle=-np.pi/2,
        verbose=False, film_model_factory=lambda cfg: SmoothFilmModel(cfg)
    )

    # Берём результаты
    s2 = eq.stage2
    if s2 is None:
        raise RuntimeError("stage2 is None")

    eps_eq = eq.epsilon
    h_min_um = s2.reynolds.h_min * 1e6
    p_max_MPa = s2.reynolds.p_max / 1e6
    P_loss_W = s2.losses.P_friction
    Q_cm3_s = s2.flow.Q_total * 1e6  # м³/с → см³/с
    f = s2.friction.mu_friction

    return eps_eq, h_min_um, p_max_MPa, P_loss_W, Q_cm3_s, f


def solve_rough(config: BearingConfig, Ra_out: float, Ra_shaft: float) -> Tuple:
    """
    Решить с Patir-Cheng (шероховатость включена).

    Ra_cell = Ra_out (нет текстуры).

    Returns:
        (eps_eq, h_min_um, p_max_MPa, P_loss_W, Q_cm3_s, f,
         lambda_min, lambda_max, phi_x_min, phi_x_max, frac_lambda_lt_1, mean_phi_x,
         Aphi_min, Aphi_max, H3_min, H3_max)
    """
    roughness_params = RoughnessParams(
        Ra_shaft=Ra_shaft,
        Ra_out=Ra_out,
        Ra_cell=Ra_out,  # Ra_cell = Ra_out (без пятнистости)
    )

    def film_model_factory_with_roughness(cfg):
        return SmoothFilmModel(cfg)

    # Вспомогательная функция для расчёта с roughness
    def compute_with_roughness(cfg, film_model):
        """Решить уравнение Рейнольдса с учётом шероховатости."""
        phi, Z, _, _ = cfg.create_grid()
        H = film_model.H(phi, Z)

        # Вычисляем поля шероховатости
        rough_result = compute_roughness_fields(
            H, phi, Z, cfg.c,
            roughness_params, texture_mask=None
        )

        # Решаем с flow factors
        reynolds_result = solve_reynolds(
            cfg, film_model,
            phi_x=rough_result.phi_x,
            phi_z=rough_result.phi_z,
            sigma_star=rough_result.sigma_star,
            lambda_field=rough_result.lambda_field,
            frac_lambda_lt_1=rough_result.frac_lambda_lt_1,
        )

        return reynolds_result, rough_result

    # Создаём собственный film_model_factory, который включает roughness
    # Для find_equilibrium нам нужен factory, который просто создаёт SmoothFilmModel
    # А roughness мы применим в финальном расчёте

    eq = find_equilibrium(
        config, W_ext=BASE_W_EXT, load_angle=-np.pi/2,
        verbose=False, film_model_factory=film_model_factory_with_roughness
    )

    # Теперь делаем финальный расчёт с roughness на найденной позиции
    final_config = BearingConfig(
        R=config.R, L=config.L, c=config.c,
        epsilon=eq.epsilon, phi0=eq.phi0,
        n_rpm=config.n_rpm, mu=config.mu,
        n_phi=config.n_phi, n_z=config.n_z,
    )

    film_model = SmoothFilmModel(final_config)
    reynolds_result, rough_result = compute_with_roughness(final_config, film_model)

    # Вычисляем stage2
    forces = compute_forces(reynolds_result, final_config)
    s2 = compute_stage2(reynolds_result, final_config)

    eps_eq = eq.epsilon
    h_min_um = reynolds_result.h_min * 1e6
    p_max_MPa = reynolds_result.p_max / 1e6
    P_loss_W = s2.losses.P_friction
    Q_cm3_s = s2.flow.Q_total * 1e6
    f = s2.friction.mu_friction

    # Статистика roughness
    lambda_min = rough_result.lambda_min
    lambda_max = rough_result.lambda_max
    phi_x_min = float(np.min(rough_result.phi_x))
    phi_x_max = float(np.max(rough_result.phi_x))
    frac_lambda_lt_1 = rough_result.frac_lambda_lt_1
    mean_phi_x = float(np.mean(rough_result.phi_x))

    # Debug: Aphi = phi_x * H^3
    phi, Z, _, _ = final_config.create_grid()
    H = film_model.H(phi, Z)
    H3 = H ** 3
    Aphi = rough_result.phi_x * H3

    Aphi_min = float(np.min(Aphi))
    Aphi_max = float(np.max(Aphi))
    H3_min = float(np.min(H3))
    H3_max = float(np.max(H3))

    return (eps_eq, h_min_um, p_max_MPa, P_loss_W, Q_cm3_s, f,
            lambda_min, lambda_max, phi_x_min, phi_x_max, frac_lambda_lt_1, mean_phi_x,
            Aphi_min, Aphi_max, H3_min, H3_max)


# ============================================================================
# СЕРИЯ A: Sweep по c и Ra с решателем
# ============================================================================

def run_series_a():
    """
    Серия A: управление lambda через c и Ra.

    Для каждой комбинации (c, Ra_level):
    1. Запуск Smooth (без Patir-Cheng)
    2. Запуск Rough (с Patir-Cheng)
    3. Вычисление дельт
    """
    print("=" * 70)
    print("СЕРИЯ A: Sweep по зазору c и шероховатости Ra")
    print("=" * 70)
    print(f"c values: {C_VALUES_UM} мкм")
    print(f"Ra levels: {list(RA_LEVELS.keys())}")
    print()

    results = []

    for c_um, c in zip(C_VALUES_UM, C_VALUES):
        print(f"\n--- c = {c_um} мкм ---")

        config = create_config(c=c)

        # Smooth (базовый)
        try:
            smooth = solve_smooth(config)
            eps_smooth, h_min_smooth, p_max_smooth, P_loss_smooth, Q_smooth, f_smooth = smooth
            print(f"  Smooth: ε={eps_smooth:.4f}, h_min={h_min_smooth:.2f}μm, P_loss={P_loss_smooth:.1f}W")
        except Exception as e:
            print(f"  Smooth FAILED: {e}")
            continue

        for Ra_level, Ra_params in RA_LEVELS.items():
            Ra_out = Ra_params['Ra_out']
            Ra_shaft = Ra_params['Ra_shaft']

            try:
                rough = solve_rough(config, Ra_out, Ra_shaft)
                (eps_rough, h_min_rough, p_max_rough, P_loss_rough, Q_rough, f_rough,
                 lambda_min, lambda_max, phi_x_min, phi_x_max, frac_lt_1, mean_phi_x,
                 Aphi_min, Aphi_max, H3_min, H3_max) = rough

                # Дельты
                dP_loss_pct = 100 * (P_loss_rough - P_loss_smooth) / P_loss_smooth
                dp_max_pct = 100 * (p_max_rough - p_max_smooth) / p_max_smooth
                dh_min_pct = 100 * (h_min_rough - h_min_smooth) / h_min_smooth
                deps_pct = 100 * (eps_rough - eps_smooth) / eps_smooth

                # Флаг эффекта
                rough_effect = (abs(dP_loss_pct) > EFFECT_THRESHOLD_PCT or
                               abs(dp_max_pct) > EFFECT_THRESHOLD_PCT or
                               abs(deps_pct) > EFFECT_THRESHOLD_PCT)

                # Режим lambda
                if lambda_min > 5:
                    lambda_regime = "hydro"
                elif lambda_min <= 3:
                    lambda_regime = "mixed_like"
                else:
                    lambda_regime = "transition"

                print(f"    {Ra_level}: λ_min={lambda_min:.2f}, φx_min={phi_x_min:.4f}, "
                      f"ΔP_loss={dP_loss_pct:+.2f}%, effect={rough_effect}")

                # Debug: Aphi vs H3
                print(f"           Aphi=[{Aphi_min:.4f}, {Aphi_max:.4f}], "
                      f"H³=[{H3_min:.4f}, {H3_max:.4f}], mean(φx)={mean_phi_x:.4f}")

                results.append({
                    'c_um': c_um,
                    'Ra_level': Ra_level,
                    'Ra_out_um': Ra_out * 1e6,
                    'Ra_shaft_um': Ra_shaft * 1e6,
                    # Smooth
                    'eps_smooth': eps_smooth,
                    'h_min_smooth_um': h_min_smooth,
                    'p_max_smooth_MPa': p_max_smooth,
                    'P_loss_smooth_W': P_loss_smooth,
                    'Q_smooth_cm3s': Q_smooth,
                    'f_smooth': f_smooth,
                    # Rough
                    'eps_rough': eps_rough,
                    'h_min_rough_um': h_min_rough,
                    'p_max_rough_MPa': p_max_rough,
                    'P_loss_rough_W': P_loss_rough,
                    'Q_rough_cm3s': Q_rough,
                    'f_rough': f_rough,
                    # Roughness stats
                    'lambda_min': lambda_min,
                    'lambda_max': lambda_max,
                    'phi_x_min': phi_x_min,
                    'phi_x_max': phi_x_max,
                    'frac_lambda_lt_1': frac_lt_1,
                    'mean_phi_x': mean_phi_x,
                    # Debug
                    'Aphi_min': Aphi_min,
                    'Aphi_max': Aphi_max,
                    'H3_min': H3_min,
                    'H3_max': H3_max,
                    # Deltas
                    'dP_loss_pct': dP_loss_pct,
                    'dp_max_pct': dp_max_pct,
                    'dh_min_pct': dh_min_pct,
                    'deps_pct': deps_pct,
                    # Flags
                    'rough_effect': rough_effect,
                    'lambda_regime': lambda_regime,
                })

            except Exception as e:
                print(f"    {Ra_level} FAILED: {e}")
                import traceback
                traceback.print_exc()

    return results


# ============================================================================
# СЕРИЯ B: Контрольный тест без решателя
# ============================================================================

def run_series_b():
    """
    Серия B: проверка модуля roughness на заданном H.

    Сгенерировать H вручную: H = 1 + eps*cos(phi - phi0)
    Для eps = 0.2, 0.6, 0.9 прогнать compute_roughness_fields.
    """
    print("\n" + "=" * 70)
    print("СЕРИЯ B: Контрольный тест модуля roughness (без решателя)")
    print("=" * 70)

    # Сетка
    n_phi, n_z = 180, 50
    phi = np.linspace(0, 2*np.pi, n_phi, endpoint=False)
    Z = np.linspace(-1, 1, n_z)
    PHI, _ = np.meshgrid(phi, Z, indexing='ij')
    phi0 = np.radians(45)

    eps_values = [0.2, 0.6, 0.9]
    c_test = [20e-6, 50e-6]  # два значения c
    Ra_test = ['Low', 'High']  # два уровня Ra

    results = []

    for eps in eps_values:
        H = 1.0 + eps * np.cos(PHI - phi0)

        for c in c_test:
            c_um = c * 1e6

            for Ra_level in Ra_test:
                Ra_out = RA_LEVELS[Ra_level]['Ra_out']
                Ra_shaft = RA_LEVELS[Ra_level]['Ra_shaft']

                roughness_params = RoughnessParams(
                    Ra_shaft=Ra_shaft,
                    Ra_out=Ra_out,
                    Ra_cell=Ra_out,
                )

                # Вычисляем roughness fields
                rough_result = compute_roughness_fields(
                    H, phi, Z, c,
                    roughness_params, texture_mask=None
                )

                lambda_min = rough_result.lambda_min
                lambda_max = rough_result.lambda_max
                phi_x_min = float(np.min(rough_result.phi_x))
                phi_x_max = float(np.max(rough_result.phi_x))
                frac_lt_1 = rough_result.frac_lambda_lt_1
                mean_phi_x = float(np.mean(rough_result.phi_x))

                print(f"  ε={eps}, c={c_um}μm, {Ra_level}: "
                      f"λ=[{lambda_min:.2f}, {lambda_max:.2f}], "
                      f"φx=[{phi_x_min:.4f}, {phi_x_max:.4f}], "
                      f"frac(λ<1)={frac_lt_1:.2%}")

                results.append({
                    'epsilon': eps,
                    'c_um': c_um,
                    'Ra_level': Ra_level,
                    'Ra_out_um': Ra_out * 1e6,
                    'Ra_shaft_um': Ra_shaft * 1e6,
                    'lambda_min': lambda_min,
                    'lambda_max': lambda_max,
                    'phi_x_min': phi_x_min,
                    'phi_x_max': phi_x_max,
                    'frac_lambda_lt_1': frac_lt_1,
                    'mean_phi_x': mean_phi_x,
                })

    return results


# ============================================================================
# ГРАФИКИ
# ============================================================================

def plot_results(results_a: List[Dict]):
    """Построить графики по результатам серии A."""

    if not results_a:
        print("Нет данных для графиков")
        return

    df = pd.DataFrame(results_a)

    # График 1: dP_loss_pct vs lambda_min
    fig, ax = plt.subplots(figsize=(10, 6))

    for Ra_level in RA_LEVELS.keys():
        subset = df[df['Ra_level'] == Ra_level]
        ax.scatter(subset['lambda_min'], subset['dP_loss_pct'],
                   label=Ra_level, s=100, alpha=0.7)

        # Соединяем точки линией
        subset_sorted = subset.sort_values('lambda_min')
        ax.plot(subset_sorted['lambda_min'], subset_sorted['dP_loss_pct'],
                '--', alpha=0.5)

    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax.axhline(y=EFFECT_THRESHOLD_PCT, color='r', linestyle=':',
               label=f'Порог эффекта ±{EFFECT_THRESHOLD_PCT}%')
    ax.axhline(y=-EFFECT_THRESHOLD_PCT, color='r', linestyle=':')
    ax.axvline(x=3, color='orange', linestyle='--', alpha=0.7,
               label='λ=3 (граница mixed)')
    ax.axvline(x=5, color='green', linestyle='--', alpha=0.7,
               label='λ=5 (граница hydro)')

    ax.set_xlabel('λ_min (минимальный параметр плёнки)', fontsize=12)
    ax.set_ylabel('ΔP_loss, %', fontsize=12)
    ax.set_title('Влияние шероховатости на потери мощности\n(Rough - Smooth)', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUT_DIR / 'dP_loss_vs_lambda_min.png', dpi=150)
    plt.close()
    print(f"График сохранён: {OUT_DIR / 'dP_loss_vs_lambda_min.png'}")

    # График 2: dp_max_pct vs lambda_min
    fig, ax = plt.subplots(figsize=(10, 6))

    for Ra_level in RA_LEVELS.keys():
        subset = df[df['Ra_level'] == Ra_level]
        ax.scatter(subset['lambda_min'], subset['dp_max_pct'],
                   label=Ra_level, s=100, alpha=0.7)
        subset_sorted = subset.sort_values('lambda_min')
        ax.plot(subset_sorted['lambda_min'], subset_sorted['dp_max_pct'],
                '--', alpha=0.5)

    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax.axhline(y=EFFECT_THRESHOLD_PCT, color='r', linestyle=':')
    ax.axhline(y=-EFFECT_THRESHOLD_PCT, color='r', linestyle=':')
    ax.axvline(x=3, color='orange', linestyle='--', alpha=0.7)
    ax.axvline(x=5, color='green', linestyle='--', alpha=0.7)

    ax.set_xlabel('λ_min', fontsize=12)
    ax.set_ylabel('Δp_max, %', fontsize=12)
    ax.set_title('Влияние шероховатости на максимальное давление', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUT_DIR / 'dp_max_vs_lambda_min.png', dpi=150)
    plt.close()
    print(f"График сохранён: {OUT_DIR / 'dp_max_vs_lambda_min.png'}")

    # График 3: phi_x_min vs lambda_min
    fig, ax = plt.subplots(figsize=(10, 6))

    for Ra_level in RA_LEVELS.keys():
        subset = df[df['Ra_level'] == Ra_level]
        ax.scatter(subset['lambda_min'], subset['phi_x_min'],
                   label=Ra_level, s=100, alpha=0.7)
        subset_sorted = subset.sort_values('lambda_min')
        ax.plot(subset_sorted['lambda_min'], subset_sorted['phi_x_min'],
                '--', alpha=0.5)

    ax.axhline(y=1.0, color='k', linestyle='-', linewidth=0.5, label='φx=1 (гладкий)')
    ax.axvline(x=3, color='orange', linestyle='--', alpha=0.7)
    ax.axvline(x=5, color='green', linestyle='--', alpha=0.7)

    ax.set_xlabel('λ_min', fontsize=12)
    ax.set_ylabel('φx_min (минимальный flow factor)', fontsize=12)
    ax.set_title('Flow factor vs параметр плёнки', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUT_DIR / 'phi_x_min_vs_lambda_min.png', dpi=150)
    plt.close()
    print(f"График сохранён: {OUT_DIR / 'phi_x_min_vs_lambda_min.png'}")


# ============================================================================
# ИТОГОВЫЙ ВЫВОД
# ============================================================================

def print_summary(results_a: List[Dict]):
    """Вывести итоговую сводку."""

    if not results_a:
        print("\nНет данных для сводки")
        return

    df = pd.DataFrame(results_a)

    print("\n" + "=" * 70)
    print("ИТОГОВАЯ СВОДКА")
    print("=" * 70)

    # Случаи, где эффект обнаружен
    effect_cases = df[df['rough_effect'] == True]

    print(f"\nВсего режимов: {len(df)}")
    print(f"С эффектом шероховатости (|Δ| > {EFFECT_THRESHOLD_PCT}%): {len(effect_cases)}")

    if len(effect_cases) > 0:
        print("\nРежимы с заметным эффектом:")
        print("-" * 70)
        for _, row in effect_cases.iterrows():
            print(f"  c={row['c_um']:3.0f}μm, {row['Ra_level']:4s}: "
                  f"λ_min={row['lambda_min']:5.2f}, φx_min={row['phi_x_min']:.4f}, "
                  f"ΔP_loss={row['dP_loss_pct']:+5.2f}%, Δp_max={row['dp_max_pct']:+5.2f}%")

    # Статистика по режимам
    print("\nСтатистика по режиму λ:")
    for regime in ['mixed_like', 'transition', 'hydro']:
        regime_df = df[df['lambda_regime'] == regime]
        if len(regime_df) > 0:
            print(f"  {regime:12s}: {len(regime_df)} случаев, "
                  f"mean |ΔP_loss|={regime_df['dP_loss_pct'].abs().mean():.2f}%")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Главная функция."""

    print("=" * 70)
    print("T3: Модуль тестирования влияния шероховатости Patir-Cheng")
    print("=" * 70)
    print(f"Выходная директория: {OUT_DIR}")
    print()

    # Создаём директорию
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Серия A
    results_a = run_series_a()

    # Сохраняем CSV
    if results_a:
        df_a = pd.DataFrame(results_a)
        csv_path = OUT_DIR / 'roughness_influence.csv'
        df_a.to_csv(csv_path, index=False)
        print(f"\nCSV сохранён: {csv_path}")

    # Серия B
    results_b = run_series_b()

    # Сохраняем CSV
    if results_b:
        df_b = pd.DataFrame(results_b)
        csv_path = OUT_DIR / 'roughness_control.csv'
        df_b.to_csv(csv_path, index=False)
        print(f"CSV сохранён: {csv_path}")

    # Графики
    plot_results(results_a)

    # Сводка
    print_summary(results_a)

    print("\n" + "=" * 70)
    print("ЗАВЕРШЕНО")
    print("=" * 70)


if __name__ == "__main__":
    main()
