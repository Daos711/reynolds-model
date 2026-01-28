#!/usr/bin/env python3
"""
T3: Модуль тестирования влияния шероховатости Patir-Cheng.

Цель: Доказать на серии режимов, что Patir-Cheng заметно влияет
при малом lambda (1-3) и почти не влияет при lambda >> 1.

Серия A: Sweep по (c, W_ext, n_rpm, Ra) с решателем
Серия B: Контрольный тест модуля roughness без решателя

Расширенная версия:
- Sweep по нагрузке W_ext и скорости n_rpm
- Опциональный shear factor для чувствительности P_loss
- Флаг clip_lambda для сравнения

Выходные файлы:
- results/roughness_sandbox/roughness_influence.csv
- results/roughness_sandbox/roughness_control.csv
- results/roughness_sandbox/dP_loss_vs_lambda_min.png
- results/roughness_sandbox/dp_max_vs_lambda_min.png
- results/roughness_sandbox/phi_x_min_vs_lambda_min.png
- results/roughness_sandbox/dh_min_vs_lambda_min.png
- results/roughness_sandbox/dP_loss_shear_vs_lambda_min.png (если включён shear)
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple
import argparse

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

# ============================================================================
# РЕЖИМЫ SWEEP (можно расширять)
# ============================================================================

# Базовый sweep (быстрый, как раньше)
C_VALUES_UM_BASIC = [10, 20, 30, 50]

# Расширенный sweep
C_VALUES_UM_EXTENDED = [10, 20, 30, 50]
W_EXT_VALUES = [10e3, 30e3, 50e3, 70e3]  # Н (нагрузка)
N_RPM_VALUES = [1000, 2000, 3000]         # об/мин (скорость)

# Уровни шероховатости
RA_LEVELS = {
    'Low': {'Ra_out': 0.5e-6, 'Ra_shaft': 0.3e-6},
    'Mid': {'Ra_out': 2.0e-6, 'Ra_shaft': 1.0e-6},
    'High': {'Ra_out': 8.0e-6, 'Ra_shaft': 4.0e-6},
}

# Порог эффекта шероховатости
EFFECT_THRESHOLD_PCT = 1.0  # %

# ============================================================================
# ОПЦИИ РАСШИРЕННОГО РЕЖИМА
# ============================================================================

# Включить расширенный sweep (W_ext, n_rpm)
EXTENDED_SWEEP = True

# Включить shear factor для P_loss
USE_SHEAR_FACTOR = True

# Параметры shear factor: k_shear(λ) = 1 + a * exp(-b * λ)
SHEAR_FACTOR_A = 0.3
SHEAR_FACTOR_B = 0.8

# Флаг clip lambda (True = как в roughness.py, λ_eff = max(λ, 1))
CLIP_LAMBDA = True

OUT_DIR = Path("results/roughness_sandbox")


# ============================================================================
# SHEAR FACTOR
# ============================================================================

def shear_factor(lambda_field: np.ndarray, a: float = SHEAR_FACTOR_A, b: float = SHEAR_FACTOR_B) -> np.ndarray:
    """
    Тестовый множитель для Couette-трения.
    k_shear(λ) = 1 + a * exp(-b * λ)

    При λ → ∞: k_shear → 1 (гладкий)
    При λ → 0: k_shear → 1 + a (увеличенное трение)

    Args:
        lambda_field: поле λ = H/σ*
        a: амплитуда эффекта (default 0.3 = +30% при λ→0)
        b: скорость затухания (default 0.8)

    Returns:
        k_shear: множитель для τ_couette
    """
    return 1.0 + a * np.exp(-b * lambda_field)


def compute_P_loss_with_shear(config: BearingConfig, reynolds_result, lambda_field: np.ndarray) -> float:
    """
    Вычислить P_loss с учётом shear factor.

    P_loss_shear = ∫∫ τ_couette * k_shear(λ) * U dA

    Это упрощённая модель - просто масштабируем P_friction.
    """
    # Получаем базовый P_loss
    s2 = compute_stage2(reynolds_result, config)
    P_loss_base = s2.losses.P_friction

    # Средний shear factor
    k_shear = shear_factor(lambda_field)
    k_shear_mean = float(np.mean(k_shear))

    # Масштабируем (упрощённо: P_loss ~ τ ~ k_shear)
    P_loss_shear = P_loss_base * k_shear_mean

    return P_loss_shear, k_shear_mean


# ============================================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ============================================================================

def create_config(c: float, mu: float = BASE_MU, n_rpm: int = BASE_N_RPM,
                  n_phi: int = 180, n_z: int = 50) -> BearingConfig:
    """Создать конфигурацию подшипника."""
    return BearingConfig(
        R=BASE_R,
        L=BASE_L,
        c=c,
        epsilon=0.5,  # начальное значение, будет найдено через find_equilibrium
        phi0=np.radians(45),
        n_rpm=n_rpm,
        mu=mu,
        n_phi=n_phi,
        n_z=n_z,
    )


def solve_smooth(config: BearingConfig, W_ext: float) -> Tuple:
    """
    Решить для гладкого подшипника (без Patir-Cheng).

    Returns:
        (eps_eq, h_min_um, p_max_MPa, P_loss_W, Q_cm3_s, f)
    """
    eq = find_equilibrium(
        config, W_ext=W_ext, load_angle=-np.pi/2,
        verbose=False, film_model_factory=lambda cfg: SmoothFilmModel(cfg)
    )

    s2 = eq.stage2
    if s2 is None:
        raise RuntimeError("stage2 is None")

    eps_eq = eq.epsilon
    h_min_um = s2.reynolds.h_min * 1e6
    p_max_MPa = s2.reynolds.p_max / 1e6
    P_loss_W = s2.losses.P_friction
    Q_cm3_s = s2.flow.Q_total * 1e6
    f = s2.friction.mu_friction

    return eps_eq, h_min_um, p_max_MPa, P_loss_W, Q_cm3_s, f


def solve_rough(config: BearingConfig, Ra_out: float, Ra_shaft: float,
                W_ext: float, use_shear: bool = False) -> Tuple:
    """
    Решить с Patir-Cheng (шероховатость включена).

    Ra_cell = Ra_out (нет текстуры).

    Returns:
        (eps_eq, h_min_um, p_max_MPa, P_loss_W, Q_cm3_s, f,
         lambda_min, lambda_max, phi_x_min, phi_x_max, frac_lambda_lt_1, mean_phi_x,
         Aphi_min, Aphi_max, H3_min, H3_max,
         P_loss_shear_W, k_shear_mean)
    """
    roughness_params = RoughnessParams(
        Ra_shaft=Ra_shaft,
        Ra_out=Ra_out,
        Ra_cell=Ra_out,
    )

    def film_model_factory_with_roughness(cfg):
        return SmoothFilmModel(cfg)

    def compute_with_roughness(cfg, film_model):
        """Решить уравнение Рейнольдса с учётом шероховатости."""
        phi, Z, _, _ = cfg.create_grid()
        H = film_model.H(phi, Z)

        rough_result = compute_roughness_fields(
            H, phi, Z, cfg.c,
            roughness_params, texture_mask=None
        )

        reynolds_result = solve_reynolds(
            cfg, film_model,
            phi_x=rough_result.phi_x,
            phi_z=rough_result.phi_z,
            sigma_star=rough_result.sigma_star,
            lambda_field=rough_result.lambda_field,
            frac_lambda_lt_1=rough_result.frac_lambda_lt_1,
        )

        return reynolds_result, rough_result

    eq = find_equilibrium(
        config, W_ext=W_ext, load_angle=-np.pi/2,
        verbose=False, film_model_factory=film_model_factory_with_roughness
    )

    final_config = BearingConfig(
        R=config.R, L=config.L, c=config.c,
        epsilon=eq.epsilon, phi0=eq.phi0,
        n_rpm=config.n_rpm, mu=config.mu,
        n_phi=config.n_phi, n_z=config.n_z,
    )

    film_model = SmoothFilmModel(final_config)
    reynolds_result, rough_result = compute_with_roughness(final_config, film_model)

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

    # Shear factor
    P_loss_shear_W = None
    k_shear_mean = None
    if use_shear:
        P_loss_shear_W, k_shear_mean = compute_P_loss_with_shear(
            final_config, reynolds_result, rough_result.lambda_field
        )

    return (eps_eq, h_min_um, p_max_MPa, P_loss_W, Q_cm3_s, f,
            lambda_min, lambda_max, phi_x_min, phi_x_max, frac_lambda_lt_1, mean_phi_x,
            Aphi_min, Aphi_max, H3_min, H3_max,
            P_loss_shear_W, k_shear_mean)


def get_lambda_zone(lambda_min: float) -> str:
    """Определить зону λ."""
    if lambda_min < 1:
        return "severe"
    elif lambda_min <= 3:
        return "mixed"
    elif lambda_min <= 5:
        return "transition"
    else:
        return "hydro"


# ============================================================================
# СЕРИЯ A: Sweep с решателем
# ============================================================================

def run_series_a(extended: bool = EXTENDED_SWEEP, use_shear: bool = USE_SHEAR_FACTOR):
    """
    Серия A: управление lambda через c, W_ext, n_rpm и Ra.

    Для каждой комбинации:
    1. Запуск Smooth (без Patir-Cheng)
    2. Запуск Rough (с Patir-Cheng)
    3. Вычисление дельт
    """
    print("=" * 70)
    print("СЕРИЯ A: Sweep по параметрам")
    print("=" * 70)

    if extended:
        c_values_um = C_VALUES_UM_EXTENDED
        w_ext_values = W_EXT_VALUES
        n_rpm_values = N_RPM_VALUES
        total = len(c_values_um) * len(w_ext_values) * len(n_rpm_values) * len(RA_LEVELS)
        print(f"Расширенный режим: {total} комбинаций")
        print(f"  c: {c_values_um} мкм")
        print(f"  W_ext: {[w/1000 for w in w_ext_values]} кН")
        print(f"  n_rpm: {n_rpm_values} об/мин")
    else:
        c_values_um = C_VALUES_UM_BASIC
        w_ext_values = [BASE_W_EXT]
        n_rpm_values = [BASE_N_RPM]
        total = len(c_values_um) * len(RA_LEVELS)
        print(f"Базовый режим: {total} комбинаций")
        print(f"  c: {c_values_um} мкм")

    print(f"  Ra levels: {list(RA_LEVELS.keys())}")
    print(f"  Shear factor: {use_shear}")
    print(f"  Clip lambda: {CLIP_LAMBDA}")
    print()

    results = []
    count = 0

    for c_um in c_values_um:
        c = c_um * 1e-6

        for W_ext in w_ext_values:
            W_ext_kN = W_ext / 1000

            for n_rpm in n_rpm_values:
                count += 1
                print(f"\n[{count}/{total}] c={c_um}μm, W={W_ext_kN}kN, n={n_rpm}rpm")

                config = create_config(c=c, n_rpm=n_rpm)

                # Smooth (базовый)
                try:
                    smooth = solve_smooth(config, W_ext)
                    eps_smooth, h_min_smooth, p_max_smooth, P_loss_smooth, Q_smooth, f_smooth = smooth
                    print(f"  Smooth: ε={eps_smooth:.4f}, h_min={h_min_smooth:.2f}μm")
                except Exception as e:
                    print(f"  Smooth FAILED: {e}")
                    continue

                for Ra_level, Ra_params in RA_LEVELS.items():
                    Ra_out = Ra_params['Ra_out']
                    Ra_shaft = Ra_params['Ra_shaft']

                    try:
                        rough = solve_rough(config, Ra_out, Ra_shaft, W_ext, use_shear=use_shear)
                        (eps_rough, h_min_rough, p_max_rough, P_loss_rough, Q_rough, f_rough,
                         lambda_min, lambda_max, phi_x_min, phi_x_max, frac_lt_1, mean_phi_x,
                         Aphi_min, Aphi_max, H3_min, H3_max,
                         P_loss_shear_W, k_shear_mean) = rough

                        # Дельты
                        dP_loss_pct = 100 * (P_loss_rough - P_loss_smooth) / P_loss_smooth
                        dp_max_pct = 100 * (p_max_rough - p_max_smooth) / p_max_smooth
                        dh_min_pct = 100 * (h_min_rough - h_min_smooth) / h_min_smooth
                        deps_pct = 100 * (eps_rough - eps_smooth) / eps_smooth

                        # Shear factor дельта
                        dP_loss_shear_pct = None
                        if use_shear and P_loss_shear_W is not None:
                            dP_loss_shear_pct = 100 * (P_loss_shear_W - P_loss_smooth) / P_loss_smooth

                        # Флаг эффекта
                        rough_effect = (abs(dP_loss_pct) > EFFECT_THRESHOLD_PCT or
                                       abs(dp_max_pct) > EFFECT_THRESHOLD_PCT or
                                       abs(deps_pct) > EFFECT_THRESHOLD_PCT)

                        # Зона lambda
                        lambda_zone = get_lambda_zone(lambda_min)

                        # Режим lambda (для совместимости)
                        if lambda_min > 5:
                            lambda_regime = "hydro"
                        elif lambda_min <= 3:
                            lambda_regime = "mixed_like"
                        else:
                            lambda_regime = "transition"

                        shear_info = ""
                        if use_shear and dP_loss_shear_pct is not None:
                            shear_info = f", ΔP_shear={dP_loss_shear_pct:+.2f}%"

                        print(f"    {Ra_level}: λ_min={lambda_min:.2f}, φx={phi_x_min:.3f}, "
                              f"Δp_max={dp_max_pct:+.1f}%{shear_info}")

                        results.append({
                            # Параметры режима
                            'c_um': c_um,
                            'W_ext_kN': W_ext_kN,
                            'n_rpm': n_rpm,
                            'Ra_level': Ra_level,
                            'Ra_out_um': Ra_out * 1e6,
                            'Ra_shaft_um': Ra_shaft * 1e6,
                            'clip_lambda_used': CLIP_LAMBDA,
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
                            # Shear factor
                            'k_shear_mean': k_shear_mean,
                            'P_loss_shear_W': P_loss_shear_W,
                            'dP_loss_shear_pct': dP_loss_shear_pct,
                            # Deltas
                            'dP_loss_pct': dP_loss_pct,
                            'dp_max_pct': dp_max_pct,
                            'dh_min_pct': dh_min_pct,
                            'deps_pct': deps_pct,
                            # Flags
                            'rough_effect': rough_effect,
                            'lambda_regime': lambda_regime,
                            'lambda_zone': lambda_zone,
                        })

                    except Exception as e:
                        print(f"    {Ra_level} FAILED: {e}")

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

    n_phi, n_z = 180, 50
    phi = np.linspace(0, 2*np.pi, n_phi, endpoint=False)
    Z = np.linspace(-1, 1, n_z)
    PHI, _ = np.meshgrid(phi, Z, indexing='ij')
    phi0 = np.radians(45)

    eps_values = [0.2, 0.6, 0.9]
    c_test = [20e-6, 50e-6]
    Ra_test = ['Low', 'High']

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

                # Shear factor stats
                k_shear = shear_factor(rough_result.lambda_field)
                k_shear_mean = float(np.mean(k_shear))
                k_shear_max = float(np.max(k_shear))

                print(f"  ε={eps}, c={c_um}μm, {Ra_level}: "
                      f"λ=[{lambda_min:.2f}, {lambda_max:.2f}], "
                      f"φx=[{phi_x_min:.4f}, {phi_x_max:.4f}], "
                      f"k_shear={k_shear_mean:.3f}")

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
                    'k_shear_mean': k_shear_mean,
                    'k_shear_max': k_shear_max,
                    'lambda_zone': get_lambda_zone(lambda_min),
                })

    return results


# ============================================================================
# ГРАФИКИ
# ============================================================================

def plot_results(results_a: List[Dict], use_shear: bool = USE_SHEAR_FACTOR):
    """Построить графики по результатам серии A."""

    if not results_a:
        print("Нет данных для графиков")
        return

    df = pd.DataFrame(results_a)

    # Цвета для Ra levels
    colors = {'Low': 'green', 'Mid': 'orange', 'High': 'red'}

    # График 1: dP_loss_pct vs lambda_min
    fig, ax = plt.subplots(figsize=(10, 6))

    for Ra_level in RA_LEVELS.keys():
        subset = df[df['Ra_level'] == Ra_level]
        ax.scatter(subset['lambda_min'], subset['dP_loss_pct'],
                   label=Ra_level, s=60, alpha=0.7, c=colors[Ra_level])

    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax.axhline(y=EFFECT_THRESHOLD_PCT, color='gray', linestyle=':', alpha=0.7)
    ax.axhline(y=-EFFECT_THRESHOLD_PCT, color='gray', linestyle=':',alpha=0.7)
    ax.axvline(x=1, color='red', linestyle='--', alpha=0.5, label='λ=1 (severe)')
    ax.axvline(x=3, color='orange', linestyle='--', alpha=0.5, label='λ=3 (mixed)')
    ax.axvline(x=5, color='green', linestyle='--', alpha=0.5, label='λ=5 (hydro)')

    ax.set_xlabel('λ_min (минимальный параметр плёнки)', fontsize=12)
    ax.set_ylabel('ΔP_loss, %', fontsize=12)
    ax.set_title('Влияние шероховатости на потери мощности\n(Rough - Smooth)', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUT_DIR / 'dP_loss_vs_lambda_min.png', dpi=150)
    plt.close()
    print(f"График: {OUT_DIR / 'dP_loss_vs_lambda_min.png'}")

    # График 2: dp_max_pct vs lambda_min
    fig, ax = plt.subplots(figsize=(10, 6))

    for Ra_level in RA_LEVELS.keys():
        subset = df[df['Ra_level'] == Ra_level]
        ax.scatter(subset['lambda_min'], subset['dp_max_pct'],
                   label=Ra_level, s=60, alpha=0.7, c=colors[Ra_level])

    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax.axhline(y=10, color='gray', linestyle=':', alpha=0.7, label='±10%')
    ax.axhline(y=-10, color='gray', linestyle=':', alpha=0.7)
    ax.axvline(x=1, color='red', linestyle='--', alpha=0.5)
    ax.axvline(x=3, color='orange', linestyle='--', alpha=0.5)
    ax.axvline(x=5, color='green', linestyle='--', alpha=0.5)

    ax.set_xlabel('λ_min', fontsize=12)
    ax.set_ylabel('Δp_max, %', fontsize=12)
    ax.set_title('Влияние шероховатости на максимальное давление', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUT_DIR / 'dp_max_vs_lambda_min.png', dpi=150)
    plt.close()
    print(f"График: {OUT_DIR / 'dp_max_vs_lambda_min.png'}")

    # График 3: phi_x_min vs lambda_min
    fig, ax = plt.subplots(figsize=(10, 6))

    for Ra_level in RA_LEVELS.keys():
        subset = df[df['Ra_level'] == Ra_level]
        ax.scatter(subset['lambda_min'], subset['phi_x_min'],
                   label=Ra_level, s=60, alpha=0.7, c=colors[Ra_level])

    ax.axhline(y=1.0, color='k', linestyle='-', linewidth=0.5, label='φx=1 (гладкий)')
    ax.axvline(x=1, color='red', linestyle='--', alpha=0.5)
    ax.axvline(x=3, color='orange', linestyle='--', alpha=0.5)
    ax.axvline(x=5, color='green', linestyle='--', alpha=0.5)

    ax.set_xlabel('λ_min', fontsize=12)
    ax.set_ylabel('φx_min (минимальный flow factor)', fontsize=12)
    ax.set_title('Flow factor vs параметр плёнки', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUT_DIR / 'phi_x_min_vs_lambda_min.png', dpi=150)
    plt.close()
    print(f"График: {OUT_DIR / 'phi_x_min_vs_lambda_min.png'}")

    # График 4: dh_min_pct vs lambda_min (НОВЫЙ)
    fig, ax = plt.subplots(figsize=(10, 6))

    for Ra_level in RA_LEVELS.keys():
        subset = df[df['Ra_level'] == Ra_level]
        ax.scatter(subset['lambda_min'], subset['dh_min_pct'],
                   label=Ra_level, s=60, alpha=0.7, c=colors[Ra_level])

    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax.axhline(y=EFFECT_THRESHOLD_PCT, color='gray', linestyle=':', alpha=0.7)
    ax.axhline(y=-EFFECT_THRESHOLD_PCT, color='gray', linestyle=':', alpha=0.7)
    ax.axvline(x=1, color='red', linestyle='--', alpha=0.5)
    ax.axvline(x=3, color='orange', linestyle='--', alpha=0.5)
    ax.axvline(x=5, color='green', linestyle='--', alpha=0.5)

    ax.set_xlabel('λ_min', fontsize=12)
    ax.set_ylabel('Δh_min, %', fontsize=12)
    ax.set_title('Влияние шероховатости на минимальную толщину плёнки', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUT_DIR / 'dh_min_vs_lambda_min.png', dpi=150)
    plt.close()
    print(f"График: {OUT_DIR / 'dh_min_vs_lambda_min.png'}")

    # График 5: dP_loss_shear_pct vs lambda_min (если включён shear)
    if use_shear and 'dP_loss_shear_pct' in df.columns:
        df_shear = df[df['dP_loss_shear_pct'].notna()]
        if len(df_shear) > 0:
            fig, ax = plt.subplots(figsize=(10, 6))

            for Ra_level in RA_LEVELS.keys():
                subset = df_shear[df_shear['Ra_level'] == Ra_level]
                ax.scatter(subset['lambda_min'], subset['dP_loss_shear_pct'],
                           label=Ra_level, s=60, alpha=0.7, c=colors[Ra_level])

            ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
            ax.axhline(y=1, color='gray', linestyle=':', alpha=0.7, label='±1%')
            ax.axhline(y=-1, color='gray', linestyle=':', alpha=0.7)
            ax.axvline(x=1, color='red', linestyle='--', alpha=0.5)
            ax.axvline(x=3, color='orange', linestyle='--', alpha=0.5)
            ax.axvline(x=5, color='green', linestyle='--', alpha=0.5)

            ax.set_xlabel('λ_min', fontsize=12)
            ax.set_ylabel('ΔP_loss (с shear factor), %', fontsize=12)
            ax.set_title(f'Влияние шероховатости на P_loss с shear factor\n'
                         f'k_shear(λ) = 1 + {SHEAR_FACTOR_A}·exp(-{SHEAR_FACTOR_B}·λ)', fontsize=14)
            ax.legend()
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(OUT_DIR / 'dP_loss_shear_vs_lambda_min.png', dpi=150)
            plt.close()
            print(f"График: {OUT_DIR / 'dP_loss_shear_vs_lambda_min.png'}")


# ============================================================================
# ИТОГОВЫЙ ВЫВОД
# ============================================================================

def print_summary(results_a: List[Dict], use_shear: bool = USE_SHEAR_FACTOR):
    """Вывести итоговую сводку."""

    if not results_a:
        print("\nНет данных для сводки")
        return

    df = pd.DataFrame(results_a)

    print("\n" + "=" * 70)
    print("ИТОГОВАЯ СВОДКА")
    print("=" * 70)

    print(f"\nВсего режимов: {len(df)}")

    # Случаи, где эффект обнаружен
    effect_cases = df[df['rough_effect'] == True]
    print(f"С эффектом шероховатости (|Δ| > {EFFECT_THRESHOLD_PCT}%): {len(effect_cases)}")

    # Статистика по зонам λ (расширенная)
    print("\n" + "-" * 70)
    print("Статистика по зонам λ:")
    print("-" * 70)
    print(f"{'Зона':<15} {'N':>5} {'mean|Δp_max|':>12} {'mean|ΔP_loss|':>13}", end="")
    if use_shear:
        print(f" {'mean|ΔP_shear|':>14}", end="")
    print()
    print("-" * 70)

    for zone in ['hydro', 'transition', 'mixed', 'severe']:
        zone_df = df[df['lambda_zone'] == zone]
        if len(zone_df) > 0:
            mean_dp_max = zone_df['dp_max_pct'].abs().mean()
            mean_dP_loss = zone_df['dP_loss_pct'].abs().mean()

            bounds = {
                'hydro': 'λ>5',
                'transition': '3<λ≤5',
                'mixed': '1<λ≤3',
                'severe': 'λ≤1'
            }

            print(f"  {zone:<8} ({bounds[zone]:<6}): {len(zone_df):>3}", end="")
            print(f"      {mean_dp_max:>6.2f}%       {mean_dP_loss:>6.2f}%", end="")

            if use_shear and 'dP_loss_shear_pct' in df.columns:
                shear_df = zone_df[zone_df['dP_loss_shear_pct'].notna()]
                if len(shear_df) > 0:
                    mean_dP_shear = shear_df['dP_loss_shear_pct'].abs().mean()
                    print(f"        {mean_dP_shear:>6.2f}%", end="")
            print()

    # Топ-5 по эффекту
    if len(df) > 0:
        print("\n" + "-" * 70)
        print("Топ-5 режимов по |Δp_max|:")
        print("-" * 70)
        top5 = df.nlargest(5, 'dp_max_pct', keep='first')
        for _, row in top5.iterrows():
            print(f"  c={row['c_um']:2.0f}μm, W={row['W_ext_kN']:.0f}kN, n={row['n_rpm']:.0f}rpm, "
                  f"{row['Ra_level']:4s}: λ_min={row['lambda_min']:5.2f}, "
                  f"Δp_max={row['dp_max_pct']:+6.1f}%")

    # Проверка критериев приёмки
    print("\n" + "-" * 70)
    print("Проверка критериев:")
    print("-" * 70)

    n_hydro = len(df[df['lambda_zone'] == 'hydro'])
    n_transition = len(df[df['lambda_zone'] == 'transition'])
    n_mixed = len(df[df['lambda_zone'] == 'mixed'])
    n_severe = len(df[df['lambda_zone'] == 'severe'])

    check_hydro = "✓" if n_hydro >= 5 else "✗"
    check_trans = "✓" if n_transition >= 5 else "✗"
    check_mixed = "✓" if n_mixed >= 5 else "✗"

    print(f"  {check_hydro} hydro (λ>5): {n_hydro} точек (нужно ≥5)")
    print(f"  {check_trans} transition (3<λ≤5): {n_transition} точек (нужно ≥5)")
    print(f"  {check_mixed} mixed (1<λ≤3): {n_mixed} точек (нужно ≥5)")
    print(f"    severe (λ≤1): {n_severe} точек")

    # Проверка эффекта
    dp_max_effect = df[(df['lambda_min'] < 3) & (df['dp_max_pct'].abs() > 10)]
    check_dp = "✓" if len(dp_max_effect) > 0 else "✗"
    print(f"  {check_dp} |Δp_max| > 10% при λ<3: {len(dp_max_effect)} случаев")

    if use_shear and 'dP_loss_shear_pct' in df.columns:
        shear_effect = df[(df['lambda_min'] < 3) & (df['dP_loss_shear_pct'].notna()) &
                          (df['dP_loss_shear_pct'].abs() > 1)]
        check_shear = "✓" if len(shear_effect) > 0 else "✗"
        print(f"  {check_shear} |ΔP_loss_shear| > 1% при λ<3: {len(shear_effect)} случаев")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Главная функция."""

    parser = argparse.ArgumentParser(description='Тестирование влияния шероховатости Patir-Cheng')
    parser.add_argument('--basic', action='store_true', help='Базовый режим (без расширенного sweep)')
    parser.add_argument('--no-shear', action='store_true', help='Отключить shear factor')
    args = parser.parse_args()

    extended = not args.basic
    use_shear = not args.no_shear

    print("=" * 70)
    print("T3: Модуль тестирования влияния шероховатости Patir-Cheng")
    print("=" * 70)
    print(f"Режим: {'расширенный' if extended else 'базовый'}")
    print(f"Shear factor: {'включён' if use_shear else 'выключен'}")
    print(f"Clip lambda: {CLIP_LAMBDA}")
    print(f"Выходная директория: {OUT_DIR}")
    print()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Серия A
    results_a = run_series_a(extended=extended, use_shear=use_shear)

    if results_a:
        df_a = pd.DataFrame(results_a)
        csv_path = OUT_DIR / 'roughness_influence.csv'
        df_a.to_csv(csv_path, index=False)
        print(f"\nCSV: {csv_path}")

    # Серия B
    results_b = run_series_b()

    if results_b:
        df_b = pd.DataFrame(results_b)
        csv_path = OUT_DIR / 'roughness_control.csv'
        df_b.to_csv(csv_path, index=False)
        print(f"CSV: {csv_path}")

    # Графики
    plot_results(results_a, use_shear=use_shear)

    # Сводка
    print_summary(results_a, use_shear=use_shear)

    print("\n" + "=" * 70)
    print("ЗАВЕРШЕНО")
    print("=" * 70)


if __name__ == "__main__":
    main()
