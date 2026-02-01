#!/usr/bin/env python3
"""
Этап 11: Параметрический анализ текстуры (ЦКП).

Исследование влияния геометрических параметров текстуры
на трибологические и динамические характеристики подшипника.

Два режима:
- lite: грубая сетка, быстрый расчёт (~5-15 мин)
- full: нормальная сетка, полный расчёт (~1-2 часа)

Использование:
    python run_stage11_parametric.py --mode lite
    python run_stage11_parametric.py --mode full --dynamics
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import time

from bearing_solver.parametric import (
    ParametricConfig,
    TextureFactors,
    ParametricResult,
    run_parametric_study,
    run_single_case,
    fit_rsm_model,
    find_optimum,
    local_refinement,
)

OUT_DIR = Path("results/stage11_parametric")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# КОНФИГУРАЦИЯ
# ============================================================================

# Подшипник (от препода)
BEARING_CONFIG = ParametricConfig(
    R=34.5e-3,        # 69/2 мм
    B=103.5e-3,       # мм
    c=60e-6,          # 60 мкм
    epsilon=0.8,      # фиксирован
    n_rpm=2980,
    mu=0.037,         # VG100 при 60°C
)

# Диапазоны факторов (от ТЗ препода)
# Полуоси = оси/2: 50-500 мкм → 25-250 мкм
FACTOR_RANGES = {
    'h':     (10e-6, 210e-6),      # глубина: 10-210 мкм
    'a':     (25e-6, 250e-6),      # полуось a: 25-250 мкм
    'b':     (25e-6, 250e-6),      # полуось b: 25-250 мкм
    'N_phi': (4, 12),              # рядов по φ
    'N_z':   (2, 6),               # рядов по z
}

# Расширенные диапазоны N для эксперимента с бо́льшим эффектом текстуры
FACTOR_RANGES_EXTENDED = {
    'h':     (10e-6, 210e-6),      # глубина: 10-210 мкм
    'a':     (25e-6, 250e-6),      # полуось a: 25-250 мкм
    'b':     (25e-6, 250e-6),      # полуось b: 25-250 мкм
    'N_phi': (20, 60),             # рядов по φ (больше!)
    'N_z':   (10, 20),             # рядов по z (больше!)
}

# Режимы сетки
GRID_LITE = (60, 20)    # быстрый расчёт
GRID_FULL = (120, 40)   # нормальный расчёт
GRID_FINE = (180, 60)   # точный расчёт


# ============================================================================
# ПРОВЕРКА ЕДИНИЦ
# ============================================================================

def check_model_units():
    """Проверить единицы поля H."""
    from bearing_solver import BearingConfig, solve_reynolds

    config = BearingConfig(
        R=BEARING_CONFIG.R,
        L=BEARING_CONFIG.B,
        c=BEARING_CONFIG.c,
        epsilon=0.5,
        phi0=np.pi,
        n_rpm=BEARING_CONFIG.n_rpm,
        mu=BEARING_CONFIG.mu,
        n_phi=60,
        n_z=20,
    )

    sol = solve_reynolds(config, film_model=None)

    H_min, H_max = np.min(sol.H), np.max(sol.H)

    print("Проверка единиц H:")
    print(f"  min(H) = {H_min:.6f}")
    print(f"  max(H) = {H_max:.6f}")
    print(f"  h_min = c * min(H) = {BEARING_CONFIG.c * H_min * 1e6:.2f} мкм")

    if 0.1 < H_max < 10:
        print("  -> H безразмерный (h/c) - OK")
        return True
    else:
        print("  -> ВНИМАНИЕ: проверить единицы!")
        return False


# ============================================================================
# ОСНОВНОЙ РАСЧЁТ
# ============================================================================

def run_study(mode: str = 'lite', compute_dynamics: bool = False, n_jobs: int = 1,
               extended: bool = False):
    """Запустить параметрическое исследование."""

    suffix = f"{mode}_extended" if extended else mode
    factor_ranges = FACTOR_RANGES_EXTENDED if extended else FACTOR_RANGES

    print("=" * 60)
    print(f"ПАРАМЕТРИЧЕСКИЙ АНАЛИЗ ТЕКСТУРЫ — режим: {mode.upper()}" +
          (" (EXTENDED)" if extended else ""))
    print("=" * 60)

    # Проверка единиц
    check_model_units()

    # Выбрать сетку
    if mode == 'lite':
        n_phi, n_z = GRID_LITE
    elif mode == 'full':
        n_phi, n_z = GRID_FULL
    else:
        n_phi, n_z = GRID_FINE

    print(f"\nПодшипник: D={BEARING_CONFIG.R*2*1000:.1f}мм, "
          f"B={BEARING_CONFIG.B*1000:.1f}мм, c={BEARING_CONFIG.c*1e6:.0f}мкм")
    print(f"Режим: eps={BEARING_CONFIG.epsilon}, n={BEARING_CONFIG.n_rpm} rpm")
    print(f"Сетка: {n_phi}x{n_z}")
    print(f"Динамика: {'ДА' if compute_dynamics else 'НЕТ'}")
    if extended:
        print(f"Extended N: N_phi={factor_ranges['N_phi']}, N_z={factor_ranges['N_z']}")

    # Запустить
    start = time.time()

    results_df = run_parametric_study(
        pconfig=BEARING_CONFIG,
        factor_ranges=factor_ranges,
        n_phi_grid=n_phi,
        n_z_grid=n_z,
        compute_dynamics=compute_dynamics,
        n_jobs=n_jobs,
        verbose=True,
    )

    elapsed = time.time() - start
    print(f"\nВремя расчёта: {elapsed/60:.1f} мин")

    # Сохранить результаты
    results_df.to_csv(OUT_DIR / f"results_{suffix}.csv", index=False)
    print(f"Результаты: {OUT_DIR}/results_{suffix}.csv")

    return results_df, suffix, factor_ranges


# ============================================================================
# АНАЛИЗ РЕЗУЛЬТАТОВ
# ============================================================================

def analyze_results(results_df: pd.DataFrame, suffix: str = 'lite',
                    factor_ranges: dict = None):
    """Анализ и визуализация результатов."""

    if factor_ranges is None:
        factor_ranges = FACTOR_RANGES

    print("\n" + "=" * 60)
    print("АНАЛИЗ РЕЗУЛЬТАТОВ")
    print("=" * 60)

    # Статистика
    valid = results_df[results_df['valid']]
    invalid = results_df[~results_df['valid']]

    print(f"\nВсего точек: {len(results_df)}")
    print(f"Валидных: {len(valid)}")
    print(f"Невалидных (перекрытие): {len(invalid)}")

    if len(valid) == 0:
        print("Нет валидных точек для анализа!")
        return None, None

    # Статистика по метрикам
    mu_min, mu_max = valid['mu_friction'].min(), valid['mu_friction'].max()
    mu_range_pct = (mu_max - mu_min) / mu_min * 100

    print(f"\nСтатистика (валидные точки):")
    print(f"  mu_friction: min={mu_min:.6f}, max={mu_max:.6f}, "
          f"range={mu_range_pct:.2f}%")
    print(f"  W: min={valid['W'].min():.1f} Н, max={valid['W'].max():.1f} Н")
    print(f"  h_min: min={valid['h_min'].min()*1e6:.2f} мкм, "
          f"max={valid['h_min'].max()*1e6:.2f} мкм")
    print(f"  fill_factor: min={valid['fill_factor'].min()*100:.4f}%, "
          f"max={valid['fill_factor'].max()*100:.4f}%")
    print(f"  N_total: min={valid['N_total'].min()}, max={valid['N_total'].max()}")

    # --- RSM для коэффициента трения ---
    print("\n--- RSM: Коэффициент трения ---")
    factors = ['h', 'a', 'b', 'N_phi', 'N_z']

    rsm_mu = fit_rsm_model(valid, factors, 'mu_friction')
    opt_mu = None

    if 'error' not in rsm_mu:
        print(f"R^2 = {rsm_mu['r2']:.4f} (точек: {rsm_mu['n_points']})")

        opt_mu = find_optimum(rsm_mu, factor_ranges, minimize=True)
        if 'error' not in opt_mu:
            print(f"\nОптимум mu_friction (RSM):")
            print(f"  h = {opt_mu['h']*1e6:.1f} мкм")
            print(f"  a = {opt_mu['a']*1e6:.1f} мкм")
            print(f"  b = {opt_mu['b']*1e6:.1f} мкм")
            print(f"  N_phi = {opt_mu['N_phi']:.1f}")
            print(f"  N_z = {opt_mu['N_z']:.1f}")
            print(f"  mu_friction = {opt_mu['response']:.6f}")
    else:
        print(f"Ошибка RSM: {rsm_mu['error']}")

    # --- RSM для несущей способности ---
    print("\n--- RSM: Несущая способность ---")
    rsm_W = fit_rsm_model(valid, factors, 'W')
    opt_W = None

    if 'error' not in rsm_W:
        print(f"R^2 = {rsm_W['r2']:.4f}")

        opt_W = find_optimum(rsm_W, factor_ranges, minimize=False)
        if 'error' not in opt_W:
            print(f"\nОптимум W (RSM):")
            print(f"  h = {opt_W['h']*1e6:.1f} мкм")
            print(f"  a = {opt_W['a']*1e6:.1f} мкм")
            print(f"  b = {opt_W['b']*1e6:.1f} мкм")
            print(f"  N_phi = {opt_W['N_phi']:.1f}")
            print(f"  N_z = {opt_W['N_z']:.1f}")
            print(f"  W = {opt_W['response']:.1f} Н")
    else:
        print(f"Ошибка RSM: {rsm_W['error']}")

    # --- Графики ---
    plot_results(valid, suffix)

    # --- Сводка ---
    write_summary(valid, rsm_mu, rsm_W, opt_mu, opt_W, suffix, factor_ranges)

    return rsm_mu, rsm_W


def plot_results(df: pd.DataFrame, mode: str):
    """Построить графики."""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. μ vs глубина h
    ax1 = axes[0, 0]
    scatter1 = ax1.scatter(df['h']*1e6, df['mu_friction'],
                           c=df['N_total'], cmap='viridis', alpha=0.7)
    ax1.set_xlabel('Глубина h, мкм')
    ax1.set_ylabel('Коэффициент трения mu')
    ax1.set_title('mu vs глубина (цвет = N_total)')
    ax1.grid(True, alpha=0.3)
    plt.colorbar(scatter1, ax=ax1, label='N_total')

    # 2. W vs глубина h
    ax2 = axes[0, 1]
    scatter2 = ax2.scatter(df['h']*1e6, df['W'],
                           c=df['fill_factor']*100, cmap='plasma', alpha=0.7)
    ax2.set_xlabel('Глубина h, мкм')
    ax2.set_ylabel('Несущая способность W, Н')
    ax2.set_title('W vs глубина (цвет = fill_factor %)')
    ax2.grid(True, alpha=0.3)
    plt.colorbar(scatter2, ax=ax2, label='fill %')

    # 3. μ vs fill_factor
    ax3 = axes[1, 0]
    ax3.scatter(df['fill_factor']*100, df['mu_friction'],
                c=df['h']*1e6, cmap='coolwarm', alpha=0.7)
    ax3.set_xlabel('Fill factor, %')
    ax3.set_ylabel('Коэффициент трения mu')
    ax3.set_title('mu vs fill_factor (цвет = h мкм)')
    ax3.grid(True, alpha=0.3)

    # 4. μ vs N_total
    ax4 = axes[1, 1]
    ax4.scatter(df['N_total'], df['mu_friction'],
                c=df['a']*1e6, cmap='viridis', alpha=0.7)
    ax4.set_xlabel('Число лунок N_total')
    ax4.set_ylabel('Коэффициент трения mu')
    ax4.set_title('mu vs N_total (цвет = a мкм)')
    ax4.grid(True, alpha=0.3)

    fig.suptitle(f'Параметрический анализ текстуры ({mode})', fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(OUT_DIR / f'parametric_scatter_{mode}.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"График: {OUT_DIR}/parametric_scatter_{mode}.png")

    # --- Дополнительный график: влияние размеров лунки ---
    fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))

    ax1 = axes2[0]
    ax1.scatter(df['a']*1e6, df['mu_friction'], c=df['b']*1e6, cmap='viridis', alpha=0.7)
    ax1.set_xlabel('Полуось a (по z), мкм')
    ax1.set_ylabel('Коэффициент трения mu')
    ax1.set_title('mu vs a (цвет = b мкм)')
    ax1.grid(True, alpha=0.3)

    ax2 = axes2[1]
    ax2.scatter(df['b']*1e6, df['mu_friction'], c=df['a']*1e6, cmap='plasma', alpha=0.7)
    ax2.set_xlabel('Полуось b (по phi), мкм')
    ax2.set_ylabel('Коэффициент трения mu')
    ax2.set_title('mu vs b (цвет = a мкм)')
    ax2.grid(True, alpha=0.3)

    fig2.tight_layout()
    fig2.savefig(OUT_DIR / f'parametric_dimples_{mode}.png', dpi=150, bbox_inches='tight')
    plt.close(fig2)


def write_summary(df: pd.DataFrame, rsm_mu: dict, rsm_W: dict,
                  opt_mu: dict, opt_W: dict, suffix: str,
                  factor_ranges: dict = None):
    """Записать сводку."""

    if factor_ranges is None:
        factor_ranges = FACTOR_RANGES

    with open(OUT_DIR / f'summary_{suffix}.txt', 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("ПАРАМЕТРИЧЕСКИЙ АНАЛИЗ ТЕКСТУРЫ — СВОДКА\n")
        f.write("=" * 60 + "\n\n")

        f.write("ПОДШИПНИК\n")
        f.write("-" * 40 + "\n")
        f.write(f"D = {BEARING_CONFIG.R*2*1000:.1f} мм\n")
        f.write(f"B = {BEARING_CONFIG.B*1000:.1f} мм\n")
        f.write(f"c = {BEARING_CONFIG.c*1e6:.0f} мкм\n")
        f.write(f"epsilon = {BEARING_CONFIG.epsilon}\n")
        f.write(f"n = {BEARING_CONFIG.n_rpm} об/мин\n")
        f.write(f"mu = {BEARING_CONFIG.mu} Па*с\n\n")

        f.write("ДИАПАЗОНЫ ФАКТОРОВ\n")
        f.write("-" * 40 + "\n")
        for name, (lo, hi) in factor_ranges.items():
            if name in ['h', 'a', 'b']:
                f.write(f"{name}: {lo*1e6:.0f} - {hi*1e6:.0f} мкм\n")
            else:
                f.write(f"{name}: {lo:.0f} - {hi:.0f}\n")
        f.write("\n")

        f.write("СТАТИСТИКА РАСЧЁТА\n")
        f.write("-" * 40 + "\n")
        f.write(f"Всего точек ЦКП: {len(df)}\n")
        f.write(f"Время расчёта: см. вывод консоли\n\n")

        f.write("РЕЗУЛЬТАТЫ RSM\n")
        f.write("-" * 40 + "\n")
        if 'error' not in rsm_mu:
            f.write(f"mu_friction: R^2 = {rsm_mu['r2']:.4f}\n")
        if 'error' not in rsm_W:
            f.write(f"W: R^2 = {rsm_W['r2']:.4f}\n")
        f.write("\n")

        if opt_mu and 'error' not in opt_mu:
            f.write("ОПТИМУМ mu_friction (минимум)\n")
            f.write("-" * 40 + "\n")
            f.write(f"h = {opt_mu['h']*1e6:.1f} мкм\n")
            f.write(f"a = {opt_mu['a']*1e6:.1f} мкм\n")
            f.write(f"b = {opt_mu['b']*1e6:.1f} мкм\n")
            f.write(f"N_phi = {opt_mu['N_phi']:.1f}\n")
            f.write(f"N_z = {opt_mu['N_z']:.1f}\n")
            f.write(f"mu_friction = {opt_mu['response']:.6f}\n\n")

        if opt_W and 'error' not in opt_W:
            f.write("ОПТИМУМ W (максимум)\n")
            f.write("-" * 40 + "\n")
            f.write(f"h = {opt_W['h']*1e6:.1f} мкм\n")
            f.write(f"a = {opt_W['a']*1e6:.1f} мкм\n")
            f.write(f"b = {opt_W['b']*1e6:.1f} мкм\n")
            f.write(f"N_phi = {opt_W['N_phi']:.1f}\n")
            f.write(f"N_z = {opt_W['N_z']:.1f}\n")
            f.write(f"W = {opt_W['response']:.1f} Н\n\n")

        f.write("=" * 60 + "\n")

    print(f"Сводка: {OUT_DIR}/summary_{suffix}.txt")


# ============================================================================
# ЛОКАЛЬНАЯ ДОВОДКА
# ============================================================================

def run_refinement(rsm_mu: dict, mode: str):
    """Выполнить локальную доводку оптимума."""

    print("\n" + "=" * 60)
    print("ЛОКАЛЬНАЯ ДОВОДКА")
    print("=" * 60)

    if rsm_mu is None or 'error' in rsm_mu:
        print("RSM не доступна, пропуск.")
        return

    opt_mu = find_optimum(rsm_mu, FACTOR_RANGES, minimize=True)
    if 'error' in opt_mu:
        print(f"Ошибка: {opt_mu['error']}")
        return

    # Выбрать сетку
    if mode == 'lite':
        n_phi, n_z = GRID_LITE
    else:
        n_phi, n_z = GRID_FULL

    refined = local_refinement(
        BEARING_CONFIG, opt_mu, 'mu_friction',
        minimize=True, n_phi_grid=n_phi, n_z_grid=n_z, verbose=True
    )

    if 'error' not in refined:
        print(f"\nЛучший результат после доводки:")
        print(f"  N_phi = {refined['N_phi']}")
        print(f"  N_z = {refined['N_z']}")
        print(f"  mu_friction = {refined['mu_friction']:.6f}")

        # Сохранить
        with open(OUT_DIR / f'refinement_{mode}.txt', 'w', encoding='utf-8') as f:
            f.write("ЛОКАЛЬНАЯ ДОВОДКА\n")
            f.write("=" * 40 + "\n\n")
            f.write(f"h = {refined['h']*1e6:.1f} мкм\n")
            f.write(f"a = {refined['a']*1e6:.1f} мкм\n")
            f.write(f"b = {refined['b']*1e6:.1f} мкм\n")
            f.write(f"N_phi = {refined['N_phi']}\n")
            f.write(f"N_z = {refined['N_z']}\n")
            f.write(f"mu_friction = {refined['mu_friction']:.6f}\n")

        print(f"Сохранено: {OUT_DIR}/refinement_{mode}.txt")


# ============================================================================
# ТЕСТОВЫЙ РАСЧЁТ ОДНОЙ ТОЧКИ
# ============================================================================

def test_single_point():
    """Тестовый расчёт одной точки с диагностикой текстуры."""
    from bearing_solver import BearingConfig, solve_reynolds
    from bearing_solver.parametric import GridTexturedFilmModel

    print("\n" + "=" * 60)
    print("ТЕСТ: РАСЧЁТ ОДНОЙ ТОЧКИ С ДИАГНОСТИКОЙ ТЕКСТУРЫ")
    print("=" * 60)

    # Центральная точка
    factors = TextureFactors(
        h=110e-6,       # 110 мкм
        a=137.5e-6,     # 137.5 мкм
        b=137.5e-6,     # 137.5 мкм
        N_phi=8,
        N_z=4,
    )

    print(f"\nПараметры текстуры:")
    print(f"  h = {factors.h*1e6:.1f} мкм")
    print(f"  a = {factors.a*1e6:.1f} мкм")
    print(f"  b = {factors.b*1e6:.1f} мкм")
    print(f"  N_phi = {factors.N_phi}")
    print(f"  N_z = {factors.N_z}")
    print(f"  N_total = {factors.N_total}")

    gap_phi, gap_z = factors.get_gaps(BEARING_CONFIG.R, BEARING_CONFIG.B)
    print(f"  gap_phi = {gap_phi*1e6:.1f} мкм")
    print(f"  gap_z = {gap_z*1e6:.1f} мкм")
    print(f"  valid = {factors.validate(BEARING_CONFIG.R, BEARING_CONFIG.B)}")
    print(f"  fill_factor = {factors.get_fill_factor(BEARING_CONFIG.R, BEARING_CONFIG.B)*100:.2f}%")

    # Прямая проверка текстуры
    print("\n--- Диагностика текстурного поля (SUBGRID) ---")

    config = BearingConfig(
        R=BEARING_CONFIG.R,
        L=BEARING_CONFIG.B,
        c=BEARING_CONFIG.c,
        epsilon=BEARING_CONFIG.epsilon,
        phi0=np.pi,
        n_rpm=BEARING_CONFIG.n_rpm,
        mu=BEARING_CONFIG.mu,
        n_phi=90,
        n_z=30,
    )

    film_model = GridTexturedFilmModel(config, factors)

    # Получаем сетку
    phi, Z, d_phi, d_Z = config.create_grid()

    # Вычисляем H с текстурой
    H_textured = film_model.H(phi, Z)
    H_smooth = film_model._smooth_model.H(phi, Z)
    texture_field = H_textured - H_smooth

    # Статистика
    print(f"  Сетка: {len(phi)}×{len(Z)} = {len(phi)*len(Z)} ячеек")
    print(f"  d_phi = {np.degrees(d_phi):.2f}°, dz = {d_Z * config.L/2 * 1e3:.3f} мм")
    print(f"  Размер ячейки: {config.R * d_phi * 1e3:.3f} мм × {d_Z * config.L/2 * 1e3:.3f} мм")
    print(f"  Размер лунки: {2*factors.b*1e3:.3f} мм × {2*factors.a*1e3:.3f} мм")
    print(f"  Отношение (лунка/ячейка): {2*factors.b/(config.R * d_phi):.4f} × {2*factors.a/(d_Z * config.L/2):.4f}")
    print(f"  Лунок добавлено: {film_model._n_cells_added}")
    print(f"  dH_max (безразм.): {film_model._dH_max:.6f}")
    print(f"  dH_max (размерн.): {film_model._dH_max * config.c * 1e6:.3f} мкм")
    print(f"  texture_field: min={texture_field.min():.6f}, max={texture_field.max():.6f}")
    print(f"  H_smooth: min={H_smooth.min():.4f}, max={H_smooth.max():.4f}")
    print(f"  H_textured: min={H_textured.min():.4f}, max={H_textured.max():.4f}")

    # Объём одной лунки
    V_dimple = (2.0/3.0) * factors.h * np.pi * factors.a * factors.b
    print(f"\n  Объём одной лунки: V = {V_dimple*1e9:.6f} мм³")

    # Полный расчёт
    print("\n--- Полный расчёт (с поиском phi0) ---")

    result = run_single_case(
        BEARING_CONFIG, factors,
        n_phi_grid=90, n_z_grid=30,
        compute_dynamics=False,
        verbose=True
    )

    print(f"\nРезультат:")
    print(f"  phi0_eq = {np.degrees(result.phi0_equilibrium):.1f} deg")
    print(f"  W = {result.W:.1f} Н")
    print(f"  Fx = {result.Fx:.3f} Н")
    print(f"  Fy = {result.Fy:.1f} Н")
    print(f"  mu_friction = {result.mu_friction:.6f}")
    print(f"  p_max = {result.p_max/1e6:.2f} МПа")
    print(f"  h_min = {result.h_min*1e6:.2f} мкм")
    print(f"  calc_time = {result.calc_time:.1f} s")
    print(f"  converged = {result.converged}")

    # Сравнение с гладкой поверхностью
    print("\n--- Сравнение с гладкой поверхностью ---")
    from bearing_solver import SmoothFilmModel
    from bearing_solver.forces import compute_forces, compute_friction

    config_smooth = BearingConfig(
        R=BEARING_CONFIG.R,
        L=BEARING_CONFIG.B,
        c=BEARING_CONFIG.c,
        epsilon=BEARING_CONFIG.epsilon,
        phi0=result.phi0_equilibrium,
        n_rpm=BEARING_CONFIG.n_rpm,
        mu=BEARING_CONFIG.mu,
        n_phi=90,
        n_z=30,
    )
    sol_smooth = solve_reynolds(config_smooth, film_model=SmoothFilmModel(config_smooth))
    forces_smooth = compute_forces(sol_smooth, config_smooth)
    friction_smooth = compute_friction(sol_smooth, config_smooth, forces_smooth)

    print(f"  Гладкая: W = {forces_smooth.W:.1f} Н, mu = {friction_smooth.mu_friction:.6f}")
    print(f"  С текст: W = {result.W:.1f} Н, mu = {result.mu_friction:.6f}")
    print(f"  Изменение W: {(result.W - forces_smooth.W)/forces_smooth.W*100:+.2f}%")
    print(f"  Изменение mu: {(result.mu_friction - friction_smooth.mu_friction)/friction_smooth.mu_friction*100:+.2f}%")

    return result


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Параметрический анализ текстуры')
    parser.add_argument('--mode', choices=['lite', 'full', 'fine'], default='lite',
                        help='Режим расчёта (lite/full/fine)')
    parser.add_argument('--dynamics', action='store_true',
                        help='Считать K, C (медленнее)')
    parser.add_argument('--test', action='store_true',
                        help='Только тестовый расчёт одной точки')
    parser.add_argument('--refine', action='store_true',
                        help='Локальная доводка после RSM')
    parser.add_argument('--extended', action='store_true',
                        help='Расширенные диапазоны N_phi/N_z для большего эффекта текстуры')
    parser.add_argument('--jobs', type=int, default=1,
                        help='Число параллельных процессов (1 = последовательно)')
    args = parser.parse_args()

    if args.test:
        test_single_point()
        return

    results_df, suffix, factor_ranges = run_study(
        mode=args.mode,
        compute_dynamics=args.dynamics,
        n_jobs=args.jobs,
        extended=args.extended
    )
    rsm_mu, rsm_W = analyze_results(results_df, suffix=suffix, factor_ranges=factor_ranges)

    if args.refine:
        run_refinement(rsm_mu, mode=args.mode)

    print(f"\nГотово! Результаты в {OUT_DIR}")


if __name__ == "__main__":
    main()
