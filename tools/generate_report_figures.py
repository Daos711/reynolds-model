#!/usr/bin/env python3
"""
Генератор графиков для научного отчёта.

Использование:
    python tools/generate_report_figures.py --all      # все этапы
    python tools/generate_report_figures.py --stage 1  # только этап 1
    python tools/generate_report_figures.py --fast     # быстрый режим (меньше точек)

Выходная папка: results/report_figures/
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
import argparse
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bearing_solver import (
    BearingConfig,
    SmoothFilmModel,
    TexturedFilmModel,
    TextureParams,
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
# НАСТРОЙКИ
# ============================================================================

plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'legend.fontsize': 11,
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'savefig.bbox': 'tight',
})

OUT_DIR = Path("results/report_figures")

# Рабочая точка (базовые параметры)
BASE_R = 0.050        # м
BASE_L = 0.050        # м
BASE_C = 50e-6        # м (50 мкм)
BASE_N_RPM = 3000     # об/мин
BASE_W_EXT = 50e3     # Н (50 кН)
BASE_MU = 0.057       # Па·с (T=50°C)
BASE_MASS = 30.0      # кг
BASE_EPSILON = 0.6    # эксцентриситет для статических графиков

# Вязкость по температуре
MU_BY_TEMP = {40: 0.098, 50: 0.057, 60: 0.037, 70: 0.025}

# Режим (глобальная переменная, устанавливается из args)
FAST_MODE = False


def create_config(c=BASE_C, epsilon=BASE_EPSILON, phi0=0.0,
                  n_rpm=BASE_N_RPM, mu=BASE_MU, n_phi=180, n_z=50):
    """Создать конфигурацию подшипника."""
    return BearingConfig(
        R=BASE_R, L=BASE_L, c=c,
        epsilon=epsilon, phi0=phi0,
        n_rpm=n_rpm, mu=mu,
        n_phi=n_phi, n_z=n_z,
    )


def save_fig(fig, filename):
    """Сохранить фигуру."""
    path = OUT_DIR / filename
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ {filename}")


# ============================================================================
# ЭТАП 1: Решатель Рейнольдса
# ============================================================================

def generate_stage1_figures():
    """Этап 1: Решатель Рейнольдса."""
    print("\n" + "=" * 60)
    print("ЭТАП 1: Решатель Рейнольдса")
    print("=" * 60)

    config = create_config()
    film_model = SmoothFilmModel(config)
    result = solve_reynolds(config, film_model)

    phi_deg = np.degrees(result.phi)
    n_z_mid = len(result.Z) // 2

    # fig1_1: Профиль давления P(φ) при Z=0
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(phi_deg, result.P[:, n_z_mid], 'b-', linewidth=2)
    ax.set_xlabel('φ, градусы')
    ax.set_ylabel('P (безразмерное)')
    ax.set_title('Профиль давления P(φ) при Z=0')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 360)
    save_fig(fig, 'fig1_1_pressure_profile.png')

    # fig1_2: 3D поверхность P(φ, Z)
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    PHI, Z = np.meshgrid(phi_deg, result.Z, indexing='ij')
    ax.plot_surface(PHI, Z, result.P, cmap='viridis', alpha=0.8)
    ax.set_xlabel('φ, градусы')
    ax.set_ylabel('Z (безразм.)')
    ax.set_zlabel('P (безразм.)')
    ax.set_title('Поле давления P(φ, Z)')
    save_fig(fig, 'fig1_2_pressure_3d.png')

    # fig1_3: Толщина плёнки H(φ) при Z=0
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(phi_deg, result.H[:, n_z_mid], 'g-', linewidth=2)
    ax.set_xlabel('φ, градусы')
    ax.set_ylabel('H (безразмерное)')
    ax.set_title('Профиль толщины плёнки H(φ) при Z=0')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 360)
    save_fig(fig, 'fig1_3_film_thickness.png')


# ============================================================================
# ЭТАП 2: Силы и трение
# ============================================================================

def generate_stage2_figures():
    """Этап 2: Силы и трение."""
    print("\n" + "=" * 60)
    print("ЭТАП 2: Силы и трение")
    print("=" * 60)

    eps_values = np.linspace(0.1, 0.9, 17 if not FAST_MODE else 9)

    W_list, pmax_list, hmin_list = [], [], []
    f_list, Ploss_list, Q_list = [], [], []

    for eps in eps_values:
        config = create_config(epsilon=eps)
        film_model = SmoothFilmModel(config)
        result = solve_reynolds(config, film_model)
        forces = compute_forces(result, config)
        s2 = compute_stage2(result, config)

        W_list.append(forces.W / 1000)  # кН
        pmax_list.append(result.p_max / 1e6)  # МПа
        hmin_list.append(result.h_min * 1e6)  # мкм
        f_list.append(s2.friction.mu_friction)
        Ploss_list.append(s2.losses.P_friction)  # Вт
        Q_list.append(s2.flow.Q_total * 1e6)  # см³/с

    # fig2_1: W vs ε
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(eps_values, W_list, 'b-o', linewidth=2, markersize=6)
    ax.set_xlabel('ε (эксцентриситет)')
    ax.set_ylabel('W, кН')
    ax.set_title('Несущая способность W(ε)')
    ax.grid(True, alpha=0.3)
    save_fig(fig, 'fig2_1_W_vs_epsilon.png')

    # fig2_2: p_max vs ε
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(eps_values, pmax_list, 'r-o', linewidth=2, markersize=6)
    ax.set_xlabel('ε (эксцентриситет)')
    ax.set_ylabel('p_max, МПа')
    ax.set_title('Максимальное давление p_max(ε)')
    ax.grid(True, alpha=0.3)
    save_fig(fig, 'fig2_2_pmax_vs_epsilon.png')

    # fig2_3: h_min vs ε
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(eps_values, hmin_list, 'g-o', linewidth=2, markersize=6)
    ax.set_xlabel('ε (эксцентриситет)')
    ax.set_ylabel('h_min, мкм')
    ax.set_title('Минимальная толщина плёнки h_min(ε)')
    ax.grid(True, alpha=0.3)
    save_fig(fig, 'fig2_3_hmin_vs_epsilon.png')

    # fig2_4: f vs ε
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(eps_values, f_list, 'm-o', linewidth=2, markersize=6)
    ax.set_xlabel('ε (эксцентриситет)')
    ax.set_ylabel('f (коэффициент трения)')
    ax.set_title('Коэффициент трения f(ε)')
    ax.grid(True, alpha=0.3)
    save_fig(fig, 'fig2_4_friction_vs_epsilon.png')

    # fig2_5: P_loss vs ε
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(eps_values, Ploss_list, 'c-o', linewidth=2, markersize=6)
    ax.set_xlabel('ε (эксцентриситет)')
    ax.set_ylabel('P_loss, Вт')
    ax.set_title('Потери мощности P_loss(ε)')
    ax.grid(True, alpha=0.3)
    save_fig(fig, 'fig2_5_Ploss_vs_epsilon.png')

    # fig2_6: Q vs ε
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(eps_values, Q_list, '-o', linewidth=2, markersize=6, color='orange')
    ax.set_xlabel('ε (эксцентриситет)')
    ax.set_ylabel('Q, см³/с')
    ax.set_title('Расход смазки Q(ε)')
    ax.grid(True, alpha=0.3)
    save_fig(fig, 'fig2_6_Q_vs_epsilon.png')


# ============================================================================
# ЭТАП 3: Поиск равновесия
# ============================================================================

def generate_stage3_figures():
    """Этап 3: Поиск равновесия."""
    print("\n" + "=" * 60)
    print("ЭТАП 3: Поиск равновесия")
    print("=" * 60)

    W_values = np.linspace(10e3, 80e3, 8 if not FAST_MODE else 4)  # 10-80 кН

    eps_list, phi0_list, hmin_list = [], [], []

    for W_ext in W_values:
        config = create_config(epsilon=0.5)  # начальное
        try:
            eq = find_equilibrium(
                config, W_ext=W_ext, load_angle=-np.pi/2,
                verbose=False,
                film_model_factory=lambda cfg: SmoothFilmModel(cfg)
            )
            eps_list.append(eq.epsilon)
            phi0_list.append(np.degrees(eq.phi0))
            hmin_list.append(eq.stage2.reynolds.h_min * 1e6 if eq.stage2 else np.nan)
        except Exception as e:
            print(f"  W={W_ext/1000:.0f}kN failed: {e}")
            eps_list.append(np.nan)
            phi0_list.append(np.nan)
            hmin_list.append(np.nan)

    W_kN = W_values / 1000

    # fig3_1: ε vs W
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(W_kN, eps_list, 'b-o', linewidth=2, markersize=8)
    ax.set_xlabel('W_ext, кН')
    ax.set_ylabel('ε (эксцентриситет)')
    ax.set_title('Эксцентриситет равновесия ε(W)')
    ax.grid(True, alpha=0.3)
    save_fig(fig, 'fig3_1_epsilon_vs_W.png')

    # fig3_2: φ₀ vs W
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(W_kN, phi0_list, 'r-o', linewidth=2, markersize=8)
    ax.set_xlabel('W_ext, кН')
    ax.set_ylabel('φ₀, градусы')
    ax.set_title('Угол линии центров φ₀(W)')
    ax.grid(True, alpha=0.3)
    save_fig(fig, 'fig3_2_phi0_vs_W.png')

    # fig3_3: h_min vs W
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(W_kN, hmin_list, 'g-o', linewidth=2, markersize=8)
    ax.set_xlabel('W_ext, кН')
    ax.set_ylabel('h_min, мкм')
    ax.set_title('Минимальная толщина плёнки h_min(W)')
    ax.grid(True, alpha=0.3)
    save_fig(fig, 'fig3_3_hmin_vs_W.png')


# ============================================================================
# ЭТАП 4: Динамические коэффициенты
# ============================================================================

def generate_stage4_figures():
    """Этап 4: Динамические коэффициенты."""
    print("\n" + "=" * 60)
    print("ЭТАП 4: Динамические коэффициенты")
    print("=" * 60)

    eps_values = np.linspace(0.2, 0.8, 7 if not FAST_MODE else 4)

    Kxx_list, Kxy_list, Kyx_list, Kyy_list = [], [], [], []
    Cxx_list, Cxy_list, Cyx_list, Cyy_list = [], [], [], []

    base_config = create_config(epsilon=0.5)  # базовая конфигурация
    phi0 = 0.0  # стандартное положение (h_min при φ=π)

    for eps in eps_values:
        try:
            dyn = compute_dynamic_coefficients(
                base_config, epsilon=eps, phi0=phi0,
                film_model_factory=lambda cfg: SmoothFilmModel(cfg)
            )

            # Жёсткости (МН/м)
            Kxx_list.append(dyn.Kxx / 1e6)
            Kxy_list.append(dyn.Kxy / 1e6)
            Kyx_list.append(dyn.Kyx / 1e6)
            Kyy_list.append(dyn.Kyy / 1e6)

            # Демпфирование (кН·с/м)
            Cxx_list.append(dyn.Cxx / 1e3)
            Cxy_list.append(dyn.Cxy / 1e3)
            Cyx_list.append(dyn.Cyx / 1e3)
            Cyy_list.append(dyn.Cyy / 1e3)

        except Exception as e:
            print(f"  ε={eps:.2f} failed: {e}")
            for lst in [Kxx_list, Kxy_list, Kyx_list, Kyy_list,
                        Cxx_list, Cxy_list, Cyx_list, Cyy_list]:
                lst.append(np.nan)

    # fig4_1: K vs ε
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(eps_values, Kxx_list, 'b-o', label='K_xx', linewidth=2)
    ax.plot(eps_values, Kxy_list, 'r-s', label='K_xy', linewidth=2)
    ax.plot(eps_values, Kyx_list, 'g-^', label='K_yx', linewidth=2)
    ax.plot(eps_values, Kyy_list, 'm-d', label='K_yy', linewidth=2)
    ax.set_xlabel('ε (эксцентриситет)')
    ax.set_ylabel('K, МН/м')
    ax.set_title('Коэффициенты жёсткости K(ε)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    save_fig(fig, 'fig4_1_K_vs_epsilon.png')

    # fig4_2: C vs ε
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(eps_values, Cxx_list, 'b-o', label='C_xx', linewidth=2)
    ax.plot(eps_values, Cxy_list, 'r-s', label='C_xy', linewidth=2)
    ax.plot(eps_values, Cyx_list, 'g-^', label='C_yx', linewidth=2)
    ax.plot(eps_values, Cyy_list, 'm-d', label='C_yy', linewidth=2)
    ax.set_xlabel('ε (эксцентриситет)')
    ax.set_ylabel('C, кН·с/м')
    ax.set_title('Коэффициенты демпфирования C(ε)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    save_fig(fig, 'fig4_2_C_vs_epsilon.png')


# ============================================================================
# ЭТАП 5: Устойчивость
# ============================================================================

def generate_stage5_figures():
    """Этап 5: Устойчивость."""
    print("\n" + "=" * 60)
    print("ЭТАП 5: Устойчивость")
    print("=" * 60)

    n_values = np.linspace(1000, 5000, 9 if not FAST_MODE else 5)

    margin_list, gamma_list = [], []
    eigenvalues_all = []

    for n_rpm in n_values:
        config = create_config(n_rpm=n_rpm, epsilon=0.5)

        try:
            # Находим равновесие для данной скорости
            eq = find_equilibrium(
                config, W_ext=BASE_W_EXT, load_angle=-np.pi/2,
                verbose=False,
                film_model_factory=lambda cfg: SmoothFilmModel(cfg)
            )

            # Динамические коэффициенты
            dyn = compute_dynamic_coefficients(
                config, epsilon=eq.epsilon, phi0=eq.phi0,
                film_model_factory=lambda cfg: SmoothFilmModel(cfg)
            )

            # Анализ устойчивости
            K = np.array([[dyn.Kxx, dyn.Kxy], [dyn.Kyx, dyn.Kyy]])
            C = np.array([[dyn.Cxx, dyn.Cxy], [dyn.Cyx, dyn.Cyy]])
            stab = analyze_stability(K, C, BASE_MASS, n_rpm)

            margin_list.append(stab.stability_margin)
            gamma_list.append(stab.whirl_ratio)
            eigenvalues_all.append(stab.eigenvalues)

        except Exception as e:
            print(f"  n={n_rpm:.0f}rpm failed: {e}")
            margin_list.append(np.nan)
            gamma_list.append(np.nan)
            eigenvalues_all.append([])

    # fig5_1: margin vs n
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(n_values, margin_list, 'b-o', linewidth=2, markersize=8)
    ax.axhline(y=0, color='r', linestyle='--', alpha=0.7, label='Граница устойчивости')
    ax.set_xlabel('n, об/мин')
    ax.set_ylabel('Запас устойчивости, 1/с')
    ax.set_title('Запас устойчивости margin(n)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    save_fig(fig, 'fig5_1_margin_vs_n.png')

    # fig5_2: γ vs n
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(n_values, gamma_list, 'g-o', linewidth=2, markersize=8)
    ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.7, label='γ = 0.5 (критическое)')
    ax.set_xlabel('n, об/мин')
    ax.set_ylabel('γ (whirl ratio)')
    ax.set_title('Whirl ratio γ(n)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    save_fig(fig, 'fig5_2_whirl_vs_n.png')

    # fig5_3: Собственные значения на комплексной плоскости
    fig, ax = plt.subplots(figsize=(10, 6))

    # Собираем все собственные значения
    for i, (n_rpm, eigs) in enumerate(zip(n_values, eigenvalues_all)):
        if len(eigs) > 0:
            re_parts = [e.real for e in eigs]
            im_parts = [e.imag for e in eigs]
            color = plt.cm.viridis(i / len(n_values))
            ax.scatter(re_parts, im_parts, c=[color]*len(eigs), s=50,
                       label=f'n={n_rpm:.0f}' if i % 2 == 0 else None)

    ax.axvline(x=0, color='r', linestyle='--', alpha=0.7, label='Re(λ) = 0')
    ax.set_xlabel('Re(λ), 1/с')
    ax.set_ylabel('Im(λ), 1/с')
    ax.set_title('Собственные значения системы')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    save_fig(fig, 'fig5_3_eigenvalues.png')


# ============================================================================
# ЭТАП 6: Микрорельеф по ГОСТ
# ============================================================================

def generate_stage6_figures():
    """Этап 6: Микрорельеф по ГОСТ."""
    print("\n" + "=" * 60)
    print("ЭТАП 6: Микрорельеф по ГОСТ")
    print("=" * 60)

    # fig6_1: Схема расположения лунок
    fig, ax = plt.subplots(figsize=(10, 6))

    # Генерируем текстуру для визуализации
    tex_params = TextureParams(
        a=0.5e-3, b=0.5e-3, h_star=0.1, Fn=0.15, pattern='phyllotaxis'
    )
    config = create_config()
    tex_model = TexturedFilmModel(config, tex_params)

    phi = np.linspace(0, 2*np.pi, 360)
    Z = np.linspace(-1, 1, 100)
    PHI, ZZ = np.meshgrid(phi, Z, indexing='ij')

    H_tex = tex_model.H(phi, Z)
    H_smooth = SmoothFilmModel(config).H(phi, Z)
    texture_field = H_tex - H_smooth

    # Показываем текстуру
    c = ax.contourf(np.degrees(PHI), ZZ, texture_field, levels=20, cmap='Blues')
    plt.colorbar(c, ax=ax, label='Глубина лунок (безразм.)')
    ax.set_xlabel('φ, градусы')
    ax.set_ylabel('Z (безразм.)')
    ax.set_title(f'Расположение лунок (паттерн: phyllotaxis, Fn={tex_params.Fn})')
    save_fig(fig, 'fig6_1_texture_pattern.png')

    # Sweep по Fn
    Fn_values = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30] if not FAST_MODE else [0.10, 0.20, 0.30]
    Ploss_list, pmax_list = [], []

    # Находим равновесие для гладкого
    config_base = create_config(epsilon=0.5)
    eq_smooth = find_equilibrium(
        config_base, W_ext=BASE_W_EXT, load_angle=-np.pi/2,
        verbose=False, film_model_factory=lambda cfg: SmoothFilmModel(cfg)
    )

    for Fn in Fn_values:
        tex_params = TextureParams(a=0.5e-3, b=0.5e-3, h_star=0.1, Fn=Fn, pattern='phyllotaxis')
        config = create_config(epsilon=eq_smooth.epsilon, phi0=eq_smooth.phi0)

        tex_model = TexturedFilmModel(config, tex_params)
        result = solve_reynolds(config, tex_model)
        s2 = compute_stage2(result, config)

        Ploss_list.append(s2.losses.P_friction)
        pmax_list.append(result.p_max / 1e6)

    Fn_pct = [f * 100 for f in Fn_values]

    # fig6_2: P_loss vs Fn
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(Fn_pct, Ploss_list, 'b-o', linewidth=2, markersize=8)
    ax.set_xlabel('Fn (площадь покрытия), %')
    ax.set_ylabel('P_loss, Вт')
    ax.set_title('Потери мощности P_loss(Fn)')
    ax.grid(True, alpha=0.3)
    save_fig(fig, 'fig6_2_Ploss_vs_Fn.png')

    # fig6_3: p_max vs Fn
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(Fn_pct, pmax_list, 'r-o', linewidth=2, markersize=8)
    ax.set_xlabel('Fn (площадь покрытия), %')
    ax.set_ylabel('p_max, МПа')
    ax.set_title('Максимальное давление p_max(Fn)')
    ax.grid(True, alpha=0.3)
    save_fig(fig, 'fig6_3_pmax_vs_Fn.png')

    # Sweep по h*
    h_star_values = [0.05, 0.10, 0.15, 0.20, 0.30] if not FAST_MODE else [0.05, 0.15, 0.30]
    Ploss_h_list = []

    for h_star in h_star_values:
        tex_params = TextureParams(a=0.5e-3, b=0.5e-3, h_star=h_star, Fn=0.15, pattern='phyllotaxis')
        config = create_config(epsilon=eq_smooth.epsilon, phi0=eq_smooth.phi0)

        tex_model = TexturedFilmModel(config, tex_params)
        result = solve_reynolds(config, tex_model)
        s2 = compute_stage2(result, config)

        Ploss_h_list.append(s2.losses.P_friction)

    # fig6_4: P_loss vs h*
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(h_star_values, Ploss_h_list, 'g-o', linewidth=2, markersize=8)
    ax.set_xlabel('h* (относительная глубина)')
    ax.set_ylabel('P_loss, Вт')
    ax.set_title('Потери мощности P_loss(h*)')
    ax.grid(True, alpha=0.3)
    save_fig(fig, 'fig6_4_Ploss_vs_depth.png')


# ============================================================================
# ЭТАП 7: Шероховатость Patir-Cheng
# ============================================================================

def generate_stage7_figures():
    """Этап 7: Шероховатость Patir-Cheng."""
    print("\n" + "=" * 60)
    print("ЭТАП 7: Шероховатость Patir-Cheng")
    print("=" * 60)

    # Читаем данные из CSV (если есть)
    csv_path = Path("results/roughness_sandbox/roughness_influence.csv")

    if csv_path.exists():
        print("  Используем данные из roughness_sandbox...")
        df = pd.read_csv(csv_path)
    else:
        print("  CSV не найден, генерируем данные...")
        # Генерируем упрощённые данные
        df = generate_roughness_data()

    # Добавляем вертикальные линии на всех графиках
    def add_lambda_lines(ax):
        ax.axvline(x=1, color='red', linestyle='--', alpha=0.5, label='λ=1')
        ax.axvline(x=3, color='orange', linestyle='--', alpha=0.5, label='λ=3')
        ax.axvline(x=5, color='green', linestyle='--', alpha=0.5, label='λ=5')

    # fig7_1: φx vs λ_min
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {'Low': 'green', 'Mid': 'orange', 'High': 'red'}
    for Ra_level in ['Low', 'Mid', 'High']:
        subset = df[df['Ra_level'] == Ra_level]
        ax.scatter(subset['lambda_min'], subset['phi_x_min'],
                   label=Ra_level, s=50, alpha=0.7, c=colors[Ra_level])
    add_lambda_lines(ax)
    ax.axhline(y=1.0, color='k', linestyle='-', linewidth=0.5)
    ax.set_xlabel('λ_min (параметр плёнки)')
    ax.set_ylabel('φx_min (flow factor)')
    ax.set_title('Flow factor φx(λ_min)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    save_fig(fig, 'fig7_1_phi_x_vs_lambda.png')

    # fig7_2: Δp_max vs λ_min
    fig, ax = plt.subplots(figsize=(10, 6))
    for Ra_level in ['Low', 'Mid', 'High']:
        subset = df[df['Ra_level'] == Ra_level]
        ax.scatter(subset['lambda_min'], subset['dp_max_pct'],
                   label=Ra_level, s=50, alpha=0.7, c=colors[Ra_level])
    add_lambda_lines(ax)
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax.set_xlabel('λ_min')
    ax.set_ylabel('Δp_max, %')
    ax.set_title('Влияние шероховатости на p_max')
    ax.legend()
    ax.grid(True, alpha=0.3)
    save_fig(fig, 'fig7_2_dpmax_vs_lambda.png')

    # fig7_3: ΔP_loss vs λ_min
    fig, ax = plt.subplots(figsize=(10, 6))
    for Ra_level in ['Low', 'Mid', 'High']:
        subset = df[df['Ra_level'] == Ra_level]
        ax.scatter(subset['lambda_min'], subset['dP_loss_pct'],
                   label=Ra_level, s=50, alpha=0.7, c=colors[Ra_level])
    add_lambda_lines(ax)
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax.set_xlabel('λ_min')
    ax.set_ylabel('ΔP_loss, %')
    ax.set_title('Влияние шероховатости на P_loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    save_fig(fig, 'fig7_3_dPloss_vs_lambda.png')

    # fig7_4: ΔP_loss_shear vs λ_min (если есть)
    if 'dP_loss_shear_pct' in df.columns:
        df_shear = df[df['dP_loss_shear_pct'].notna()]
        if len(df_shear) > 0:
            fig, ax = plt.subplots(figsize=(10, 6))
            for Ra_level in ['Low', 'Mid', 'High']:
                subset = df_shear[df_shear['Ra_level'] == Ra_level]
                if len(subset) > 0:
                    ax.scatter(subset['lambda_min'], subset['dP_loss_shear_pct'],
                               label=Ra_level, s=50, alpha=0.7, c=colors[Ra_level])
            add_lambda_lines(ax)
            ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
            ax.set_xlabel('λ_min')
            ax.set_ylabel('ΔP_loss (с shear factor), %')
            ax.set_title('Влияние шероховатости на P_loss с shear factor')
            ax.legend()
            ax.grid(True, alpha=0.3)
            save_fig(fig, 'fig7_4_dPloss_shear_vs_lambda.png')


def generate_roughness_data():
    """Генерация упрощённых данных для этапа 7 (если CSV нет)."""
    # Используем аналитические формулы Patir-Cheng
    lambda_vals = np.linspace(0.5, 15, 30)

    results = []
    for lam in lambda_vals:
        phi_x = 1 - 0.9 * np.exp(-0.56 * max(lam, 1))
        for Ra_level, scale in [('Low', 1.0), ('Mid', 0.3), ('High', 0.1)]:
            results.append({
                'lambda_min': lam * scale if Ra_level != 'Low' else lam,
                'phi_x_min': phi_x if Ra_level == 'Low' else 1 - 0.9 * np.exp(-0.56 * max(lam * scale, 1)),
                'dp_max_pct': max(0, (1/phi_x - 1) * 100) if phi_x > 0.5 else 100,
                'dP_loss_pct': 0.1 * (1 - phi_x) * 100,
                'Ra_level': Ra_level,
            })

    return pd.DataFrame(results)


# ============================================================================
# ЭТАП 8: Параметрика и оптимизация
# ============================================================================

def generate_stage8_figures():
    """Этап 8: Параметрика и оптимизация."""
    print("\n" + "=" * 60)
    print("ЭТАП 8: Параметрика и оптимизация")
    print("=" * 60)

    # Читаем данные из Stage 8 CSV (если есть)
    pareto_path = Path("results/stage8/pareto_top5.csv")
    clearance_path = Path("results/stage8/clearance_sweep.csv")
    robustness_c_path = Path("results/stage8/robustness_clearance.csv")
    robustness_t_path = Path("results/stage8/robustness_temperature.csv")

    # fig8_1: Pareto-фронт
    if pareto_path.exists():
        df_pareto = pd.read_csv(pareto_path)

        # Создаём фигуру с двумя колонками: график + легенда
        fig = plt.figure(figsize=(14, 7))
        gs = fig.add_gridspec(1, 2, width_ratios=[2, 1], wspace=0.05)
        ax = fig.add_subplot(gs[0])
        ax_legend = fig.add_subplot(gs[1])

        # Цвета по паттерну текстуры
        pattern_colors = {
            'spiral': '#1f77b4',      # синий
            'grid': '#2ca02c',         # зелёный
            'phyllotaxis': '#d62728',  # красный
            'hexagonal': '#9467bd',    # фиолетовый
            'random': '#8c564b',       # коричневый
        }

        # Извлекаем паттерн из имени или колонки
        if 'pattern' in df_pareto.columns:
            patterns = df_pareto['pattern'].values
        else:
            patterns = ['unknown'] * len(df_pareto)

        # Рисуем точки с номерами
        for idx, (i, row) in enumerate(df_pareto.iterrows()):
            pattern = patterns[idx] if idx < len(patterns) else 'unknown'
            color = pattern_colors.get(pattern, '#7f7f7f')
            ax.scatter(row['P_loss_W'], row['p_max_MPa'],
                      s=250, c=color, alpha=0.8, edgecolors='black', linewidths=1)
            # Номер внутри точки
            ax.annotate(str(idx + 1), (row['P_loss_W'], row['p_max_MPa']),
                       fontsize=10, fontweight='bold', color='white',
                       ha='center', va='center')

        ax.set_xlabel('P_loss, Вт')
        ax.set_ylabel('p_max, МПа')
        ax.set_title('Pareto-фронт: p_max vs P_loss (топ-5)')
        ax.grid(True, alpha=0.3)

        # Легенда-таблица справа
        ax_legend.axis('off')

        # Формируем текст легенды
        legend_lines = ["№  Конфигурация", "─" * 35]
        for idx, (i, row) in enumerate(df_pareto.iterrows()):
            # Форматируем имя более читабельно
            name = row['name']
            # Парсим параметры из имени
            p_loss = row['P_loss_W']
            p_max = row['p_max_MPa']
            pattern = patterns[idx] if idx < len(patterns) else ''
            legend_lines.append(f"{idx+1:2d}. {name[:25]}")
            legend_lines.append(f"    P_loss={p_loss:.0f} Вт, p_max={p_max:.1f} МПа")

        legend_text = "\n".join(legend_lines)
        ax_legend.text(0.02, 0.98, legend_text, transform=ax_legend.transAxes,
                      fontsize=10, verticalalignment='top', fontfamily='monospace',
                      bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # Легенда по паттернам (цвета)
        unique_patterns = list(set(patterns))
        for i, pat in enumerate(unique_patterns):
            if pat in pattern_colors:
                ax_legend.scatter([], [], c=pattern_colors[pat], s=100,
                                 label=pat, edgecolors='black')
        if unique_patterns:
            ax_legend.legend(loc='lower left', title='Паттерн',
                           bbox_to_anchor=(0.02, 0.02), fontsize=10)

        save_fig(fig, 'fig8_1_pareto.png')
    else:
        print("  Нет данных для Pareto (pareto_top5.csv)")

    # fig8_2: ΔP_loss vs c (робастность по зазору)
    if robustness_c_path.exists():
        df_c = pd.read_csv(robustness_c_path)

        fig, ax = plt.subplots(figsize=(10, 6))

        # Группируем по имени
        for name in df_c['name'].unique():
            subset = df_c[df_c['name'] == name]
            label = 'Гладкий' if name == 'Гладкий' else name[:20]
            linestyle = '--' if name == 'Гладкий' else '-'
            ax.plot(subset['c_um'], subset['P_loss_W'], linestyle + 'o',
                    label=label, linewidth=2, markersize=6)

        ax.set_xlabel('c, мкм')
        ax.set_ylabel('P_loss, Вт')
        ax.set_title('Робастность по зазору: P_loss(c)')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        save_fig(fig, 'fig8_2_dPloss_vs_c.png')
    else:
        print("  Нет данных для робастности по c")

    # fig8_3: ΔP_loss vs T (робастность по температуре)
    if robustness_t_path.exists():
        df_t = pd.read_csv(robustness_t_path)

        fig, ax = plt.subplots(figsize=(10, 6))

        for name in df_t['name'].unique():
            subset = df_t[df_t['name'] == name]
            label = 'Гладкий' if name == 'Гладкий' else name[:20]
            linestyle = '--' if name == 'Гладкий' else '-'
            ax.plot(subset['T_C'], subset['P_loss_W'], linestyle + 'o',
                    label=label, linewidth=2, markersize=6)

        ax.set_xlabel('T, °C')
        ax.set_ylabel('P_loss, Вт')
        ax.set_title('Робастность по температуре: P_loss(T)')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        save_fig(fig, 'fig8_3_dPloss_vs_T.png')
    else:
        print("  Нет данных для робастности по T")

    # fig8_4: h_min vs c (гладкий vs текстура)
    if robustness_c_path.exists():
        df_c = pd.read_csv(robustness_c_path)

        fig, ax = plt.subplots(figsize=(10, 6))

        for name in df_c['name'].unique():
            subset = df_c[df_c['name'] == name]
            label = 'Гладкий' if name == 'Гладкий' else name[:20]
            linestyle = '--' if name == 'Гладкий' else '-'
            ax.plot(subset['c_um'], subset['h_min_um'], linestyle + 'o',
                    label=label, linewidth=2, markersize=6)

        ax.axhline(y=10, color='r', linestyle=':', alpha=0.7, label='h_min = 10 мкм')
        ax.set_xlabel('c, мкм')
        ax.set_ylabel('h_min, мкм')
        ax.set_title('Минимальная толщина плёнки h_min(c)')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        save_fig(fig, 'fig8_4_hmin_vs_c.png')
    else:
        print("  Нет данных для h_min(c)")


# ============================================================================
# MAIN
# ============================================================================

def main():
    global FAST_MODE

    parser = argparse.ArgumentParser(description='Генерация графиков для отчёта')
    parser.add_argument('--stage', type=int, help='Номер этапа (1-8)')
    parser.add_argument('--all', action='store_true', help='Все этапы')
    parser.add_argument('--fast', action='store_true', help='Быстрый режим (меньше точек)')
    args = parser.parse_args()

    FAST_MODE = args.fast

    print("=" * 60)
    print("Генератор графиков для научного отчёта")
    print("=" * 60)
    print(f"Режим: {'быстрый' if FAST_MODE else 'полный'}")
    print(f"Выходная папка: {OUT_DIR}")
    print()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    stage_funcs = {
        1: generate_stage1_figures,
        2: generate_stage2_figures,
        3: generate_stage3_figures,
        4: generate_stage4_figures,
        5: generate_stage5_figures,
        6: generate_stage6_figures,
        7: generate_stage7_figures,
        8: generate_stage8_figures,
    }

    if args.stage:
        if args.stage in stage_funcs:
            stage_funcs[args.stage]()
        else:
            print(f"Неверный номер этапа: {args.stage}")
            return
    else:
        # По умолчанию все этапы
        for i in range(1, 9):
            try:
                stage_funcs[i]()
            except Exception as e:
                print(f"  Ошибка в этапе {i}: {e}")
                import traceback
                traceback.print_exc()

    print("\n" + "=" * 60)
    print(f"Готово! Графики сохранены в {OUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
