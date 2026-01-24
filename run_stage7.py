#!/usr/bin/env python3
"""
Этап 7: Шероховатость Patir-Cheng (flow factors).

Тесты:
1. Сходимость к гладкой модели при Ra→0
2. Тренды по Ra_shaft
3. Тренды по Ra_cell
4. Сравнение 4-х режимов (главный тест)
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

from bearing_solver import (
    BearingConfig,
    SmoothFilmModel,
    TexturedFilmModel,
    TextureParams,
    RoughnessParams,
    solve_reynolds,
    compute_forces,
    compute_roughness_fields,
    find_equilibrium,
    compute_dynamic_coefficients,
    analyze_stability,
)


def create_test_config(n_phi=180, n_z=50):
    """Создать тестовую конфигурацию подшипника."""
    return BearingConfig(
        R=0.050,        # радиус 50 мм
        L=0.050,        # длина 50 мм
        c=50e-6,        # зазор 50 мкм
        epsilon=0.6,
        phi0=np.radians(45),
        n_rpm=3000,
        mu=0.04,
        n_phi=n_phi,
        n_z=n_z,
    )


def solve_with_roughness(config, film_model, roughness_params, texture_mask=None):
    """
    Решить уравнение Рейнольдса с учётом шероховатости.

    Args:
        config: конфигурация подшипника
        film_model: модель плёнки
        roughness_params: параметры шероховатости
        texture_mask: маска текстуры (опционально)

    Returns:
        reynolds_result, roughness_result
    """
    # Сначала вычисляем H на сетке
    phi, Z, _, _ = config.create_grid()
    H = film_model.H(phi, Z)

    # Вычисляем поля шероховатости
    roughness_result = compute_roughness_fields(
        H, phi, Z, config.c,
        roughness_params, texture_mask
    )

    # Решаем с flow factors
    reynolds_result = solve_reynolds(
        config, film_model,
        phi_x=roughness_result.phi_x,
        phi_z=roughness_result.phi_z,
        sigma_star=roughness_result.sigma_star,
        lambda_field=roughness_result.lambda_field,
        frac_lambda_lt_1=roughness_result.frac_lambda_lt_1,
    )

    return reynolds_result, roughness_result


def run_convergence_test():
    """
    Тест 1: При Ra→0 должно совпадать с гладкой моделью.
    """
    print("=" * 60)
    print("ТЕСТ 1: Сходимость к гладкой модели при Ra→0")
    print("=" * 60)

    config = create_test_config(n_phi=120, n_z=30)
    film_model = SmoothFilmModel(config)

    # Гладкая модель (эталон)
    reynolds_smooth = solve_reynolds(config, film_model)
    forces_smooth = compute_forces(reynolds_smooth, config)

    print(f"\nГладкая модель (эталон):")
    print(f"  W = {forces_smooth.W/1000:.2f} кН")
    print(f"  p_max = {reynolds_smooth.p_max/1e6:.2f} МПа")

    # С шероховатостью Ra=0 (все компоненты)
    roughness_params = RoughnessParams(
        Ra_shaft=0.0,
        Ra_out=0.0,
        Ra_cell=0.0,
    )

    reynolds_rough, rough_result = solve_with_roughness(
        config, film_model, roughness_params
    )
    forces_rough = compute_forces(reynolds_rough, config)

    print(f"\nС шероховатостью Ra=0:")
    print(f"  W = {forces_rough.W/1000:.2f} кН")
    print(f"  p_max = {reynolds_rough.p_max/1e6:.2f} МПа")
    print(f"  frac(λ<1) = {rough_result.frac_lambda_lt_1:.2%}")

    # Проверка
    W_diff = abs(forces_rough.W - forces_smooth.W) / forces_smooth.W * 100
    p_diff = abs(reynolds_rough.p_max - reynolds_smooth.p_max) / reynolds_smooth.p_max * 100

    print(f"\nРазница:")
    print(f"  ΔW = {W_diff:.2f}%")
    print(f"  Δp_max = {p_diff:.2f}%")

    passed = W_diff < 0.5 and p_diff < 0.5
    status = "OK" if passed else "FAIL"
    print(f"\nРезультат: [{status}]")

    return passed


def run_Ra_shaft_study(plot=True):
    """
    Тест 2: Влияние Ra_shaft при фиксированных Ra_out, Ra_cell.
    """
    print("\n" + "=" * 60)
    print("ТЕСТ 2: Влияние Ra_shaft")
    print("=" * 60)
    print("Фикс: Ra_out=1.25 мкм, Ra_cell=0.63 мкм")

    config = create_test_config(n_phi=120, n_z=30)
    film_model = SmoothFilmModel(config)

    Ra_shaft_values = [0, 0.5e-6, 0.63e-6, 0.8e-6]
    results = []

    for Ra_shaft in Ra_shaft_values:
        roughness_params = RoughnessParams(
            Ra_shaft=Ra_shaft,
            Ra_out=1.25e-6,
            Ra_cell=0.63e-6,
        )

        reynolds, rough = solve_with_roughness(config, film_model, roughness_params)
        forces = compute_forces(reynolds, config)

        results.append({
            'Ra_shaft_um': Ra_shaft * 1e6,
            'W': forces.W,
            'p_max': reynolds.p_max,
            'h_min': reynolds.h_min,
            'frac_lambda_lt_1': rough.frac_lambda_lt_1,
            'lambda_min': rough.lambda_min,
        })

        print(f"Ra_shaft={Ra_shaft*1e6:.2f} мкм: W={forces.W/1000:.2f} кН, "
              f"p_max={reynolds.p_max/1e6:.2f} МПа, "
              f"frac(λ<1)={rough.frac_lambda_lt_1:.1%}")

    # Сохраняем CSV
    out_dir = Path("results/stage7")
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(results)
    df.to_csv(out_dir / 'roughness_shaft_study.csv', index=False)
    print(f"\nCSV сохранён: {out_dir / 'roughness_shaft_study.csv'}")

    if plot:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        Ra_arr = [r['Ra_shaft_um'] for r in results]

        ax1 = axes[0]
        ax1.plot(Ra_arr, [r['W']/1000 for r in results], 'o-', color='blue')
        ax1.set_xlabel('Ra_shaft, мкм')
        ax1.set_ylabel('W, кН')
        ax1.set_title('Несущая способность vs Ra_shaft')
        ax1.grid(True, alpha=0.3)

        ax2 = axes[1]
        ax2.plot(Ra_arr, [r['p_max']/1e6 for r in results], 's-', color='red')
        ax2.set_xlabel('Ra_shaft, мкм')
        ax2.set_ylabel('p_max, МПа')
        ax2.set_title('Максимальное давление vs Ra_shaft')
        ax2.grid(True, alpha=0.3)

        ax3 = axes[2]
        ax3.plot(Ra_arr, [r['frac_lambda_lt_1']*100 for r in results], '^-', color='orange')
        ax3.set_xlabel('Ra_shaft, мкм')
        ax3.set_ylabel('frac(λ<1), %')
        ax3.set_title('Доля узлов λ<1 vs Ra_shaft')
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(out_dir / 'Ra_shaft_study.png', dpi=150)
        print(f"График сохранён: {out_dir / 'Ra_shaft_study.png'}")
        plt.close()

    return results


def run_Ra_cell_study(plot=True):
    """
    Тест 3: Влияние Ra_cell при фиксированных Ra_shaft, Ra_out.
    """
    print("\n" + "=" * 60)
    print("ТЕСТ 3: Влияние Ra_cell")
    print("=" * 60)
    print("Фикс: Ra_shaft=0.63 мкм, Ra_out=1.25 мкм")

    config = create_test_config(n_phi=120, n_z=30)

    # Используем текстурированную модель для texture_mask
    tex_params = TextureParams(
        a=0.5e-3, b=0.5e-3, h_depth=5e-6, Fn=0.15
    )
    tex_model = TexturedFilmModel(config, tex_params)

    # Получаем маску текстуры
    phi, Z, _, _ = config.create_grid()
    texture_mask = tex_model.texture_mask(phi, Z)

    Ra_cell_values = [0.25e-6, 0.32e-6, 0.5e-6, 0.63e-6, 0.8e-6, 1.0e-6, 1.25e-6]
    results = []

    for Ra_cell in Ra_cell_values:
        roughness_params = RoughnessParams(
            Ra_shaft=0.63e-6,
            Ra_out=1.25e-6,
            Ra_cell=Ra_cell,
        )

        reynolds, rough = solve_with_roughness(
            config, tex_model, roughness_params, texture_mask
        )
        forces = compute_forces(reynolds, config)

        results.append({
            'Ra_cell_um': Ra_cell * 1e6,
            'W': forces.W,
            'p_max': reynolds.p_max,
            'h_min': reynolds.h_min,
            'frac_lambda_lt_1': rough.frac_lambda_lt_1,
            'lambda_min': rough.lambda_min,
        })

        print(f"Ra_cell={Ra_cell*1e6:.2f} мкм: W={forces.W/1000:.2f} кН, "
              f"p_max={reynolds.p_max/1e6:.2f} МПа, "
              f"frac(λ<1)={rough.frac_lambda_lt_1:.1%}")

    # Сохраняем CSV
    out_dir = Path("results/stage7")
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(results)
    df.to_csv(out_dir / 'roughness_cell_study.csv', index=False)
    print(f"\nCSV сохранён: {out_dir / 'roughness_cell_study.csv'}")

    if plot:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        Ra_arr = [r['Ra_cell_um'] for r in results]

        ax1 = axes[0]
        ax1.plot(Ra_arr, [r['W']/1000 for r in results], 'o-', color='blue')
        ax1.set_xlabel('Ra_cell, мкм')
        ax1.set_ylabel('W, кН')
        ax1.set_title('Несущая способность vs Ra_cell')
        ax1.grid(True, alpha=0.3)

        ax2 = axes[1]
        ax2.plot(Ra_arr, [r['p_max']/1e6 for r in results], 's-', color='red')
        ax2.set_xlabel('Ra_cell, мкм')
        ax2.set_ylabel('p_max, МПа')
        ax2.set_title('Максимальное давление vs Ra_cell')
        ax2.grid(True, alpha=0.3)

        ax3 = axes[2]
        ax3.plot(Ra_arr, [r['frac_lambda_lt_1']*100 for r in results], '^-', color='orange')
        ax3.set_xlabel('Ra_cell, мкм')
        ax3.set_ylabel('frac(λ<1), %')
        ax3.set_title('Доля узлов λ<1 vs Ra_cell')
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(out_dir / 'Ra_cell_study.png', dpi=150)
        print(f"График сохранён: {out_dir / 'Ra_cell_study.png'}")
        plt.close()

    return results


def run_four_modes_comparison(W_ext=50e3, mass=30.0, plot=True):
    """
    Тест 4: Сравнение 4-х режимов.

    A) Гладкий без текстуры, без шероховатости
    B) Гладкий + шероховатость
    C) Текстура ГОСТ без шероховатости
    D) Текстура ГОСТ + шероховатость
    """
    print("\n" + "=" * 60)
    print("ТЕСТ 4: Сравнение 4-х режимов")
    print("=" * 60)
    print(f"W_ext = {W_ext/1000:.0f} кН, mass = {mass:.0f} кг")
    print("\nРежимы:")
    print("  A) Гладкий, без шероховатости")
    print("  B) Гладкий + шероховатость")
    print("  C) Текстура ГОСТ, без шероховатости")
    print("  D) Текстура ГОСТ + шероховатость")

    base_config = create_test_config(n_phi=180, n_z=50)
    n_rpm = base_config.n_rpm

    # Параметры текстуры
    tex_params = TextureParams(
        a=0.5e-3, b=0.5e-3, h_depth=5e-6, Fn=0.15
    )

    # Параметры шероховатости
    roughness_params = RoughnessParams(
        Ra_shaft=0.63e-6,
        Ra_out=1.25e-6,
        Ra_cell=0.63e-6,
    )

    # Фабрики моделей
    def smooth_factory(cfg):
        return SmoothFilmModel(cfg)

    def texture_factory(cfg):
        return TexturedFilmModel(cfg, tex_params)

    # 4 режима
    modes = {
        'A': ('Гладкий', smooth_factory, False),
        'B': ('Гладкий+Ra', smooth_factory, True),
        'C': ('Текстура', texture_factory, False),
        'D': ('Текстура+Ra', texture_factory, True),
    }

    results = {}

    for mode_key, (mode_name, factory, use_rough) in modes.items():
        print(f"\n--- Режим {mode_key}: {mode_name} ---")

        # Этап 3: Равновесие
        # Для режимов с шероховатостью нужно модифицировать find_equilibrium
        # Пока используем без шероховатости для поиска равновесия
        eq = find_equilibrium(
            base_config, W_ext=W_ext, load_angle=-np.pi/2,
            verbose=False, film_model_factory=factory
        )
        print(f"  Этап 3: ε = {eq.epsilon:.4f}, φ₀ = {np.degrees(eq.phi0):.1f}°")

        # Создаём конфиг для равновесной позиции
        eq_config = BearingConfig(
            R=base_config.R, L=base_config.L, c=base_config.c,
            epsilon=eq.epsilon, phi0=eq.phi0,
            n_rpm=base_config.n_rpm, mu=base_config.mu,
            n_phi=180, n_z=50
        )

        film_model = factory(eq_config)

        # Этап 2: Решение Рейнольдса (с/без шероховатости)
        if use_rough:
            # Получаем texture_mask если модель текстурированная
            phi, Z, _, _ = eq_config.create_grid()
            if hasattr(film_model, 'texture_mask'):
                texture_mask = film_model.texture_mask(phi, Z)
            else:
                texture_mask = None

            reynolds, rough = solve_with_roughness(
                eq_config, film_model, roughness_params, texture_mask
            )
            frac_lt_1 = rough.frac_lambda_lt_1
        else:
            reynolds = solve_reynolds(eq_config, film_model)
            frac_lt_1 = 0.0

        forces = compute_forces(reynolds, eq_config)
        print(f"  Этап 2: W = {forces.W/1000:.2f} кН, p_max = {reynolds.p_max/1e6:.2f} МПа, "
              f"h_min = {reynolds.h_min*1e6:.2f} мкм")
        if use_rough:
            print(f"          frac(λ<1) = {frac_lt_1:.1%}")

        # Этап 4: K, C
        coeffs = compute_dynamic_coefficients(
            base_config, eq.epsilon, eq.phi0,
            delta_e=0.01, delta_v_star=0.01,
            n_phi=180, n_z=50, verbose=False,
            film_model_factory=factory
        )
        print(f"  Этап 4: Kxx = {coeffs.Kxx/1e6:.0f} МН/м, Cxx = {coeffs.Cxx/1e3:.0f} кН·с/м")

        # Этап 5: Устойчивость
        stab = analyze_stability(coeffs.K, coeffs.C, mass, n_rpm)
        status = "УСТОЙЧИВО" if stab.is_stable else "НЕУСТОЙЧИВО"
        print(f"  Этап 5: {status}, margin = {stab.stability_margin:.1f} 1/с, γ = {stab.whirl_ratio:.3f}")

        results[mode_key] = {
            'mode': mode_name,
            'epsilon': eq.epsilon,
            'phi0_deg': np.degrees(eq.phi0),
            'W': forces.W,
            'p_max': reynolds.p_max,
            'h_min': reynolds.h_min,
            'Kxx': coeffs.Kxx,
            'Kyy': coeffs.Kyy,
            'Cxx': coeffs.Cxx,
            'Cyy': coeffs.Cyy,
            'margin': stab.stability_margin,
            'whirl_ratio': stab.whirl_ratio,
            'is_stable': stab.is_stable,
            'frac_lambda_lt_1': frac_lt_1,
        }

    # Сравнительная таблица
    print("\n" + "=" * 60)
    print("СРАВНИТЕЛЬНАЯ ТАБЛИЦА")
    print("=" * 60)

    params = [
        ('epsilon', '', 1, 4),
        ('h_min', 'мкм', 1e6, 2),
        ('p_max', 'МПа', 1e-6, 2),
        ('Kxx', 'МН/м', 1e-6, 0),
        ('Cxx', 'кН·с/м', 1e-3, 0),
        ('margin', '1/с', 1, 1),
        ('whirl_ratio', '', 1, 3),
        ('frac_lambda_lt_1', '%', 100, 1),
    ]

    header = f"{'Параметр':<20} {'A':<12} {'B':<12} {'C':<12} {'D':<12}"
    print(header)
    print("-" * len(header))

    for param, unit, scale, decimals in params:
        vals = [results[k][param] * scale for k in ['A', 'B', 'C', 'D']]
        fmt = f"{{:.{decimals}f}}"
        line = f"{param + ' ' + unit:<20} " + " ".join(f"{fmt.format(v):<12}" for v in vals)
        print(line)

    # Сохраняем CSV
    out_dir = Path("results/stage7")
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame([results[k] for k in ['A', 'B', 'C', 'D']])
    df.to_csv(out_dir / 'four_modes_comparison.csv', index=False)
    print(f"\nCSV сохранён: {out_dir / 'four_modes_comparison.csv'}")

    # Визуализация поля λ для режима D
    if plot:
        eq_config_D = BearingConfig(
            R=base_config.R, L=base_config.L, c=base_config.c,
            epsilon=results['D']['epsilon'],
            phi0=np.radians(results['D']['phi0_deg']),
            n_rpm=base_config.n_rpm, mu=base_config.mu,
            n_phi=180, n_z=50
        )
        film_model_D = texture_factory(eq_config_D)
        phi, Z, _, _ = eq_config_D.create_grid()
        H = film_model_D.H(phi, Z)
        texture_mask = film_model_D.texture_mask(phi, Z)

        rough_D = compute_roughness_fields(
            H, phi, Z, eq_config_D.c, roughness_params, texture_mask
        )

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        PHI, ZZ = np.meshgrid(np.degrees(phi), Z, indexing='ij')

        # λ field
        ax1 = axes[0]
        c1 = ax1.contourf(PHI, ZZ, rough_D.lambda_field, levels=50, cmap='viridis')
        plt.colorbar(c1, ax=ax1, label='λ = H/σ*')
        ax1.set_xlabel('φ, °')
        ax1.set_ylabel('Z')
        ax1.set_title(f'Параметр плёнки λ (режим D)\nfrac(λ<1) = {rough_D.frac_lambda_lt_1:.1%}')

        # φ_x field
        ax2 = axes[1]
        c2 = ax2.contourf(PHI, ZZ, rough_D.phi_x, levels=50, cmap='RdYlGn')
        plt.colorbar(c2, ax=ax2, label='φ_x')
        ax2.set_xlabel('φ, °')
        ax2.set_ylabel('Z')
        ax2.set_title('Flow factor φ_x (режим D)')

        # σ* field
        ax3 = axes[2]
        c3 = ax3.contourf(PHI, ZZ, rough_D.sigma_star, levels=50, cmap='hot')
        plt.colorbar(c3, ax=ax3, label='σ*')
        ax3.set_xlabel('φ, °')
        ax3.set_ylabel('Z')
        ax3.set_title('Безразмерная шероховатость σ* (режим D)')

        plt.tight_layout()
        plt.savefig(out_dir / 'roughness_fields_mode_D.png', dpi=150)
        print(f"График сохранён: {out_dir / 'roughness_fields_mode_D.png'}")
        plt.close()

    return results


def main():
    print("ЭТАП 7: Шероховатость Patir-Cheng (flow factors)")
    print("=" * 60)
    print("ВАЖНО: φ_s = 0 (shear flow factor отключён)")
    print("       При λ < 1 используется λ_eff = max(λ, 1.0)")
    print("=" * 60)

    out_dir = Path("results/stage7")
    out_dir.mkdir(parents=True, exist_ok=True)

    all_passed = True

    # Тест 1
    passed = run_convergence_test()
    if not passed:
        all_passed = False

    # Тест 2
    run_Ra_shaft_study(plot=True)

    # Тест 3
    run_Ra_cell_study(plot=True)

    # Тест 4
    run_four_modes_comparison(plot=True)

    print("\n" + "=" * 60)
    if all_passed:
        print("ЭТАП 7: ВСЕ ТЕСТЫ ПРОЙДЕНЫ")
    else:
        print("ЭТАП 7: ЕСТЬ ПРОБЛЕМЫ (см. выше)")
    print("=" * 60)


if __name__ == "__main__":
    main()
