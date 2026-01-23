#!/usr/bin/env python3
"""
Этап 6: Микрорельеф по ГОСТ 24773-81.

Тесты:
1. h_depth = 0 → результат совпадает с SmoothFilmModel
2. Визуализация H(φ,Z) — ячейки видны
3. Сравнение гладкий vs текстурированный по Этапам 2-5
4. Влияние параметров текстуры (Fn, h_depth)
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path

from bearing_solver import (
    BearingConfig,
    SmoothFilmModel,
    TexturedFilmModel,
    TextureParams,
    solve_reynolds,
    compute_forces,
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


def run_zero_depth_test():
    """
    Тест 1: h_depth = 0 должен давать тот же результат, что SmoothFilmModel.
    """
    print("=" * 60)
    print("ТЕСТ 1: h_depth = 0 → совпадение с гладкой моделью")
    print("=" * 60)

    config = create_test_config(n_phi=90, n_z=25)

    # Гладкая модель
    smooth_model = SmoothFilmModel(config)
    result_smooth = solve_reynolds(config, film_model=smooth_model)
    forces_smooth = compute_forces(result_smooth, config)

    # Текстурированная с h_depth = 0
    tex_params = TextureParams(h_depth=0.0, Fn=0.15)
    tex_model = TexturedFilmModel(config, tex_params)
    result_tex = solve_reynolds(config, film_model=tex_model)
    forces_tex = compute_forces(result_tex, config)

    print(f"\nГладкая модель:")
    print(f"  W = {forces_smooth.W/1000:.2f} кН")
    print(f"  p_max = {result_smooth.p_max/1e6:.2f} МПа")

    print(f"\nТекстурированная (h_depth=0):")
    print(f"  W = {forces_tex.W/1000:.2f} кН")
    print(f"  p_max = {result_tex.p_max/1e6:.2f} МПа")

    # Проверка
    W_diff = abs(forces_tex.W - forces_smooth.W) / forces_smooth.W * 100
    p_diff = abs(result_tex.p_max - result_smooth.p_max) / result_smooth.p_max * 100

    print(f"\nРазница:")
    print(f"  ΔW = {W_diff:.2f}%")
    print(f"  Δp_max = {p_diff:.2f}%")

    passed = W_diff < 1.0 and p_diff < 1.0
    status = "OK" if passed else "FAIL"
    print(f"\nРезультат: [{status}]")

    return passed


def run_visualization_test(plot=True):
    """
    Тест 2: Визуализация толщины плёнки с текстурой.
    """
    print("\n" + "=" * 60)
    print("ТЕСТ 2: Визуализация H(φ, Z)")
    print("=" * 60)

    config = create_test_config(n_phi=180, n_z=50)

    # Текстурированная модель
    tex_params = TextureParams(
        a=0.5e-3,      # полуось 0.5 мм
        b=0.5e-3,      # полуось 0.5 мм
        h_depth=5e-6,  # глубина 5 мкм
        Fn=0.15,       # 15% заполнение
        pattern='phyllotaxis'
    )
    tex_model = TexturedFilmModel(config, tex_params)

    stats = tex_model.get_texture_stats()
    print(f"\nПараметры текстуры:")
    print(f"  Число лунок N = {stats['N_cells']}")
    print(f"  Полуоси: a = {stats['a_mm']:.2f} мм, b = {stats['b_mm']:.2f} мм")
    print(f"  Глубина: h = {stats['h_depth_um']:.1f} мкм (h* = {stats['h_depth_star']:.4f})")
    print(f"  Fn заданное = {stats['Fn']:.2%}")
    print(f"  Fn фактическое = {stats['Fn_actual']:.2%}")
    print(f"  Паттерн: {stats['pattern']}")

    # Вычисляем поле H
    phi, Z, _, _ = config.create_grid()
    H = tex_model.H(phi, Z)
    H_smooth = SmoothFilmModel(config).H(phi, Z)
    delta_H = H - H_smooth  # только текстура

    print(f"\nСтатистика H:")
    print(f"  H_min = {H.min():.4f}, H_max = {H.max():.4f}")
    print(f"  ΔH_min = {delta_H.min():.6f}, ΔH_max = {delta_H.max():.6f}")

    if plot:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # H (полная толщина)
        ax1 = axes[0, 0]
        PHI, ZZ = np.meshgrid(np.degrees(phi), Z, indexing='ij')
        c1 = ax1.contourf(PHI, ZZ, H, levels=50, cmap='viridis')
        plt.colorbar(c1, ax=ax1, label='H')
        ax1.set_xlabel('φ, °')
        ax1.set_ylabel('Z')
        ax1.set_title('Полная толщина плёнки H(φ, Z)')

        # ΔH (только текстура)
        ax2 = axes[0, 1]
        c2 = ax2.contourf(PHI, ZZ, delta_H, levels=50, cmap='RdBu_r')
        plt.colorbar(c2, ax=ax2, label='ΔH*')
        ax2.set_xlabel('φ, °')
        ax2.set_ylabel('Z')
        ax2.set_title('Текстура ΔH*(φ, Z)')

        # Центры лунок
        ax3 = axes[1, 0]
        ax3.scatter(np.degrees(tex_model.centers_phi),
                   tex_model.centers_z * 1000,
                   s=5, alpha=0.5)
        ax3.set_xlabel('φ, °')
        ax3.set_ylabel('z, мм')
        ax3.set_title(f'Центры лунок (N={stats["N_cells"]})')
        ax3.set_xlim(0, 360)
        ax3.set_ylim(-config.L/2*1000, config.L/2*1000)
        ax3.grid(True, alpha=0.3)

        # Срез H при Z=0
        ax4 = axes[1, 1]
        j_mid = len(Z) // 2
        ax4.plot(np.degrees(phi), H[:, j_mid], 'b-', label='H (текстура)', linewidth=1)
        ax4.plot(np.degrees(phi), H_smooth[:, j_mid], 'r--', label='H (гладкая)', linewidth=1)
        ax4.set_xlabel('φ, °')
        ax4.set_ylabel('H')
        ax4.set_title('Срез H(φ) при Z=0')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        out_dir = Path("results/stage6")
        out_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_dir / 'texture_visualization.png', dpi=150)
        print(f"\nГрафик сохранён: {out_dir / 'texture_visualization.png'}")
        plt.close()

    return True


def run_comparison_test(W_ext=50e3, mass=30.0, plot=True):
    """
    Тест 3: Сравнение гладкий vs текстурированный по всем этапам.

    Режим A: Фиксированное положение (ε, φ₀), смотрим W, p_max
    Режим B: Фиксированная нагрузка W_ext, находим НОВОЕ равновесие
    """
    print("\n" + "=" * 60)
    print("ТЕСТ 3: Сравнение гладкий vs текстурированный")
    print("=" * 60)

    base_config = create_test_config(n_phi=180, n_z=50)
    n_rpm = base_config.n_rpm

    # Параметры текстуры по ГОСТ
    tex_params = TextureParams(
        a=0.5e-3,      # 0.5 мм
        b=0.5e-3,      # 0.5 мм
        h_depth=5e-6,  # 5 мкм
        Fn=0.15,       # 15%
        pattern='phyllotaxis'
    )

    # Фабрики для создания моделей с учётом разных конфигураций (ε, φ₀)
    def smooth_factory(cfg):
        return SmoothFilmModel(cfg)

    def texture_factory(cfg):
        return TexturedFilmModel(cfg, tex_params)

    print("\n--- РЕЖИМ A: Фиксированное положение (ε=0.6, φ₀=45°) ---")
    print("Сравниваем W, p_max при одинаковом (ε, φ₀)")

    results_A = {}
    for name, factory in [('Гладкая', smooth_factory), ('Текстура', texture_factory)]:
        film_model = factory(base_config)
        reynolds = solve_reynolds(base_config, film_model=film_model)
        forces = compute_forces(reynolds, base_config)
        results_A[name] = {'W': forces.W, 'p_max': reynolds.p_max}
        print(f"  {name}: W = {forces.W/1000:.2f} кН, p_max = {reynolds.p_max/1e6:.2f} МПа")

    dW = (results_A['Текстура']['W'] - results_A['Гладкая']['W']) / results_A['Гладкая']['W'] * 100
    dp = (results_A['Текстура']['p_max'] - results_A['Гладкая']['p_max']) / results_A['Гладкая']['p_max'] * 100
    print(f"  Разница: ΔW = {dW:+.1f}%, Δp_max = {dp:+.1f}%")

    print(f"\n--- РЕЖИМ B: Фиксированная нагрузка W_ext = {W_ext/1000:.0f} кН ---")
    print("Для каждой модели находим своё равновесие (ε, φ₀)")

    results_B = {}

    for name, factory in [('Гладкая', smooth_factory), ('Текстура', texture_factory)]:
        print(f"\n{name} модель:")

        # Этап 3: Равновесие с текстурой
        eq = find_equilibrium(
            base_config, W_ext=W_ext, load_angle=-np.pi/2,
            verbose=False, film_model_factory=factory
        )
        print(f"  Этап 3: ε = {eq.epsilon:.4f}, φ₀ = {np.degrees(eq.phi0):.1f}°")

        # Этап 2 при равновесии (для h_min)
        eq_config = BearingConfig(
            R=base_config.R, L=base_config.L, c=base_config.c,
            epsilon=eq.epsilon, phi0=eq.phi0,
            n_rpm=base_config.n_rpm, mu=base_config.mu,
            n_phi=180, n_z=50
        )
        film_model = factory(eq_config)
        reynolds = solve_reynolds(eq_config, film_model=film_model)
        forces = compute_forces(reynolds, eq_config)
        print(f"  Этап 2: W = {forces.W/1000:.2f} кН, p_max = {reynolds.p_max/1e6:.2f} МПа, h_min = {reynolds.h_min*1e6:.2f} мкм")

        # Этап 4: K, C с текстурой
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

        results_B[name] = {
            'W': forces.W,
            'p_max': reynolds.p_max,
            'h_min': reynolds.h_min,
            'epsilon': eq.epsilon,
            'phi0': eq.phi0,
            'Kxx': coeffs.Kxx,
            'Kyy': coeffs.Kyy,
            'Cxx': coeffs.Cxx,
            'Cyy': coeffs.Cyy,
            'margin': stab.stability_margin,
            'whirl_ratio': stab.whirl_ratio,
            'is_stable': stab.is_stable,
        }

    # Сравнение режима B
    print("\n" + "-" * 40)
    print("Сравнение РЕЖИМ B (Текстура vs Гладкая):")

    smooth = results_B['Гладкая']
    tex = results_B['Текстура']

    for param, unit, scale in [
        ('epsilon', '', 1),
        ('h_min', 'мкм', 1e6),
        ('Kxx', 'МН/м', 1e-6),
        ('Kyy', 'МН/м', 1e-6),
        ('Cxx', 'кН·с/м', 1e-3),
        ('Cyy', 'кН·с/м', 1e-3),
        ('margin', '1/с', 1),
    ]:
        v_smooth = smooth[param] * scale
        v_tex = tex[param] * scale
        diff = (v_tex - v_smooth) / abs(v_smooth) * 100 if v_smooth != 0 else 0
        print(f"  {param}: {v_smooth:.4f} → {v_tex:.4f} {unit} ({diff:+.1f}%)")

    return {'A': results_A, 'B': results_B}


def run_fn_study(plot=True):
    """
    Тест 4: Влияние плотности заполнения Fn.
    """
    print("\n" + "=" * 60)
    print("ТЕСТ 4: Влияние плотности заполнения Fn")
    print("=" * 60)

    config = create_test_config(n_phi=120, n_z=30)

    Fn_values = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25]
    results = []

    for Fn in Fn_values:
        if Fn == 0:
            film_model = SmoothFilmModel(config)
            N_cells = 0
        else:
            tex_params = TextureParams(
                a=0.5e-3, b=0.5e-3, h_depth=5e-6, Fn=Fn
            )
            film_model = TexturedFilmModel(config, tex_params)
            N_cells = film_model.N_cells

        reynolds = solve_reynolds(config, film_model=film_model)
        forces = compute_forces(reynolds, config)

        results.append({
            'Fn': Fn,
            'N_cells': N_cells,
            'W': forces.W,
            'p_max': reynolds.p_max,
            'h_min': reynolds.h_min,
        })

        print(f"Fn={Fn:.0%}: N={N_cells:4d}, W={forces.W/1000:.2f} кН, "
              f"p_max={reynolds.p_max/1e6:.2f} МПа, h_min={reynolds.h_min*1e6:.2f} мкм")

    if plot:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        Fn_arr = [r['Fn']*100 for r in results]

        ax1 = axes[0]
        ax1.plot(Fn_arr, [r['W']/1000 for r in results], 'o-', color='blue')
        ax1.set_xlabel('Fn, %')
        ax1.set_ylabel('W, кН')
        ax1.set_title('Несущая способность vs Fn')
        ax1.grid(True, alpha=0.3)

        ax2 = axes[1]
        ax2.plot(Fn_arr, [r['p_max']/1e6 for r in results], 's-', color='red')
        ax2.set_xlabel('Fn, %')
        ax2.set_ylabel('p_max, МПа')
        ax2.set_title('Максимальное давление vs Fn')
        ax2.grid(True, alpha=0.3)

        ax3 = axes[2]
        ax3.plot(Fn_arr, [r['h_min']*1e6 for r in results], '^-', color='green')
        ax3.set_xlabel('Fn, %')
        ax3.set_ylabel('h_min, мкм')
        ax3.set_title('Минимальная толщина плёнки vs Fn')
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()
        out_dir = Path("results/stage6")
        out_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_dir / 'fn_study.png', dpi=150)
        print(f"\nГрафик сохранён: {out_dir / 'fn_study.png'}")
        plt.close()

    return results


def run_depth_study(plot=True):
    """
    Тест 5: Влияние глубины лунок h_depth.
    """
    print("\n" + "=" * 60)
    print("ТЕСТ 5: Влияние глубины лунок h_depth")
    print("=" * 60)

    config = create_test_config(n_phi=120, n_z=30)

    # Глубины в мкм
    h_depths_um = [0, 2, 5, 10, 15, 20]
    results = []

    for h_um in h_depths_um:
        h_depth = h_um * 1e-6

        if h_depth == 0:
            film_model = SmoothFilmModel(config)
        else:
            tex_params = TextureParams(
                a=0.5e-3, b=0.5e-3, h_depth=h_depth, Fn=0.15
            )
            film_model = TexturedFilmModel(config, tex_params)

        reynolds = solve_reynolds(config, film_model=film_model)
        forces = compute_forces(reynolds, config)

        results.append({
            'h_depth_um': h_um,
            'h_depth_star': h_depth / config.c,
            'W': forces.W,
            'p_max': reynolds.p_max,
            'h_min': reynolds.h_min,
        })

        print(f"h={h_um:2d} мкм (h*={h_depth/config.c:.3f}): "
              f"W={forces.W/1000:.2f} кН, p_max={reynolds.p_max/1e6:.2f} МПа")

    if plot:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        h_arr = [r['h_depth_um'] for r in results]

        ax1 = axes[0]
        ax1.plot(h_arr, [r['W']/1000 for r in results], 'o-', color='blue')
        ax1.set_xlabel('h_depth, мкм')
        ax1.set_ylabel('W, кН')
        ax1.set_title('Несущая способность vs глубина лунок')
        ax1.grid(True, alpha=0.3)

        ax2 = axes[1]
        ax2.plot(h_arr, [r['p_max']/1e6 for r in results], 's-', color='red')
        ax2.set_xlabel('h_depth, мкм')
        ax2.set_ylabel('p_max, МПа')
        ax2.set_title('Максимальное давление vs глубина лунок')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        out_dir = Path("results/stage6")
        out_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_dir / 'depth_study.png', dpi=150)
        print(f"\nГрафик сохранён: {out_dir / 'depth_study.png'}")
        plt.close()

    return results


def main():
    parser = argparse.ArgumentParser(description="Этап 6: Микрорельеф ГОСТ 24773-81")
    parser.add_argument("--test", type=int, default=0,
                       help="Номер теста (0=все, 1-5)")
    parser.add_argument("--plot", action="store_true", default=True)
    parser.add_argument("--no-plot", dest="plot", action="store_false")
    args = parser.parse_args()

    print("ЭТАП 6: Микрорельеф по ГОСТ 24773-81")
    print("=" * 60)

    all_passed = True

    if args.test == 0 or args.test == 1:
        passed = run_zero_depth_test()
        if not passed:
            all_passed = False

    if args.test == 0 or args.test == 2:
        run_visualization_test(plot=args.plot)

    if args.test == 0 or args.test == 3:
        run_comparison_test(plot=args.plot)

    if args.test == 0 or args.test == 4:
        run_fn_study(plot=args.plot)

    if args.test == 0 or args.test == 5:
        run_depth_study(plot=args.plot)

    print("\n" + "=" * 60)
    if all_passed:
        print("ЭТАП 6: ВСЕ ТЕСТЫ ПРОЙДЕНЫ")
    else:
        print("ЭТАП 6: ЕСТЬ ПРОБЛЕМЫ (см. выше)")
    print("=" * 60)


if __name__ == "__main__":
    main()
