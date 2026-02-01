#!/usr/bin/env python3
"""
Этап 9: Валидация модели по экспериментальным и численным данным.

Источники:
1. Patel et al. - экспериментальный стенд с ISOVG46 (без магнитного поля)
2. Gwynllyw et al. - численный бенчмарк (constant viscosity, long bearing)

МЕТОДИКА:
- Patel: сравнение p_max на midplane (где стоял датчик) с экспериментом
- Gwynllyw: сравнение p_max_global с оцифрованными данными из графиков
- Grid convergence: фиксируем (ε, φ₀), варьируем только сетку
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

from bearing_solver.validation import (
    PatelCase, GwynllywCase,
    PATEL_EXPERIMENTAL, GWYNLLYW_REFERENCE,
    run_patel_validation,
    run_patel_validation_fixed_position,
    run_gwynllyw_validation,
    create_patel_config,
)
from bearing_solver import solve_reynolds, find_equilibrium

OUT_DIR = Path("results/stage9_validation")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# БЛОК 1: ВАЛИДАЦИЯ ПО PATEL (ISOVG46, без магнитного поля)
# ============================================================================

def run_patel_block():
    """
    Валидация по экспериментальным данным Patel et al.

    ВАЖНО: Сравниваем p_max на MIDPLANE, потому что датчик давления
    в эксперименте был установлен в центральной плоскости.
    """
    print("=" * 60)
    print("БЛОК 1: Валидация по Patel et al.")
    print("(ISOVG46, без магнитного поля)")
    print("=" * 60)

    results = []

    for (n_rpm, W), exp_data in PATEL_EXPERIMENTAL.items():
        print(f"\nРежим: {n_rpm} rpm, {W} N...")

        res = run_patel_validation(n_rpm, W, n_phi=180, n_z=50,
                                    epsilon_bounds=(0.01, 0.98))

        p_max_exp = exp_data["p_max_kPa"]

        # Сравниваем с p_max на midplane (не global!)
        p_max_model_mid = res["p_max_mid_kPa"]
        p_max_model_global = res["p_max_global_kPa"]

        error_mid_pct = abs(p_max_model_mid - p_max_exp) / p_max_exp * 100
        error_global_pct = abs(p_max_model_global - p_max_exp) / p_max_exp * 100

        print(f"  ε = {res['epsilon']:.3f}, φ₀ = {res['phi0_deg']:.1f}°")
        print(f"  p_max_mid:    модель = {p_max_model_mid:.1f} kPa, эксп = {p_max_exp} kPa, ошибка = {error_mid_pct:.1f}%")
        print(f"  p_max_global: модель = {p_max_model_global:.1f} kPa (для справки)")

        results.append({
            "n_rpm": n_rpm,
            "W_N": W,
            "epsilon": res["epsilon"],
            "phi0_deg": res["phi0_deg"],
            "p_max_mid_model_kPa": p_max_model_mid,
            "p_max_global_model_kPa": p_max_model_global,
            "p_max_exp_kPa": p_max_exp,
            "error_mid_pct": error_mid_pct,
            "error_global_pct": error_global_pct,
            "h_min_um": res["h_min_um"],
        })

        # График профиля давления на midplane
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(res["phi_deg"], res["p_Pa"] / 1000, 'b-', linewidth=2, label='Модель (midplane)')
        ax.axhline(y=p_max_exp, color='r', linestyle='--',
                   label=f'Эксперимент p_max = {p_max_exp} kPa')
        ax.set_xlabel("φ, градусы")
        ax.set_ylabel("Давление, кПа")
        ax.set_title(f"Patel (ISOVG46): {n_rpm} rpm, {W} N\nε = {res['epsilon']:.3f}, ошибка = {error_mid_pct:.1f}%")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 360)
        ax.set_ylim(bottom=0)

        fig.savefig(OUT_DIR / f"patel_{n_rpm}rpm_{W}N.png", dpi=150, bbox_inches='tight')
        plt.close(fig)

    df = pd.DataFrame(results)
    df.to_csv(OUT_DIR / "patel_comparison.csv", index=False)

    # Сводный график: модель vs эксперимент (по midplane!)
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(df["p_max_exp_kPa"], df["p_max_mid_model_kPa"], s=100, c='blue', label='p_max midplane')
    ax.scatter(df["p_max_exp_kPa"], df["p_max_global_model_kPa"], s=60, c='green',
               marker='^', alpha=0.5, label='p_max global (для справки)')

    max_p = max(df["p_max_exp_kPa"].max(), df["p_max_mid_model_kPa"].max()) * 1.1
    ax.plot([0, max_p], [0, max_p], 'k--', label='Идеальное совпадение')
    ax.fill_between([0, max_p], [0, 0.7*max_p], [0, 1.3*max_p],
                    alpha=0.1, color='green', label='±30%')

    for _, row in df.iterrows():
        ax.annotate(f"{int(row['n_rpm'])}/{int(row['W_N'])}",
                    (row["p_max_exp_kPa"], row["p_max_mid_model_kPa"]),
                    textcoords="offset points", xytext=(5, 5), fontsize=8)

    ax.set_xlabel("p_max эксперимент (ISOVG46), кПа")
    ax.set_ylabel("p_max модель, кПа")
    ax.set_title("Валидация по Patel: модель vs эксперимент\n(сравнение по midplane)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, max_p)
    ax.set_ylim(0, max_p)

    fig.savefig(OUT_DIR / "patel_summary.png", dpi=150, bbox_inches='tight')
    plt.close(fig)

    return df


# ============================================================================
# БЛОК 2: ВАЛИДАЦИЯ ПО GWYNLLYW (constant viscosity, long bearing)
# ============================================================================

def run_gwynllyw_block():
    """
    Валидация по численному бенчмарку Gwynllyw et al.

    ВАЖНО: Сравниваем с данными для CONSTANT VISCOSITY case (линия b),
    а не piezoviscous. Референсные значения оцифрованы с графиков.
    """
    print("\n" + "=" * 60)
    print("БЛОК 2: Валидация по Gwynllyw et al.")
    print("(constant viscosity, long bearing L/D=4)")
    print("=" * 60)

    results = []

    for epsilon, ref_data in GWYNLLYW_REFERENCE.items():
        print(f"\nЭксцентриситет: ε = {epsilon}...")

        res = run_gwynllyw_validation(epsilon, n_phi=180, n_z=50)

        p_max_ref = ref_data["p_max_MPa"]
        p_max_model = res["p_max_global_MPa"]  # для long bearing mid ≈ global
        error_pct = abs(p_max_model - p_max_ref) / p_max_ref * 100

        print(f"  p_max: модель = {p_max_model:.1f} MPa, референс = {p_max_ref} MPa")
        print(f"  Ошибка: {error_pct:.1f}%")
        print(f"  (референс оцифрован с {ref_data['description']})")

        results.append({
            "epsilon": epsilon,
            "p_max_model_MPa": p_max_model,
            "p_max_ref_MPa": p_max_ref,
            "error_pct": error_pct,
            "h_min_um": res["h_min_um"],
            "source": ref_data["description"],
        })

        # График профиля давления
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(res["phi_deg"], res["p_Pa"] / 1e6, 'b-', linewidth=2, label='Модель')
        ax.axhline(y=p_max_ref, color='r', linestyle='--',
                   label=f'Референс p_max ≈ {p_max_ref} MPa (digitized)')
        ax.set_xlabel("φ, градусы")
        ax.set_ylabel("Давление, МПа")
        ax.set_title(f"Gwynllyw (constant viscosity): ε = {epsilon}\nL/D = 4, ошибка = {error_pct:.1f}%")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 360)
        ax.set_ylim(bottom=0)

        fig.savefig(OUT_DIR / f"gwynllyw_e{int(epsilon*100):02d}.png",
                    dpi=150, bbox_inches='tight')
        plt.close(fig)

    df = pd.DataFrame(results)
    df.to_csv(OUT_DIR / "gwynllyw_comparison.csv", index=False)

    return df


# ============================================================================
# БЛОК 3: СЕТОЧНАЯ СХОДИМОСТЬ (фиксированное положение!)
# ============================================================================

def run_grid_convergence():
    """
    Проверка сеточной сходимости.

    МЕТОДИКА (по рекомендации эксперта):
    1. Найти равновесие на САМОЙ ТОНКОЙ сетке (360×100)
    2. Зафиксировать (ε, φ₀)
    3. Прогнать solve_reynolds() на разных сетках при ФИКСИРОВАННЫХ (ε, φ₀)
    4. Смотреть как меняется p_max от сетки (без влияния ошибки поиска равновесия)
    """
    print("\n" + "=" * 60)
    print("БЛОК 3: Сеточная сходимость")
    print("(фиксированное положение ε, φ₀)")
    print("=" * 60)

    # Тестовый режим
    n_rpm, W = 250, 300  # средний режим для надёжности

    # Шаг 1: Находим равновесие на самой тонкой сетке
    print(f"\nШаг 1: Поиск равновесия на сетке 360×100 для {n_rpm} rpm, {W} N...")
    config_fine = create_patel_config(n_rpm, epsilon=0.5, n_phi=360, n_z=100)
    eq = find_equilibrium(
        config_fine,
        W_ext=W,
        load_angle=-np.pi/2,
        epsilon_bounds=(0.01, 0.98),
        verbose=False
    )
    epsilon_fixed = eq.epsilon
    phi0_fixed = eq.phi0
    print(f"  Найдено: ε = {epsilon_fixed:.4f}, φ₀ = {np.degrees(phi0_fixed):.2f}°")

    # Шаг 2: Прогоняем на разных сетках с ФИКСИРОВАННЫМ положением
    print(f"\nШаг 2: Расчёты на разных сетках при фиксированных ε, φ₀...")

    grids = [
        (60, 15),
        (90, 25),
        (120, 35),
        (180, 50),
        (240, 70),
        (360, 100),
    ]

    results = []

    for n_phi, n_z in grids:
        print(f"  Сетка {n_phi}×{n_z}...", end=" ")

        res = run_patel_validation_fixed_position(
            n_rpm, epsilon_fixed, phi0_fixed,
            n_phi=n_phi, n_z=n_z
        )

        results.append({
            "n_phi": n_phi,
            "n_z": n_z,
            "N_cells": n_phi * n_z,
            "p_max_mid_kPa": res["p_max_mid_kPa"],
            "p_max_global_kPa": res["p_max_global_kPa"],
            "h_min_um": res["h_min_um"],
            "converged": res["converged"],
            "iterations": res["iterations"],
        })
        print(f"p_max_mid = {res['p_max_mid_kPa']:.2f} kPa")

    df = pd.DataFrame(results)

    # Вычисляем изменение относительно предыдущей сетки
    df["change_mid_pct"] = df["p_max_mid_kPa"].pct_change().abs() * 100
    df["change_global_pct"] = df["p_max_global_kPa"].pct_change().abs() * 100

    # Добавляем информацию о фиксированном положении
    df["epsilon_fixed"] = epsilon_fixed
    df["phi0_fixed_deg"] = np.degrees(phi0_fixed)

    df.to_csv(OUT_DIR / "grid_convergence.csv", index=False)

    # График
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # p_max vs N_cells
    ax1.plot(df["N_cells"], df["p_max_mid_kPa"], 'bo-', markersize=10, linewidth=2, label='p_max midplane')
    ax1.plot(df["N_cells"], df["p_max_global_kPa"], 'g^--', markersize=8, linewidth=1.5, label='p_max global')
    ax1.set_xlabel("Число ячеек сетки")
    ax1.set_ylabel("p_max, кПа")
    ax1.set_title(f"Сеточная сходимость\n(ε = {epsilon_fixed:.4f}, φ₀ = {np.degrees(phi0_fixed):.1f}° — фиксированы)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    for _, row in df.iterrows():
        ax1.annotate(f"{int(row['n_phi'])}×{int(row['n_z'])}",
                    (row["N_cells"], row["p_max_mid_kPa"]),
                    textcoords="offset points", xytext=(5, 10), fontsize=8)

    # Изменение vs N_cells
    ax2.bar(range(len(df)-1), df["change_mid_pct"].iloc[1:], color='blue', alpha=0.7)
    ax2.axhline(y=2, color='r', linestyle='--', label='Критерий 2%')
    ax2.set_xticks(range(len(df)-1))
    ax2.set_xticklabels([f"{int(df.iloc[i+1]['n_phi'])}×{int(df.iloc[i+1]['n_z'])}" for i in range(len(df)-1)], rotation=45)
    ax2.set_xlabel("Сетка")
    ax2.set_ylabel("Изменение p_max, %")
    ax2.set_title("Относительное изменение при измельчении сетки")
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    fig.tight_layout()
    fig.savefig(OUT_DIR / "grid_convergence.png", dpi=150, bbox_inches='tight')
    plt.close(fig)

    return df


# ============================================================================
# БЛОК 4: СВОДКА ВАЛИДАЦИИ
# ============================================================================

def write_validation_summary(patel_df, gwynllyw_df, grid_df):
    """Записать итоговый отчёт о валидации."""
    print("\n" + "=" * 60)
    print("БЛОК 4: Сводка валидации")
    print("=" * 60)

    with open(OUT_DIR / "validation_summary.txt", "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("ОТЧЁТ О ВАЛИДАЦИИ МОДЕЛИ ПОДШИПНИКА\n")
        f.write("=" * 70 + "\n\n")

        # Методика
        f.write("МЕТОДИКА ВАЛИДАЦИИ\n")
        f.write("-" * 40 + "\n")
        f.write("1. Patel: сравнение p_max на MIDPLANE (где стоял датчик)\n")
        f.write("   Источник данных: ISOVG46 (обычное масло, без магнитного поля)\n")
        f.write("2. Gwynllyw: сравнение p_max с оцифрованными графиками\n")
        f.write("   Источник данных: constant viscosity case (линия b)\n")
        f.write("3. Grid convergence: ФИКСИРОВАННОЕ положение (ε, φ₀)\n\n")

        # Patel
        f.write("1. ВАЛИДАЦИЯ ПО PATEL ET AL. (ЭКСПЕРИМЕНТАЛЬНАЯ)\n")
        f.write("-" * 40 + "\n")
        mean_error_patel = patel_df["error_mid_pct"].mean()
        max_error_patel = patel_df["error_mid_pct"].max()
        min_error_patel = patel_df["error_mid_pct"].min()
        f.write(f"   Число кейсов: {len(patel_df)}\n")
        f.write(f"   Ошибка (p_max midplane): min={min_error_patel:.1f}%, mean={mean_error_patel:.1f}%, max={max_error_patel:.1f}%\n")
        patel_ok = max_error_patel < 50
        f.write(f"   Критерий: max < 50%\n")
        f.write(f"   Статус: {'PASS' if patel_ok else 'FAIL'}\n\n")

        # Gwynllyw
        f.write("2. ВАЛИДАЦИЯ ПО GWYNLLYW ET AL. (ЧИСЛЕННАЯ)\n")
        f.write("-" * 40 + "\n")
        mean_error_gwyn = gwynllyw_df["error_pct"].mean()
        max_error_gwyn = gwynllyw_df["error_pct"].max()
        f.write(f"   Число кейсов: {len(gwynllyw_df)}\n")
        f.write(f"   Ошибка: mean={mean_error_gwyn:.1f}%, max={max_error_gwyn:.1f}%\n")
        f.write(f"   Примечание: референсные значения оцифрованы с графиков (±10-15%)\n")
        gwyn_ok = max_error_gwyn < 30
        f.write(f"   Критерий: max < 30%\n")
        f.write(f"   Статус: {'PASS' if gwyn_ok else 'FAIL'}\n\n")

        # Сетка
        f.write("3. СЕТОЧНАЯ СХОДИМОСТЬ\n")
        f.write("-" * 40 + "\n")
        last_change = grid_df["change_mid_pct"].iloc[-1] if len(grid_df) > 1 else 0
        f.write(f"   Фиксированное положение: ε = {grid_df['epsilon_fixed'].iloc[0]:.4f}\n")
        f.write(f"   Изменение p_max на последнем шаге: {last_change:.2f}%\n")
        grid_ok = last_change < 5
        f.write(f"   Критерий: < 5%\n")
        f.write(f"   Статус: {'PASS' if grid_ok else 'FAIL'}\n\n")

        # Итог
        f.write("=" * 70 + "\n")
        all_ok = patel_ok and gwyn_ok and grid_ok
        if all_ok:
            f.write("ИТОГ: МОДЕЛЬ ВАЛИДНА\n")
            f.write("\nМодель корректно воспроизводит:\n")
            f.write("- Экспериментальные данные Patel (ISOVG46)\n")
            f.write("- Численный бенчмарк Gwynllyw (constant viscosity)\n")
            f.write("- Сеточная сходимость достигнута\n")
        else:
            f.write("ИТОГ: ТРЕБУЕТСЯ ДОРАБОТКА\n")
            if not patel_ok:
                f.write("- Большая ошибка по Patel (проверить вязкость, температуру)\n")
            if not gwyn_ok:
                f.write("- Большая ошибка по Gwynllyw (проверить L/D ratio)\n")
            if not grid_ok:
                f.write("- Сетка не сошлась (нужна более тонкая сетка)\n")
        f.write("=" * 70 + "\n")

    with open(OUT_DIR / "validation_summary.txt", "r", encoding="utf-8") as f:
        print(f.read())


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("ЭТАП 9: ВАЛИДАЦИЯ МОДЕЛИ ПОДШИПНИКА")
    print("=" * 60)

    patel_df = run_patel_block()
    gwynllyw_df = run_gwynllyw_block()
    grid_df = run_grid_convergence()

    write_validation_summary(patel_df, gwynllyw_df, grid_df)

    print(f"\nРезультаты сохранены в: {OUT_DIR}")


if __name__ == "__main__":
    main()
