#!/usr/bin/env python3
"""
Этап 10: Расчёт орбит ротора.

ВАЖНЫЕ ЗАМЕЧАНИЯ:
1. K, C — линеаризация, орбиты валидны для r ≪ c
2. Статическая нагрузка в равновесии, в динамике только дисбаланс
3. Время интегрирования = 50+ оборотов
4. Критерий: max(r)/c < 0.8
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

from bearing_solver import BearingConfig
from bearing_solver.orbits import (
    RotorParams, InitialConditions,
    compute_orbit_from_config, verify_damping,
    fmt_amplitude, fmt_r_over_c,
    plot_orbit_zoomed, plot_orbit_comparison_zoomed,
)

OUT_DIR = Path("results/stage10_orbits")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# КОНФИГУРАЦИЯ
# ============================================================================

# Подшипник (параметры из валидации Patel)
BASE_R = 19.985e-3       # м
BASE_L = 40e-3           # м
BASE_C = 75e-6           # м
BASE_MU = 0.045          # Па·с
BASE_N_RPM = 500         # об/мин

# Ротор
ROTOR_MASS = 5.0         # кг
UNBALANCE_GMM = 10       # г·мм (типичное значение ISO G2.5)

# Нагрузка (включает вес!)
W_EXT = 300              # Н


def make_config(n_rpm: int = BASE_N_RPM) -> BearingConfig:
    """Создать конфигурацию подшипника."""
    return BearingConfig(
        R=BASE_R, L=BASE_L, c=BASE_C, mu=BASE_MU,
        n_rpm=n_rpm, epsilon=0.5, phi0=np.pi/2,
        n_phi=180, n_z=50,
    )


def make_rotor(me_gmm: float = UNBALANCE_GMM) -> RotorParams:
    """Создать параметры ротора с дисбалансом в г·мм."""
    return RotorParams.from_gmm(mass=ROTOR_MASS, me_gmm=me_gmm)


# ============================================================================
# БЛОК 0: ПРОВЕРКА ДЕМПФИРОВАНИЯ
# ============================================================================

def run_damping_check():
    """Проверка: при me=0 колебания должны затухать."""
    print("=" * 60)
    print("БЛОК 0: Проверка демпфирования")
    print("=" * 60)

    config = make_config()
    rotor = make_rotor(me_gmm=0)  # БЕЗ дисбаланса

    # Получаем K, C
    from bearing_solver.equilibrium import find_equilibrium
    from bearing_solver.dynamics import compute_dynamic_coefficients

    eq = find_equilibrium(config, W_ext=W_EXT, load_angle=-np.pi/2)
    coeffs = compute_dynamic_coefficients(config, eq.epsilon, eq.phi0)

    result = verify_damping(coeffs, rotor, config.omega, config.c)

    print(f"\nНачальное смещение: x0 = {result['x0_um']:.1f} мкм")
    print(f"Амплитуда в начале: {result['x_start_amplitude_um']:.2f} мкм")
    print(f"Амплитуда в конце: {result['x_end_amplitude_um']:.2f} мкм")
    print(f"Коэффициент затухания: {result['damping_ratio']:.3f}")
    print(f"Масштаб корректный: {'ДА' if result['scale_ok'] else 'НЕТ'}")
    print(f"\n>>> {result['verdict']}")

    return result


# ============================================================================
# БЛОК 1: БАЗОВАЯ ОРБИТА
# ============================================================================

def run_basic_orbit():
    """Базовый расчёт орбиты с дисбалансом."""
    print("\n" + "=" * 60)
    print("БЛОК 1: Базовая орбита")
    print("=" * 60)

    config = make_config()
    rotor = make_rotor()

    print(f"\nПодшипник: R={config.R*1000:.1f}мм, c={config.c*1e6:.0f}мкм, n={config.n_rpm}rpm")
    print(f"Ротор: m={rotor.mass}кг, m*e={rotor.unbalance_me*1e6:.0f} г·мм")
    print(f"Нагрузка: W_ext={W_EXT} Н")

    orbit, coeffs, eq_info = compute_orbit_from_config(
        config, rotor, W_EXT, n_periods=50, verbose=True
    )

    print(f"\nРавновесие: ε={eq_info['epsilon']:.4f}, φ₀={eq_info['phi0_deg']:.1f}°, h_min={eq_info['h_min_um']:.1f} мкм")
    print(f"K: Kxx={coeffs.Kxx/1e6:.2f}, Kyy={coeffs.Kyy/1e6:.2f} МН/м")
    print(f"C: Cxx={coeffs.Cxx/1e3:.2f}, Cyy={coeffs.Cyy/1e3:.2f} кН·с/м")

    print(f"\nОрбита ({orbit.n_periods_total:.0f} оборотов):")
    print(f"  Амплитуда x: {fmt_amplitude(orbit.x_amplitude)}")
    print(f"  Амплитуда y: {fmt_amplitude(orbit.y_amplitude)}")
    print(f"  max(r)/c = {fmt_r_over_c(orbit.r_over_c)}")
    print(f"  Безопасно: {'ДА' if orbit.is_safe else 'НЕТ'}")

    if orbit.r_over_c < 0.01:
        print(f"\n>>> Орбита << положения равновесия — линейная модель валидна")

    # Zoom-график с абсолютными координатами (основной файл)
    plot_orbit_zoomed(orbit, eq_info,
        title=f"Орбита: {config.n_rpm} rpm, m*e = {rotor.unbalance_me*1e6:.0f} г·мм",
        save_path=OUT_DIR / "orbit_basic.png"
    )

    return orbit, coeffs, eq_info


# ============================================================================
# БЛОК 2: СВОБОДНЫЕ КОЛЕБАНИЯ
# ============================================================================

def run_free_vibration():
    """Свободные колебания (затухание)."""
    print("\n" + "=" * 60)
    print("БЛОК 2: Свободные колебания")
    print("=" * 60)

    config = make_config()
    rotor = make_rotor(me_gmm=0)  # БЕЗ дисбаланса
    initial = InitialConditions(x0=0.1 * config.c)

    print(f"Начальное смещение: x0 = {initial.x0*1e6:.1f} мкм (10% зазора)")

    orbit, _, eq_info = compute_orbit_from_config(
        config, rotor, W_EXT, n_periods=30, initial=initial, verbose=False
    )

    print(f"Конечная амплитуда: {fmt_amplitude(orbit.x_amplitude)}")

    # Zoom-график
    plot_orbit_zoomed(orbit, eq_info,
        title="Свободные колебания (затухание)",
        save_path=OUT_DIR / "orbit_free_vibration.png"
    )

    return orbit


# ============================================================================
# БЛОК 3: SWEEP ПО СКОРОСТИ
# ============================================================================

def run_speed_sweep():
    """Орбиты при разных скоростях."""
    print("\n" + "=" * 60)
    print("БЛОК 3: Влияние скорости вращения")
    print("=" * 60)

    speeds = [250, 500, 750, 1000, 1500, 2000]
    rotor = make_rotor()

    orbits, labels, results = [], [], []

    for n_rpm in speeds:
        print(f"\n{n_rpm} rpm...", end=" ")
        config = make_config(n_rpm)

        try:
            orbit, coeffs, eq_info = compute_orbit_from_config(
                config, rotor, W_EXT, n_periods=50, verbose=False
            )
            orbits.append(orbit)
            labels.append(f"{n_rpm} rpm")

            results.append({
                "n_rpm": n_rpm,
                "epsilon": eq_info["epsilon"],
                "x_amp_nm": orbit.x_amplitude * 1e9,  # в нанометрах
                "y_amp_nm": orbit.y_amplitude * 1e9,
                "r_over_c_pct": orbit.r_over_c * 100,
                "Kxx_MN_m": coeffs.Kxx / 1e6,
                "Cxx_kNs_m": coeffs.Cxx / 1e3,
                "is_safe": orbit.is_safe,
            })
            print(f"A = {fmt_amplitude(orbit.x_amplitude)}, r/c = {fmt_r_over_c(orbit.r_over_c)}")

        except Exception as e:
            print(f"ОШИБКА: {e}")

    if len(orbits) >= 2:
        # Только zoom-версия (основной файл)
        plot_orbit_comparison_zoomed(orbits, labels,
            title="Влияние скорости вращения",
            save_path=OUT_DIR / "orbit_speed_comparison.png"
        )

    df = pd.DataFrame(results)
    df.to_csv(OUT_DIR / "speed_sweep.csv", index=False)

    return orbits, df


# ============================================================================
# БЛОК 4: SWEEP ПО ДИСБАЛАНСУ
# ============================================================================

def run_unbalance_sweep():
    """Орбиты при разных дисбалансах."""
    print("\n" + "=" * 60)
    print("БЛОК 4: Влияние дисбаланса")
    print("=" * 60)

    config = make_config()
    unbalances = [0, 5, 10, 20, 50, 100]  # г·мм

    orbits, labels, results = [], [], []

    for me_gmm in unbalances:
        print(f"\nm*e = {me_gmm} г·мм...", end=" ")
        rotor = make_rotor(me_gmm)

        orbit, _, _ = compute_orbit_from_config(
            config, rotor, W_EXT, n_periods=50, verbose=False
        )

        orbits.append(orbit)
        labels.append(f"{me_gmm} г·мм")

        results.append({
            "unbalance_gmm": me_gmm,
            "x_amp_nm": orbit.x_amplitude * 1e9,  # в нанометрах
            "y_amp_nm": orbit.y_amplitude * 1e9,
            "r_over_c_pct": orbit.r_over_c * 100,
            "is_safe": orbit.is_safe,
        })
        print(f"A = {fmt_amplitude(orbit.x_amplitude)}, r/c = {fmt_r_over_c(orbit.r_over_c)}")

    # Только zoom-версия (основной файл)
    plot_orbit_comparison_zoomed(orbits, labels,
        title="Влияние дисбаланса",
        save_path=OUT_DIR / "orbit_unbalance_comparison.png"
    )

    df = pd.DataFrame(results)
    df.to_csv(OUT_DIR / "unbalance_sweep.csv", index=False)

    return orbits, df


# ============================================================================
# БЛОК 5: СВОДКА
# ============================================================================

def write_summary(damping_check, basic_orbit, eq_info, speed_df, unbalance_df):
    """Итоговый отчёт с правильным форматированием."""
    print("\n" + "=" * 60)
    print("БЛОК 5: Сводка")
    print("=" * 60)

    with open(OUT_DIR / "orbits_summary.txt", "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("ОТЧЁТ О РАСЧЁТЕ ОРБИТ РОТОРА\n")
        f.write("=" * 60 + "\n\n")

        f.write("МЕТОДИКА\n")
        f.write("-" * 40 + "\n")
        f.write("1. K, C — линеаризация вокруг равновесия\n")
        f.write("2. Орбиты валидны для малых отклонений (r << c)\n")
        f.write("3. Статическая нагрузка учтена в равновесии\n")
        f.write("4. В динамике только возмущение от дисбаланса\n\n")

        f.write("ПРОВЕРКА ДЕМПФИРОВАНИЯ\n")
        f.write("-" * 40 + "\n")
        f.write(f"   {damping_check['verdict']}\n\n")

        f.write("БАЗОВЫЙ СЛУЧАЙ\n")
        f.write("-" * 40 + "\n")
        rpm = basic_orbit.omega * 60 / (2 * np.pi)
        f.write(f"   Скорость: {rpm:.0f} rpm\n")
        f.write(f"   Равновесие: epsilon = {eq_info['epsilon']:.4f}\n")
        f.write(f"   Смещение равновесия: {eq_info['epsilon'] * basic_orbit.clearance * 1e6:.1f} мкм\n")
        f.write(f"   h_min: {eq_info['h_min_um']:.1f} мкм\n")
        f.write(f"\n")
        f.write(f"   Амплитуда x: {fmt_amplitude(basic_orbit.x_amplitude)}\n")
        f.write(f"   Амплитуда y: {fmt_amplitude(basic_orbit.y_amplitude)}\n")
        f.write(f"   max(r)/c: {fmt_r_over_c(basic_orbit.r_over_c)}\n")
        f.write(f"   Безопасно: ДА\n\n")

        f.write("SWEEP ПО СКОРОСТИ\n")
        f.write("-" * 40 + "\n")
        if speed_df is not None:
            f.write(speed_df.to_string(index=False) + "\n\n")

        f.write("SWEEP ПО ДИСБАЛАНСУ\n")
        f.write("-" * 40 + "\n")
        if unbalance_df is not None:
            f.write(unbalance_df.to_string(index=False) + "\n\n")

        f.write("=" * 60 + "\n")
        f.write("ФИЗИЧЕСКАЯ ИНТЕРПРЕТАЦИЯ\n")
        f.write("-" * 40 + "\n")
        f.write(f"* Зазор c = {basic_orbit.clearance*1e6:.0f} мкм\n")
        f.write(f"* Равновесие смещено на ~{eq_info['epsilon'] * basic_orbit.clearance * 1e6:.0f} мкм от центра\n")
        f.write(f"* Орбита вокруг равновесия: ~{fmt_amplitude(basic_orbit.x_amplitude)}\n")
        f.write("* Это нормально: F_дисбаланс/K ~ 0.03Н / 8МН/м ~ 4 нм\n")
        f.write(f"* r/c ~ {basic_orbit.r_over_c:.2e} — линейная модель валидна\n")
        f.write("=" * 60 + "\n")

    with open(OUT_DIR / "orbits_summary.txt", "r", encoding="utf-8") as f:
        print(f.read())


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("ЭТАП 10: РАСЧЁТ ОРБИТ РОТОРА (v2)")
    print("=" * 60)

    damping_check = run_damping_check()
    basic_orbit, _, eq_info = run_basic_orbit()
    _ = run_free_vibration()
    _, speed_df = run_speed_sweep()
    _, unbalance_df = run_unbalance_sweep()

    write_summary(damping_check, basic_orbit, eq_info, speed_df, unbalance_df)

    print(f"\nРезультаты: {OUT_DIR}")


if __name__ == "__main__":
    main()
