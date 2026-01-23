"""
Этап 3: Поиск положения равновесия вала.

Задача: по заданной внешней нагрузке (Wx, Wy) найти (ε, φ₀),
при которых несущая способность плёнки уравновешивает нагрузку.

ВЕКТОРНАЯ ПОСТАНОВКА:
    Fx(ε, φ₀) = Wx
    Fy(ε, φ₀) = Wy

Переменные оптимизации — вектор эксцентриситета:
    ex = ε·cos(φ₀)
    ey = ε·sin(φ₀)

Это более устойчиво и универсально для асимметричных случаев (этапы 6-7).
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np
from scipy.optimize import root, minimize, brentq, minimize_scalar

from .config import BearingConfig
from .reynolds_solver import solve_reynolds, ReynoldsResult
from .forces import compute_forces, compute_stage2, Stage2Result, BearingForces


@dataclass
class EquilibriumResult:
    """Результат поиска равновесия."""

    # Найденное положение
    epsilon: float          # эксцентриситет равновесия
    phi0: float             # угол линии центров, рад
    ex: float               # компонента вектора эксцентриситета x
    ey: float               # компонента вектора эксцентриситета y

    # Внешняя нагрузка (вход)
    W_ext: float            # заданная нагрузка (модуль), Н
    Wx_ext: float           # компонента x, Н
    Wy_ext: float           # компонента y, Н
    load_angle: float       # угол нагрузки, рад

    # Достигнутое равновесие
    W_achieved: float       # достигнутая несущая способность, Н
    Fx_achieved: float      # Fx при равновесии, Н
    Fy_achieved: float      # Fy при равновесии, Н

    # Невязки
    residual_W: float       # |W - W_ext| / W_ext (по модулю)
    residual_vec: float     # |F - W_ext| / W_ext (векторная)

    # Сходимость
    converged: bool         # сошлось ли решение
    iterations: int         # число итераций
    message: str            # сообщение от солвера

    # Полные результаты этапов 1-2 (на финальной сетке)
    stage2: Optional[Stage2Result] = None


def _create_config_for_equilibrium(
    base_config: BearingConfig,
    epsilon: float,
    phi0: float,
    n_phi: int = None,
    n_z: int = None
) -> BearingConfig:
    """Создать конфигурацию с заданными ε и φ₀."""
    return BearingConfig(
        R=base_config.R,
        L=base_config.L,
        c=base_config.c,
        epsilon=epsilon,
        phi0=phi0,
        n_rpm=base_config.n_rpm,
        mu=base_config.mu,
        n_phi=n_phi or base_config.n_phi,
        n_z=n_z or base_config.n_z,
    )


def find_equilibrium(
    base_config: BearingConfig,
    W_ext: float,
    load_angle: float = -np.pi/2,  # вертикально вниз
    epsilon_bounds: tuple = (0.05, 0.95),
    tol: float = 1e-2,
    max_iter: int = 50,
    coarse_grid: tuple = (90, 25),
    fine_grid: tuple = (180, 50),
    verbose: bool = False
) -> EquilibriumResult:
    """
    Найти положение равновесия вала (ВЕКТОРНАЯ ПОСТАНОВКА).

    Двухэтапный подход:
    1. Найти ε методом Брента из условия |W| = W_ext
    2. Уточнить (ex, ey) методом Ньютона для Fx = Wx, Fy = Wy

    Args:
        base_config: базовая конфигурация подшипника
        W_ext: модуль внешней нагрузки, Н
        load_angle: направление нагрузки, рад (по умолчанию вниз = -π/2)
        epsilon_bounds: границы поиска ε
        tol: относительная точность
        max_iter: максимум итераций
        coarse_grid: грубая сетка (n_phi, n_z) для оптимизации
        fine_grid: тонкая сетка для финального расчёта
        verbose: печатать ход решения

    Returns:
        EquilibriumResult
    """
    # Компоненты внешней нагрузки
    Wx_ext = W_ext * np.cos(load_angle)
    Wy_ext = W_ext * np.sin(load_angle)

    call_count = [0]

    def compute_forces_for_eps_phi(eps, phi0):
        """Вычислить силы для заданных ε, φ₀."""
        call_count[0] += 1
        config = _create_config_for_equilibrium(
            base_config, eps, phi0,
            n_phi=coarse_grid[0], n_z=coarse_grid[1]
        )
        reynolds = solve_reynolds(config)
        forces = compute_forces(reynolds, config)
        return forces.Fx, forces.Fy, forces.W, forces.force_angle, forces.attitude_angle

    # ЭТАП 1: Найти ε методом Брента
    if verbose:
        print(f"Этап 1: Поиск ε (Brent)")

    def estimate_phi0_for_eps(eps):
        """Оценка φ₀ для гладкого подшипника."""
        # При заданной нагрузке вниз (load_angle = -π/2):
        # force_angle должен быть ≈ +π/2 (вверх, против нагрузки)
        # force_angle = phi0 + π/2 + attitude_angle (примерно)
        # attitude ≈ 70°·(1-ε) для гладкого подшипника
        attitude = np.radians(70 * (1 - eps))
        # force_angle ≈ load_angle + π = π/2
        # π/2 ≈ phi0 + π/2 + attitude => phi0 ≈ -attitude
        phi0 = load_angle + np.pi - np.pi/2 - attitude
        return phi0 % (2 * np.pi)

    def W_residual(eps):
        phi0 = estimate_phi0_for_eps(eps)
        Fx, Fy, W, _, _ = compute_forces_for_eps_phi(eps, phi0)
        if verbose and call_count[0] % 5 == 0:
            print(f"  ε={eps:.4f}, φ₀={np.degrees(phi0):.1f}°, W={W/1000:.2f} кН")
        return W - W_ext

    try:
        W_low = W_residual(epsilon_bounds[0])
        W_high = W_residual(epsilon_bounds[1])

        if W_low * W_high < 0:
            eps_stage1 = brentq(W_residual, epsilon_bounds[0], epsilon_bounds[1], xtol=0.001)
        else:
            # Корень вне интервала - минимизация
            res = minimize_scalar(lambda e: abs(W_residual(e)), bounds=epsilon_bounds, method='bounded')
            eps_stage1 = res.x
    except Exception:
        res = minimize_scalar(lambda e: abs(W_residual(e)), bounds=epsilon_bounds, method='bounded')
        eps_stage1 = res.x

    phi0_stage1 = estimate_phi0_for_eps(eps_stage1)

    if verbose:
        Fx1, Fy1, W1, _, _ = compute_forces_for_eps_phi(eps_stage1, phi0_stage1)
        print(f"После этапа 1: ε={eps_stage1:.4f}, φ₀={np.degrees(phi0_stage1):.1f}°")
        print(f"  W={W1/1000:.3f} кН, Fx={Fx1/1000:.3f}, Fy={Fy1/1000:.3f}")

    # ЭТАП 2: Уточнить (ex, ey) минимизацией невязки
    if verbose:
        print(f"Этап 2: Уточнение вектора (minimize)")

    ex0 = eps_stage1 * np.cos(phi0_stage1)
    ey0 = eps_stage1 * np.sin(phi0_stage1)

    def objective(x):
        ex, ey = x
        eps = np.sqrt(ex**2 + ey**2)

        # Штраф за выход за границы
        if eps < epsilon_bounds[0] or eps > epsilon_bounds[1]:
            return 1e20

        phi0 = np.arctan2(ey, ex)
        Fx, Fy, W, _, _ = compute_forces_for_eps_phi(eps, phi0)

        # Минимизируем |F - W_ext|²
        return (Fx - Wx_ext)**2 + (Fy - Wy_ext)**2

    result = minimize(
        objective,
        [ex0, ey0],
        method='Nelder-Mead',
        options={'maxiter': max_iter, 'xatol': 0.001, 'fatol': (tol * W_ext)**2}
    )

    ex_found, ey_found = result.x
    eps_found = np.sqrt(ex_found**2 + ey_found**2)
    phi0_found = np.arctan2(ey_found, ex_found) % (2 * np.pi)

    # Ограничиваем ε
    eps_found = np.clip(eps_found, epsilon_bounds[0], epsilon_bounds[1])
    ex_found = eps_found * np.cos(phi0_found)
    ey_found = eps_found * np.sin(phi0_found)

    # Финальный расчёт на тонкой сетке
    config_final = _create_config_for_equilibrium(
        base_config, eps_found, phi0_found,
        n_phi=fine_grid[0], n_z=fine_grid[1]
    )
    reynolds_final = solve_reynolds(config_final)
    stage2_final = compute_stage2(reynolds_final, config_final, return_tau_field=False)

    Fx_achieved = stage2_final.forces.Fx
    Fy_achieved = stage2_final.forces.Fy
    W_achieved = stage2_final.forces.W

    # Невязки
    residual_W = abs(W_achieved - W_ext) / W_ext
    residual_vec = np.sqrt((Fx_achieved - Wx_ext)**2 + (Fy_achieved - Wy_ext)**2) / W_ext

    converged = residual_vec < tol

    if verbose:
        print(f"\nРезультат: ε={eps_found:.4f}, φ₀={np.degrees(phi0_found):.1f}°")
        print(f"ex={ex_found:.4f}, ey={ey_found:.4f}")
        print(f"W={W_achieved/1000:.3f} кН")
        print(f"Fx={Fx_achieved/1000:.3f} кН (цель: {Wx_ext/1000:.3f})")
        print(f"Fy={Fy_achieved/1000:.3f} кН (цель: {Wy_ext/1000:.3f})")
        print(f"Невязка вект: {residual_vec*100:.4f}%")

    return EquilibriumResult(
        epsilon=eps_found,
        phi0=phi0_found,
        ex=ex_found,
        ey=ey_found,
        W_ext=W_ext,
        Wx_ext=Wx_ext,
        Wy_ext=Wy_ext,
        load_angle=load_angle,
        W_achieved=W_achieved,
        Fx_achieved=Fx_achieved,
        Fy_achieved=Fy_achieved,
        residual_W=residual_W,
        residual_vec=residual_vec,
        converged=converged,
        iterations=call_count[0],
        message="Brent + Nelder-Mead",
        stage2=stage2_final
    )


def find_equilibrium_1d(
    base_config: BearingConfig,
    W_ext: float,
    load_angle: float = -np.pi/2,
    epsilon_bounds: tuple = (0.05, 0.95),
    tol: float = 1e-2,
    coarse_grid: tuple = (90, 25),
    fine_grid: tuple = (180, 50),
    verbose: bool = False
) -> EquilibriumResult:
    """
    Упрощённый 1D поиск (только для гладкого подшипника).

    Ищет ε из условия |W(ε)| = W_ext методом Брента.
    φ₀ вычисляется из условия направления силы.

    Быстрее, но менее универсально (не для асимметричных случаев).
    """
    Wx_ext = W_ext * np.cos(load_angle)
    Wy_ext = W_ext * np.sin(load_angle)

    call_count = [0]

    def estimate_phi0(eps):
        """Оценка φ₀ для гладкого подшипника."""
        # attitude_angle ≈ 70°·(1-ε) - эмпирика
        attitude = np.radians(70 * (1 - eps))
        # При нагрузке вниз (load_angle = -π/2), сила должна быть вверх (π/2)
        # force_angle ≈ load_angle + π
        # force_angle ≈ phi0 + π/2 + attitude (из геометрии)
        # => phi0 ≈ load_angle + π - π/2 - attitude
        phi0 = load_angle + np.pi - np.pi/2 - attitude
        return phi0 % (2 * np.pi)

    def W_residual(eps):
        """Невязка |W(ε) - W_ext|."""
        phi0 = estimate_phi0(eps)
        call_count[0] += 1

        config = _create_config_for_equilibrium(
            base_config, eps, phi0,
            n_phi=coarse_grid[0], n_z=coarse_grid[1]
        )
        reynolds = solve_reynolds(config)
        forces = compute_forces(reynolds, config)

        if verbose and call_count[0] % 3 == 0:
            print(f"  iter {call_count[0]:3d}: ε={eps:.4f}, W={forces.W/1000:.2f} кН")

        return forces.W - W_ext

    if verbose:
        print(f"1D поиск: W_ext={W_ext/1000:.2f} кН")

    # Поиск Брентом
    try:
        W_low = W_residual(epsilon_bounds[0])
        W_high = W_residual(epsilon_bounds[1])

        if W_low * W_high < 0:
            eps_found = brentq(W_residual, epsilon_bounds[0], epsilon_bounds[1],
                               xtol=tol/10)
        else:
            if verbose:
                print("  Корень вне интервала, минимизация...")
            res = minimize_scalar(
                lambda e: abs(W_residual(e)),
                bounds=epsilon_bounds,
                method='bounded'
            )
            eps_found = res.x
    except Exception:
        res = minimize_scalar(
            lambda e: abs(W_residual(e)),
            bounds=epsilon_bounds,
            method='bounded'
        )
        eps_found = res.x

    phi0_found = estimate_phi0(eps_found)

    # Финальный расчёт на тонкой сетке
    config_final = _create_config_for_equilibrium(
        base_config, eps_found, phi0_found,
        n_phi=fine_grid[0], n_z=fine_grid[1]
    )
    reynolds_final = solve_reynolds(config_final)
    stage2_final = compute_stage2(reynolds_final, config_final)

    Fx = stage2_final.forces.Fx
    Fy = stage2_final.forces.Fy
    W = stage2_final.forces.W

    residual_W = abs(W - W_ext) / W_ext
    residual_vec = np.sqrt((Fx - Wx_ext)**2 + (Fy - Wy_ext)**2) / W_ext

    ex = eps_found * np.cos(phi0_found)
    ey = eps_found * np.sin(phi0_found)

    return EquilibriumResult(
        epsilon=eps_found,
        phi0=phi0_found,
        ex=ex,
        ey=ey,
        W_ext=W_ext,
        Wx_ext=Wx_ext,
        Wy_ext=Wy_ext,
        load_angle=load_angle,
        W_achieved=W,
        Fx_achieved=Fx,
        Fy_achieved=Fy,
        residual_W=residual_W,
        residual_vec=residual_vec,
        converged=residual_W < tol,
        iterations=call_count[0],
        message="1D Brent",
        stage2=stage2_final
    )
