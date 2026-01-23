"""
Этап 3: Поиск положения равновесия вала.

Задача: по заданной внешней нагрузке W_ext найти (ε, φ₀),
при которых несущая способность плёнки уравновешивает нагрузку.

Условие равновесия (векторное):
    W(ε, φ₀) = W_ext

То есть:
    Fx(ε, φ₀) = Wx_ext
    Fy(ε, φ₀) = Wy_ext
"""

from dataclasses import dataclass
from typing import Optional, Callable
import numpy as np
from scipy.optimize import root, minimize_scalar, brentq

from .config import BearingConfig
from .reynolds_solver import solve_reynolds, ReynoldsResult
from .forces import compute_forces, compute_stage2, Stage2Result, BearingForces


@dataclass
class EquilibriumResult:
    """Результат поиска равновесия."""

    # Найденное положение
    epsilon: float          # эксцентриситет равновесия
    phi0: float             # угол линии центров, рад

    # Внешняя нагрузка (вход)
    W_ext: float            # заданная нагрузка, Н
    load_angle: float       # угол нагрузки, рад

    # Достигнутое равновесие
    W_achieved: float       # достигнутая несущая способность, Н
    Fx_achieved: float      # Fx при равновесии, Н
    Fy_achieved: float      # Fy при равновесии, Н

    # Невязка
    residual: float         # |W - W_ext| / W_ext
    residual_abs: float     # |W - W_ext|, Н

    # Сходимость
    converged: bool         # сошлось ли решение
    iterations: int         # число итераций (вызовов функции)
    message: str            # сообщение от солвера

    # Полные результаты этапов 1-2
    stage2: Optional[Stage2Result] = None


def _create_config_for_equilibrium(
    base_config: BearingConfig,
    epsilon: float,
    phi0: float
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
        n_phi=base_config.n_phi,
        n_z=base_config.n_z,
    )


def find_equilibrium(
    base_config: BearingConfig,
    W_ext: float,
    load_angle: float = -np.pi/2,  # вертикально вниз (270° = -90°)
    epsilon_bounds: tuple = (0.05, 0.95),
    tol: float = 1e-2,  # 1% по умолчанию
    max_iter: int = 50,
    verbose: bool = False
) -> EquilibriumResult:
    """
    Найти положение равновесия вала.

    Использует двухэтапный подход:
    1. Найти ε методом Брента (1D поиск по |W|)
    2. Уточнить φ₀ из условия направления силы

    Args:
        base_config: базовая конфигурация подшипника (ε и φ₀ будут изменяться)
        W_ext: внешняя нагрузка, Н
        load_angle: направление нагрузки, рад (по умолчанию вниз = -π/2)
        epsilon_bounds: границы поиска ε
        tol: относительная точность по невязке
        max_iter: максимум итераций
        verbose: печатать ход решения

    Returns:
        EquilibriumResult с найденным положением равновесия
    """
    from scipy.optimize import brentq, minimize_scalar

    # Компоненты внешней нагрузки
    Wx_ext = W_ext * np.cos(load_angle)
    Wy_ext = W_ext * np.sin(load_angle)

    # Счётчик вызовов
    call_count = [0]

    # Кэш для результатов (чтобы не пересчитывать)
    cache = {}

    def compute_W_for_eps(eps, phi0):
        """Вычислить W для заданных ε, φ₀."""
        key = (round(eps, 6), round(phi0, 6))
        if key in cache:
            return cache[key]

        call_count[0] += 1
        config = _create_config_for_equilibrium(base_config, eps, phi0)
        reynolds = solve_reynolds(config)
        forces = compute_forces(reynolds, config)

        cache[key] = (forces.W, forces.Fx, forces.Fy, forces.force_angle)
        return cache[key]

    def estimate_phi0(eps):
        """
        Оценка φ₀ из физических соображений.

        Сила направлена примерно под углом (φ₀ + π/2 + attitude_angle).
        Для уравновешивания нагрузки: force_angle ≈ load_angle + π
        """
        # Attitude angle зависит от ε (эмпирическая формула)
        # При ε→0: α→90°, при ε→1: α→0°
        attitude_approx = np.radians(70 * (1 - eps))

        # Вектор силы должен быть направлен против нагрузки
        # force_angle = load_angle + π
        # force_angle ≈ phi0 + π/2 + attitude_angle
        # => phi0 ≈ load_angle + π - π/2 - attitude_angle
        phi0 = load_angle + np.pi/2 - attitude_approx
        return phi0 % (2 * np.pi)

    def W_residual(eps):
        """Невязка |W(ε) - W_ext| для поиска Брентом."""
        phi0 = estimate_phi0(eps)
        W, Fx, Fy, _ = compute_W_for_eps(eps, phi0)

        if verbose and call_count[0] % 5 == 0:
            print(f"  iter {call_count[0]:3d}: ε={eps:.4f}, φ₀={np.degrees(phi0):6.1f}°, "
                  f"W={W/1000:.2f} кН, target={W_ext/1000:.2f} кН")

        return W - W_ext

    if verbose:
        print(f"Целевая нагрузка: W_ext={W_ext/1000:.2f} кН, угол={np.degrees(load_angle):.1f}°")

    # Этап 1: Найти ε методом Брента
    try:
        # Проверяем знаки на границах
        W_low = W_residual(epsilon_bounds[0])
        W_high = W_residual(epsilon_bounds[1])

        if W_low * W_high < 0:
            # Есть корень в интервале
            eps_found = brentq(W_residual, epsilon_bounds[0], epsilon_bounds[1],
                               xtol=tol, maxiter=max_iter)
        else:
            # Корень вне интервала - используем минимизацию
            if verbose:
                print(f"  Корень вне интервала, используем минимизацию")
            result = minimize_scalar(
                lambda e: abs(W_residual(e)),
                bounds=epsilon_bounds,
                method='bounded',
                options={'maxiter': max_iter}
            )
            eps_found = result.x

    except Exception as e:
        if verbose:
            print(f"  [!] Ошибка Brent: {e}, используем минимизацию")
        result = minimize_scalar(
            lambda e: abs(W_residual(e)),
            bounds=epsilon_bounds,
            method='bounded'
        )
        eps_found = result.x

    # Этап 2: Уточнить φ₀
    phi0_found = estimate_phi0(eps_found)

    # Финальный расчёт с найденными параметрами
    config_final = _create_config_for_equilibrium(base_config, eps_found, phi0_found)
    reynolds_final = solve_reynolds(config_final)
    stage2_final = compute_stage2(reynolds_final, config_final, return_tau_field=False)

    Fx_achieved = stage2_final.forces.Fx
    Fy_achieved = stage2_final.forces.Fy
    W_achieved = stage2_final.forces.W

    # Невязка по модулю силы
    residual_W = abs(W_achieved - W_ext)
    residual_rel = residual_W / W_ext if W_ext > 0 else residual_W

    # Невязка по компонентам (для информации)
    residual_vec = np.sqrt((Fx_achieved - Wx_ext)**2 + (Fy_achieved - Wy_ext)**2)

    converged = residual_rel < tol

    if verbose:
        print(f"\nРезультат: ε={eps_found:.4f}, φ₀={np.degrees(phi0_found):.1f}°")
        print(f"W={W_achieved/1000:.3f} кН, невязка={residual_rel*100:.4f}%")
        print(f"Сходимость: {converged}, итераций: {call_count[0]}")

    return EquilibriumResult(
        epsilon=eps_found,
        phi0=phi0_found,
        W_ext=W_ext,
        load_angle=load_angle,
        W_achieved=W_achieved,
        Fx_achieved=Fx_achieved,
        Fy_achieved=Fy_achieved,
        residual=residual_rel,
        residual_abs=residual_W,
        converged=converged,
        iterations=call_count[0],
        message="Brent + phi0 estimation",
        stage2=stage2_final
    )


def find_equilibrium_1d(
    base_config: BearingConfig,
    W_ext: float,
    load_angle: float = -np.pi/2,
    epsilon_bounds: tuple = (0.1, 0.95),
    tol: float = 1e-4,
    verbose: bool = False
) -> EquilibriumResult:
    """
    Упрощённый поиск равновесия (1D по ε).

    Предполагается, что φ₀ можно вычислить из условия направления силы.
    Ищем только ε такой, что |W(ε)| = W_ext.

    Это быстрее, но менее точно для сложных случаев.
    """
    call_count = [0]

    def W_residual(eps):
        """Невязка |W(ε) - W_ext|."""
        call_count[0] += 1

        # φ₀ вычисляем из условия: вектор силы направлен против нагрузки
        # attitude_angle ≈ 55° для средних ε, корректируем
        attitude_estimate = np.radians(60 - 30 * eps)  # грубая оценка
        phi0 = load_angle + np.pi - attitude_estimate
        phi0 = phi0 % (2 * np.pi)

        try:
            config = _create_config_for_equilibrium(base_config, eps, phi0)
            reynolds = solve_reynolds(config)
            forces = compute_forces(reynolds, config)

            if verbose and call_count[0] % 3 == 0:
                print(f"  iter {call_count[0]:3d}: ε={eps:.4f}, W={forces.W/1000:.2f} кН")

            return forces.W - W_ext

        except Exception:
            return 1e10 if eps > 0.5 else -1e10

    # Поиск корня методом Брента
    try:
        eps_found = brentq(W_residual, epsilon_bounds[0], epsilon_bounds[1], xtol=tol)
    except ValueError:
        # Если корень не найден, используем минимизацию
        result = minimize_scalar(
            lambda e: abs(W_residual(e)),
            bounds=epsilon_bounds,
            method='bounded'
        )
        eps_found = result.x

    # Финальный расчёт
    attitude_estimate = np.radians(60 - 30 * eps_found)
    phi0_found = (load_angle + np.pi - attitude_estimate) % (2 * np.pi)

    config_final = _create_config_for_equilibrium(base_config, eps_found, phi0_found)
    reynolds_final = solve_reynolds(config_final)
    stage2_final = compute_stage2(reynolds_final, config_final)

    W_achieved = stage2_final.forces.W
    residual_abs = abs(W_achieved - W_ext)
    residual_rel = residual_abs / W_ext

    return EquilibriumResult(
        epsilon=eps_found,
        phi0=phi0_found,
        W_ext=W_ext,
        load_angle=load_angle,
        W_achieved=W_achieved,
        Fx_achieved=stage2_final.forces.Fx,
        Fy_achieved=stage2_final.forces.Fy,
        residual=residual_rel,
        residual_abs=residual_abs,
        converged=residual_rel < tol * 10,  # более мягкий критерий
        iterations=call_count[0],
        message="1D search (Brent)",
        stage2=stage2_final
    )
