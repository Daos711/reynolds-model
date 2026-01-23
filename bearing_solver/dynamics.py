"""
Этап 4: Динамические коэффициенты подшипника.

Матрицы жёсткости K и демпфирования C:

    [Fx]   [Kxx  Kxy] [Δx]   [Cxx  Cxy] [ẋ]
    [Fy] = [Kyx  Kyy] [Δy] + [Cyx  Cyy] [ẏ]

Знаки:
    - Силы Fx, Fy — от плёнки к валу (реакция опоры)
    - K: жёсткость определяется как Kij = -∂Fi/∂xj
      (минус, т.к. при смещении +Δx сила уменьшается)
    - C: демпфирование определяется из squeeze-члена

Методы расчёта:
    - K: конечные разности по положению (ex, ey)
    - C: решение с squeeze-членом при единичных vx*, vy*

Нормировка скоростей:
    vx* = (R/c) · (ẋ/U) = (R/c) · ẋ/(ωR) = ẋ/(ωc)
    vy* = (R/c) · (ẏ/U) = ẏ/(ωc)

    Для единичного vx* = 1: ẋ = ωc

Типичные диапазоны:
    K ~ 10⁷ - 10⁹ Н/м
    C ~ 10⁴ - 10⁶ Н·с/м
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Callable
import numpy as np

from .config import BearingConfig
from .reynolds_solver import solve_reynolds, ReynoldsSolver, ReynoldsResult
from .forces import compute_forces, BearingForces
from .film_models import FilmModel


@dataclass
class DynamicCoefficients:
    """Динамические коэффициенты подшипника."""

    # Жёсткость, Н/м
    Kxx: float
    Kxy: float
    Kyx: float
    Kyy: float

    # Демпфирование, Н·с/м
    Cxx: float
    Cxy: float
    Cyx: float
    Cyy: float

    # Безразмерные коэффициенты (для сравнения с литературой)
    Kxx_dimless: float
    Kxy_dimless: float
    Kyx_dimless: float
    Kyy_dimless: float
    Cxx_dimless: float
    Cxy_dimless: float
    Cyx_dimless: float
    Cyy_dimless: float

    # Параметры расчёта
    delta_e: float          # смещение для K (безразмерное)
    delta_v_star: float     # смещение для C (безразмерное)

    # Исходное положение
    epsilon: float
    phi0: float
    ex: float
    ey: float

    @property
    def K(self) -> np.ndarray:
        """Матрица жёсткости 2x2, Н/м."""
        return np.array([[self.Kxx, self.Kxy],
                        [self.Kyx, self.Kyy]])

    @property
    def C(self) -> np.ndarray:
        """Матрица демпфирования 2x2, Н·с/м."""
        return np.array([[self.Cxx, self.Cxy],
                        [self.Cyx, self.Cyy]])

    @property
    def K_dimless(self) -> np.ndarray:
        """Безразмерная матрица жёсткости 2x2."""
        return np.array([[self.Kxx_dimless, self.Kxy_dimless],
                        [self.Kyx_dimless, self.Kyy_dimless]])

    @property
    def C_dimless(self) -> np.ndarray:
        """Безразмерная матрица демпфирования 2x2."""
        return np.array([[self.Cxx_dimless, self.Cxy_dimless],
                        [self.Cyx_dimless, self.Cyy_dimless]])


def _create_config_for_position(
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


def compute_forces_at_position(
    base_config: BearingConfig,
    ex: float,
    ey: float,
    n_phi: int = 180,
    n_z: int = 50,
    film_model_factory: Optional[Callable[[BearingConfig], FilmModel]] = None
) -> Tuple[float, float]:
    """
    Вычислить силы Fx, Fy для заданного положения (ex, ey).

    Returns:
        (Fx, Fy) в Н
    """
    epsilon = np.sqrt(ex**2 + ey**2)
    phi0 = np.arctan2(ey, ex)

    # Ограничиваем ε
    epsilon = np.clip(epsilon, 0.01, 0.99)

    config = _create_config_for_position(base_config, epsilon, phi0, n_phi, n_z)
    film_model = film_model_factory(config) if film_model_factory else None
    reynolds = solve_reynolds(config, film_model=film_model)
    forces = compute_forces(reynolds, config)

    return forces.Fx, forces.Fy


def compute_forces_with_squeeze(
    base_config: BearingConfig,
    epsilon: float,
    phi0: float,
    vx_star: float,
    vy_star: float,
    n_phi: int = 180,
    n_z: int = 50,
    film_model_factory: Optional[Callable[[BearingConfig], FilmModel]] = None
) -> Tuple[float, float]:
    """
    Вычислить силы Fx, Fy со squeeze-членом.

    dH/dt* = vx*·cos(φ) + vy*·sin(φ)

    Args:
        vx_star, vy_star: безразмерные скорости

    Returns:
        (Fx, Fy) в Н
    """
    config = _create_config_for_position(base_config, epsilon, phi0, n_phi, n_z)

    # Создаём сетку для squeeze-члена
    phi, Z, _, _ = config.create_grid()
    PHI, _ = np.meshgrid(phi, Z, indexing='ij')

    # dH/dt* = vx*·cos(φ) + vy*·sin(φ)
    dH_dt_star = vx_star * np.cos(PHI) + vy_star * np.sin(PHI)

    # Решаем с squeeze-членом
    film_model = film_model_factory(config) if film_model_factory else None
    reynolds = solve_reynolds(config, film_model=film_model, dH_dt_star=dH_dt_star)
    forces = compute_forces(reynolds, config)

    return forces.Fx, forces.Fy


def compute_stiffness(
    base_config: BearingConfig,
    ex: float,
    ey: float,
    delta_e: float = 0.01,
    n_phi: int = 180,
    n_z: int = 50,
    film_model_factory: Optional[Callable[[BearingConfig], FilmModel]] = None
) -> Tuple[float, float, float, float]:
    """
    Вычислить коэффициенты жёсткости методом конечных разностей.

    Kij = -∂Fi/∂xj ≈ -(F(xj+δ) - F(xj-δ)) / (2δ)

    Args:
        ex, ey: текущее положение (безразмерные компоненты)
        delta_e: шаг для конечных разностей (безразмерный)

    Returns:
        (Kxx, Kxy, Kyx, Kyy) в Н/м
    """
    c = base_config.c  # зазор для размерности
    delta_x = delta_e * c  # размерный шаг, м
    delta_y = delta_e * c

    # Смещения по x
    Fx_xp, Fy_xp = compute_forces_at_position(base_config, ex + delta_e, ey, n_phi, n_z, film_model_factory)
    Fx_xm, Fy_xm = compute_forces_at_position(base_config, ex - delta_e, ey, n_phi, n_z, film_model_factory)

    # Смещения по y
    Fx_yp, Fy_yp = compute_forces_at_position(base_config, ex, ey + delta_e, n_phi, n_z, film_model_factory)
    Fx_ym, Fy_ym = compute_forces_at_position(base_config, ex, ey - delta_e, n_phi, n_z, film_model_factory)

    # Жёсткость: Kij = -∂Fi/∂xj
    Kxx = -(Fx_xp - Fx_xm) / (2 * delta_x)
    Kxy = -(Fx_yp - Fx_ym) / (2 * delta_y)
    Kyx = -(Fy_xp - Fy_xm) / (2 * delta_x)
    Kyy = -(Fy_yp - Fy_ym) / (2 * delta_y)

    return Kxx, Kxy, Kyx, Kyy


def compute_damping(
    base_config: BearingConfig,
    epsilon: float,
    phi0: float,
    delta_v_star: float = 0.01,
    n_phi: int = 180,
    n_z: int = 50,
    film_model_factory: Optional[Callable[[BearingConfig], FilmModel]] = None
) -> Tuple[float, float, float, float]:
    """
    Вычислить коэффициенты демпфирования из squeeze-члена.

    Cij = Fi(vj*=1) / ẋj_real

    где ẋj_real = vj* · ω · c (размерная скорость)

    Args:
        epsilon, phi0: текущее положение
        delta_v_star: безразмерная скорость для расчёта

    Returns:
        (Cxx, Cxy, Cyx, Cyy) в Н·с/м
    """
    omega = base_config.omega
    c = base_config.c

    # Базовые силы (без squeeze)
    Fx_base, Fy_base = compute_forces_with_squeeze(
        base_config, epsilon, phi0, 0.0, 0.0, n_phi, n_z, film_model_factory
    )

    # Силы при vx* = delta_v_star
    Fx_vx, Fy_vx = compute_forces_with_squeeze(
        base_config, epsilon, phi0, delta_v_star, 0.0, n_phi, n_z, film_model_factory
    )

    # Силы при vy* = delta_v_star
    Fx_vy, Fy_vy = compute_forces_with_squeeze(
        base_config, epsilon, phi0, 0.0, delta_v_star, n_phi, n_z, film_model_factory
    )

    # Изменение сил от squeeze
    dFx_dvx = (Fx_vx - Fx_base) / delta_v_star
    dFy_dvx = (Fy_vx - Fy_base) / delta_v_star
    dFx_dvy = (Fx_vy - Fx_base) / delta_v_star
    dFy_dvy = (Fy_vy - Fy_base) / delta_v_star

    # Перевод в размерное демпфирование
    # По соглашению ротординамики: C = -∂F/∂ẋ (положительное C означает диссипацию)
    # vx* = ẋ / (ω·c)  =>  ∂F/∂ẋ = (∂F/∂vx*) / (ω·c)

    factor = 1.0 / (omega * c)  # перевод из ∂F/∂v* в ∂F/∂ẋ

    # Знак минус: C_ij = -∂F_i/∂ẋ_j
    Cxx = -dFx_dvx * factor
    Cxy = -dFx_dvy * factor
    Cyx = -dFy_dvx * factor
    Cyy = -dFy_dvy * factor

    return Cxx, Cxy, Cyx, Cyy


def compute_dynamic_coefficients(
    base_config: BearingConfig,
    epsilon: float,
    phi0: float,
    delta_e: float = 0.01,
    delta_v_star: float = 0.01,
    n_phi: int = 180,
    n_z: int = 50,
    verbose: bool = False,
    film_model_factory: Optional[Callable[[BearingConfig], FilmModel]] = None
) -> DynamicCoefficients:
    """
    Вычислить полный набор динамических коэффициентов.

    Args:
        base_config: базовая конфигурация подшипника
        epsilon: относительный эксцентриситет
        phi0: угол линии центров, рад
        delta_e: шаг для K (безразмерный, доля от c)
        delta_v_star: шаг для C (безразмерная скорость)
        n_phi, n_z: размер сетки
        verbose: печатать ход расчёта

    Returns:
        DynamicCoefficients
    """
    ex = epsilon * np.cos(phi0)
    ey = epsilon * np.sin(phi0)

    if verbose:
        print(f"Расчёт K и C для ε={epsilon:.4f}, φ₀={np.degrees(phi0):.1f}°")
        print(f"  ex={ex:.4f}, ey={ey:.4f}")
        print(f"  delta_e={delta_e}, delta_v*={delta_v_star}")

    # Жёсткость
    if verbose:
        print("Расчёт жёсткости K...")
    Kxx, Kxy, Kyx, Kyy = compute_stiffness(
        base_config, ex, ey, delta_e, n_phi, n_z, film_model_factory
    )

    # Демпфирование
    if verbose:
        print("Расчёт демпфирования C...")
    Cxx, Cxy, Cyx, Cyy = compute_damping(
        base_config, epsilon, phi0, delta_v_star, n_phi, n_z, film_model_factory
    )

    # Безразмерные коэффициенты
    # K* = K · c / F₀, где F₀ = 3μUR²L/c²
    # C* = C · ω·c / F₀
    F0 = base_config.force_scale
    c = base_config.c
    omega = base_config.omega

    K_scale = c / F0  # для K* = K · K_scale
    C_scale = omega * c / F0  # для C* = C · C_scale

    if verbose:
        print(f"\nРезультаты:")
        print(f"  Kxx = {Kxx/1e6:.2f} МН/м")
        print(f"  Kxy = {Kxy/1e6:.2f} МН/м")
        print(f"  Kyx = {Kyx/1e6:.2f} МН/м")
        print(f"  Kyy = {Kyy/1e6:.2f} МН/м")
        print(f"  Cxx = {Cxx/1e3:.2f} кН·с/м")
        print(f"  Cxy = {Cxy/1e3:.2f} кН·с/м")
        print(f"  Cyx = {Cyx/1e3:.2f} кН·с/м")
        print(f"  Cyy = {Cyy/1e3:.2f} кН·с/м")

    return DynamicCoefficients(
        Kxx=Kxx, Kxy=Kxy, Kyx=Kyx, Kyy=Kyy,
        Cxx=Cxx, Cxy=Cxy, Cyx=Cyx, Cyy=Cyy,
        Kxx_dimless=Kxx * K_scale,
        Kxy_dimless=Kxy * K_scale,
        Kyx_dimless=Kyx * K_scale,
        Kyy_dimless=Kyy * K_scale,
        Cxx_dimless=Cxx * C_scale,
        Cxy_dimless=Cxy * C_scale,
        Cyx_dimless=Cyx * C_scale,
        Cyy_dimless=Cyy * C_scale,
        delta_e=delta_e,
        delta_v_star=delta_v_star,
        epsilon=epsilon,
        phi0=phi0,
        ex=ex,
        ey=ey,
    )


def check_delta_sensitivity(
    base_config: BearingConfig,
    epsilon: float,
    phi0: float,
    delta_values: list = None,
    n_phi: int = 180,
    n_z: int = 50,
    verbose: bool = True
) -> dict:
    """
    Проверить чувствительность K и C к выбору delta.

    Рекомендуется вариация ±2× от номинала.

    Args:
        delta_values: список значений delta для проверки
                     (по умолчанию [0.005, 0.01, 0.02])

    Returns:
        dict с результатами для каждого delta
    """
    if delta_values is None:
        delta_values = [0.005, 0.01, 0.02]

    results = {}

    for delta in delta_values:
        if verbose:
            print(f"\n=== delta = {delta} ===")

        coeffs = compute_dynamic_coefficients(
            base_config, epsilon, phi0,
            delta_e=delta, delta_v_star=delta,
            n_phi=n_phi, n_z=n_z,
            verbose=verbose
        )
        results[delta] = coeffs

    # Анализ чувствительности
    if verbose and len(delta_values) >= 2:
        print("\n=== Анализ чувствительности ===")
        ref_delta = delta_values[len(delta_values)//2]  # средний
        ref = results[ref_delta]

        for delta, coeffs in results.items():
            if delta == ref_delta:
                continue

            dKxx = abs(coeffs.Kxx - ref.Kxx) / abs(ref.Kxx) * 100
            dKyy = abs(coeffs.Kyy - ref.Kyy) / abs(ref.Kyy) * 100
            dCxx = abs(coeffs.Cxx - ref.Cxx) / abs(ref.Cxx) * 100 if ref.Cxx != 0 else 0
            dCyy = abs(coeffs.Cyy - ref.Cyy) / abs(ref.Cyy) * 100 if ref.Cyy != 0 else 0

            print(f"delta={delta} vs {ref_delta}:")
            print(f"  ΔKxx = {dKxx:.1f}%, ΔKyy = {dKyy:.1f}%")
            print(f"  ΔCxx = {dCxx:.1f}%, ΔCyy = {dCyy:.1f}%")

    return results
