"""
Расчёт сил, трения и расхода смазки (Этап 2).

Формулы:
    Силы:
        F̄ₓ = ∫∫ P·cos(φ) dφ dZ
        F̄ᵧ = ∫∫ P·sin(φ) dφ dZ
        W = sqrt(Fx² + Fy²)

    Трение:
        τ = μ·U/h + (h/2)·∂p/∂x
        где ∂p/∂x = (1/R)·∂p/∂φ

    Расход (на торцах):
        qz = -(h³/12μ)·∂p/∂z
        Q = ∫ qz·R dφ
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np
from numba import njit

from .config import BearingConfig
from .reynolds_solver import ReynoldsResult


@dataclass
class BearingForces:
    """Результаты расчёта сил."""
    Fx: float              # радиальная сила, Н
    Fy: float              # тангенциальная сила, Н
    W: float               # результирующая несущая способность, Н
    force_angle: float     # угол вектора силы от φ=0, рад
    attitude_angle: float  # attitude angle от линии центров (h_min), рад

    # Безразмерные силы
    Fx_dimless: float
    Fy_dimless: float
    W_dimless: float


@dataclass
class BearingFriction:
    """Результаты расчёта трения."""
    F_friction: float      # сила трения, Н
    mu_friction: float     # коэффициент трения
    tau_max: float         # максимальное касательное напряжение, Па
    tau_mean: float        # среднее касательное напряжение, Па

    # Поле касательного напряжения
    tau: Optional[np.ndarray] = None  # shape (n_phi, n_z), Па


@dataclass
class BearingFlow:
    """Результаты расчёта расхода."""
    Q_total: float         # суммарный расход, м³/с
    Q_plus: float          # расход на торце Z=+1, м³/с
    Q_minus: float         # расход на торце Z=-1, м³/с


@dataclass
class BearingLosses:
    """Потери мощности."""
    P_friction: float      # потери на трение, Вт
    P_total: float         # общие потери, Вт


@dataclass
class Stage2Result:
    """Полный результат этапа 2."""
    # От этапа 1
    reynolds: ReynoldsResult

    # Этап 2
    forces: BearingForces
    friction: BearingFriction
    flow: BearingFlow
    losses: BearingLosses


# =============================================================================
# Numba-ускоренные функции
# =============================================================================

@njit(cache=True)
def _integrate_forces(
    P: np.ndarray,
    phi: np.ndarray,
    d_phi: float,
    d_Z: float,
    n_phi: int,
    n_z: int
) -> tuple:
    """
    Интегрирование сил по полю давления.

    F̄ₓ = ∫∫ P·cos(φ) dφ dZ
    F̄ᵧ = ∫∫ P·sin(φ) dφ dZ
    """
    Fx_dimless = 0.0
    Fy_dimless = 0.0

    for i in range(n_phi):
        cos_phi = np.cos(phi[i])
        sin_phi = np.sin(phi[i])

        for j in range(n_z):
            # Трапецеидальное интегрирование
            weight = 1.0
            if j == 0 or j == n_z - 1:
                weight = 0.5

            Fx_dimless += P[i, j] * cos_phi * weight
            Fy_dimless += P[i, j] * sin_phi * weight

    Fx_dimless *= d_phi * d_Z
    Fy_dimless *= d_phi * d_Z

    return Fx_dimless, Fy_dimless


@njit(cache=True)
def _compute_shear_stress(
    P: np.ndarray,
    H: np.ndarray,
    phi: np.ndarray,
    d_phi: float,
    n_phi: int,
    n_z: int,
    mu: float,
    U: float,
    c: float,
    R: float,
    pressure_scale: float
) -> np.ndarray:
    """
    Вычислить касательное напряжение τ(φ, Z).

    τ = μ·U/h + (h/2)·∂p/∂x

    где:
        h = H·c (размерная толщина)
        ∂p/∂x = (1/R)·∂p/∂φ = (pressure_scale/R)·∂P/∂φ
    """
    tau = np.zeros((n_phi, n_z))

    for i in range(n_phi):
        i_plus = (i + 1) % n_phi
        i_minus = (i - 1 + n_phi) % n_phi

        for j in range(n_z):
            h = H[i, j] * c  # размерная толщина

            # Производная давления по φ (центральные разности)
            dP_dphi = (P[i_plus, j] - P[i_minus, j]) / (2 * d_phi)

            # ∂p/∂x = (pressure_scale/R) × ∂P/∂φ
            dp_dx = (pressure_scale / R) * dP_dphi

            # Касательное напряжение: τ = μU/h + (h/2)·dp/dx
            # Примечание: знак второго члена зависит от направления
            tau_couette = mu * U / h           # вязкое (Куэтта)
            tau_pressure = (h / 2) * dp_dx     # от градиента давления

            tau[i, j] = tau_couette + tau_pressure

    return tau


@njit(cache=True)
def _compute_shear_stress_components(
    P: np.ndarray,
    H: np.ndarray,
    phi: np.ndarray,
    d_phi: float,
    n_phi: int,
    n_z: int,
    mu: float,
    U: float,
    c: float,
    R: float,
    pressure_scale: float
) -> tuple:
    """
    Вычислить компоненты касательного напряжения отдельно.

    Returns:
        tau_couette: вязкий член μU/h (Куэтта)
        tau_pressure: член от градиента давления (h/2)·dp/dx (Пуазёйль)
        tau_total: сумма
    """
    tau_couette = np.zeros((n_phi, n_z))
    tau_pressure = np.zeros((n_phi, n_z))

    for i in range(n_phi):
        i_plus = (i + 1) % n_phi
        i_minus = (i - 1 + n_phi) % n_phi

        for j in range(n_z):
            h = H[i, j] * c

            dP_dphi = (P[i_plus, j] - P[i_minus, j]) / (2 * d_phi)
            dp_dx = (pressure_scale / R) * dP_dphi

            tau_couette[i, j] = mu * U / h
            tau_pressure[i, j] = (h / 2) * dp_dx

    tau_total = tau_couette + tau_pressure
    return tau_couette, tau_pressure, tau_total


@njit(cache=True)
def _integrate_friction(
    tau: np.ndarray,
    d_phi: float,
    d_Z: float,
    n_phi: int,
    n_z: int,
    R: float,
    L: float
) -> float:
    """
    Интегрирование силы трения.

    F_friction = ∫∫ |τ| dA

    где dA = R·dφ × (L/2)·dZ
    """
    F_friction = 0.0

    for i in range(n_phi):
        for j in range(n_z):
            weight = 1.0
            if j == 0 or j == n_z - 1:
                weight = 0.5

            F_friction += np.abs(tau[i, j]) * weight

    # dA = R·dφ × (L/2)·dZ
    F_friction *= R * d_phi * (L / 2) * d_Z

    return F_friction


@njit(cache=True)
def _compute_flow_rate(
    P: np.ndarray,
    H: np.ndarray,
    d_phi: float,
    d_Z: float,
    n_phi: int,
    n_z: int,
    mu: float,
    c: float,
    R: float,
    L: float,
    pressure_scale: float
) -> tuple:
    """
    Вычислить расход смазки на торцах.

    qz = -(h³/12μ)·∂p/∂z

    где:
        ∂p/∂z = (2/L)·∂p/∂Z = (2·pressure_scale/L)·∂P/∂Z

    Q = ∫ qz·R dφ на торцах
    """
    Q_plus = 0.0   # расход на Z = +1
    Q_minus = 0.0  # расход на Z = -1

    for i in range(n_phi):
        # На торце Z = +1 (j = n_z - 1)
        j = n_z - 1
        h = H[i, j] * c
        # Односторонняя разность (внутрь)
        dP_dZ = (P[i, j] - P[i, j-1]) / d_Z
        dp_dz = (2 * pressure_scale / L) * dP_dZ
        qz_plus = -(h**3 / (12 * mu)) * dp_dz
        Q_plus += qz_plus * R * d_phi

        # На торце Z = -1 (j = 0)
        j = 0
        h = H[i, j] * c
        dP_dZ = (P[i, j+1] - P[i, j]) / d_Z
        dp_dz = (2 * pressure_scale / L) * dP_dZ
        qz_minus = -(h**3 / (12 * mu)) * dp_dz
        Q_minus += np.abs(qz_minus) * R * d_phi  # abs т.к. вытекает наружу

    return Q_plus, Q_minus


# =============================================================================
# Основные функции расчёта
# =============================================================================

def compute_forces(
    result: ReynoldsResult,
    config: BearingConfig
) -> BearingForces:
    """
    Вычислить силы от давления масляной плёнки.

    Args:
        result: результат решения уравнения Рейнольдса
        config: конфигурация подшипника

    Returns:
        BearingForces с компонентами сил
    """
    phi = result.phi
    P = result.P
    n_phi = len(phi)
    n_z = len(result.Z)

    d_phi = phi[1] - phi[0] if n_phi > 1 else 2*np.pi
    d_Z = result.Z[1] - result.Z[0] if n_z > 1 else 2.0

    # Интегрируем безразмерные силы
    Fx_dimless, Fy_dimless = _integrate_forces(
        P, phi, d_phi, d_Z, n_phi, n_z
    )

    W_dimless = np.sqrt(Fx_dimless**2 + Fy_dimless**2)

    # Размерные силы: F = force_scale × F̄
    Fx = Fx_dimless * config.force_scale
    Fy = Fy_dimless * config.force_scale
    W = W_dimless * config.force_scale

    # Угол вектора силы от φ=0
    force_angle = np.arctan2(Fy, Fx)

    # Attitude angle: угол от линии центров (h_min при φ = φ₀ + π)
    # attitude_angle = |force_angle - (phi0 + π)|
    h_min_angle = config.phi0 + np.pi
    attitude_angle = abs(force_angle - h_min_angle)
    # Нормализуем к [0, π]
    if attitude_angle > np.pi:
        attitude_angle = 2 * np.pi - attitude_angle

    return BearingForces(
        Fx=Fx,
        Fy=Fy,
        W=W,
        force_angle=force_angle,
        attitude_angle=attitude_angle,
        Fx_dimless=Fx_dimless,
        Fy_dimless=Fy_dimless,
        W_dimless=W_dimless
    )


def compute_friction(
    result: ReynoldsResult,
    config: BearingConfig,
    forces: BearingForces,
    return_field: bool = False
) -> BearingFriction:
    """
    Вычислить трение масляной плёнки.

    Args:
        result: результат решения уравнения Рейнольдса
        config: конфигурация подшипника
        forces: результат расчёта сил (для коэффициента трения)
        return_field: вернуть поле τ(φ, Z)

    Returns:
        BearingFriction
    """
    phi = result.phi
    P = result.P
    H = result.H
    n_phi = len(phi)
    n_z = len(result.Z)

    d_phi = phi[1] - phi[0] if n_phi > 1 else 2*np.pi
    d_Z = result.Z[1] - result.Z[0] if n_z > 1 else 2.0

    # Поле касательного напряжения
    tau = _compute_shear_stress(
        P, H, phi, d_phi, n_phi, n_z,
        config.mu, config.U, config.c, config.R,
        config.pressure_scale
    )

    # Интегрируем силу трения
    F_friction = _integrate_friction(
        tau, d_phi, d_Z, n_phi, n_z,
        config.R, config.L
    )

    # Коэффициент трения
    mu_friction = F_friction / forces.W if forces.W > 0 else 0.0

    # Статистика по τ
    tau_max = np.max(np.abs(tau))
    tau_mean = np.mean(np.abs(tau))

    return BearingFriction(
        F_friction=F_friction,
        mu_friction=mu_friction,
        tau_max=tau_max,
        tau_mean=tau_mean,
        tau=tau if return_field else None
    )


def get_shear_stress_components(
    result: ReynoldsResult,
    config: BearingConfig
) -> tuple:
    """
    Получить компоненты касательного напряжения для диагностики.

    Returns:
        tau_couette: μU/h (вязкий член, Куэтта)
        tau_pressure: (h/2)·dp/dx (от градиента давления, Пуазёйль)
        tau_total: сумма
    """
    phi = result.phi
    P = result.P
    H = result.H
    n_phi = len(phi)
    n_z = len(result.Z)

    d_phi = phi[1] - phi[0] if n_phi > 1 else 2*np.pi

    tau_couette, tau_pressure, tau_total = _compute_shear_stress_components(
        P, H, phi, d_phi, n_phi, n_z,
        config.mu, config.U, config.c, config.R,
        config.pressure_scale
    )

    return tau_couette, tau_pressure, tau_total


def compute_flow(
    result: ReynoldsResult,
    config: BearingConfig
) -> BearingFlow:
    """
    Вычислить расход смазки.

    Args:
        result: результат решения уравнения Рейнольдса
        config: конфигурация подшипника

    Returns:
        BearingFlow
    """
    phi = result.phi
    P = result.P
    H = result.H
    n_phi = len(phi)
    n_z = len(result.Z)

    d_phi = phi[1] - phi[0] if n_phi > 1 else 2*np.pi
    d_Z = result.Z[1] - result.Z[0] if n_z > 1 else 2.0

    Q_plus, Q_minus = _compute_flow_rate(
        P, H, d_phi, d_Z, n_phi, n_z,
        config.mu, config.c, config.R, config.L,
        config.pressure_scale
    )

    Q_total = Q_plus + Q_minus

    return BearingFlow(
        Q_total=Q_total,
        Q_plus=Q_plus,
        Q_minus=Q_minus
    )


def compute_losses(
    config: BearingConfig,
    friction: BearingFriction
) -> BearingLosses:
    """
    Вычислить потери мощности.

    P_friction = F_friction × U
    """
    P_friction = friction.F_friction * config.U

    return BearingLosses(
        P_friction=P_friction,
        P_total=P_friction  # На этом этапе только трение
    )


def compute_stage2(
    result: ReynoldsResult,
    config: BearingConfig,
    return_tau_field: bool = False
) -> Stage2Result:
    """
    Выполнить полный расчёт этапа 2.

    Args:
        result: результат решения уравнения Рейнольдса
        config: конфигурация подшипника
        return_tau_field: вернуть поле касательных напряжений

    Returns:
        Stage2Result со всеми результатами
    """
    forces = compute_forces(result, config)
    friction = compute_friction(result, config, forces, return_field=return_tau_field)
    flow = compute_flow(result, config)
    losses = compute_losses(config, friction)

    return Stage2Result(
        reynolds=result,
        forces=forces,
        friction=friction,
        flow=flow,
        losses=losses
    )
