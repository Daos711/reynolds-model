"""
Этап 7: Шероховатость Patir-Cheng (flow factors).

Модифицированное уравнение Рейнольдса:
    ∂/∂φ(φ_x·H³·∂P/∂φ) + (D/L)²·∂/∂Z(φ_z·H³·∂P/∂Z) = ∂(H + φ_s·σ*)/∂φ

где:
    λ = H / σ* — параметр плёнки (film thickness ratio)
    σ* = σ/c — безразмерная шероховатость
    σ = √(σ_shaft² + σ_bushing²) — комбинированная шероховатость
    σ = 1.25 × Ra (RMS ≈ 1.25 × Ra для Гауссова распределения)

Flow factors (изотропный случай, Patir & Cheng 1978):
    φ_x(λ) = φ_z(λ) = 1 - 0.9·exp(-0.56·λ)  для λ ≥ 1
    φ_s(λ) = 0  (shear flow factor отключён)

Примечание: При λ < 1 (смешанный режим) используем λ_eff = max(λ, 1.0)
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
from numba import njit


# Коэффициент перевода Ra → σ (RMS)
RA_TO_SIGMA = 1.25


def sigma_from_Ra(Ra: float) -> float:
    """
    Преобразовать Ra (средняя арифметическая шероховатость) в σ (RMS).

    σ = 1.25 × Ra (для Гауссова распределения высот)

    Args:
        Ra: шероховатость Ra, м

    Returns:
        σ: RMS шероховатость, м
    """
    return RA_TO_SIGMA * Ra


def combined_sigma(sigma_shaft: float, sigma_bushing: np.ndarray) -> np.ndarray:
    """
    Вычислить комбинированную шероховатость σ(φ,Z).

    σ = √(σ_shaft² + σ_bushing(φ,Z)²)

    Args:
        sigma_shaft: RMS шероховатость вала, м (скаляр)
        sigma_bushing: RMS шероховатость втулки, м (массив shape (n_phi, n_z))

    Returns:
        combined: комбинированная шероховатость, м (shape (n_phi, n_z))
    """
    return np.sqrt(sigma_shaft**2 + sigma_bushing**2)


def build_Ra_bushing_field(
    phi: np.ndarray,
    Z: np.ndarray,
    texture_mask: np.ndarray,
    Ra_out: float,
    Ra_cell: float
) -> np.ndarray:
    """
    Построить поле Ra_bushing(φ,Z) с учётом текстуры.

    Внутри лунок: Ra = Ra_cell
    Вне лунок: Ra = Ra_out

    Args:
        phi: углы сетки, shape (n_phi,)
        Z: осевые координаты, shape (n_z,)
        texture_mask: маска лунок, shape (n_phi, n_z), True = внутри лунки
        Ra_out: Ra вне лунок, м
        Ra_cell: Ra внутри лунок, м

    Returns:
        Ra_field: поле Ra(φ,Z), shape (n_phi, n_z), м
    """
    Ra_field = np.full((len(phi), len(Z)), Ra_out)
    Ra_field[texture_mask] = Ra_cell
    return Ra_field


@njit(cache=True)
def _flow_factors_numba(
    lambda_field: np.ndarray,
    n_phi: int,
    n_z: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Вычислить flow factors Patir-Cheng (Numba-ускорено).

    φ_x = φ_z = 1 - 0.9·exp(-0.56·λ)  для λ ≥ 1
    φ_s = 0 (shear flow отключён)

    При λ < 1: используем λ_eff = max(λ, 1.0)

    Returns:
        phi_x, phi_z, phi_s, count_lambda_lt_1
    """
    phi_x = np.zeros((n_phi, n_z))
    phi_z = np.zeros((n_phi, n_z))
    phi_s = np.zeros((n_phi, n_z))

    count_lambda_lt_1 = 0

    for i in range(n_phi):
        for j in range(n_z):
            lam = lambda_field[i, j]

            # Подсчёт узлов с λ < 1
            if lam < 1.0:
                count_lambda_lt_1 += 1
                lam = 1.0  # λ_eff = max(λ, 1.0)

            # Изотропные flow factors
            phi_x[i, j] = 1.0 - 0.9 * np.exp(-0.56 * lam)
            phi_z[i, j] = phi_x[i, j]
            phi_s[i, j] = 0.0  # shear flow отключён

    return phi_x, phi_z, phi_s, count_lambda_lt_1


def flow_factors_PC(
    lambda_field: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Вычислить flow factors Patir-Cheng.

    Изотропный случай:
        φ_x(λ) = φ_z(λ) = 1 - 0.9·exp(-0.56·λ)
        φ_s(λ) = 0 (shear flow отключён)

    При λ < 1 используем λ_eff = max(λ, 1.0) и логируем долю таких узлов.

    Args:
        lambda_field: поле λ = H/σ*, shape (n_phi, n_z)

    Returns:
        phi_x: pressure flow factor по φ, shape (n_phi, n_z)
        phi_z: pressure flow factor по Z, shape (n_phi, n_z)
        phi_s: shear flow factor (всегда 0), shape (n_phi, n_z)
        frac_lambda_lt_1: доля узлов с λ < 1
    """
    n_phi, n_z = lambda_field.shape

    phi_x, phi_z, phi_s, count_lt_1 = _flow_factors_numba(
        lambda_field, n_phi, n_z
    )

    total_nodes = n_phi * n_z
    frac_lambda_lt_1 = count_lt_1 / total_nodes if total_nodes > 0 else 0.0

    return phi_x, phi_z, phi_s, frac_lambda_lt_1


@dataclass
class RoughnessParams:
    """Параметры шероховатости для расчёта."""

    # Шероховатость вала (цапфы)
    Ra_shaft: float = 0.63e-6  # м (0.63 мкм)

    # Шероховатость втулки вне лунок
    Ra_out: float = 1.25e-6  # м (1.25 мкм, фиксировано)

    # Шероховатость втулки внутри лунок
    Ra_cell: float = 0.63e-6  # м

    def __post_init__(self):
        if self.Ra_shaft < 0:
            raise ValueError(f"Ra_shaft must be >= 0, got {self.Ra_shaft}")
        if self.Ra_out < 0:
            raise ValueError(f"Ra_out must be >= 0, got {self.Ra_out}")
        if self.Ra_cell < 0:
            raise ValueError(f"Ra_cell must be >= 0, got {self.Ra_cell}")

    @property
    def sigma_shaft(self) -> float:
        """RMS шероховатость вала, м."""
        return sigma_from_Ra(self.Ra_shaft)

    @property
    def sigma_out(self) -> float:
        """RMS шероховатость втулки вне лунок, м."""
        return sigma_from_Ra(self.Ra_out)

    @property
    def sigma_cell(self) -> float:
        """RMS шероховатость втулки внутри лунок, м."""
        return sigma_from_Ra(self.Ra_cell)


@dataclass
class RoughnessResult:
    """Результат расчёта с шероховатостью."""

    # Flow factors (поля)
    phi_x: np.ndarray       # shape (n_phi, n_z)
    phi_z: np.ndarray       # shape (n_phi, n_z)
    phi_s: np.ndarray       # shape (n_phi, n_z)

    # Поля шероховатости
    sigma_field: np.ndarray   # комбинированная σ(φ,Z), м
    sigma_star: np.ndarray    # безразмерная σ* = σ/c
    lambda_field: np.ndarray  # λ = H/σ*

    # Статистика
    frac_lambda_lt_1: float   # доля узлов с λ < 1
    lambda_min: float         # минимальное значение λ
    lambda_max: float         # максимальное значение λ

    # Параметры
    Ra_shaft: float
    Ra_out: float
    Ra_cell: float


def compute_roughness_fields(
    H: np.ndarray,
    phi: np.ndarray,
    Z: np.ndarray,
    c: float,
    roughness_params: RoughnessParams,
    texture_mask: Optional[np.ndarray] = None
) -> RoughnessResult:
    """
    Вычислить все поля, связанные с шероховатостью.

    Args:
        H: безразмерная толщина плёнки, shape (n_phi, n_z)
        phi: углы сетки, shape (n_phi,)
        Z: осевые координаты, shape (n_z,)
        c: радиальный зазор, м
        roughness_params: параметры шероховатости
        texture_mask: маска лунок (True = внутри лунки), shape (n_phi, n_z)
                     Если None, вся поверхность имеет Ra_out

    Returns:
        RoughnessResult со всеми полями
    """
    n_phi, n_z = H.shape

    # Маска текстуры (по умолчанию - нет лунок)
    if texture_mask is None:
        texture_mask = np.zeros((n_phi, n_z), dtype=bool)

    # Поле Ra_bushing(φ,Z)
    Ra_bushing_field = build_Ra_bushing_field(
        phi, Z, texture_mask,
        Ra_out=roughness_params.Ra_out,
        Ra_cell=roughness_params.Ra_cell
    )

    # σ_bushing(φ,Z)
    sigma_bushing = Ra_bushing_field * RA_TO_SIGMA

    # Комбинированная σ
    sigma_field = combined_sigma(roughness_params.sigma_shaft, sigma_bushing)

    # Безразмерная σ* = σ/c
    sigma_star = sigma_field / c

    # Параметр плёнки λ = H / σ*
    # Защита от деления на ноль
    sigma_star_safe = np.maximum(sigma_star, 1e-10)
    lambda_field = H / sigma_star_safe

    # Если σ* ≈ 0, то λ → ∞ (гладкая поверхность)
    lambda_field = np.where(sigma_star < 1e-10, 1e10, lambda_field)

    # Flow factors
    phi_x, phi_z, phi_s, frac_lt_1 = flow_factors_PC(lambda_field)

    return RoughnessResult(
        phi_x=phi_x,
        phi_z=phi_z,
        phi_s=phi_s,
        sigma_field=sigma_field,
        sigma_star=sigma_star,
        lambda_field=lambda_field,
        frac_lambda_lt_1=frac_lt_1,
        lambda_min=float(np.min(lambda_field)),
        lambda_max=float(np.max(lambda_field)),
        Ra_shaft=roughness_params.Ra_shaft,
        Ra_out=roughness_params.Ra_out,
        Ra_cell=roughness_params.Ra_cell,
    )
