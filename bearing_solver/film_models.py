"""
Модели толщины масляной плёнки.

По ТЗ film_model должен предоставлять callbacks:
    - H(φ, Z) — толщина плёнки
    - dH_dphi(φ, Z) — производная по φ
    - dH_dt_star(φ, Z) — (опц.) производная по безразмерному времени

Это обеспечивает универсальность для smooth/texture/roughness.
"""

from typing import Protocol, runtime_checkable
import numpy as np

from .config import BearingConfig


@runtime_checkable
class FilmModel(Protocol):
    """
    Протокол для модели толщины плёнки.

    Все методы возвращают безразмерные величины (H = h/c).
    """

    def H(self, phi: np.ndarray, Z: np.ndarray) -> np.ndarray:
        """
        Безразмерная толщина плёнки H = h/c.

        Args:
            phi: углы, shape (n_phi,) или 2D meshgrid
            Z: осевые координаты, shape (n_z,) или 2D meshgrid

        Returns:
            H: толщина плёнки, shape (n_phi, n_z)
        """
        ...

    def dH_dphi(self, phi: np.ndarray, Z: np.ndarray) -> np.ndarray:
        """
        Производная ∂H/∂φ.

        Args:
            phi: углы
            Z: осевые координаты

        Returns:
            dH/dφ: shape (n_phi, n_z)
        """
        ...


class SmoothFilmModel:
    """
    Модель гладкого подшипника без рельефа и шероховатости.

    H = 1 + ε·cos(φ - φ₀)

    Минимум зазора при φ - φ₀ = π (h_min = c(1-ε))
    Максимум зазора при φ - φ₀ = 0 (h_max = c(1+ε))
    """

    def __init__(self, config: BearingConfig):
        """
        Args:
            config: конфигурация подшипника
        """
        self.config = config
        self.epsilon = config.epsilon
        self.phi0 = config.phi0

    def H(self, phi: np.ndarray, Z: np.ndarray) -> np.ndarray:
        """
        Безразмерная толщина плёнки.

        H = 1 + ε·cos(φ - φ₀)
        """
        # Создаём 2D сетку если входы 1D
        if phi.ndim == 1 and Z.ndim == 1:
            PHI, _ = np.meshgrid(phi, Z, indexing='ij')
        else:
            PHI = phi

        return 1.0 + self.epsilon * np.cos(PHI - self.phi0)

    def dH_dphi(self, phi: np.ndarray, Z: np.ndarray) -> np.ndarray:
        """
        Производная ∂H/∂φ = -ε·sin(φ - φ₀)
        """
        if phi.ndim == 1 and Z.ndim == 1:
            PHI, _ = np.meshgrid(phi, Z, indexing='ij')
        else:
            PHI = phi

        return -self.epsilon * np.sin(PHI - self.phi0)

    def dH_dt_star(self, phi: np.ndarray, Z: np.ndarray) -> np.ndarray:
        """
        Производная по безразмерному времени ∂H/∂t* = 0 для стационарного случая.

        Для динамических расчётов (этап 4) будет ненулевой.
        """
        if phi.ndim == 1 and Z.ndim == 1:
            return np.zeros((len(phi), len(Z)))
        return np.zeros_like(phi)

    @property
    def h_min(self) -> float:
        """Минимальная размерная толщина, м."""
        return self.config.c * (1 - self.epsilon)

    @property
    def h_max(self) -> float:
        """Максимальная размерная толщина, м."""
        return self.config.c * (1 + self.epsilon)

    @property
    def H_min(self) -> float:
        """Минимальная безразмерная толщина."""
        return 1 - self.epsilon

    @property
    def H_max(self) -> float:
        """Максимальная безразмерная толщина."""
        return 1 + self.epsilon

    def update_position(self, epsilon: float, phi0: float):
        """Обновить положение вала (для итеративных расчётов)."""
        self.epsilon = epsilon
        self.phi0 = phi0


class FilmModelWithFlowFactors(SmoothFilmModel):
    """
    Базовый класс для моделей с flow factors (Patir-Cheng).

    Для этапа 7: шероховатость.
    """

    def phi_x(self, phi: np.ndarray, Z: np.ndarray) -> np.ndarray:
        """Pressure flow factor по φ (=1 для гладкого)."""
        if phi.ndim == 1 and Z.ndim == 1:
            return np.ones((len(phi), len(Z)))
        return np.ones_like(phi)

    def phi_z(self, phi: np.ndarray, Z: np.ndarray) -> np.ndarray:
        """Pressure flow factor по Z (=1 для гладкого)."""
        if phi.ndim == 1 and Z.ndim == 1:
            return np.ones((len(phi), len(Z)))
        return np.ones_like(phi)

    def phi_s(self, phi: np.ndarray, Z: np.ndarray) -> np.ndarray:
        """Shear flow factor (=0 для гладкого)."""
        if phi.ndim == 1 and Z.ndim == 1:
            return np.zeros((len(phi), len(Z)))
        return np.zeros_like(phi)
