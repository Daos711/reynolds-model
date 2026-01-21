"""
Модели толщины масляной плёнки.

Все модели реализуют протокол FilmModel с методами:
- H(phi, Z) — безразмерная толщина плёнки
- dH_dphi(phi, Z) — производная по φ
- dH_dt_star(phi, Z) — производная по безразмерному времени (опционально)
"""

from abc import ABC, abstractmethod
from typing import Protocol, runtime_checkable
import numpy as np

from .config import BearingConfig


@runtime_checkable
class FilmModel(Protocol):
    """
    Протокол для моделей толщины плёнки.

    Все координаты безразмерные:
    - phi ∈ [0, 2π] — окружная координата
    - Z ∈ [-1, 1] — осевая координата
    - H = h/c — безразмерная толщина
    """

    def H(self, phi: np.ndarray, Z: np.ndarray) -> np.ndarray:
        """
        Безразмерная толщина плёнки H = h/c.

        Args:
            phi: окружные координаты, shape (n_phi,) или (n_phi, n_z)
            Z: осевые координаты, shape (n_z,) или (n_phi, n_z)

        Returns:
            H: безразмерная толщина, shape (n_phi, n_z)
        """
        ...

    def dH_dphi(self, phi: np.ndarray, Z: np.ndarray) -> np.ndarray:
        """
        Производная толщины по φ: ∂H/∂φ.

        Args:
            phi: окружные координаты
            Z: осевые координаты

        Returns:
            dH/dφ, shape (n_phi, n_z)
        """
        ...


class SmoothFilmModel:
    """
    Модель гладкого подшипника (без текстуры и шероховатости).

    Толщина плёнки:
        H = 1 + ε·cos(φ - φ₀)

    где:
        ε — эксцентриситет
        φ₀ — угол положения линии центров
    """

    def __init__(self, config: BearingConfig):
        """
        Args:
            config: конфигурация подшипника
        """
        self.config = config
        self._epsilon = config.epsilon
        self._phi0 = config.phi0

    @property
    def epsilon(self) -> float:
        """Эксцентриситет."""
        return self._epsilon

    @epsilon.setter
    def epsilon(self, value: float) -> None:
        if not 0 < value < 1:
            raise ValueError("Эксцентриситет должен быть в (0, 1)")
        self._epsilon = value

    @property
    def phi0(self) -> float:
        """Угол положения линии центров, рад."""
        return self._phi0

    @phi0.setter
    def phi0(self, value: float) -> None:
        self._phi0 = value

    def H(self, phi: np.ndarray, Z: np.ndarray) -> np.ndarray:
        """
        Безразмерная толщина плёнки.

        H = 1 + ε·cos(φ - φ₀)

        Args:
            phi: окружные координаты, shape (n_phi,) или 2D
            Z: осевые координаты, shape (n_z,) или 2D

        Returns:
            H: безразмерная толщина, shape (n_phi, n_z)
        """
        # Приводим к 2D сетке если нужно
        if phi.ndim == 1 and Z.ndim == 1:
            PHI, _ = np.meshgrid(phi, Z, indexing='ij')
        else:
            PHI = phi

        return 1.0 + self._epsilon * np.cos(PHI - self._phi0)

    def dH_dphi(self, phi: np.ndarray, Z: np.ndarray) -> np.ndarray:
        """
        Производная толщины по φ.

        ∂H/∂φ = -ε·sin(φ - φ₀)

        Args:
            phi: окружные координаты
            Z: осевые координаты

        Returns:
            ∂H/∂φ, shape (n_phi, n_z)
        """
        if phi.ndim == 1 and Z.ndim == 1:
            PHI, _ = np.meshgrid(phi, Z, indexing='ij')
        else:
            PHI = phi

        return -self._epsilon * np.sin(PHI - self._phi0)

    def dH_dt_star(self, phi: np.ndarray, Z: np.ndarray) -> np.ndarray:
        """
        Производная толщины по безразмерному времени.

        Для стационарного случая ∂H/∂t* = 0.

        Returns:
            Нулевой массив, shape (n_phi, n_z)
        """
        if phi.ndim == 1 and Z.ndim == 1:
            return np.zeros((len(phi), len(Z)))
        return np.zeros_like(phi)

    def h_min(self) -> float:
        """
        Минимальная безразмерная толщина плёнки.

        H_min = 1 - ε (при φ - φ₀ = π)
        """
        return 1.0 - self._epsilon

    def h_max(self) -> float:
        """
        Максимальная безразмерная толщина плёнки.

        H_max = 1 + ε (при φ - φ₀ = 0)
        """
        return 1.0 + self._epsilon

    def update_position(self, epsilon: float, phi0: float) -> None:
        """
        Обновить положение вала.

        Args:
            epsilon: новый эксцентриситет
            phi0: новый угол положения линии центров
        """
        self.epsilon = epsilon
        self._phi0 = phi0


class FilmModelWithFlowFactors(ABC):
    """
    Базовый класс для моделей с flow factors (для шероховатости).

    Flow factors модифицируют уравнение Рейнольдса:
    ∂/∂φ(φₓ·H³·∂P/∂φ) + (D/L)²·∂/∂Z(φᵧ·H³·∂P/∂Z) = ∂(H·φₛ)/∂φ
    """

    @abstractmethod
    def H(self, phi: np.ndarray, Z: np.ndarray) -> np.ndarray:
        """Безразмерная толщина плёнки."""
        ...

    @abstractmethod
    def dH_dphi(self, phi: np.ndarray, Z: np.ndarray) -> np.ndarray:
        """Производная толщины по φ."""
        ...

    @abstractmethod
    def phi_x(self, phi: np.ndarray, Z: np.ndarray) -> np.ndarray:
        """
        Flow factor φₓ для течения в окружном направлении.

        Для гладкого подшипника φₓ = 1.
        """
        ...

    @abstractmethod
    def phi_z(self, phi: np.ndarray, Z: np.ndarray) -> np.ndarray:
        """
        Flow factor φᵧ для течения в осевом направлении.

        Для гладкого подшипника φᵧ = 1.
        """
        ...

    @abstractmethod
    def phi_s(self, phi: np.ndarray, Z: np.ndarray) -> np.ndarray:
        """
        Shear flow factor φₛ.

        Для гладкого подшипника φₛ = 1.
        """
        ...


class SmoothFilmModelWithFlowFactors(FilmModelWithFlowFactors):
    """
    Гладкий подшипник с интерфейсом flow factors.

    Все flow factors равны 1 (эквивалентно SmoothFilmModel).
    """

    def __init__(self, config: BearingConfig):
        self._base = SmoothFilmModel(config)

    @property
    def epsilon(self) -> float:
        return self._base.epsilon

    @epsilon.setter
    def epsilon(self, value: float) -> None:
        self._base.epsilon = value

    @property
    def phi0(self) -> float:
        return self._base.phi0

    @phi0.setter
    def phi0(self, value: float) -> None:
        self._base.phi0 = value

    def H(self, phi: np.ndarray, Z: np.ndarray) -> np.ndarray:
        return self._base.H(phi, Z)

    def dH_dphi(self, phi: np.ndarray, Z: np.ndarray) -> np.ndarray:
        return self._base.dH_dphi(phi, Z)

    def phi_x(self, phi: np.ndarray, Z: np.ndarray) -> np.ndarray:
        """Flow factor = 1 для гладкого подшипника."""
        H = self.H(phi, Z)
        return np.ones_like(H)

    def phi_z(self, phi: np.ndarray, Z: np.ndarray) -> np.ndarray:
        """Flow factor = 1 для гладкого подшипника."""
        H = self.H(phi, Z)
        return np.ones_like(H)

    def phi_s(self, phi: np.ndarray, Z: np.ndarray) -> np.ndarray:
        """Shear flow factor = 1 для гладкого подшипника."""
        H = self.H(phi, Z)
        return np.ones_like(H)

    def update_position(self, epsilon: float, phi0: float) -> None:
        self._base.update_position(epsilon, phi0)
