"""
Этап 6: Микрорельеф по ГОСТ 24773-81.

Текстурированная поверхность подшипника с эллипсоидальными лунками.

Безразмерная геометрия:
    H(φ, Z) = 1 + ex·cos(φ) + ey·sin(φ) + ΔH*(φ, Z)

    где ΔH* = Δh/c — безразмерная глубина текстуры

Форма ячейки — эллипсоидальная лунка:
    r² = (Δφ·R/b)² + (Δz/a)²
    if r² <= 1: ΔH* = h_depth* × sqrt(1 - r²)

Генерация центров:
    - Regular grid: равномерная сетка с шагами Ss, Sz
    - Phyllotaxis: золотой угол α ≈ 137.5°

Связь Fn ↔ геометрия:
    N = (2πRL / (πab)) × Fn = (2RL/ab) × Fn
"""

from dataclasses import dataclass
from typing import Optional, Tuple, List, Literal
import numpy as np
from numba import njit

from .config import BearingConfig
from .film_models import FilmModel, SmoothFilmModel


@dataclass
class TextureParams:
    """Параметры текстуры поверхности."""

    # Размеры эллипсоидальной лунки (в метрах)
    a: float = 0.5e-3       # полуось по Z, м
    b: float = 0.5e-3       # полуось по φ (дуга), м
    h_depth: float = 5e-6   # глубина лунки, м

    # Плотность заполнения
    Fn: float = 0.15        # доля площади, занятая лунками (0-1)

    # Метод генерации центров
    pattern: Literal['regular', 'phyllotaxis'] = 'phyllotaxis'

    # Для regular grid
    theta: float = 0.0      # угол поворота сетки, рад

    def __post_init__(self):
        if not 0 <= self.Fn <= 1:
            raise ValueError(f"Fn must be in [0, 1], got {self.Fn}")
        if self.h_depth < 0:
            raise ValueError(f"h_depth must be >= 0, got {self.h_depth}")


@njit(cache=True)
def _wrap_to_pi(angle: float) -> float:
    """Привести угол к диапазону [-π, π]."""
    while angle > np.pi:
        angle -= 2 * np.pi
    while angle < -np.pi:
        angle += 2 * np.pi
    return angle


@njit(cache=True)
def _compute_texture_field_numba(
    phi: np.ndarray,
    Z: np.ndarray,
    centers_phi: np.ndarray,
    centers_z: np.ndarray,
    R: float,
    L: float,
    a: float,
    b: float,
    h_depth_star: float
) -> np.ndarray:
    """
    Вычислить поле текстуры на сетке (Numba-ускоренно).

    Args:
        phi: углы сетки [0, 2π), shape (n_phi,)
        Z: осевые координаты [-1, 1], shape (n_z,)
        centers_phi: углы центров лунок, shape (N,)
        centers_z: z-координаты центров (размерные, м), shape (N,)
        R: радиус подшипника, м
        L: длина подшипника, м
        a, b: полуоси лунки, м
        h_depth_star: безразмерная глубина лунки (h_depth/c)

    Returns:
        texture_field: shape (n_phi, n_z)
    """
    n_phi = len(phi)
    n_z = len(Z)
    N = len(centers_phi)

    texture_field = np.zeros((n_phi, n_z))

    for i in range(n_phi):
        for j in range(n_z):
            # Размерная z-координата текущей точки
            z_dim = Z[j] * (L / 2)  # Z ∈ [-1, 1] → z ∈ [-L/2, L/2]

            for k in range(N):
                # Разность углов с учётом периодичности
                d_phi = _wrap_to_pi(phi[i] - centers_phi[k])

                # Разность по z (размерная)
                d_z = z_dim - centers_z[k]

                # Нормированное расстояние до центра лунки
                # r² = (Δφ·R/b)² + (Δz/a)²
                r_sq = (d_phi * R / b)**2 + (d_z / a)**2

                if r_sq <= 1.0:
                    # Внутри лунки: эллипсоидальный профиль
                    delta_H = h_depth_star * np.sqrt(1.0 - r_sq)
                    texture_field[i, j] += delta_H

    return texture_field


def generate_regular_centers(
    R: float,
    L: float,
    a: float,
    b: float,
    Fn: float,
    theta: float = 0.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Генерация центров лунок на регулярной сетке.

    Fn ≈ π·a·b / (Ss × Sz)

    Args:
        R: радиус, м
        L: длина, м
        a, b: полуоси лунки, м
        Fn: плотность заполнения
        theta: угол поворота сетки, рад

    Returns:
        (centers_phi, centers_z) — массивы координат центров
    """
    # Площадь одной лунки
    S_cell = np.pi * a * b

    # Площадь поверхности цилиндра
    S_total = 2 * np.pi * R * L

    # Число лунок
    N = int(S_total * Fn / S_cell)
    if N < 1:
        N = 1

    # Шаги сетки (примерно квадратные ячейки)
    # Ss × Sz = S_cell / Fn = π·a·b / Fn
    aspect = (2 * np.pi * R) / L  # отношение периметра к длине
    Sz = np.sqrt(S_cell / Fn / aspect)
    Ss = S_cell / Fn / Sz

    # Число ячеек по каждому направлению
    n_phi = max(1, int(2 * np.pi * R / Ss))
    n_z = max(1, int(L / Sz))

    # Создаём сетку
    phi_vals = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    z_vals = np.linspace(-L/2 + Sz/2, L/2 - Sz/2, n_z)

    centers_phi = []
    centers_z = []

    for i, phi_c in enumerate(phi_vals):
        for j, z_c in enumerate(z_vals):
            # Поворот на угол theta
            # (применяем сдвиг по φ, зависящий от z)
            phi_shifted = phi_c + theta * (z_c / (L/2))
            phi_shifted = phi_shifted % (2 * np.pi)

            centers_phi.append(phi_shifted)
            centers_z.append(z_c)

    return np.array(centers_phi), np.array(centers_z)


def generate_phyllotaxis_centers(
    R: float,
    L: float,
    a: float,
    b: float,
    Fn: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Генерация центров лунок по филлотаксису (золотой угол).

    φ_k = (k × α) mod 2π,  α ≈ 137.508° (золотой угол)
    z_k = L × (k + 0.5) / N - L/2

    Args:
        R: радиус, м
        L: длина, м
        a, b: полуоси лунки, м
        Fn: плотность заполнения

    Returns:
        (centers_phi, centers_z)
    """
    # Площадь одной лунки
    S_cell = np.pi * a * b

    # Площадь поверхности цилиндра
    S_total = 2 * np.pi * R * L

    # Число лунок
    N = int(S_total * Fn / S_cell)
    if N < 1:
        N = 1

    # Золотой угол в радианах
    golden_angle = np.pi * (3 - np.sqrt(5))  # ≈ 137.508°

    centers_phi = np.zeros(N)
    centers_z = np.zeros(N)

    for k in range(N):
        centers_phi[k] = (k * golden_angle) % (2 * np.pi)
        centers_z[k] = L * (k + 0.5) / N - L/2

    return centers_phi, centers_z


class TexturedFilmModel(FilmModel):
    """
    Модель толщины плёнки с текстурой поверхности.

    H(φ, Z) = H_smooth(φ, Z) + ΔH*(φ, Z)

    где ΔH* — кешированное поле текстуры.
    """

    def __init__(
        self,
        config: BearingConfig,
        texture_params: Optional[TextureParams] = None
    ):
        """
        Args:
            config: конфигурация подшипника
            texture_params: параметры текстуры (по умолчанию ГОСТ 24773-81)
        """
        self.config = config
        self.params = texture_params or TextureParams()

        # Базовая гладкая модель
        self._smooth_model = SmoothFilmModel(config)

        # Безразмерная глубина
        self.h_depth_star = self.params.h_depth / config.c

        # Генерируем центры лунок
        if self.params.pattern == 'regular':
            self.centers_phi, self.centers_z = generate_regular_centers(
                R=config.R, L=config.L,
                a=self.params.a, b=self.params.b,
                Fn=self.params.Fn, theta=self.params.theta
            )
        else:  # phyllotaxis
            self.centers_phi, self.centers_z = generate_phyllotaxis_centers(
                R=config.R, L=config.L,
                a=self.params.a, b=self.params.b,
                Fn=self.params.Fn
            )

        self.N_cells = len(self.centers_phi)

        # Кешированное поле текстуры (вычисляется при первом вызове)
        self._texture_field: Optional[np.ndarray] = None
        self._cached_phi: Optional[np.ndarray] = None
        self._cached_Z: Optional[np.ndarray] = None

    def _compute_texture_field(self, phi: np.ndarray, Z: np.ndarray) -> np.ndarray:
        """Вычислить и закешировать поле текстуры."""
        # Проверяем, нужно ли пересчитывать
        if (self._texture_field is not None and
            self._cached_phi is not None and
            self._cached_Z is not None and
            len(phi) == len(self._cached_phi) and
            len(Z) == len(self._cached_Z) and
            np.allclose(phi, self._cached_phi) and
            np.allclose(Z, self._cached_Z)):
            return self._texture_field

        # Вычисляем новое поле
        self._texture_field = _compute_texture_field_numba(
            phi, Z,
            self.centers_phi, self.centers_z,
            self.config.R, self.config.L,
            self.params.a, self.params.b,
            self.h_depth_star
        )
        self._cached_phi = phi.copy()
        self._cached_Z = Z.copy()

        return self._texture_field

    def H(self, phi: np.ndarray, Z: np.ndarray) -> np.ndarray:
        """
        Безразмерная толщина плёнки с текстурой.

        H = H_smooth + ΔH*
        """
        H_smooth = self._smooth_model.H(phi, Z)
        texture = self._compute_texture_field(phi, Z)
        return H_smooth + texture

    def dH_dphi(self, phi: np.ndarray, Z: np.ndarray) -> np.ndarray:
        """
        Производная ∂H/∂φ.

        Для текстуры используем численную производную,
        т.к. аналитическая сложна из-за разрывов.
        """
        # Гладкая часть — аналитически
        dH_smooth = self._smooth_model.dH_dphi(phi, Z)

        # Текстурная часть — численно
        texture = self._compute_texture_field(phi, Z)
        d_phi = phi[1] - phi[0] if len(phi) > 1 else 2 * np.pi / 180

        # Центральные разности с периодичностью
        dH_texture = np.zeros_like(texture)
        n_phi = len(phi)
        for i in range(n_phi):
            i_plus = (i + 1) % n_phi
            i_minus = (i - 1 + n_phi) % n_phi
            dH_texture[i, :] = (texture[i_plus, :] - texture[i_minus, :]) / (2 * d_phi)

        return dH_smooth + dH_texture

    def get_texture_stats(self) -> dict:
        """Получить статистику текстуры."""
        return {
            'N_cells': self.N_cells,
            'a_mm': self.params.a * 1000,
            'b_mm': self.params.b * 1000,
            'h_depth_um': self.params.h_depth * 1e6,
            'h_depth_star': self.h_depth_star,
            'Fn': self.params.Fn,
            'pattern': self.params.pattern,
            'Fn_actual': self._compute_actual_Fn(),
        }

    def _compute_actual_Fn(self) -> float:
        """Вычислить фактическую плотность заполнения."""
        S_cell = np.pi * self.params.a * self.params.b
        S_total = 2 * np.pi * self.config.R * self.config.L
        return self.N_cells * S_cell / S_total
