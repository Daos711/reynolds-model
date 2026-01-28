"""
Этап 6-8: Микрорельеф по ГОСТ 24773-81.

Текстурированная поверхность подшипника с эллипсоидальными лунками.

Безразмерная геометрия:
    H(φ, Z) = 1 + ex·cos(φ) + ey·sin(φ) + ΔH*(φ, Z)

    где ΔH* = h_star = h_depth/c — безразмерная глубина текстуры

Форма ячейки — эллипсоидальная лунка:
    r² = (Δφ·R/b)² + (Δz/a)²
    if r² <= 1: ΔH* = h_star × sqrt(1 - r²)

Генерация центров:
    - Regular (linear): равномерная сетка с шагами Ss, Sz
    - Phyllotaxis: золотой угол α ≈ 137.5°
    - Spiral: многозаходная спираль

Связь Fn ↔ геометрия:
    N = (2πRL / (πab)) × Fn = (2RL/ab) × Fn
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Literal, Union
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

    # Глубина лунки — ОДИН из двух способов задания:
    # Способ 1: размерная глубина (м)
    h_depth: Optional[float] = None
    # Способ 2: безразмерная глубина h_star = h_depth/c (предпочтительно!)
    h_star: Optional[float] = None

    # Плотность заполнения (от ВСЕЙ площади подшипника)
    Fn: float = 0.15        # доля площади, занятая лунками (0-1)

    # Метод генерации центров
    pattern: Literal['regular', 'phyllotaxis', 'spiral'] = 'phyllotaxis'

    # Для regular grid
    theta: float = 0.0      # угол поворота сетки, рад

    # Для spiral
    n_starts: int = 1       # число заходов спирали (T)
    spiral_angle: Optional[float] = None  # угол между точками (None = золотой)

    # Сектор текстуры (опционально)
    phi_min: float = 0.0    # начальный угол сектора, рад
    phi_max: float = 2 * np.pi  # конечный угол сектора, рад

    def __post_init__(self):
        if not 0 <= self.Fn <= 1:
            raise ValueError(f"Fn must be in [0, 1], got {self.Fn}")

        # Проверка задания глубины
        if self.h_depth is None and self.h_star is None:
            # По умолчанию h_star = 0.1
            self.h_star = 0.1
        if self.h_depth is not None and self.h_star is not None:
            raise ValueError("Specify either h_depth or h_star, not both")

        # Нормализация сектора
        self.phi_min = self.phi_min % (2 * np.pi)
        self.phi_max = self.phi_max % (2 * np.pi)
        if self.phi_max == 0:
            self.phi_max = 2 * np.pi

    def get_h_depth(self, c: float) -> float:
        """Получить размерную глубину h_depth для заданного зазора c."""
        if self.h_depth is not None:
            return self.h_depth
        return self.h_star * c

    def get_h_star(self, c: float) -> float:
        """Получить безразмерную глубину h_star для заданного зазора c."""
        if self.h_star is not None:
            return self.h_star
        return self.h_depth / c

    @property
    def is_full_circumference(self) -> bool:
        """Текстура на всей окружности?"""
        return abs(self.phi_max - self.phi_min - 2*np.pi) < 0.01 or \
               (self.phi_min == 0 and self.phi_max >= 2*np.pi - 0.01)

    def get_sector_fraction(self) -> float:
        """Доля окружности, занятая сектором."""
        if self.is_full_circumference:
            return 1.0
        if self.phi_max > self.phi_min:
            return (self.phi_max - self.phi_min) / (2 * np.pi)
        else:
            return (2*np.pi - self.phi_min + self.phi_max) / (2 * np.pi)


@njit(cache=True)
def _wrap_to_pi(angle: float) -> float:
    """Привести угол к диапазону [-π, π]."""
    while angle > np.pi:
        angle -= 2 * np.pi
    while angle < -np.pi:
        angle += 2 * np.pi
    return angle


@njit(cache=True)
def _compute_texture_field_scatter(
    phi: np.ndarray,
    z_dim: np.ndarray,
    centers_phi: np.ndarray,
    centers_z: np.ndarray,
    R: float,
    a: float,
    b: float,
    h_depth_star: float
) -> np.ndarray:
    """
    Вычислить поле текстуры SCATTER-алгоритмом (Numba-ускоренно).

    Вместо проверки всех N лунок для каждой точки сетки,
    обновляем только локальный патч вокруг каждой лунки.

    Сложность: O(N × m²) вместо O(n_phi × n_z × N)
    При N=1000, m=7: 49k операций вместо 9M — в 180 раз быстрее!

    Args:
        phi: углы сетки [0, 2π), shape (n_phi,)
        z_dim: размерные z-координаты, м, shape (n_z,)
        centers_phi: углы центров лунок, shape (N,)
        centers_z: z-координаты центров (размерные, м), shape (N,)
        R: радиус подшипника, м
        a, b: полуоси лунки, м
        h_depth_star: безразмерная глубина лунки (h_depth/c)

    Returns:
        texture_field: shape (n_phi, n_z)
    """
    n_phi = len(phi)
    n_z = len(z_dim)
    N = len(centers_phi)

    texture_field = np.zeros((n_phi, n_z))

    if N == 0:
        return texture_field

    dphi = phi[1] - phi[0] if n_phi > 1 else 2 * np.pi / 180
    dz = z_dim[1] - z_dim[0] if n_z > 1 else z_dim[-1] - z_dim[0]

    # Радиус влияния в индексах (сколько узлов покрывает лунка)
    m_phi = int(np.ceil((b / R) / dphi)) + 1
    m_z = int(np.ceil(a / abs(dz))) + 1

    # Предвычисляем обратные величины
    inv_b_R = R / b
    inv_a = 1.0 / a

    phi_0 = phi[0]
    z_0 = z_dim[0]

    for k in range(N):
        # Индекс центра лунки на сетке
        i0 = int(round((centers_phi[k] - phi_0) / dphi)) % n_phi
        j0 = int(round((centers_z[k] - z_0) / dz))

        # Обходим только локальный патч вокруг лунки
        for di in range(-m_phi, m_phi + 1):
            i = (i0 + di) % n_phi  # периодичность по φ!

            d_phi = phi[i] - centers_phi[k]
            # wrap to [-π, π]
            if d_phi > np.pi:
                d_phi -= 2 * np.pi
            elif d_phi < -np.pi:
                d_phi += 2 * np.pi

            for dj in range(-m_z, m_z + 1):
                j = j0 + dj
                if j < 0 or j >= n_z:
                    continue

                d_z = z_dim[j] - centers_z[k]
                r_sq = (d_phi * inv_b_R)**2 + (d_z * inv_a)**2

                if r_sq <= 1.0:
                    texture_field[i, j] += h_depth_star * np.sqrt(1.0 - r_sq)

    return texture_field


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

    ОБЁРТКА для совместимости: конвертирует Z в z_dim и вызывает scatter.

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
    # Конвертируем безразмерные Z в размерные z_dim
    n_z = len(Z)
    z_dim = np.empty(n_z)
    half_L = L / 2.0
    for j in range(n_z):
        z_dim[j] = Z[j] * half_L

    return _compute_texture_field_scatter(
        phi, z_dim, centers_phi, centers_z, R, a, b, h_depth_star
    )


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
    Fn: float,
    phi_min: float = 0.0,
    phi_max: float = 2 * np.pi
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Генерация центров лунок по филлотаксису (золотой угол).

    φ_k = (k × α) mod 2π,  α ≈ 137.508° (золотой угол)
    z_k = L × (k + 0.5) / N - L/2

    Args:
        R: радиус, м
        L: длина, м
        a, b: полуоси лунки, м
        Fn: плотность заполнения (от ВСЕЙ поверхности)
        phi_min, phi_max: границы сектора (рад)

    Returns:
        (centers_phi, centers_z)
    """
    # Площадь одной лунки
    S_cell = np.pi * a * b

    # Площадь поверхности цилиндра
    S_total = 2 * np.pi * R * L

    # Число лунок (от всей поверхности!)
    N = int(S_total * Fn / S_cell)
    if N < 1:
        N = 1

    # Золотой угол в радианах
    golden_angle = np.pi * (3 - np.sqrt(5))  # ≈ 137.508°

    # Проверяем, нужен ли сектор
    is_full = abs(phi_max - phi_min - 2*np.pi) < 0.01 or \
              (phi_min == 0 and phi_max >= 2*np.pi - 0.01)

    if is_full:
        # Полная окружность
        centers_phi = np.zeros(N)
        centers_z = np.zeros(N)

        for k in range(N):
            centers_phi[k] = (k * golden_angle) % (2 * np.pi)
            centers_z[k] = L * (k + 0.5) / N - L/2
    else:
        # Сектор — генерируем больше точек и фильтруем
        sector_frac = (phi_max - phi_min) / (2 * np.pi) if phi_max > phi_min \
                      else (2*np.pi - phi_min + phi_max) / (2 * np.pi)

        # Нужно примерно N * sector_frac точек в секторе
        N_target = max(1, int(N * sector_frac))

        centers_phi = []
        centers_z = []
        k = 0
        max_attempts = N * 10  # защита от бесконечного цикла

        while len(centers_phi) < N_target and k < max_attempts:
            phi_k = (k * golden_angle) % (2 * np.pi)
            z_k = L * (k + 0.5) / max(N, N_target) - L/2

            # Проверяем попадание в сектор
            in_sector = (phi_min <= phi_k <= phi_max) if phi_max > phi_min \
                        else (phi_k >= phi_min or phi_k <= phi_max)

            if in_sector and -L/2 + a <= z_k <= L/2 - a:
                centers_phi.append(phi_k)
                centers_z.append(z_k)

            k += 1

        centers_phi = np.array(centers_phi)
        centers_z = np.array(centers_z)

    return centers_phi, centers_z


def generate_spiral_centers(
    R: float,
    L: float,
    a: float,
    b: float,
    Fn: float,
    n_starts: int = 1,
    spiral_angle: Optional[float] = None,
    phi_min: float = 0.0,
    phi_max: float = 2 * np.pi
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Генерация центров лунок по многозаходной спирали.

    Args:
        R: радиус, м
        L: длина, м
        a, b: полуоси лунки, м
        Fn: плотность заполнения
        n_starts: число заходов спирали (T)
        spiral_angle: угол между точками (None = золотой угол)
        phi_min, phi_max: границы сектора

    Returns:
        (centers_phi, centers_z)
    """
    # Площадь одной лунки
    S_cell = np.pi * a * b

    # Площадь поверхности
    S_total = 2 * np.pi * R * L

    # Число лунок
    N_total = int(S_total * Fn / S_cell)
    if N_total < 1:
        N_total = 1

    # Угол между точками
    if spiral_angle is None:
        alpha = np.pi * (3 - np.sqrt(5))  # золотой угол
    else:
        alpha = spiral_angle

    # Шаг по оси
    c_axial = L / max(1, N_total // n_starts)

    centers_phi = []
    centers_z = []

    # Проверяем сектор
    is_full = abs(phi_max - phi_min - 2*np.pi) < 0.01 or \
              (phi_min == 0 and phi_max >= 2*np.pi - 0.01)

    for t in range(n_starts):
        phi_0 = 2 * np.pi * t / n_starts  # начальный угол захода

        n_points = N_total // n_starts
        for n in range(n_points):
            phi_n = (phi_0 + n * alpha) % (2 * np.pi)
            z_n = -L/2 + a + n * c_axial

            if z_n > L/2 - a:
                break

            # Проверка сектора
            in_sector = is_full or \
                        ((phi_min <= phi_n <= phi_max) if phi_max > phi_min
                         else (phi_n >= phi_min or phi_n <= phi_max))

            if in_sector:
                centers_phi.append(phi_n)
                centers_z.append(z_n)

    return np.array(centers_phi), np.array(centers_z)


def validate_texture(
    centers_phi: np.ndarray,
    centers_z: np.ndarray,
    a: float,
    b: float,
    R: float,
    L: float,
    phi_min: float = 0.0,
    phi_max: float = 2 * np.pi
) -> Tuple[bool, str]:
    """
    Проверка валидности текстуры.

    Args:
        centers_phi: углы центров лунок, рад
        centers_z: z-координаты центров, м
        a, b: полуоси эллипса, м
        R: радиус подшипника, м
        L: длина подшипника, м
        phi_min, phi_max: границы сектора

    Returns:
        (valid, message)
    """
    n = len(centers_phi)

    if n == 0:
        return True, "No dimples"

    # 1. Проверка границ (торцы)
    for i in range(n):
        z = centers_z[i]
        if z < -L/2 + a or z > L/2 - a:
            return False, f"Лунка {i} выходит за торец: z={z*1000:.2f} мм"

    # 2. Проверка сектора
    is_full = abs(phi_max - phi_min - 2*np.pi) < 0.01 or \
              (phi_min == 0 and phi_max >= 2*np.pi - 0.01)

    if not is_full:
        for i in range(n):
            phi = centers_phi[i] % (2 * np.pi)
            in_sector = (phi_min <= phi <= phi_max) if phi_max > phi_min \
                        else (phi >= phi_min or phi <= phi_max)
            if not in_sector:
                return False, f"Лунка {i} вне сектора: φ={np.degrees(phi):.1f}°"

    # 3. Проверка неперекрытия эллипсов
    for i in range(n):
        phi_i, z_i = centers_phi[i], centers_z[i]
        for j in range(i + 1, n):
            phi_j, z_j = centers_phi[j], centers_z[j]

            # Расстояние по окружности с периодичностью
            dphi = abs(phi_i - phi_j)
            dphi = min(dphi, 2 * np.pi - dphi)
            dx = R * dphi
            dz = abs(z_i - z_j)

            # Критерий перекрытия эллипсов: (dx/(2b))² + (dz/(2a))² >= 1
            overlap_param = (dx / (2 * b))**2 + (dz / (2 * a))**2
            if overlap_param < 1.0:
                return False, f"Перекрытие лунок {i} и {j}: param={overlap_param:.3f}"

    return True, "OK"


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

        # Безразмерная глубина (используем h_star или вычисляем из h_depth)
        self.h_depth_star = self.params.get_h_star(config.c)

        # Генерируем центры лунок
        if self.params.pattern == 'regular':
            self.centers_phi, self.centers_z = generate_regular_centers(
                R=config.R, L=config.L,
                a=self.params.a, b=self.params.b,
                Fn=self.params.Fn, theta=self.params.theta
            )
        elif self.params.pattern == 'spiral':
            self.centers_phi, self.centers_z = generate_spiral_centers(
                R=config.R, L=config.L,
                a=self.params.a, b=self.params.b,
                Fn=self.params.Fn,
                n_starts=self.params.n_starts,
                spiral_angle=self.params.spiral_angle,
                phi_min=self.params.phi_min,
                phi_max=self.params.phi_max
            )
        else:  # phyllotaxis
            self.centers_phi, self.centers_z = generate_phyllotaxis_centers(
                R=config.R, L=config.L,
                a=self.params.a, b=self.params.b,
                Fn=self.params.Fn,
                phi_min=self.params.phi_min,
                phi_max=self.params.phi_max
            )

        self.N_cells = len(self.centers_phi)

        # Валидация текстуры
        self._is_valid, self._validation_msg = validate_texture(
            self.centers_phi, self.centers_z,
            self.params.a, self.params.b,
            config.R, config.L,
            self.params.phi_min, self.params.phi_max
        )

        # Кешированное поле текстуры (вычисляется при первом вызове)
        self._texture_field: Optional[np.ndarray] = None
        self._cached_phi: Optional[np.ndarray] = None
        self._cached_Z: Optional[np.ndarray] = None

    @property
    def is_valid(self) -> bool:
        """Валидна ли текстура (нет перекрытий, в границах)."""
        return self._is_valid

    @property
    def validation_message(self) -> str:
        """Сообщение о валидации."""
        return self._validation_msg

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
        h_depth_m = self.params.get_h_depth(self.config.c)
        return {
            'N_cells': self.N_cells,
            'a_mm': self.params.a * 1000,
            'b_mm': self.params.b * 1000,
            'h_depth_um': h_depth_m * 1e6,
            'h_star': self.h_depth_star,
            'Fn': self.params.Fn,
            'pattern': self.params.pattern,
            'Fn_actual': self._compute_actual_Fn(),
            'is_valid': self._is_valid,
            'validation_msg': self._validation_msg,
            'phi_min_deg': np.degrees(self.params.phi_min),
            'phi_max_deg': np.degrees(self.params.phi_max),
            'is_full_circumference': self.params.is_full_circumference,
        }

    def _compute_actual_Fn(self) -> float:
        """Вычислить фактическую плотность заполнения."""
        S_cell = np.pi * self.params.a * self.params.b
        S_total = 2 * np.pi * self.config.R * self.config.L
        return self.N_cells * S_cell / S_total

    def texture_mask(self, phi: np.ndarray, Z: np.ndarray) -> np.ndarray:
        """
        Вычислить маску текстуры: True внутри лунки, False вне.

        Args:
            phi: углы сетки, shape (n_phi,)
            Z: осевые координаты [-1, 1], shape (n_z,)

        Returns:
            mask: булева маска, shape (n_phi, n_z)
        """
        return _compute_texture_mask_numba(
            phi, Z,
            self.centers_phi, self.centers_z,
            self.config.R, self.config.L,
            self.params.a, self.params.b
        )


@njit(cache=True)
def _compute_texture_mask_numba(
    phi: np.ndarray,
    Z: np.ndarray,
    centers_phi: np.ndarray,
    centers_z: np.ndarray,
    R: float,
    L: float,
    a: float,
    b: float
) -> np.ndarray:
    """
    Вычислить маску текстуры (Numba-ускорено).

    Точка внутри лунки, если r² = (Δφ·R/b)² + (Δz/a)² <= 1

    Returns:
        mask: булева маска, shape (n_phi, n_z)
    """
    n_phi = len(phi)
    n_z = len(Z)
    N = len(centers_phi)

    mask = np.zeros((n_phi, n_z), dtype=np.bool_)

    for i in range(n_phi):
        for j in range(n_z):
            # Размерная z-координата текущей точки
            z_dim = Z[j] * (L / 2)

            for k in range(N):
                # Разность углов с учётом периодичности
                d_phi = _wrap_to_pi(phi[i] - centers_phi[k])

                # Разность по z (размерная)
                d_z = z_dim - centers_z[k]

                # Нормированное расстояние до центра лунки
                r_sq = (d_phi * R / b)**2 + (d_z / a)**2

                if r_sq <= 1.0:
                    mask[i, j] = True
                    break  # Уже внутри какой-то лунки

    return mask
