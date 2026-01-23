"""
Решатель уравнения Рейнольдса для гидродинамического подшипника.

Безразмерное уравнение:
    ∂/∂φ(H³ ∂P/∂φ) + (D/L)² ∂/∂Z(H³ ∂P/∂Z) = ∂H/∂φ + 2·∂H/∂t*

Граничные условия:
    - По φ: периодичность P(φ=0) = P(φ=2π)
    - По Z: P(Z=±1) = 0 (атмосферное давление на торцах)
    - Кавитация: P ≥ 0

Squeeze-член (∂H/∂t*):
    Для расчёта демпфирования используется:
    ∂H/∂t* = vx*·cos(φ) + vy*·sin(φ)
    где vx* = (R/Uc)·ẋ, vy* = (R/Uc)·ẏ

Ускорение через Numba JIT для параметрических исследований.
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np
from numba import njit

from .config import BearingConfig
from .film_models import FilmModel, SmoothFilmModel


@dataclass
class ReynoldsResult:
    """
    Результат решения уравнения Рейнольдса.

    Все поля в безразмерных координатах, кроме явно указанных.
    """
    # Сетка
    phi: np.ndarray          # углы [0, 2π), shape (n_phi,)
    Z: np.ndarray            # осевые координаты [-1, 1], shape (n_z,)

    # Поля
    P: np.ndarray            # безразмерное давление, shape (n_phi, n_z)
    H: np.ndarray            # безразмерная толщина плёнки, shape (n_phi, n_z)

    # Характеристики
    h_min: float             # минимальная толщина плёнки, м (размерная!)
    h_min_dimless: float     # минимальная безразмерная толщина
    P_max: float             # максимальное безразмерное давление
    p_max: float             # максимальное размерное давление, Па

    # Информация о решении
    converged: bool
    iterations: int
    residual: float

    def get_dimensional_pressure(self, config: BearingConfig) -> np.ndarray:
        """Получить размерное поле давления, Па."""
        return self.P * config.pressure_scale


# ============================================================================
# Numba-ускоренные функции для SOR
# ============================================================================

@njit(cache=True)
def _precompute_H3_halves(H3: np.ndarray, n_phi: int, n_z: int):
    """
    Предвычислить H³ на полуцелых узлах.

    Returns:
        H3_phi_plus, H3_phi_minus, H3_z_plus, H3_z_minus
    """
    H3_phi_plus = np.zeros((n_phi, n_z))
    H3_phi_minus = np.zeros((n_phi, n_z))
    H3_z_plus = np.zeros((n_phi, n_z))
    H3_z_minus = np.zeros((n_phi, n_z))

    for i in range(n_phi):
        i_plus = (i + 1) % n_phi
        i_minus = (i - 1 + n_phi) % n_phi
        for j in range(n_z):
            H3_phi_plus[i, j] = 0.5 * (H3[i, j] + H3[i_plus, j])
            H3_phi_minus[i, j] = 0.5 * (H3[i, j] + H3[i_minus, j])
            if j < n_z - 1:
                H3_z_plus[i, j] = 0.5 * (H3[i, j] + H3[i, j + 1])
            if j > 0:
                H3_z_minus[i, j] = 0.5 * (H3[i, j] + H3[i, j - 1])

    return H3_phi_plus, H3_phi_minus, H3_z_plus, H3_z_minus


@njit(cache=True)
def _sor_iteration(
    P: np.ndarray,
    rhs: np.ndarray,
    H3_phi_plus: np.ndarray,
    H3_phi_minus: np.ndarray,
    H3_z_plus: np.ndarray,
    H3_z_minus: np.ndarray,
    a_phi: float,
    a_z: float,
    omega: float,
    n_phi: int,
    n_z: int
) -> float:
    """
    Одна итерация SOR.

    Args:
        rhs: правая часть уравнения (∂H/∂φ + 2·∂H/∂t*)

    Возвращает максимальное изменение (для проверки сходимости).
    """
    max_change = 0.0

    # Gauss-Seidel с релаксацией (последовательный обход для SOR!)
    for i in range(n_phi):
        i_plus = (i + 1) % n_phi
        i_minus = (i - 1 + n_phi) % n_phi

        for j in range(1, n_z - 1):  # исключаем границы по Z
            # Коэффициенты
            c_i_plus = a_phi * H3_phi_plus[i, j]
            c_i_minus = a_phi * H3_phi_minus[i, j]
            c_j_plus = a_z * H3_z_plus[i, j]
            c_j_minus = a_z * H3_z_minus[i, j]
            c_center = c_i_plus + c_i_minus + c_j_plus + c_j_minus

            if c_center < 1e-15:
                continue

            # Сумма соседей
            P_sum = (c_i_plus * P[i_plus, j] +
                     c_i_minus * P[i_minus, j] +
                     c_j_plus * P[i, j + 1] +
                     c_j_minus * P[i, j - 1])

            # Новое значение
            P_new = (P_sum - rhs[i, j]) / c_center

            # SOR с релаксацией
            P_new = P[i, j] + omega * (P_new - P[i, j])

            # Кавитация: P ≥ 0
            if P_new < 0.0:
                P_new = 0.0

            change = abs(P_new - P[i, j])
            if change > max_change:
                max_change = change

            P[i, j] = P_new

    return max_change


@njit(cache=True)
def _compute_residual(
    P: np.ndarray,
    H3: np.ndarray,
    rhs: np.ndarray,
    a_phi: float,
    a_z: float,
    n_phi: int,
    n_z: int
) -> float:
    """
    Вычислить среднюю невязку PDE только для активной зоны (P > 0).

    Args:
        rhs: правая часть уравнения (∂H/∂φ + 2·∂H/∂t*)
    """
    sum_residual = 0.0
    count_active = 0

    for i in range(n_phi):
        i_plus = (i + 1) % n_phi
        i_minus = (i - 1 + n_phi) % n_phi

        for j in range(1, n_z - 1):
            if P[i, j] <= 0:
                continue

            count_active += 1

            # H³ на полуцелых узлах
            H3_ip = 0.5 * (H3[i, j] + H3[i_plus, j])
            H3_im = 0.5 * (H3[i, j] + H3[i_minus, j])
            H3_jp = 0.5 * (H3[i, j] + H3[i, j + 1])
            H3_jm = 0.5 * (H3[i, j] + H3[i, j - 1])

            # Дискретный оператор
            lhs = (a_phi * (H3_ip * (P[i_plus, j] - P[i, j]) -
                           H3_im * (P[i, j] - P[i_minus, j])) +
                   a_z * (H3_jp * (P[i, j + 1] - P[i, j]) -
                          H3_jm * (P[i, j] - P[i, j - 1])))

            res = abs(lhs - rhs[i, j])
            sum_residual += res

    if count_active > 0:
        return sum_residual / count_active
    return 0.0


@njit(cache=True)
def _solve_sor_numba(
    H: np.ndarray,
    rhs: np.ndarray,
    d_phi: float,
    d_Z: float,
    D_L_sq: float,
    n_phi: int,
    n_z: int,
    omega: float,
    max_iter: int,
    min_iter: int,
    tol: float
) -> tuple:
    """
    Решение методом SOR с Numba.

    Args:
        rhs: правая часть уравнения (∂H/∂φ + 2·∂H/∂t*)

    Returns:
        P: поле давления
        converged: сходимость
        iterations: число итераций
        residual: невязка
    """
    P = np.zeros((n_phi, n_z))
    H3 = H ** 3

    # Коэффициенты
    a_phi = 1.0 / (d_phi ** 2)
    a_z = D_L_sq / (d_Z ** 2)

    # Предвычисляем H³ на полуцелых узлах
    H3_phi_plus, H3_phi_minus, H3_z_plus, H3_z_minus = _precompute_H3_halves(H3, n_phi, n_z)

    converged = False
    iterations = 0

    for iteration in range(max_iter):
        max_change = _sor_iteration(
            P, rhs,
            H3_phi_plus, H3_phi_minus,
            H3_z_plus, H3_z_minus,
            a_phi, a_z, omega,
            n_phi, n_z
        )

        iterations = iteration + 1

        # Сходимость только после минимума итераций
        if iteration >= min_iter and max_change < tol:
            converged = True
            break

    # Вычисляем невязку PDE
    residual = _compute_residual(P, H3, rhs, a_phi, a_z, n_phi, n_z)

    return P, converged, iterations, residual


# ============================================================================
# Основной класс решателя
# ============================================================================

class ReynoldsSolver:
    """
    Решатель уравнения Рейнольдса методом конечных разностей.

    Поддерживает:
    - SOR-итерации с Numba-ускорением (по умолчанию)
    - Кавитация через условие P ≥ 0
    """

    def __init__(
        self,
        config: BearingConfig,
        film_model: Optional[FilmModel] = None,
    ):
        """
        Args:
            config: конфигурация подшипника
            film_model: модель толщины плёнки (по умолчанию SmoothFilmModel)
        """
        self.config = config
        self.film_model = film_model or SmoothFilmModel(config)

        # Параметры SOR
        self.omega_sor = 1.7      # параметр релаксации
        self.max_iter = 10000     # максимум итераций
        self.min_iter = 100       # минимум итераций
        self.tol = 1e-8           # допуск сходимости

    def solve(self, dH_dt_star: Optional[np.ndarray] = None) -> ReynoldsResult:
        """
        Решить уравнение Рейнольдса.

        Args:
            dH_dt_star: безразмерная скорость изменения толщины плёнки
                        (squeeze-член для расчёта демпфирования).
                        Если задано, RHS = ∂H/∂φ + 2·∂H/∂t*
                        Формула: dH_dt_star = vx*·cos(φ) + vy*·sin(φ)

        Returns:
            ReynoldsResult с полями давления и толщины плёнки
        """
        # Создаём сетку
        phi, Z, d_phi, d_Z = self.config.create_grid()
        n_phi = len(phi)
        n_z = len(Z)

        # Вычисляем толщину плёнки через film_model (callbacks)
        H = self.film_model.H(phi, Z)
        dH_dphi = self.film_model.dH_dphi(phi, Z)

        # Правая часть уравнения Рейнольдса
        rhs = dH_dphi.copy()
        if dH_dt_star is not None:
            rhs = rhs + 2.0 * dH_dt_star

        # Параметр (D/L)²
        D_L_sq = self.config.D_L_ratio ** 2

        # Решаем SOR с Numba
        P, converged, iterations, residual = _solve_sor_numba(
            H, rhs, d_phi, d_Z, D_L_sq,
            n_phi, n_z,
            self.omega_sor, self.max_iter, self.min_iter, self.tol
        )

        # Применяем условие кавитации (на всякий случай)
        P = np.maximum(P, 0.0)

        # Вычисляем характеристики
        h_min_dimless = np.min(H)
        h_min = h_min_dimless * self.config.c
        P_max = np.max(P)
        p_max = P_max * self.config.pressure_scale

        return ReynoldsResult(
            phi=phi,
            Z=Z,
            P=P,
            H=H,
            h_min=h_min,
            h_min_dimless=h_min_dimless,
            P_max=P_max,
            p_max=p_max,
            converged=converged,
            iterations=iterations,
            residual=residual,
        )


def solve_reynolds(
    config: BearingConfig,
    film_model: Optional[FilmModel] = None,
    dH_dt_star: Optional[np.ndarray] = None,
) -> ReynoldsResult:
    """
    Удобная функция для решения уравнения Рейнольдса.

    Args:
        config: конфигурация подшипника
        film_model: модель толщины плёнки
        dH_dt_star: безразмерная скорость изменения толщины плёнки
                    (squeeze-член для расчёта демпфирования)

    Returns:
        ReynoldsResult
    """
    solver = ReynoldsSolver(config, film_model)
    return solver.solve(dH_dt_star=dH_dt_star)
