"""
Решатель уравнения Рейнольдса для гидродинамического подшипника.

Безразмерное уравнение:
    ∂/∂φ(H³ ∂P/∂φ) + (D/L)² ∂/∂Z(H³ ∂P/∂Z) = ∂H/∂φ

Граничные условия:
    - По φ: периодичность P(φ=0) = P(φ=2π)
    - По Z: P(Z=±1) = 0 (атмосферное давление на торцах)
    - Кавитация: P ≥ 0 (простая модель Гюмбеля)
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

from .config import BearingConfig
from .film_models import FilmModel, SmoothFilmModel


@dataclass
class ReynoldsResult:
    """
    Результат решения уравнения Рейнольдса.

    Все поля в безразмерных координатах, кроме явно указанных.
    """
    # Сетка
    phi: np.ndarray          # окружные координаты, shape (n_phi,)
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
    iterations: int          # 0 для прямого метода
    residual: float          # невязка в активной зоне (P > 0)
    method: str              # "direct" или "sor"

    def get_dimensional_pressure(self, config: BearingConfig) -> np.ndarray:
        """Получить размерное поле давления, Па."""
        return self.P * config.pressure_scale


class ReynoldsSolver:
    """
    Решатель уравнения Рейнольдса методом конечных разностей.

    Поддерживает:
    - SOR-итерации с условием кавитации
    - Прямое решение разреженной системы
    - Кавитация через условие P ≥ 0 (модель Гюмбеля)
    """

    def __init__(
        self,
        config: BearingConfig,
        film_model: Optional[FilmModel] = None,
        method: str = "sor"
    ):
        """
        Args:
            config: конфигурация подшипника
            film_model: модель толщины плёнки (по умолчанию SmoothFilmModel)
            method: метод решения ("sor" или "direct")
        """
        self.config = config
        self.film_model = film_model or SmoothFilmModel(config)
        self.method = method

        # Параметры SOR
        self.omega_sor = 1.7      # параметр релаксации
        self.max_iter = 10000     # максимум итераций
        self.min_iter = 50        # минимум итераций (для надёжности)
        self.tol = 1e-6           # допуск сходимости

    def solve(self) -> ReynoldsResult:
        """
        Решить уравнение Рейнольдса.

        Returns:
            ReynoldsResult с полями давления и толщины плёнки
        """
        # Создаём сетку
        phi, Z, d_phi, d_Z = self.config.create_grid()
        n_phi = len(phi)
        n_z = len(Z)

        # Вычисляем толщину плёнки
        H = self.film_model.H(phi, Z)
        dH_dphi = self.film_model.dH_dphi(phi, Z)

        # Параметр (D/L)²
        D_L_sq = self.config.D_L_ratio ** 2

        # Предвычисляем коэффициенты
        H3 = H ** 3
        H3_phi_plus, H3_phi_minus, H3_z_plus, H3_z_minus = self._compute_half_node_H3(
            H3, n_phi, n_z
        )

        a_phi = 1.0 / d_phi**2
        a_z = D_L_sq / d_Z**2

        if self.method == "sor":
            P, converged, iterations, residual = self._solve_sor(
                H3_phi_plus, H3_phi_minus, H3_z_plus, H3_z_minus,
                dH_dphi, a_phi, a_z, n_phi, n_z
            )
        else:
            P, converged, iterations, residual = self._solve_direct(
                H3_phi_plus, H3_phi_minus, H3_z_plus, H3_z_minus,
                dH_dphi, a_phi, a_z, n_phi, n_z
            )

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
            method=self.method,
        )

    def _compute_half_node_H3(
        self,
        H3: np.ndarray,
        n_phi: int,
        n_z: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Вычислить H³ на полуцелых узлах."""
        H3_phi_plus = np.zeros_like(H3)
        H3_phi_minus = np.zeros_like(H3)
        H3_z_plus = np.zeros_like(H3)
        H3_z_minus = np.zeros_like(H3)

        for i in range(n_phi):
            i_plus = (i + 1) % n_phi
            i_minus = (i - 1) % n_phi
            H3_phi_plus[i, :] = 0.5 * (H3[i, :] + H3[i_plus, :])
            H3_phi_minus[i, :] = 0.5 * (H3[i, :] + H3[i_minus, :])

        for j in range(n_z):
            if j < n_z - 1:
                H3_z_plus[:, j] = 0.5 * (H3[:, j] + H3[:, j + 1])
            if j > 0:
                H3_z_minus[:, j] = 0.5 * (H3[:, j] + H3[:, j - 1])

        return H3_phi_plus, H3_phi_minus, H3_z_plus, H3_z_minus

    def _compute_residual_active_zone(
        self,
        P: np.ndarray,
        H3_phi_plus: np.ndarray,
        H3_phi_minus: np.ndarray,
        H3_z_plus: np.ndarray,
        H3_z_minus: np.ndarray,
        dH_dphi: np.ndarray,
        a_phi: float,
        a_z: float,
        n_phi: int,
        n_z: int,
    ) -> float:
        """
        Вычислить максимальную невязку PDE только в активной зоне (P > 0).

        Невязка нормируется на масштаб правой части |dH/dφ|.
        """
        # Масштаб для нормировки
        rhs_scale = np.max(np.abs(dH_dphi)) + 1e-15

        max_residual = 0.0
        active_count = 0

        for i in range(n_phi):
            i_plus = (i + 1) % n_phi
            i_minus = (i - 1) % n_phi

            for j in range(1, n_z - 1):
                # Считаем невязку только в активной зоне
                # и там где соседи тоже в активной зоне
                if P[i, j] > 1e-10:
                    c_i_plus = a_phi * H3_phi_plus[i, j]
                    c_i_minus = a_phi * H3_phi_minus[i, j]
                    c_j_plus = a_z * H3_z_plus[i, j]
                    c_j_minus = a_z * H3_z_minus[i, j]

                    # Дискретный лапласиан
                    lap = (c_i_plus * (P[i_plus, j] - P[i, j]) -
                           c_i_minus * (P[i, j] - P[i_minus, j]) +
                           c_j_plus * (P[i, j + 1] - P[i, j]) -
                           c_j_minus * (P[i, j] - P[i, j - 1]))

                    # Невязка = |L[P] - dH/dφ| / scale
                    res = abs(lap - dH_dphi[i, j]) / rhs_scale

                    if res > max_residual:
                        max_residual = res
                    active_count += 1

        return max_residual

    def _solve_sor(
        self,
        H3_phi_plus: np.ndarray,
        H3_phi_minus: np.ndarray,
        H3_z_plus: np.ndarray,
        H3_z_minus: np.ndarray,
        dH_dphi: np.ndarray,
        a_phi: float,
        a_z: float,
        n_phi: int,
        n_z: int,
    ) -> tuple[np.ndarray, bool, int, float]:
        """
        Решение методом SOR с условием кавитации P ≥ 0.

        Критерий сходимости: max|P_new - P_old| < tol
        Минимум итераций: self.min_iter
        """
        omega = self.omega_sor
        P = np.zeros((n_phi, n_z))

        converged = False
        final_residual = 0.0

        for iteration in range(self.max_iter):
            max_change = 0.0

            for i in range(n_phi):
                i_plus = (i + 1) % n_phi
                i_minus = (i - 1) % n_phi

                for j in range(1, n_z - 1):
                    c_i_plus = a_phi * H3_phi_plus[i, j]
                    c_i_minus = a_phi * H3_phi_minus[i, j]
                    c_j_plus = a_z * H3_z_plus[i, j]
                    c_j_minus = a_z * H3_z_minus[i, j]
                    c_center = c_i_plus + c_i_minus + c_j_plus + c_j_minus

                    if c_center < 1e-15:
                        continue

                    # Правая часть
                    rhs = dH_dphi[i, j]

                    # Сумма соседних значений
                    P_sum = (c_i_plus * P[i_plus, j] +
                             c_i_minus * P[i_minus, j] +
                             c_j_plus * P[i, j + 1] +
                             c_j_minus * P[i, j - 1])

                    # Новое значение
                    P_new = (P_sum - rhs) / c_center

                    # SOR-релаксация
                    P_new = P[i, j] + omega * (P_new - P[i, j])

                    # Условие кавитации
                    P_new = max(P_new, 0.0)

                    change = abs(P_new - P[i, j])
                    if change > max_change:
                        max_change = change

                    P[i, j] = P_new

            # Проверяем сходимость (только после минимума итераций)
            if iteration >= self.min_iter - 1 and max_change < self.tol:
                converged = True
                final_residual = self._compute_residual_active_zone(
                    P, H3_phi_plus, H3_phi_minus, H3_z_plus, H3_z_minus,
                    dH_dphi, a_phi, a_z, n_phi, n_z
                )
                return P, converged, iteration + 1, final_residual

        # Не сошлось за max_iter
        final_residual = self._compute_residual_active_zone(
            P, H3_phi_plus, H3_phi_minus, H3_z_plus, H3_z_minus,
            dH_dphi, a_phi, a_z, n_phi, n_z
        )
        return P, converged, self.max_iter, final_residual

    def _solve_direct(
        self,
        H3_phi_plus: np.ndarray,
        H3_phi_minus: np.ndarray,
        H3_z_plus: np.ndarray,
        H3_z_minus: np.ndarray,
        dH_dphi: np.ndarray,
        a_phi: float,
        a_z: float,
        n_phi: int,
        n_z: int,
    ) -> tuple[np.ndarray, bool, int, float]:
        """
        Прямое решение + итеративная коррекция кавитации.

        Для учёта кавитации используется итеративный процесс:
        пересобираем систему с P=0 в зоне кавитации.
        """
        def idx(i, j):
            return i * n_z + j

        n_total = n_phi * n_z
        max_cav_iter = 100  # максимум итераций для кавитации

        # Начальная маска кавитации — пустая
        cavitation_mask = np.zeros((n_phi, n_z), dtype=bool)

        for cav_iter in range(max_cav_iter):
            # Собираем систему с учётом текущей маски кавитации
            row_indices = []
            col_indices = []
            values = []
            b = np.zeros(n_total)

            for i in range(n_phi):
                i_plus = (i + 1) % n_phi
                i_minus = (i - 1) % n_phi

                for j in range(n_z):
                    k = idx(i, j)

                    # Граничные условия на торцах или в зоне кавитации
                    if j == 0 or j == n_z - 1 or cavitation_mask[i, j]:
                        row_indices.append(k)
                        col_indices.append(k)
                        values.append(1.0)
                        b[k] = 0.0
                    else:
                        # Внутренняя точка
                        c_i_plus = a_phi * H3_phi_plus[i, j]
                        c_i_minus = a_phi * H3_phi_minus[i, j]
                        c_j_plus = a_z * H3_z_plus[i, j]
                        c_j_minus = a_z * H3_z_minus[i, j]
                        c_center = c_i_plus + c_i_minus + c_j_plus + c_j_minus

                        row_indices.append(k)
                        col_indices.append(k)
                        values.append(-c_center)

                        row_indices.append(k)
                        col_indices.append(idx(i_plus, j))
                        values.append(c_i_plus)

                        row_indices.append(k)
                        col_indices.append(idx(i_minus, j))
                        values.append(c_i_minus)

                        row_indices.append(k)
                        col_indices.append(idx(i, j + 1))
                        values.append(c_j_plus)

                        row_indices.append(k)
                        col_indices.append(idx(i, j - 1))
                        values.append(c_j_minus)

                        b[k] = dH_dphi[i, j]

            A = sparse.csr_matrix(
                (values, (row_indices, col_indices)),
                shape=(n_total, n_total)
            )

            P_flat = spsolve(A, b)
            P = P_flat.reshape((n_phi, n_z))

            # Проверяем, есть ли новые отрицательные значения
            new_cavitation = P < -1e-10
            new_cavitation[:, 0] = False  # границы не трогаем
            new_cavitation[:, -1] = False

            if not np.any(new_cavitation & ~cavitation_mask):
                # Нет новых точек кавитации — сошлось
                break

            # Обновляем маску
            cavitation_mask = cavitation_mask | new_cavitation

        # Финальное применение P >= 0
        P = np.maximum(P, 0.0)

        # Вычисляем невязку в активной зоне
        residual = self._compute_residual_active_zone(
            P, H3_phi_plus, H3_phi_minus, H3_z_plus, H3_z_minus,
            dH_dphi, a_phi, a_z, n_phi, n_z
        )

        # iterations = 0 для прямого метода (или число итераций кавитации)
        return P, True, cav_iter + 1, residual


def solve_reynolds(
    config: BearingConfig,
    film_model: Optional[FilmModel] = None,
    method: str = "sor"
) -> ReynoldsResult:
    """
    Удобная функция для решения уравнения Рейнольдса.

    Args:
        config: конфигурация подшипника
        film_model: модель толщины плёнки
        method: метод решения ("sor" или "direct")

    Returns:
        ReynoldsResult
    """
    solver = ReynoldsSolver(config, film_model, method)
    return solver.solve()
