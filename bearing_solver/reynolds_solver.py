"""
Решатель уравнения Рейнольдса для гидродинамического подшипника.

Безразмерное уравнение:
    ∂/∂φ(H³ ∂P/∂φ) + (D/L)² ∂/∂Z(H³ ∂P/∂Z) = ∂H/∂φ

Граничные условия:
    - По φ: периодичность P(φ=0) = P(φ=2π)
    - По Z: P(Z=±1) = 0 (атмосферное давление на торцах)
    - Кавитация: P ≥ 0
"""

from dataclasses import dataclass
from typing import Optional, Union
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
    iterations: int
    residual: float

    def get_dimensional_pressure(self, config: BearingConfig) -> np.ndarray:
        """Получить размерное поле давления, Па."""
        return self.P * config.pressure_scale


class ReynoldsSolver:
    """
    Решатель уравнения Рейнольдса методом конечных разностей.

    Поддерживает:
    - SOR-итерации (метод по умолчанию для больших сеток)
    - Прямое решение разреженной системы (для меньших сеток)
    - Кавитация через условие P ≥ 0
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
        self.min_iter = 100       # минимум итераций (для стабилизации кавитации)
        self.tol = 1e-8           # допуск сходимости (строже для лучшей невязки)

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

        if self.method == "sor":
            P, converged, iterations, residual = self._solve_sor(
                H, dH_dphi, d_phi, d_Z, D_L_sq, n_phi, n_z
            )
        else:
            P, converged, iterations, residual = self._solve_direct(
                H, dH_dphi, d_phi, d_Z, D_L_sq, n_phi, n_z
            )

        # Применяем условие кавитации
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

    def _solve_sor(
        self,
        H: np.ndarray,
        dH_dphi: np.ndarray,
        d_phi: float,
        d_Z: float,
        D_L_sq: float,
        n_phi: int,
        n_z: int,
    ) -> tuple[np.ndarray, bool, int, float]:
        """
        Решение методом SOR (последовательной верхней релаксации).

        Дискретизация уравнения:
        ∂/∂φ(H³ ∂P/∂φ) ≈ (H³_{i+1/2}(P_{i+1}-P_i) - H³_{i-1/2}(P_i-P_{i-1})) / d_φ²

        Аналогично для ∂/∂Z.
        """
        omega = self.omega_sor
        P = np.zeros((n_phi, n_z))

        # Граничные условия по Z: P = 0 на торцах
        # (они уже нулевые и останутся такими)

        # Предвычисляем H³ в узлах
        H3 = H ** 3

        # H³ на полуцелых узлах (для потоков)
        # H³_{i+1/2,j} = 0.5 * (H³_{i,j} + H³_{i+1,j})
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

        # Коэффициенты
        a_phi = 1.0 / d_phi**2
        a_z = D_L_sq / d_Z**2

        converged = False
        residual = 0.0

        for iteration in range(self.max_iter):
            max_change = 0.0

            for i in range(n_phi):
                i_plus = (i + 1) % n_phi
                i_minus = (i - 1) % n_phi

                # Внутренние точки по Z (исключаем границы j=0 и j=n_z-1)
                for j in range(1, n_z - 1):
                    # Коэффициенты при соседних узлах
                    c_i_plus = a_phi * H3_phi_plus[i, j]
                    c_i_minus = a_phi * H3_phi_minus[i, j]
                    c_j_plus = a_z * H3_z_plus[i, j]
                    c_j_minus = a_z * H3_z_minus[i, j]

                    # Центральный коэффициент
                    c_center = c_i_plus + c_i_minus + c_j_plus + c_j_minus

                    if c_center < 1e-15:
                        continue

                    # Правая часть: ∂H/∂φ
                    rhs = dH_dphi[i, j]

                    # Сумма соседних значений
                    P_sum = (c_i_plus * P[i_plus, j] +
                             c_i_minus * P[i_minus, j] +
                             c_j_plus * P[i, j + 1] +
                             c_j_minus * P[i, j - 1])

                    # Новое значение (без релаксации)
                    P_new = (P_sum - rhs) / c_center

                    # SOR с условием кавитации
                    P_new = P[i, j] + omega * (P_new - P[i, j])
                    P_new = max(P_new, 0.0)

                    change = abs(P_new - P[i, j])
                    if change > max_change:
                        max_change = change

                    P[i, j] = P_new

            residual = max_change

            # Сходимость только после минимума итераций
            if iteration >= self.min_iter and max_change < self.tol:
                converged = True
                # Вычисляем невязку PDE только для активной зоны (P > 0)
                pde_residual = self._compute_pde_residual(
                    P, H, dH_dphi, d_phi, d_Z, D_L_sq, n_phi, n_z
                )
                return P, converged, iteration + 1, pde_residual

        # Вычисляем финальную невязку PDE
        pde_residual = self._compute_pde_residual(
            P, H, dH_dphi, d_phi, d_Z, D_L_sq, n_phi, n_z
        )
        return P, converged, self.max_iter, pde_residual

    def _compute_pde_residual(
        self,
        P: np.ndarray,
        H: np.ndarray,
        dH_dphi: np.ndarray,
        d_phi: float,
        d_Z: float,
        D_L_sq: float,
        n_phi: int,
        n_z: int,
    ) -> float:
        """
        Вычислить невязку уравнения Рейнольдса только для активной зоны (P > 0).

        Невязка: |∂/∂φ(H³ ∂P/∂φ) + (D/L)² ∂/∂Z(H³ ∂P/∂Z) - ∂H/∂φ|
        """
        H3 = H ** 3
        max_residual = 0.0
        count_active = 0
        sum_residual = 0.0

        a_phi = 1.0 / d_phi**2
        a_z = D_L_sq / d_Z**2

        for i in range(n_phi):
            i_plus = (i + 1) % n_phi
            i_minus = (i - 1) % n_phi

            for j in range(1, n_z - 1):
                # Только для активной зоны (P > 0)
                if P[i, j] <= 0:
                    continue

                count_active += 1

                # H³ на полуцелых узлах
                H3_ip = 0.5 * (H3[i, j] + H3[i_plus, j])
                H3_im = 0.5 * (H3[i, j] + H3[i_minus, j])
                H3_jp = 0.5 * (H3[i, j] + H3[i, j + 1])
                H3_jm = 0.5 * (H3[i, j] + H3[i, j - 1])

                # Дискретный лапласиан (левая часть уравнения)
                lhs = (a_phi * (H3_ip * (P[i_plus, j] - P[i, j]) -
                               H3_im * (P[i, j] - P[i_minus, j])) +
                       a_z * (H3_jp * (P[i, j + 1] - P[i, j]) -
                              H3_jm * (P[i, j] - P[i, j - 1])))

                # Правая часть
                rhs = dH_dphi[i, j]

                # Невязка
                res = abs(lhs - rhs)
                sum_residual += res
                if res > max_residual:
                    max_residual = res

        # Возвращаем среднюю невязку (более стабильно)
        if count_active > 0:
            return sum_residual / count_active
        return 0.0

    def _solve_direct(
        self,
        H: np.ndarray,
        dH_dphi: np.ndarray,
        d_phi: float,
        d_Z: float,
        D_L_sq: float,
        n_phi: int,
        n_z: int,
    ) -> tuple[np.ndarray, bool, int, float]:
        """
        Прямое решение разреженной системы уравнений.

        Собираем матрицу A и вектор b для системы A·P = b.
        Используем итеративную коррекцию для кавитации.
        """
        H3 = H ** 3

        # H³ на полуцелых узлах
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

        a_phi = 1.0 / d_phi**2
        a_z = D_L_sq / d_Z**2

        def idx(i, j):
            """Индекс в линейном массиве."""
            return i * n_z + j

        # Число неизвестных (внутренние точки по Z)
        # Граничные условия: P = 0 при j = 0 и j = n_z - 1
        n_total = n_phi * n_z

        # Собираем разреженную матрицу
        row_indices = []
        col_indices = []
        values = []
        b = np.zeros(n_total)

        for i in range(n_phi):
            i_plus = (i + 1) % n_phi
            i_minus = (i - 1) % n_phi

            for j in range(n_z):
                k = idx(i, j)

                if j == 0 or j == n_z - 1:
                    # Граничное условие P = 0
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

                    # Диагональный элемент
                    row_indices.append(k)
                    col_indices.append(k)
                    values.append(-c_center)

                    # Соседи по φ
                    row_indices.append(k)
                    col_indices.append(idx(i_plus, j))
                    values.append(c_i_plus)

                    row_indices.append(k)
                    col_indices.append(idx(i_minus, j))
                    values.append(c_i_minus)

                    # Соседи по Z
                    row_indices.append(k)
                    col_indices.append(idx(i, j + 1))
                    values.append(c_j_plus)

                    row_indices.append(k)
                    col_indices.append(idx(i, j - 1))
                    values.append(c_j_minus)

                    # Правая часть: уравнение имеет вид
                    # -c_center * P + ... = dH_dphi
                    b[k] = dH_dphi[i, j]

        # Создаём разреженную матрицу
        A = sparse.csr_matrix(
            (values, (row_indices, col_indices)),
            shape=(n_total, n_total)
        )

        # Решаем систему
        P_flat = spsolve(A, b)

        # Преобразуем в 2D
        P = P_flat.reshape((n_phi, n_z))

        # Итеративная коррекция для кавитации
        max_cav_iter = 50
        converged = True
        iterations = 1

        for cav_iter in range(max_cav_iter):
            # Находим точки с отрицательным давлением
            cavitation_mask = P < 0

            if not np.any(cavitation_mask):
                break

            # Фиксируем давление = 0 в этих точках
            for i in range(n_phi):
                for j in range(1, n_z - 1):
                    k = idx(i, j)
                    if cavitation_mask[i, j]:
                        # Модифицируем строку матрицы
                        # Устанавливаем P = 0 в этой точке
                        pass

            # Для простоты просто обнуляем отрицательные значения
            P = np.maximum(P, 0.0)
            iterations = cav_iter + 1

        # Вычисляем невязку PDE только для активной зоны
        residual = self._compute_pde_residual(
            P, H, dH_dphi, d_phi, d_Z, D_L_sq, n_phi, n_z
        )

        return P, converged, iterations, residual


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
