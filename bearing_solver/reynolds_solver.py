"""
Решатель уравнения Рейнольдса для гидродинамического подшипника.

Оптимизирован с помощью Numba JIT для максимальной производительности.
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

try:
    from numba import njit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    # Заглушки если Numba не установлена
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    prange = range

from .config import BearingConfig
from .film_models import FilmModel, SmoothFilmModel


@dataclass
class ReynoldsResult:
    """Результат решения уравнения Рейнольдса."""
    phi: np.ndarray
    Z: np.ndarray
    P: np.ndarray
    H: np.ndarray
    h_min: float
    h_min_dimless: float
    P_max: float
    p_max: float
    converged: bool
    iterations: int
    residual: float
    method: str

    def get_dimensional_pressure(self, config: BearingConfig) -> np.ndarray:
        return self.P * config.pressure_scale


# ============================================================================
# Numba-оптимизированные функции
# ============================================================================

@njit(cache=True, parallel=True)
def _sor_iteration_numba(
    P: np.ndarray,
    c_phi_plus: np.ndarray,
    c_phi_minus: np.ndarray,
    c_z_plus: np.ndarray,
    c_z_minus: np.ndarray,
    c_center: np.ndarray,
    dH_dphi: np.ndarray,
    omega: float,
    n_phi: int,
    n_z: int,
) -> tuple:
    """Одна итерация SOR с Numba (параллельно по i)."""
    max_change = 0.0

    for i in prange(n_phi):
        i_plus = (i + 1) % n_phi
        i_minus = (i - 1) % n_phi

        for j in range(1, n_z - 1):
            cc = c_center[i, j]
            if cc < 1e-15:
                continue

            P_sum = (c_phi_plus[i, j] * P[i_plus, j] +
                     c_phi_minus[i, j] * P[i_minus, j] +
                     c_z_plus[i, j] * P[i, j + 1] +
                     c_z_minus[i, j] * P[i, j - 1])

            P_new = (P_sum - dH_dphi[i, j]) / cc
            P_new = P[i, j] + omega * (P_new - P[i, j])

            if P_new < 0.0:
                P_new = 0.0

            change = abs(P_new - P[i, j])
            if change > max_change:
                max_change = change

            P[i, j] = P_new

    return P, max_change


@njit(cache=True, parallel=True)
def _jacobi_iteration_numba(
    P: np.ndarray,
    P_new: np.ndarray,
    c_phi_plus: np.ndarray,
    c_phi_minus: np.ndarray,
    c_z_plus: np.ndarray,
    c_z_minus: np.ndarray,
    c_center: np.ndarray,
    dH_dphi: np.ndarray,
    omega: float,
    n_phi: int,
    n_z: int,
) -> float:
    """Итерация Jacobi с релаксацией (полностью параллельно)."""
    max_change = 0.0

    for i in prange(n_phi):
        i_plus = (i + 1) % n_phi
        i_minus = (i - 1) % n_phi

        for j in range(1, n_z - 1):
            cc = c_center[i, j]
            if cc < 1e-15:
                P_new[i, j] = 0.0
                continue

            P_sum = (c_phi_plus[i, j] * P[i_plus, j] +
                     c_phi_minus[i, j] * P[i_minus, j] +
                     c_z_plus[i, j] * P[i, j + 1] +
                     c_z_minus[i, j] * P[i, j - 1])

            val = (P_sum - dH_dphi[i, j]) / cc
            val = P[i, j] + omega * (val - P[i, j])

            if val < 0.0:
                val = 0.0

            change = abs(val - P[i, j])
            if change > max_change:
                max_change = change

            P_new[i, j] = val

    return max_change


@njit(cache=True)
def _compute_residual_numba(
    P: np.ndarray,
    c_phi_plus: np.ndarray,
    c_phi_minus: np.ndarray,
    c_z_plus: np.ndarray,
    c_z_minus: np.ndarray,
    dH_dphi: np.ndarray,
    n_phi: int,
    n_z: int,
    rhs_scale: float,
) -> float:
    """Вычислить невязку в активной зоне."""
    max_res = 0.0

    for i in range(n_phi):
        i_plus = (i + 1) % n_phi
        i_minus = (i - 1) % n_phi

        for j in range(1, n_z - 1):
            if P[i, j] > 1e-10:
                lap = (c_phi_plus[i, j] * (P[i_plus, j] - P[i, j]) -
                       c_phi_minus[i, j] * (P[i, j] - P[i_minus, j]) +
                       c_z_plus[i, j] * (P[i, j + 1] - P[i, j]) -
                       c_z_minus[i, j] * (P[i, j] - P[i, j - 1]))

                res = abs(lap - dH_dphi[i, j]) / rhs_scale
                if res > max_res:
                    max_res = res

    return max_res


# ============================================================================
# Основной класс решателя
# ============================================================================

class ReynoldsSolver:
    """
    Решатель уравнения Рейнольдса.

    Методы:
    - "direct": прямое решение СЛАУ (scipy.sparse) — быстро для малых сеток
    - "sor": SOR с Numba JIT — быстро для больших сеток
    - "jacobi": Jacobi с Numba (полностью параллельно)
    """

    def __init__(
        self,
        config: BearingConfig,
        film_model: Optional[FilmModel] = None,
        method: str = "direct"
    ):
        self.config = config
        self.film_model = film_model or SmoothFilmModel(config)
        self.method = method

        # Параметры итерационных методов
        self.omega = 1.5          # релаксация (1.0-1.9)
        self.max_iter = 10000
        self.min_iter = 50
        self.tol = 1e-6

    def solve(self) -> ReynoldsResult:
        phi, Z, d_phi, d_Z = self.config.create_grid()
        n_phi, n_z = len(phi), len(Z)

        H = self.film_model.H(phi, Z)
        dH_dphi = self.film_model.dH_dphi(phi, Z)

        D_L_sq = self.config.D_L_ratio ** 2
        a_phi = 1.0 / d_phi**2
        a_z = D_L_sq / d_Z**2

        # H³ на полуцелых узлах
        H3 = H ** 3
        H3_phi_plus = 0.5 * (H3 + np.roll(H3, -1, axis=0))
        H3_phi_minus = 0.5 * (H3 + np.roll(H3, 1, axis=0))
        H3_z_plus = np.zeros_like(H3)
        H3_z_minus = np.zeros_like(H3)
        H3_z_plus[:, :-1] = 0.5 * (H3[:, :-1] + H3[:, 1:])
        H3_z_minus[:, 1:] = 0.5 * (H3[:, 1:] + H3[:, :-1])

        # Коэффициенты
        c_phi_plus = a_phi * H3_phi_plus
        c_phi_minus = a_phi * H3_phi_minus
        c_z_plus = a_z * H3_z_plus
        c_z_minus = a_z * H3_z_minus
        c_center = c_phi_plus + c_phi_minus + c_z_plus + c_z_minus
        c_center = np.where(c_center < 1e-15, 1.0, c_center)

        # Выбор метода
        if self.method == "sor":
            P, converged, iterations, residual = self._solve_sor(
                c_phi_plus, c_phi_minus, c_z_plus, c_z_minus,
                c_center, dH_dphi, n_phi, n_z
            )
        elif self.method == "jacobi":
            P, converged, iterations, residual = self._solve_jacobi(
                c_phi_plus, c_phi_minus, c_z_plus, c_z_minus,
                c_center, dH_dphi, n_phi, n_z
            )
        else:
            P, converged, iterations, residual = self._solve_direct(
                c_phi_plus, c_phi_minus, c_z_plus, c_z_minus,
                dH_dphi, n_phi, n_z
            )

        # Результаты
        h_min_dimless = np.min(H)
        h_min = h_min_dimless * self.config.c
        P_max = np.max(P)
        p_max = P_max * self.config.pressure_scale

        return ReynoldsResult(
            phi=phi, Z=Z, P=P, H=H,
            h_min=h_min, h_min_dimless=h_min_dimless,
            P_max=P_max, p_max=p_max,
            converged=converged, iterations=iterations,
            residual=residual, method=self.method,
        )

    def _solve_sor(self, c_phi_plus, c_phi_minus, c_z_plus, c_z_minus,
                   c_center, dH_dphi, n_phi, n_z):
        """SOR с Numba JIT."""
        P = np.zeros((n_phi, n_z))
        converged = False

        for iteration in range(self.max_iter):
            P, max_change = _sor_iteration_numba(
                P, c_phi_plus, c_phi_minus, c_z_plus, c_z_minus,
                c_center, dH_dphi, self.omega, n_phi, n_z
            )

            if iteration >= self.min_iter - 1 and max_change < self.tol:
                converged = True
                break

        rhs_scale = np.max(np.abs(dH_dphi)) + 1e-15
        residual = _compute_residual_numba(
            P, c_phi_plus, c_phi_minus, c_z_plus, c_z_minus,
            dH_dphi, n_phi, n_z, rhs_scale
        )

        return P, converged, iteration + 1, residual

    def _solve_jacobi(self, c_phi_plus, c_phi_minus, c_z_plus, c_z_minus,
                      c_center, dH_dphi, n_phi, n_z):
        """Jacobi с релаксацией (полностью параллельно)."""
        P = np.zeros((n_phi, n_z))
        P_new = np.zeros((n_phi, n_z))
        converged = False

        for iteration in range(self.max_iter):
            max_change = _jacobi_iteration_numba(
                P, P_new, c_phi_plus, c_phi_minus, c_z_plus, c_z_minus,
                c_center, dH_dphi, self.omega, n_phi, n_z
            )
            P, P_new = P_new, P  # swap

            if iteration >= self.min_iter - 1 and max_change < self.tol:
                converged = True
                break

        rhs_scale = np.max(np.abs(dH_dphi)) + 1e-15
        residual = _compute_residual_numba(
            P, c_phi_plus, c_phi_minus, c_z_plus, c_z_minus,
            dH_dphi, n_phi, n_z, rhs_scale
        )

        return P, converged, iteration + 1, residual

    def _solve_direct(self, c_phi_plus, c_phi_minus, c_z_plus, c_z_minus,
                      dH_dphi, n_phi, n_z):
        """Прямое решение СЛАУ + итеративная коррекция кавитации."""
        n_total = n_phi * n_z
        max_cav_iter = 50
        cavitation_mask = np.zeros((n_phi, n_z), dtype=bool)

        for cav_iter in range(max_cav_iter):
            row_idx = []
            col_idx = []
            data = []
            b = np.zeros(n_total)

            for i in range(n_phi):
                i_plus = (i + 1) % n_phi
                i_minus = (i - 1) % n_phi

                for j in range(n_z):
                    k = i * n_z + j

                    if j == 0 or j == n_z - 1 or cavitation_mask[i, j]:
                        row_idx.append(k)
                        col_idx.append(k)
                        data.append(1.0)
                        b[k] = 0.0
                    else:
                        cp = c_phi_plus[i, j]
                        cm = c_phi_minus[i, j]
                        czp = c_z_plus[i, j]
                        czm = c_z_minus[i, j]
                        cc = cp + cm + czp + czm

                        row_idx.append(k)
                        col_idx.append(k)
                        data.append(-cc)

                        row_idx.append(k)
                        col_idx.append(i * n_z + j + 1)
                        data.append(czp)

                        row_idx.append(k)
                        col_idx.append(i * n_z + j - 1)
                        data.append(czm)

                        row_idx.append(k)
                        col_idx.append(i_plus * n_z + j)
                        data.append(cp)

                        row_idx.append(k)
                        col_idx.append(i_minus * n_z + j)
                        data.append(cm)

                        b[k] = dH_dphi[i, j]

            A = sparse.csr_matrix((data, (row_idx, col_idx)), shape=(n_total, n_total))
            P_flat = spsolve(A, b)
            P = P_flat.reshape((n_phi, n_z))

            new_cav = P < -1e-10
            new_cav[:, 0] = False
            new_cav[:, -1] = False

            if not np.any(new_cav & ~cavitation_mask):
                break

            cavitation_mask = cavitation_mask | new_cav

        P = np.maximum(P, 0.0)

        c_center = c_phi_plus + c_phi_minus + c_z_plus + c_z_minus
        c_center = np.where(c_center < 1e-15, 1.0, c_center)
        rhs_scale = np.max(np.abs(dH_dphi)) + 1e-15

        residual = _compute_residual_numba(
            P, c_phi_plus, c_phi_minus, c_z_plus, c_z_minus,
            dH_dphi, n_phi, n_z, rhs_scale
        )

        return P, True, cav_iter + 1, residual


def solve_reynolds(
    config: BearingConfig,
    film_model: Optional[FilmModel] = None,
    method: str = "direct"
) -> ReynoldsResult:
    """
    Решить уравнение Рейнольдса.

    Args:
        config: конфигурация подшипника
        film_model: модель толщины плёнки
        method: "direct" (рекомендуется), "sor", или "jacobi"

    Returns:
        ReynoldsResult
    """
    solver = ReynoldsSolver(config, film_model, method)
    return solver.solve()
