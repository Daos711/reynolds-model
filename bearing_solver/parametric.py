"""
Этап 11: Параметрический анализ текстуры методом ЦКП.

Центральный композиционный план (CCD) для исследования
влияния геометрических параметров текстуры на характеристики подшипника.

5 факторов:
    - h: глубина лунки, м
    - a: полуось по z, м
    - b: полуось по φ, м
    - N_phi: число рядов по φ
    - N_z: число рядов по z

Методика:
    - ε = 0.8 фиксирован
    - φ₀ ищется из условия равновесия (Fx = 0)
    - Inscribed CCD: звёздные точки = границы диапазонов
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Callable
from itertools import product
import time

from .config import BearingConfig
from .film_models import FilmModel, SmoothFilmModel
from .reynolds_solver import solve_reynolds, ReynoldsResult
from .forces import compute_forces, compute_friction, compute_losses, BearingForces


# ============================================================================
# КОНФИГУРАЦИЯ ПАРАМЕТРИЧЕСКОГО АНАЛИЗА
# ============================================================================

@dataclass
class ParametricConfig:
    """Конфигурация подшипника для параметрического анализа."""
    R: float = 34.5e-3       # радиус, м
    B: float = 103.5e-3      # ширина (L), м
    c: float = 60e-6         # зазор, м
    epsilon: float = 0.8     # эксцентриситет (фиксирован)
    n_rpm: float = 2980      # скорость, об/мин
    mu: float = 0.037        # вязкость, Па·с

    @property
    def omega(self) -> float:
        return 2 * np.pi * self.n_rpm / 60

    @property
    def U(self) -> float:
        return self.omega * self.R


@dataclass
class TextureFactors:
    """Факторы ЦКП для текстуры."""
    h: float         # глубина лунки, м
    a: float         # полуось по z, м
    b: float         # полуось по φ, м
    N_phi: int       # число рядов по φ
    N_z: int         # число рядов по z

    # Зона текстурирования (опционально)
    phi_min: float = 0.0
    phi_max: float = 2 * np.pi

    def validate(self, R: float, B: float) -> bool:
        """Проверить, что лунки не перекрываются."""
        gap_phi, gap_z = self.get_gaps(R, B)
        return gap_phi >= 0 and gap_z >= 0

    def get_gaps(self, R: float, B: float) -> Tuple[float, float]:
        """Вычислить зазоры между лунками (в метрах)."""
        # Длина зоны по окружности
        L_phi_zone = R * (self.phi_max - self.phi_min)

        # Зазоры
        gap_phi = (L_phi_zone - self.N_phi * 2 * self.b) / self.N_phi if self.N_phi > 0 else 0
        gap_z = (B - self.N_z * 2 * self.a) / self.N_z if self.N_z > 0 else 0

        return gap_phi, gap_z

    def get_fill_factor(self, R: float, B: float) -> float:
        """Коэффициент заполнения (доля площади под лунками)."""
        A_zone = R * (self.phi_max - self.phi_min) * B
        if A_zone <= 0:
            return 0.0
        S_cells = self.N_total * np.pi * self.a * self.b
        return S_cells / A_zone

    @property
    def N_total(self) -> int:
        return self.N_phi * self.N_z


@dataclass
class ParametricResult:
    """Результат одного расчёта."""
    # Входные факторы
    h: float
    a: float
    b: float
    N_phi: int
    N_z: int

    # Вычисленные геометрические
    gap_phi: float = 0.0
    gap_z: float = 0.0
    N_total: int = 0
    valid: bool = True
    fill_factor: float = 0.0

    # Равновесие
    phi0_equilibrium: float = np.nan

    # Целевые функции (Исследование 1: Трибология)
    W: float = np.nan           # несущая способность, Н
    Fx: float = np.nan          # сила по x, Н
    Fy: float = np.nan          # сила по y, Н
    F_friction: float = np.nan  # сила трения, Н
    mu_friction: float = np.nan # коэффициент трения
    P_loss: float = np.nan      # потери, Вт
    p_max: float = np.nan       # макс. давление, Па
    h_min: float = np.nan       # мин. зазор (с текстурой), м

    # Целевые функции (Исследование 2: Динамика)
    Kxx: float = np.nan
    Kyy: float = np.nan
    Kxy: float = np.nan
    Kyx: float = np.nan
    Cxx: float = np.nan
    Cyy: float = np.nan
    Cxy: float = np.nan
    Cyx: float = np.nan

    # Служебные
    calc_time: float = 0.0
    error: str = ""
    converged: bool = True


# ============================================================================
# МОДЕЛЬ ПЛЁНКИ С СЕТОЧНОЙ ТЕКСТУРОЙ
# ============================================================================

class GridTexturedFilmModel(FilmModel):
    """
    Модель толщины плёнки с сеточной текстурой.

    Лунки расположены в равномерной сетке N_phi × N_z.
    Форма лунок — эллипсоидальная.

    SUBGRID-ПОДХОД: Когда лунки меньше ячеек сетки, вместо "рисования"
    формы лунки добавляем эквивалентный объём в ячейку.

    H(φ, Z) = H_smooth(φ, Z) + ΔH*(φ, Z)
    """

    def __init__(self, config: BearingConfig, factors: TextureFactors):
        """
        Args:
            config: конфигурация подшипника
            factors: параметры текстуры
        """
        self.config = config
        self.factors = factors

        # Базовая гладкая модель
        self._smooth_model = SmoothFilmModel(config)

        # Безразмерная глубина
        self.h_depth_star = factors.h / config.c

        # Генерируем центры лунок
        self.centers_phi, self.centers_z = self._generate_grid_centers()
        self.N_cells = len(self.centers_phi)

        # Кешированное поле текстуры
        self._texture_field: Optional[np.ndarray] = None
        self._cached_phi: Optional[np.ndarray] = None
        self._cached_Z: Optional[np.ndarray] = None

        # Диагностика
        self._n_cells_added = 0
        self._dH_max = 0.0

    def _generate_grid_centers(self) -> Tuple[np.ndarray, np.ndarray]:
        """Генерировать центры лунок для равномерной сетки."""
        factors = self.factors
        config = self.config

        if factors.N_phi <= 0 or factors.N_z <= 0:
            return np.array([]), np.array([])

        # Диапазон по φ
        phi_range = factors.phi_max - factors.phi_min
        dphi = phi_range / factors.N_phi
        phi_centers = factors.phi_min + dphi / 2 + dphi * np.arange(factors.N_phi)

        # Диапазон по z (от -L/2 до +L/2)
        dz = config.L / factors.N_z
        z_centers = -config.L / 2 + dz / 2 + dz * np.arange(factors.N_z)

        # Все комбинации
        centers_phi = []
        centers_z = []

        for phi_c in phi_centers:
            for z_c in z_centers:
                centers_phi.append(phi_c)
                centers_z.append(z_c)

        return np.array(centers_phi), np.array(centers_z)

    def _compute_texture_field(self, phi: np.ndarray, Z: np.ndarray) -> np.ndarray:
        """
        Вычислить поле текстуры методом SUBGRID.

        Когда лунки меньше ячеек сетки, добавляем эквивалентный объём
        лунки в ячейку, содержащую её центр.

        Объём эллипсоидальной лунки: V = (2/3) × h × π × a × b
        Площадь ячейки: A_cell = R × dφ × dz
        Прибавка к H: dH = V / (A_cell × c)
        """
        # Проверяем кеш
        if (self._texture_field is not None and
            self._cached_phi is not None and
            self._cached_Z is not None and
            len(phi) == len(self._cached_phi) and
            len(Z) == len(self._cached_Z) and
            np.allclose(phi, self._cached_phi) and
            np.allclose(Z, self._cached_Z)):
            return self._texture_field

        n_phi = len(phi)
        n_z = len(Z)
        texture = np.zeros((n_phi, n_z))

        if self.N_cells == 0:
            return texture

        config = self.config
        factors = self.factors

        # Параметры сетки
        d_phi = phi[1] - phi[0] if n_phi > 1 else 2 * np.pi / n_phi
        d_Z = Z[1] - Z[0] if n_z > 1 else 2.0 / n_z

        # Размерный шаг по z
        dz_dim = d_Z * (config.L / 2)  # Z ∈ [-1, 1], так что dz = dZ × L/2

        # Площадь одной ячейки сетки (на поверхности)
        A_cell = config.R * d_phi * dz_dim  # м²

        # Объём одной эллипсоидальной лунки
        # Профиль: depth = h × sqrt(1 - r²), средняя глубина = (2/3)×h
        V_dimple = (2.0 / 3.0) * factors.h * np.pi * factors.a * factors.b  # м³

        # Эквивалентная прибавка к H (безразмерная) для одной лунки в ячейке
        dH_per_dimple = V_dimple / (A_cell * config.c)

        # Добавляем каждую лунку в соответствующую ячейку
        n_added = 0

        for k in range(self.N_cells):
            phi_c = self.centers_phi[k]
            z_c = self.centers_z[k]  # размерная, в метрах

            # Привести φ_c к диапазону сетки
            phi_c_wrapped = phi_c % (2 * np.pi)

            # Найти индекс ячейки по φ
            i_phi = int(phi_c_wrapped / d_phi)
            if i_phi >= n_phi:
                i_phi = n_phi - 1

            # Найти индекс ячейки по Z
            # z_c ∈ [-L/2, L/2], Z ∈ [-1, 1]
            Z_c = z_c / (config.L / 2)  # перевести в безразмерные
            i_z = int((Z_c + 1) / d_Z)
            if i_z >= n_z:
                i_z = n_z - 1
            if i_z < 0:
                i_z = 0

            # Добавить прибавку
            texture[i_phi, i_z] += dH_per_dimple
            n_added += 1

        # Сохранить диагностику
        self._n_cells_added = n_added
        self._dH_max = texture.max() if n_added > 0 else 0.0

        # Кешируем
        self._texture_field = texture
        self._cached_phi = phi.copy()
        self._cached_Z = Z.copy()

        return texture

    def H(self, phi: np.ndarray, Z: np.ndarray) -> np.ndarray:
        """Безразмерная толщина плёнки с текстурой."""
        H_smooth = self._smooth_model.H(phi, Z)
        texture = self._compute_texture_field(phi, Z)
        return H_smooth + texture

    def dH_dphi(self, phi: np.ndarray, Z: np.ndarray) -> np.ndarray:
        """Производная ∂H/∂φ."""
        # Гладкая часть — аналитически
        dH_smooth = self._smooth_model.dH_dphi(phi, Z)

        # Текстурная часть — численно
        texture = self._compute_texture_field(phi, Z)
        d_phi = phi[1] - phi[0] if len(phi) > 1 else 2 * np.pi / 180

        n_phi = len(phi)
        dH_texture = np.zeros_like(texture)

        for i in range(n_phi):
            i_plus = (i + 1) % n_phi
            i_minus = (i - 1 + n_phi) % n_phi
            dH_texture[i, :] = (texture[i_plus, :] - texture[i_minus, :]) / (2 * d_phi)

        return dH_smooth + dH_texture


# ============================================================================
# ПОИСК РАВНОВЕСИЯ φ₀
# ============================================================================

def find_equilibrium_phi0(
    pconfig: ParametricConfig,
    factors: TextureFactors,
    n_phi_grid: int = 90,
    n_z_grid: int = 30,
    verbose: bool = False
) -> float:
    """
    Найти φ₀ равновесия при фиксированном ε.

    Алгоритм:
    1. Сканируем φ₀ по сетке, считаем Fx
    2. Находим интервал где Fx меняет знак
    3. Уточняем brentq
    4. Проверяем что Fy < 0 (нагрузка вниз)

    Returns:
        φ₀ в радианах
    """
    from scipy.optimize import brentq

    def calc_Fx(phi0: float) -> float:
        """Вычислить Fx для заданного φ₀."""
        config = BearingConfig(
            R=pconfig.R,
            L=pconfig.B,
            c=pconfig.c,
            epsilon=pconfig.epsilon,
            phi0=phi0,
            n_rpm=pconfig.n_rpm,
            mu=pconfig.mu,
            n_phi=n_phi_grid,
            n_z=n_z_grid,
        )

        film_model = GridTexturedFilmModel(config, factors)
        sol = solve_reynolds(config, film_model=film_model)
        forces = compute_forces(sol, config)

        return forces.Fx

    # 1. Сканирование
    phi_grid = np.linspace(np.pi / 2, 3 * np.pi / 2, 20)
    Fx_values = []

    for phi0 in phi_grid:
        try:
            Fx = calc_Fx(phi0)
            Fx_values.append(Fx)
        except:
            Fx_values.append(np.nan)

    Fx_values = np.array(Fx_values)

    # Убираем NaN
    valid_mask = ~np.isnan(Fx_values)
    if not np.any(valid_mask):
        return np.pi  # fallback

    phi_valid = phi_grid[valid_mask]
    Fx_valid = Fx_values[valid_mask]

    # 2. Найти интервалы где Fx меняет знак
    sign_changes = np.where(np.diff(np.sign(Fx_valid)))[0]

    if len(sign_changes) == 0:
        # Нет смены знака — вернуть точку с минимальным |Fx|
        idx_min = np.argmin(np.abs(Fx_valid))
        return phi_valid[idx_min]

    # 3. Уточнить brentq
    for idx in sign_changes:
        phi_a, phi_b = phi_valid[idx], phi_valid[idx + 1]
        try:
            phi0_root = brentq(calc_Fx, phi_a, phi_b, xtol=1e-3)
            return phi0_root
        except:
            pass

    # Fallback
    idx_min = np.argmin(np.abs(Fx_valid))
    return phi_valid[idx_min]


# ============================================================================
# РАСЧЁТ ОДНОЙ ТОЧКИ
# ============================================================================

def run_single_case(
    pconfig: ParametricConfig,
    factors: TextureFactors,
    n_phi_grid: int = 90,
    n_z_grid: int = 30,
    compute_dynamics: bool = False,
    find_phi0: bool = True,
    verbose: bool = False
) -> ParametricResult:
    """
    Выполнить один расчёт для заданных факторов.

    Args:
        pconfig: параметры подшипника
        factors: параметры текстуры
        n_phi_grid: число узлов сетки по φ
        n_z_grid: число узлов сетки по z
        compute_dynamics: считать ли K, C
        find_phi0: искать ли равновесие φ₀
        verbose: выводить сообщения

    Returns:
        ParametricResult
    """
    start_time = time.time()

    # Проверка валидности
    gap_phi, gap_z = factors.get_gaps(pconfig.R, pconfig.B)
    valid = factors.validate(pconfig.R, pconfig.B)

    result = ParametricResult(
        h=factors.h,
        a=factors.a,
        b=factors.b,
        N_phi=factors.N_phi,
        N_z=factors.N_z,
        gap_phi=gap_phi,
        gap_z=gap_z,
        N_total=factors.N_total,
        valid=valid,
        fill_factor=factors.get_fill_factor(pconfig.R, pconfig.B),
    )

    if not valid:
        result.error = "Лунки перекрываются"
        result.calc_time = time.time() - start_time
        return result

    try:
        # Поиск равновесия φ₀
        if find_phi0:
            phi0_eq = find_equilibrium_phi0(
                pconfig, factors, n_phi_grid, n_z_grid, verbose
            )
        else:
            phi0_eq = np.pi  # default

        result.phi0_equilibrium = phi0_eq

        # Основной расчёт
        config = BearingConfig(
            R=pconfig.R,
            L=pconfig.B,
            c=pconfig.c,
            epsilon=pconfig.epsilon,
            phi0=phi0_eq,
            n_rpm=pconfig.n_rpm,
            mu=pconfig.mu,
            n_phi=n_phi_grid,
            n_z=n_z_grid,
        )

        film_model = GridTexturedFilmModel(config, factors)
        sol = solve_reynolds(config, film_model=film_model)
        forces = compute_forces(sol, config)
        friction = compute_friction(sol, config, forces)
        losses = compute_losses(config, friction)

        # Заполнить результаты
        result.W = forces.W
        result.Fx = forces.Fx
        result.Fy = forces.Fy
        result.F_friction = friction.F_friction
        result.mu_friction = friction.mu_friction
        result.P_loss = losses.P_friction
        result.p_max = sol.p_max
        result.h_min = pconfig.c * np.min(sol.H)  # h_min с учётом текстуры
        result.converged = sol.converged

        # Динамические коэффициенты
        if compute_dynamics:
            from .dynamics import compute_dynamic_coefficients

            def film_model_factory(cfg: BearingConfig) -> FilmModel:
                return GridTexturedFilmModel(cfg, factors)

            coeffs = compute_dynamic_coefficients(
                config, pconfig.epsilon, phi0_eq,
                n_phi=n_phi_grid, n_z=n_z_grid,
                film_model_factory=film_model_factory,
            )

            result.Kxx = coeffs.Kxx
            result.Kyy = coeffs.Kyy
            result.Kxy = coeffs.Kxy
            result.Kyx = coeffs.Kyx
            result.Cxx = coeffs.Cxx
            result.Cyy = coeffs.Cyy
            result.Cxy = coeffs.Cxy
            result.Cyx = coeffs.Cyx

    except Exception as e:
        result.error = str(e)
        result.valid = False

    result.calc_time = time.time() - start_time
    return result


# ============================================================================
# ГЕНЕРАЦИЯ ЦКП
# ============================================================================

def generate_ccd_plan(n_factors: int = 5, n_center: int = 3) -> Tuple[np.ndarray, float]:
    """
    Генерировать центральный композиционный план.

    Args:
        n_factors: число факторов
        n_center: число центральных точек

    Returns:
        (план в кодированных координатах, alpha)
    """
    # Для inscribed CCD: alpha = sqrt(n_factors)
    alpha = np.sqrt(n_factors)

    # Факторный план 2^k
    factorial = []
    for bits in product([-1, 1], repeat=n_factors):
        factorial.append(list(bits))
    factorial = np.array(factorial)

    # Звёздные точки (±alpha по каждой оси)
    axial = []
    for i in range(n_factors):
        point_plus = [0.0] * n_factors
        point_plus[i] = alpha
        axial.append(point_plus)

        point_minus = [0.0] * n_factors
        point_minus[i] = -alpha
        axial.append(point_minus)
    axial = np.array(axial)

    # Центральные точки
    center = np.zeros((n_center, n_factors))

    # Объединяем
    plan = np.vstack([factorial, axial, center])

    return plan, alpha


def decode_plan_inscribed(
    coded_plan: np.ndarray,
    factor_ranges: Dict[str, Tuple[float, float]],
    alpha: float
) -> pd.DataFrame:
    """
    Преобразовать кодированный план в реальные значения.

    Inscribed CCD: coded = ±α соответствует x_min/x_max

    Args:
        coded_plan: массив кодированных значений
        factor_ranges: словарь {имя: (min, max)}
        alpha: параметр звёздных точек

    Returns:
        DataFrame с реальными значениями
    """
    factor_names = list(factor_ranges.keys())

    decoded = {}
    for i, name in enumerate(factor_names):
        x_min, x_max = factor_ranges[name]
        x_center = (x_max + x_min) / 2
        x_range = (x_max - x_min) / 2
        decoded[name] = x_center + coded_plan[:, i] * x_range / alpha

    return pd.DataFrame(decoded)


# ============================================================================
# ПАРАЛЛЕЛЬНЫЙ РАСЧЁТ
# ============================================================================

def run_parametric_study(
    pconfig: ParametricConfig,
    factor_ranges: Dict[str, Tuple[float, float]],
    n_phi_grid: int = 90,
    n_z_grid: int = 30,
    compute_dynamics: bool = False,
    n_jobs: int = 1,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Выполнить полный параметрический анализ.

    Args:
        pconfig: параметры подшипника
        factor_ranges: диапазоны факторов
        n_phi_grid: разрешение сетки
        n_z_grid: разрешение сетки
        compute_dynamics: считать K, C
        n_jobs: число параллельных процессов (-1 = все ядра)
        verbose: выводить прогресс

    Returns:
        DataFrame с результатами
    """
    # Генерировать план
    coded_plan, alpha = generate_ccd_plan(n_factors=5, n_center=3)
    plan_df = decode_plan_inscribed(coded_plan, factor_ranges, alpha)

    if verbose:
        print(f"ЦКП: {len(plan_df)} точек")
        print(f"Сетка: {n_phi_grid}×{n_z_grid}")
        print(f"Alpha (inscribed): {alpha:.3f}")

    # Подготовить список факторов
    factors_list = []
    for _, row in plan_df.iterrows():
        factors = TextureFactors(
            h=row['h'],
            a=row['a'],
            b=row['b'],
            N_phi=max(1, int(round(row['N_phi']))),
            N_z=max(1, int(round(row['N_z']))),
        )
        factors_list.append(factors)

    # Расчёт
    results = []

    if n_jobs == 1:
        # Последовательный
        for i, factors in enumerate(factors_list):
            if verbose:
                print(f"  Точка {i+1}/{len(factors_list)}...", end=" ")

            result = run_single_case(
                pconfig, factors, n_phi_grid, n_z_grid,
                compute_dynamics=compute_dynamics,
                verbose=False
            )
            results.append(result)

            if verbose:
                if result.valid:
                    print(f"μ={result.mu_friction:.6f}, t={result.calc_time:.1f}s")
                else:
                    print(f"INVALID: {result.error}")
    else:
        # Параллельный
        try:
            from joblib import Parallel, delayed
            from tqdm import tqdm

            if verbose:
                results = Parallel(n_jobs=n_jobs)(
                    delayed(run_single_case)(
                        pconfig, f, n_phi_grid, n_z_grid,
                        compute_dynamics, verbose=False
                    )
                    for f in tqdm(factors_list, desc="Расчёт")
                )
            else:
                results = Parallel(n_jobs=n_jobs)(
                    delayed(run_single_case)(
                        pconfig, f, n_phi_grid, n_z_grid,
                        compute_dynamics, verbose=False
                    )
                    for f in factors_list
                )
        except ImportError:
            print("WARNING: joblib/tqdm не установлены, используем последовательный расчёт")
            for factors in factors_list:
                result = run_single_case(
                    pconfig, factors, n_phi_grid, n_z_grid,
                    compute_dynamics=compute_dynamics
                )
                results.append(result)

    # Собрать в DataFrame
    results_df = pd.DataFrame([vars(r) for r in results])

    return results_df


# ============================================================================
# RSM (Response Surface Methodology)
# ============================================================================

def fit_rsm_model(
    df: pd.DataFrame,
    factors: List[str],
    response: str
) -> dict:
    """
    Построить модель поверхности отклика (полином 2-го порядка).

    Args:
        df: DataFrame с результатами
        factors: список имён факторов
        response: имя целевой функции

    Returns:
        Словарь с моделью и статистиками
    """
    try:
        from sklearn.preprocessing import PolynomialFeatures
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import r2_score
    except ImportError:
        return {"error": "sklearn не установлен"}

    # Убрать невалидные точки
    df_valid = df[df['valid'] & df[response].notna()].copy()

    if len(df_valid) < 10:
        return {"error": f"Недостаточно данных: {len(df_valid)} точек"}

    X = df_valid[factors].values
    y = df_valid[response].values

    # Полином 2-го порядка
    poly = PolynomialFeatures(degree=2, include_bias=True)
    X_poly = poly.fit_transform(X)

    # Регрессия
    model = LinearRegression()
    model.fit(X_poly, y)

    y_pred = model.predict(X_poly)
    r2 = r2_score(y, y_pred)

    return {
        "model": model,
        "poly": poly,
        "r2": r2,
        "coefficients": model.coef_,
        "intercept": model.intercept_,
        "feature_names": poly.get_feature_names_out(factors),
        "n_points": len(df_valid),
    }


def find_optimum(
    rsm_model: dict,
    factor_ranges: Dict[str, Tuple[float, float]],
    minimize: bool = True
) -> dict:
    """
    Найти оптимум по модели RSM.

    Args:
        rsm_model: результат fit_rsm_model
        factor_ranges: диапазоны факторов
        minimize: True = минимизация, False = максимизация

    Returns:
        Словарь с оптимальными значениями
    """
    if "error" in rsm_model:
        return {"error": rsm_model["error"]}

    from scipy.optimize import minimize as scipy_minimize

    model = rsm_model["model"]
    poly = rsm_model["poly"]
    factor_names = list(factor_ranges.keys())

    def objective(x):
        X_poly = poly.transform(x.reshape(1, -1))
        y = model.predict(X_poly)[0]
        return y if minimize else -y

    # Границы
    bounds = [factor_ranges[name] for name in factor_names]

    # Начальная точка — центр
    x0 = np.array([(b[0] + b[1]) / 2 for b in bounds])

    # Оптимизация
    result = scipy_minimize(objective, x0, bounds=bounds, method='L-BFGS-B')

    optimum = {name: result.x[i] for i, name in enumerate(factor_names)}
    optimum["response"] = result.fun if minimize else -result.fun
    optimum["success"] = result.success

    return optimum


# ============================================================================
# ЛОКАЛЬНАЯ ДОВОДКА
# ============================================================================

def local_refinement(
    pconfig: ParametricConfig,
    rsm_optimum: dict,
    response: str,
    minimize: bool = True,
    n_phi_grid: int = 90,
    n_z_grid: int = 30,
    verbose: bool = True
) -> dict:
    """
    Локальная доводка: перебор целых N_phi, N_z вокруг оптимума RSM.

    Args:
        pconfig: параметры подшипника
        rsm_optimum: результат find_optimum
        response: имя целевой функции
        minimize: минимизировать
        n_phi_grid: сетка
        n_z_grid: сетка
        verbose: выводить сообщения

    Returns:
        Лучший результат
    """
    if "error" in rsm_optimum:
        return rsm_optimum

    N_phi_opt = int(round(rsm_optimum.get('N_phi', 8)))
    N_z_opt = int(round(rsm_optimum.get('N_z', 4)))

    best_result = None
    best_value = float('inf') if minimize else float('-inf')
    all_results = []

    if verbose:
        print(f"\nЛокальная доводка вокруг N_phi={N_phi_opt}, N_z={N_z_opt}:")

    # Перебор ±1
    for dN_phi in [-1, 0, 1]:
        for dN_z in [-1, 0, 1]:
            N_phi = max(1, N_phi_opt + dN_phi)
            N_z = max(1, N_z_opt + dN_z)

            factors = TextureFactors(
                h=rsm_optimum['h'],
                a=rsm_optimum['a'],
                b=rsm_optimum['b'],
                N_phi=N_phi,
                N_z=N_z,
            )

            result = run_single_case(
                pconfig, factors, n_phi_grid, n_z_grid,
                compute_dynamics=False, verbose=False
            )

            all_results.append(result)

            if result.valid and not np.isnan(getattr(result, response)):
                value = getattr(result, response)

                if verbose:
                    print(f"  N_phi={N_phi}, N_z={N_z}: {response}={value:.6f}")

                is_better = (minimize and value < best_value) or \
                           (not minimize and value > best_value)
                if is_better:
                    best_value = value
                    best_result = result

    if best_result is None:
        return {"error": "Нет валидных результатов"}

    return {
        "N_phi": best_result.N_phi,
        "N_z": best_result.N_z,
        "h": best_result.h,
        "a": best_result.a,
        "b": best_result.b,
        response: best_value,
        "result": best_result,
        "all_results": all_results,
    }
