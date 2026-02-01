"""
Валидационные кейсы для верификации модели.

Источники:
- Patel et al.: Hydrodynamic journal bearing lubricated with a ferrofluid
  (валидация по данным для ISOVG46 без магнитного поля)
- Gwynllyw et al.: On the effects of a piezoviscous lubricant on the dynamics
  (валидация по данным для constant viscosity case)
"""

from dataclasses import dataclass
from typing import Tuple, Optional, Callable
import numpy as np

from .config import BearingConfig
from .reynolds_solver import solve_reynolds, ReynoldsResult
from .equilibrium import find_equilibrium
from .film_models import FilmModel


# ============================================================================
# КЕЙС A: PATEL ET AL. (ЭКСПЕРИМЕНТАЛЬНЫЙ)
# ============================================================================

@dataclass
class PatelCase:
    """
    Параметры экспериментального стенда Patel et al.

    ВАЖНО: Используются данные для обычного масла ISOVG46,
    а не для феррожидкости. Это baseline для классического
    уравнения Рейнольдса без магнитных членов.
    """
    D_bearing: float = 40.12e-3      # м
    D_journal: float = 39.97e-3      # м
    L: float = 40e-3                  # м
    Ra_bearing: float = 0.8e-6       # м
    Ra_journal: float = 0.3e-6       # м
    rho: float = 875.0               # кг/м³
    mu: float = 0.045                # Па·с (45 cP при 27°C)

    @property
    def R(self) -> float:
        return self.D_journal / 2

    @property
    def c(self) -> float:
        return (self.D_bearing - self.D_journal) / 2


# Экспериментальные данные p_max для ISOVG46 (Table I статьи Patel)
# Ключ: (n_rpm, W_newton)
PATEL_EXPERIMENTAL = {
    (250, 150): {"p_max_kPa": 207, "description": "250 rpm, 150 N, ISOVG46"},
    (500, 150): {"p_max_kPa": 234, "description": "500 rpm, 150 N, ISOVG46"},
    (650, 150): {"p_max_kPa": 234, "description": "650 rpm, 150 N, ISOVG46"},
    (250, 300): {"p_max_kPa": 552, "description": "250 rpm, 300 N, ISOVG46"},
    (500, 300): {"p_max_kPa": 627, "description": "500 rpm, 300 N, ISOVG46"},
    (650, 300): {"p_max_kPa": 662, "description": "650 rpm, 300 N, ISOVG46"},
    (250, 450): {"p_max_kPa": 1006, "description": "250 rpm, 450 N, ISOVG46"},
    (500, 450): {"p_max_kPa": 1020, "description": "500 rpm, 450 N, ISOVG46"},
    (650, 450): {"p_max_kPa": 820, "description": "650 rpm, 450 N, ISOVG46"},
}


# ============================================================================
# КЕЙС B: GWYNLLYW ET AL. (ЧИСЛЕННЫЙ БЕНЧМАРК)
# ============================================================================

@dataclass
class GwynllywCase:
    """
    Параметры численного бенчмарка Gwynllyw et al.

    ВАЖНО: Сравнение с случаем constant viscosity (линия b на графиках),
    а не piezoviscous. Для приближения "long bearing" используем L/D = 4.
    """
    R_journal: float = 31.25e-3      # м
    R_bearing: float = 31.29e-3      # м
    L_D_ratio: float = 4.0           # для приближения long bearing
    omega: float = 250.0             # рад/с
    mu: float = 5.7e-3               # Па·с
    rho: float = 820.0               # кг/м³

    @property
    def R(self) -> float:
        return self.R_journal

    @property
    def c(self) -> float:
        return self.R_bearing - self.R_journal

    @property
    def n_rpm(self) -> float:
        return self.omega * 60 / (2 * np.pi)

    @property
    def L(self) -> float:
        return self.L_D_ratio * 2 * self.R


# Референсные значения p_max (оцифрованы с графиков Figures 14b, 15b)
GWYNLLYW_REFERENCE = {
    0.93: {"p_max_MPa": 80, "description": "ε=0.93, digitized from Fig.14(b), constant viscosity"},
    0.95: {"p_max_MPa": 120, "description": "ε=0.95, digitized from Fig.15(b), constant viscosity"},
}


# ============================================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ============================================================================

def create_patel_config(n_rpm: float, epsilon: float = 0.5, phi0: float = None,
                        n_phi: int = 180, n_z: int = 50) -> BearingConfig:
    """Создать BearingConfig для кейса Patel."""
    case = PatelCase()
    if phi0 is None:
        phi0 = np.pi / 2  # нагрузка вниз по умолчанию
    return BearingConfig(
        R=case.R,
        L=case.L,
        c=case.c,
        epsilon=epsilon,
        phi0=phi0,
        n_rpm=n_rpm,
        mu=case.mu,
        n_phi=n_phi,
        n_z=n_z,
    )


def create_gwynllyw_config(epsilon: float,
                           n_phi: int = 180, n_z: int = 50) -> BearingConfig:
    """Создать BearingConfig для кейса Gwynllyw (long bearing)."""
    case = GwynllywCase()
    return BearingConfig(
        R=case.R,
        L=case.L,
        c=case.c,
        epsilon=epsilon,
        phi0=np.pi/2,
        n_rpm=case.n_rpm,
        mu=case.mu,
        n_phi=n_phi,
        n_z=n_z,
    )


def get_pressure_at_midplane(result: ReynoldsResult,
                              config: BearingConfig) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Извлечь профиль давления p(φ) на среднем сечении Z=0.

    Returns:
        phi_deg: углы в градусах, shape (n_phi,)
        p_Pa: давление в Па, shape (n_phi,)
        p_max_mid_Pa: максимум давления на midplane, Па
    """
    j_mid = len(result.Z) // 2
    phi_deg = np.degrees(result.phi)
    p_Pa = result.P[:, j_mid] * config.pressure_scale
    p_max_mid_Pa = np.max(p_Pa)
    return phi_deg, p_Pa, p_max_mid_Pa


def run_patel_validation(n_rpm: float, W: float,
                         n_phi: int = 180, n_z: int = 50,
                         epsilon_bounds: Tuple[float, float] = (0.01, 0.98),
                         film_model_factory: Optional[Callable[[BearingConfig], FilmModel]] = None) -> dict:
    """
    Запустить валидацию для одного режима Patel.

    Args:
        n_rpm: скорость вращения, об/мин
        W: нагрузка, Н
        epsilon_bounds: границы поиска эксцентриситета
        film_model_factory: фабрика модели плёнки (None = гладкая)

    Returns:
        dict с результатами, включая p_max_global и p_max_mid
    """
    # Создаём начальную конфигурацию
    config_init = create_patel_config(n_rpm, epsilon=0.5, n_phi=n_phi, n_z=n_z)

    # Находим равновесие при заданной нагрузке
    eq = find_equilibrium(
        config_init,
        W_ext=W,
        load_angle=-np.pi/2,
        epsilon_bounds=epsilon_bounds,
        verbose=False,
        film_model_factory=film_model_factory
    )

    # Создаём конфигурацию с найденным эксцентриситетом
    config = create_patel_config(n_rpm, epsilon=eq.epsilon, phi0=eq.phi0,
                                  n_phi=n_phi, n_z=n_z)

    # Решаем уравнение Рейнольдса
    if film_model_factory is not None:
        film_model = film_model_factory(config)
        result = solve_reynolds(config, film_model=film_model)
    else:
        result = solve_reynolds(config)

    # Извлекаем p_max на midplane (для сравнения с экспериментом!)
    phi_deg, p_Pa, p_max_mid_Pa = get_pressure_at_midplane(result, config)

    # Глобальный p_max (для справки)
    p_max_global_Pa = result.p_max

    return {
        "n_rpm": n_rpm,
        "W": W,
        "epsilon": eq.epsilon,
        "phi0_deg": np.degrees(eq.phi0),
        # Два значения p_max!
        "p_max_global_Pa": p_max_global_Pa,
        "p_max_global_kPa": p_max_global_Pa / 1000,
        "p_max_mid_Pa": p_max_mid_Pa,
        "p_max_mid_kPa": p_max_mid_Pa / 1000,
        # Профиль на midplane
        "phi_deg": phi_deg,
        "p_Pa": p_Pa,
        "h_min_um": result.h_min * 1e6,
    }


def run_patel_validation_fixed_position(n_rpm: float, epsilon: float, phi0: float,
                                         n_phi: int = 180, n_z: int = 50,
                                         film_model_factory: Optional[Callable[[BearingConfig], FilmModel]] = None) -> dict:
    """
    Запустить расчёт для фиксированного положения (для grid convergence).

    БЕЗ поиска равновесия — epsilon и phi0 заданы.
    """
    config = create_patel_config(n_rpm, epsilon=epsilon, phi0=phi0,
                                  n_phi=n_phi, n_z=n_z)

    if film_model_factory is not None:
        film_model = film_model_factory(config)
        result = solve_reynolds(config, film_model=film_model)
    else:
        result = solve_reynolds(config)

    phi_deg, p_Pa, p_max_mid_Pa = get_pressure_at_midplane(result, config)

    return {
        "n_rpm": n_rpm,
        "epsilon": epsilon,
        "phi0_deg": np.degrees(phi0),
        "p_max_global_kPa": result.p_max / 1000,
        "p_max_mid_kPa": p_max_mid_Pa / 1000,
        "h_min_um": result.h_min * 1e6,
        "converged": result.converged,
        "iterations": result.iterations,
    }


def run_gwynllyw_validation(epsilon: float,
                            n_phi: int = 180, n_z: int = 50,
                            film_model_factory: Optional[Callable[[BearingConfig], FilmModel]] = None) -> dict:
    """
    Запустить валидацию для одного ε Gwynllyw.

    Эксцентриситет задаётся напрямую (без поиска равновесия).
    """
    config = create_gwynllyw_config(epsilon, n_phi=n_phi, n_z=n_z)

    if film_model_factory is not None:
        film_model = film_model_factory(config)
        result = solve_reynolds(config, film_model=film_model)
    else:
        result = solve_reynolds(config)

    phi_deg, p_Pa, p_max_mid_Pa = get_pressure_at_midplane(result, config)

    return {
        "epsilon": epsilon,
        "p_max_global_Pa": result.p_max,
        "p_max_global_MPa": result.p_max / 1e6,
        "p_max_mid_MPa": p_max_mid_Pa / 1e6,
        "phi_deg": phi_deg,
        "p_Pa": p_Pa,
        "h_min_um": result.h_min * 1e6,
    }
