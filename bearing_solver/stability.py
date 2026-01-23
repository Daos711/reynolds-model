"""
Этап 5: Анализ устойчивости ротора (Jeffcott rotor).

Модель:
    Масса m в центре, координаты (x, y)
    Подшипник задаёт силы через K (жёсткость) и C (демпфирование)

Уравнение движения:
    m·ẍ = -Kxx·x - Kxy·y - Cxx·ẋ - Cxy·ẏ
    m·ÿ = -Kyx·x - Kyy·y - Cyx·ẋ - Cyy·ẏ

Матрица состояния (z = [x, y, ẋ, ẏ]ᵀ):
    ż = A·z

        [  0       0       1       0   ]
    A = [  0       0       0       1   ]
        [-Kxx/m  -Kxy/m  -Cxx/m  -Cxy/m]
        [-Kyx/m  -Kyy/m  -Cyx/m  -Cyy/m]

Собственные значения:
    λᵢ = eig(A)
    Для комплексной пары λ = σ ± iωd:
    - σ — декремент затухания
    - ωd — демпфированная частота (частота вихревого движения)

Критерий устойчивости:
    - Устойчиво, если все Re(λᵢ) < 0
    - Запас устойчивости = -max(Re(λ))

Whirl ratio:
    γ = ωd / Ω, где Ω = 2πn/60 — угловая скорость вала
    Типичное значение для oil whirl: γ ≈ 0.4...0.5
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np
from numpy.linalg import eig

from .config import BearingConfig
from .dynamics import DynamicCoefficients


@dataclass
class StabilityResult:
    """Результат анализа устойчивости."""

    # Собственные значения (4 штуки, могут быть комплексными)
    eigenvalues: np.ndarray

    # Устойчивость
    is_stable: bool
    stability_margin: float      # -max(Re(λ)), положительный = устойчиво

    # Доминирующая мода (с наибольшим Re)
    dominant_eigenvalue: complex
    dominant_real: float         # Re(λ_dom)
    dominant_imag: float         # |Im(λ_dom)|
    dominant_freq_hz: float      # |Im(λ_dom)| / (2π), Гц

    # Whirl ratio (отношение частоты вихря к скорости вращения)
    whirl_ratio: float           # ωd / Ω
    shaft_speed_hz: float        # Ω / (2π), Гц

    # Параметры расчёта
    mass: float                  # масса ротора, кг
    n_rpm: float                 # скорость вращения, об/мин

    # Режим демпфирования
    damping_ratio: float         # C²/(4mK) для диагонали, >1 = overdamped
    is_overdamped: bool          # True если C² > 4mK

    def __str__(self) -> str:
        status = "УСТОЙЧИВО" if self.is_stable else "НЕУСТОЙЧИВО"
        return (
            f"Устойчивость: {status}\n"
            f"  Запас: {self.stability_margin:.2f} 1/с\n"
            f"  Доминирующее λ: {self.dominant_real:.2f} ± {self.dominant_imag:.2f}i\n"
            f"  Частота вихря: {self.dominant_freq_hz:.1f} Гц\n"
            f"  Whirl ratio: {self.whirl_ratio:.3f}"
        )


def build_state_matrix(
    K: np.ndarray,
    C: np.ndarray,
    mass: float
) -> np.ndarray:
    """
    Построить матрицу состояния A для модели Jeffcott rotor.

    Args:
        K: матрица жёсткости 2x2, Н/м
        C: матрица демпфирования 2x2, Н·с/м
        mass: масса ротора, кг

    Returns:
        A: матрица состояния 4x4
    """
    # z = [x, y, ẋ, ẏ]ᵀ
    # ż = A·z
    #
    # ẋ = ẋ
    # ẏ = ẏ
    # ẍ = (-Kxx·x - Kxy·y - Cxx·ẋ - Cxy·ẏ) / m
    # ÿ = (-Kyx·x - Kyy·y - Cyx·ẋ - Cyy·ẏ) / m

    A = np.zeros((4, 4))

    # Верхняя часть: ẋ = ẋ, ẏ = ẏ
    A[0, 2] = 1.0  # dx/dt = ẋ
    A[1, 3] = 1.0  # dy/dt = ẏ

    # Нижняя часть: уравнения движения
    A[2, 0] = -K[0, 0] / mass  # -Kxx/m
    A[2, 1] = -K[0, 1] / mass  # -Kxy/m
    A[2, 2] = -C[0, 0] / mass  # -Cxx/m
    A[2, 3] = -C[0, 1] / mass  # -Cxy/m

    A[3, 0] = -K[1, 0] / mass  # -Kyx/m
    A[3, 1] = -K[1, 1] / mass  # -Kyy/m
    A[3, 2] = -C[1, 0] / mass  # -Cyx/m
    A[3, 3] = -C[1, 1] / mass  # -Cyy/m

    return A


def analyze_stability(
    K: np.ndarray,
    C: np.ndarray,
    mass: float,
    n_rpm: float
) -> StabilityResult:
    """
    Провести анализ устойчивости для заданных K, C, массы и скорости.

    Args:
        K: матрица жёсткости 2x2, Н/м
        C: матрица демпфирования 2x2, Н·с/м
        mass: масса ротора, кг
        n_rpm: скорость вращения, об/мин

    Returns:
        StabilityResult
    """
    # Строим матрицу состояния
    A = build_state_matrix(K, C, mass)

    # Вычисляем собственные значения
    eigenvalues, _ = eig(A)

    # Сортируем по убыванию действительной части
    idx = np.argsort(-eigenvalues.real)
    eigenvalues = eigenvalues[idx]

    # Доминирующее собственное значение (с наибольшим Re)
    dominant = eigenvalues[0]
    dominant_real = dominant.real
    dominant_imag = abs(dominant.imag)

    # Устойчивость
    max_real = np.max(eigenvalues.real)
    is_stable = max_real < 0
    stability_margin = -max_real

    # Частота вихря
    dominant_freq_hz = dominant_imag / (2 * np.pi)

    # Скорость вращения вала
    omega_shaft = 2 * np.pi * n_rpm / 60  # рад/с
    shaft_speed_hz = n_rpm / 60

    # Whirl ratio
    if omega_shaft > 0:
        whirl_ratio = dominant_imag / omega_shaft
    else:
        whirl_ratio = 0.0

    # Оценка режима демпфирования (по диагонали для простоты)
    # Для m·ẍ + c·ẋ + k·x = 0: overdamped если c² > 4mk
    # damping_ratio = c²/(4mk), >1 означает overdamped
    K_diag = 0.5 * (abs(K[0, 0]) + abs(K[1, 1]))
    C_diag = 0.5 * (abs(C[0, 0]) + abs(C[1, 1]))
    if mass > 0 and K_diag > 0:
        damping_ratio = C_diag**2 / (4 * mass * K_diag)
    else:
        damping_ratio = 0.0
    is_overdamped = damping_ratio > 1.0

    return StabilityResult(
        eigenvalues=eigenvalues,
        is_stable=is_stable,
        stability_margin=stability_margin,
        dominant_eigenvalue=dominant,
        dominant_real=dominant_real,
        dominant_imag=dominant_imag,
        dominant_freq_hz=dominant_freq_hz,
        whirl_ratio=whirl_ratio,
        shaft_speed_hz=shaft_speed_hz,
        mass=mass,
        n_rpm=n_rpm,
        damping_ratio=damping_ratio,
        is_overdamped=is_overdamped,
    )


def analyze_stability_from_coefficients(
    coeffs: DynamicCoefficients,
    mass: float,
    n_rpm: float
) -> StabilityResult:
    """
    Анализ устойчивости из объекта DynamicCoefficients.

    Args:
        coeffs: динамические коэффициенты
        mass: масса ротора, кг
        n_rpm: скорость вращения, об/мин

    Returns:
        StabilityResult
    """
    return analyze_stability(coeffs.K, coeffs.C, mass, n_rpm)


def find_stability_threshold(
    base_config: BearingConfig,
    mass: float,
    W_ext: float,
    n_rpm_range: Tuple[float, float] = (500, 10000),
    n_points: int = 50,
    verbose: bool = False
) -> Tuple[Optional[float], List[dict]]:
    """
    Найти порог устойчивости по скорости вращения.

    Возвращает скорость, при которой система становится неустойчивой.

    Args:
        base_config: базовая конфигурация подшипника
        mass: масса ротора, кг
        W_ext: внешняя нагрузка, Н
        n_rpm_range: диапазон скоростей (min, max), об/мин
        n_points: число точек для сканирования
        verbose: печатать ход расчёта

    Returns:
        (threshold_rpm, results_list)
        threshold_rpm: скорость потери устойчивости (None если всегда устойчиво/неустойчиво)
        results_list: список словарей с результатами для каждой скорости
    """
    from .equilibrium import find_equilibrium
    from .dynamics import compute_dynamic_coefficients

    n_values = np.linspace(n_rpm_range[0], n_rpm_range[1], n_points)
    results = []
    threshold_rpm = None
    prev_stable = None

    for n_rpm in n_values:
        # Создаём конфигурацию с новой скоростью
        config = BearingConfig(
            R=base_config.R,
            L=base_config.L,
            c=base_config.c,
            epsilon=base_config.epsilon,
            phi0=base_config.phi0,
            n_rpm=n_rpm,
            mu=base_config.mu,
            n_phi=base_config.n_phi,
            n_z=base_config.n_z,
        )

        try:
            # Находим равновесие
            eq = find_equilibrium(config, W_ext=W_ext, load_angle=-np.pi/2, verbose=False)

            # Вычисляем K, C
            coeffs = compute_dynamic_coefficients(
                config, eq.epsilon, eq.phi0,
                delta_e=0.01, delta_v_star=0.01,
                n_phi=180, n_z=50, verbose=False
            )

            # Анализ устойчивости
            stab = analyze_stability(coeffs.K, coeffs.C, mass, n_rpm)

            result = {
                'n_rpm': n_rpm,
                'epsilon': eq.epsilon,
                'phi0': eq.phi0,
                'Kxx': coeffs.Kxx, 'Kxy': coeffs.Kxy,
                'Kyx': coeffs.Kyx, 'Kyy': coeffs.Kyy,
                'Cxx': coeffs.Cxx, 'Cxy': coeffs.Cxy,
                'Cyx': coeffs.Cyx, 'Cyy': coeffs.Cyy,
                'max_real': stab.dominant_real,
                'is_stable': stab.is_stable,
                'stability_margin': stab.stability_margin,
                'whirl_ratio': stab.whirl_ratio,
                'freq_hz': stab.dominant_freq_hz,
            }
            results.append(result)

            if verbose:
                status = "OK" if stab.is_stable else "UNSTABLE"
                print(f"n={n_rpm:.0f}: ε={eq.epsilon:.3f}, "
                      f"max_Re={stab.dominant_real:.1f}, γ={stab.whirl_ratio:.3f} [{status}]")

            # Проверяем переход через порог
            if prev_stable is not None and prev_stable and not stab.is_stable:
                threshold_rpm = n_rpm

            prev_stable = stab.is_stable

        except Exception as e:
            if verbose:
                print(f"n={n_rpm:.0f}: ОШИБКА - {e}")
            continue

    return threshold_rpm, results
