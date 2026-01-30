"""
Расчёт орбит ротора на основе модели Джеффкотта.

ВАЖНО:
- K, C — линеаризация вокруг равновесия
- Орбиты валидны для малых отклонений (r ≪ c)
- Статическая нагрузка уже учтена в равновесии
"""

import numpy as np
from scipy.integrate import solve_ivp
from dataclasses import dataclass
from typing import Optional, Tuple, Callable
import matplotlib.pyplot as plt

from .dynamics import DynamicCoefficients


@dataclass
class RotorParams:
    """Параметры ротора."""
    mass: float                      # масса ротора, кг

    # Дисбаланс
    unbalance_mass: float = 0.0      # масса дисбаланса, кг
    unbalance_eccentricity: float = 0.0  # эксцентриситет дисбаланса, м
    unbalance_phase: float = 0.0     # начальная фаза, рад

    @property
    def unbalance_me(self) -> float:
        """Дисбаланс m*e, кг·м."""
        return self.unbalance_mass * self.unbalance_eccentricity

    @classmethod
    def from_gmm(cls, mass: float, me_gmm: float, phase: float = 0.0):
        """
        Создать RotorParams из дисбаланса в г·мм.

        Args:
            mass: масса ротора, кг
            me_gmm: дисбаланс m*e в г·мм
            phase: начальная фаза, рад
        """
        # m*e [г·мм] = m*e [кг·м] * 1e6
        # Фиксируем e = 1 мм, тогда m = me_gmm грамм
        return cls(
            mass=mass,
            unbalance_mass=me_gmm * 1e-3,      # грамм → кг
            unbalance_eccentricity=1e-3,        # 1 мм
            unbalance_phase=phase,
        )


@dataclass
class OrbitResult:
    """Результат расчёта орбиты."""
    t: np.ndarray            # время, с
    x: np.ndarray            # смещение по x, м
    y: np.ndarray            # смещение по y, м
    vx: np.ndarray           # скорость по x, м/с
    vy: np.ndarray           # скорость по y, м/с
    omega: float             # угловая скорость, рад/с
    mass: float              # масса ротора, кг
    clearance: float         # зазор подшипника, м

    # Характеристики (на установившемся участке)
    x_amplitude: float       # амплитуда по x, м
    y_amplitude: float       # амплитуда по y, м
    max_displacement: float  # максимальное смещение от центра, м
    steady_state_start: int  # индекс начала установившегося режима
    n_periods_total: float   # общее число оборотов

    @property
    def r_over_c(self) -> float:
        """Отношение max(r)/c — ключевой показатель."""
        return self.max_displacement / self.clearance

    @property
    def is_bounded(self) -> bool:
        """Орбита внутри зазора с запасом 20%."""
        return self.r_over_c < 0.8

    @property
    def is_safe(self) -> bool:
        """Орбита внутри 50% зазора."""
        return self.r_over_c < 0.5


@dataclass
class InitialConditions:
    """Начальные условия."""
    x0: float = 0.0
    y0: float = 0.0
    vx0: float = 0.0
    vy0: float = 0.0


def compute_t_end(omega: float, n_periods: int = 50, t_min: float = 0.2) -> float:
    """Время интегрирования: минимум n_periods оборотов."""
    T = 2 * np.pi / omega
    return max(n_periods * T, t_min)


def build_state_equations(K: np.ndarray, C: np.ndarray,
                          mass: float, omega: float,
                          rotor: RotorParams) -> Callable:
    """
    Построить правую часть для solve_ivp.

    Уравнение (в отклонениях от равновесия):
    m·r̈ + C·ṙ + K·r = F_unbalance(t)

    Статическая нагрузка НЕ включена — она в равновесии!
    """
    me = rotor.unbalance_me
    phase0 = rotor.unbalance_phase

    def equations(t, z):
        x, y, vx, vy = z

        # Только дисбаланс — вращающаяся сила
        Fx = me * omega**2 * np.cos(omega * t + phase0)
        Fy = me * omega**2 * np.sin(omega * t + phase0)

        # Реакция подшипника (линеаризованная)
        Fx -= K[0, 0]*x + K[0, 1]*y + C[0, 0]*vx + C[0, 1]*vy
        Fy -= K[1, 0]*x + K[1, 1]*y + C[1, 0]*vx + C[1, 1]*vy

        return [vx, vy, Fx/mass, Fy/mass]

    return equations


def compute_orbit(coeffs: DynamicCoefficients,
                  rotor: RotorParams,
                  omega: float,
                  clearance: float,
                  n_periods: int = 50,
                  n_points_per_period: int = 40,
                  initial: Optional[InitialConditions] = None,
                  method: str = 'RK45') -> OrbitResult:
    """
    Рассчитать орбиту ротора.

    Args:
        coeffs: динамические коэффициенты K, C (в СИ: Н/м и Н·с/м!)
        rotor: параметры ротора (масса, дисбаланс)
        omega: угловая скорость, рад/с
        clearance: зазор подшипника, м
        n_periods: минимальное число оборотов
        n_points_per_period: точек на оборот
        initial: начальные условия
        method: метод интегрирования
    """
    if initial is None:
        initial = InitialConditions()

    # Время интегрирования (с учётом t_min)
    T_period = 2 * np.pi / omega
    t_end = compute_t_end(omega, n_periods)

    # ВАЖНО: n_points от РЕАЛЬНОГО t_end, не от n_periods!
    n_periods_eff = t_end / T_period
    n_points = int(n_periods_eff * n_points_per_period)

    z0 = [initial.x0, initial.y0, initial.vx0, initial.vy0]
    equations = build_state_equations(coeffs.K, coeffs.C, rotor.mass, omega, rotor)

    t_eval = np.linspace(0, t_end, n_points)

    sol = solve_ivp(
        equations, [0, t_end], z0,
        method=method, t_eval=t_eval,
        rtol=1e-9, atol=1e-12
    )

    t = sol.t
    x, y, vx, vy = sol.y

    # Установившийся режим — последние 50%
    steady_start = len(t) // 2
    x_ss, y_ss = x[steady_start:], y[steady_start:]

    x_amplitude = (np.max(x_ss) - np.min(x_ss)) / 2
    y_amplitude = (np.max(y_ss) - np.min(y_ss)) / 2
    max_displacement = np.max(np.sqrt(x**2 + y**2))

    return OrbitResult(
        t=t, x=x, y=y, vx=vx, vy=vy,
        omega=omega, mass=rotor.mass, clearance=clearance,
        x_amplitude=x_amplitude, y_amplitude=y_amplitude,
        max_displacement=max_displacement,
        steady_state_start=steady_start,
        n_periods_total=t_end / T_period,
    )


def compute_orbit_from_config(config, rotor: RotorParams, W_ext: float,
                               n_periods: int = 50,
                               initial: Optional[InitialConditions] = None,
                               verbose: bool = False) -> Tuple[OrbitResult, DynamicCoefficients, dict]:
    """
    Полная цепочка: равновесие → K,C → орбита.

    Args:
        config: BearingConfig
        rotor: параметры ротора
        W_ext: внешняя нагрузка (уже включает вес!), Н
        n_periods: число оборотов

    Returns:
        (OrbitResult, DynamicCoefficients, equilibrium_info)
    """
    from .equilibrium import find_equilibrium
    from .dynamics import compute_dynamic_coefficients

    # 1. Равновесие при полной нагрузке W_ext
    if verbose:
        print(f"Поиск равновесия при W_ext = {W_ext} Н...")
    eq = find_equilibrium(config, W_ext=W_ext, load_angle=-np.pi/2, verbose=verbose)

    # 2. Линеаризованные K, C
    if verbose:
        print(f"Расчёт K, C при ε = {eq.epsilon:.4f}...")
    coeffs = compute_dynamic_coefficients(
        config, eq.epsilon, eq.phi0,
        delta_e=0.01, delta_v_star=0.01,
        n_phi=180, n_z=50, verbose=verbose
    )

    # 3. Орбита (только возмущения от дисбаланса!)
    if verbose:
        print(f"Интегрирование орбиты ({n_periods} оборотов)...")
    orbit = compute_orbit(
        coeffs, rotor, config.omega, config.c,
        n_periods=n_periods, initial=initial
    )

    eq_info = {
        "epsilon": eq.epsilon,
        "phi0_deg": np.degrees(eq.phi0),
        "h_min_um": (1 - eq.epsilon) * config.c * 1e6,  # минимальный зазор в равновесии
        "W_ext": W_ext,
    }

    return orbit, coeffs, eq_info


def verify_damping(coeffs: DynamicCoefficients,
                   rotor: RotorParams,
                   omega: float,
                   clearance: float,
                   x0_fraction: float = 0.1) -> dict:
    """
    Проверка знаков K, C: при нулевом дисбалансе и начальном
    смещении колебания должны затухать.

    Также проверяет масштаб: амплитуды должны быть в микронах,
    не в миллиметрах или нанометрах (иначе единицы K,C неправильные).

    Args:
        x0_fraction: начальное смещение как доля от зазора
    """
    rotor_no_unb = RotorParams(mass=rotor.mass)  # без дисбаланса
    initial = InitialConditions(x0=x0_fraction * clearance)

    orbit = compute_orbit(
        coeffs, rotor_no_unb, omega, clearance,
        n_periods=30, initial=initial
    )

    # Сравниваем амплитуды в начале и в конце
    n_quarter = len(orbit.t) // 4
    x_start = np.max(np.abs(orbit.x[:n_quarter]))
    x_end = np.max(np.abs(orbit.x[-n_quarter:]))

    is_damped = x_end < 0.5 * x_start  # должно затухнуть хотя бы в 2 раза

    # Проверка масштаба: амплитуды должны быть порядка начального смещения
    scale_ok = 0.01 * initial.x0 < x_start < 10 * initial.x0

    return {
        "x0_um": initial.x0 * 1e6,
        "x_start_amplitude_um": x_start * 1e6,
        "x_end_amplitude_um": x_end * 1e6,
        "damping_ratio": x_end / x_start if x_start > 0 else 0,
        "is_damped": is_damped,
        "scale_ok": scale_ok,
        "verdict": _get_verdict(is_damped, scale_ok),
    }


def _get_verdict(is_damped: bool, scale_ok: bool) -> str:
    """Формирует вердикт проверки."""
    if not scale_ok:
        return "ОШИБКА: неправильный масштаб! Проверить единицы K, C (должны быть Н/м и Н·с/м)"
    if not is_damped:
        return "ОШИБКА: колебания не затухают! Проверить знаки K, C"
    return "OK: демпфирование работает, масштаб корректный"


def plot_orbit(orbit: OrbitResult,
               title: str = "Орбита ротора",
               show_transient: bool = True,
               save_path: Optional[str] = None):
    """Построить графики орбиты."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    i_ss = orbit.steady_state_start
    c = orbit.clearance

    # 1. Орбита x-y
    ax1 = axes[0, 0]
    if show_transient:
        ax1.plot(orbit.x[:i_ss]*1e6, orbit.y[:i_ss]*1e6,
                 'b-', alpha=0.3, linewidth=0.5, label='Переходный')
    ax1.plot(orbit.x[i_ss:]*1e6, orbit.y[i_ss:]*1e6,
             'b-', linewidth=1, label='Установившийся')
    ax1.plot(orbit.x[-1]*1e6, orbit.y[-1]*1e6, 'ro', markersize=6)

    # Граница зазора
    theta = np.linspace(0, 2*np.pi, 100)
    ax1.plot(c*1e6*np.cos(theta), c*1e6*np.sin(theta),
             'r--', linewidth=2, label=f'Зазор c = {c*1e6:.0f} мкм')
    ax1.plot(0.5*c*1e6*np.cos(theta), 0.5*c*1e6*np.sin(theta),
             'g:', linewidth=1, label='50% зазора')

    ax1.set_xlabel('x, мкм')
    ax1.set_ylabel('y, мкм')
    ax1.set_title('Орбита центра вала')
    ax1.axis('equal')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=8)

    # 2. x(t), y(t)
    ax2 = axes[0, 1]
    t_ms = orbit.t * 1000
    ax2.plot(t_ms, orbit.x*1e6, 'b-', label='x(t)', linewidth=0.8)
    ax2.plot(t_ms, orbit.y*1e6, 'r-', label='y(t)', linewidth=0.8)
    ax2.axhline(y=c*1e6, color='k', linestyle='--', alpha=0.3)
    ax2.axhline(y=-c*1e6, color='k', linestyle='--', alpha=0.3)
    ax2.set_xlabel('Время, мс')
    ax2.set_ylabel('Смещение, мкм')
    ax2.set_title('Смещения x(t), y(t)')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # 3. Радиус r(t)
    ax3 = axes[1, 0]
    r = np.sqrt(orbit.x**2 + orbit.y**2)
    ax3.plot(t_ms, r*1e6, 'b-', linewidth=0.8)
    ax3.axhline(y=c*1e6, color='r', linestyle='--', label=f'c = {c*1e6:.0f} мкм')
    ax3.axhline(y=0.5*c*1e6, color='g', linestyle=':', label='0.5·c')
    ax3.set_xlabel('Время, мс')
    ax3.set_ylabel('r = √(x²+y²), мкм')
    ax3.set_title('Радиальное смещение')
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    # 4. Информация
    ax4 = axes[1, 1]
    ax4.axis('off')

    rpm = orbit.omega * 60 / (2*np.pi)
    info = f"""
ПАРАМЕТРЫ
─────────────────────────
Скорость: {rpm:.0f} rpm ({orbit.omega:.1f} рад/с)
Масса ротора: {orbit.mass:.2f} кг
Зазор: {c*1e6:.0f} мкм
Число оборотов: {orbit.n_periods_total:.0f}

РЕЗУЛЬТАТЫ (установившийся режим)
─────────────────────────
Амплитуда x: {orbit.x_amplitude*1e6:.2f} мкм
Амплитуда y: {orbit.y_amplitude*1e6:.2f} мкм
Max смещение: {orbit.max_displacement*1e6:.2f} мкм

ОЦЕНКА
─────────────────────────
max(r)/c = {orbit.r_over_c:.3f} ({orbit.r_over_c*100:.1f}%)
Внутри зазора: {'ДА' if orbit.is_bounded else 'НЕТ'}
Безопасно (<50%): {'ДА' if orbit.is_safe else 'НЕТ'}
    """
    ax4.text(0.05, 0.95, info, transform=ax4.transAxes,
             fontsize=10, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    fig.suptitle(title, fontsize=14, fontweight='bold')
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

    return fig


def plot_orbit_comparison(orbits: list, labels: list,
                          title: str = "Сравнение орбит",
                          save_path: Optional[str] = None):
    """Сравнить несколько орбит."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(orbits)))

    # Орбиты
    ax1 = axes[0]
    for orbit, label, color in zip(orbits, labels, colors):
        i_ss = orbit.steady_state_start
        ax1.plot(orbit.x[i_ss:]*1e6, orbit.y[i_ss:]*1e6,
                 '-', color=color, linewidth=1.5, label=label)

    # Зазор (берём от первой орбиты)
    c = orbits[0].clearance
    theta = np.linspace(0, 2*np.pi, 100)
    ax1.plot(c*1e6*np.cos(theta), c*1e6*np.sin(theta), 'k--', linewidth=2, label='Зазор')

    ax1.set_xlabel('x, мкм')
    ax1.set_ylabel('y, мкм')
    ax1.set_title('Установившиеся орбиты')
    ax1.axis('equal')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=9)

    # Амплитуды и r/c
    ax2 = axes[1]
    x_pos = np.arange(len(orbits))
    r_over_c = [o.r_over_c * 100 for o in orbits]

    ax2.bar(x_pos, r_over_c, color=[colors[i] for i in range(len(orbits))], alpha=0.7)
    ax2.axhline(y=80, color='r', linestyle='--', label='Предел (80%)')
    ax2.axhline(y=50, color='g', linestyle=':', label='Безопасно (50%)')

    ax2.set_xlabel('Кейс')
    ax2.set_ylabel('max(r)/c, %')
    ax2.set_title('Относительное смещение')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(labels, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    fig.suptitle(title, fontsize=14, fontweight='bold')
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

    return fig
