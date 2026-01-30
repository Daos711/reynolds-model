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


# ============================================================================
# ФУНКЦИИ ФОРМАТИРОВАНИЯ
# ============================================================================

def fmt_amplitude(val_m: float) -> str:
    """
    Форматировать амплитуду с автовыбором единиц.

    < 0.1 мкм → показывать в нм
    >= 0.1 мкм → показывать в мкм
    """
    val_um = val_m * 1e6
    if abs(val_um) < 0.1:
        return f"{val_um * 1000:.2f} нм"
    elif abs(val_um) < 10:
        return f"{val_um:.4f} мкм"
    else:
        return f"{val_um:.2f} мкм"


def fmt_r_over_c(ratio: float) -> str:
    """
    Форматировать r/c с достаточной точностью.

    < 0.01% → научная нотация
    >= 0.01% → обычный формат
    """
    pct = ratio * 100
    if pct < 0.01:
        return f"{pct:.2e}%"
    elif pct < 1:
        return f"{pct:.4f}%"
    else:
        return f"{pct:.2f}%"


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


def plot_orbit_zoomed(orbit: OrbitResult,
                      eq_info: dict,
                      title: str = "Орбита ротора",
                      save_path: Optional[str] = None):
    """
    Построить графики орбиты с zoom и абсолютными координатами.

    Включает:
    1. Орбита в абсолютных координатах (видно положение равновесия)
    2. Zoom на саму орбиту (автомасштаб)
    3. x(t), y(t) с адекватным масштабом
    4. Информационный блок
    """
    fig = plt.figure(figsize=(16, 10))

    # Получаем данные
    i_ss = orbit.steady_state_start
    c = orbit.clearance

    # Положение равновесия (из eq_info)
    epsilon = eq_info.get("epsilon", 0.5)
    phi0_rad = np.radians(eq_info.get("phi0_deg", 90))

    # Абсолютные координаты равновесия
    x_eq = -epsilon * c * np.cos(phi0_rad)
    y_eq = -epsilon * c * np.sin(phi0_rad)

    # Абсолютные координаты орбиты
    x_abs = x_eq + orbit.x
    y_abs = y_eq + orbit.y

    # === 1. Орбита в абсолютных координатах (полный вид) ===
    ax1 = fig.add_subplot(2, 3, 1)

    # Граница зазора
    theta = np.linspace(0, 2*np.pi, 100)
    ax1.plot(c*1e6*np.cos(theta), c*1e6*np.sin(theta),
             'r--', linewidth=2, label='Зазор')
    ax1.plot(0.5*c*1e6*np.cos(theta), 0.5*c*1e6*np.sin(theta),
             'g:', linewidth=1, label='50% зазора')

    # Положение равновесия
    ax1.plot(x_eq*1e6, y_eq*1e6, 'ko', markersize=10, label='Равновесие')

    # Орбита (установившаяся)
    ax1.plot(x_abs[i_ss:]*1e6, y_abs[i_ss:]*1e6, 'b-', linewidth=1.5, label='Орбита')

    ax1.set_xlabel('x, мкм')
    ax1.set_ylabel('y, мкм')
    ax1.set_title('Абсолютные координаты')
    ax1.axis('equal')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=8)

    # === 2. ZOOM на орбиту (автомасштаб) ===
    ax2 = fig.add_subplot(2, 3, 2)

    # Определяем масштаб по реальным данным
    x_ss = orbit.x[i_ss:]
    y_ss = orbit.y[i_ss:]

    r_max = max(np.max(np.abs(x_ss)), np.max(np.abs(y_ss)))

    # Выбираем единицы для zoom
    if r_max * 1e6 < 0.1:  # < 0.1 мкм → нанометры
        scale = 1e9
        unit = "нм"
    else:
        scale = 1e6
        unit = "мкм"

    # Орбита в выбранных единицах
    ax2.plot(x_ss * scale, y_ss * scale, 'b-', linewidth=1.5)
    ax2.plot(x_ss[0] * scale, y_ss[0] * scale, 'go', markersize=8, label='Начало')
    ax2.plot(x_ss[-1] * scale, y_ss[-1] * scale, 'ro', markersize=6, label='Конец')

    # Автомасштаб с запасом 20%
    lim = r_max * scale * 1.2
    if lim > 0:
        ax2.set_xlim(-lim, lim)
        ax2.set_ylim(-lim, lim)

    ax2.set_xlabel(f'x, {unit}')
    ax2.set_ylabel(f'y, {unit}')
    ax2.set_title(f'ZOOM орбиты (масштаб: {unit})')
    ax2.axis('equal')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=8)

    # === 3. x(t), y(t) с автомасштабом ===
    ax3 = fig.add_subplot(2, 3, 3)

    t_ms = orbit.t * 1000

    ax3.plot(t_ms, orbit.x * scale, 'b-', linewidth=0.8, label='x(t)')
    ax3.plot(t_ms, orbit.y * scale, 'r-', linewidth=0.8, label='y(t)')

    ax3.set_xlabel('Время, мс')
    ax3.set_ylabel(f'Смещение, {unit}')
    ax3.set_title('Колебания x(t), y(t)')
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    # === 4. r(t) с автомасштабом ===
    ax4 = fig.add_subplot(2, 3, 4)

    r = np.sqrt(orbit.x**2 + orbit.y**2)
    ax4.plot(t_ms, r * scale, 'b-', linewidth=0.8)

    ax4.set_xlabel('Время, мс')
    ax4.set_ylabel(f'r = sqrt(x^2+y^2), {unit}')
    ax4.set_title('Радиальное смещение')
    ax4.grid(True, alpha=0.3)

    # === 5. Сравнение масштабов (столбчатая) ===
    ax5 = fig.add_subplot(2, 3, 5)

    categories = ['Зазор c', 'Положение\nравновесия', 'Амплитуда\nорбиты']
    values_um = [
        c * 1e6,
        epsilon * c * 1e6,
        orbit.x_amplitude * 1e6
    ]

    colors_bar = ['red', 'orange', 'blue']
    bars = ax5.bar(categories, values_um, color=colors_bar, alpha=0.7)

    # Подписи значений
    for bar, val in zip(bars, values_um):
        height = bar.get_height()
        if val < 0.1:
            label = f"{val*1000:.2f} нм"
        else:
            label = f"{val:.2f} мкм"
        ax5.annotate(label, xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

    ax5.set_ylabel('мкм')
    ax5.set_title('Сравнение масштабов')
    ax5.set_yscale('log')  # логарифмическая шкала!
    ax5.grid(True, alpha=0.3, axis='y')

    # === 6. Информация ===
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis('off')

    rpm = orbit.omega * 60 / (2 * np.pi)

    info = f"""
ПАРАМЕТРЫ
-----------------------
Скорость: {rpm:.0f} rpm
Масса ротора: {orbit.mass:.2f} кг
Зазор: {c*1e6:.0f} мкм

РАВНОВЕСИЕ
-----------------------
epsilon = {epsilon:.4f}
Смещение: {epsilon*c*1e6:.1f} мкм от центра
h_min = {(1-epsilon)*c*1e6:.1f} мкм

ОРБИТА (установившийся режим)
-----------------------
Амплитуда x: {fmt_amplitude(orbit.x_amplitude)}
Амплитуда y: {fmt_amplitude(orbit.y_amplitude)}
Max r: {fmt_amplitude(orbit.max_displacement)}

r/c = {fmt_r_over_c(orbit.r_over_c)}

ВЫВОД
-----------------------
Орбита << положения равновесия
Линейная модель валидна
    """

    ax6.text(0.05, 0.95, info, transform=ax6.transAxes,
             fontsize=10, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    fig.suptitle(title, fontsize=14, fontweight='bold')
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

    return fig


def plot_orbit_comparison_zoomed(orbits: list, labels: list,
                                  title: str = "Сравнение орбит",
                                  save_path: Optional[str] = None):
    """Сравнить орбиты с автомасштабом."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    colors = plt.cm.tab10(np.linspace(0, 1, len(orbits)))

    # Определяем общий масштаб
    all_x = np.concatenate([o.x[o.steady_state_start:] for o in orbits])
    all_y = np.concatenate([o.y[o.steady_state_start:] for o in orbits])
    r_max = max(np.max(np.abs(all_x)), np.max(np.abs(all_y)))

    if r_max * 1e6 < 0.1:
        scale, unit = 1e9, "нм"
    else:
        scale, unit = 1e6, "мкм"

    # === 1. Орбиты (zoom) ===
    ax1 = axes[0]
    for orbit, label, color in zip(orbits, labels, colors):
        i_ss = orbit.steady_state_start
        ax1.plot(orbit.x[i_ss:] * scale, orbit.y[i_ss:] * scale,
                 '-', color=color, linewidth=1.5, label=label)

    ax1.set_xlabel(f'x, {unit}')
    ax1.set_ylabel(f'y, {unit}')
    ax1.set_title(f'Орбиты (масштаб: {unit})')
    ax1.axis('equal')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=8)

    # === 2. Амплитуды ===
    ax2 = axes[1]
    x_amps = [o.x_amplitude * scale for o in orbits]
    y_amps = [o.y_amplitude * scale for o in orbits]

    x_pos = np.arange(len(orbits))
    width = 0.35

    ax2.bar(x_pos - width/2, x_amps, width, label='x', color='blue', alpha=0.7)
    ax2.bar(x_pos + width/2, y_amps, width, label='y', color='red', alpha=0.7)

    ax2.set_xlabel('Кейс')
    ax2.set_ylabel(f'Амплитуда, {unit}')
    ax2.set_title('Амплитуды колебаний')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(labels, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    # === 3. r/c (логарифм если нужно) ===
    ax3 = axes[2]
    r_over_c_pct = [o.r_over_c * 100 for o in orbits]

    bars = ax3.bar(x_pos, r_over_c_pct, color=[colors[i] for i in range(len(orbits))], alpha=0.7)

    # Подписи
    for bar, val in zip(bars, r_over_c_pct):
        height = bar.get_height()
        if val < 0.01:
            label = f"{val:.2e}%"
        else:
            label = f"{val:.4f}%"
        ax3.annotate(label, xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8, rotation=45)

    ax3.set_xlabel('Кейс')
    ax3.set_ylabel('r/c, %')
    ax3.set_title('Относительное смещение')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(labels, rotation=45, ha='right')
    ax3.grid(True, alpha=0.3, axis='y')

    # Логарифмическая шкала если разброс большой
    if max(r_over_c_pct) / (min(r_over_c_pct) + 1e-10) > 100:
        ax3.set_yscale('log')

    fig.suptitle(title, fontsize=14, fontweight='bold')
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

    return fig


def plot_orbit_overview(orbit: OrbitResult, eq_info: dict,
                        title: str = "Положение вала в подшипнике",
                        save_path: Optional[str] = None):
    """
    Показать где вал находится в подшипнике (не орбиту!).

    Используется для понимания общей картины: насколько вал смещён
    от центра подшипника в положении равновесия.
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    c = orbit.clearance
    epsilon = eq_info.get("epsilon", 0.5)
    phi0_rad = np.radians(eq_info.get("phi0_deg", 90))

    # Граница зазора
    theta = np.linspace(0, 2*np.pi, 100)
    ax.plot(c*1e6*np.cos(theta), c*1e6*np.sin(theta),
            'r-', linewidth=3, label='Граница зазора')

    # Положение равновесия
    x_eq = -epsilon * c * np.cos(phi0_rad) * 1e6
    y_eq = -epsilon * c * np.sin(phi0_rad) * 1e6

    ax.plot(x_eq, y_eq, 'ko', markersize=15, label=f'Центр вала (epsilon={epsilon:.3f})')

    # Вал (круг радиусом условно 15% зазора для наглядности)
    r_shaft_vis = 0.15 * c * 1e6
    ax.add_patch(plt.Circle((x_eq, y_eq), r_shaft_vis,
                            color='blue', alpha=0.5, label='Вал (схематично)'))

    # h_min
    ax.annotate(f'h_min = {eq_info.get("h_min_um", (1-epsilon)*c*1e6):.1f} мкм',
                xy=(x_eq + r_shaft_vis*0.7, y_eq - r_shaft_vis*0.7),
                fontsize=10)

    ax.set_xlabel('x, мкм')
    ax.set_ylabel('y, мкм')
    ax.set_title(title)
    ax.axis('equal')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')

    # Установить пределы чуть больше зазора
    lim = c * 1e6 * 1.2
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

    return fig
