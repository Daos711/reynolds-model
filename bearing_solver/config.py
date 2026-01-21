"""
Конфигурация параметров подшипника и смазки.
"""

from dataclasses import dataclass, field
from typing import Dict
import numpy as np


@dataclass
class LubricantVG100:
    """
    Свойства смазки VG100.

    Attributes:
        rho: плотность, кг/м³
        viscosity_table: динамическая вязкость μ(T), Па·с
    """
    rho: float = 900.0  # кг/м³

    # Динамическая вязкость при разных температурах, Па·с
    viscosity_table: Dict[int, float] = field(default_factory=lambda: {
        40: 0.098,
        50: 0.057,
        60: 0.037,
        70: 0.025,
    })

    def get_viscosity(self, temperature: float) -> float:
        """
        Получить вязкость при заданной температуре.
        Интерполяция между табличными значениями.

        Args:
            temperature: температура, °C

        Returns:
            Динамическая вязкость, Па·с
        """
        temps = sorted(self.viscosity_table.keys())
        viscs = [self.viscosity_table[t] for t in temps]

        if temperature <= temps[0]:
            return viscs[0]
        if temperature >= temps[-1]:
            return viscs[-1]

        # Логарифмическая интерполяция (вязкость-температура)
        log_viscs = np.log(viscs)
        return np.exp(np.interp(temperature, temps, log_viscs))


@dataclass
class BearingConfig:
    """
    Конфигурация гидродинамического подшипника.

    Все размерные величины в СИ (м, Па·с, рад/с и т.д.).

    Attributes:
        R: радиус подшипника, м
        L: длина подшипника, м
        c: радиальный зазор, м
        n_rpm: скорость вращения, об/мин
        mu: динамическая вязкость смазки, Па·с
        epsilon: эксцентриситет (0 < ε < 1)
        phi0: угол положения линии центров, рад
        n_phi: количество точек сетки по φ
        n_z: количество точек сетки по Z
    """
    # Геометрия
    R: float = 0.0345       # радиус, м (D = 69 мм)
    L: float = 0.1035       # длина, м (B/D = 1.5)
    c: float = 50e-6        # зазор, м

    # Режим работы
    n_rpm: float = 2980     # об/мин
    mu: float = 0.057       # Па·с (VG100 при 50°C)

    # Положение вала
    epsilon: float = 0.6    # эксцентриситет
    phi0: float = 0.0       # угол положения линии центров, рад

    # Параметры сетки
    n_phi: int = 180        # точек по окружности
    n_z: int = 50           # точек по длине

    @property
    def D(self) -> float:
        """Диаметр подшипника, м."""
        return 2 * self.R

    @property
    def omega(self) -> float:
        """Угловая скорость, рад/с."""
        return 2 * np.pi * self.n_rpm / 60

    @property
    def U(self) -> float:
        """Линейная скорость поверхности вала, м/с."""
        return self.omega * self.R

    @property
    def D_L_ratio(self) -> float:
        """Отношение D/L."""
        return self.D / self.L

    @property
    def L_D_ratio(self) -> float:
        """Отношение L/D."""
        return self.L / self.D

    @property
    def force_scale(self) -> float:
        """
        Масштаб сил для перехода к размерным величинам.
        F = force_scale * F_bar
        где F_bar — безразмерная сила.
        """
        return 3 * self.mu * self.U * self.R**2 * self.L / self.c**2

    @property
    def pressure_scale(self) -> float:
        """
        Масштаб давления.
        p = pressure_scale * P
        где P — безразмерное давление.

        Из нормировки: P = p·c² / (6μUR)
        Следовательно: p = P · 6μUR / c²
        """
        return 6 * self.mu * self.U * self.R / self.c**2

    def create_grid(self) -> tuple[np.ndarray, np.ndarray, float, float]:
        """
        Создать расчётную сетку.

        Returns:
            phi: массив углов [0, 2π], shape (n_phi,)
            Z: массив осевых координат [-1, 1], shape (n_z,)
            d_phi: шаг по φ
            d_Z: шаг по Z
        """
        # φ ∈ [0, 2π)
        phi = np.linspace(0, 2 * np.pi, self.n_phi, endpoint=False)
        # Z ∈ [-1, 1]
        Z = np.linspace(-1, 1, self.n_z)

        d_phi = phi[1] - phi[0]
        d_Z = Z[1] - Z[0]

        return phi, Z, d_phi, d_Z

    def validate(self) -> None:
        """Проверить корректность параметров."""
        if self.R <= 0:
            raise ValueError("Радиус R должен быть положительным")
        if self.L <= 0:
            raise ValueError("Длина L должна быть положительной")
        if self.c <= 0:
            raise ValueError("Зазор c должен быть положительным")
        if not 0 < self.epsilon < 1:
            raise ValueError("Эксцентриситет epsilon должен быть в (0, 1)")
        if self.mu <= 0:
            raise ValueError("Вязкость mu должна быть положительной")
        if self.n_rpm <= 0:
            raise ValueError("Скорость n_rpm должна быть положительной")
        if self.n_phi < 10 or self.n_z < 5:
            raise ValueError("Сетка слишком грубая")


# Стандартные параметры для тестов
DEFAULT_CONFIG = BearingConfig(
    R=0.0345,
    L=0.1035,
    c=50e-6,
    epsilon=0.6,
    phi0=0.0,
    n_rpm=2980,
    mu=0.057,
    n_phi=180,
    n_z=50,
)
