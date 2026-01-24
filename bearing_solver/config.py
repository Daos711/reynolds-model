"""
Конфигурация подшипника и параметры расчёта.

Нормировка по ТЗ:
    - x = R·φ
    - Z = 2z/L ∈ [-1, 1], следовательно dz = (L/2)·dZ
    - H = h/c
    - U = ω·R, где ω = 2πn/60
    - Безразмерное давление: P = p·c² / (6μUR)
    - Масштаб сил: F = (3μUR²L/c²) × F̄
      (коэффициент 3 = 6/2 из-за dz = (L/2)·dZ при интегрировании)
"""

from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class LubricantVG100:
    """Смазка VG100 - свойства."""

    rho: float = 900.0  # кг/м³, плотность

    # Динамическая вязкость μ(T), Па·с
    viscosity_table: dict = field(default_factory=lambda: {
        40: 0.098,
        50: 0.057,
        60: 0.037,
        70: 0.025
    })

    def get_viscosity(self, T: float) -> float:
        """Интерполяция вязкости по температуре."""
        temps = sorted(self.viscosity_table.keys())
        viscs = [self.viscosity_table[t] for t in temps]
        return np.interp(T, temps, viscs)


@dataclass
class BearingConfig:
    """
    Конфигурация гидродинамического подшипника.

    Параметры:
        R: радиус подшипника, м
        L: длина подшипника, м
        c: радиальный зазор, м
        epsilon: эксцентриситет (0 < ε < 1)
        phi0: угол положения линии центров, рад
        n_rpm: скорость вращения, об/мин
        mu: динамическая вязкость, Па·с
        n_phi: число точек сетки по φ
        n_z: число точек сетки по Z
    """

    R: float              # радиус, м
    L: float              # длина, м
    c: float              # зазор, м
    epsilon: float        # эксцентриситет
    phi0: float = 0.0     # угол положения, рад
    n_rpm: float = 2980   # скорость, об/мин
    mu: float = 0.057     # вязкость, Па·с
    n_phi: int = 180      # точек по φ
    n_z: int = 50         # точек по Z

    def __post_init__(self):
        self.validate()

    def validate(self):
        """Проверка параметров."""
        assert 0 < self.R, "Радиус должен быть положительным"
        assert 0 < self.L, "Длина должна быть положительной"
        assert 0 < self.c, "Зазор должен быть положительным"
        assert 0 < self.epsilon < 1, "Эксцентриситет должен быть в (0, 1)"
        assert self.n_rpm > 0, "Скорость должна быть положительной"
        assert self.mu > 0, "Вязкость должна быть положительной"
        assert self.n_phi >= 36, "Минимум 36 точек по φ"
        assert self.n_z >= 10, "Минимум 10 точек по Z"

    @property
    def D(self) -> float:
        """Диаметр, м."""
        return 2 * self.R

    @property
    def L_D_ratio(self) -> float:
        """Отношение L/D."""
        return self.L / self.D

    @property
    def D_L_ratio(self) -> float:
        """Отношение D/L (для уравнения Рейнольдса)."""
        return self.D / self.L

    @property
    def omega(self) -> float:
        """Угловая скорость, рад/с."""
        return 2 * np.pi * self.n_rpm / 60

    @property
    def U(self) -> float:
        """Линейная скорость поверхности, м/с."""
        return self.omega * self.R

    @property
    def pressure_scale(self) -> float:
        """
        Масштаб давления, Па.

        p = P × pressure_scale
        где P - безразмерное давление

        По ТЗ: P = p·c² / (6μUR), следовательно p = P × (6μUR/c²)
        """
        return 6 * self.mu * self.U * self.R / (self.c ** 2)

    @property
    def force_scale(self) -> float:
        """
        Масштаб сил, Н.

        F = F̄ × force_scale
        где F̄ - безразмерная сила

        Вывод:
            p = P × (6μUR/c²)
            dA = R·dφ × dz = R·dφ × (L/2)·dZ  (т.к. Z = 2z/L)
            F = ∫∫ p dA = (6μUR/c²) × R × (L/2) × ∫∫ P dφ dZ
            F = (3μUR²L/c²) × F̄
        """
        return 3 * self.mu * self.U * self.R**2 * self.L / (self.c ** 2)

    def create_grid(self) -> tuple[np.ndarray, np.ndarray, float, float]:
        """
        Создать расчётную сетку.

        Returns:
            phi: массив углов [0, 2π), shape (n_phi,)
            Z: массив осевых координат [-1, 1], shape (n_z,)
            d_phi: шаг по φ
            d_Z: шаг по Z
        """
        # φ ∈ [0, 2π) - периодическая координата
        d_phi = 2 * np.pi / self.n_phi
        phi = np.linspace(0, 2*np.pi - d_phi, self.n_phi)

        # Z ∈ [-1, 1] - включая границы для ГУ
        Z = np.linspace(-1, 1, self.n_z)
        d_Z = Z[1] - Z[0]

        return phi, Z, d_phi, d_Z

    def info(self) -> str:
        """Строка с информацией о конфигурации."""
        return f"""Конфигурация подшипника:
  R = {self.R*1000:.2f} мм, D = {self.D*1000:.2f} мм
  L = {self.L*1000:.2f} мм, L/D = {self.L_D_ratio:.2f}
  c = {self.c*1e6:.1f} мкм
  ε = {self.epsilon}, φ₀ = {np.degrees(self.phi0):.1f}°
  n = {self.n_rpm} об/мин, ω = {self.omega:.2f} рад/с
  U = {self.U:.2f} м/с
  μ = {self.mu} Па·с
  Сетка: {self.n_phi}×{self.n_z}
  Масштаб давления: {self.pressure_scale:.2e} Па
  Масштаб сил: {self.force_scale:.2e} Н"""
