"""
Визуализация результатов расчёта подшипника.
"""

from typing import Optional
from pathlib import Path
import numpy as np

from .config import BearingConfig
from .reynolds_solver import ReynoldsResult


def plot_pressure_field(
    result: ReynoldsResult,
    config: BearingConfig,
    ax=None,
    dimensional: bool = True,
    title: str = "Поле давления",
):
    """
    Построить контурную карту давления.

    Args:
        result: результат решения
        config: конфигурация подшипника
        ax: оси matplotlib (если None, создаются новые)
        dimensional: True для размерного давления (МПа), False для безразмерного
        title: заголовок графика

    Returns:
        ax: оси matplotlib
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))

    PHI, Z = np.meshgrid(result.phi, result.Z, indexing='ij')
    phi_deg = np.degrees(PHI)

    if dimensional:
        P_plot = result.P * config.pressure_scale / 1e6  # МПа
        label = "p, МПа"
    else:
        P_plot = result.P
        label = "P (безразм.)"

    contour = ax.contourf(phi_deg, Z, P_plot, levels=50, cmap='jet')
    ax.set_xlabel("φ, град")
    ax.set_ylabel("Z")
    ax.set_title(title)

    cbar = plt.colorbar(contour, ax=ax)
    cbar.set_label(label)

    # Отмечаем положение минимального зазора
    phi0_deg = np.degrees(config.phi0)
    ax.axvline(phi0_deg + 180, color='white', linestyle='--', linewidth=1,
               label=f"h_min (φ={phi0_deg+180:.0f}°)")

    return ax


def plot_film_thickness(
    result: ReynoldsResult,
    config: BearingConfig,
    ax=None,
    dimensional: bool = True,
    title: str = "Толщина масляной плёнки",
):
    """
    Построить контурную карту толщины плёнки.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))

    PHI, Z = np.meshgrid(result.phi, result.Z, indexing='ij')
    phi_deg = np.degrees(PHI)

    if dimensional:
        H_plot = result.H * config.c * 1e6  # мкм
        label = "h, мкм"
    else:
        H_plot = result.H
        label = "H (безразм.)"

    contour = ax.contourf(phi_deg, Z, H_plot, levels=50, cmap='viridis')
    ax.set_xlabel("φ, град")
    ax.set_ylabel("Z")
    ax.set_title(title)

    cbar = plt.colorbar(contour, ax=ax)
    cbar.set_label(label)

    return ax


def plot_pressure_profile(
    result: ReynoldsResult,
    config: BearingConfig,
    z_index: Optional[int] = None,
    ax=None,
    dimensional: bool = True,
    title: str = "Профиль давления по окружности",
):
    """
    Построить профиль давления вдоль окружности при фиксированном Z.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))

    if z_index is None:
        z_index = len(result.Z) // 2  # середина

    phi_deg = np.degrees(result.phi)

    if dimensional:
        P_plot = result.P[:, z_index] * config.pressure_scale / 1e6
        ylabel = "p, МПа"
    else:
        P_plot = result.P[:, z_index]
        ylabel = "P (безразм.)"

    ax.plot(phi_deg, P_plot, 'b-', linewidth=2)
    ax.set_xlabel("φ, град")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title} (Z = {result.Z[z_index]:.2f})")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 360)
    ax.axhline(0, color='k', linewidth=0.5)

    # Отмечаем важные точки
    phi0_deg = np.degrees(config.phi0)
    ax.axvline(phi0_deg, color='g', linestyle='--', alpha=0.7, label='h_max')
    ax.axvline(phi0_deg + 180, color='r', linestyle='--', alpha=0.7, label='h_min')
    ax.legend()

    return ax


def plot_summary(
    result: ReynoldsResult,
    config: BearingConfig,
    save_path: Optional[str] = None,
    show_info: bool = True
):
    """
    Построить сводную диаграмму с несколькими графиками.

    Args:
        result: результат решения
        config: конфигурация подшипника
        save_path: путь для сохранения (опционально)
        show_info: показывать информационную панель
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Поле давления
    plot_pressure_field(result, config, ax=axes[0, 0])

    # Поле толщины
    plot_film_thickness(result, config, ax=axes[0, 1])

    # Профиль давления в центре
    plot_pressure_profile(result, config, ax=axes[1, 0])

    # Информация о расчёте
    axes[1, 1].axis('off')
    if show_info:
        info_text = f"""ПАРАМЕТРЫ ПОДШИПНИКА:
  Радиус R = {config.R*1000:.1f} мм
  Длина L = {config.L*1000:.1f} мм
  Зазор c = {config.c*1e6:.1f} мкм
  L/D = {config.L_D_ratio:.2f}

РЕЖИМ РАБОТЫ:
  Эксцентриситет ε = {config.epsilon}
  Угол φ₀ = {np.degrees(config.phi0):.1f}°
  Скорость n = {config.n_rpm} об/мин
  Вязкость μ = {config.mu} Па·с

РЕЗУЛЬТАТЫ:
  Максимальное давление p_max = {result.p_max/1e6:.2f} МПа
  Минимальная толщина h_min = {result.h_min*1e6:.2f} мкм

СЕТКА: {config.n_phi} × {config.n_z}
Сходимость: {"Да" if result.converged else "Нет"}
Итераций: {result.iterations}
Невязка: {result.residual:.2e}"""
        axes[1, 1].text(0.1, 0.5, info_text, transform=axes[1, 1].transAxes,
                        fontsize=10, verticalalignment='center', fontfamily='monospace')

    plt.tight_layout()

    if save_path:
        # Создаём директорию если не существует
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Сохранено: {save_path}")

    return fig
