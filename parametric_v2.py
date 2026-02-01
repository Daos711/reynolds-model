#!/usr/bin/env python3
"""
Параметрический модуль v2: Правильное разрешение геометрии текстуры.

В отличие от subgrid-подхода, здесь лунки разрешаются на сетке,
создавая локальные градиенты ∂H/∂φ, которые генерируют
гидродинамический эффект текстуры.

Ключевой элемент — эллипсоидальный профиль:
    H += H_p * sqrt(1 - (Δφ/B)² - ((Z-Z_c)/A)²)

Использование:
    python parametric_v2.py --single --epsilon 0.6
    python parametric_v2.py --sweep-epsilon
    python parametric_v2.py --zone-study
    python parametric_v2.py --sweep-rows
"""

import numpy as np
from numba import njit
from joblib import Parallel, delayed
from dataclasses import dataclass, field
from typing import Tuple, List, Optional, Dict
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
import argparse
import time

# Директория для результатов
OUT_DIR = Path("results/parametric_v2")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# ПАРАМЕТРЫ
# ============================================================================

@dataclass
class BearingParams:
    """Параметры подшипника."""
    R: float = 0.035        # радиус подшипника (м)
    c: float = 0.00005      # радиальный зазор (м)
    L: float = 0.056        # длина подшипника (м)
    n: float = 2980         # обороты (об/мин)
    eta: float = 0.01105    # вязкость (Па·с)
    epsilon: float = 0.6    # эксцентриситет

    @property
    def omega(self) -> float:
        """Угловая скорость (рад/с)."""
        return 2 * np.pi * self.n / 60

    @property
    def U(self) -> float:
        """Линейная скорость на поверхности вала (м/с)."""
        return self.omega * (self.R - self.c)

    @property
    def pressure_scale(self) -> float:
        """Масштаб давления."""
        return (6 * self.eta * self.U * self.R) / (self.c**2)

    @property
    def load_scale(self) -> float:
        """Масштаб нагрузки."""
        return self.pressure_scale * (self.R * self.L) / 2

    @property
    def friction_scale(self) -> float:
        """Масштаб силы трения."""
        return (self.eta * self.U * self.R * self.L) / self.c


@dataclass
class TextureParams:
    """Параметры текстуры."""
    a: float = 0.003        # полуось по Z (м), ~3 мм
    b: float = 0.003        # полуось по φ (м), ~3 мм
    h: float = 0.00001      # глубина лунки (м), 10 мкм
    N_phi: int = 8          # число лунок по окружности (в зоне текстуры)
    N_z: int = 11           # число лунок по длине
    phi_min: float = np.pi / 2      # начало зоны текстуры (рад) = 90°
    phi_max: float = 3 * np.pi / 2  # конец зоны текстуры (рад) = 270°

    def get_dimple_centers(self, R: float, L: float) -> Tuple[np.ndarray, np.ndarray]:
        """Вычислить координаты центров лунок."""
        # Безразмерные полуоси
        A = 2 * self.a / L    # по Z
        B = self.b / R        # по φ

        # Центры по φ (в зоне phi_min..phi_max)
        phi_range = self.phi_max - self.phi_min
        if self.N_phi > 1:
            delta_phi_gap = (phi_range - 2 * self.N_phi * B) / (self.N_phi - 1)
            delta_phi_center = 2 * B + delta_phi_gap
        else:
            delta_phi_center = 0
        phi_centers = self.phi_min + B + delta_phi_center * np.arange(self.N_phi)

        # Центры по Z (на всю длину [-1, 1])
        if self.N_z > 1:
            delta_Z_gap = (2 - 2 * self.N_z * A) / (self.N_z - 1)
            delta_Z_center = 2 * A + delta_Z_gap
        else:
            delta_Z_center = 0
        Z_centers = -1 + A + delta_Z_center * np.arange(self.N_z)

        # Сетка центров
        phi_grid, Z_grid = np.meshgrid(phi_centers, Z_centers)
        return phi_grid.flatten(), Z_grid.flatten()


@dataclass
class MeshParams:
    """Параметры сетки."""
    num_phi: int = 100   # число точек по φ
    num_z: int = 100     # число точек по Z


# ============================================================================
# СОЗДАНИЕ СЕТКИ И ЗАЗОРА
# ============================================================================

def create_mesh(mesh: MeshParams) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    """
    Создание сетки координат.

    Returns:
        phi_1D, Z_1D, Phi_mesh, Z_mesh, d_phi, d_Z
    """
    phi_1D = np.linspace(0, 2 * np.pi, mesh.num_phi)
    Z_1D = np.linspace(-1, 1, mesh.num_z)
    Phi_mesh, Z_mesh = np.meshgrid(phi_1D, Z_1D)
    d_phi = phi_1D[1] - phi_1D[0]
    d_Z = Z_1D[1] - Z_1D[0]
    return phi_1D, Z_1D, Phi_mesh, Z_mesh, d_phi, d_Z


def create_H_smooth(Phi_mesh: np.ndarray, epsilon: float) -> np.ndarray:
    """Зазор гладкого подшипника: H = 1 + ε·cos(φ)."""
    return 1 + epsilon * np.cos(Phi_mesh)


def create_H_textured(H0: np.ndarray, texture: TextureParams,
                      Phi_mesh: np.ndarray, Z_mesh: np.ndarray,
                      bearing: BearingParams) -> np.ndarray:
    """
    Зазор с эллипсоидальными лунками.

    Ключевая формула:
        H += H_p * sqrt(1 - (Δφ/B)² - ((Z-Z_c)/A)²)

    где:
        H_p = h/c — безразмерная глубина
        A = 2a/L — безразмерная полуось по Z
        B = b/R — безразмерная полуось по φ
    """
    H = H0.copy()

    # Безразмерные параметры
    H_p = texture.h / bearing.c
    A = 2 * texture.a / bearing.L
    B = texture.b / bearing.R

    # Центры лунок
    phi_centers, Z_centers = texture.get_dimple_centers(bearing.R, bearing.L)

    for k in range(len(phi_centers)):
        phi_c = phi_centers[k]
        Z_c = Z_centers[k]

        # Угловое расстояние с учётом периодичности
        delta_phi = np.arctan2(np.sin(Phi_mesh - phi_c), np.cos(Phi_mesh - phi_c))

        # Уравнение эллипса
        expr = (delta_phi / B)**2 + ((Z_mesh - Z_c) / A)**2

        # Точки внутри лунки
        inside = expr <= 1

        # Эллипсоидальный профиль — ключевой момент!
        H[inside] += H_p * np.sqrt(1 - expr[inside])

    return H


# ============================================================================
# РЕШАТЕЛЬ РЕЙНОЛЬДСА (NUMBA)
# ============================================================================

@njit
def solve_reynolds_numba(H: np.ndarray, d_phi: float, d_Z: float,
                         R: float, L: float,
                         omega: float = 1.5, tol: float = 1e-5,
                         max_iter: int = 20000) -> Tuple[np.ndarray, float, int]:
    """
    Решатель уравнения Рейнольдса методом Гаусса-Зейделя.

    Безразмерное уравнение:
        ∂/∂φ(H³ ∂P/∂φ) + (D/L)² ∂/∂Z(H³ ∂P/∂Z) = ∂H/∂φ

    Args:
        H: безразмерный зазор (N_Z × N_phi)
        d_phi: шаг по φ
        d_Z: шаг по Z
        R: радиус подшипника
        L: длина подшипника
        omega: параметр релаксации (1.0-1.8)
        tol: точность сходимости
        max_iter: максимум итераций

    Returns:
        P: давление (безразмерное)
        delta: финальная погрешность
        iteration: число итераций
    """
    N_Z, N_phi = H.shape
    P = np.zeros((N_Z, N_phi))

    # Вычисляем H на полуцелых узлах
    H_i_plus_half = np.zeros((N_Z, N_phi))
    H_i_minus_half = np.zeros((N_Z, N_phi))
    H_j_plus_half = np.zeros((N_Z, N_phi))
    H_j_minus_half = np.zeros((N_Z, N_phi))

    for i in range(N_Z):
        for j in range(N_phi - 1):
            H_i_plus_half[i, j] = 0.5 * (H[i, j] + H[i, j + 1])
        H_i_plus_half[i, N_phi - 1] = 0.5 * (H[i, N_phi - 1] + H[i, 0])

    for i in range(N_Z):
        for j in range(1, N_phi):
            H_i_minus_half[i, j] = 0.5 * (H[i, j] + H[i, j - 1])
        H_i_minus_half[i, 0] = 0.5 * (H[i, 0] + H[i, N_phi - 1])

    for i in range(N_Z - 1):
        for j in range(N_phi):
            H_j_plus_half[i, j] = 0.5 * (H[i, j] + H[i + 1, j])

    for i in range(1, N_Z):
        for j in range(N_phi):
            H_j_minus_half[i, j] = 0.5 * (H[i, j] + H[i - 1, j])

    # Коэффициент (D/L)²
    D_over_L = 2 * R / L
    alpha_sq = (D_over_L * d_phi / d_Z)**2

    # Коэффициенты A, B, C, D
    A = H_i_plus_half**3
    B = H_i_minus_half**3
    C = alpha_sq * H_j_plus_half**3
    D_coef = alpha_sq * H_j_minus_half**3

    # Суммарный коэффициент E
    E = A + B + C + D_coef

    # Правая часть F = d_phi * (H_{i+1/2} - H_{i-1/2})
    F = np.zeros((N_Z, N_phi))
    for i in range(N_Z):
        for j in range(N_phi):
            F[i, j] = d_phi * (H_i_plus_half[i, j] - H_i_minus_half[i, j])

    # Итерационный процесс
    delta = 1.0
    iteration = 0

    while delta > tol and iteration < max_iter:
        delta = 0.0
        norm_P = 0.0

        for i in range(1, N_Z - 1):
            for j in range(1, N_phi - 1):
                Ai = A[i, j]
                Bi = B[i, j]
                Ci = C[i, j]
                Di = D_coef[i, j]
                Ei = E[i, j]
                Fi = F[i, j]

                P_old_ij = P[i, j]

                # Индексы с периодичностью по φ
                j_plus = (j + 1) % N_phi
                j_minus = (j - 1 + N_phi) % N_phi

                # Обновление давления
                P_new = (Ai * P[i, j_plus] + Bi * P[i, j_minus] +
                        Ci * P[i + 1, j] + Di * P[i - 1, j] - Fi) / Ei

                # Условие кавитации: P >= 0
                if P_new < 0:
                    P_new = 0.0

                # Релаксация
                P[i, j] = P_old_ij + omega * (P_new - P_old_ij)

                delta += abs(P[i, j] - P_old_ij)
                norm_P += abs(P[i, j])

        # Периодические граничные условия по φ
        for i in range(N_Z):
            P[i, 0] = P[i, N_phi - 2]
            P[i, N_phi - 1] = P[i, 1]

        # Граничные условия Дирихле по Z
        for j in range(N_phi):
            P[0, j] = 0.0
            P[N_Z - 1, j] = 0.0

        # Нормализация погрешности
        if norm_P > 1e-10:
            delta /= norm_P

        iteration += 1

    return P, delta, iteration


# ============================================================================
# РАСЧЁТ ХАРАКТЕРИСТИК
# ============================================================================

def compute_dP_dphi(P: np.ndarray, d_phi: float) -> np.ndarray:
    """Производная давления по φ (центральные разности)."""
    N_Z, N_phi = P.shape
    dP_dphi = np.zeros_like(P)

    # Центральные разности с учётом периодичности
    dP_dphi[:, 1:-1] = (P[:, 2:] - P[:, :-2]) / (2 * d_phi)
    dP_dphi[:, 0] = (P[:, 1] - P[:, -2]) / (2 * d_phi)
    dP_dphi[:, -1] = dP_dphi[:, 0]  # Периодичность

    return dP_dphi


def compute_characteristics(P: np.ndarray, H: np.ndarray,
                           Phi_mesh: np.ndarray, Z_mesh: np.ndarray,
                           phi_1D: np.ndarray, Z_1D: np.ndarray,
                           d_phi: float,
                           bearing: BearingParams) -> Dict:
    """
    Расчёт характеристик подшипника.

    Returns:
        dict с W, F_friction, mu, Q и другими параметрами
    """
    cos_phi = np.cos(Phi_mesh)
    sin_phi = np.sin(Phi_mesh)

    # Безразмерные компоненты нагрузки
    F_r = np.trapz(np.trapz(P * cos_phi, phi_1D, axis=1), Z_1D)
    F_t = np.trapz(np.trapz(P * sin_phi, phi_1D, axis=1), Z_1D)
    F_dimless = np.sqrt(F_r**2 + F_t**2)

    # Размерная нагрузка
    W = F_dimless * bearing.load_scale

    # Производная давления
    dP_dphi = compute_dP_dphi(P, d_phi)

    # Безразмерная сила трения
    # f* = ∫∫ [1/H + 3H·∂P/∂φ] dφ dZ
    Integrand = (1 / H) + 3 * H * dP_dphi
    f_dimless = np.trapz(np.trapz(Integrand, phi_1D, axis=1), Z_1D)

    # Размерная сила трения
    F_friction = f_dimless * bearing.friction_scale

    # Коэффициент трения
    mu = F_friction / W if W > 0 else 0.0

    # Расход смазки
    # Q* = ∫∫ [H - 0.5·H³·∂P/∂φ] dφ dZ
    q_integrand = H - 0.5 * H**3 * dP_dphi
    Q_dimless = np.trapz(np.trapz(q_integrand, phi_1D, axis=1), Z_1D)
    Q = bearing.U * bearing.c * bearing.R * Q_dimless

    # Максимальное давление
    P_max = np.max(P) * bearing.pressure_scale

    return {
        'W': W,
        'F_friction': F_friction,
        'mu': mu,
        'Q': Q,
        'P_max': P_max,
        'F_r': F_r * bearing.load_scale,
        'F_t': F_t * bearing.load_scale,
    }


# ============================================================================
# ОДИНОЧНЫЙ РАСЧЁТ
# ============================================================================

def run_single(bearing: BearingParams, texture: TextureParams,
               mesh: MeshParams, verbose: bool = True) -> Dict:
    """
    Одиночный расчёт: сравнение гладкого и текстурированного подшипника.
    """
    if verbose:
        print("\n" + "=" * 60)
        print("ОДИНОЧНЫЙ РАСЧЁТ")
        print("=" * 60)
        print(f"Подшипник: R={bearing.R*1000:.1f}мм, L={bearing.L*1000:.1f}мм, c={bearing.c*1e6:.0f}мкм")
        print(f"Режим: ε={bearing.epsilon}, n={bearing.n} об/мин")
        print(f"Сетка: {mesh.num_phi}×{mesh.num_z}")
        print(f"Текстура: a={texture.a*1000:.1f}мм, b={texture.b*1000:.1f}мм, h={texture.h*1e6:.0f}мкм")
        print(f"          N_phi={texture.N_phi}, N_z={texture.N_z}")
        print(f"          Зона: {np.degrees(texture.phi_min):.0f}°-{np.degrees(texture.phi_max):.0f}°")

    # Создание сетки
    phi_1D, Z_1D, Phi_mesh, Z_mesh, d_phi, d_Z = create_mesh(mesh)

    # Зазор гладкого подшипника
    H_smooth = create_H_smooth(Phi_mesh, bearing.epsilon)

    # Зазор с текстурой
    H_textured = create_H_textured(H_smooth, texture, Phi_mesh, Z_mesh, bearing)

    # Решение для гладкого
    t0 = time.time()
    P_smooth, delta_s, iter_s = solve_reynolds_numba(
        H_smooth, d_phi, d_Z, bearing.R, bearing.L)
    t_smooth = time.time() - t0

    # Решение для текстурированного
    t0 = time.time()
    P_textured, delta_t, iter_t = solve_reynolds_numba(
        H_textured, d_phi, d_Z, bearing.R, bearing.L)
    t_textured = time.time() - t0

    if verbose:
        print(f"\nГладкий: {iter_s} итер, delta={delta_s:.2e}, время={t_smooth:.2f}с")
        print(f"Текстур.: {iter_t} итер, delta={delta_t:.2e}, время={t_textured:.2f}с")

    # Характеристики
    chars_smooth = compute_characteristics(
        P_smooth, H_smooth, Phi_mesh, Z_mesh, phi_1D, Z_1D, d_phi, bearing)
    chars_textured = compute_characteristics(
        P_textured, H_textured, Phi_mesh, Z_mesh, phi_1D, Z_1D, d_phi, bearing)

    # Сравнение
    delta_W = (chars_textured['W'] - chars_smooth['W']) / chars_smooth['W'] * 100
    delta_mu = (chars_textured['mu'] - chars_smooth['mu']) / chars_smooth['mu'] * 100
    delta_Q = (chars_textured['Q'] - chars_smooth['Q']) / chars_smooth['Q'] * 100

    if verbose:
        print(f"\n--- Результаты ---")
        print(f"{'Параметр':<20} {'Гладкий':>15} {'Текстурир.':>15} {'Изменение':>12}")
        print("-" * 65)
        print(f"{'W, Н':<20} {chars_smooth['W']:>15.1f} {chars_textured['W']:>15.1f} {delta_W:>+11.2f}%")
        print(f"{'F_трения, Н':<20} {chars_smooth['F_friction']:>15.2f} {chars_textured['F_friction']:>15.2f}")
        print(f"{'μ':<20} {chars_smooth['mu']:>15.6f} {chars_textured['mu']:>15.6f} {delta_mu:>+11.2f}%")
        print(f"{'Q, л/с':<20} {chars_smooth['Q']*1000:>15.4f} {chars_textured['Q']*1000:>15.4f} {delta_Q:>+11.2f}%")
        print(f"{'P_max, МПа':<20} {chars_smooth['P_max']/1e6:>15.2f} {chars_textured['P_max']/1e6:>15.2f}")

    return {
        'smooth': chars_smooth,
        'textured': chars_textured,
        'delta_W_pct': delta_W,
        'delta_mu_pct': delta_mu,
        'delta_Q_pct': delta_Q,
        'H_smooth': H_smooth,
        'H_textured': H_textured,
        'P_smooth': P_smooth,
        'P_textured': P_textured,
        'Phi_mesh': Phi_mesh,
        'Z_mesh': Z_mesh,
    }


# ============================================================================
# ИССЛЕДОВАНИЕ ПО ЭКСЦЕНТРИСИТЕТУ
# ============================================================================

def compute_for_epsilon(epsilon: float, bearing: BearingParams,
                        texture: TextureParams, mesh: MeshParams) -> Dict:
    """Расчёт для одного значения ε (для параллелизации)."""
    bearing_copy = BearingParams(
        R=bearing.R, c=bearing.c, L=bearing.L,
        n=bearing.n, eta=bearing.eta, epsilon=epsilon
    )

    phi_1D, Z_1D, Phi_mesh, Z_mesh, d_phi, d_Z = create_mesh(mesh)

    H_smooth = create_H_smooth(Phi_mesh, epsilon)
    H_textured = create_H_textured(H_smooth, texture, Phi_mesh, Z_mesh, bearing_copy)

    P_smooth, _, _ = solve_reynolds_numba(H_smooth, d_phi, d_Z, bearing.R, bearing.L)
    P_textured, _, _ = solve_reynolds_numba(H_textured, d_phi, d_Z, bearing.R, bearing.L)

    chars_smooth = compute_characteristics(
        P_smooth, H_smooth, Phi_mesh, Z_mesh, phi_1D, Z_1D, d_phi, bearing_copy)
    chars_textured = compute_characteristics(
        P_textured, H_textured, Phi_mesh, Z_mesh, phi_1D, Z_1D, d_phi, bearing_copy)

    return {
        'epsilon': epsilon,
        'W_smooth': chars_smooth['W'],
        'W_textured': chars_textured['W'],
        'mu_smooth': chars_smooth['mu'],
        'mu_textured': chars_textured['mu'],
        'Q_smooth': chars_smooth['Q'],
        'Q_textured': chars_textured['Q'],
    }


def run_sweep_epsilon(bearing: BearingParams, texture: TextureParams,
                      mesh: MeshParams,
                      eps_min: float = 0.1, eps_max: float = 0.8,
                      eps_num: int = 15, n_jobs: int = -1) -> Dict:
    """Параметрическое исследование по эксцентриситету."""
    print("\n" + "=" * 60)
    print("ИССЛЕДОВАНИЕ ПО ЭКСЦЕНТРИСИТЕТУ")
    print("=" * 60)

    epsilon_values = np.linspace(eps_min, eps_max, eps_num)

    print(f"ε ∈ [{eps_min}, {eps_max}], {eps_num} точек")
    print(f"Сетка: {mesh.num_phi}×{mesh.num_z}")
    print(f"Параллельных процессов: {n_jobs if n_jobs > 0 else 'все'}")

    t0 = time.time()
    results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(compute_for_epsilon)(eps, bearing, texture, mesh)
        for eps in epsilon_values
    )
    elapsed = time.time() - t0

    print(f"\nВремя расчёта: {elapsed:.1f} с")

    # Собрать результаты
    data = {
        'epsilon': [r['epsilon'] for r in results],
        'W_smooth': [r['W_smooth'] for r in results],
        'W_textured': [r['W_textured'] for r in results],
        'mu_smooth': [r['mu_smooth'] for r in results],
        'mu_textured': [r['mu_textured'] for r in results],
        'Q_smooth': [r['Q_smooth'] for r in results],
        'Q_textured': [r['Q_textured'] for r in results],
    }

    # Построить графики
    plot_epsilon_sweep(data)

    return data


def plot_epsilon_sweep(data: Dict):
    """Графики исследования по ε."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    eps = data['epsilon']

    # W(ε)
    ax1 = axes[0]
    ax1.plot(eps, data['W_smooth'], 'o-', label='Гладкий', linewidth=2)
    ax1.plot(eps, data['W_textured'], 's-', label='Текстурированный', linewidth=2)
    ax1.set_xlabel('ε', fontsize=12)
    ax1.set_ylabel('W, Н', fontsize=12)
    ax1.set_title('Несущая способность', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # μ(ε)
    ax2 = axes[1]
    ax2.plot(eps, data['mu_smooth'], 'o-', label='Гладкий', linewidth=2)
    ax2.plot(eps, data['mu_textured'], 's-', label='Текстурированный', linewidth=2)
    ax2.set_xlabel('ε', fontsize=12)
    ax2.set_ylabel('μ', fontsize=12)
    ax2.set_title('Коэффициент трения', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Q(ε)
    ax3 = axes[2]
    ax3.plot(eps, [q*1000 for q in data['Q_smooth']], 'o-', label='Гладкий', linewidth=2)
    ax3.plot(eps, [q*1000 for q in data['Q_textured']], 's-', label='Текстурированный', linewidth=2)
    ax3.set_xlabel('ε', fontsize=12)
    ax3.set_ylabel('Q, л/с', fontsize=12)
    ax3.set_title('Расход смазки', fontsize=14)
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    fig.suptitle('Влияние эксцентриситета', fontsize=16, fontweight='bold')
    fig.tight_layout()
    fig.savefig(OUT_DIR / 'epsilon_sweep.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"График: {OUT_DIR}/epsilon_sweep.png")


# ============================================================================
# ИССЛЕДОВАНИЕ ЗОН ТЕКСТУРЫ
# ============================================================================

def run_zone_study(bearing: BearingParams, texture: TextureParams,
                   mesh: MeshParams) -> Dict:
    """Исследование влияния угловой зоны текстуры."""
    print("\n" + "=" * 60)
    print("ИССЛЕДОВАНИЕ ЗОН ТЕКСТУРЫ")
    print("=" * 60)

    # Сначала гладкий подшипник
    phi_1D, Z_1D, Phi_mesh, Z_mesh, d_phi, d_Z = create_mesh(mesh)
    H_smooth = create_H_smooth(Phi_mesh, bearing.epsilon)
    P_smooth, _, _ = solve_reynolds_numba(H_smooth, d_phi, d_Z, bearing.R, bearing.L)
    chars_smooth = compute_characteristics(
        P_smooth, H_smooth, Phi_mesh, Z_mesh, phi_1D, Z_1D, d_phi, bearing)

    print(f"Гладкий: W={chars_smooth['W']:.1f} Н, μ={chars_smooth['mu']:.6f}")

    # Зоны для сравнения
    zones = [
        ("0°-360° (полный)", 0, 360),
        ("45°-315°", 45, 315),
        ("60°-300°", 60, 300),
        ("90°-270° (стандарт)", 90, 270),
        ("120°-240°", 120, 240),
        ("135°-225° (узкая)", 135, 225),
        ("90°-180° (сходящ.)", 90, 180),
        ("180°-270° (расход.)", 180, 270),
    ]

    results = []

    print(f"\n{'Зона':<22} {'W, Н':>10} {'μ':>10} {'ΔW%':>8} {'Δμ%':>8}")
    print("-" * 65)

    for zone_name, phi_min_deg, phi_max_deg in zones:
        tex = TextureParams(
            a=texture.a, b=texture.b, h=texture.h,
            N_phi=texture.N_phi, N_z=texture.N_z,
            phi_min=np.radians(phi_min_deg),
            phi_max=np.radians(phi_max_deg),
        )

        H_textured = create_H_textured(H_smooth, tex, Phi_mesh, Z_mesh, bearing)
        P_textured, _, _ = solve_reynolds_numba(H_textured, d_phi, d_Z, bearing.R, bearing.L)
        chars = compute_characteristics(
            P_textured, H_textured, Phi_mesh, Z_mesh, phi_1D, Z_1D, d_phi, bearing)

        delta_W = (chars['W'] - chars_smooth['W']) / chars_smooth['W'] * 100
        delta_mu = (chars['mu'] - chars_smooth['mu']) / chars_smooth['mu'] * 100

        results.append({
            'zone_name': zone_name,
            'phi_min': phi_min_deg,
            'phi_max': phi_max_deg,
            'W': chars['W'],
            'mu': chars['mu'],
            'delta_W_pct': delta_W,
            'delta_mu_pct': delta_mu,
        })

        print(f"{zone_name:<22} {chars['W']:>10.1f} {chars['mu']:>10.6f} {delta_W:>+8.2f} {delta_mu:>+8.2f}")

    # Найти лучшую зону
    best_mu = min(results, key=lambda x: x['delta_mu_pct'])
    best_W = max(results, key=lambda x: x['delta_W_pct'])

    print(f"\nЛучшая зона для μ: {best_mu['zone_name']} (Δμ={best_mu['delta_mu_pct']:+.2f}%)")
    print(f"Лучшая зона для W: {best_W['zone_name']} (ΔW={best_W['delta_W_pct']:+.2f}%)")

    # График
    plot_zone_study(results, chars_smooth)

    return {'smooth': chars_smooth, 'zones': results}


def plot_zone_study(results: List[Dict], smooth: Dict):
    """График исследования зон."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    names = [r['zone_name'] for r in results]
    delta_mu = [r['delta_mu_pct'] for r in results]
    delta_W = [r['delta_W_pct'] for r in results]

    # Δμ
    ax1 = axes[0]
    colors = ['green' if d < 0 else 'red' for d in delta_mu]
    bars1 = ax1.bar(range(len(names)), delta_mu, color=colors, edgecolor='black', alpha=0.7)
    ax1.axhline(0, color='black', linewidth=1)
    ax1.set_xticks(range(len(names)))
    ax1.set_xticklabels(names, rotation=45, ha='right', fontsize=9)
    ax1.set_ylabel('Δμ, %', fontsize=12)
    ax1.set_title('Изменение коэффициента трения', fontsize=14)
    ax1.grid(True, alpha=0.3, axis='y')

    # ΔW
    ax2 = axes[1]
    colors = ['green' if d > 0 else 'red' for d in delta_W]
    bars2 = ax2.bar(range(len(names)), delta_W, color=colors, edgecolor='black', alpha=0.7)
    ax2.axhline(0, color='black', linewidth=1)
    ax2.set_xticks(range(len(names)))
    ax2.set_xticklabels(names, rotation=45, ha='right', fontsize=9)
    ax2.set_ylabel('ΔW, %', fontsize=12)
    ax2.set_title('Изменение несущей способности', fontsize=14)
    ax2.grid(True, alpha=0.3, axis='y')

    fig.suptitle('Влияние угловой зоны текстуры', fontsize=16, fontweight='bold')
    fig.tight_layout()
    fig.savefig(OUT_DIR / 'zone_study.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"График: {OUT_DIR}/zone_study.png")


# ============================================================================
# ИССЛЕДОВАНИЕ РЯДНОСТИ
# ============================================================================

def run_sweep_rows(bearing: BearingParams, texture: TextureParams,
                   mesh: MeshParams,
                   n_phi_range: List[int] = None,
                   n_z_range: List[int] = None,
                   n_jobs: int = -1) -> Dict:
    """Исследование влияния числа рядов лунок."""
    print("\n" + "=" * 60)
    print("ИССЛЕДОВАНИЕ РЯДНОСТИ")
    print("=" * 60)

    if n_phi_range is None:
        n_phi_range = [4, 6, 8, 10, 12]
    if n_z_range is None:
        n_z_range = [5, 7, 9, 11, 13]

    # Гладкий
    phi_1D, Z_1D, Phi_mesh, Z_mesh, d_phi, d_Z = create_mesh(mesh)
    H_smooth = create_H_smooth(Phi_mesh, bearing.epsilon)
    P_smooth, _, _ = solve_reynolds_numba(H_smooth, d_phi, d_Z, bearing.R, bearing.L)
    chars_smooth = compute_characteristics(
        P_smooth, H_smooth, Phi_mesh, Z_mesh, phi_1D, Z_1D, d_phi, bearing)

    print(f"Гладкий: W={chars_smooth['W']:.1f} Н, μ={chars_smooth['mu']:.6f}")
    print(f"N_phi: {n_phi_range}")
    print(f"N_z: {n_z_range}")

    results = []

    for N_phi in n_phi_range:
        for N_z in n_z_range:
            tex = TextureParams(
                a=texture.a, b=texture.b, h=texture.h,
                N_phi=N_phi, N_z=N_z,
                phi_min=texture.phi_min, phi_max=texture.phi_max,
            )

            H_textured = create_H_textured(H_smooth, tex, Phi_mesh, Z_mesh, bearing)
            P_textured, _, _ = solve_reynolds_numba(H_textured, d_phi, d_Z, bearing.R, bearing.L)
            chars = compute_characteristics(
                P_textured, H_textured, Phi_mesh, Z_mesh, phi_1D, Z_1D, d_phi, bearing)

            delta_W = (chars['W'] - chars_smooth['W']) / chars_smooth['W'] * 100
            delta_mu = (chars['mu'] - chars_smooth['mu']) / chars_smooth['mu'] * 100

            results.append({
                'N_phi': N_phi,
                'N_z': N_z,
                'N_total': N_phi * N_z,
                'W': chars['W'],
                'mu': chars['mu'],
                'delta_W_pct': delta_W,
                'delta_mu_pct': delta_mu,
            })

            print(f"N_phi={N_phi:2d}, N_z={N_z:2d}: ΔW={delta_W:+6.2f}%, Δμ={delta_mu:+6.2f}%")

    # Найти оптимум
    best_mu = min(results, key=lambda x: x['delta_mu_pct'])
    print(f"\nОптимум для μ: N_phi={best_mu['N_phi']}, N_z={best_mu['N_z']} (Δμ={best_mu['delta_mu_pct']:+.2f}%)")

    # График
    plot_rows_sweep(results, n_phi_range, n_z_range)

    return {'smooth': chars_smooth, 'results': results}


def plot_rows_sweep(results: List[Dict], n_phi_range: List[int], n_z_range: List[int]):
    """Heatmap исследования рядности."""
    import pandas as pd

    # Создать матрицу Δμ
    delta_mu_matrix = np.zeros((len(n_z_range), len(n_phi_range)))

    for r in results:
        i = n_z_range.index(r['N_z'])
        j = n_phi_range.index(r['N_phi'])
        delta_mu_matrix[i, j] = r['delta_mu_pct']

    fig, ax = plt.subplots(figsize=(10, 8))

    im = ax.imshow(delta_mu_matrix, cmap='RdYlGn_r', aspect='auto')

    ax.set_xticks(range(len(n_phi_range)))
    ax.set_xticklabels(n_phi_range)
    ax.set_yticks(range(len(n_z_range)))
    ax.set_yticklabels(n_z_range)

    ax.set_xlabel('N_phi (число рядов по φ)', fontsize=12)
    ax.set_ylabel('N_z (число рядов по Z)', fontsize=12)
    ax.set_title('Изменение коэффициента трения Δμ, %', fontsize=14)

    # Аннотации
    for i in range(len(n_z_range)):
        for j in range(len(n_phi_range)):
            val = delta_mu_matrix[i, j]
            color = 'white' if abs(val) > 2 else 'black'
            ax.text(j, i, f'{val:+.1f}%', ha='center', va='center', color=color, fontsize=9)

    fig.colorbar(im, ax=ax, label='Δμ, %')
    fig.tight_layout()
    fig.savefig(OUT_DIR / 'rows_sweep.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"График: {OUT_DIR}/rows_sweep.png")


# ============================================================================
# ВИЗУАЛИЗАЦИЯ 3D
# ============================================================================

def plot_3d_comparison(result: Dict, bearing: BearingParams):
    """3D-графики H и P для гладкого и текстурированного."""
    fig = plt.figure(figsize=(16, 12))

    Phi = result['Phi_mesh']
    Z = result['Z_mesh']

    cases = [
        ('Гладкий', result['H_smooth'], result['P_smooth']),
        ('Текстурированный', result['H_textured'], result['P_textured']),
    ]

    for i, (title, H, P) in enumerate(cases):
        # H
        ax1 = fig.add_subplot(2, 2, 2*i + 1, projection='3d')
        surf1 = ax1.plot_surface(Phi, Z, H, cmap='viridis', alpha=0.8)
        ax1.set_xlabel('φ, рад')
        ax1.set_ylabel('Z')
        ax1.set_zlabel('H')
        ax1.set_title(f'{title}: Зазор H')
        fig.colorbar(surf1, ax=ax1, shrink=0.5)

        # P
        ax2 = fig.add_subplot(2, 2, 2*i + 2, projection='3d')
        surf2 = ax2.plot_surface(Phi, Z, P * bearing.pressure_scale / 1e6, cmap='plasma', alpha=0.8)
        ax2.set_xlabel('φ, рад')
        ax2.set_ylabel('Z')
        ax2.set_zlabel('P, МПа')
        ax2.set_title(f'{title}: Давление P')
        fig.colorbar(surf2, ax=ax2, shrink=0.5)

    fig.tight_layout()
    fig.savefig(OUT_DIR / 'comparison_3d.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"График: {OUT_DIR}/comparison_3d.png")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Параметрический модуль v2')

    # Режимы
    parser.add_argument('--single', action='store_true', help='Одиночный расчёт')
    parser.add_argument('--sweep-epsilon', action='store_true', help='Исследование по ε')
    parser.add_argument('--zone-study', action='store_true', help='Исследование зон текстуры')
    parser.add_argument('--sweep-rows', action='store_true', help='Исследование рядности')
    parser.add_argument('--plot-3d', action='store_true', help='3D-визуализация')

    # Параметры подшипника
    parser.add_argument('--epsilon', type=float, default=0.6, help='Эксцентриситет')
    parser.add_argument('--R', type=float, default=0.035, help='Радиус, м')
    parser.add_argument('--L', type=float, default=0.056, help='Длина, м')
    parser.add_argument('--c', type=float, default=0.00005, help='Зазор, м')
    parser.add_argument('--n', type=float, default=2980, help='Обороты, об/мин')
    parser.add_argument('--eta', type=float, default=0.01105, help='Вязкость, Па·с')

    # Параметры текстуры
    parser.add_argument('--a', type=float, default=3.0, help='Полуось a, мм')
    parser.add_argument('--b', type=float, default=3.0, help='Полуось b, мм')
    parser.add_argument('--h', type=float, default=10.0, help='Глубина лунки, мкм')
    parser.add_argument('--N-phi', type=int, default=8, help='Число лунок по φ')
    parser.add_argument('--N-z', type=int, default=11, help='Число лунок по Z')

    # Параметры сетки
    parser.add_argument('--mesh', type=int, default=100, help='Размер сетки (100/200/500)')

    # Параметры исследований
    parser.add_argument('--eps-min', type=float, default=0.1, help='Мин. ε')
    parser.add_argument('--eps-max', type=float, default=0.8, help='Макс. ε')
    parser.add_argument('--eps-num', type=int, default=15, help='Число точек по ε')
    parser.add_argument('--jobs', type=int, default=-1, help='Число параллельных процессов')

    args = parser.parse_args()

    # Создать объекты параметров
    bearing = BearingParams(
        R=args.R, c=args.c, L=args.L,
        n=args.n, eta=args.eta, epsilon=args.epsilon
    )

    texture = TextureParams(
        a=args.a * 1e-3,  # мм → м
        b=args.b * 1e-3,
        h=args.h * 1e-6,  # мкм → м
        N_phi=args.N_phi,
        N_z=args.N_z,
    )

    mesh = MeshParams(num_phi=args.mesh, num_z=args.mesh)

    # Выполнить выбранный режим
    if args.single or not any([args.sweep_epsilon, args.zone_study, args.sweep_rows]):
        result = run_single(bearing, texture, mesh)

        if args.plot_3d:
            plot_3d_comparison(result, bearing)

    if args.sweep_epsilon:
        run_sweep_epsilon(bearing, texture, mesh,
                         eps_min=args.eps_min, eps_max=args.eps_max,
                         eps_num=args.eps_num, n_jobs=args.jobs)

    if args.zone_study:
        run_zone_study(bearing, texture, mesh)

    if args.sweep_rows:
        run_sweep_rows(bearing, texture, mesh, n_jobs=args.jobs)

    print(f"\nГотово! Результаты в {OUT_DIR}")


if __name__ == "__main__":
    main()
