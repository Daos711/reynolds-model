"""
Тесты для решателя уравнения Рейнольдса.

Критерии проверки этапа 1:
1. Максимум давления в сходящейся части зазора (φ < π)
2. Сеточная сходимость
3. P_max растёт с ε
4. P ≥ 0 (кавитация)
"""

import numpy as np
import pytest

from bearing_solver import BearingConfig, solve_reynolds, SmoothFilmModel


@pytest.fixture
def base_config():
    """Базовая конфигурация из ТЗ."""
    return BearingConfig(
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


class TestBasicSolver:
    """Базовые тесты решателя."""

    def test_solver_runs(self, base_config):
        """Решатель выполняется без ошибок."""
        result = solve_reynolds(base_config)
        assert result.converged
        assert result.iterations > 0

    def test_pressure_non_negative(self, base_config):
        """Давление неотрицательно (кавитация)."""
        result = solve_reynolds(base_config)
        assert np.all(result.P >= 0)

    def test_boundary_conditions_z(self, base_config):
        """P = 0 на торцах (Z = ±1)."""
        result = solve_reynolds(base_config)
        # Проверяем границы
        assert np.allclose(result.P[:, 0], 0, atol=1e-10)
        assert np.allclose(result.P[:, -1], 0, atol=1e-10)

    def test_pressure_max_in_converging_zone(self, base_config):
        """Максимум давления в сходящейся зоне (0 < φ < π)."""
        result = solve_reynolds(base_config)
        i_max = np.argmax(np.max(result.P, axis=1))
        phi_max = result.phi[i_max]
        assert 0 < phi_max < np.pi, f"φ_max = {np.degrees(phi_max):.1f}° не в (0°, 180°)"


class TestFilmModel:
    """Тесты модели толщины плёнки."""

    def test_smooth_film_h_min(self, base_config):
        """H_min = 1 - ε."""
        model = SmoothFilmModel(base_config)
        assert np.isclose(model.H_min, 1 - base_config.epsilon)

    def test_smooth_film_h_max(self, base_config):
        """H_max = 1 + ε."""
        model = SmoothFilmModel(base_config)
        assert np.isclose(model.H_max, 1 + base_config.epsilon)

    def test_film_thickness_range(self, base_config):
        """H в диапазоне [1-ε, 1+ε]."""
        result = solve_reynolds(base_config)
        eps = base_config.epsilon
        assert np.min(result.H) >= 1 - eps - 1e-10
        assert np.max(result.H) <= 1 + eps + 1e-10


class TestGridConvergence:
    """Тесты сеточной сходимости."""

    def test_grid_convergence(self, base_config):
        """Решение сходится при измельчении сетки."""
        grids = [(90, 25), (180, 50), (360, 100)]
        P_max_values = []

        for n_phi, n_z in grids:
            cfg = BearingConfig(
                R=base_config.R, L=base_config.L, c=base_config.c,
                epsilon=base_config.epsilon, phi0=base_config.phi0,
                n_rpm=base_config.n_rpm, mu=base_config.mu,
                n_phi=n_phi, n_z=n_z
            )
            result = solve_reynolds(cfg)
            P_max_values.append(result.P_max)

        # Относительная разница между средней и мелкой сеткой < 1%
        rel_diff = abs(P_max_values[2] - P_max_values[1]) / P_max_values[2]
        assert rel_diff < 0.01, f"Сеточная сходимость не достигнута: {rel_diff*100:.2f}%"


class TestPhysicalBehavior:
    """Тесты физического поведения."""

    def test_p_max_increases_with_epsilon(self, base_config):
        """P_max растёт с увеличением ε."""
        epsilons = [0.4, 0.6, 0.8]
        P_max_values = []

        for eps in epsilons:
            cfg = BearingConfig(
                R=base_config.R, L=base_config.L, c=base_config.c,
                epsilon=eps, phi0=base_config.phi0,
                n_rpm=base_config.n_rpm, mu=base_config.mu,
                n_phi=90, n_z=25  # Быстрая сетка для теста
            )
            result = solve_reynolds(cfg)
            P_max_values.append(result.P_max)

        # P_max должно монотонно расти
        assert P_max_values[1] > P_max_values[0]
        assert P_max_values[2] > P_max_values[1]

    def test_h_min_decreases_with_epsilon(self, base_config):
        """h_min уменьшается с увеличением ε."""
        epsilons = [0.4, 0.6, 0.8]
        h_min_values = []

        for eps in epsilons:
            cfg = BearingConfig(
                R=base_config.R, L=base_config.L, c=base_config.c,
                epsilon=eps, phi0=base_config.phi0,
                n_rpm=base_config.n_rpm, mu=base_config.mu,
                n_phi=90, n_z=25
            )
            result = solve_reynolds(cfg)
            h_min_values.append(result.h_min)

        # h_min должно монотонно уменьшаться
        assert h_min_values[1] < h_min_values[0]
        assert h_min_values[2] < h_min_values[1]


class TestScaling:
    """Тесты масштабирования."""

    def test_pressure_scale(self, base_config):
        """Проверка масштаба давления."""
        # p = P × (6μUR/c²)
        expected = 6 * base_config.mu * base_config.U * base_config.R / (base_config.c ** 2)
        assert np.isclose(base_config.pressure_scale, expected)

    def test_force_scale(self, base_config):
        """Проверка масштаба сил."""
        # F = (6μUR²L/c²) × F̄
        expected = 6 * base_config.mu * base_config.U * base_config.R**2 * base_config.L / (base_config.c ** 2)
        assert np.isclose(base_config.force_scale, expected)

    def test_dimensional_pressure(self, base_config):
        """Размерное давление вычисляется правильно."""
        result = solve_reynolds(base_config)
        p_dim = result.get_dimensional_pressure(base_config)
        assert np.allclose(p_dim, result.P * base_config.pressure_scale)
