"""
Тесты для решателя уравнения Рейнольдса.

Критерии проверки:
1. Сеточная сходимость — решение не должно существенно меняться при сгущении
2. Качественная проверка — максимум давления в сходящейся зоне
3. Golden grid — сравнение с эталонным решением на мелкой сетке
"""

import numpy as np
import pytest

from bearing_solver.config import BearingConfig
from bearing_solver.film_models import SmoothFilmModel
from bearing_solver.reynolds_solver import ReynoldsSolver, solve_reynolds


class TestBasicSolver:
    """Базовые тесты решателя."""

    def test_solver_runs(self):
        """Проверка, что решатель запускается без ошибок."""
        config = BearingConfig(
            R=0.0345,
            L=0.1035,
            c=50e-6,
            epsilon=0.6,
            phi0=0.0,
            n_rpm=2980,
            mu=0.057,
            n_phi=90,
            n_z=25,
        )

        result = solve_reynolds(config)

        assert result.converged or result.iterations > 0
        assert result.P.shape == (config.n_phi, config.n_z)
        assert result.H.shape == (config.n_phi, config.n_z)

    def test_cavitation_condition(self):
        """Проверка условия кавитации P >= 0."""
        config = BearingConfig(epsilon=0.7, n_phi=90, n_z=25)
        result = solve_reynolds(config)

        assert np.all(result.P >= 0), "Давление не должно быть отрицательным"

    def test_boundary_conditions_z(self):
        """Проверка граничных условий P = 0 на торцах."""
        config = BearingConfig(epsilon=0.6, n_phi=90, n_z=25)
        result = solve_reynolds(config)

        # P = 0 на торцах (Z = -1 и Z = 1)
        assert np.allclose(result.P[:, 0], 0), "P должно быть 0 при Z = -1"
        assert np.allclose(result.P[:, -1], 0), "P должно быть 0 при Z = 1"

    def test_pressure_maximum_location(self):
        """Проверка расположения максимума давления в сходящейся зоне."""
        config = BearingConfig(
            epsilon=0.7,
            phi0=0.0,
            n_phi=180,
            n_z=50,
        )
        result = solve_reynolds(config)

        # Находим положение максимума давления
        i_max, j_max = np.unravel_index(np.argmax(result.P), result.P.shape)
        phi_max = result.phi[i_max]

        # Максимум давления должен быть в сходящейся зоне: 0 < φ - φ₀ < π
        # т.е. между 0 и π (при φ₀ = 0)
        assert 0 < phi_max < np.pi, (
            f"Максимум давления должен быть в сходящейся зоне, "
            f"φ_max = {np.degrees(phi_max):.1f}°"
        )

        # Максимум по Z должен быть около центра (Z ≈ 0)
        z_max = result.Z[j_max]
        assert abs(z_max) < 0.3, (
            f"Максимум давления должен быть около центра, Z_max = {z_max:.2f}"
        )


class TestGridConvergence:
    """Тесты сеточной сходимости."""

    def test_grid_convergence(self):
        """Проверка сходимости при сгущении сетки."""
        config_base = BearingConfig(
            R=0.0345,
            L=0.1035,
            c=50e-6,
            epsilon=0.6,
            n_rpm=2980,
            mu=0.057,
        )

        # Грубая сетка
        config_coarse = BearingConfig(**{
            **config_base.__dict__,
            'n_phi': 90,
            'n_z': 25,
        })

        # Средняя сетка
        config_medium = BearingConfig(**{
            **config_base.__dict__,
            'n_phi': 180,
            'n_z': 50,
        })

        # Мелкая сетка
        config_fine = BearingConfig(**{
            **config_base.__dict__,
            'n_phi': 360,
            'n_z': 100,
        })

        result_coarse = solve_reynolds(config_coarse)
        result_medium = solve_reynolds(config_medium)
        result_fine = solve_reynolds(config_fine)

        # Максимальное давление должно сходиться
        P_max_coarse = result_coarse.P_max
        P_max_medium = result_medium.P_max
        P_max_fine = result_fine.P_max

        # Относительное изменение при сгущении должно уменьшаться
        delta_coarse_medium = abs(P_max_medium - P_max_coarse) / P_max_fine
        delta_medium_fine = abs(P_max_fine - P_max_medium) / P_max_fine

        print(f"\nСеточная сходимость P_max:")
        print(f"  Грубая (90×25):   P_max = {P_max_coarse:.4f}")
        print(f"  Средняя (180×50): P_max = {P_max_medium:.4f}")
        print(f"  Мелкая (360×100): P_max = {P_max_fine:.4f}")
        print(f"  Δ(грубая→средняя) / P_fine = {delta_coarse_medium:.2%}")
        print(f"  Δ(средняя→мелкая) / P_fine = {delta_medium_fine:.2%}")

        # Разница между средней и мелкой сеткой должна быть меньше 10%
        assert delta_medium_fine < 0.10, (
            f"Сходимость недостаточна: Δ = {delta_medium_fine:.2%}"
        )


class TestPhysicalBehavior:
    """Тесты физического поведения."""

    def test_eccentricity_effect(self):
        """При увеличении эксцентриситета P_max должен расти."""
        results = []
        epsilons = [0.3, 0.5, 0.7, 0.9]

        for eps in epsilons:
            config = BearingConfig(epsilon=eps, n_phi=180, n_z=50)
            result = solve_reynolds(config)
            results.append(result.P_max)

        print("\nВлияние эксцентриситета на P_max:")
        for eps, p_max in zip(epsilons, results):
            print(f"  ε = {eps}: P_max = {p_max:.4f}")

        # Давление должно монотонно расти с эксцентриситетом
        for i in range(len(results) - 1):
            assert results[i] < results[i + 1], (
                f"P_max должен расти с ε: P_max(ε={epsilons[i]}) = {results[i]:.4f} "
                f">= P_max(ε={epsilons[i+1]}) = {results[i+1]:.4f}"
            )

    def test_film_thickness_minimum(self):
        """Проверка минимальной толщины плёнки."""
        config = BearingConfig(
            epsilon=0.6,
            c=50e-6,
            n_phi=180,
            n_z=50,
        )
        result = solve_reynolds(config)

        # Минимальная безразмерная толщина: H_min = 1 - ε
        expected_h_min_dimless = 1.0 - config.epsilon

        assert np.isclose(result.h_min_dimless, expected_h_min_dimless, rtol=0.01), (
            f"H_min = {result.h_min_dimless:.4f}, ожидалось {expected_h_min_dimless:.4f}"
        )

        # Размерная минимальная толщина
        expected_h_min = expected_h_min_dimless * config.c
        assert np.isclose(result.h_min, expected_h_min, rtol=0.01), (
            f"h_min = {result.h_min*1e6:.2f} мкм, ожидалось {expected_h_min*1e6:.2f} мкм"
        )


class TestSolverMethods:
    """Тесты различных методов решения."""

    def test_direct_vs_sor(self):
        """Сравнение прямого метода и SOR."""
        config = BearingConfig(
            epsilon=0.6,
            n_phi=90,
            n_z=25,
        )

        result_direct = solve_reynolds(config, method="direct")
        result_sor = solve_reynolds(config, method="sor")

        # Результаты должны быть близки
        rel_diff = abs(result_direct.P_max - result_sor.P_max) / result_direct.P_max

        print(f"\nСравнение методов:")
        print(f"  Direct: P_max = {result_direct.P_max:.4f}")
        print(f"  SOR:    P_max = {result_sor.P_max:.4f}")
        print(f"  Относительная разница: {rel_diff:.2%}")

        assert rel_diff < 0.05, f"Методы дают слишком разные результаты: {rel_diff:.2%}"


def run_golden_grid_test():
    """
    Тест с эталонной (golden) сеткой.

    Вычисляет решение на очень мелкой сетке (720×200) и сохраняет
    ключевые характеристики для последующего сравнения.
    """
    print("=" * 60)
    print("GOLDEN GRID TEST")
    print("=" * 60)

    config = BearingConfig(
        R=0.0345,
        L=0.1035,
        c=50e-6,
        epsilon=0.6,
        phi0=0.0,
        n_rpm=2980,
        mu=0.057,
        n_phi=720,
        n_z=200,
    )

    print(f"\nПараметры подшипника:")
    print(f"  R = {config.R*1000:.1f} мм")
    print(f"  L = {config.L*1000:.1f} мм")
    print(f"  c = {config.c*1e6:.1f} мкм")
    print(f"  ε = {config.epsilon}")
    print(f"  n = {config.n_rpm} об/мин")
    print(f"  μ = {config.mu} Па·с")
    print(f"  L/D = {config.L_D_ratio:.2f}")
    print(f"\nСетка: {config.n_phi} × {config.n_z}")

    print("\nРешение уравнения Рейнольдса...")
    result = solve_reynolds(config, method="direct")

    print(f"\nРезультаты (GOLDEN GRID):")
    print(f"  P_max (безразмерное) = {result.P_max:.6f}")
    print(f"  p_max = {result.p_max/1e6:.3f} МПа")
    print(f"  H_min = {result.h_min_dimless:.4f}")
    print(f"  h_min = {result.h_min*1e6:.2f} мкм")
    print(f"  Сходимость: {result.converged}")
    print(f"  Итераций: {result.iterations}")

    # Положение максимума
    i_max, j_max = np.unravel_index(np.argmax(result.P), result.P.shape)
    phi_max_deg = np.degrees(result.phi[i_max])
    z_max = result.Z[j_max]

    print(f"\nПоложение максимума давления:")
    print(f"  φ_max = {phi_max_deg:.1f}°")
    print(f"  Z_max = {z_max:.3f}")

    return result


if __name__ == "__main__":
    # Запуск теста с эталонной сеткой
    run_golden_grid_test()

    # Запуск pytest
    pytest.main([__file__, "-v"])
