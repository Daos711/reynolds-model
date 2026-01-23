"""
bearing_solver - Пакет для расчёта гидродинамического подшипника скольжения.

Этап 1: Базовый решатель уравнения Рейнольдса.
Этап 2: Расчёт сил, трения и расхода смазки.
Этап 3: Поиск положения равновесия вала.
"""

from .config import BearingConfig, LubricantVG100
from .film_models import FilmModel, SmoothFilmModel, FilmModelWithFlowFactors
from .reynolds_solver import ReynoldsSolver, ReynoldsResult, solve_reynolds
from .forces import (
    BearingForces,
    BearingFriction,
    BearingFlow,
    BearingLosses,
    Stage2Result,
    compute_forces,
    compute_friction,
    compute_flow,
    compute_losses,
    compute_stage2,
    get_shear_stress_components,
)
from .equilibrium import (
    EquilibriumResult,
    find_equilibrium,
    find_equilibrium_1d,
)
from .dynamics import (
    DynamicCoefficients,
    compute_dynamic_coefficients,
    compute_stiffness,
    compute_damping,
    check_delta_sensitivity,
)
from .stability import (
    StabilityResult,
    build_state_matrix,
    analyze_stability,
    analyze_stability_from_coefficients,
    find_stability_threshold,
)
from .texture import (
    TextureParams,
    TexturedFilmModel,
    generate_regular_centers,
    generate_phyllotaxis_centers,
)

__version__ = "0.6.0"

__all__ = [
    # Конфигурация
    "BearingConfig",
    "LubricantVG100",
    # Модели плёнки
    "FilmModel",
    "SmoothFilmModel",
    "FilmModelWithFlowFactors",
    # Этап 1: Решатель
    "ReynoldsSolver",
    "ReynoldsResult",
    "solve_reynolds",
    # Этап 2: Силы, трение, расход
    "BearingForces",
    "BearingFriction",
    "BearingFlow",
    "BearingLosses",
    "Stage2Result",
    "compute_forces",
    "compute_friction",
    "compute_flow",
    "compute_losses",
    "compute_stage2",
    "get_shear_stress_components",
    # Этап 3: Равновесие
    "EquilibriumResult",
    "find_equilibrium",
    "find_equilibrium_1d",
    # Этап 4: Динамика
    "DynamicCoefficients",
    "compute_dynamic_coefficients",
    "compute_stiffness",
    "compute_damping",
    "check_delta_sensitivity",
    # Этап 5: Устойчивость
    "StabilityResult",
    "build_state_matrix",
    "analyze_stability",
    "analyze_stability_from_coefficients",
    "find_stability_threshold",
    # Этап 6: Текстура
    "TextureParams",
    "TexturedFilmModel",
    "generate_regular_centers",
    "generate_phyllotaxis_centers",
]
