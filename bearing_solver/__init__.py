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
    validate_texture,
)
from .roughness import (
    RoughnessParams,
    RoughnessResult,
    sigma_from_Ra,
    combined_sigma,
    build_Ra_bushing_field,
    flow_factors_PC,
    compute_roughness_fields,
)
from .validation import (
    PatelCase,
    GwynllywCase,
    PATEL_EXPERIMENTAL,
    GWYNLLYW_REFERENCE,
    run_patel_validation,
    run_patel_validation_fixed_position,
    run_gwynllyw_validation,
    create_patel_config,
    create_gwynllyw_config,
    get_pressure_at_midplane,
)
from .orbits import (
    RotorParams,
    InitialConditions,
    OrbitResult,
    compute_orbit,
    compute_orbit_from_config,
    verify_damping,
    plot_orbit,
    plot_orbit_comparison,
)

__version__ = "0.7.0"

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
    "validate_texture",
    # Этап 7: Шероховатость
    "RoughnessParams",
    "RoughnessResult",
    "sigma_from_Ra",
    "combined_sigma",
    "build_Ra_bushing_field",
    "flow_factors_PC",
    "compute_roughness_fields",
    # Этап 9: Валидация
    "PatelCase",
    "GwynllywCase",
    "PATEL_EXPERIMENTAL",
    "GWYNLLYW_REFERENCE",
    "run_patel_validation",
    "run_patel_validation_fixed_position",
    "run_gwynllyw_validation",
    "create_patel_config",
    "create_gwynllyw_config",
    "get_pressure_at_midplane",
    # Этап 10: Орбиты
    "RotorParams",
    "InitialConditions",
    "OrbitResult",
    "compute_orbit",
    "compute_orbit_from_config",
    "verify_damping",
    "plot_orbit",
    "plot_orbit_comparison",
]
