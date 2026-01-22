"""
bearing_solver - Пакет для расчёта гидродинамического подшипника скольжения.

Этап 1: Базовый решатель уравнения Рейнольдса.
Этап 2: Расчёт сил, трения и расхода смазки.
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
)

__version__ = "0.2.0"

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
]
