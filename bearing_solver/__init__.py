"""
Bearing Solver - Hydrodynamic Journal Bearing Analysis Package

Пакет для расчёта гидродинамического подшипника скольжения
на основе уравнения Рейнольдса.
"""

from .config import BearingConfig, LubricantVG100
from .film_models import FilmModel, SmoothFilmModel
from .reynolds_solver import ReynoldsSolver, ReynoldsResult, solve_reynolds

__version__ = "0.1.0"
__all__ = [
    "BearingConfig",
    "LubricantVG100",
    "FilmModel",
    "SmoothFilmModel",
    "ReynoldsSolver",
    "ReynoldsResult",
    "solve_reynolds",
]
