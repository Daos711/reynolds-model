"""
bearing_solver - Пакет для расчёта гидродинамического подшипника скольжения.

Этап 1: Базовый решатель уравнения Рейнольдса.
"""

from .config import BearingConfig, LubricantVG100
from .film_models import FilmModel, SmoothFilmModel, FilmModelWithFlowFactors
from .reynolds_solver import ReynoldsSolver, ReynoldsResult, solve_reynolds

__version__ = "0.1.0"

__all__ = [
    # Конфигурация
    "BearingConfig",
    "LubricantVG100",
    # Модели плёнки
    "FilmModel",
    "SmoothFilmModel",
    "FilmModelWithFlowFactors",
    # Решатель
    "ReynoldsSolver",
    "ReynoldsResult",
    "solve_reynolds",
]
