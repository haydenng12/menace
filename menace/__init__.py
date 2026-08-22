"""MENACE: Matchbox Educable Noughts And Crosses Engine."""

from .players import MenacePlayer, RandomPlayer
from .training import TrainingStats, train_against_random

__all__ = ["MenacePlayer", "RandomPlayer", "TrainingStats", "train_against_random"]
