"""
Módulo de inicialización del paquete src.
"""

from .maps import LogisticMap, TentMap
from .lyapunov import (
    calculate_lyapunov_exponent,
    lyapunov_logistic_spectrum,
    lyapunov_tent_spectrum,
    analytical_lyapunov_tent,
    analytical_lyapunov_logistic_fixed_point,
)
from .bifurcation import calculate_bifurcation_data
from .sensitivity import simulate_trajectory_divergence
from .cobweb import generate_cobweb_data

__all__ = [
    "LogisticMap",
    "TentMap",
    "calculate_lyapunov_exponent",
    "lyapunov_logistic_spectrum",
    "lyapunov_tent_spectrum",
    "analytical_lyapunov_tent",
    "analytical_lyapunov_logistic_fixed_point",
    "calculate_bifurcation_data",
    "simulate_trajectory_divergence",
    "generate_cobweb_data",
]
