"""
Módulo para simular la sensibilidad a las condiciones iniciales
y verificar la tasa de divergencia exponencial guiada por el exponente de Lyapunov.
"""

from typing import Tuple, Dict, Any
import numpy as np
from .maps import LogisticMap, TentMap


def simulate_trajectory_divergence(
    map_instance: LogisticMap | TentMap,
    x0: float = 0.4,
    delta0: float = 1e-8,
    n_steps: int = 60,
) -> Dict[str, np.ndarray]:
    r"""
    Simula dos trayectorias que comienzan a una distancia infinitesimal delta0:
        x_0^{(1)} = x_0
        x_0^{(2)} = x_0 + \delta_0

    Calcula la evolución de la separación:
        \delta_n = |x_n^{(1)} - x_n^{(2)}|
        \ln(\delta_n) \approx \ln(\delta_0) + \lambda \cdot n   (en régimen lineal antes de saturación)

    Retorna:
    --------
    Dict con arrays:
        'steps': n = 0, 1, ..., n_steps - 1
        'x1': trayectoria 1
        'x2': trayectoria 2
        'delta': diferencia absoluta |x1 - x2|
        'log_delta': logaritmo natural de la diferencia
    """
    x1_arr = np.empty(n_steps, dtype=np.float64)
    x2_arr = np.empty(n_steps, dtype=np.float64)
    delta_arr = np.empty(n_steps, dtype=np.float64)

    x1 = float(x0)
    x2 = float(x0 + delta0)

    for i in range(n_steps):
        x1_arr[i] = x1
        x2_arr[i] = x2
        delta = abs(x1 - x2)
        delta_arr[i] = delta if delta > 1e-16 else 1e-16

        x1 = map_instance.evaluate(x1)
        x2 = map_instance.evaluate(x2)

    return {
        'steps': np.arange(n_steps, dtype=int),
        'x1': x1_arr,
        'x2': x2_arr,
        'delta': delta_arr,
        'log_delta': np.log(delta_arr),
    }
