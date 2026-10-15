"""
Módulo para el cálculo eficiente del diagrama de bifurcación
para los modelos Logístico y de Tienda.
"""

from typing import Tuple
import numpy as np


def calculate_bifurcation_data(
    map_type: str,
    params: np.ndarray,
    n_transient: int = 500,
    n_retained: int = 200,
    x0: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    r"""
    Calcula los puntos del diagrama de bifurcación de manera vectorizada.

    Parámetros:
    -----------
    map_type : str
        'logistic' o 'tent'
    params : np.ndarray
        Array de valores del parámetro de control (r para logístico, p para tienda).
    n_transient : int
        Iteraciones previas descartadas para eliminar el transitorio.
    n_retained : int
        Iteraciones retenidas en el atractor por cada valor del parámetro.
    x0 : float
        Condición inicial.

    Retorna:
    --------
    Tuple[np.ndarray, np.ndarray] :
        - param_coords: Coordenadas del parámetro repetidas para cada punto retenido.
        - x_coords: Coordenadas del estado x en el atractor.
    """
    param_arr = np.asarray(params, dtype=np.float64)
    n_points = len(param_arr)

    x = np.full(n_points, x0, dtype=np.float64)
    if x0 == 0.5:
        x += 1e-5

    # 1. Eliminar transitorio
    if map_type.lower() == 'logistic':
        for _ in range(n_transient):
            x = param_arr * x * (1.0 - x)
            singular = (x <= 0.0) | (x >= 1.0)
            if np.any(singular):
                x[singular] += 1e-6

        # 2. Guardar puntos en el atractor
        retained = np.empty((n_retained, n_points), dtype=np.float64)
        for i in range(n_retained):
            x = param_arr * x * (1.0 - x)
            retained[i, :] = x

    elif map_type.lower() == 'tent':
        for _ in range(n_transient):
            x = np.where(x < 0.5, param_arr * x, param_arr * (1.0 - x))
            singular = (x == 0.5) | (x == 0.0)
            if np.any(singular):
                x[singular] += 1e-6

        retained = np.empty((n_retained, n_points), dtype=np.float64)
        for i in range(n_retained):
            x = np.where(x < 0.5, param_arr * x, param_arr * (1.0 - x))
            retained[i, :] = x
    else:
        raise ValueError(f"map_type desconocido: {map_type}. Use 'logistic' o 'tent'.")

    # Aplanar arrays para visualización scatter eficiente
    # param_arr repetido n_retained veces
    param_coords = np.tile(param_arr, n_retained)
    x_coords = retained.ravel()

    return param_coords, x_coords
