"""
Módulo para generar diagramas de telaraña (cobweb plots)
para sistemas dinámicos discretos 1D.
"""

from typing import Tuple, Union
import numpy as np
from .maps import LogisticMap, TentMap


def generate_cobweb_data(
    map_instance: Union[LogisticMap, TentMap],
    x0: float = 0.2,
    n_steps: int = 50,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    r"""
    Genera los puntos necesarios para trazar el diagrama de telaraña:
    1. Curva del mapa: y = f(x)
    2. Línea identidad: y = x
    3. Trayectoria de escalones de telaraña: (x0, 0) -> (x0, x1) -> (x1, x1) -> (x1, x2) ...

    Retorna:
    --------
    x_curve : np.ndarray
        Dominio continuo [0, 1].
    y_curve : np.ndarray
        Imagen f(x).
    cobweb_x : np.ndarray
        Coordenadas x del trazo escalonado.
    cobweb_y : np.ndarray
        Coordenadas y del trazo escalonado.
    """
    x_curve = np.linspace(0.0, 1.0, 600)
    y_curve = map_instance.evaluate(x_curve)

    cobweb_x = [x0]
    cobweb_y = [0.0]

    current_x = x0
    for _ in range(n_steps):
        next_x = float(map_instance.evaluate(current_x))
        # Paso vertical a f(x)
        cobweb_x.extend([current_x, current_x])
        cobweb_y.extend([current_x, next_x])
        # Paso horizontal a la identidad y = x
        cobweb_x.extend([current_x, next_x])
        cobweb_y.extend([next_x, next_x])
        current_x = next_x

    return x_curve, y_curve, np.array(cobweb_x), np.array(cobweb_y)
