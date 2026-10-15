"""
Módulo de cálculo y análisis del Exponente de Lyapunov para mapas 1D.
"""

from typing import Union, Tuple, Optional
import numpy as np
from .maps import LogisticMap, TentMap


def calculate_lyapunov_exponent(
    map_instance: Union[LogisticMap, TentMap],
    x0: float = 0.5,
    n_transient: int = 500,
    n_iterations: int = 5000,
    eps: float = 1e-14,
) -> float:
    r"""
    Calcula numéricamente el exponente de Lyapunov para un mapa unidimensional:
        \lambda = \lim_{N \to \infty} \frac{1}{N} \sum_{n=0}^{N-1} \ln |f'(x_n)|

    Parámetros:
    -----------
    map_instance : LogisticMap o TentMap
        Instancia del mapa dinámico con parámetro configurado.
    x0 : float
        Condición inicial (por defecto 0.5, si es singular se perturba levemente).
    n_transient : int
        Número de iteraciones transitorias descartadas para alcanzar el atractor.
    n_iterations : int
        Número de iteraciones promediadas para calcular el exponente.
    eps : float
        Umbral inferior para evitar \ln(0).

    Retorna:
    --------
    float : Valor numérico estimado del exponente de Lyapunov \lambda.
    """
    # Evitar singularidades exactas en la condición inicial
    x = float(x0)
    if x in (0.0, 0.5, 1.0):
        x += 1e-6

    # 1. Fase transitoria (eliminar estados iniciales alejados del atractor)
    for _ in range(n_transient):
        x = map_instance.evaluate(x)
        if x in (0.0, 0.5, 1.0):
            x += 1e-6

    # 2. Acumulación del logaritmo de la derivada a lo largo de la órbita
    lyap_sum = 0.0
    for _ in range(n_iterations):
        deriv = abs(map_instance.derivative(x))
        if deriv < eps:
            deriv = eps
        lyap_sum += np.log(deriv)
        x = map_instance.evaluate(x)
        if x in (0.0, 0.5, 1.0):
            x += 1e-6

    return float(lyap_sum / n_iterations)


def lyapunov_logistic_spectrum(
    r_values: np.ndarray,
    x0: float = 0.5,
    n_transient: int = 500,
    n_iterations: int = 2000,
    eps: float = 1e-14,
) -> np.ndarray:
    r"""
    Calcula el espectro de Lyapunov vectorizado para un conjunto de parámetros r del mapa logístico:
        x_{n+1} = r x_n (1 - x_n)
    """
    r_arr = np.asarray(r_values, dtype=np.float64)
    n_points = len(r_arr)

    # Inicialización de estados
    x = np.full(n_points, x0, dtype=np.float64)
    # Pequeña perturbación si x0 == 0.5
    if x0 == 0.5:
        x += 1e-5

    # Transitorio
    for _ in range(n_transient):
        x = r_arr * x * (1.0 - x)
        # Control de singularidades
        singular = (x == 0.5) | (x <= 0.0) | (x >= 1.0)
        if np.any(singular):
            x[singular] += 1e-6

    # Acumulación de Lyapunov
    lyap_sum = np.zeros(n_points, dtype=np.float64)
    for _ in range(n_iterations):
        deriv = np.abs(r_arr * (1.0 - 2.0 * x))
        deriv = np.maximum(deriv, eps)
        lyap_sum += np.log(deriv)
        x = r_arr * x * (1.0 - x)
        singular = (x == 0.5) | (x <= 0.0) | (x >= 1.0)
        if np.any(singular):
            x[singular] += 1e-6

    return lyap_sum / n_iterations


def lyapunov_tent_spectrum(
    p_values: np.ndarray,
    x0: float = 0.35,
    n_transient: int = 300,
    n_iterations: int = 1500,
    eps: float = 1e-14,
) -> np.ndarray:
    r"""
    Calcula el exponente de Lyapunov numérico para un conjunto de parámetros p del mapa de tienda.
    Para cada p > 0, |f'(x)| = p casi en todas partes, por lo que numéricamente converge a ln(p).
    """
    p_arr = np.asarray(p_values, dtype=np.float64)
    n_points = len(p_arr)

    x = np.full(n_points, x0, dtype=np.float64)

    # Transitorio
    for _ in range(n_transient):
        x = np.where(x < 0.5, p_arr * x, p_arr * (1.0 - x))
        # Si cae exactamente en 0.5 o 0, perturbar para mantener la trayectoria dinámica
        singular = (x == 0.5) | (x == 0.0)
        if np.any(singular):
            x[singular] += 1e-6

    # Acumulación
    lyap_sum = np.zeros(n_points, dtype=np.float64)
    for _ in range(n_iterations):
        deriv = np.abs(np.where(x < 0.5, p_arr, -p_arr))
        deriv = np.maximum(deriv, eps)
        lyap_sum += np.log(deriv)
        x = np.where(x < 0.5, p_arr * x, p_arr * (1.0 - x))
        singular = (x == 0.5) | (x == 0.0)
        if np.any(singular):
            x[singular] += 1e-6

    return lyap_sum / n_iterations


def analytical_lyapunov_tent(p_values: np.ndarray) -> np.ndarray:
    r"""
    Retorna la solución analítica exacta del exponente de Lyapunov para el mapa de tienda:
        \lambda(p) = \ln(p)
    """
    p_arr = np.asarray(p_values, dtype=np.float64)
    with np.errstate(divide='ignore'):
        res = np.log(p_arr)
    return res


def analytical_lyapunov_logistic_fixed_point(r_values: np.ndarray) -> np.ndarray:
    r"""
    Retorna la solución analítica del exponente de Lyapunov para el punto fijo estable
    del mapa logístico x* = 1 - 1/r, válido en el régimen 1 < r \le 3:
        \lambda(r) = \ln |2 - r|
    """
    r_arr = np.asarray(r_values, dtype=np.float64)
    valid = (r_arr >= 1.0) & (r_arr <= 3.0)
    res = np.full_like(r_arr, np.nan, dtype=np.float64)
    diff = np.abs(2.0 - r_arr[valid])
    with np.errstate(divide='ignore'):
        res[valid] = np.log(np.maximum(diff, 1e-15))
    return res
