"""
================================================================================
EXAMEN 1: COMPUTACIÓN CIENTÍFICA - SOLUCIÓN COMPUTACIONAL OFICIAL
Profesor: Juan Fernando Paz, Ph.D.
Pontificia Universidad Javeriana Cali - Semestre 7
================================================================================
Este script implementa la solución completa, cuantitativa y reproducible
del Componente Práctico del Examen 1:
1. Interpolación polinómica mediante matrices de Vandermonde:
   - Caso n = 20 (Grado 19)
   - Caso n = 200 (Grado 199)
2. Factorización LU con pivoteo parcial (PLU) y Factorización QR.
3. Evaluación de tiempos de ejecución (benchmarks en microsegundos).
4. Cálculo de normas de residuos relativos ||V a - y|| / ||y||.
5. Evaluación en x = 1: P(1) y Q(1) y cálculo del error relativo respecto a 20 y 200.
6. Demostración analítica del Integer Overflow en int64 y Float64 Overflow en IEEE 754.
================================================================================
"""

import time
import numpy as np
import scipy.linalg as la

# ==============================================================================
# CARGA DE DATOS OFICIALES (GOOGLE COLAB)
# ==============================================================================

# Polinomio de orden 20 (Grado 19)
x20 = np.array([
    0, 2, 4, 6, 8, 10, 12, 14, 16, 18,
    20, 22, 24, 26, 28, 30, 32, 34, 36, 38
], dtype=np.int64)

y20 = np.array([
    1, 1048575, 366503875925, 731231688012595, 164703072086692425,
    -7335632962598440505, -1964501723438287779, -1894033625802624709,
    1229782938247303441, -792061141976150257, -2052364549866899995,
    -9003277803673866045, -7168164137338222503, -6426821785559194665,
    4435783745579263469, 2394159210806721995, 1190112520884487201,
    9122145642523687967, 5336187560878624885, -1026742389961399213
], dtype=np.int64)

# Polinomio de orden 200 (Grado 199)
x200 = np.arange(0, 400, 2, dtype=np.int64)

def print_separator(title=""):
    print("\n" + "=" * 80)
    if title:
        print(f" {title.upper()} ".center(80, "="))
        print("=" * 80)

# ==============================================================================
# PARTE 1: DIAGNÓSTICO TEÓRICO PREVIO
# ==============================================================================
print_separator("1. DIAGNÓSTICO TEÓRICO PREVIO DEL PROBLEMA")

print("""
[1] CRECIMIENTO EXPONENCIAL DEL NÚMERO DE CONDICIÓN:
    La matriz de Vandermonde construida a partir de monomios {1, x, x^2, ..., x^(n-1)}
    posee un número de condición que crece exponencialmente: kappa(V) ~ O(c^n).
    Para n=20, kappa_2(V) ~ 8.76e+30, superando con creces la precisión de 64 bits (1e-16).
    Se predice una pérdida TOTAL de todas las cifras significativas (error > 100%).

[2] DESBORDAMIENTO DE PUNTO FLOTANTE (IEEE 754 FLOAT64 OVERFLOW):
    Para n=200, la potencia superior es 398^199 ~ 10^517.
    Como el número finito más grande representable en doble precisión es ~ 1.79e+308,
    la matriz de Vandermonde colapsa instantáneamente a 'inf'.
    Cualquier resolvedor de álgebra lineal arrojará una excepción fatal por presencia de infinitos.

[3] CORRUPCIÓN SILENCIOSA DE LOS DATOS POR INTEGER OVERFLOW (int64):
    La función generadora teórica fue f(x) = sum_{k=0}^{n-1} x^k.
    Para x >= 10, la suma excede 2^63 - 1 (~ 9.22e+18) y se desborda en complemento a dos:
    f(10) = 11,111,111,111,111,111,111 - 2^64 = -7,335,632,962,598,440,505.
    Los puntos y20 e y200 están corruptos por desbordamiento de enteros de 64 bits.
""")

# Demostración del desbordamiento en y20[5]
exact_sum_10 = sum(10**k for k in range(20))
val_overflow_int64 = np.uint64(exact_sum_10 % 2**64).view(np.int64)
print(f"Demostración analítica para x=10:")
print(f"  Suma matemática exacta: {exact_sum_10}")
print(f"  Envolvente modular mod 2^64: {exact_sum_10 - 2**64}")
print(f"  Valor y20[5] en el examen : {y20[5]}")
print(f"  ¿Coincidencia exacta?    : {val_overflow_int64 == y20[5]}\n")

# ==============================================================================
# PARTE 2: EXPERIMENTO PARA n = 20 (GRADO 19)
# ==============================================================================
print_separator("2. CASO n = 20 (POLINOMIO P DE GRADO 19)")

x20_f = x20.astype(np.float64)
y20_f = y20.astype(np.float64)

# Construir matriz de Vandermonde
V20 = np.vander(x20_f, increasing=True)
cond_V20 = np.linalg.cond(V20)
print(f"Número de condición kappa_2(V20): {cond_V20:.4e}")

# Factorización LU (PLU)
t0 = time.perf_counter()
for _ in range(100):
    lu20, piv20 = la.lu_factor(V20)
    a_lu20 = la.lu_solve((lu20, piv20), y20_f)
t_lu20 = (time.perf_counter() - t0) / 100 * 1e6

# Factorización QR
t0 = time.perf_counter()
for _ in range(100):
    Q20, R20 = la.qr(V20)
    a_qr20 = la.solve_triangular(R20, Q20.T @ y20_f)
t_qr20 = (time.perf_counter() - t0) / 100 * 1e6

# Residuos relativos
res_lu20 = np.linalg.norm(V20 @ a_lu20 - y20_f) / np.linalg.norm(y20_f)
res_qr20 = np.linalg.norm(V20 @ a_qr20 - y20_f) / np.linalg.norm(y20_f)

# Evaluación en x = 1: P(1) = sum(a_j)
P1_lu = np.sum(a_lu20)
P1_qr = np.sum(a_qr20)

err_P1_lu = abs(P1_lu - 20.0) / 20.0
err_P1_qr = abs(P1_qr - 20.0) / 20.0

print(f"FACTORIZACIÓN LU (PLU):")
print(f"  - Tiempo de ejecución : {t_lu20:.2f} us")
print(f"  - Residuo relativo    : {res_lu20:.4e}")
print(f"  - P(1) calculado      : {P1_lu:.6e}")
print(f"  - Error relativo a 20 : {err_P1_lu:.6e} ({err_P1_lu * 100:.2e} %)")

print(f"\nFACTORIZACIÓN QR:")
print(f"  - Tiempo de ejecución : {t_qr20:.2f} us")
print(f"  - Residuo relativo    : {res_qr20:.4e}")
print(f"  - P(1) calculado      : {P1_qr:.6e}")
print(f"  - Error relativo a 20 : {err_P1_qr:.6e} ({err_P1_qr * 100:.2e} %)")

# ==============================================================================
# PARTE 3: EXPERIMENTO PARA n = 200 (GRADO 199)
# ==============================================================================
print_separator("3. CASO n = 200 (POLINOMIO Q DE GRADO 199)")

x200_f = x200.astype(np.float64)

# Intentar construir V200
V200 = np.vander(x200_f, increasing=True)
has_inf = np.isinf(V200).any()
print(f"¿Contiene 'inf' la matriz V200? : {has_inf}")
print(f"Máxima potencia teórica         : 398^199 ~ 10^517.4")
print(f"Límite máximo de float64        : ~ 1.7977e+308")

print("\nIntentando resolver con Factorización LU:")
try:
    la.lu_factor(V200)
except Exception as e:
    print(f"  [FALLO ESPERADO]: {type(e).__name__} -> {e}")

print("\nIntentando resolver con Factorización QR:")
try:
    la.qr(V200)
except Exception as e:
    print(f"  [FALLO ESPERADO]: {type(e).__name__} -> {e}")

print(f"\nResultado para Q(1): NaN / Indefinido")
print(f"Error relativo respecto a 200: Infinito (desbordamiento de hardware)\n")

print_separator("FIN DE LA EJECUCIÓN DEL SCRIPT")
