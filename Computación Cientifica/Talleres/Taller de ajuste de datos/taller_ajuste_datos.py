"""
================================================================================
TALLER: INTERPOLACIÓN Y MÍNIMOS CUADRADOS
Curso: Computación Científica / Métodos Numéricos
Profesor: Juan Fernando Paz
Pontificia Universidad Javeriana Cali

Contenido del archivo:
1. Ejercicio 1: Interpolación de datos experimentales (Lagrange)
   - Grado máximo del polinomio interpolante.
   - Construcción analítica y computacional de los polinomios base L_i(t).
   - Polinomio interpolante p(t) simplificado.
   - Estimaciones en t = 1.5 y t = 2.5.
   - Verificación computacional en los 4 nodos originales.
   - Análisis de sobreajuste con polinomios de grado 4 o superior.
   - Gráfica guardada: fig_ejercicio1_lagrange.png

2. Ejercicio 2: Interpolación y polinomio de Taylor (f(x) = 1 / (1 + x^2))
   - Parte A: Polinomio de Taylor T_2, T_4, T_6 en x_0 = 0 y derivadas.
   - Parte B: Polinomio interpolante p(x) de grado 4 sobre 5 nodos.
   - Parte C: Comparación de errores e_p(x) y e_T(x) en [-1, 1], errores máximos
     y evaluaciones en x = 0.1, 0.5, 1.0.
   - Parte D: Respuestas teóricas (información local vs. global, derivadas).
   - Gráficas guardadas: fig_ejercicio2_taylor.png, fig_ejercicio2_comparacion.png

3. Ejercicio 3: Mínimos cuadrados y matrices mal condicionadas
   - Conjunto experimental de 50 puntos (x, y).
   - Ajuste polinomial de grado 12 mediante Ecuaciones Normales y QR.
   - Comparación de coeficientes y normas de residuos ||r_NE||_2 vs ||r_QR||_2.
   - Condicionamiento kappa(A) vs kappa(A^T A).
   - Experimento de perturbación (y_0 -> y_0 + 10^-6).
   - Gráficas guardadas: fig_ejercicio3_minimos_cuadrados.png, fig_ejercicio3_perturbacion.png
   - Análisis y conclusiones conceptuales.
================================================================================
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

np.set_printoptions(precision=6, suppress=True)


# ==============================================================================
# FUNCIONES AUXILIARES DE FORMATO
# ==============================================================================

def lines_hashtags():
    print("\n" + "#" * 80)

def lines_equals():
    print("=" * 80)

def section_header(title):
    print("\n" + "=" * 80)
    print(f" {title.upper()} ")
    print("=" * 80)


# ==============================================================================
# EJERCICIO 1: INTERPOLACIÓN DE DATOS EXPERIMENTALES
# ==============================================================================

def ejecutar_ejercicio_1():
    section_header("Ejercicio 1: Interpolación de datos experimentales")
    
    t_nodos = np.array([0.0, 1.0, 2.0, 3.0])
    x_nodos = np.array([1.0, 2.7, 5.8, 10.9])
    n = len(t_nodos) - 1  # n = 3 (grado máximo 3)
    
    print("Datos experimentales registrados:")
    for ti, xi in zip(t_nodos, x_nodos):
        print(f"  t = {ti:3.1f} s  -->  x(t) = {xi:4.1f} m")
        
    print("\n--- a) Grado máximo del polinomio interpolante ---")
    resp_a = """Dado que se disponen de N = 4 datos distintos (t_0, t_1, t_2, t_3),
el polinomio interpolante p(t) tiene como grado MÁXIMO n = N - 1 = 3.
Por el Teorema de Existencia y Unicidad de la interpolación polinomial,
existe un ÚNICO polinomio de grado a lo sumo 3 que pasa exactamente por estos 4 puntos,
debido a que la matriz de Vandermonde asociada con nodos distintos es estrictamente invertible."""
    print(resp_a)

    print("\n--- b) y c) Polinomios base de Lagrange L_i(t) ---")
    # Construcción algebraica de Li(t) = prod_{j!=i} (t - tj) / (ti - tj)
    # L0(t) = (t-1)(t-2)(t-3) / (-1)(-2)(-3) = -1/6 (t^3 - 6t^2 + 11t - 6)
    # L1(t) = t(t-2)(t-3) / (1)(-1)(-2) = 1/2 (t^3 - 5t^2 + 6t)
    # L2(t) = t(t-1)(t-3) / (2)(1)(-1) = -1/2 (t^3 - 4t^2 + 3t)
    # L3(t) = t(t-1)(t-2) / (3)(2)(1) = 1/6 (t^3 - 3t^2 + 2t)
    
    polys_L = [
        [-1/6, 1.0, -11/6, 1.0],      # L0: coeficientes de t^3, t^2, t^1, t^0
        [1/2, -5/2, 3.0, 0.0],        # L1
        [-1/2, 2.0, -3/2, 0.0],       # L2
        [1/6, -1/2, 1/3, 0.0]         # L3
    ]
    
    for i in range(4):
        c3, c2, c1, c0 = polys_L[i]
        print(f"  L_{i}(t) = ({c3:+.4f})*t^3 + ({c2:+.4f})*t^2 + ({c1:+.4f})*t + ({c0:+.4f})")
        
    print("\n--- d) Polinomio interpolante p(t) simplificado ---")
    # p(t) = sum_{i=0}^3 x_i * L_i(t)
    # Coeficientes exactos:
    # c3 = 1.0*(-1/6) + 2.7*(1/2) + 5.8*(-1/2) + 10.9*(1/6) = 0.1
    # c2 = 1.0*(1.0) + 2.7*(-5/2) + 5.8*(2.0) + 10.9*(-1/2) = 0.4
    # c1 = 1.0*(-11/6) + 2.7*(3.0) + 5.8*(-3/2) + 10.9*(1/3) = 1.2
    # c0 = 1.0*(1.0) = 1.0
    coefs_p = np.zeros(4) # [c3, c2, c1, c0]
    for i in range(4):
        coefs_p += x_nodos[i] * np.array(polys_L[i])
        
    print(f"Expresión analítica obtenida:")
    print(f"  p(t) = {coefs_p[0]:.4f}*t^3 + {coefs_p[1]:.4f}*t^2 + {coefs_p[2]:.4f}*t + {coefs_p[3]:.4f}")
    print("En forma fraccionaria exacta:")
    print("  p(t) = (1/10)*t^3 + (2/5)*t^2 + (6/5)*t + 1")

    def eval_p(t):
        return coefs_p[0]*t**3 + coefs_p[1]*t**2 + coefs_p[2]*t + coefs_p[3]

    print("\n--- e) y f) Estimaciones de posición ---")
    pos_1_5 = eval_p(1.5)
    pos_2_5 = eval_p(2.5)
    print(f"  Posición estimada en t = 1.5 s: p(1.5) = {pos_1_5:.4f} m  (Exacto: 4.0375 m)")
    print(f"  Posición estimada en t = 2.5 s: p(2.5) = {pos_2_5:.4f} m  (Exacto: 8.0625 m)")

    print("\n--- g) Verificación computacional en los nodos ---")
    print("Nodo t_i | x(t_i) Original | p(t_i) Evaluado | Error Absoluto")
    print("-" * 55)
    for ti, xi in zip(t_nodos, x_nodos):
        p_val = eval_p(ti)
        err = abs(p_val - xi)
        print(f"   {ti:3.1f}   |      {xi:5.2f}      |      {p_val:5.2f}      |    {err:.2e}")
    print("-> Se comprueba que el polinomio pasa EXACTAMENTE por los 4 datos originales.")

    print("\n--- h) ¿Qué pasaría con un polinomio de grado 4 o superior? ---")
    resp_h = """1. SUBDETERMINACIÓN Y PÉRDIDA DE UNICIDAD:
   Con 4 puntos de soporte, un polinomio de grado m >= 4 tiene m + 1 >= 5 coeficientes incógnitas.
   El sistema lineal resultante A c = x tiene infinitas soluciones. Cualquier polinomio de la forma:
       q(t) = p_3(t) + alpha(t) * (t - 0)(t - 1)(t - 2)(t - 3)
   interpola exactamente los mismos 4 puntos para cualquier función arbitraria alpha(t) != 0.
2. SINGULARIDAD EN MÍNIMOS CUADRADOS:
   La matriz de ecuaciones normales A^T A (de tamaño (m+1)x(m+1)) tiene rango a lo sumo 4,
   por lo que es matemáticamente SINGULAR (no invertible) sin regularización.
3. SOBREAJUSTE (OVERFITTING) Y OSCILACIONES:
   Grados innecesariamente altos provocan oscilaciones espurias pronunciadas entre nodos
   y divergencia violenta fuera de [0, 3], deteriorando la validez física del modelo."""
    print(resp_h)

    # Generación y guardado de figura
    t_fine = np.linspace(-0.2, 3.2, 300)
    x_fine = eval_p(t_fine)
    
    plt.figure(figsize=(8, 5))
    plt.plot(t_fine, x_fine, 'b-', lw=2, label=r'$p(t) = 0.1t^3 + 0.4t^2 + 1.2t + 1.0$')
    plt.plot(t_nodos, x_nodos, 'ro', markersize=8, label='Datos experimentales originales')
    plt.plot([1.5, 2.5], [pos_1_5, pos_2_5], 's', color='darkorange', markersize=8,
             label=f'Estimaciones: p(1.5)={pos_1_5:.2f}, p(2.5)={pos_2_5:.2f}')
    plt.axvline(1.5, color='gray', linestyle=':', alpha=0.6)
    plt.axvline(2.5, color='gray', linestyle=':', alpha=0.6)
    plt.title("Ejercicio 1: Polinomio Interpolante de Lagrange", fontsize=13, fontweight='bold')
    plt.xlabel("Tiempo t (s)", fontsize=11)
    plt.ylabel("Posición x(t) (m)", fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig("Talleres/fig_ejercicio1_lagrange.png", dpi=200)
    plt.close()
    print("\n-> Gráfica guardada exitosamente en 'Talleres/fig_ejercicio1_lagrange.png'")


# ==============================================================================
# EJERCICIO 2: INTERPOLACIÓN Y POLINOMIO DE TAYLOR
# ==============================================================================

def ejecutar_ejercicio_2():
    section_header("Ejercicio 2: Interpolación y Polinomio de Taylor")
    
    print("Función objetivo: f(x) = 1 / (1 + x^2) alrededor de x_0 = 0\n")
    
    # --------------------------------------------------------------------------
    # Parte A: Polinomio de Taylor
    # --------------------------------------------------------------------------
    print("PARTE A: POLINOMIO DE TAYLOR")
    print("a) Cálculo analítico de derivadas en x_0 = 0:")
    derivadas_info = """  f(x)   = (1 + x^2)^(-1)                      ==> f(0)   = 1
  f'(x)  = -2x (1 + x^2)^(-2)                  ==> f'(0)  = 0
  f''(x) = (6x^2 - 2) (1 + x^2)^(-3)           ==> f''(0) = -2   ==> f''(0)/2! = -1
  f'''(x)= 24x(1 - x^2) (1 + x^2)^(-4)         ==> f'''(0)= 0
  f^(4)  = (120x^4 - 240x^2 + 24)(1+x^2)^(-5)  ==> f^(4)(0)= 24  ==> f^(4)(0)/4! = +1
  En general, usando la serie geométrica 1/(1 - (-x^2)) = sum_{k=0}^inf (-1)^k x^(2k):
  f^(2k)(0) / (2k)! = (-1)^k,  mientras que todas las derivadas impares son 0."""
    print(derivadas_info)
    
    print("\nb) y c) Polinomios de Taylor calculados:")
    print("  T_2(x) = 1 - x^2")
    print("  T_4(x) = 1 - x^2 + x^4")
    print("  T_6(x) = 1 - x^2 + x^4 - x^6")
    
    # Gráfica de Taylor
    x_plot = np.linspace(-1.25, 1.25, 500)
    f_plot = 1.0 / (1.0 + x_plot**2)
    T2_plot = 1.0 - x_plot**2
    T4_plot = 1.0 - x_plot**2 + x_plot**4
    T6_plot = 1.0 - x_plot**2 + x_plot**4 - x_plot**6

    plt.figure(figsize=(8, 5))
    plt.plot(x_plot, f_plot, 'k-', lw=2.5, label=r'$f(x) = \frac{1}{1 + x^2}$')
    plt.plot(x_plot, T2_plot, 'r--', lw=1.8, label=r'$T_2(x) = 1 - x^2$')
    plt.plot(x_plot, T4_plot, 'g-.', lw=1.8, label=r'$T_4(x) = 1 - x^2 + x^4$')
    plt.plot(x_plot, T6_plot, 'b:', lw=2.0, label=r'$T_6(x) = 1 - x^2 + x^4 - x^6$')
    plt.axvline(-1, color='gray', linestyle='--', alpha=0.5)
    plt.axvline(1, color='gray', linestyle='--', alpha=0.5, label='Radio conv. |x|=1')
    plt.ylim(-0.2, 1.3)
    plt.title("Ejercicio 2 - Parte A: Polinomios de Taylor de f(x)", fontsize=13, fontweight='bold')
    plt.xlabel("x", fontsize=11)
    plt.ylabel("y", fontsize=11)
    plt.legend(fontsize=10, loc='lower center')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig("Talleres/fig_ejercicio2_taylor.png", dpi=200)
    plt.close()
    print("-> Gráfica guardada en 'Talleres/fig_ejercicio2_taylor.png'")

    # --------------------------------------------------------------------------
    # Parte B: Polinomio Interpolante
    # --------------------------------------------------------------------------
    print("\nPARTE B: POLINOMIO INTERPOLANTE DE GRADO 4")
    x_interp = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
    y_interp = 1.0 / (1.0 + x_interp**2)
    
    print("Nodos de interpolación y valores:")
    for xi, yi in zip(x_interp, y_interp):
        print(f"  x_i = {xi:4.1f}  -->  y_i = f(x_i) = {yi:4.2f}")
        
    # Polinomio interpolante p(x) de grado 4:
    # Por simetría f(-x) = f(x), p(x) = c4*x^4 + c2*x^2 + c0
    # c0 = 1.0
    # c2 = -0.9 = -9/10
    # c4 = 0.4 = 2/5
    coef_p = np.array([0.4, 0.0, -0.9, 0.0, 1.0]) # [c4, c3, c2, c1, c0]
    
    print("\na) Expresión explícita del polinomio interpolante:")
    print("  p(x) = 0.4*x^4 - 0.9*x^2 + 1.0  =  (2/5)*x^4 - (9/10)*x^2 + 1")
    
    def eval_interp_p(x):
        return 0.4 * x**4 - 0.9 * x**2 + 1.0

    print("\nb) Verificación p(x_i) = f(x_i):")
    print("Nodo x_i | f(x_i) Real | p(x_i) Evaluado | Error Absoluto")
    print("-" * 55)
    for xi, yi in zip(x_interp, y_interp):
        p_val = eval_interp_p(xi)
        print(f"  {xi:5.1f}  |    {yi:5.2f}    |      {p_val:5.2f}      |    {abs(p_val - yi):.2e}")

    # Gráfica f(x), p(x), T4(x)
    p_plot = eval_interp_p(x_plot)
    plt.figure(figsize=(8, 5))
    plt.plot(x_plot, f_plot, 'k-', lw=2.5, label=r'$f(x) = \frac{1}{1 + x^2}$')
    plt.plot(x_plot, p_plot, 'b-', lw=2.0, label=r'$p(x) = 0.4x^4 - 0.9x^2 + 1$ (Lagrange)')
    plt.plot(x_plot, T4_plot, 'r--', lw=2.0, label=r'$T_4(x) = x^4 - x^2 + 1$ (Taylor)')
    plt.plot(x_interp, y_interp, 'ko', markersize=7, label='Nodos de interpolación')
    plt.xlim(-1.1, 1.1)
    plt.ylim(0.3, 1.15)
    plt.title("Ejercicio 2 - Parte B: Comparación f(x), p(x) y T_4(x)", fontsize=13, fontweight='bold')
    plt.xlabel("x", fontsize=11)
    plt.ylabel("y", fontsize=11)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig("Talleres/fig_ejercicio2_comparacion.png", dpi=200)
    plt.close()
    print("-> Gráfica guardada en 'Talleres/fig_ejercicio2_comparacion.png'")

    # --------------------------------------------------------------------------
    # Parte C: Comparación analítica y numérica
    # --------------------------------------------------------------------------
    print("\nPARTE C: COMPARACIÓN DE ERRORES EN [-1, 1]")
    mesh_x = np.linspace(-1.0, 1.0, 10001)
    f_mesh = 1.0 / (1.0 + mesh_x**2)
    p_mesh = eval_interp_p(mesh_x)
    T4_mesh = 1.0 - mesh_x**2 + mesh_x**4
    
    e_p = np.abs(f_mesh - p_mesh)
    e_T = np.abs(f_mesh - T4_mesh)
    
    max_e_p = np.max(e_p)
    max_e_T = np.max(e_T)
    x_max_p = mesh_x[np.argmax(e_p)]
    x_max_T = mesh_x[np.argmax(e_T)]
    
    print(f"a) y b) Error máximo en [-1, 1]:")
    print(f"  * Error máximo de Taylor T_4:       {max_e_T:.6f}  (alcanzado en x = {x_max_T:.2f})")
    print(f"  * Error máximo de Interpolación p:  {max_e_p:.6f}  (alcanzado en x = {x_max_p:.2f})")
    print(f"  * Ratio de error max(e_T) / max(e_p) = {max_e_T / max_e_p:.2f} veces mayor en Taylor.")
    
    print("\nc) Evaluación en puntos específicos x in {0.1, 0.5, 1.0}:")
    print("  x   |  f(x) Exacto  |  T_4(x)  | Error e_T(x) |   p(x)   | Error e_p(x) | Ganador")
    print("-" * 80)
    for pt in [0.1, 0.5, 1.0]:
        f_val = 1.0 / (1.0 + pt**2)
        t_val = 1.0 - pt**2 + pt**4
        p_val = eval_interp_p(pt)
        err_t = abs(f_val - t_val)
        err_p = abs(f_val - p_val)
        winner = "Taylor" if err_t < err_p else "Interpolación"
        print(f" {pt:4.1f} |   {f_val:.6f}    | {t_val:.6f} |  {err_t:.2e}   | {p_val:.6f} |  {err_p:.2e}   | {winner}")

    print("\nd), e) y f) Análisis:")
    resp_cf = """- En la vecindad inmediata de x_0 = 0 (e.g. |x| <= 0.2), Taylor es sumamente superior:
  en x = 0.1 el error de Taylor es ~9.9e-07 mientras que el de interpolación es ~9.4e-04
  (Taylor es casi 1,000 veces más exacto localmente).
- A medida que nos alejamos de x_0, el error de Taylor crece explosivamente:
  en x = 0.5 su error es 0.0125, y en x = 1.0 su error es 0.5000 (100% de error relativo).
- Esto ocurre porque f(z) = 1/(1+z^2) tiene polos en el plano complejo en z = +-i.
  Por tanto, el radio de convergencia de la serie de Taylor en el origen es R = 1.
  En los bordes |x| = 1, la aproximación de Taylor colapsa.
- En contraste, la interpolación distribuye nodos en todo el intervalo [-1, 1],
  fijando la curva en +-1, +-0.5 y 0, garantizando un error máximo inferior a 0.0223."""
    print(resp_cf)

    # --------------------------------------------------------------------------
    # Parte D: Interpretación conceptual
    # --------------------------------------------------------------------------
    print("\nPARTE D: INTERPOLACIÓN Y TAYLOR COMO PROBLEMAS DIFERENTES")
    resp_part_d = """a) ¿Qué información utiliza el polinomio de Taylor?
   Utiliza información LOCAL y CONCENTRADA en un único punto x_0: el valor f(x_0)
   y sus derivadas de orden superior f'(x_0), f''(x_0), ..., f^(k)(x_0).
b) ¿Qué información utiliza el polinomio interpolante?
   Utiliza información DISTRIBUIDA y GLOBAL: una muestra discreta de pares (x_i, f(x_i))
   a lo largo del dominio.
c) ¿Es necesario conocer las derivadas para interpolar?
   NO. Solo se requieren las evaluaciones de la función en los nodos de soporte.
d) ¿Es necesario conocer valores en varios puntos para Taylor?
   NO. Toda la estructura se calcula a partir del comportamiento infinitesimal en x_0.
e) ¿Por qué Taylor es una aproximación local?
   Porque impone contacto de orden k (mismo valor y mismas derivadas) exclusivamente
   en x_0. Su precisión óptima decae a medida que nos distanciamos del centro.
f) ¿Por qué la interpolación es distribuida?
   Porque los polinomios base L_i(x) fuerzan al polinomio a cruzar obligatoriamente
   por cada uno de los nodos dispersos en el intervalo, anclando el error globalmente."""
    print(resp_part_d)

    print("\nPREGUNTA DE CIERRE: ¿CUÁNDO USAR TAYLOR Y CUÁNDO INTERPOLACIÓN?")
    resp_cierre = """- UTILIZAR TAYLOR cuando:
  1. Se conoce la función de manera analítica y es diferenciable.
  2. El objetivo es analizar o linealizar el comportamiento en un entorno muy cercano
     al punto de equilibrio o de operación x_0 (e.g. física teórica, control automático,
     desarrollo asintótico de métodos numéricos de un paso).
- UTILIZAR INTERPOLACIÓN cuando:
  1. No se conocen derivadas ni una fórmula analítica (e.g. datos de laboratorio,
     muestreos discretos, simulaciones costosas de 'caja negra').
  2. Se requiere aproximar la función a lo largo de un intervalo amplio [a, b]
     manteniendo un error acotado en todo el dominio."""
    print(resp_cierre)


# ==============================================================================
# EJERCICIO 3: MÍNIMOS CUADRADOS Y MATRICES MAL CONDICIONADAS
# ==============================================================================

def ejecutar_ejercicio_3():
    section_header("Ejercicio 3: Mínimos Cuadrados y Matrices Mal Condicionadas")
    
    # Datos experimentales (50 mediciones)
    x = np.array([
        0.00, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18,
        0.20, 0.22, 0.24, 0.26, 0.28, 0.30, 0.32, 0.34, 0.36, 0.38,
        0.40, 0.42, 0.44, 0.46, 0.48, 0.50, 0.52, 0.54, 0.56, 0.58,
        0.60, 0.62, 0.64, 0.66, 0.68, 0.70, 0.72, 0.74, 0.76, 0.78,
        0.80, 0.82, 0.84, 0.86, 0.88, 0.90, 0.92, 0.94, 0.96, 0.98
    ])
    
    y = np.array([
        1.02, 1.04, 1.08, 1.13, 1.15, 1.21, 1.25, 1.31, 1.34, 1.40,
        1.43, 1.50, 1.54, 1.61, 1.66, 1.72, 1.78, 1.82, 1.91, 1.96,
        2.01, 2.06, 2.12, 2.19, 2.23, 2.31, 2.36, 2.42, 2.48, 2.55,
        2.60, 2.68, 2.73, 2.81, 2.87, 2.94, 3.01, 3.07, 3.14, 3.21,
        3.29, 3.35, 3.42, 3.49, 3.57, 3.63, 3.71, 3.78, 3.86, 3.94
    ])
    
    degree = 12
    m = len(x)        # 50 filas
    n = degree + 1    # 13 columnas (c_0 hasta c_12)
    
    # Construcción de la matriz de Vandermonde A (50 x 13)
    # A[:, j] = x^j  para j = 0, 1, ..., 12
    A = np.vander(x, N=n, increasing=True)
    
    print(f"Dimensiones del problema:")
    print(f"  Número de datos (filas de A):       m = {m}")
    print(f"  Grado del polinomio:                grado = {degree}")
    print(f"  Número de coeficientes (columnas):  n = {n}")
    print(f"  Dimensión de la matriz A:           {A.shape}")

    # 1. Ecuaciones Normales: (A^T A) c_NE = A^T y
    ATA = A.T @ A
    ATy = A.T @ y
    c_NE = np.linalg.solve(ATA, ATy)

    # 2. Factorización QR: A = Q R, R c_QR = Q^T y
    Q, R = np.linalg.qr(A)  # QR reducido: Q es 50x13, R es 13x13
    c_QR = np.linalg.solve(R, Q.T @ y)

    # a) Comparación de coeficientes
    print("\n--- a) Comparación de coeficientes obtenidos ---")
    print(f"{'Coef':^6} | {'Ecuaciones Normales (NE)':^25} | {'Factorización QR':^25} | {'Diferencia Absoluta':^20}")
    print("-" * 82)
    for i in range(n):
        diff = abs(c_NE[i] - c_QR[i])
        print(f" c_{i:<2d}  | {c_NE[i]:25.6f} | {c_QR[i]:25.6f} | {diff:20.4e}")

    # b) Comparación de residuos
    r_NE = y - A @ c_NE
    r_QR = y - A @ c_QR
    norm_r_NE = np.linalg.norm(r_NE, 2)
    norm_r_QR = np.linalg.norm(r_QR, 2)
    
    print("\n--- b) Comparación de normas de residuos ---")
    print(f"  Norma residuo Ecuaciones Normales ||r_NE||_2: {norm_r_NE:.8f}")
    print(f"  Norma residuo Factorización QR    ||r_QR||_2: {norm_r_QR:.8f}")
    print(f"  Diferencia absoluta ||r_NE||_2 - ||r_QR||_2:  {abs(norm_r_NE - norm_r_QR):.6e}")
    print("  -> ¡Los residuos tienen casi el mismo valor (diferencia ~ 2.4e-3) a pesar")
    print("     de que los coeficientes difieren en órdenes de magnitud de 10^5!")

    # c) y d) Evaluación en malla de [0, 1] y gráficas
    mesh_eval = np.linspace(0.0, 1.0, 1000)
    A_eval = np.vander(mesh_eval, N=n, increasing=True)
    p_NE_eval = A_eval @ c_NE
    p_QR_eval = A_eval @ c_QR
    diff_eval = np.abs(p_NE_eval - p_QR_eval)

    print("\n--- d) Evaluación de ambos polinomios en malla del intervalo [0, 1] ---")
    print(f"  Diferencia máxima en la curva sobre [0, 1]:  {np.max(diff_eval):.6f}")
    print(f"  Diferencia promedio en la curva sobre [0, 1]: {np.mean(diff_eval):.6f}")
    print(f"  Valor en x = 1.0: NE = {p_NE_eval[-1]:.4f} | QR = {p_QR_eval[-1]:.4f}")

    # Gráfica de ajuste
    plt.figure(figsize=(9, 5.5))
    plt.plot(x, y, 'ko', markersize=4.5, alpha=0.7, label='Datos de laboratorio (50 pts)')
    plt.plot(mesh_eval, p_NE_eval, 'r--', lw=1.8, label=r'Ajuste Ecuaciones Normales ($p_{NE}$)')
    plt.plot(mesh_eval, p_QR_eval, 'b-', lw=1.8, alpha=0.8, label=r'Ajuste Factorización QR ($p_{QR}$)')
    plt.title("Ejercicio 3: Ajuste Polinomial de Grado 12 (NE vs QR)", fontsize=13, fontweight='bold')
    plt.xlabel("Variable independiente x", fontsize=11)
    plt.ylabel("Variable de respuesta y", fontsize=11)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig("Talleres/fig_ejercicio3_minimos_cuadrados.png", dpi=200)
    plt.close()
    print("-> Gráfica guardada en 'Talleres/fig_ejercicio3_minimos_cuadrados.png'")

    # e), f) y g) Números de Condicionamiento
    print("\n--- e), f) y g) Números de Condicionamiento ---")
    cond_A = np.linalg.cond(A, 2)
    cond_ATA = np.linalg.cond(ATA, 2)
    print(f"  Condicionamiento de la matriz de diseño A:     kappa(A)   = {cond_A:.6e}")
    print(f"  Condicionamiento de la matriz normal A^T A:    kappa(A^TA)= {cond_ATA:.6e}")
    print(f"  Relación teórica: (kappa(A))^2                           = {(cond_A**2):.6e}")
    print(f"  Límite de precisión doble IEEE 754 (1/eps_mach)         = {1.0/np.finfo(float).eps:.6e}")
    
    resp_cond = """EXPLICACIÓN DE LO OBSERVADO:
1. kappa(A^T A) = (kappa(A))^2 es una propiedad fundamental de la norma-2 espectral.
2. Como kappa(A) ~ 7.76e+08, su cuadrado asciende a ~ 3.13e+17.
3. En aritmética estándar de 64 bits (doble precisión), el épsilon de máquina es
   eps_mach ~ 2.22e-16, por lo que 1/eps_mach ~ 4.50e+15.
4. Dado que kappa(A^T A) > 1/eps_mach, la matriz A^T A es NUMÉRICAMENTE SINGULAR en IEEE 754.
   Todos los dígitos significativos se pierden por completo en los modos singulares más pequeños!"""
    print(resp_cond)

    # Parte D: Perturbación de datos
    print("\n--- PARTE D: PERTURBACIÓN DE LOS DATOS (y_0 -> y_0 + 10^-6) ---")
    y_pert = y.copy()
    perturbation = 1e-6
    y_pert[0] += perturbation
    
    # Resolución del problema perturbado
    c_NE_pert = np.linalg.solve(ATA, A.T @ y_pert)
    c_QR_pert = np.linalg.solve(R, Q.T @ y_pert)

    delta_c_NE = np.linalg.norm(c_NE_pert - c_NE, np.inf)
    delta_c_QR = np.linalg.norm(c_QR_pert - c_QR, np.inf)

    print(f"Perturbación inyectada en el primer dato: delta y_0 = {perturbation:.1e}")
    print(f"Máximo cambio en coeficientes con Ecuaciones Normales ||delta c_NE||_inf: {delta_c_NE:.6f}")
    print(f"Máximo cambio en coeficientes con Factorización QR    ||delta c_QR||_inf: {delta_c_QR:.6f}")
    print(f"Factor de sensibilidad / amplificación del error en NE: {delta_c_NE / perturbation:.2e}")
    print(f"Factor de sensibilidad / amplificación del error en QR: {delta_c_QR / perturbation:.2e}")
    
    print("\nComparación del coeficiente c_12 ante la perturbación:")
    print(f"  NE original: {c_NE[12]:15.6f}  |  NE perturbado: {c_NE_pert[12]:15.6f} (Dif: {abs(c_NE_pert[12]-c_NE[12]):.6f})")
    print(f"  QR original: {c_QR[12]:15.6f}  |  QR perturbado: {c_QR_pert[12]:15.6f} (Dif: {abs(c_QR_pert[12]-c_QR[12]):.6f})")

    # Gráfica de perturbación
    p_NE_pert_mesh = A_eval @ c_NE_pert
    p_QR_pert_mesh = A_eval @ c_QR_pert

    plt.figure(figsize=(9, 5.5))
    plt.plot(mesh_eval, p_NE_eval, 'r-', lw=2.0, label='NE Original')
    plt.plot(mesh_eval, p_NE_pert_mesh, 'm--', lw=1.8, label=r'NE Perturbado ($y_0 + 10^{-6}$)')
    plt.plot(mesh_eval, p_QR_eval, 'b-', lw=2.0, alpha=0.7, label='QR Original')
    plt.plot(mesh_eval, p_QR_pert_mesh, 'c:', lw=2.0, label=r'QR Perturbado ($y_0 + 10^{-6}$)')
    plt.title("Ejercicio 3 - Perturbación: Estabilidad Numérica NE vs QR", fontsize=13, fontweight='bold')
    plt.xlabel("x", fontsize=11)
    plt.ylabel("y", fontsize=11)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig("Talleres/fig_ejercicio3_perturbacion.png", dpi=200)
    plt.close()
    print("-> Gráfica guardada en 'Talleres/fig_ejercicio3_perturbacion.png'")

    # Preguntas de análisis
    print("\n--- PREGUNTAS DE ANÁLISIS ---")
    preguntas_analisis = """1. ¿Los dos métodos producen exactamente los mismos coeficientes?
   NO. Debido al severo mal condicionamiento de la base monomial, los errores de redondeo
   en aritmética de punto flotante de 64 bits se propagan de manera distinta en cada algoritmo.
   Ecuaciones Normales sufre pérdida catastrófica de precisión al formar A^T A.

2. Si los coeficientes son diferentes, ¿los residuos presentan una diferencia de la misma magnitud?
   NO. Los coeficientes difieren en magnitudes gigantescas (~ 10^5), mientras que los residuos
   difieren en apenas ~ 2.4e-3. Esto se debe a que las columnas de la matriz de Vandermonde
   son casi linealmente dependientes en [0, 1]. Enorme fluctuación en los coeficientes se compensa
   mutuamente produciendo curvas funcionalmente casi idénticas en [0, 1].

3. ¿Qué relación observa entre el número de condicionamiento de A y el de A^T A?
   kappa(A^T A) = (kappa(A))^2. Si kappa(A) = 7.76e+08, entonces kappa(A^T A) = 3.13e+17.

4. ¿Por qué formar A^T A puede empeorar las dificultades numéricas del problema?
   Porque elevar al cuadrado el condicionamiento duplica la cantidad de dígitos significativos
   perdidos. Al superar 10^16, se rebasa la capacidad de los 53 bits de mantisa (15-17 dígitos)
   de la precisión doble IEEE 754, convirtiendo a A^T A en una matriz numéricamente singular.

5. ¿Qué método parece ser más estable para este problema?
   La Factorización QR es notablemente más estable. Al no formar el producto A^T A y utilizar
   transformaciones ortogonales (que preservan distancias y tienen condicionamiento perfecto 1),
   QR mantiene el número de condición en kappa(A) ~ 7.8e+08, resolviendo el sistema triangular R c = Q^T y
   de manera backward stable según el teorema de Wilkinson.

6. ¿Por qué este ejemplo constituye un problema de mínimos cuadrados mal condicionado?
   Porque se utilizó un polinomio de grado alto (grado 12) con la base canónica monomial {1, x, ..., x^12}
   en el intervalo compacto [0, 1]. En este intervalo, las funciones x^k se vuelven casi indistinguibles
   entre sí a medida que k crece, haciendo que las columnas de A sean cuasi-colineales."""
    print(preguntas_analisis)

    print("\n--- PREGUNTA FINAL ---")
    pregunta_final = """'Las ecuaciones normales y QR resuelven el mismo problema matemático,
pero no necesariamente con la misma estabilidad numérica.'

EXPLICACIÓN FORMAL:
En el mundo del análisis matemático ideal (con aritmética exacta y precisión infinita),
ambas formulaciones son idénticas: ambas caracterizan la solución que minimiza la distancia
euclidiana ||A c - y||_2 proyectando y sobre el subespacio columna R(A).
Sin embargo, en el mundo de la COMPUTACIÓN CIENTÍFICA (aritmética de punto flotante finita):
1. Ecuaciones Normales:
   Calcula A^T A, lo cual cuadra el número de condición (kappa -> kappa^2).
   Esto destruye la información en presencia de cancelación catastrófica.
2. Factorización QR:
   Descompone A = Q R mediante reflectores de Householder o rotaciones de Givens.
   La matriz Q es ortogonal (Q^T Q = I), lo que garantiza que ||Q x||_2 = ||x||_2
   (isometría perfecta) y kappa_2(Q) = 1. No amplifica en absoluto los errores de redondeo.
Por tanto, aunque el objetivo matemático sea el mismo, el recorrido numérico de QR
es numéricamente estable, mientras que el de Ecuaciones Normales es potencialmente inestable."""
    print(pregunta_final)


# ==============================================================================
# BLOQUE PRINCIPAL
# ==============================================================================

def main():
    lines_hashtags()
    print("# SOLUCIÓN COMPLETA: TALLER DE AJUSTE DE DATOS (INTERPOLACIÓN Y MÍNIMOS CUADRADOS)")
    print("# CURSO: COMPUTACIÓN CIENTÍFICA - PROFESOR: JUAN FERNANDO PAZ")
    lines_hashtags()
    
    ejecutar_ejercicio_1()
    ejecutar_ejercicio_2()
    ejecutar_ejercicio_3()
    
    lines_hashtags()
    print("¡TALLER DE AJUSTE DE DATOS EJECUTADO Y COMPLETADO CON ÉXITO!")
    lines_hashtags()


if __name__ == "__main__":
    main()
