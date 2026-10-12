"""
================================================================================
TALLER 2: MODELO ECONÓMICO MULTISECTORIAL MEDIANTE FACTORIZACIÓN PLU
Curso: Computación Científica / Métodos Numéricos

Contenido:
1. Actividad 1: Justificación de la necesidad del pivoteo parcial.
2. Actividad 2: Implementación de Factorización PLU (PA = LU) mediante
   eliminación de Gauss con pivoteo parcial.
3. Actividad 3: Verificación numérica de la factorización (||PA - LU||_2).
4. Actividades 4 y 5: Transformación del sistema (Ly = Pb, Ux = y) y cálculo de
   los vectores de producción x^(1), x^(2), x^(3) para los tres escenarios.
5. Actividad 6: Verificación mediante el vector residuo r^(k) = Ax^(k) - b^(k)
   y su norma euclidiana ||r^(k)||_2.
6. Sección 3: Análisis económico multisectorial (preguntas 1 a 6).
7. Sección 4: Análisis de tiempo de ejecución (Matriz Inversa vs. PLU con SciPy).
================================================================================
"""

import time
import numpy as np
from scipy.linalg import lu, solve_triangular

np.set_printoptions(precision=4, suppress=True)


# ==============================================================================
# FUNCIONES AUXILIARES DE FORMATO
# ==============================================================================

def linesPrintsHashtags():
    print("#" * 80)

def linesPrintsEquals():
    print("=" * 80)


# ==============================================================================
# MATRIZ DE REQUERIMIENTOS INTERSECTORIALES A (30x30)
# ==============================================================================

MATRIZ_A = np.array([
    [0.,4.,8.,5.,7.,3.,7.,8.,5.,4.,8.,8.,3.,6.,5.,2.,8.,6.,2.,5.,1.,6.,9.,1.,3.,7.,4.,9.,3.,5.],
    [3.,7.,5.,9.,7.,2.,4.,9.,2.,9.,5.,2.,4.,7.,8.,3.,1.,4.,2.,8.,4.,2.,6.,6.,4.,6.,2.,2.,4.,8.],
    [7.,9.,8.,5.,2.,5.,8.,9.,9.,1.,9.,7.,9.,8.,1.,8.,8.,3.,1.,8.,3.,3.,1.,5.,7.,9.,7.,9.,8.,2.],
    [1.,7.,7.,8.,5.,3.,8.,6.,3.,1.,3.,5.,3.,1.,5.,7.,7.,9.,3.,7.,1.,4.,4.,5.,7.,7.,4.,7.,3.,6.],
    [2.,9.,5.,6.,4.,7.,9.,7.,1.,1.,9.,9.,4.,9.,3.,7.,6.,8.,9.,5.,1.,3.,8.,6.,8.,9.,4.,1.,1.,4.],
    [7.,2.,3.,1.,5.,0.,8.,1.,1.,2.,2.,6.,7.,5.,1.,1.,3.,2.,5.,6.,7.,4.,7.,8.,1.,6.,8.,5.,4.,2.],
    [6.,6.,1.,9.,6.,3.,4.,4.,3.,3.,3.,4.,7.,4.,9.,1.,8.,7.,2.,8.,1.,9.,9.,2.,7.,3.,7.,9.,4.,1.],
    [2.,1.,5.,5.,7.,9.,9.,3.,3.,3.,4.,8.,6.,8.,1.,8.,4.,1.,8.,4.,6.,8.,4.,3.,9.,3.,9.,2.,2.,2.],
    [6.,3.,9.,4.,1.,4.,1.,5.,4.,8.,8.,7.,3.,1.,1.,3.,6.,7.,6.,6.,6.,3.,6.,8.,2.,5.,1.,1.,5.,3.],
    [4.,3.,1.,1.,5.,6.,3.,9.,5.,8.,1.,5.,3.,1.,4.,5.,7.,1.,3.,2.,9.,6.,3.,8.,8.,2.,6.,7.,2.,2.],
    [1.,8.,1.,9.,6.,7.,7.,3.,2.,9.,0.,7.,9.,4.,4.,1.,8.,3.,7.,2.,2.,7.,6.,3.,9.,6.,6.,1.,4.,6.],
    [6.,5.,1.,8.,5.,5.,7.,4.,6.,4.,3.,7.,8.,4.,2.,3.,1.,8.,3.,7.,5.,5.,7.,9.,5.,1.,1.,2.,6.,9.],
    [8.,5.,1.,7.,5.,6.,7.,3.,3.,5.,6.,9.,5.,1.,4.,5.,5.,7.,4.,1.,5.,7.,6.,5.,4.,2.,4.,3.,1.,8.],
    [5.,4.,8.,7.,2.,1.,4.,8.,2.,3.,1.,1.,3.,5.,3.,1.,1.,8.,2.,3.,2.,3.,7.,1.,8.,2.,3.,9.,7.,4.],
    [5.,2.,8.,4.,9.,5.,9.,4.,5.,9.,8.,3.,1.,3.,4.,2.,1.,7.,8.,7.,5.,1.,7.,7.,9.,3.,9.,1.,1.,4.],
    [9.,6.,3.,1.,4.,9.,3.,9.,7.,4.,3.,5.,5.,3.,9.,0.,5.,4.,5.,7.,9.,7.,5.,7.,5.,3.,7.,2.,9.,1.],
    [6.,7.,8.,9.,2.,2.,5.,5.,6.,3.,8.,1.,6.,4.,1.,7.,9.,4.,4.,6.,3.,6.,7.,3.,7.,3.,2.,4.,8.,9.],
    [7.,1.,3.,9.,1.,9.,8.,1.,6.,5.,6.,5.,6.,5.,5.,4.,3.,3.,4.,9.,2.,9.,1.,1.,5.,6.,6.,3.,7.,9.],
    [8.,6.,8.,5.,8.,4.,8.,2.,5.,9.,4.,6.,1.,9.,1.,5.,4.,3.,6.,2.,3.,5.,9.,2.,8.,2.,5.,7.,8.,1.],
    [6.,1.,2.,1.,5.,9.,6.,1.,1.,2.,9.,3.,1.,5.,7.,6.,1.,5.,5.,6.,3.,5.,7.,5.,5.,5.,3.,1.,5.,9.],
    [1.,3.,4.,1.,1.,8.,2.,8.,7.,2.,6.,6.,3.,2.,1.,6.,5.,9.,1.,7.,0.,5.,2.,3.,7.,6.,2.,6.,2.,2.],
    [2.,3.,2.,4.,9.,6.,1.,8.,7.,3.,1.,5.,4.,8.,1.,1.,4.,8.,5.,2.,6.,5.,2.,3.,9.,7.,7.,6.,8.,4.],
    [8.,4.,8.,9.,3.,3.,2.,3.,3.,5.,5.,2.,6.,5.,6.,1.,5.,9.,2.,1.,9.,9.,9.,6.,8.,1.,4.,1.,8.,1.],
    [3.,4.,8.,6.,7.,8.,2.,8.,3.,7.,3.,7.,2.,6.,3.,3.,9.,7.,5.,7.,9.,1.,7.,6.,9.,1.,4.,9.,4.,3.],
    [9.,2.,4.,6.,2.,8.,8.,1.,3.,9.,5.,6.,4.,2.,8.,6.,5.,9.,1.,5.,6.,5.,6.,6.,7.,4.,8.,7.,9.,7.],
    [3.,3.,8.,5.,4.,8.,6.,2.,4.,4.,6.,6.,1.,8.,6.,3.,9.,2.,8.,3.,5.,6.,6.,4.,3.,0.,1.,4.,1.,1.],
    [6.,5.,4.,3.,1.,6.,2.,8.,5.,7.,2.,8.,2.,4.,1.,5.,9.,1.,9.,8.,6.,7.,3.,1.,5.,2.,5.,8.,9.,2.],
    [3.,3.,1.,8.,6.,8.,9.,5.,3.,5.,5.,5.,3.,4.,9.,9.,2.,8.,8.,7.,8.,3.,6.,1.,7.,1.,9.,1.,1.,7.],
    [2.,5.,8.,8.,3.,7.,3.,6.,6.,3.,7.,3.,8.,4.,1.,4.,7.,4.,6.,9.,1.,5.,7.,1.,4.,6.,2.,5.,1.,6.],
    [6.,4.,8.,4.,8.,8.,5.,4.,2.,3.,3.,4.,4.,5.,4.,8.,6.,6.,8.,7.,8.,2.,9.,1.,8.,4.,4.,5.,8.,5.]
], dtype=float)


# ==============================================================================
# ALGORITMOS DE FACTORIZACIÓN PLU Y SUSTITUCIÓN
# ==============================================================================

def factorizacion_plu(A):
    """
    Calcula la descomposición PA = LU mediante eliminación de Gauss
    con pivoteo parcial.
    """
    A = np.array(A, dtype=float)
    n = len(A)

    U = np.copy(A)
    L = np.eye(n, dtype=float)
    P = np.eye(n, dtype=float)

    for k in range(n - 1):
        # 1. Buscar el pivote de mayor valor absoluto en la columna k (desde fila k hasta n-1)
        piv = k
        mayor = abs(U[k, k])
        for i in range(k + 1, n):
            if abs(U[i, k]) > mayor:
                mayor = abs(U[i, k])
                piv = i

        # 2. Si la fila con mayor valor no es la fila k, intercambiar filas
        if piv != k:
            # Intercambio de filas en U
            temp_U = np.copy(U[k])
            U[k] = U[piv]
            U[piv] = temp_U

            # Intercambio de filas en P
            temp_P = np.copy(P[k])
            P[k] = P[piv]
            P[piv] = temp_P

            # Intercambio de los multiplicadores ya calculados en L (columnas 0 a k-1)
            for col in range(k):
                temp_L = L[k, col]
                L[k, col] = L[piv, col]
                L[piv, col] = temp_L

        # 3. Eliminación hacia adelante para hacer ceros debajo del pivote U[k, k]
        for i in range(k + 1, n):
            factor = U[i, k] / U[k, k]
            L[i, k] = factor
            for j in range(k, n):
                U[i, j] = U[i, j] - factor * U[k, j]

    return P, L, U


def sustitucion_adelante(L, b):
    """
    Resuelve el sistema triangular inferior: L @ y = b
    """
    L = np.array(L, dtype=float)
    b = np.array(b, dtype=float)
    n = len(b)
    y = np.zeros(n, dtype=float)

    for i in range(n):
        suma = 0.0
        for k in range(i):
            suma += L[i, k] * y[k]
        y[i] = (b[i] - suma) / L[i, i]

    return y


def sustitucion_atras(U, y):
    """
    Resuelve el sistema triangular superior: U @ x = y
    """
    U = np.array(U, dtype=float)
    y = np.array(y, dtype=float)
    n = len(y)
    x = np.zeros(n, dtype=float)

    for i in range(n - 1, -1, -1):
        suma = 0.0
        for k in range(i + 1, n):
            suma += U[i, k] * x[k]
        x[i] = (y[i] - suma) / U[i, i]

    return x


def resolver_sistema_plu(P, L, U, b):
    """
    Resuelve Ax = b usando PA = LU:
    1. Multiplicar Pb (reordenar el vector b)
    2. Resolver Ly = Pb (sustitución hacia adelante)
    3. Resolver Ux = y  (sustitución hacia atrás)
    """
    Pb = P @ b
    y = sustitucion_adelante(L, Pb)
    x = sustitucion_atras(U, y)
    return x, y, Pb


def obtener_top_indices(vector, cantidad=3):
    """
    Retorna los índices de los mayores valores del vector en orden descendente.
    """
    ordenados = np.argsort(vector)
    top = []
    for i in range(1, cantidad + 1):
        top.append(int(ordenados[-i]))
    return top


# ==============================================================================
# SECCIÓN 2: ACTIVIDADES DEL TALLER (ACTIVIDADES 1 A 6)
# ==============================================================================

def ejecutar_actividades(A):
    linesPrintsEquals()
    print("SECCIÓN 2: ACTIVIDADES DE FACTORIZACIÓN PLU")
    linesPrintsEquals()

    # --- Actividad 1 ---
    print("\n--- Actividad 1: ¿Por qué es necesario utilizar pivoteo parcial? ---")
    ceros_diagonal = []
    for i in range(len(A)):
        if A[i, i] == 0.0:
            ceros_diagonal.append(i)

    print(f"Primer elemento de la diagonal A[0, 0] = {A[0, 0]}")
    print(f"Filas con ceros en la diagonal principal de A: {ceros_diagonal}")

    explicacion_act_1 = """
    EXPLICACIÓN:
    1. Evitar división por cero:
       En la matriz A del modelo económico, el primer pivote es A[0, 0] = 0 (y además
       existen otros ceros en la diagonal principal en las posiciones 5, 10, 15, 20 y 25).
       Si intentáramos aplicar eliminación de Gauss o el método de Crout directamente sin
       intercambiar filas, en el primer paso se dividiría entre A[0, 0] = 0, generando
       una indeterminación matemática (división por cero).

    2. Estabilidad numérica:
       El pivoteo parcial busca en la columna actual (desde la diagonal hacia abajo) el
       elemento de mayor valor absoluto y lo intercambia a la posición del pivote mediante
       la matriz de permutación P. Así se evitan divisiones por cero y se impide que
       pivotes muy pequeños amplifiquen los errores de redondeo.
    """
    print(explicacion_act_1)

    # --- Actividad 2 ---
    linesPrintsEquals()
    print("--- Actividad 2: Obtención de las matrices P, L y U (PA = LU) ---")
    linesPrintsEquals()

    P, L, U = factorizacion_plu(A)

    print(f"Dimensiones de P: {P.shape}")
    print(f"Dimensiones de L: {L.shape}")
    print(f"Dimensiones de U: {U.shape}")

    print("\nMuestra de la matriz P (primeras 5 filas y 5 columnas):")
    print(P[0:5, 0:5])

    print("\nMuestra de la matriz L (primeras 5 filas y 5 columnas):")
    print(L[0:5, 0:5])

    print("\nMuestra de la matriz U (primeras 5 filas y 5 columnas):")
    print(U[0:5, 0:5])

    # --- Actividad 3 ---
    print("\n")
    linesPrintsEquals()
    print("--- Actividad 3: Verificación numérica de la factorización ||PA - LU||_2 ---")
    linesPrintsEquals()

    PA = P @ A
    LU = L @ U
    norma_factorizacion = np.linalg.norm(PA - LU, 2)

    print(f"Norma euclidiana ||PA - LU||_2 = {norma_factorizacion:.4e}")
    print("¿Es (PA) igual a (LU)?:", np.allclose(PA, LU))
    print("Interpretación: Como el valor es del orden de 1e-14 (prácticamente cero),")
    print("se comprueba que la factorización PA = LU es numéricamente exacta.")

    # --- Actividades 4, 5 y 6 ---
    print("\n")
    linesPrintsEquals()
    print("--- Actividades 4, 5 y 6: Resolución de los 3 escenarios y cálculo de residuos ---")
    linesPrintsEquals()

    # Fijamos semilla para reproducibilidad de los tres vectores en el intervalo [50, 500]
    np.random.seed(42)

    # Generamos los 3 escenarios dentro del intervalo [50, 500]:
    # - Escenario 1 (Normal): valores intermedios en [180, 320]
    # - Escenario 2 (Expansión): aumento de demanda respecto al normal, en [240, 500]
    # - Escenario 3 (Desaceleración): disminución de demanda respecto al normal, en [50, 220]
    b1_normal = np.random.uniform(180.0, 320.0, size=30)
    b2_expansion = b1_normal + np.random.uniform(60.0, 180.0, size=30)
    b3_desaceleracion = b1_normal - np.random.uniform(60.0, 130.0, size=30)

    nombres_escenarios = [
        "Escenario 1 (Condiciones económicas normales)",
        "Escenario 2 (Expansión económica - aumento de demanda)",
        "Escenario 3 (Desaceleración económica - disminución de demanda)"
    ]
    vectores_b = [b1_normal, b2_expansion, b3_desaceleracion]
    soluciones_x = []

    for k in range(3):
        b_k = vectores_b[k]
        x_k, y_k, Pb_k = resolver_sistema_plu(P, L, U, b_k)
        soluciones_x.append(x_k)

        # Actividad 6: Residuo r^(k) = A x^(k) - b^(k) y norma ||r^(k)||_2
        r_k = (A @ x_k) - b_k
        norma_residuo = np.linalg.norm(r_k, 2)

        print(f"\n>> {nombres_escenarios[k]}:")
        print(f"   Rango del vector de demanda b^({k+1}): mín = {b_k.min():.2f}, máx = {b_k.max():.2f}")
        print(f"   Primeros 5 elementos de b^({k+1}):  {b_k[0:5]}")
        print(f"   Primeros 5 elementos de y^({k+1}) (Ly = Pb): {y_k[0:5]}")
        print(f"   Primeros 5 elementos de x^({k+1}) (Ux = y):  {x_k[0:5]}")
        print(f"   Primeros 5 elementos del residuo r^({k+1}):  {r_k[0:5]}")
        print(f"   Norma euclidiana del residuo ||r^({k+1})||_2 = {norma_residuo:.4e}")

    return P, L, U, vectores_b, soluciones_x


# ==============================================================================
# SECCIÓN 3: ANÁLISIS ECONÓMICO
# ==============================================================================

def ejecutar_analisis_economico(soluciones_x):
    print("\n")
    linesPrintsEquals()
    print("SECCIÓN 3: ANÁLISIS ECONÓMICO")
    linesPrintsEquals()

    x1 = soluciones_x[0]
    x2 = soluciones_x[1]
    x3 = soluciones_x[2]

    top_1 = obtener_top_indices(x1, 3)
    top_2 = obtener_top_indices(x2, 3)
    top_3 = obtener_top_indices(x3, 3)

    print("\n1. ¿Qué sectores requieren un mayor nivel de producción en cada escenario?")
    print("   - Escenario 1 (Normal):")
    for idx in top_1:
        print(f"     * Sector {idx + 1} (índice {idx}): {x1[idx]:.4f}")
    print("   - Escenario 2 (Expansión):")
    for idx in top_2:
        print(f"     * Sector {idx + 1} (índice {idx}): {x2[idx]:.4f}")
    print("   - Escenario 3 (Desaceleración):")
    for idx in top_3:
        print(f"     * Sector {idx + 1} (índice {idx}): {x3[idx]:.4f}")

    # Pregunta 2: Mayor cambio entre normal y expansión
    cambio_expansion = np.abs(x2 - x1)
    top_cambio_exp = obtener_top_indices(cambio_expansion, 3)

    print("\n2. ¿Qué sectores presentan el mayor cambio en su nivel de producción entre el")
    print("   escenario normal y el escenario de expansión?")
    for idx in top_cambio_exp:
        diferencia = x2[idx] - x1[idx]
        print(f"     * Sector {idx + 1} (índice {idx}): cambio de {diferencia:+.4f} unidades ( magnitud = {cambio_expansion[idx]:.4f} )")

    # Pregunta 3: Más afectados durante la desaceleración
    cambio_desaceleracion = np.abs(x1 - x3)
    top_cambio_des = obtener_top_indices(cambio_desaceleracion, 3)

    print("\n3. ¿Qué sectores se ven más afectados durante la desaceleración económica?")
    for idx in top_cambio_des:
        diferencia = x3[idx] - x1[idx]
        print(f"     * Sector {idx + 1} (índice {idx}): variación de {diferencia:+.4f} unidades ( magnitud = {cambio_desaceleracion[idx]:.4f} )")

    respuestas_teoricas = """
4. ¿Por qué la matriz A puede mantenerse constante mientras cambia el vector b?
   RESPUESTA:
   Porque la matriz A representa las relaciones productivas y tecnológicas internas entre
   los 30 sectores (cuánto insumo necesita cada sector de los demás para producir). Estas
   relaciones técnicas dependen de la infraestructura industrial y no cambian de un día
   para otro. En cambio, el vector b representa la demanda final del mercado, la cual varía
   constantemente según el estado de la economía (normal, expansión o desaceleración).

5. Si durante el día se reciben 100 nuevos vectores de demanda, ¿sería necesario volver a
   calcular las matrices P, L y U?
   RESPUESTA:
   No. Como la matriz A permanece constante, las matrices P, L y U se calculan una sola vez.
   La descomposición PA = LU depende únicamente de la matriz A y es independiente de b.

6. Explique qué operaciones deben realizarse nuevamente para cada nuevo vector b.
   RESPUESTA:
   Para cada nuevo vector b únicamente se deben realizar tres pasos de bajo costo O(n^2):
   1) Multiplicar la matriz de permutación por el nuevo vector: Pb (reordenar las filas de b).
   2) Resolver el sistema triangular inferior Ly = Pb mediante sustitución hacia adelante.
   3) Resolver el sistema triangular superior Ux = y mediante sustitución hacia atrás.
    """
    print(respuestas_teoricas)


# ==============================================================================
# SECCIÓN 4: ANÁLISIS DE TIEMPO (INVERSA vs. FACTORIZACIÓN PLU)
# ==============================================================================

def ejecutar_analisis_tiempo(A, vectores_b):
    print("\n")
    linesPrintsEquals()
    print("SECCIÓN 4: ANÁLISIS DE TIEMPO (x = A^-1 b vs. scipy.linalg.lu)")
    linesPrintsEquals()

    b_prueba = vectores_b[0]

    # Llamada previa para inicializar las librerías en memoria antes de medir el tiempo
    _ = np.linalg.inv(A) @ b_prueba
    P_w, L_w, U_w = lu(A)
    _ = solve_triangular(U_w, solve_triangular(L_w, P_w.T @ b_prueba, lower=True, unit_diagonal=True), lower=False)

    # 1. Solución analítica con la matriz inversa: Ainv = np.linalg.inv(A)
    inicio_inv = time.perf_counter()
    Ainv = np.linalg.inv(A)
    x_inv = Ainv @ b_prueba
    tiempo_inv = time.perf_counter() - inicio_inv

    # 2. Solución usando factorización PLU de SciPy: from scipy.linalg import lu
    # Nota: scipy.linalg.lu(A) retorna P, L, U tales que A = P @ L @ U (es decir, P.T @ A = L @ U)
    inicio_lu = time.perf_counter()
    P_scipy, L_scipy, U_scipy = lu(A)
    Pb_scipy = P_scipy.T @ b_prueba
    y_scipy = solve_triangular(L_scipy, Pb_scipy, lower=True, unit_diagonal=True)
    x_lu = solve_triangular(U_scipy, y_scipy, lower=False)
    tiempo_lu = time.perf_counter() - inicio_lu

    # 3. Simulación del caso de recibir 100 nuevos vectores de demanda en el día (Pregunta 5)
    np.random.seed(42)
    B_100 = np.random.uniform(50.0, 500.0, size=(30, 100))

    # Si se recalcula la inversa para cada vector vs reutilizar la factorización PLU
    inicio_100_inv = time.perf_counter()
    for col in range(100):
        b_nuevo = B_100[:, col]
        Ainv_temp = np.linalg.inv(A)
        x_temp = Ainv_temp @ b_nuevo
    tiempo_100_inv = time.perf_counter() - inicio_100_inv

    inicio_100_lu = time.perf_counter()
    P_fijo, L_fijo, U_fijo = lu(A)
    y_100 = solve_triangular(L_fijo, P_fijo.T @ B_100, lower=True, unit_diagonal=True)
    x_100 = solve_triangular(U_fijo, y_100, lower=False)
    tiempo_100_lu = time.perf_counter() - inicio_100_lu

    diferencia_soluciones = np.linalg.norm(x_inv - x_lu, 2)

    print(f"\nVerificación de que ambos métodos dan la misma solución:")
    print(f"  ||x_inv - x_lu||_2 = {diferencia_soluciones:.4e}")

    print(f"\nComparación de tiempos de ejecución (1 escenario):")
    print(f"  * Solución con inversa (np.linalg.inv):      {tiempo_inv:.6f} segundos")
    print(f"  * Solución con PLU (scipy.linalg.lu):        {tiempo_lu:.6f} segundos")

    print(f"\nComparación resolviendo los 100 escenarios de demanda del día:")
    print(f"  * Calculando np.linalg.inv en cada consulta: {tiempo_100_inv:.6f} segundos")
    print(f"  * Factorizando una vez con lu(A) y reusando: {tiempo_100_lu:.6f} segundos")

    analisis_tiempo = """
    CONCLUSIÓN DEL ANÁLISIS DE TIEMPO:
    - Calcular la matriz inversa A^-1 tiene un costo de ~ 2*n^3 operaciones, que es tres veces
      más costoso que calcular la factorización PLU (~ (2/3)*n^3 operaciones).
    - Además, cuando se resuelven múltiples vectores de demanda b (como los 100 vectores del día),
      con PLU la matriz A se descompone una sola vez en P, L y U, y cada nuevo vector se resuelve
      únicamente con sustituciones hacia adelante y hacia atrás en tiempo O(n^2).
    - Numéricamente, usar la matriz inversa acumula más errores de redondeo de punto flotante,
      mientras que la factorización PLU con pivoteo parcial es más estable y eficiente.
    """
    print(analisis_tiempo)


# ==============================================================================
# BLOQUE DE EJECUCIÓN PRINCIPAL
# ==============================================================================

def main():
    linesPrintsHashtags()
    print("# RESOLUCIÓN COMPLETA: TALLER 2 - FACTORIZACIÓN PLU Y MODELO ECONÓMICO")
    linesPrintsHashtags()

    P, L, U, vectores_b, soluciones_x = ejecutar_actividades(MATRIZ_A)
    ejecutar_analisis_economico(soluciones_x)
    ejecutar_analisis_tiempo(MATRIZ_A, vectores_b)

    linesPrintsHashtags()
    print("TALLER 2 COMPLETADO")
    linesPrintsHashtags()

main()
