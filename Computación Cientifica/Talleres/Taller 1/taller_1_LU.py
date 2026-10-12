"""
================================================================================
TALLER 1: FACTORIZACIÓN LU (MÉTODO DE CROUT) Y APLICACIONES
Curso: Computación Científica / Métodos Numéricos

Contenido:
1. Implementación de Factorización LU (Algoritmo de Crout).
2. Algoritmos de Sustitución Hacia Adelante (Ly = b) y Hacia Atrás (Ux = y).
3. Ejercicio 1: Planificación de producción de tres productos (A, B, C).
   - Solución mediante factorización de Crout.
   - Análisis de eficiencia computacional (LU vs. Eliminación Gaussiana).
   - Solución y comparación: lu_factor/lu_solve vs. resolución independiente.
4. Ejercicio 2: Estimación de emisiones contaminantes en una red de 50 sensores.
   - Generación sintética y acondicionamiento con dominancia diagonal estricta.
   - Factorización, resolución, cálculo del vector residuo y norma euclidiana ||r||_2.
   - Análisis sobre la reutilización de factores ante nuevos vectores de medición.
=================================================================================
"""

import numpy as np
from scipy.linalg import lu_factor, lu_solve

np.set_printoptions(precision=4, suppress=True)


# ==============================================================================
# FUNCIONES AUXILIARES DE FORMATO
# ==============================================================================

def linesPrintsHashtags():
    print("#" * 80)

def linesPrintsEquals():
    print("=" * 80)


# ==============================================================================
# ALGORITMOS DE FACTORIZACIÓN Y RESOLUCIÓN (MÉTODO DE CROUT)
# ==============================================================================

def factorizacion_crout(A):
    """
    Calcula la descomposición A = L @ U usando el método de Crout.
     
    Formulas (indexación base 0):
    Para cada etapa j de 0 a n-1:
      1. Columna j de L (para i = j, ..., n-1):
         L[i, j] = A[i, j] - sum_{k=0}^{j-1} (L[i, k] * U[k, j])
         
      2. Fila j de U (para k2 = j+1, ..., n-1):
         U[j, k2] = (A[j, k2] - sum_{r=0}^{j-1} (L[j, r] * U[r, k2])) / L[j, j]
    """

    A = np.array(A, dtype=float)
    n = A.shape[0]
    
    # Inicialización de matrices
    L = np.zeros((n, n), dtype=float)
    U = np.eye(n, dtype=float)  # Diagonal con 1s
    
    for j in range(n):
        # 1. Construcción de los elementos de la columna j de L (desde la diagonal hacia abajo)
        for i in range(j, n):
            suma_L = 0.0
            for k in range(j):
                suma_L += L[i, k] * U[k, j]
            L[i, j] = A[i, j] - suma_L
        
        # Se calcula la fila j de U solo si el pivote es diferente de cero
        if L[j, j] != 0.0:
            # 2. Construcción de los elementos de la fila j de U (a la derecha de la diagonal)
            for k2 in range(j + 1, n):
                suma_U = 0.0
                for r in range(j):
                    suma_U += L[j, r] * U[r, k2]
                U[j, k2] = (A[j, k2] - suma_U) / L[j, j]
            
    return L, U


def sustitucion_adelante(L, b):
    """
    Resuelve el sistema triangular inferior: L @ y = b
    
    Formula:
      y[i] = (b[i] - sum_{k=0}^{i-1} L[i, k] * y[k]) / L[i, i]
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
    
    Nota: En el método de Crout, U[i, i] = 1, por lo que la división es por 1.
    Formula:
      x[i] = (y[i] - sum_{k=i+1}^{n-1} U[i, k] * x[k]) / U[i, i]
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


def resolver_sistema_lu(L, U, b):
    """
    Resuelve Ax = b dado A = LU mediante:
    1. Ly = b (Sustitución hacia adelante)
    2. Ux = y (Sustitución hacia atrás)
    """
    y = sustitucion_adelante(L, b)
    x = sustitucion_atras(U, y)
    return x, y


# ==============================================================================
# EJERCICIO 1: CANTIDAD DE PRODUCTOS GENERADOS
# ==============================================================================

def ejecutar_ejercicio_1():
    linesPrintsEquals()
    print("EJERCICIO 1: PLANIFICACIÓN DE PRODUCCIÓN (MATRICES LU)")
    linesPrintsEquals()
    
    # Definición de la matriz de consumo de materia prima (Acero, Plástico, Aluminio)
    # Fila 1: Acero    (2*A + 1*B + 1*C = b1)
    # Fila 2: Plástico (1*A + 3*B + 1*C = b2)
    # Fila 3: Aluminio (1*A + 1*B + 2*C = b3)
    A = np.array([
        [2.0, 1.0, 1.0],
        [1.0, 3.0, 1.0],
        [1.0, 1.0, 2.0]
    ])
    
    print("\n--- Actividad 1: Factorización LU de la matriz de coeficientes ---")
    print("Matriz A (Coeficientes de Insumos):")
    print(A)
    
    L, U = factorizacion_crout(A)
    
    print("\nMatriz L (Triangular Inferior - Crout):")
    print(L)
    print("\nMatriz U (Triangular Superior con 1s en la diagonal):")
    print(U)
    
    # Verificación matricial A == L @ U
    print("\nVerificación de la descomposición (L @ U):")
    print(L @ U)
    print("¿Es igual (L @ U) a (A)?:", np.allclose(A, L @ U))
    
    print("\n")
    
    # Actividad 2: Escenarios de disponibilidad de materia prima
    print("\n")
    linesPrintsEquals()
    print("--- Actividad 2: Solución para los Tres Escenarios de Producción ---")
    linesPrintsEquals()
    
    escenarios = {
        "Escenario 1 (Acero: 100kg, Plástico: 120kg, Aluminio: 90kg)": np.array([100.0, 120.0, 90.0]),
        "Escenario 2 (Acero: 80kg,  Plástico: 100kg, Aluminio: 110kg)": np.array([80.0, 100.0, 110.0]),
        "Escenario 3 (Acero: 150kg, Plástico: 160kg, Aluminio: 140kg)": np.array([150.0, 160.0, 140.0])
    }
    
    productos = ["Producto A (x1)", "Producto B (x2)", "Producto C (x3)"]
    
    for nombre_escenario, b in escenarios.items():
        print(f"\n>> {nombre_escenario}:")
        print(f"   Vector de recursos b = {b}")
        
        # Paso 1: Ly = b (Sustitución hacia adelante)
        y = sustitucion_adelante(L, b)
        print(f"   Vector intermedio y (Ly = b): {y}")
        
        # Paso 2: Ux = y (Sustitución hacia atrás)
        x = sustitucion_atras(U, y)
        print(f"   Solución de producción x (Ux = y):")
        for i in range(len(productos)):
            print(f"     * {productos[i]}: {x[i]:.4f} unidades")
    
    # Actividad 3: Justificación Teorica
    print("\n")
    linesPrintsEquals()
    print("--- Actividad 3: Explicación de Eficiencia Computacional ---")
    analisis_actividad_3 = """
    ANÁLISIS DE EFICIENCIA (LU vs. Eliminación Gaussiana):
    
    1. Costo de la Eliminación Gaussiana:
       - Resolver un sistema Ax = b desde cero requiere triangularizar la matriz aumentada [A|b],
         lo cual tiene un costo computacional de O(n^3) operaciones.
       - Si tenemos 'm' escenarios distintos (en este caso m = 3), repetir Gauss para cada uno 
         costaría m * O(n^3) operaciones.
         
    2. Complejidad con Factorización LU:
       - La descomposición A = LU se calcula UNA SOLA VEZ, con un costo de O(n^3).
       - Para cada nuevo escenario 'b', solo se resuelven dos sistemas triangulares:
         * Ly = b mediante sustitución hacia adelante: O(n^2).
         * Ux = y mediante sustitución hacia atrás: O(n^2).
       - El costo total para 'm' escenarios es: O(n^3) + m * O(2 * n^2).
       
    3. Conclusion:
        Para sistemas grandes (n grande) o múltiples escenarios (m grande), el término O(n^3)
        domina drásticamente. 
        Factorizar una sola vez ahorra un trabajo computacional inmenso, ya que pasar de O(n^3)
        a O(n^2) para cada nuevo vector de demanda/recursos acelera radicalmente el tiempo de respuesta.
    """
    print(analisis_actividad_3)
    
    # Actividad 4: Resolución con SciPy y Comparacion
    print("\n")
    linesPrintsEquals()
    print("--- Actividad 4: Resolución con SciPy (lu_factor y lu_solve) ---")
    
    # SciPy utiliza internamente factorización PLU (LAPACK dgetrf / dgetrs)
    lu, piv = lu_factor(A)
    print("Factorización LU compacta obtenida con scipy.linalg.lu_factor:")
    print(lu)
    print("Vector de pivoteo:", piv)
    
    print("\nResultados obtenidos con SciPy lu_solve:")
    for nombre_escenario, b in escenarios.items():
        x_scipy = lu_solve((lu, piv), b)
        print(f"   {nombre_escenario}:")
        print(f"     x_scipy = {x_scipy}")
        
    comparativa_scipy = """
    COMPARACIÓN ENTRE IMPLEMENTACIÓN MANUAL Y SCIPY:
    - Precision: Ambos métodos obtienen exactamente la misma solución matemática.
    - Metodología: La función manual aplica Crout directamente en el orden original. 
      SciPy utiliza factorización PLU con pivoteo parcial automático, lo cual reordena 
      estratégicamente las filas para evitar divisiones entre ceros o números muy pequeños, 
      garantizando estabilidad numérica.
    - Rendimiento: Para producción y proyectos a gran escala, lu_solve de SciPy es 
      ampliamente superior en velocidad debido a su procesamiento optimizado de alto rendimiento.
    """
    print(comparativa_scipy)


# ==============================================================================
# EJERCICIO 2: ESTIMACIÓN DE EMISIONES EN RED DE SENSORES
# ==============================================================================

def modificar_a_diagonal_dominante_estricta(A):
    """
    Garantiza que la matriz A sea estrictamente dominante por filas.
    
    Formula solicitada:
      A[i, i] = sum_{j=0, j != i}^{n-1} |A[i, j]| + 1
      
    Importancia matemática:
    Una matriz estrictamente diagonalmente dominante cumple:
      |A[i, i]| > sum_{j != i} |A[i, j]|
    Por el Teorema de los Círculos de Gershgorin, esto garantiza:
    1. Que la matriz es invertible (det(A) != 0).
    2. Que la factorización LU (sin pivoteo) existe y es numéricamente estable.
    """
    A_mod = np.copy(A)
    n = A_mod.shape[0]
    for i in range(n):
        suma_fuera_diagonal = 0.0
        for j in range(n):
            if j != i:
                suma_fuera_diagonal += abs(A_mod[i, j])
        A_mod[i, i] = suma_fuera_diagonal + 1.0
    return A_mod


def ejecutar_ejercicio_2():
    print("\n")
    linesPrintsEquals()
    print("EJERCICIO 2: ESTIMACION DE EMISIONES EN RED DE SENSORES")
    linesPrintsEquals()
    
    # Fijamos semilla para reproducibilidad
    np.random.seed(42)
    n = 50
    
    print(f"\n--- Actividad 1: Generación de Matriz A ({n}x{n}) en el intervalo [0, 1] ---")
    A_sintetica = np.random.uniform(0.0, 1.0, size=(n, n))
    print(f"Dimensiones de A: {A_sintetica.shape}")
    print("Muestra de submatriz A[0:3, 0:3]:")
    print(A_sintetica[0:3, 0:3])
    
    print("\n--- Actividad 2: Modificación a Diagonal Dominante Estricta ---")
    A_dominante = modificar_a_diagonal_dominante_estricta(A_sintetica)
    print(f"Nuevo valor A[0,0]: {A_dominante[0,0]:.4f} (Suma de fila sin diagonal + 1)")
    print("Muestra de submatriz A_dominante[0:3, 0:3]:")
    print(A_dominante[0:3, 0:3])
    
    # Verificación de condición de dominancia
    es_dominante = True
    for i in range(n):
        suma_fila = 0.0
        for j in range(n):
            if j != i:
                suma_fila += abs(A_dominante[i, j])
        if A_dominante[i, i] <= suma_fila:
            es_dominante = False
    print("¿Se cumple la dominancia diagonal estricta en todas las 50 filas?:", es_dominante)
    
    print(f"\n--- Actividad 3: Generación del Vector de Mediciones b (50 elementos en [0, 100]) ---")
    b = np.random.uniform(0.0, 100.0, size=n)
    print(f"Dimensiones de b: {b.shape}")
    print("Primeras 5 mediciones de sensores b[0:5]:")
    print(b[0:5])
    
    print("\n--- Actividad 4: Factorización LU con el Método de Crout ---")
    L, U = factorizacion_crout(A_dominante)
    print(f"Matriz L generada: forma {L.shape}, es triangular inferior.")
    print(f"Matriz U generada: forma {U.shape}, es triangular superior con U[i,i] = 1.")
    
    # Error de descomposición ||A - LU||
    error_factorizacion = np.linalg.norm(A_dominante - (L @ U))
    print(f"Error de aproximación ||A - LU||: {error_factorizacion:.2e}")
    
    print("\n--- Actividad 5: Resolución mediante Sustitución Hacia Adelante y Hacia Atrás ---")
    # Paso 1: Ly = b
    y = sustitucion_adelante(L, b)
    # Paso 2: Ux = y
    x = sustitucion_atras(U, y)
    
    print("Intensidades estimadas de las primeras 5 fuentes de emisión x[0:5]:")
    print(x[0:5])
    print(f"Rango de intensidades calculadas: mín = {x.min():.4f}, máx = {x.max():.4f}")
    
    print("\n--- Actividad 6: Verificación de la Solución (Vector Residuo y Norma ||r||_2) ---")
    # Vector residuo: r = Ax - b
    r = A_dominante @ x - b
    norma_residuo = np.linalg.norm(r, 2)
    
    print(f"Primeros 5 elementos del vector residuo r[0:5]:")
    print(r[0:5])
    print(f"\nValor de la Norma Euclidiana ||r||_2 = {norma_residuo:.2e}")
    print("Interpretacion: La norma es prácticamente cero (del orden de precisión de máquina ~1e-14),")
    print("lo que confirma que la solución encontrada 'x' es numéricamente exacta.")
    
    # Pregunta de Análisis
    print("\n" + "-" * 80)
    print("--- Pregunta de Análisis del Ejercicio 2 ---")
    pregunta_analisis_2 = """
    PREGUNTA:
    Suponga que la red de sensores permanece igual, pero se realizan nuevas mediciones
    en otro momento (es decir, A permanece constante pero cambia el vector b).
    ¿Qué parte del procedimiento podría reutilizarse para resolver el nuevo sistema 
    de manera más eficiente?
    
    RESPUESTA Y EXPLICACIÓN:
    1. Procedimiento a Reutilizar:
       - Se reutilizan directamente las matrices L y U ya calculadas (Factorización A = LU).
       
    2. Procedimiento a Ejecutar para cada nueva medición b_nueva:
       - Únicamente se ejecutan las dos sustituciones:
         1) Sustitución hacia adelante: L * y = b_nueva
         2) Sustitución hacia atrás:    U * x = y
         
    3. Justificación del Ahorro Computacional:
       - La matriz A representa la geometría física fija entre las 50 fuentes y los 50 sensores 
         (distancias, factores de dispersión atmosférica constante, etc.).
       - Calcular A = LU requiere ~ (2/3)*50^3 ≈ 83,333 operaciones.
       - En cambio, resolver con L y U solo requiere ~ 2 * 50^2 = 5,000 operaciones por cada muestra.
    """
    print(pregunta_analisis_2)


# ==============================================================================
# BLOQUE DE EJECUCIÓN PRINCIPAL
# ==============================================================================

def main():
    linesPrintsHashtags()
    print("# RESOLUCIÓN COMPLETA: TALLER 1 - FACTORIZACIÓN LU (MÉTODO DE CROUT)")
    linesPrintsHashtags()

    ejecutar_ejercicio_1()
    ejecutar_ejercicio_2()
    
    linesPrintsHashtags()
    print("TALLER 1 COMPLETADO")
    linesPrintsHashtags()

main()