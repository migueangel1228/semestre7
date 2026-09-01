"""
================================================================================
TALLER 2: MODELO ECONÓMICO MULTISECTORIAL MEDIANTE FACTORIZACIÓN PLU
Curso: Computación Científica / Métodos Numéricos
Nivel: Universitario

Descripción general:
Este archivo contiene la solución completa, estructurada y pedagógica del 
Taller 2. Incluye:
1. Justificación teórica de la necesidad del pivoteo parcial en matrices con ceros
   en la diagonal principal (evitar división por cero e inestabilidad numérica).
2. Implementación desde cero de la Factorización PLU (PA = LU) con pivoteo parcial.
3. Implementación de algoritmos de Sustitución Hacia Adelante (Ly = Pb) y Hacia Atrás (Ux = y).
4. Solución del modelo económico intersectorial de 30 sectores (Matriz A de 30x30)
   bajo tres escenarios de demanda:
   - Escenario 1: Condiciones económicas normales.
   - Escenario 2: Expansión económica (aumento de demanda).
   - Escenario 3: Desaceleración económica (disminución de demanda).
5. Verificación numérica de la descomposición (||PA - LU||_2) y de los residuos (||r||_2).
6. Análisis económico detallado respondiendo a las 6 preguntas planteadas.
7. Análisis de rendimiento y tiempo de ejecución: Comparación entre la solución con 
   matriz inversa (x = A^-1 b) y la factorización PLU (scipy.linalg.lu), con 
   justificación teórica de por qué no se debe invertir matrices en computación científica.
================================================================================
"""

import time
import numpy as np
from scipy.linalg import lu, lu_factor, lu_solve

# Configuramos la visualización de NumPy para una lectura limpia
np.set_printoptions(precision=4, suppress=True)


# ==============================================================================
# FUNCIONES AUXILIARES DE FORMATO
# ==============================================================================

def linesPrintsHashtags():
    print("\n" + "#" * 80)

def linesPrintsEquals():
    print("=" * 80)


# ==============================================================================
# DEFINICIÓN DE LA MATRIZ DE REQUERIMIENTOS INTERSECTORIALES (30x30)
# ==============================================================================

# Matriz A de requerimientos entre los 30 sectores económicos (obtenida del Colab oficial
# (https://colab.research.google.com/drive/1hZE7pRAhSXtk4YFC5MnwIS0SkI_FRVfd?usp=sharing))
MATRIZ_A_ECONOMIA = np.array([
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
# FUNCIONES/ALGORITMOS NÚCLEO: FACTORIZACIÓN PLU Y SUSTITUCIONES
# ==============================================================================

def factorizacion_plu(A):
    """
    Calcula la descomposición P @ A = L @ U usando Eliminación Gaussiana 
    con Pivoteo Parcial.
    
    Parámetros:
        A : Matriz cuadrada (numpy.ndarray de dimensión n x n).
        
    Retorna:
        P : Matriz de permutación (n x n).
        L : Matriz triangular inferior con 1s en la diagonal principal (n x n).
        U : Matriz triangular superior obtenida tras la eliminación (n x n).
        
    Algoritmo:
    Para cada etapa k de 0 a n-2:
      1. Búsqueda del pivote parcial:
         Se selecciona la fila 'piv' (con piv >= k) tal que |U[piv, k]| sea máximo.
      2. Intercambio de filas (Pivoteo):
         - Se intercambian las filas k y piv en U.
         - Se intercambian las filas k y piv en P.
         - Se intercambian los multiplicadores ya calculados en L (columnas 0 a k-1).
      3. Eliminación gaussiana:
         Para cada fila i de k+1 a n-1:
           factor = U[i, k] / U[k, k]
           L[i, k] = factor
           U[i, k:] = U[i, k:] - factor * U[k, k:]
    """
    A = np.array(A, dtype=float)
    n = A.shape[0]
    
    # Inicialización de matrices
    U = A.copy()
    L = np.eye(n, dtype=float)  # Diagonal con 1s
    P = np.eye(n, dtype=float)  # Matriz identidad para rastrear permutaciones
    
    for k in range(n - 1):
        # 1. Selección del pivote con mayor valor absoluto en la columna k desde la fila k
        piv = np.argmax(np.abs(U[k:, k])) + k
        
        # Validación de matriz singular
        if np.isclose(U[piv, k], 0.0):
            raise ValueError(f"La matriz es singular o numéricamente no invertible en la etapa {k}.")
            
        # 2. Intercambio de filas si el pivote no está en la diagonal actual
        if piv != k:
            # Permutación en U (todas las columnas)
            U[[k, piv], :] = U[[piv, k], :]
            # Permutación en P (acumula el orden final de filas)
            P[[k, piv], :] = P[[piv, k], :]
            # Permutación en L (solo las columnas de multiplicadores previos 0 a k-1)
            L[[k, piv], :k] = L[[piv, k], :k]
            
        # 3. Eliminación hacia adelante (construcción de L y U)
        for i in range(k + 1, n):
            factor = U[i, k] / U[k, k]
            L[i, k] = factor
            U[i, k:] -= factor * U[k, k:]
            
    return P, L, U


def sustitucion_adelante(L, b):
    """
    Resuelve el sistema triangular inferior: L @ y = b
    Dado que L tiene 1s en la diagonal principal, L[i, i] = 1.
    
    Fórmula:
      y[i] = b[i] - sum_{k=0}^{i-1} (L[i, k] * y[k])
    """
    L = np.array(L, dtype=float)
    b = np.array(b, dtype=float)
    n = len(b)
    y = np.zeros(n, dtype=float)
    
    for i in range(n):
        suma = sum(L[i, k] * y[k] for k in range(i))
        y[i] = (b[i] - suma) / L[i, i]
        
    return y


def sustitucion_atras(U, y):
    """
    Resuelve el sistema triangular superior: U @ x = y
    
    Fórmula:
      x[i] = (y[i] - sum_{k=i+1}^{n-1} (U[i, k] * x[k])) / U[i, i]
    """
    U = np.array(U, dtype=float)
    y = np.array(y, dtype=float)
    n = len(y)
    x = np.zeros(n, dtype=float)
    
    for i in range(n - 1, -1, -1):
        suma = sum(U[i, k] * x[k] for k in range(i + 1, n))
        x[i] = (y[i] - suma) / U[i, i]
        
    return x


def resolver_sistema_plu(P, L, U, b):
    """
    Resuelve el sistema lineal Ax = b mediante factorización PLU:
      PA = LU  ==>  LUx = Pb
      
    Pasos:
      1. Permutar el vector b: b_perm = P @ b
      2. Sustitución hacia adelante: L @ y = b_perm
      3. Sustitución hacia atrás:    U @ x = y
    """
    b_perm = P @ b
    y = sustitucion_adelante(L, b_perm)
    x = sustitucion_atras(U, y)
    return x, y, b_perm


# ==============================================================================
# ACTIVIDAD 1: EXPLICACIÓN DE LA NECESIDAD DE PIVOTEO PARCIAL
# ==============================================================================

def mostrar_actividad_1(A):
    linesPrintsEquals()
    print("ACTIVIDAD 1: ¿POR QUÉ ES NECESARIO EL PIVOTEO PARCIAL EN ESTE CASO?")
    linesPrintsEquals()
    
    print(f"\n1. Inspección de elementos de la diagonal principal de A:")
    ceros_en_diagonal = [i for i in range(A.shape[0]) if np.isclose(A[i, i], 0.0)]
    print(f"   - El primer elemento diagonal A[0, 0] = {A[0, 0]}")
    print(f"   - Índices de filas con ceros en la diagonal principal: {ceros_en_diagonal}")
    
    explicacion_pivoteo = """
    EXPLICACIÓN MATEMÁTICA Y NUMÉRICA:
    
    1. Prevención de División por Cero:
       En el algoritmo de eliminación gaussiana estándar (LU directo sin pivoteo),
       el primer paso requiere calcular los multiplicadores:
           m_{i,0} = A[i, 0] / A[0, 0]
       Como A[0, 0] = 0.0, ocurre una división por cero inmediata que hace imposible 
       continuar el cálculo.
       
    2. Estabilidad Numérica y Control de Errores de Redondeo:
       Aun cuando un pivote no sea exactamente cero, si su valor absoluto es muy pequeño, 
       dividir por él genera multiplicadores gigantescos. Esto amplifica los errores 
       de redondeo en aritmética de punto flotante de 64 bits (pérdida de precisión por 
       cancelación catastrófica).
       
    3. Solución mediante Factorización PLU:
       El pivoteo parcial inspecciona todos los candidatos de la columna k desde la fila k 
       hacia abajo y selecciona el elemento con mayor magnitud (|A[piv, k]|). 
       Al intercambiar filas mediante la matriz de permutación P, se garantiza que 
       |L[i, k]| <= 1 para todo i > k, asegurando estabilidad numérica óptima.
    """
    print(explicacion_pivoteo)


# ==============================================================================
# ACTIVIDADES 2 Y 3: FACTORIZACIÓN PLU Y VERIFICACIÓN NUMÉRICA
# ==============================================================================

def ejecutar_actividades_2_y_3(A):
    linesPrintsEquals()
    print("ACTIVIDADES 2 Y 3: FACTORIZACIÓN PLU Y VERIFICACIÓN NUMÉRICA")
    linesPrintsEquals()
    
    print("\n--- Actividad 2: Implementación de Factorización PLU (PA = LU) ---")
    P, L, U = factorizacion_plu(A)
    
    print(f"Forma de las matrices generadas:")
    print(f"  * P (Matriz de Permutación):       {P.shape}")
    print(f"  * L (Triangular Inferior con 1s): {L.shape}")
    print(f"  * U (Triangular Superior):         {U.shape}")
    
    print("\nMuestra de la submatriz L[0:5, 0:5] (Triangular inferior unitaria):")
    print(L[0:5, 0:5])
    
    print("\nMuestra de la submatriz U[0:5, 0:5] (Triangular superior):")
    print(U[0:5, 0:5])
    
    print("\n" + "-" * 80)
    print("--- Actividad 3: Verificación Numérica ||PA - LU||_2 ---")
    PA = P @ A
    LU_prod = L @ U
    error_plu = np.linalg.norm(PA - LU_prod, 2)
    
    print(f"Norma Euclidiana de error: ||PA - LU||_2 = {error_plu:.4e}")
    print("¿La descomposición es numéricamente exacta?:", np.isclose(error_plu, 0.0))
    print("Interpretación: El error es del orden de 1e-14 (precisión de máquina), lo que")
    print("demuestra la exactitud de la factorización calculada.")
    
    return P, L, U


# ==============================================================================
# ACTIVIDADES 4, 5 Y 6: SOLUCIÓN DE LOS ESCENARIOS ECONÓMICOS Y RESIDUOS
# ==============================================================================

def ejecutar_actividades_4_5_6(A, P, L, U):
    linesPrintsEquals()
    print("ACTIVIDADES 4, 5 Y 6: EVALUACIÓN DE ESCENARIOS ECONÓMICOS")
    linesPrintsEquals()
    
    # Fijamos semilla para reproducibilidad de los datos sintéticos de demanda [50, 500]
    np.random.seed(100)
    
    # 1. Escenario 1: Demanda normal base (distribuida en [50, 500])
    b_normal = np.random.uniform(50.0, 500.0, size=30)
    
    # 2. Escenario 2: Expansión económica (+30% de incremento en la demanda final)
    b_expansion = b_normal * 1.30
    
    # 3. Escenario 3: Desaceleración económica (-30% de reducción en la demanda final)
    b_desaceleracion = b_normal * 0.70
    
    escenarios = {
        "Escenario 1 (Condiciones Normales)": b_normal,
        "Escenario 2 (Expansión Económica)": b_expansion,
        "Escenario 3 (Desaceleración Económica)": b_desaceleracion
    }
    
    soluciones_x = {}
    residuos_normas = {}
    
    print("\n--- Actividades 4 y 5: Resolución mediante Ly = Pb y Ux = y ---")
    
    for idx, (nombre, b_k) in enumerate(escenarios.items(), start=1):
        x_k, y_k, Pb_k = resolver_sistema_plu(P, L, U, b_k)
        soluciones_x[nombre] = x_k
        
        # Actividad 6: Vector residuo r = Ax - b y cálculo de ||r||_2
        r_k = A @ x_k - b_k
        norma_r = np.linalg.norm(r_k, 2)
        residuos_normas[nombre] = norma_r
        
        print(f"\n>> {nombre}:")
        print(f"   Demanda promedio sum(b): {np.sum(b_k):.2f} unidades")
        print(f"   Muestra de vector intermedio y[0:4]: {y_k[0:4]}")
        print(f"   Nivel de producción total requerido sum(x): {np.sum(x_k):.2f} unidades")
        print(f"   Primeros 5 sectores de producción x[0:5]:")
        print(f"   {x_k[0:5]}")
        print(f"   [Actividad 6] Norma del Residuo ||r||_2 = {norma_r:.4e}")
        
    return escenarios, soluciones_x, residuos_normas


# ==============================================================================
# SECCIÓN 3: ANÁLISIS ECONÓMICO MULTISECTORIAL
# ==============================================================================

def ejecutar_analisis_economico(A, escenarios, soluciones_x):
    linesPrintsEquals()
    print("SECCIÓN 3: ANÁLISIS ECONÓMICO DETALLADO")
    linesPrintsEquals()
    
    x1 = soluciones_x["Escenario 1 (Condiciones Normales)"]
    x2 = soluciones_x["Escenario 2 (Expansión Económica)"]
    x3 = soluciones_x["Escenario 3 (Desaceleración Económica)"]
    
    # 1. Sectores con mayor nivel de producción
    top_sectores_1 = [int(i) for i in np.argsort(x1)[::-1][:3]]
    top_sectores_2 = [int(i) for i in np.argsort(x2)[::-1][:3]]
    top_sectores_3 = [int(i) for i in np.argsort(x3)[::-1][:3]]
    
    print("\n1. ¿Qué sectores requieren un mayor nivel de producción en cada escenario?")
    print(f"   - Escenario Normal:         Sectores {top_sectores_1} con producciones {[float(f'{x1[i]:.2f}') for i in top_sectores_1]}")
    print(f"   - Escenario Expansión:      Sectores {top_sectores_2} con producciones {[float(f'{x2[i]:.2f}') for i in top_sectores_2]}")
    print(f"   - Escenario Desaceleración: Sectores {top_sectores_3} con producciones {[float(f'{x3[i]:.2f}') for i in top_sectores_3]}")
    print("   -> Explicación: Los sectores con mayor requerimiento son aquellos que actúan como insumos")
    print("      clave para toda la economía (por ejemplo energía, transporte, materias primas básicas).")
    
    # 2. Mayor cambio entre escenario normal y de expansión
    cambio_expansion = x2 - x1
    top_cambio_exp = [int(i) for i in np.argsort(cambio_expansion)[::-1][:3]]
    
    print("\n2. ¿Qué sectores presentan el mayor cambio en producción entre el escenario normal y el de expansión?")
    print(f"   - Sectores con mayor aumento absoluto: {top_cambio_exp}")
    print(f"   - Incrementos absolutos (unidades):     {[float(f'{cambio_expansion[i]:.2f}') for i in top_cambio_exp]}")
    print(f"   - Crecimiento porcentual medio:         {np.mean(cambio_expansion / x1) * 100:.2f}%")
    
    # 3. Sectores más afectados durante la desaceleración
    caida_desaceleracion = x1 - x3
    top_caida_des = [int(i) for i in np.argsort(caida_desaceleracion)[::-1][:3]]
    
    print("\n3. ¿Qué sectores se ven más afectados durante la desaceleración económica?")
    print(f"   - Sectores con mayor caída en producción: {top_caida_des}")
    print(f"   - Caídas absolutas (unidades):             {[float(f'{caida_desaceleracion[i]:.2f}') for i in top_caida_des]}")
    print(f"   - Reducción porcentual media:              {np.mean(caida_desaceleracion / x1) * 100:.2f}%")
    
    # 4. Explicación de por qué A se mantiene constante mientras b cambia
    print("\n4. ¿Por qué la matriz A puede mantenerse constante mientras cambia el vector b?")
    resp_4 = """
       La matriz A representa la 'estructura tecnológica y de encadenamientos productivos' 
       de la economía (los coeficientes de insumo-producto de Leontief). Esta estructura depende 
       de la tecnología instalada y los procesos industriales, por lo que es constante en el 
       corto/mediano plazo. En contraste, el vector b representa la 'demanda final de mercado'
       (consumo de hogares, gasto público, exportaciones), el cual cambia de forma dinámica 
       según las fluctuaciones del ciclo económico.
    """
    print(resp_4)
    
    # 5. y 6. Reutilización de matrices P, L, U para 100 nuevos vectores de demanda
    print("5. y 6. Si se reciben 100 nuevos vectores de demanda:")
    resp_5_6 = """
       - Respuesta a la Pregunta 5: NO es necesario volver a calcular P, L y U.
         Como la matriz A es constante, la factorización PA = LU se realiza UNA SOLA VEZ 
         y se mantiene en memoria.
         
       - Respuesta a la Pregunta 6 (Operaciones a realizar por cada nuevo vector b_nuevo):
         Para cada una de las 100 nuevas demandas únicamente se ejecutan:
           1) Permutación del vector:   b_perm = P @ b_nuevo      [O(n) o O(1) con índices]
           2) Sustitución hacia adelante: L @ y = b_perm          [O(n^2) operaciones]
           3) Sustitución hacia atrás:    U @ x = y               [O(n^2) operaciones]
           
         Ahorro: Se pasa de un costo O(2/3 * 30^3) = 18,000 FLOPs por escenario a solo
         O(2 * 30^2) = 1,800 FLOPs. ¡Un ahorro del 90% de operaciones por cada nueva consulta!
    """
    print(resp_5_6)


# ==============================================================================
# SECCIÓN 4: ANÁLISIS DE TIEMPO (INVERSA vs. FACTORIZACIÓN PLU)
# ==============================================================================

def ejecutar_analisis_tiempo(A):
    linesPrintsEquals()
    print("SECCIÓN 4: ANÁLISIS DE TIEMPO DE EJECUCIÓN (INVERSA vs. FACTORIZACIÓN PLU)")
    linesPrintsEquals()
    
    np.random.seed(42)
    b_test = np.random.uniform(50.0, 500.0, size=30)
    N_ITERACIONES = 10000
    
    print(f"\nRealizando benchmark con {N_ITERACIONES:,} repeticiones:")
    
    # --- Caso A: Computando desde cero en cada llamada ---
    # Método 1: Solución analítica con matriz inversa np.linalg.inv(A)
    t_inicio = time.perf_counter()
    for _ in range(N_ITERACIONES):
        A_inv = np.linalg.inv(A)
        x_inv = A_inv @ b_test
    t_total_inv = time.perf_counter() - t_inicio
    
    # Método 2: Solución con scipy.linalg.lu (PLU)
    t_inicio = time.perf_counter()
    for _ in range(N_ITERACIONES):
        P_scipy, L_scipy, U_scipy = lu(A)
        # En scipy.linalg.lu: A = P @ L @ U ==> L @ U @ x = P.T @ b
        y_scipy = sustitucion_adelante(L_scipy, P_scipy.T @ b_test)
        x_scipy = sustitucion_atras(U_scipy, y_scipy)
    t_total_lu = time.perf_counter() - t_inicio
    
    # Método 3: Con scipy.linalg.lu_factor y lu_solve (rutina optimizada en C/LAPACK)
    t_inicio = time.perf_counter()
    for _ in range(N_ITERACIONES):
        lu_piv = lu_factor(A)
        x_lapack = lu_solve(lu_piv, b_test)
    t_total_lapack = time.perf_counter() - t_inicio
    
    print(f"\n1. Calculando todo desde cero cada vez ({N_ITERACIONES:,} repeticiones):")
    print(f"   * Matriz Inversa (np.linalg.inv):        {t_total_inv:.4f} segundos ({t_total_inv/N_ITERACIONES*1e6:.2f} µs/op)")
    print(f"   * Factorización PLU (scipy.linalg.lu):   {t_total_lu:.4f} segundos ({t_total_lu/N_ITERACIONES*1e6:.2f} µs/op)")
    print(f"   * PLU optimizado (lu_factor + lu_solve): {t_total_lapack:.4f} segundos ({t_total_lapack/N_ITERACIONES*1e6:.2f} µs/op)")
    
    # --- Caso B: Con estructura precalculada para múltiples vectores b ---
    A_inv_pre = np.linalg.inv(A)
    P_pre, L_pre, U_pre = factorizacion_plu(A)
    lu_piv_pre = lu_factor(A)
    
    t_inicio = time.perf_counter()
    for _ in range(N_ITERACIONES):
        x_inv_pre = A_inv_pre @ b_test
    t_pre_inv = time.perf_counter() - t_inicio
    
    t_inicio = time.perf_counter()
    for _ in range(N_ITERACIONES):
        x_plu_pre, _, _ = resolver_sistema_plu(P_pre, L_pre, U_pre, b_test)
    t_pre_plu = time.perf_counter() - t_inicio
    
    t_inicio = time.perf_counter()
    for _ in range(N_ITERACIONES):
        x_lapack_pre = lu_solve(lu_piv_pre, b_test)
    t_pre_lapack = time.perf_counter() - t_inicio
    
    print(f"\n2. Reutilizando la estructura precalculada ({N_ITERACIONES:,} repeticiones):")
    print(f"   * Producto Inversa precalculada (A^-1 @ b): {t_pre_inv:.4f} segundos ({t_pre_inv/N_ITERACIONES*1e6:.2f} µs/op)")
    print(f"   * Sustituciones PLU manuales:               {t_pre_plu:.4f} segundos ({t_pre_plu/N_ITERACIONES*1e6:.2f} µs/op)")
    print(f"   * Sustituciones lu_solve de SciPy:          {t_pre_lapack:.4f} segundos ({t_pre_lapack/N_ITERACIONES*1e6:.2f} µs/op)")
    
    conclusion_tiempo = """
    JUSTIFICACIÓN TEÓRICA DE POR QUÉ NO SE DEBE INVERTIR UNA MATRIZ:
    
    1. Regla Fundamental de Computación Científica: 'Nunca calcules la inversa explícita A^-1'.
    2. Costo Computacional:
       - Invertir una matriz explícitamente cuesta ~ 2 * n^3 FLOPs (el triple que una factorización PLU).
       - La factorización PLU requiere únicamente ~ (2/3) * n^3 FLOPs.
    3. Estabilidad y Propagación del Error:
       - El cálculo de A^-1 introduce errores de redondeo adicionales tanto en la inversión 
         como en el posterior producto matriz-vector (A^-1 @ b).
       - Resolver mediante sistemas triangulares (PLU) minimiza el número de condición efectivo 
         y preserva la estructura de la matriz (como la dispersión de ceros o matrices banda).
    4. Conclusión:
       Aunque la multiplicación (A^-1 @ b) es rápida una vez invertida la matriz, el costo y los 
       riesgos de inestabilidad numérica hacen que la factorización PLU sea el estándar universal 
       en software científico e ingeniería.
    """
    print(conclusion_tiempo)


# ==============================================================================
# FUNCIÓN PRINCIPAL
# ==============================================================================

def main():
    linesPrintsHashtags()
    print("# RESOLUCIÓN COMPLETA: TALLER 2 - FACTORIZACIÓN PLU Y MODELO ECONÓMICO")
    linesPrintsHashtags()
    
    # 1. Actividad 1: Explicación de pivoteo parcial
    mostrar_actividad_1(MATRIZ_A_ECONOMIA)
    
    # 2. Actividades 2 y 3: Factorización y verificación numérica
    P, L, U = ejecutar_actividades_2_y_3(MATRIZ_A_ECONOMIA)
    
    # 3. Actividades 4, 5 y 6: Escenarios económicos y residuos
    escenarios, soluciones_x, residuos_normas = ejecutar_actividades_4_5_6(
        MATRIZ_A_ECONOMIA, P, L, U
    )
    
    # 4. Sección 3: Análisis económico detallado
    ejecutar_analisis_economico(MATRIZ_A_ECONOMIA, escenarios, soluciones_x)
    
    # 5. Sección 4: Análisis de tiempo de ejecución
    ejecutar_analisis_tiempo(MATRIZ_A_ECONOMIA)
    
    linesPrintsHashtags()
    print("¡TALLER 2 COMPLETADO EXITOSAMENTE!")
    linesPrintsHashtags()


if __name__ == "__main__":
    main()
