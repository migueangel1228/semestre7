"""
================================================================================
TALLER 3: FACTORIZACIÓN QR, BASES, ORTOGONALIDAD Y PROCESO DE GRAM-SCHMIDT
Curso: Computación Científica / Métodos Numéricos
Nivel: Universitario

Descripción general:
Este archivo contiene la solución completa, estructurada y pedagógica del 
Taller 3 (Taller Teórico-Práctico de Factorización QR). Incluye:
1. Ejercicio 1: Bases, columnas e invertibilidad.
   - Análisis de independencia lineal, cálculo de rango y determinante.
   - Demostración de relaciones de equivalencia en matrices invertibles.
   - Interpretación de Ax = b como cambio de base y coordenadas.
2. Ejercicio 2: Proyecciones y proceso de Gram-Schmidt.
   - Ortogonalización y normalización paso a paso en R^2.
   - Demostración algebraica de ortogonalidad q1^T v2 = 0.
   - Extensión al caso 3x3 para la matriz A = [[1,1,1],[1,1,0],[1,0,0]].
3. Ejercicio 3: Matrices ortogonales e isometrías.
   - Propiedades algebraicas: Q^T Q = I ==> Q^-1 = Q^T.
   - Demostración de conservación de norma: ||Qx||_2 = ||x||_2.
   - Justificación de por qué una matriz ortogonal no amplifica errores.
4. Ejercicio 4: Construcción y estructura de la matriz R.
   - Deducción analítica de r_kj = q_k^T a_j.
   - Demostración de por qué R es estrictamente triangular superior.
   - Verificación de la identidad A = QR.
5. Ejercicio 5: QR frente a LU y estabilidad numérica.
   - Tabla comparativa multidimensional (LU vs QR).
   - Análisis de perturbaciones h y propagación de errores: ||Qh||_2 vs ||Ah||_2.
   - Cadena conceptual completa y reflexión epistemológica.
================================================================================
"""

import numpy as np

# Configuración de visualización de NumPy para salida clara
np.set_printoptions(precision=4, suppress=True)


# ==============================================================================
# FUNCIONES AUXILIARES DE FORMATO
# ==============================================================================

def linesPrintsHashtags():
    print("\n" + "#" * 80)

def linesPrintsEquals():
    print("=" * 80)


# ==============================================================================
# ALGORITMOS NÚCLEO: GRAM-SCHMIDT Y FACTORIZACIÓN QR
# ==============================================================================

def gram_schmidt_qr(A):
    """
    Calcula la factorización QR de una matriz A mediante el proceso de 
    Gram-Schmidt (Ortogonalización y Normalización por columnas).
    
    Parámetros:
        A : numpy.ndarray de dimensiones (m, n) con columnas lin. independientes.
        
    Retorna:
        Q : Matriz con columnas ortonormales (m, n), tal que Q^T @ Q = I.
        R : Matriz triangular superior (n, n), con r_jj > 0 y r_ij = q_i^T a_j.
        
    Algoritmo:
    Para cada columna j de 0 a n-1:
      1. Se inicializa el vector ortogonal v_j = a_j.
      2. Para cada i de 0 a j-1 (direcciones ortogonales previas):
           r_ij = q_i^T @ a_j   (coeficiente de proyección)
           v_j = v_j - r_ij * q_i (resta de la componente paralela a q_i)
      3. Se calcula la norma del vector ortogonal resultante:
           r_jj = ||v_j||_2
      4. Se normaliza para obtener el j-ésimo vector unitario:
           q_j = v_j / r_jj
    """
    A = np.array(A, dtype=float)
    m, n = A.shape
    Q = np.zeros((m, n), dtype=float)
    R = np.zeros((n, n), dtype=float)
    
    for j in range(n):
        v = A[:, j].copy()
        for i in range(j):
            # Coordenada de a_j en la dirección q_i
            R[i, j] = np.dot(Q[:, i], A[:, j])
            # Eliminación de la proyección sobre q_i
            v -= R[i, j] * Q[:, i]
            
        # Norma del vector perpendicular
        norma_v = np.linalg.norm(v)
        if np.isclose(norma_v, 0.0):
            raise ValueError(f"Las columnas de A son linealmente dependientes en la columna {j}.")
            
        R[j, j] = norma_v
        Q[:, j] = v / norma_v
        
    return Q, R


def resolver_sistema_qr(Q, R, b):
    """
    Resuelve el sistema Ax = b mediante factorización QR:
      A x = b  ==>  Q R x = b  ==>  R x = Q^T b
      
    Como R es triangular superior, se resuelve mediante sustitución hacia atrás.
    """
    b_transf = Q.T @ b
    n = len(b_transf)
    x = np.zeros(n, dtype=float)
    
    for i in range(n - 1, -1, -1):
        suma = sum(R[i, k] * x[k] for k in range(i + 1, n))
        x[i] = (b_transf[i] - suma) / R[i, i]
        
    return x, b_transf


# ==============================================================================
# EJERCICIO 1: BASES, COLUMNAS E INVERTIBILIDAD
# ==============================================================================

def ejecutar_ejercicio_1():
    linesPrintsEquals()
    print("EJERCICIO 1: BASES, COLUMNAS E INVERTIBILIDAD")
    linesPrintsEquals()
    
    A = np.array([
        [1.0, 2.0, 0.0],
        [0.0, 1.0, 1.0],
        [1.0, 3.0, 1.0]
    ])
    
    print("\nMatriz A analizada:")
    print(A)
    
    # a) Independencia lineal de las columnas
    # Verificamos si existe combinación no trivial: c1*a1 + c2*a2 + c3*a3 = 0
    # Notamos que: 2*a1 - a2 + a3 = 2*[1,0,1]^T - [2,1,3]^T + [0,1,1]^T = [0,0,0]^T
    det_A = np.linalg.det(A)
    rango_A = np.linalg.matrix_rank(A)
    
    print("\n--- Literal a) Independencia Lineal de las Columnas ---")
    print("Observamos las columnas de A:")
    print(f"  a1 = {A[:, 0]},  a2 = {A[:, 1]},  a3 = {A[:, 2]}")
    print("Combinación lineal encontrada:")
    print("  2*a1 - a2 + a3 = 2*[1,0,1]^T - [2,1,3]^T + [0,1,1]^T = [0,0,0]^T")
    print("-> Conclusión: Las columnas de A son LINEALMENTE DEPENDIENTES.")
    
    print("\n--- Literal b) Rango de la Matriz A ---")
    print(f"Rango calculado rank(A) = {rango_A}")
    print("-> Justificación: Solo hay 2 columnas linealmente independientes (por ejemplo {a1, a3}).")
    
    print("\n--- Literal c) Invertibilidad de A ---")
    print(f"Determinante det(A) = {det_A:.4f}")
    print("-> Justificación: Como det(A) = 0 (y rank(A) = 2 < 3), A NO es invertible (es singular).")
    
    print("\n--- Literal d) Relación entre las afirmaciones ---")
    resp_1d = """
    Las tres afirmaciones son EQUIVALENTES (Teorema de la Matriz Invertible en R^n):
      [A es invertible] <===> [rank(A) = n = 3] <===> [Las columnas de A son lin. independientes]
    Si una de ellas se cumple, las demás se cumplen automáticamente. Si una falla (como en 
    nuestro caso, donde rank(A)=2), todas las demás fallan simultáneamente.
    """
    print(resp_1d)
    
    print("--- Literales e) y f) Existencia, Unicidad e Interpretación de Coordenadas ---")
    resp_1ef = """
    - Literal e): Si las columnas {a1, a2, a3} forman una base de R^3, por definición de base:
      1) Generan todo R^3: Todo vector b se puede escribir como combinación lineal Ax = b (EXISTENCIA).
      2) Son linealmente independientes: Los coeficientes x = [x1, x2, x3]^T son únicos (UNICIDAD).
      
    - Literal f): En la ecuación Ax = b:
          x1 * a1 + x2 * a2 + x3 * a3 = b
      El vector 'x' representa las COORDENADAS del vector 'b' en el sistema de referencia (base)
      formado por las columnas de la matriz A.
    """
    print(resp_1ef)
    
    print("--- Pregunta Conceptual del Ejercicio 1 ---")
    resp_1_concept = """
    ¿Por qué la factorización QR parte naturalmente de las columnas de una matriz?
    Porque una matriz A = [a1 | a2 | ... | an] se interpreta geométricamente como un conjunto 
    de vectores columna que generan un subespacio. La factorización QR consiste en transformar 
    dicha base de columnas {a1, ..., an} (posiblemente oblicua o mal condicionada) en una 
    base ortonormal {q1, ..., qn}, preservando los subespacios anidados:
        Span(a1, ..., ak) = Span(q1, ..., qk) para todo k.
    """
    print(resp_1_concept)


# ==============================================================================
# EJERCICIO 2: PROYECCIONES Y PROCESO DE GRAM-SCHMIDT
# ==============================================================================

def ejecutar_ejercicio_2():
    linesPrintsEquals()
    print("EJERCICIO 2: PROYECCIONES Y PROCESO DE GRAM-SCHMIDT")
    linesPrintsEquals()
    
    # Paso a paso en R^2 con a1 = [1, 2]^T, a2 = [2, 1]^T
    a1 = np.array([1.0, 2.0])
    a2 = np.array([2.0, 1.0])
    
    print("\nVectores dados en R^2:")
    print(f"  a1 = {a1},  a2 = {a2}")
    
    # a) Norma de a1
    norma_a1 = np.linalg.norm(a1)
    print(f"\n--- a) Norma ||a1||: sqrt(1^2 + 2^2) = sqrt(5) ≈ {norma_a1:.4f}")
    
    # b) Primer vector unitario q1
    q1 = a1 / norma_a1
    print(f"--- b) Vector unitario q1 = a1 / ||a1|| = [1/sqrt(5), 2/sqrt(5)]^T ≈ {q1}")
    
    # c) Proyección de a2 sobre q1
    q1_dot_a2 = np.dot(q1, a2)  # (1*2 + 2*1)/sqrt(5) = 4/sqrt(5)
    proj_a2_q1 = q1_dot_a2 * q1 # (4/sqrt(5)) * (1/sqrt(5)*[1,2]) = [4/5, 8/5] = [0.8, 1.6]
    print(f"--- c) Proyección proj_q1(a2) = (q1^T a2)*q1:")
    print(f"       q1^T a2 = {q1_dot_a2:.4f} (4/sqrt(5))")
    print(f"       proj_q1(a2) = {proj_a2_q1} ([4/5, 8/5]^T)")
    
    # d) Vector ortogonal v2
    v2 = a2 - proj_a2_q1  # [2, 1] - [0.8, 1.6] = [1.2, -0.6] = [6/5, -3/5]
    print(f"--- d) Vector ortogonal v2 = a2 - proj_q1(a2) = {v2} ([6/5, -3/5]^T)")
    
    # e) Demostración de ortogonalidad q1^T v2 = 0
    prod_ortogonal = np.dot(q1, v2)
    print(f"--- e) Verificación de ortogonalidad q1^T v2 = {prod_ortogonal:.4e}")
    print(f"       Demostración analítica: (1/sqrt(5))*(1*(6/5) + 2*(-3/5)) = (6/5 - 6/5)/sqrt(5) = 0.")
    
    # f) Normalización de v2 para obtener q2
    norma_v2 = np.linalg.norm(v2)  # sqrt(36/25 + 9/25) = sqrt(45/25) = 3/sqrt(5)
    q2 = v2 / norma_v2  # [2/sqrt(5), -1/sqrt(5)]
    print(f"--- f) Vector unitario q2 = v2 / ||v2|| = {q2} ([2/sqrt(5), -1/sqrt(5)]^T)")
    
    # g) Verificación de base ortonormal
    print(f"--- g) Verificación de base ortonormal {{q1, q2}}:")
    print(f"       * ||q1||_2 = {np.linalg.norm(q1):.4f}")
    print(f"       * ||q2||_2 = {np.linalg.norm(q2):.4f}")
    print(f"       * q1^T q2  = {np.dot(q1, q2):.4e} (0.0)")
    
    # h) Proceso para la matriz A 3x3
    print("\n" + "-" * 80)
    print("--- h) Proceso de Gram-Schmidt para la Matriz A (3x3) ---")
    A_3x3 = np.array([
        [1.0, 1.0, 1.0],
        [1.0, 1.0, 0.0],
        [1.0, 0.0, 0.0]
    ])
    print("Matriz A (3x3):")
    print(A_3x3)
    
    Q_3x3, R_3x3 = gram_schmidt_qr(A_3x3)
    
    print("\nMatriz Q obtenida (Base Ortonormal):")
    print(Q_3x3)
    print("\nMatriz R obtenida (Triangular Superior):")
    print(R_3x3)
    
    print("\nVerificación de ortogonalidad Q^T @ Q = I:")
    print(Q_3x3.T @ Q_3x3)
    print("Error de ortogonalidad ||Q^T Q - I||_2:", np.linalg.norm(Q_3x3.T @ Q_3x3 - np.eye(3)))
    
    print("\nVerificación de reconstrucción Q @ R = A:")
    print(Q_3x3 @ R_3x3)
    print("Error de reconstrucción ||A - QR||_2:", np.linalg.norm(A_3x3 - Q_3x3 @ R_3x3))
    
    print("\n--- Pregunta Conceptual del Ejercicio 2 ---")
    resp_2_concept = """
    ¿Qué está haciendo geométricamente Gram-Schmidt al calcular v2 = a2 - (q1^T a2)q1?
    Geométricamente, descompone el vector a2 en dos componentes mutuamente perpendiculares:
      1. La sombra o proyección paralela a la dirección de q1:  (q1^T a2)q1.
      2. La componente residual o perpendicular a q1:           v2.
    Al restar la proyección, se 'elimina' toda la información que a2 compartía con q1, 
    obteniendo un vector v2 que apunta en una dimensión completamente nueva y ortogonal.
    """
    print(resp_2_concept)


# ==============================================================================
# EJERCICIO 3: MATRICES ORTOGONALES
# ==============================================================================

def ejecutar_ejercicio_3():
    linesPrintsEquals()
    print("EJERCICIO 3: MATRICES ORTOGONALES Y PROPIEDADES")
    linesPrintsEquals()
    
    resp_3_abc = """
    --- Literales a), b) y c) ---
    a) Condición de Ortonormalidad:
       Los vectores q_i, q_j deben cumplir:
         q_i^T q_j = 1  si i == j  (vectores unitarios, ||q_i|| = 1)
         q_i^T q_j = 0  si i != j  (vectores ortogonales entre sí)
         
    b) Demostración de Q^T @ Q = I:
       El elemento en la posición (i, j) del producto Q^T @ Q es exactamente el producto 
       punto entre la fila i de Q^T (es decir, el vector q_i) y la columna j de Q (el vector q_j):
           (Q^T @ Q)_{i,j} = q_i^T q_j = delta_{i,j}
       Por tanto, todas las entradas diagonales son 1 y las fuera de la diagonal son 0: Q^T Q = I.
       
    c) Demostración de Q^-1 = Q^T:
       Para cualquier matriz cuadrada n x n que cumpla Q^T @ Q = I, por unicidad de la matriz 
       inversa bilateral, se deduce directamente que:
           Q^-1 = Q^T.
    """
    print(resp_3_abc)
    
    # Demostración numérica y formal de la conservación de norma
    print("--- Literal d) Demostración de Conservación de Norma: ||Qx||_2 = ||x||_2 ---")
    resp_3d = """
    Demostración analítica:
      ||Qx||_2^2 = (Qx)^T (Qx)
                 = x^T (Q^T Q) x
                 = x^T (I) x
                 = x^T x
                 = ||x||_2^2
      Tomando la raíz cuadrada positiva en ambos lados:
          ||Qx||_2 = ||x||_2.  (Q.E.D.)
    """
    print(resp_3d)
    
    # Verificación numérica en Python
    Q_ejemplo, _ = np.linalg.qr(np.random.randn(3, 3))
    x_test = np.array([3.0, -4.0, 5.0])
    Qx_test = Q_ejemplo @ x_test
    print(f"Ejemplo numérico con x = {x_test}:")
    print(f"  * ||x||_2   = {np.linalg.norm(x_test):.6f}")
    print(f"  * ||Qx||_2  = {np.linalg.norm(Qx_test):.6f}")
    print(f"  * Diferencia absoluta: {abs(np.linalg.norm(x_test) - np.linalg.norm(Qx_test)):.2e}")
    
    print("\n--- Literales e) y f) Interpretación Geométrica y No Amplificación ---")
    resp_3ef = """
    - Literal e): Que una transformación conserve la norma significa que es una ISOMETRÍA rígida
      (una rotación pura, una reflexión o una combinación de ambas). No deforma el espacio, no 
      estira ni encoge vectores y preserva todos los ángulos y distancias euclidianas.
      
    - Literal f): ¿Puede una matriz ortogonal amplificar la longitud de un vector?
      ¡NO, JAMÁS! La relación ||Qx||_2 / ||x||_2 = 1 es exactamente constante e igual a 1 para 
      cualquier vector x != 0. Su número de condición en norma 2 es óptimo: kappa_2(Q) = 1.
    """
    print(resp_3ef)
    
    print("--- Pregunta Conceptual del Ejercicio 3 ---")
    resp_3_concept = """
    ¿Por qué las matrices ortogonales son tan importantes en Computación Científica?
    1. Estabilidad Numérica Perfecta: Al multiplicar por Q, los errores de redondeo de punto 
       flotante no se amplifican en absoluto (factor de amplificación = 1).
    2. Eficiencia Extrema: Invertir Q no requiere costosos algoritmos de eliminación ni 
       corre riesgos de división por cero; basta con transponerla (Q^-1 = Q^T), lo cual 
       es una operación inmediata y exacta de O(n^2).
    """
    print(resp_3_concept)


# ==============================================================================
# EJERCICIO 4: CONSTRUCCIÓN DE LA MATRIZ R
# ==============================================================================

def ejecutar_ejercicio_4():
    linesPrintsEquals()
    print("EJERCICIO 4: CONSTRUCCIÓN Y ESTRUCTURA DE LA MATRIZ R")
    linesPrintsEquals()
    
    resp_4 = """
    --- Literales a), b), c), d), e) y f) ---
    
    a) ¿Por qué a2 no necesita componente en la dirección de q3?
       Porque q1 y q2 se construyen a partir de {a1, a2}. El espacio generado Span(q1, q2) es 
       idéntico al plano Span(a1, a2). Como a2 pertenece por completo a este plano, su 
       proyección sobre cualquier dirección ortogonal al plano (como q3) es idénticamente 0.
       
    b) ¿Por qué a3 puede escribirse utilizando q1, q2 y q3?
       Porque {q1, q2, q3} es una base ortonormal completa del subespacio tridimensional generado 
       por {a1, a2, a3} (de hecho, es una base de todo R^3).
       
    c) Demostración de r_kj = q_k^T a_j:
       Partiendo de la combinación lineal:
           a_j = sum_{i=1}^j (r_ij * q_i)
       Multiplicamos por la izquierda por q_k^T:
           q_k^T a_j = sum_{i=1}^j r_ij * (q_k^T q_i)
       Por la ortonormalidad de la base, q_k^T q_i = 1 si i == k y 0 si i != k:
           q_k^T a_j = r_kj   (para k <= j).
       Si k > j, como a_j no tiene componente en q_k, q_k^T a_j = 0. (Q.E.D.)
       
    d) Estructura de la Matriz R:
           R = [[r11, r12, r13],
                [ 0 , r22, r23],
                [ 0 ,  0 , r33]]
                
    e) ¿Por qué R es triangular superior?
       Porque cada columna a_j de la matriz original solo involucra las primeras 'j' direcciones 
       ortonormales {q1, ..., qj} generadas hasta ese paso del algoritmo. Todos los coeficientes 
       r_ij con i > j son estrictamente cero.
       
    f) Demostración de A = QR:
       Por el producto bloque de matrices:
           Q @ R = [q1 | q2 | q3] @ [[r11, r12, r13],
                                    [ 0 , r22, r23],
                                    [ 0 ,  0 , r33]]
                 = [ r11*q1 | r12*q1 + r22*q2 | r13*q1 + r23*q2 + r33*q3 ]
                 = [ a1     | a2              | a3                       ] = A. (Q.E.D.)
    """
    print(resp_4)


# ==============================================================================
# EJERCICIO 5: QR FRENTE A LU Y ESTABILIDAD NUMÉRICA
# ==============================================================================

def ejecutar_ejercicio_5():
    linesPrintsEquals()
    print("EJERCICIO 5: COMPARACIÓN QR vs LU Y ESTABILIDAD NUMÉRICA")
    linesPrintsEquals()
    
    tabla_comparativa = """
    +---------------------------+-----------------------------------+-----------------------------------+
    | CARACTERÍSTICA            | FACTORIZACIÓN LU                  | FACTORIZACIÓN QR                  |
    +---------------------------+-----------------------------------+-----------------------------------+
    | Primer factor             | L (Triangular Inferior Unitaria)  | Q (Matriz Ortogonal / Isometría)  |
    | Segundo factor            | U (Triangular Superior)           | R (Triangular Superior)           |
    | Idea principal            | Eliminación Gaussiana por filas   | Ortogonalización por columnas     |
    | ¿Utiliza ortogonalidad?   | No                                | Sí (Garantiza Q^T Q = I)          |
    | ¿Conserva normas?         | No (Puede amplificar errores)     | Sí (||Qx||_2 = ||x||_2)           |
    | Estructura triangular     | Dos factores triangulares (L, U)  | Un factor triangular (R)          |
    | Costo computacional       | ~ (2/3) n^3 FLOPs                 | ~ (4/3) n^3 FLOPs (el doble de LU)|
    | Aplicación típica         | Sistemas cuadrados exactos Ax=b   | Mínimos Cuadrados, Autovalores QR |
    +---------------------------+-----------------------------------+-----------------------------------+
    """
    print(tabla_comparativa)
    
    print("--- Análisis de Perturbaciones y Propagación de Errores ---")
    resp_5_abcde = """
    a) Propiedad de Q: Es una matriz ortogonal (Q^T Q = I, columnas ortonormales).
    b) Conveniencia de Q^-1 = Q^T: Invertir matrices ortogonales es instantáneo y exacto (transposición).
    c) Conservación de Norma: ||Qx||_2 = ||x||_2 porque (Qx)^T (Qx) = x^T Q^T Q x = x^T x.
    
    d) Perturbación con Matriz Ortogonal Q:
       ||Q(x + h) - Qx||_2 = ||Qh||_2 = ||h||_2.
       -> El error de salida es EXACTAMENTE IGUAL al error de entrada. Cero amplificación.
       
    e) Perturbación con Matriz General A:
       A(x + h) - Ax = Ah.
       En general, ||Ah||_2 != ||h||_2. El error puede verse atenuado o severamente amplificado 
       dependiendo del condicionamiento y los valores singulares de A:
           sigma_min(A) * ||h||_2 <= ||Ah||_2 <= sigma_max(A) * ||h||_2.
           
    f) Si ||Ah||_2 > ||h||_2:
       Significa que la matriz está AMPLIFICANDO los errores de medición o redondeo. En matrices 
       mal condicionadas (casi singulares), esta amplificación puede destruir por completo la 
       precisión de los resultados. Frente a esto, una matriz ortogonal Q mantiene el factor 
       de amplificación estrictamente fijo en 1.
       
    g) Relación entre Ortogonalidad y Estabilidad Numérica:
       La ortogonalidad es el estándar de oro en estabilidad numérica. Cualquier algoritmo que 
       utilice transformaciones ortogonales (como QR, Householder, Givens o SVD) es incondicionalmente 
       estable, ya que no introduce crecimiento artificial de normas ni propagación descontrolada 
       de errores de redondeo.
    """
    print(resp_5_abcde)
    
    # Experimento numérico de perturbación
    print("-" * 80)
    print("Demostración Computacional de Propagación de Perturbaciones:")
    np.random.seed(42)
    n = 4
    Q_sim, _ = np.linalg.qr(np.random.randn(n, n))
    # Creamos una matriz A con alto número de condición
    A_ill = np.array([
        [1.0, 1.0, 1.0, 1.0],
        [1.0, 1.001, 1.0, 1.0],
        [1.0, 1.0, 1.001, 1.0],
        [1.0, 1.0, 1.0, 1.001]
    ])
    
    h = np.random.randn(n) * 1e-3
    norma_h = np.linalg.norm(h)
    norma_Qh = np.linalg.norm(Q_sim @ h)
    norma_Ah = np.linalg.norm(A_ill @ h)
    
    print(f"  * Norma de la perturbación original ||h||_2:     {norma_h:.6e}")
    print(f"  * Error propagado por matriz ortogonal ||Qh||_2: {norma_Qh:.6e} (Ratio ||Qh||/||h|| = {norma_Qh/norma_h:.6f})")
    print(f"  * Error propagado por matriz general   ||Ah||_2: {norma_Ah:.6e} (Ratio ||Ah||/||h|| = {norma_Ah/norma_h:.6f})")
    
    print("\n--- Pregunta de Cierre: Cadena Conceptual ---")
    cadena_cierre = """
    Base {a1, ..., an} 
      ==> Proyección (q_i^T a_j) q_i 
      ==> Gram-Schmidt (Resta de componentes paralelas y normalización)
      ==> Q (Matriz con base ortonormal como columnas)
      ==> R (Matriz triangular con los coeficientes de proyección)
      ==> A = Q @ R (Factorización completada)
    """
    print(cadena_cierre)
    
    print("--- Reflexión Final ---")
    reflexion_final = """
    REFLEXIÓN EPISTEMOLÓGICA:
    La factorización QR puede entenderse fundamentalmente como un proceso geométrico de cambio 
    de base: toma las columnas originales de la matriz A (que representan un sistema de ejes 
    oblicuo, posiblemente redundante o deformado) y las transforma en un sistema de ejes 
    perfectamente perpendiculares y unitarios (la matriz ortogonal Q). La matriz triangular 
    superior R actúa como el 'mapa de reconstrucción', guardando exactamente las coordenadas y 
    pesos de proyección necesarios para regresar de la base ortonormal ideal a las columnas originales.
    """
    print(reflexion_final)


# ==============================================================================
# BLOQUE DE EJECUCIÓN PRINCIPAL
# ==============================================================================

def main():
    linesPrintsHashtags()
    print("# RESOLUCIÓN COMPLETA: TALLER 3 - FACTORIZACIÓN QR Y GRAM-SCHMIDT")
    linesPrintsHashtags()
    
    # Ejercicio 1: Bases, columnas e invertibilidad
    ejecutar_ejercicio_1()
    
    # Ejercicio 2: Proyecciones y proceso de Gram-Schmidt
    ejecutar_ejercicio_2()
    
    # Ejercicio 3: Matrices ortogonales
    ejecutar_ejercicio_3()
    
    # Ejercicio 4: Construcción de la matriz R
    ejecutar_ejercicio_4()
    
    # Ejercicio 5: QR vs LU y estabilidad numérica
    ejecutar_ejercicio_5()
    
    linesPrintsHashtags()
    print("¡TALLER 3 COMPLETADO EXITOSAMENTE!")
    linesPrintsHashtags()


if __name__ == "__main__":
    main()
