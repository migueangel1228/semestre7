"""
Ejercicio - Parcial 1 - Computación Científica
Integrantes: 
1) Maria Lucía Castillo García
2) Juliana González Sánchez
3) Miguel Ángel Padilla Rosero

- Profesor: Juan Fernando Paz 


Notas sobre el formato de los print: 
En los f-string, lo que va después de los dos puntos índica CÓMO se escribe el valor. Se usan dos formas en este archivo:

{texto:<6}      rellena con espacios a la derecha hasta ocupar 6 caracteres. El símbolo < es alineacion a la izquierda. Sirve para alinear columnas.
{numero:.3e}    escribe el número en notación científica con 3 decimales.   Ejemplo: 4.864e+17
--> La notación científica se usa para los errores y los residuos porque
son números de 20 o más cifras, imposibles de leer de otro modo.

"""

### ==========================================================
### LIBRERÍAS UTILIZADAS
### ==========================================================


import numpy as np
import time # mide cuanto tarda en ejecutarse cada método
import matplotlib.pyplot as plt #librería de graficas


### ==========================================================
### CONFIGURACIÓN INICIAL
### ==========================================================


np.set_printoptions(precision=2) # Limita a 2 los decimales al imprimir arreglos de numpy

## Le indica a numpy que no interrumpa ni avise cuando un calculo se
## desborde. Los desbordamientos se detectan a proposito en el
## diagnostico, revisando si aparecen valores infinitos.
np.seterr(all="ignore")

## Los datos del enunciado estan en datos.py, que debe estar en la
## misma carpeta que este archivo.
from datos import x20, y20, x200, y200


### ==========================================================
### MATRIZ DE VANDERMONDE
### ==========================================================

def construirVandermonde(x):

    ## Cada fila i son las potencias de x[i], desde x^0 hasta x^(n-1).
    ## Se acumulan multiplicando en float64 para que un desbordamiento
    ## quede registrado como infinito y pueda detectarse en el diagnóstico.
    tam = len(x)
    V = np.zeros((tam, tam))
    for i in range(tam):
        potencia = np.float64(1.0)
        for j in range(tam):
            V[i][j] = potencia
            potencia = potencia * np.float64(x[i])
    return V


### ==========================================================
### MÉTODO DE CROUT
### ==========================================================

def crout(A):
    ## Llenar matrices L y U
    tam = len(A)
    L = np.zeros((tam, tam))  ## Una forma de llenar la matriz de ceros en numpy
    U = np.eye(tam)           ## Una manera de hacer una matriz identidad

    # i = fila de L
    # j = etapa actual (columna de L y fila de U)
    # k1, r = indices de sumatoria
    # k2 = columna de U

    for j in range(tam):
        ## Construir columna j de L
        for i in range(j, tam):
            sumaColumna = 0
            for k1 in range(0, j):
                sumaColumna += L[i][k1] * U[k1][j]
            L[i][j] = A[i][j] - sumaColumna

        ## Construir fila j de U
        for k2 in range(j + 1, tam):
            sumaFila = 0
            for r in range(0, j):
                sumaFila += L[j][r] * U[r][k2]
            U[j][k2] = (A[j][k2] - sumaFila) / L[j][j]
    return L, U


### ==========================================================
### SUSTITUCION HACIA ADELANTE
### ==========================================================

def sustitucionAdelante(L, b):
    tamL = len(L)
    tamB = len(b)
    Y = np.zeros((tamB))

    for i in range(tamL):
        suma = 0
        for k in range(i):
            suma += L[i][k] * Y[k]
        Y[i] = (b[i] - suma) / L[i][i]
    return Y


### ==========================================================
### SUSTITUCION HACIA ATRÁS
### ==========================================================

def sustitucionAtras(U, y):
    tamU = len(U)
    X = np.zeros((tamU))

    for i in range((tamU - 1), -1, -1):
        suma = 0
        for k in range(i + 1, tamU):
            suma += U[i][k] * X[k]
        X[i] = (y[i] - suma) / U[i][i]
    return X


### ==========================================================
### PROCESO DE GRAM-SCHMIDT
### ==========================================================

def gramSchmidt(A):

    ## Toma las columnas de A y las convierte en columnas ortonormales
    tam = len(A)
    Q = np.zeros((tam, tam))
    R = np.zeros((tam, tam))

    # j = columna que se esta procesando
    # i = columna ya construida sobre la que se proyecta
    # k = indice para recorrer las componentes del vector

    for j in range(tam):
        ## Copiar la columna j de A
        v = np.zeros((tam))
        for k in range(tam):
            v[k] = A[k][j]

        ## Restarle sus proyecciones sobre las columnas anteriores de Q
        for i in range(j):
            productoPunto = 0
            for k in range(tam):
                productoPunto += A[k][j] * Q[k][i]
            R[i][j] = productoPunto
            for k in range(tam):
                v[k] = v[k] - R[i][j] * Q[k][i]

        ## La norma de lo que sobro va a la diagonal de R
        sumaCuadrados = 0
        for k in range(tam):
            sumaCuadrados += v[k] ** 2
        R[j][j] = sumaCuadrados ** 0.5

        ## Normalizar para obtener la columna j de Q
        for k in range(tam):
            Q[k][j] = v[k] / R[j][j]

    return Q, R


### ==========================================================
### PRODUCTO DE Q TRANSPUESTA POR UN VECTOR
### ==========================================================

def calcularQtPorVector(Q, y):
    ## Cada componente es el producto punto de una columna de Q con y
    tam = len(Q)
    c = np.zeros((tam))

    for j in range(tam):
        suma = 0
        for k in range(tam):
            suma += Q[k][j] * y[k]
        c[j] = suma
    return c



### ==========================================================
### RESOLVER EL SISTEMA CON LU
### ==========================================================

def resolverConLU(V, y):

    ## perf_counter() devuelve un numero de segundos que solo sirve para
    ## restarlo con otra medicion. Se toma una marca antes de empezar y
    ## otra al terminar, de modo que la diferencia es lo que tardo el
    ## metodo completo: la factorizacion mas las sustituciones.
    tiempoInicio = time.perf_counter()
    L, U = crout(V)                                  ## Factorizar V = L * U
    yIntermedio = sustitucionAdelante(L, y)          ## Resolver L * y' = y
    coeficientes = sustitucionAtras(U, yIntermedio)  ## Resolver U * a = y'
    tiempoFin = time.perf_counter()
    return coeficientes, (tiempoFin - tiempoInicio)


### ==========================================================
### RESOLVER EL SISTEMA CON QR
### ==========================================================

def resolverConQR(V, y):
    ## Misma medicion de tiempo que en LU, para poder compararlos
    tiempoInicio = time.perf_counter()
    Q, R = gramSchmidt(V)                  ## Factorizar V = Q * R
    c = calcularQtPorVector(Q, y)          ## Calcular Qt * y
    coeficientes = sustitucionAtras(R, c)  ## Resolver R * a = Qt * y
    tiempoFin = time.perf_counter()

    return coeficientes, (tiempoFin - tiempoInicio), Q, R



### ==========================================================
### EVALUAR EL POLINOMIO EN UN PUNTO
### ==========================================================

def evaluarPolinomio(coeficientes, valorX):
    ## Calcula p(x) = a0 + a1*x + a2*x^2 + ... + a(n-1)*x^(n-1),
    ## Misma expresion del enunciado, término por término.
    resultado = 0.0
    for k in range(len(coeficientes)):
        resultado += coeficientes[k] * (valorX ** k)
    return resultado



### ==========================================================
### MEDIDAS DE ERROR
### ==========================================================

def calcularResiduo(V, coeficientes, y):
    ## El residuo mide que tanto se aleja V*a de y. Si la solución fuera perfecta, este vector seria de puros ceros.
    vectorResiduo = (V @ coeficientes) - y
    return np.linalg.norm(vectorResiduo)


def calcularErrorRelativo(valorObtenido, valorEsperado):
    return abs(valorObtenido - valorEsperado) / abs(valorEsperado)


def medirPerdidaOrtogonalidad(Q):
    tam = len(Q)
    return np.linalg.norm((Q.T @ Q) - np.eye(tam))




### ==========================================================
### DIAGNÓSTICO 1: los datos y[i] que entrega el enunciado
### ==========================================================

def revisarDatosOriginales(x, y):

    ## El polinomio real es p(t) = 1 + t + t^2 + ... + t^(n-1),
    ## porque p(0) = 1 y p(1) = n, que son los valores que pide el
    ## enunciado. Se recalcula con enteros de Python, que no tienen
    ## límite de tamaño, y se compara contra los datos entregados.

    tam = len(x)
    correctos = 0
    primerFallo = -1
    for i in range(tam):
        valorExacto = 0
        for k in range(tam):
            valorExacto += int(x[i]) ** k
        if valorExacto == int(y[i]):
            correctos += 1
        elif primerFallo == -1:
            primerFallo = i
    return correctos, primerFallo


### ==========================================================
### DIAGNÓSTICO 2: la matriz de Vandermonde en float64
### ==========================================================

def revisarVandermonde(x):

    tam = len(x)
    mayorX = max(x)

    ## np.log10 calcula el logaritmo en base 10, es decir, cuantos ceros
    ## tiene aproximadamente un numero. Sirve para saber de que orden de
    ## magnitud es la potencia mas grande que hara falta, y compararla
    ## con el limite de float64, que es 10^308.
    ## Ejemplo: 38^19  ->  19 * log10(38)  =~ 30, o sea del orden de 10^30.
    exponente = (tam - 1) * np.log10(float(mayorX))

    V = construirVandermonde(x)

    ## np.isinf(V) devuelve una matriz de verdadero/falso indicando que
    ## entradas quedaron en infinito, y .sum() cuenta cuantos verdaderos
    ## hay. Si el resultado es 0, ninguna entrada se desbordo.
    cantidadInfinitos = int(np.isinf(V).sum())
    esFinita = (cantidadInfinitos == 0)

    if esFinita:
        numeroCondicion = np.linalg.cond(V) ## np.linalg.cond mide que tan cerca esta la matriz de ser singular (entre más grande el número, peor)
    else:
        numeroCondicion = float("inf")
    return V, esFinita, cantidadInfinitos, exponente, numeroCondicion



### ==========================================================
### GRAFICAR UN POLINOMIO
### ==========================================================

def graficarPolinomio(coeficientes, x, y, titulo):

    ## np.isfinite(...).all() revisa que TODOS los coeficientes sean
    ## números normales, es decir, que ninguno sea infinito ni inválido.
    ## Si alguno lo fuera, no habría nada que dibujar.
    if not np.isfinite(coeficientes).all():
        print(f"No se puede graficar {titulo}: los coeficientes no son finitos.")

    else:
        ## np.linspace genera 600 valores repartidos uniformemente entre
        ## el x mas pequeño y el más grande. Son los puntos donde se
        ## evaluara el polinomio para poder trazar la curva.
        malla = np.linspace(float(min(x)), float(max(x)), 600)

        valores = np.zeros(len(malla))
        for i in range(len(malla)):
            valores[i] = evaluarPolinomio(coeficientes, malla[i])

        plt.figure(figsize=(8, 5))   ## Tamaño de la figura, en pulgadas

        ## Traza la curva del polinomio obtenido
        plt.plot(malla, valores, linewidth=1.4, label="polinomio obtenido")

        ## Traza los puntos del enunciado. El "o" indica que se dibujen
        ## como circulos sueltos y no unidos por una linea.
        plt.plot(x, y, "o", markersize=5, label="puntos del enunciado")

        ## Escala del eje vertical. Los valores van desde 1 hasta 10^21,
        ## asi que en escala normal no se distinguiria nada. "symlog"
        ## comprime el eje de forma logaritmica y, a diferencia de la
        ## escala logaritmica comun, admite valores negativos y el cero.
        plt.yscale("symlog")

        plt.title(titulo)
        plt.xlabel("x")
        plt.ylabel("p(x)   (escala simlog)")
        plt.legend()          ## Recuadro con los nombres de cada trazo
        plt.grid(alpha=0.3)   ## Cuadricula de fondo, semitransparente
        plt.tight_layout()    ## Evita que los textos queden cortados
        plt.show()            ## Abre la ventana con la grafica



### ==========================================================
### MENÚ PRINCIPAL
### ==========================================================

def mostrarMenuPrincipal():
    print()
    print("====================================")
    print("INTERPOLACIÓN POLINÓMICA - LU vs QR")
    print("====================================")
    print("1. Diagnóstico previo")
    print("2. Polinomio P (grado 19)")
    print("3. Polinomio Q (grado 199)")
    print("4. Tabla comparativa")
    print("5. Graficar los polinomios")
    print("0. Salir")
    print()


### ==========================================================
### ACTIVIDAD 1: DIAGNÓSTICO PREVIO
### ==========================================================

def ejecutarDiagnostico():

    print()
    print("DIAGNÓSTICO PREVIO")
    print("==================")
    print()

    for nombre, x, y in [("P (20 puntos)", x20, y20), ("Q (200 puntos)", x200, y200)]:

        correctos, primerFallo = revisarDatosOriginales(x, y)
        V, esFinita, infinitos, exponente, condicion = revisarVandermonde(x)

        print(f"--- {nombre} ---")
        print(f"  Mayor valor de x            : {max(x)}")
        print(f"  Mayor potencia necesaria    : {max(x)}^{len(x)-1}  =~  10^{exponente:.1f}")
        print(f"  Límite de float64           : 10^308")
        print(f"  Matriz de Vandermonde finita: {esFinita}")

        if esFinita:
            print(f"  Número de condicion de V    : {condicion:.3e}")
        else:
            print(f"  Entradas infinitas en V     : {infinitos} de {V.size}")

        print(f"  Datos y[i] correctos        : {correctos} de {len(y)}")

        if primerFallo != -1:
            print(f"  Primer dato corrupto        : indice {primerFallo}, x = {x[primerFallo]}")

        print()

    print("Lectura del diagnóstico:")
    print("- El polinomio real es p(t) = 1 + t + t^2 + ... , por eso p(1) vale 20 y 200.")
    print("- Los y[i] se entregaron en int64 y se desbordaron: la mayoria ya viene mal.")
    print("- La matriz de Vandermonde esta pesimamente condicionada, y en el caso de")
    print("  200 puntos ni siquiera cabe en float64.")
    print("- Por tanto se espera que ningun metodo recupere los coeficientes correctos.")


### ==========================================================
### ACTIVIDAD 2 y 3: RESOLVER UN CASO
### ==========================================================

def ejecutarCaso(nombrePolinomio, x, y, valorEsperado):

    print()
    print(f"POLINOMIO {nombrePolinomio}  (grado {len(x)-1})")
    print("=" * 40)
    print()

    coefLU = None
    coefQR = None

    V, esFinita, infinitos, exponente, condicion = revisarVandermonde(x)

    if not esFinita:
        print("La matriz de Vandermonde no es representable en float64.")
        print(f"Se requiere {max(x)}^{len(x)-1} =~ 10^{exponente:.1f}, muy por encima de 10^308.")
        print(f"Quedan {infinitos} entradas infinitas de {V.size}.")
        print()
        print("No es posible factorizar ni con LU ni con QR: el sistema no llega")
        print("a plantearse en aritmética de punto flotante.")

    else:
        print(f"Numero de condición de V: {condicion:.3e}")
        print()

        yFlotante = np.asarray(y, dtype=float) # np.asarray convierte el arreglo de enteros a decimales (float64), porque toda la aritmetica de los metodos se hace con decimales.

        coefLU, tiempoLU = resolverConLU(V, yFlotante)
        coefQR, tiempoQR, Q, R = resolverConQR(V, yFlotante)

        for etiqueta, coeficientes, tiempo in [("LU", coefLU, tiempoLU),
                                               ("QR", coefQR, tiempoQR)]:
            valorEnUno = evaluarPolinomio(coeficientes, 1.0)
            print(f"  {etiqueta}")
            print(f"    Tiempo de ejecución : {tiempo:.6f} s")
            print(f"    Residuo ||V a - y|| : {calcularResiduo(V, coeficientes, yFlotante):.3e}")
            print(f"    {nombrePolinomio}(1) obtenido      : {valorEnUno:.6e}")
            print(f"    {nombrePolinomio}(1) esperado      : {valorEsperado}")
            print(f"    Error relativo      : {calcularErrorRelativo(valorEnUno, valorEsperado):.3e}")
            print()

        print(f"  Perdida de ortogonalidad en Q: ||QtQ - I|| = {medirPerdidaOrtogonalidad(Q):.3e}")
        print("  (Si Gram-Schmidt hubiera funcionado bien, este valor sería cercano a 0)")
        print()
    
        print("  Los coeficientes correctos son todos iguales a 1. Primeros seis obtenidos:")
        print(f"    LU: {np.array2string(coefLU[:6], precision=3)}") ## np.array2string convierte un arreglo en texto, para poder controlar cuantos decimales se muestran al imprimirlo
        print(f"    QR: {np.array2string(coefQR[:6], precision=3)}")
    return coefLU, coefQR


### ==========================================================
### ACTIVIDAD 4: TABLA COMPARATIVA
### ==========================================================

def ejecutarTablaComparativa():

    print()
    print("TABLA COMPARATIVA")
    print("=================")
    print()

    filas = []

    for nombre, x, y, esperado in [("P", x20, y20, 20), ("Q", x200, y200, 200)]:
        V, esFinita, infinitos, exponente, condicion = revisarVandermonde(x)
        if not esFinita:
            filas.append((nombre, "LU", "No aplica", "No aplica", "No aplica", "No aplica"))
            filas.append((nombre, "QR", "No aplica", "No aplica", "No aplica", "No aplica"))

        else:
            yFlotante = np.asarray(y, dtype=float) ## Convertir los datos a decimales para operar con ellos
            coefLU, tiempoLU = resolverConLU(V, yFlotante)
            coefQR, tiempoQR, Q, R = resolverConQR(V, yFlotante)

            for etiqueta, coeficientes, tiempo in [("LU", coefLU, tiempoLU),
                                                   ("QR", coefQR, tiempoQR)]:
                valorEnUno = evaluarPolinomio(coeficientes, 1.0)
                filas.append((
                    nombre,
                    etiqueta,
                    f"{tiempo:.6f}",
                    f"{calcularResiduo(V, coeficientes, yFlotante):.3e}",
                    f"{valorEnUno:.4e}",
                    f"{calcularErrorRelativo(valorEnUno, esperado):.3e}"
                ))

    ## Los anchos (6, 8, 14, 14, 16, 14) reservan el espacio de cada columna, de modo que el encabezado y las filas queden alineados
    print(f"{'Pol.':<6}{'Metodo':<8}{'Tiempo (s)':<14}{'Residuo':<14}{'Valor en 1':<16}{'Error rel.':<14}")
    print("-" * 72)
    for fila in filas:
        print(f"{fila[0]:<6}{fila[1]:<8}{fila[2]:<14}{fila[3]:<14}{fila[4]:<16}{fila[5]:<14}")


### ==========================================================
### ACTIVIDAD 5: GRÁFICAS
### ==========================================================

def ejecutarGraficas():
    print()
    print("Generando gráficas...")
    print()
    for nombre, x, y in [("P", x20, y20), ("Q", x200, y200)]:
        V, esFinita, infinitos, exponente, condicion = revisarVandermonde(x)
        if not esFinita:
            print(f"El polinomio {nombre} no pudo construirse, no hay nada que graficar.")

        else:
           
            yFlotante = np.asarray(y, dtype=float)  ## Convertir los datos a decimales para operar con ellos
            coefLU, tiempoLU = resolverConLU(V, yFlotante)
            coefQR, tiempoQR, Q, R = resolverConQR(V, yFlotante)
            graficarPolinomio(coefLU, x, yFlotante, f"Polinomio {nombre} obtenido con LU")
            graficarPolinomio(coefQR, x, yFlotante, f"Polinomio {nombre} obtenido con QR")



def main():
    continuar = True
    while continuar:
        mostrarMenuPrincipal()
        opcion = int(input("Seleccione una opción: "))
        if opcion == 1:
            ejecutarDiagnostico()
        elif opcion == 2:
            ejecutarCaso("P", x20, y20, 20)
        elif opcion == 3:
            ejecutarCaso("Q", x200, y200, 200)
        elif opcion == 4:
            ejecutarTablaComparativa()
        elif opcion == 5:
            ejecutarGraficas()
        elif opcion == 0:
            continuar = False
        else:
            print("\nOpcion inválida.\n")

main()