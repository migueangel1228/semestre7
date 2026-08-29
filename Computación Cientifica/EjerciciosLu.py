from sys import stdin
import numpy as np
np.set_printoptions(precision=2) ## Para establecer la cantidad de decimales desde el inicio

from scipy.linalg import lu_factor, lu_solve


### ==========================================================
### MÉTODO DE CROUT
### ==========================================================

def crout(A):

    ## Llenar matrices L y U
    tam = len(A)
    L = np.zeros((tam, tam))  ## Una forma de llenar la matriz de ceros en numpy
    U = np.eye(tam)  ## Una manera de hacer una matriz identidad 

    # i = fila de L
    # j = etapa actual (columna de L y fila de U)
    # k1, r = índices de sumatoria
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
### SUSTITUCIÓN HACIA ADELANTE
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
### SUSTITUCIÓN HACIA ATRÁS
### ==========================================================

def sustitucionAtras(U, y):
    tamU = len(U)
    X = np.zeros((tamU))
    
    for i in range((tamU-1), -1, -1 ):
        suma = 0
        for k in range(i+1, tamU):
            suma += U[i][k] * X[k]
        X[i] = (y[i] - suma) / U[i][i]
    return X


#### --------- ///////// --------- ///////// --------- ///////// --------- ///////
#### --------- ///////// --------- ///////// --------- ///////// --------- ///////
#### --------- ///////// --------- ///////// --------- ///////// --------- ///////


### ==========================================================
### MATRIZ CON DIAGONAL DOMINANTE ESTRICTA
### ==========================================================
def modificarADiagonalmenteDominanteEstricta(A):
    tamA = len(A)
    for i in range(tamA):
        suma = 0
        for k in range(tamA):
            if (k != i): 
                suma += abs(A[i][k]) 
        A[i][i] = suma + 1
    return A


"""
def normaEuclidiana(vector):
    suma = 0
    for i in range(len(vector)):
        suma += vector[i] ** 2
    norma = suma ** 0.5
    return norma
"""

#### --------- ///////// --------- ///////// --------- ///////// --------- ///////
#### --------- ///////// --------- ///////// --------- ///////// --------- ///////
#### --------- ///////// --------- ///////// --------- ///////// --------- ///////


### ==========================================================
### MENÚ PRINCIPAL
### ==========================================================
def mostrarMenuPrincipal():
    print()
    print("====================================")
    print("MENÚ PRINCIPAL")
    print("====================================")
    print("1. Ejercicio 1")
    print("2. Ejercicio 2")
    print("0. Salir")
    print()



### ==========================================================
### MENÚ EJERCICIO 1: Cantidad de productos generados
### ==========================================================
def mostrarMenuEjercicio1():
    print()
    print("====================================")
    print("EJERCICIO 1: Cantidad de productos")
    print("====================================")
    print("1. Actividad 1: Factorización LU")
    print("2. Actividad 2: Escenarios")
    print("3. Actividad 3: Explicación de eficiencia")
    print("4. Actividad 4: SciPy")
    print("0. Volver")
    print()


### ======================================================================
### MENÚ EJERCICIO 2: Estimación de emisiones contaminantes (red sensores)
### ======================================================================
def mostrarMenuEjercicio2():
    print()
    print("====================================")
    print("EJERCICIO 2: Red de sensores")
    print("====================================")
    print("1. Actividad 1")
    print("2. Actividad 2")
    print("3. Actividad 3")
    print("4. Actividad 4")
    print("5. Actividad 5")
    print("6. Actividad 6")
    print("0. Volver")
    print()



def main():

    A = np.array([[2,1,1],
                  [1,3,1],
                  [1,1,2]])
    L, U = crout(A)

    continuar = True

    while continuar:
        mostrarMenuPrincipal()
        opcionPrincipal = int(input("Seleccione una opción: "))
        if opcionPrincipal == 1:
            continuarEj1 = True

            while continuarEj1:
                mostrarMenuEjercicio1()
                opcion = int(input("Seleccione una opción: "))

                if opcion == 1:
                    print("\nActividad 1:\n")
                    print("L = ")
                    print(L)
                    print("\nU = ")
                    print(U)

                elif opcion == 2:
                    arregloProductos = ["Producto A",
                                        "Producto B",
                                        "Producto C"]

                    print("\nEscenario 1:\n")

                    b = (100,120,90)
                    y = sustitucionAdelante(L, b)
                    x = sustitucionAtras(U, y)

                    for i in range(len(x)):
                        print(f"{arregloProductos[i]}: {x[i]}")

                    print("\n============================\n")

                    print("Escenario 2:\n")
                    b = (80,100,110)
                    y = sustitucionAdelante(L, b)
                    x = sustitucionAtras(U, y)

                    for i in range(len(x)):
                        print(f"{arregloProductos[i]}: {x[i]}")

                    print("\n============================\n")

                    print("Escenario 3:\n")
                    b = (150,160,140)
                    y = sustitucionAdelante(L, b)
                    x = sustitucionAtras(U, y)

                    for i in range(len(x)):
                        print(f"{arregloProductos[i]}: {x[i]}")

                elif opcion == 3:
                    ###### Respuesta: 
                    """ La factorización LU resulta más eficiente cuando se deben resolver varios sistemas con la misma matriz de 
                    coeficientes A y distintos vectores b. Esto se debe a que la eliminación gaussiana transforma la matriz A paso 
                    a paso hasta obtener una forma triangular, proceso que requiere una cantidad considerable de operaciones. Si por ejemplo se 
                    resolvieran los tres escenarios de manera independiente mediante eliminación gaussiana, habría que repetir exactamente 
                    las mismas transformaciones sobre la misma matriz A tres veces.

                    En cambio, con el método LU la descomposición A=LU se calcula una sola vez. Después, para cada nuevo vector b, únicamente es 
                    necesario resolver los sistemas triangulares Ly=b y Ux=y, utilizando sustitución hacia adelante y hacia atrás. Estos 
                    procedimientos requieren muchas menos operaciones que volver a realizar toda la eliminación gaussiana.

                    Por esta razón, cuando la matriz de coeficientes permanece fija y solo cambian los términos independientes, la factorización 
                    LU reduce el tiempo de cálculo y evita repetir trabajo que ya fue realizado."""

                    ###### Respuesta en Consola: 
                    print("Ej1 - Actividad 3:\n")
                    print("La factorización LU es más eficiente porque la matriz")
                    print("de coeficientes A se descompone una sola vez en L y U.")
                    print()

                    print("Si se aplicara eliminación gaussiana a cada escenario,")
                    print("sería necesario repetir las mismas operaciones sobre A")
                    print("para cada nuevo vector b.")
                    print()

                    print("Con LU, la factorización se reutiliza y solo se resuelven")
                    print("los sistemas triangulares Ly = b y Ux = y mediante")
                    print("sustitución hacia adelante y hacia atrás.")
                    print()

                    print("De esta forma se evita trabajo repetido y se reduce")
                    print("el costo computacional total.")


                elif opcion == 4:
                    lu, piv = lu_factor(A)               ##
                    escenario1 = np.array([100,120,90])  
                    escenario2 = np.array([80,100,110])  
                    escenario3 = np.array([150,160,140])

                    x1 = lu_solve((lu, piv), escenario1) ##
                    x2 = lu_solve((lu, piv), escenario2)
                    x3 = lu_solve((lu, piv), escenario3)

                    print("Escenario 1:")
                    print(x1)
                    print()

                    print("Escenario 2:")
                    print(x2)
                    print()

                    print("Escenario 3:")
                    print(x3)
                    print()
                    print()


                    ### Comparación respuesta: 
                    """
                    Tanto la implementación manual mediante el método de Crout como las funciones lu_factor y lu_solve de SciPy 
                    producen las mismas soluciones para los tres escenarios, ya que ambos procedimientos se basan en la factorización 
                    LU de la matriz de coeficientes. La principal diferencia radica en la implementación: en el método manual fue 
                    necesario construir explícitamente las matrices L y U, así como desarrollar los algoritmos de sustitución hacia 
                    adelante y hacia atrás para resolver los sistemas resultantes.

                    Por otro lado, SciPy encapsula todo este proceso en funciones optimizadas y probadas, lo que reduce considerablemente 
                    la cantidad de código requerido y disminuye la probabilidad de errores de implementación. Además, estas funciones están 
                    diseñadas para trabajar eficientemente con matrices de gran tamaño, por lo que suelen ofrecer un mejor rendimiento 
                    computacional que una implementación manual en Python.

                    Básicamente, la implementación manual resulta útil para comprender el funcionamiento interno de la factorización LU, mientras 
                    que SciPy constituye una alternativa más práctica y eficiente para resolver problemas reales y de mayor escala.
                    """

                    ### Comparación consola: 
                    print("Comparación:")
                    print()

                    print("Método implementado manualmente:")
                    print("- Se construyen las matrices L y U con Crout.")
                    print("- Se resuelve Ly=b mediante sustitución hacia adelante.")
                    print("- Se resuelve Ux=y mediante sustitución hacia atrás.")
                    print()

                    print("Método con SciPy:")
                    print("- lu_factor calcula la factorización LU automáticamente.")
                    print("- lu_solve utiliza dicha factorización para resolver Ax=b.")
                    print("- Requiere menos código y es más robusto.")
                    print()

                    print("Ambos métodos producen la misma solución.")
                    print("Sin embargo, SciPy está optimizado y resulta más práctico")
                    print("para problemas de mayor tamaño.")


                elif opcion == 0:
                    continuarEj1 = False

                else:
                    print("\nOpción inválida.\n")



        elif opcionPrincipal == 2:

            A = np.random.uniform(0, 1, size=(50,50))
            matrizAOriginal = np.array(A) ## Para hacer copia de la matriz original
            matDiagonDomin = modificarADiagonalmenteDominanteEstricta(A)
            bVector = np.random.uniform(0, 100, size=50)
            L, U = crout(matDiagonDomin)
            y = sustitucionAdelante(L, bVector)
            x = sustitucionAtras(U, y)
            

            continuarEj2 = True
            while continuarEj2:

                mostrarMenuEjercicio2()
                opcion = int(input("Seleccione una actividad: "))

                if opcion == 1:
                    print("Matriz A:")
                    print(matrizAOriginal)

                elif opcion == 2:
                    #B = np.array([[2,1,3], [4,5,2], [1,6,3]])
                    print("Matriz A modificada:")
                    print(matDiagonDomin)

                elif opcion == 3:
                    print("El vector b:")
                    print(bVector)

                elif opcion == 4:
                    print("Factorización con Crout:")
                    print("L = ")
                    print(L)
                    print("\nU = ")
                    print(U)

                elif opcion == 5:
                    
                    print("Vector y:")
                    print(y)
                    print()
                    print("Vector x:")
                    print(x)

                elif opcion == 6: 
                    vectorResiduo = (matDiagonDomin @ x ) - bVector

                elif opcion == 0:
                    continuarEj2 = False
                    norma = np.linalg.norm(vectorResiduo)

                    print("Vector residuo:")
                    print(vectorResiduo)

                    print()
                    print("Norma euclidiana:")
                    print(norma)

                else:
                    print("Opción inválida.")


        elif opcionPrincipal == 0:
            continuar = False


main() 