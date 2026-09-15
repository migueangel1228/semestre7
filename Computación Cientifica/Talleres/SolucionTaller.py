from numpy._core import arrayprint
from sys import stdin

import numpy as np

def crout(A):
    n = len(A)
    # Inicilizacion de matriz en 0
    L = [[0.0 for _ in range(n)] for _ in range(n)]
    U = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)] # Diagonal principal en 1
    
    for j in range(n):
        for i in range(j, n):
            cntL = 0
            for k in range(j):
                cntL += L[i][k] * U[k][j]
            L[i][j] = A[i][j] - cntL
            
        for i in range(j + 1, n):
            cntU = 0
            for k in range(j):
                cntU += L[j][k] * U[k][i]
            U[j][i] = (A[j][i] - cntU) / L[j][j]
         
    return L, U

def imprimirMatriz(nombre, M):
    print(nombre)
    for fila in M:
        print("  [ " + "  ".join(f"{val:8.4f}" for val in fila) + " ]")

def main():
    A = [[2, 1, 1],
         [1, 3, 1],
         [1, 1, 2]]
         
    L, U = crout(A)
    print("Solucion 1: ")
    imprimirMatriz("Matriz A (Original)", A)
    imprimirMatriz("Matriz L (Triangular Inferior)", L)
    imprimirMatriz("Matriz U (Triangular Superior)", U)
    print("Solucion 2: ")
    
    # Verificación L * U
    n = len(A)
    LU = [[sum(L[i][k] * U[k][j] for k in range(n)) for j in range(n)] for i in range(n)]
    imprimirMatriz("Verificación (L * U)", LU)

main() 