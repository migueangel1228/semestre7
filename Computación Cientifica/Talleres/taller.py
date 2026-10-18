"""
JULIANA GONZÁLEZ SÁNCHEZ

PROF JUAN FERNANDO PAZ

COMPUTACIÓN CIENTÍFICA

TALLER DE BÚSQUEDA DE RAÍCES


"""

import numpy as np

# Tolerancias que se van a probar en los dos problemas
tolerancias = [1e-2, 1e-4, 1e-6, 1e-8, 1e-10]
max_iteraciones = 1000


# --------------------------------------------------
# METODOS PARA ENCONTRAR RAICES
# Cada metodo devuelve la raiz aproximada, las iteraciones
# y los valores que fue calculando.
# Si no puede continuar, devuelve None como raiz.
# --------------------------------------------------

def biseccion(f, a, b, eps):
    if f(a) * f(b) > 0:
        return None, 0, []

    aproximaciones = []
    k = 0

    while (b - a) / 2 > eps and k < max_iteraciones:
        medio = (a + b) / 2
        f_medio = f(medio)
        aproximaciones.append(medio)
        k += 1

        if f_medio == 0:
            return medio, k, aproximaciones

        if f(a) * f_medio < 0:
            b = medio
        else:
            a = medio

    return (a + b) / 2, k, aproximaciones


def newton_raphson(f, df, x0, eps):
    x = x0
    aproximaciones = [x]

    for k in range(1, max_iteraciones + 1):
        derivada = df(x)

        if derivada == 0:
            return None, k - 1, aproximaciones

        nuevo = x - f(x) / derivada
        aproximaciones.append(nuevo)

        if abs(nuevo - x) < eps:
            return nuevo, k, aproximaciones

        x = nuevo

    return x, max_iteraciones, aproximaciones


def secante(f, x0, x1, eps):
    aproximaciones = [x0, x1]

    for k in range(1, max_iteraciones + 1):
        f0 = f(x0)
        f1 = f(x1)

        if f1 == f0:
            return None, k - 1, aproximaciones

        nuevo = x1 - f1 * (x1 - x0) / (f1 - f0)
        aproximaciones.append(nuevo)

        if abs(nuevo - x1) < eps:
            return nuevo, k, aproximaciones

        x0 = x1
        x1 = nuevo

    return x1, max_iteraciones, aproximaciones


def falsa_posicion(f, a, b, eps):
    if f(a) * f(b) > 0:
        return None, 0, []

    aproximaciones = []
    anterior = None

    for k in range(1, max_iteraciones + 1):
        fa = f(a)
        fb = f(b)

        if fb == fa:
            return None, k - 1, aproximaciones

        c = b - fb * (b - a) / (fb - fa)
        fc = f(c)
        aproximaciones.append(c)

        if fc == 0:
            return c, k, aproximaciones

        if anterior is not None and abs(c - anterior) < eps:
            return c, k, aproximaciones

        if fa * fc < 0:
            b = c
        else:
            a = c

        anterior = c

    return c, max_iteraciones, aproximaciones


def punto_fijo(g, x0, eps):
    x = x0
    aproximaciones = [x]

    for k in range(1, max_iteraciones + 1):
        nuevo = g(x)
        aproximaciones.append(nuevo)

        if not np.isfinite(nuevo):
            return None, k, aproximaciones

        if abs(nuevo - x) < eps:
            return nuevo, k, aproximaciones

        x = nuevo

    return x, max_iteraciones, aproximaciones


# --------------------------------------------------
# Funciones para impresión de resultados.
# --------------------------------------------------

ANCHO = 74


def titulo(texto):
    print()
    print("=" * ANCHO)
    print(texto.center(ANCHO))
    print("=" * ANCHO)


def subtitulo(texto):
    print()
    print(texto)
    print("-" * len(texto))


def dato(nombre, valor):
    print(f"  {nombre:<28}{valor}")


def comparar_metodos(titulo_tabla, metodos):
    subtitulo(titulo_tabla)

    encabezado = f"  {'Método':<16}"
    for eps in tolerancias:
        encabezado += f"{'ε = ' + format(eps, '.0e'):>11}"
    print(encabezado)
    print("  " + "-" * (len(encabezado) - 2))

    raices = {}

    for nombre, metodo in metodos.items():
        fila = f"  {nombre:<16}"
        raiz_actual = None

        for eps in tolerancias:
            raiz, k, aproximaciones = metodo(eps)
            fila += f"{k:>11}"
            raiz_actual = raiz

        raices[nombre] = raiz_actual
        print(fila)

    subtitulo("Raíz aproximada con ε = 1e-10")
    for nombre, raiz in raices.items():
        if raiz is None:
            dato(nombre, "No se pudo calcular")
        else:
            dato(nombre, f"{raiz:.10f}")

    return raices


def mostrar_aproximaciones(metodos_aprox, cantidad=5):
    """metodos_aprox: diccionario {nombre: lista de aproximaciones}"""
    subtitulo("Primeras aproximaciones")
    encabezado = f"  {'Método':<16}" + "".join(f"{'x' + str(i):>13}" for i in range(cantidad))
    print(encabezado)
    print("  " + "-" * (len(encabezado) - 2))
    for nombre, aproximaciones in metodos_aprox.items():
        fila = f"  {nombre:<16}" + "".join(f"{x:>13.6f}" for x in aproximaciones[:cantidad])
        print(fila)


# ==================================================
# ACTIVIDAD 1: TEMPERATURA CRITICA
# ==================================================

titulo("ACTIVIDAD 1: INSTANTE EN QUE SE ALCANZAN LOS 60 °C")

tiempos = np.array([0, 1, 2, 3, 4, 5], dtype=float)
temperaturas = np.array([21.0, 23.8, 28.1, 34.5, 43.2, 55.0])
temperatura_critica = 60.0

# Construimos el polinomio interpolante con Vandermonde.
# Cada fila contiene las potencias del tiempo: 1, t, t^2, ..., t^5.
V = np.vander(tiempos, N=6, increasing=True)
coeficientes = np.linalg.solve(V, temperaturas)

subtitulo("Matriz de Vandermonde")
for fila in V.astype(int):
    print("  " + "".join(f"{x:>7}" for x in fila))

subtitulo("Coeficientes del polinomio interpolante")
for i, valor in enumerate(coeficientes):
    dato(f"c{i}", f"{valor:>10.6f}")


def T1(t):
    return sum(coeficientes[i] * t**i for i in range(len(coeficientes)))


def f1(t):
    # La raiz ocurre cuando T(t) - 60 = 0.
    return T1(t) - temperatura_critica


def df1(t):
    return sum(i * coeficientes[i] * t**(i - 1)
               for i in range(1, len(coeficientes)))


subtitulo("Intervalo inicial")
dato("T(6)", f"{T1(6):.4f} °C")
dato("f(5)", f"{f1(5):.6f}")
dato("f(6)", f"{f1(6):.6f}")
dato("Cambio de signo en", "[5, 6]")


# Punto fijo.
# El despeje directo de t^5 converge pero lento (|g'| = 0.53).
# Para mejorarlo usamos g(t) = t - f(t)/16, donde 16 es parecido a f'(t*),
# así |g'(t*)| = |1 - f'(t*)/16| queda casi en cero y converge rápido.
def g1(t):
    return t - f1(t) / 16


metodos_1 = {
    "Bisección": lambda eps: biseccion(f1, 5.0, 6.0, eps),
    "Newton-Raphson": lambda eps: newton_raphson(f1, df1, 5.0, eps),
    "Secante": lambda eps: secante(f1, 5.0, 6.0, eps),
    "Falsa posición": lambda eps: falsa_posicion(f1, 5.0, 6.0, eps),
    "Punto fijo": lambda eps: punto_fijo(g1, 5.0, eps),
}

raices_1 = comparar_metodos("Número de iteraciones por método y tolerancia", metodos_1)

mostrar_aproximaciones({
    "Bisección": biseccion(f1, 5.0, 6.0, 1e-10)[2],
    "Newton-Raphson": newton_raphson(f1, df1, 5.0, 1e-10)[2],
    "Secante": secante(f1, 5.0, 6.0, 1e-10)[2],
})

raiz_tiempo = raices_1["Newton-Raphson"]
if raiz_tiempo is not None:
    minutos = int(raiz_tiempo)
    segundos = (raiz_tiempo - minutos) * 60
    subtitulo("Resultado")
    dato("Tiempo estimado", f"{raiz_tiempo:.6f} min")
    dato("Equivale a", f"{minutos} min {segundos:.1f} s")
else:
    print("\n  No se pudo calcular el tiempo estimado.")


# ==================================================
# ACTIVIDAD 2: TAMANO MAXIMO DE ENTRADA
# ==================================================

titulo("ACTIVIDAD 2: TAMAÑO MÁXIMO DE ENTRADA (PRESUPUESTO DE 100 ms)")

presupuesto = 100.0


def T2(n):
    return 0.02 * n * np.log2(n) + 0.5 * np.exp(0.003 * n)


def f2(n):
    return T2(n) - presupuesto


def df2(n):
    return (0.02 * (np.log2(n) + 1 / np.log(2))
            + 0.0015 * np.exp(0.003 * n))


subtitulo("Intervalo inicial")
for n in [100, 500, 600, 1000]:
    dato(f"f({n})", f"{f2(n):>12.6f}")
dato("Cambio de signo en", "[500, 600]")


# Para punto fijo despejamos la n que multiplica a log2(n):
#   0.02 * n * log2(n) = 100 - 0.5 * e^(0.003 n)
#   n = (100 - 0.5 * e^(0.003 n)) / (0.02 * log2(n))
def g2(n):
    return (presupuesto - 0.5 * np.exp(0.003 * n)) / (0.02 * np.log2(n))


metodos_2 = {
    "Bisección": lambda eps: biseccion(f2, 500.0, 600.0, eps),
    "Newton-Raphson": lambda eps: newton_raphson(f2, df2, 500.0, eps),
    "Secante": lambda eps: secante(f2, 500.0, 600.0, eps),
    "Falsa posición": lambda eps: falsa_posicion(f2, 500.0, 600.0, eps),
    "Punto fijo": lambda eps: punto_fijo(g2, 500.0, eps),
}

raices_2 = comparar_metodos("Número de iteraciones por método y tolerancia", metodos_2)

mostrar_aproximaciones({
    "Bisección": biseccion(f2, 500.0, 600.0, 1e-10)[2],
    "Newton-Raphson": newton_raphson(f2, df2, 500.0, 1e-10)[2],
    "Secante": secante(f2, 500.0, 600.0, 1e-10)[2],
})

raiz_entrada = raices_2["Newton-Raphson"]
if raiz_entrada is not None:
    n_max = int(np.floor(raiz_entrada))
    subtitulo("Resultado")
    dato("Raíz aproximada", f"n* = {raiz_entrada:.6f}")
    dato(f"T({n_max})", f"{T2(n_max):.4f} ms  (cumple)")
    dato(f"T({n_max + 1})", f"{T2(n_max + 1):.4f} ms  (se pasa)")
    dato("Tamaño máximo de entrada", f"n = {n_max}")
else:
    print("\n  No se pudo calcular el tamaño máximo de entrada.")