import numpy as np
import matplotlib.pyplot as plt

# ==============================================================================
# CONFIGURACIÓN VISUAL (ESTILO PUJ / ACADÉMICO)
# ==============================================================================
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    'figure.autolayout': True,
    'lines.linewidth': 2.0,
    'grid.alpha': 0.35,
    'grid.linestyle': '--'
})

PUJ_BLUE = '#003366'
DARK_BLUE = '#102C57'
TEAL_COLOR = '#008080'
ACCENT_RED = '#C0392B'
ACCENT_ORANGE = '#D35400'
ACCENT_PURPLE = '#8E44AD'

# ==============================================================================
# 1. ALGORITMOS BASE DE BÚSQUEDA DE RAÍCES (ESTRUCTURADOS: FLAGS Y CONDICIONALES)
# ==============================================================================
def biseccion(f, a, b, tol, max_iter=200):
    iters = 0
    terminado = False
    hist = []
    
    while not terminado and iters < max_iter:
        error = (b - a) / 2.0
        if error < tol:
            terminado = True
        else:
            iters += 1
            m = (a + b) / 2.0
            fm = f(m)
            hist.append((a, b, m, fm, (b - a) / 2.0))
            if f(a) * fm <= 0:
                b = m
            else:
                a = m
                
    raiz = (a + b) / 2.0
    return raiz, iters, hist

def newton_raphson(f, df, x0, tol, max_iter=200):
    iters = 0
    x = x0
    terminado = False
    hist = []
    
    while not terminado and iters < max_iter:
        iters += 1
        fx = f(x)
        dfx = df(x)
        x_next = x - fx / dfx
        error = abs(x_next - x)
        hist.append((x, fx, dfx, x_next, error))
        if error < tol:
            terminado = True
        x = x_next
        
    return x, iters, hist

def secante(f, x0, x1, tol, max_iter=200):
    iters = 0
    terminado = False
    p0 = x0
    p1 = x1
    raiz = p1
    hist = []
    
    while not terminado and iters < max_iter:
        iters += 1
        fp0 = f(p0)
        fp1 = f(p1)
        p2 = p1 - fp1 * (p1 - p0) / (fp1 - fp0)
        error = abs(p2 - p1)
        hist.append((p0, p1, fp1, p2, error))
        if error < tol:
            terminado = True
        p0 = p1
        p1 = p2
        raiz = p2
        
    return raiz, iters, hist

def falsa_posicion(f, a, b, tol, max_iter=200):
    iters = 0
    terminado = False
    c_prev = None
    raiz = a
    hist = []
    
    while not terminado and iters < max_iter:
        iters += 1
        fa = f(a)
        fb = f(b)
        c = b - fb * (b - a) / (fb - fa)
        fc = f(c)
        error = abs(c - c_prev) if c_prev is not None else abs(b - a)
        hist.append((a, b, c, fc, error))
        
        if c_prev is not None and abs(c - c_prev) < tol:
            terminado = True
        else:
            c_prev = c
            if fa * fc <= 0:
                b = c
            else:
                a = c
        raiz = c
        
    return raiz, iters, hist

def punto_fijo(g, x0, tol, max_iter=200):
    iters = 0
    x = x0
    terminado = False
    hist = []
    
    while not terminado and iters < max_iter:
        iters += 1
        x_next = g(x)
        error = abs(x_next - x)
        hist.append((x, x_next, error))
        if error < tol:
            terminado = True
        x = x_next
        
    return x, iters, hist

# ==============================================================================
# 2. ACTIVIDAD 1: EXPERIMENTO DE LABORATORIO
# ==============================================================================
t_exp = np.array([0, 1, 2, 3, 4, 5], dtype=float)
T_exp = np.array([21.0, 23.8, 28.1, 34.5, 43.2, 55.0], dtype=float)

# Matriz de Vandermonde: T(t) = c0 + c1*t + c2*t^2 + c3*t^3 + c4*t^4 + c5*t^5
V = np.vander(t_exp, increasing=True)
c_poly = np.linalg.solve(V, T_exp)

def T_func(t):
    return np.polyval(c_poly[::-1], t)

def f1(t):
    return T_func(t) - 60.0

def df1(t):
    powers = np.arange(1, len(c_poly))
    c_der = powers * c_poly[1:]
    return np.polyval(c_der[::-1], t)

# Función de punto fijo para Actividad 1
g1 = lambda t: t - f1(t) / 16.0

# ==============================================================================
# 3. ACTIVIDAD 2: SISTEMA DISTRIBUIDO
# ==============================================================================
def T2_func(n):
    return 0.02 * n * np.log2(n) + 0.5 * np.exp(0.003 * n)

def f2(n):
    return T2_func(n) - 100.0

def df2(n):
    return 0.02 * (np.log2(n) + 1.0 / np.log(2.0)) + 0.0015 * np.exp(0.003 * n)

# Función de punto fijo para Actividad 2
g2 = lambda n: n - f2(n)

# ==============================================================================
# 4. EXPERIMENTACIÓN CON DISTINTAS TOLERANCIAS
# ==============================================================================
tolerancias = [1e-2, 1e-4, 1e-6, 1e-8, 1e-10]

print("=" * 80)
print("POLINOMIO ACTIVIDAD 1:")
print(f"c = {c_poly}")
print("Verificación en nodos experimentales:")
for ti, Ti in zip(t_exp, T_exp):
    print(f"  T({ti}) = {T_func(ti):.4f} (esperado {Ti:.1f})")

print("\n" + "=" * 80)
print("TABLA COMPARATIVA - ACTIVIDAD 1")
print(f"{'Método':<18} | " + " | ".join([f"{tol:8.0e}" for tol in tolerancias]))
print("-" * 80)

metodos_act1 = {
    'Bisección': lambda tol: biseccion(f1, 5.0, 6.0, tol),
    'Newton-Raphson': lambda tol: newton_raphson(f1, df1, 5.0, tol),
    'Secante': lambda tol: secante(f1, 5.0, 6.0, tol),
    'Falsa posición': lambda tol: falsa_posicion(f1, 5.0, 6.0, tol),
    'Punto fijo': lambda tol: punto_fijo(g1, 5.0, tol)
}

for nombre, m in metodos_act1.items():
    fila = [str(m(tol)[1]) for tol in tolerancias]
    print(f"{nombre:<18} | " + " | ".join([f"{val:>8}" for val in fila]))

print("\n" + "=" * 80)
print("TABLA COMPARATIVA - ACTIVIDAD 2")
print(f"{'Método':<18} | " + " | ".join([f"{tol:8.0e}" for tol in tolerancias]))
print("-" * 80)

metodos_act2 = {
    'Bisección': lambda tol: biseccion(f2, 500.0, 600.0, tol),
    'Newton-Raphson': lambda tol: newton_raphson(f2, df2, 500.0, tol),
    'Secante': lambda tol: secante(f2, 500.0, 600.0, tol),
    'Falsa posición': lambda tol: falsa_posicion(f2, 500.0, 600.0, tol),
    'Punto fijo': lambda tol: punto_fijo(g2, 500.0, tol)
}

for nombre, m in metodos_act2.items():
    fila = [str(m(tol)[1]) for tol in tolerancias]
    print(f"{nombre:<18} | " + " | ".join([f"{val:>8}" for val in fila]))

# ==============================================================================
# 5. GENERACIÓN DE FIGURAS
# ==============================================================================

# FIGURA 1: Actividad 1 - Polinomio y Búsqueda de Raíz
fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=300)
t_dense = np.linspace(0, 6.2, 500)
T_dense = T_func(t_dense)

ax.plot(t_dense, T_dense, color=PUJ_BLUE, label=r'$T(t)$ interpolante')
ax.scatter(t_exp, T_exp, color=DARK_BLUE, s=50, zorder=5, label='Datos experimentales')
ax.axhline(60, color=ACCENT_RED, linestyle='--', label=r'$T_{\mathrm{crit}} = 60\,^\circ\mathrm{C}$')

t_star = 5.335106
ax.plot([t_star, t_star], [0, 60], color='gray', linestyle=':')
ax.scatter([t_star], [60], color=ACCENT_RED, s=70, zorder=6,
           label=f'Raíz: $t^* \\approx {t_star:.2f}$ min')

ax.set_xlabel('Tiempo $t$ [min]')
ax.set_ylabel(r'Temperatura $T(t)$ [$^\circ$C]')
ax.set_title('Actividad 1: Curva térmica interpolante y temperatura crítica')
ax.legend(loc='upper left', frameon=True)
ax.grid(True)
ax.set_xlim(0, 6.2)
ax.set_ylim(15, 75)
plt.savefig('fig_actividad1_polinomio_temperatura.png')
plt.close()
print("Guardada: fig_actividad1_polinomio_temperatura.png")

# FIGURA 2: Actividad 2 - Tiempo de ejecución vs n
fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=300)
n_dense = np.linspace(450, 650, 500)
T2_dense = T2_func(n_dense)

ax.plot(n_dense, T2_dense, color=PUJ_BLUE, label=r'$T(n) = 0.02n\log_2(n) + 0.5e^{0.003n}$')
ax.axhline(100, color=ACCENT_RED, linestyle='--', label='Presupuesto límite: 100 ms')

n_star = 537.447345
ax.scatter([n_star], [100], color=ACCENT_RED, s=70, zorder=6,
           label=f'Raíz continua: $n^* \\approx {n_star:.2f}$')

# Puntos enteros 537 y 538
t_537 = T2_func(537)
t_538 = T2_func(538)
ax.scatter([537], [t_537], color='green', s=60, zorder=6,
           label=f'$n = 537$: $T(537) \\approx {t_537:.2f}$ ms $\\leq 100$')
ax.scatter([538], [t_538], color=ACCENT_ORANGE, s=60, zorder=6,
           label=f'$n = 538$: $T(538) \\approx {t_538:.2f}$ ms $> 100$')

ax.set_xlabel('Tamaño de entrada $n$')
ax.set_ylabel('Tiempo de ejecución $T(n)$ [ms]')
ax.set_title('Actividad 2: Tiempo de procesamiento vs. Tamaño de entrada')
ax.legend(loc='upper left', frameon=True)
ax.grid(True)
ax.set_xlim(480, 620)
ax.set_ylim(80, 120)
plt.savefig('fig_actividad2_tiempo_servicio.png')
plt.close()
print("Guardada: fig_actividad2_tiempo_servicio.png")

# FIGURA 3: Comparación de Convergencia
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), dpi=300)

tol_ref = 1e-10
colores = {
    'Bisección': PUJ_BLUE,
    'Newton-Raphson': ACCENT_RED,
    'Secante': TEAL_COLOR,
    'Falsa posición': ACCENT_ORANGE,
    'Punto fijo': ACCENT_PURPLE
}

# Historial Actividad 1
_, _, h_bis1 = biseccion(f1, 5.0, 6.0, tol_ref)
_, _, h_newt1 = newton_raphson(f1, df1, 5.0, tol_ref)
_, _, h_sec1 = secante(f1, 5.0, 6.0, tol_ref)
_, _, h_fpos1 = falsa_posicion(f1, 5.0, 6.0, tol_ref)
_, _, h_pf1 = punto_fijo(g1, 5.0, tol_ref)

ax1.semilogy(range(1, len(h_bis1)+1), [item[4] for item in h_bis1], marker='o', markersize=3, label='Bisección', color=colores['Bisección'])
ax1.semilogy(range(1, len(h_newt1)+1), [item[4] for item in h_newt1], marker='s', markersize=3, label='Newton-Raphson', color=colores['Newton-Raphson'])
ax1.semilogy(range(1, len(h_sec1)+1), [item[4] for item in h_sec1], marker='^', markersize=3, label='Secante', color=colores['Secante'])
ax1.semilogy(range(1, len(h_fpos1)+1), [item[4] for item in h_fpos1], marker='d', markersize=3, label='Falsa posición', color=colores['Falsa posición'])
ax1.semilogy(range(1, len(h_pf1)+1), [item[2] for item in h_pf1], marker='x', markersize=3, label='Punto fijo', color=colores['Punto fijo'])

ax1.set_title('Actividad 1: Evolución del error $(\\varepsilon = 10^{-10})$')
ax1.set_xlabel('Iteración $k$')
ax1.set_ylabel(r'Error / Diferencia $|x_{k+1} - x_k|$')
ax1.grid(True)
ax1.legend()

# Historial Actividad 2
_, _, h_bis2 = biseccion(f2, 500.0, 600.0, tol_ref)
_, _, h_newt2 = newton_raphson(f2, df2, 500.0, tol_ref)
_, _, h_sec2 = secante(f2, 500.0, 600.0, tol_ref)
_, _, h_fpos2 = falsa_posicion(f2, 500.0, 600.0, tol_ref)
_, _, h_pf2 = punto_fijo(g2, 500.0, tol_ref)

ax2.semilogy(range(1, len(h_bis2)+1), [item[4] for item in h_bis2], marker='o', markersize=3, label='Bisección', color=colores['Bisección'])
ax2.semilogy(range(1, len(h_newt2)+1), [item[4] for item in h_newt2], marker='s', markersize=3, label='Newton-Raphson', color=colores['Newton-Raphson'])
ax2.semilogy(range(1, len(h_sec2)+1), [item[4] for item in h_sec2], marker='^', markersize=3, label='Secante', color=colores['Secante'])
ax2.semilogy(range(1, len(h_fpos2)+1), [item[4] for item in h_fpos2], marker='d', markersize=3, label='Falsa posición', color=colores['Falsa posición'])
ax2.semilogy(range(1, len(h_pf2)+1), [item[2] for item in h_pf2], marker='x', markersize=3, label='Punto fijo', color=colores['Punto fijo'])

ax2.set_title('Actividad 2: Evolución del error $(\\varepsilon = 10^{-10})$')
ax2.set_xlabel('Iteración $k$')
ax2.set_ylabel(r'Error / Diferencia $|x_{k+1} - x_k|$')
ax2.grid(True)
ax2.legend()

plt.savefig('fig_comparacion_convergencia.png')
plt.close()
print("Guardada: fig_comparacion_convergencia.png")
