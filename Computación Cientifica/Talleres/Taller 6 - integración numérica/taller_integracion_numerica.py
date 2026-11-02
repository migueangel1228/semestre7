"""
================================================================================
TALLER DE INTEGRACIÓN NUMÉRICA
Curso: Computación Científica / Métodos Numéricos
Profesor: Juan Fernando Paz
Estudiante: Miguel Angel Padilla Rosero
Pontificia Universidad Javeriana Cali

Contenido:
1. Ejercicio 1: Función polinómica I = int_0^3 (x^2 + 2x + 1) dx
   - Valor exacto analítico.
   - Riemann izquierda, derecha, punto medio y trapecio.
   - Comparación de errores para diferentes valores de n.
   - Gráfica guardada: fig_ejercicio1_convergencia.png

2. Ejercicio 2: Función trigonométrica I = int_0^pi sin(x) dx
   - Valor exacto analítico.
   - Métodos numéricos y análisis de simetría.
   - Efecto de incrementar n.
   - Gráfica guardada: fig_ejercicio2_error_trapecio.png

3. Ejercicio 3: Función exponencial I = int_0^2 e^x dx
   - Valor exacto analítico.
   - Comparación teórica y numérica: trapecio vs rectángulos.
   - Número de subintervalos para tolerancia 10^-6 (cota teórica vs cómputo real).
   - Gráfica guardada: fig_ejercicio3_exponencial.png

4. Ejercicio 4: Datos experimentales
   - Integración mediante trapecio y rectángulos (izquierda y derecha).
   - Discusión profunda sobre limitaciones en datos experimentales.
   - Gráfica guardada: fig_ejercicio4_datos_experimentales.png

5. Ejercicio 5: Oscilaciones de alta frecuencia I = int_0^1 e^-x sin(40x^2 + 5x) dx
   - Frecuencia instantánea y regiones de oscilación.
   - Trapecio y Simpson 1/3 para n = 20, 40, 80, 160, 320.
   - Errores absolutos respecto a referencia de alta precisión.
   - Estimación del orden experimental de convergencia.
   - Menor n para tolerancia 10^-6.
   - Gráfica guardada: fig_ejercicio5_oscilaciones.png

6. Ejercicio 6: Integral altamente oscilatoria I = int_0^1 sin(1/(x + 0.01)) dx
   - Análisis de frecuencia instantánea hiperbólica omega(x) = 1/(x+0.01)^2.
   - Trapecio y Simpson 1/3 para n = 50, 100, 200, 400, 800, 1600.
   - Errores absolutos respecto a referencia de alta precisión.
   - Justificación teórica de la ineficiencia de mallas uniformes (criterio de Nyquist).
   - Estrategia de refinamiento local (malla adaptativa / cuadratura adaptativa) y comparación.
   - Gráfica guardada: fig_ejercicio6_altamente_oscilatoria.png
================================================================================
"""

import os
import numpy as np
import scipy.integrate as spi
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

np.set_printoptions(precision=8, suppress=True)

# Directorio donde reside el script para guardar figuras de manera robusta
DIR_ACTUAL = os.path.dirname(os.path.abspath(__file__))

# Configuración estética de gráficas
plt.rcParams.update({
    'font.size': 10,
    'font.family': 'serif',
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 13,
    'figure.autolayout': True,
    'lines.linewidth': 1.8,
    'grid.alpha': 0.35,
    'grid.linestyle': '--'
})

PUJ_BLUE = '#003366'
DARK_BLUE = '#102C57'
TEAL_COLOR = '#008080'
ACCENT_RED = '#C0392B'
ACCENT_ORANGE = '#D35400'
ACCENT_PURPLE = '#8E44AD'
ACCENT_GREEN = '#27AE60'


# ==============================================================================
# FUNCIONES AUXILIARES DE FORMATO
# ==============================================================================

def section_header(title):
    print("\n" + "=" * 80)
    print(f" {title.upper()} ")
    print("=" * 80)

def sub_header(title):
    print("\n--- " + title + " ---")


# ==============================================================================
# ALGORITMOS BASE DE INTEGRACIÓN NUMÉRICA
# ==============================================================================

def riemann_izquierda(f, a, b, n):
    """Suma de Riemann por el extremo izquierdo."""
    h = (b - a) / n
    x = a + np.arange(n) * h
    return h * np.sum(f(x))

def riemann_derecha(f, a, b, n):
    """Suma de Riemann por el extremo derecho."""
    h = (b - a) / n
    x = a + np.arange(1, n + 1) * h
    return h * np.sum(f(x))

def punto_medio(f, a, b, n):
    """Regla del punto medio compuesta."""
    h = (b - a) / n
    x_mid = a + (np.arange(n) + 0.5) * h
    return h * np.sum(f(x_mid))

def trapecio(f, a, b, n):
    """Regla del trapecio compuesta con nodos uniformes."""
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    y = f(x)
    return h * (0.5 * y[0] + np.sum(y[1:-1]) + 0.5 * y[-1])

def simpson_13(f, a, b, n):
    """Regla de Simpson 1/3 compuesta (requiere n par)."""
    if n % 2 != 0:
        raise ValueError("El número de subintervalos n debe ser par para Simpson 1/3.")
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    y = f(x)
    return (h / 3.0) * (y[0] + 4.0 * np.sum(y[1:-1:2]) + 2.0 * np.sum(y[2:-1:2]) + y[-1])

def trapecio_no_uniforme(x, y):
    """Regla del trapecio para una malla no uniforme x arbitraria."""
    dx = np.diff(x)
    return np.sum(0.5 * (y[:-1] + y[1:]) * dx)

def cuadratura_adaptativa_simpson(f, a, b, tol, max_depth=50):
    """
    Cuadratura adaptativa recursiva de Simpson con control local de error.
    Retorna la aproximación, el número total de evaluaciones y los nodos visitados.
    """
    eval_count = [0]
    nodes = []

    def simpson_local(fa, fb, fc, a_loc, b_loc):
        return (b_loc - a_loc) / 6.0 * (fa + 4.0 * fc + fb)

    fa = f(a)
    fb = f(b)
    c = 0.5 * (a + b)
    fc = f(c)
    eval_count[0] += 3
    nodes.extend([a, c, b])

    def _rec(a_loc, b_loc, f_a, f_b, f_c, S, tol_loc, depth):
        c_loc = 0.5 * (a_loc + b_loc)
        d_loc = 0.5 * (a_loc + c_loc)
        e_loc = 0.5 * (c_loc + b_loc)
        f_d = f(d_loc)
        f_e = f(e_loc)
        eval_count[0] += 2
        nodes.extend([d_loc, e_loc])

        S_left = simpson_local(f_a, f_c, f_d, a_loc, c_loc)
        S_right = simpson_local(f_c, f_b, f_e, c_loc, b_loc)
        S2 = S_left + S_right
        err_est = abs(S2 - S) / 15.0

        if err_est < tol_loc or depth >= max_depth:
            # Corrección de Richardson de orden 6
            return S2 + (S2 - S) / 15.0

        return (_rec(a_loc, c_loc, f_a, f_c, f_d, S_left, tol_loc / 2.0, depth + 1) +
                _rec(c_loc, b_loc, f_c, f_b, f_e, S_right, tol_loc / 2.0, depth + 1))

    S_init = simpson_local(fa, fb, fc, a, b)
    resultado = _rec(a, b, fa, fb, fc, S_init, tol, 1)
    return resultado, eval_count[0], np.array(sorted(nodes))


# ==============================================================================
# EJERCICIO 1: FUNCIÓN POLINÓMICA
# ==============================================================================

def ejecutar_ejercicio_1():
    section_header("Ejercicio 1: Función polinómica I = int_0^3 (x^2 + 2x + 1) dx")

    # 1. Valor exacto
    f = lambda x: x**2 + 2*x + 1
    # F(x) = x^3/3 + x^2 + x
    # F(3) = 27/3 + 9 + 3 = 9 + 9 + 3 = 21
    # F(0) = 0 => I = 21
    I_exacto = 21.0
    print(f"1. Valor exacto analítico:")
    print(f"   I = [x^3/3 + x^2 + x]_0^3 = (27/3 + 9 + 3) - 0 = 21.0")

    # 2, 3, 4, 5. Implementación y comparación de errores
    n_valores = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    
    print("\n5. Tabla comparativa de aproximaciones y errores absolutos:")
    encabezado = f"{'n':>5} | {'Riem. Izq':>12} {'Err Izq':>10} | {'Riem. Der':>12} {'Err Der':>10} | {'Pto Medio':>12} {'Err PM':>10} | {'Trapecio':>12} {'Err Trap':>10}"
    print("-" * len(encabezado))
    print(encabezado)
    print("-" * len(encabezado))

    hist_rl, hist_rr, hist_pm, hist_tr = [], [], [], []
    for n in n_valores:
        rl = riemann_izquierda(f, 0, 3, n)
        rr = riemann_derecha(f, 0, 3, n)
        pm = punto_medio(f, 0, 3, n)
        tr = trapecio(f, 0, 3, n)
        
        err_rl = abs(rl - I_exacto)
        err_rr = abs(rr - I_exacto)
        err_pm = abs(pm - I_exacto)
        err_tr = abs(tr - I_exacto)

        hist_rl.append(err_rl)
        hist_rr.append(err_rr)
        hist_pm.append(err_pm)
        hist_tr.append(err_tr)

        print(f"{n:5d} | {rl:12.6f} {err_rl:10.4e} | {rr:12.6f} {err_rr:10.4e} | {pm:12.6f} {err_pm:10.4e} | {tr:12.6f} {err_tr:10.4e}")
    print("-" * len(encabezado))

    # Análisis teórico de constantes de error
    print("\nAnálisis del orden asintótico:")
    print("Para f''(x) = 2 (constante):")
    print("  * Error teórico Trapecio: E_T = -(b-a)^3/(12 n^2) f'' = -27/(12 n^2) * 2 = -4.5 / n^2")
    print("  * Error teórico Punto Medio: E_M = (b-a)^3/(24 n^2) f'' = 27/(24 n^2) * 2 = 2.25 / n^2")
    print("  * Observe que |E_T| = 2 * |E_M| exactamente para cualquier n.")

    # Generación de figura
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Subplot 1: Interpretación geométrica con n=4
    x_dense = np.linspace(0, 3, 300)
    y_dense = f(x_dense)
    axes[0].plot(x_dense, y_dense, color=PUJ_BLUE, label=r'$f(x) = x^2 + 2x + 1$', lw=2.5)

    n_demo = 4
    h_demo = 3.0 / n_demo
    x_nodes = np.linspace(0, 3, n_demo + 1)
    y_nodes = f(x_nodes)
    
    # Trapecios sombreados
    for i in range(n_demo):
        xs = [x_nodes[i], x_nodes[i], x_nodes[i+1], x_nodes[i+1]]
        ys = [0, y_nodes[i], y_nodes[i+1], 0]
        axes[0].fill(xs, ys, color=PUJ_BLUE, alpha=0.15, edgecolor=PUJ_BLUE, linestyle='--')

    axes[0].plot(x_nodes, y_nodes, 'o--', color=ACCENT_RED, label=f'Trapecio compuesto ($n={n_demo}$)')
    axes[0].set_title(r'Interpretación Geométrica: Regla del Trapecio ($n=4$)', fontweight='bold')
    axes[0].set_xlabel('$x$')
    axes[0].set_ylabel('$f(x)$')
    axes[0].set_xlim(-0.1, 3.1)
    axes[0].set_ylim(0, 18)
    axes[0].legend(loc='upper left')
    axes[0].grid(True)

    # Subplot 2: Gráfica de convergencia del error
    axes[1].loglog(n_valores, hist_rl, 'o-', color=ACCENT_ORANGE, label='Riemann Izquierda $O(n^{-1})$')
    axes[1].loglog(n_valores, hist_rr, 's-', color=ACCENT_PURPLE, label='Riemann Derecha $O(n^{-1})$')
    axes[1].loglog(n_valores, hist_tr, '^-', color=PUJ_BLUE, label=r'Trapecio $O(n^{-2})$ ($|E_T| = 4.5/n^2$)')
    axes[1].loglog(n_valores, hist_pm, 'd-', color=ACCENT_GREEN, label=r'Punto Medio $O(n^{-2})$ ($|E_M| = 2.25/n^2$)')
    
    # Guías teóricas de pendiente
    n_ref = np.array(n_valores)
    axes[1].loglog(n_ref, 20.0 / n_ref, 'k:', alpha=0.6, label=r'Pendiente teórica $-1$')
    axes[1].loglog(n_ref, 4.5 / (n_ref**2), 'k--', alpha=0.6, label=r'Pendiente teórica $-2$')

    axes[1].set_title('Convergencia del Error Absoluto vs $n$ (Escala Log-Log)', fontweight='bold')
    axes[1].set_xlabel('Número de subintervalos ($n$)')
    axes[1].set_ylabel('Error absoluto $|I - I_n|$')
    axes[1].legend(loc='lower left')
    axes[1].grid(True, which='both')

    ruta_fig = os.path.join(DIR_ACTUAL, 'fig_ejercicio1_convergencia.png')
    fig.savefig(ruta_fig, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Gráfica guardada: {ruta_fig}")


# ==============================================================================
# EJERCICIO 2: FUNCIÓN TRIGONOMÉTRICA
# ==============================================================================

def ejecutar_ejercicio_2():
    section_header("Ejercicio 2: Función trigonométrica I = int_0^pi sin(x) dx")

    f = lambda x: np.sin(x)
    # 1. Valor exacto analítico
    # I = [-cos(x)]_0^pi = -cos(pi) - (-cos(0)) = 1 - (-1) = 2.0
    I_exacto = 2.0
    print("1. Valor exacto analítico:")
    print("   I = [-cos(x)]_0^pi = -(-1) - (-1) = 2.0")

    # 2. Aproximaciones
    n_valores = [2, 4, 8, 16, 32, 64, 128, 256]
    
    print("\n2. Tabla de aproximaciones y errores absolutos:")
    encabezado = f"{'n':>5} | {'Trapecio':>14} {'Error Trap':>12} | {'Pto Medio':>14} {'Error PM':>12} | {'Riem. Izq':>14} {'Error RI':>12}"
    print("-" * len(encabezado))
    print(encabezado)
    print("-" * len(encabezado))

    errores_tr, errores_pm, errores_ri = [], [], []
    for n in n_valores:
        tr = trapecio(f, 0, np.pi, n)
        pm = punto_medio(f, 0, np.pi, n)
        ri = riemann_izquierda(f, 0, np.pi, n)
        
        err_tr = abs(tr - I_exacto)
        err_pm = abs(pm - I_exacto)
        err_ri = abs(ri - I_exacto)

        errores_tr.append(err_tr)
        errores_pm.append(err_pm)
        errores_ri.append(err_ri)

        print(f"{n:5d} | {tr:14.8f} {err_tr:12.4e} | {pm:14.8f} {err_pm:12.4e} | {ri:14.8f} {err_ri:12.4e}")
    print("-" * len(encabezado))

    # 3. Análisis de simetría y orden de convergencia
    print("\n3. Análisis del efecto de incrementar n y la simetría de la función:")
    print("   * f(0) = sin(0) = 0 y f(pi) = sin(pi) = 0.")
    print("   * Por consiguiente, Riem_Izq(n) = Riem_Der(n) = Trapecio(n) = h * sum_{i=1}^{n-1} sin(x_i).")
    print("   * La simetría respecto a x = pi/2 anula el término de primer orden en la fórmula de Euler-Maclaurin:")
    print("     T(h) - I = h^2/12 * [f'(pi) - f'(0)] + O(h^4) = h^2/12 * [-1 - 1] = -h^2 / 6.")
    print("     Por ende, el error del trapecio decae cuadráticamente como O(n^-2).")

    # Orden experimental del trapecio
    print("\nOrden experimental de convergencia del Trapecio:")
    for i in range(len(n_valores) - 1):
        p_exp = np.log2(errores_tr[i] / errores_tr[i+1])
        print(f"   n: {n_valores[i]:3d} -> {n_valores[i+1]:3d} | Orden p = {p_exp:.4f}")

    # 4. Gráfica del error absoluto del trapecio en función de n
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Escala lineal
    axes[0].plot(n_valores, errores_tr, 'o-', color=PUJ_BLUE, label=r'Error Trapecio $|I - T(n)|$')
    axes[0].set_title(r'Error Absoluto del Trapecio vs $n$ (Escala Lineal)', fontweight='bold')
    axes[0].set_xlabel('Número de subintervalos ($n$)')
    axes[0].set_ylabel('Error absoluto $|I - T(n)|$')
    axes[0].grid(True)
    axes[0].legend()

    # Escala log-log
    axes[1].loglog(n_valores, errores_tr, 'o-', color=PUJ_BLUE, lw=2, label=r'Error Trapecio $|I - T(n)|$')
    axes[1].loglog(n_valores, errores_pm, 's--', color=ACCENT_GREEN, lw=1.5, label=r'Error Punto Medio $|I - M(n)|$')
    # Guía asintótica: E_T ~ (pi^2 / 6) / n^2
    guia_teorica = (np.pi**2 / 6.0) / (np.array(n_valores)**2)
    axes[1].loglog(n_valores, guia_teorica, 'r:', lw=2, label=r'Asíntota Euler-Maclaurin: $\frac{\pi^2}{6 n^2}$')

    axes[1].set_title(r'Error Absoluto vs $n$ (Escala Log-Log: Orden Cuadrático)', fontweight='bold')
    axes[1].set_xlabel('Número de subintervalos ($n$)')
    axes[1].set_ylabel('Error absoluto')
    axes[1].grid(True, which='both')
    axes[1].legend()

    ruta_fig = os.path.join(DIR_ACTUAL, 'fig_ejercicio2_error_trapecio.png')
    fig.savefig(ruta_fig, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Gráfica guardada: {ruta_fig}")


# ==============================================================================
# EJERCICIO 3: EXPONENCIAL
# ==============================================================================

def ejecutar_ejercicio_3():
    section_header("Ejercicio 3: Función exponencial I = int_0^2 e^x dx")

    f = lambda x: np.exp(x)
    # 1. Valor exacto analítico
    I_exacto = np.exp(2.0) - 1.0
    print(f"1. Valor exacto analítico:")
    print(f"   I = [e^x]_0^2 = e^2 - 1 = {I_exacto:.10f}")

    # 2. Comparación teórica y numérica
    print("\n2. Comparación entre Regla del Trapecio y Rectángulos:")
    print("   * Función f(x) = e^x es estrictamente creciente y convexa (f'(x) = e^x > 0, f''(x) = e^x > 0).")
    print("   * Rectángulo Izquierdo subestima sistemáticamente: E_RL ~ h/2 [f(b) - f(a)] = (e^2 - 1) / n")
    print("   * Rectángulo Derecho sobreestima sistemáticamente: E_RR ~ -h/2 [f(b) - f(a)] = -(e^2 - 1) / n")
    print("   * Trapecio sobreestima con orden superior O(h^2): E_T ~ -(b-a)^3/(12 n^2) f'' = -2/(3 n^2) e^xi")
    print("   * Punto Medio subestima con orden superior O(h^2): E_PM ~ (b-a)^3/(24 n^2) f'' = 1/(3 n^2) e^eta")

    n_test = [2, 4, 8, 16, 32, 64, 128]
    print("\n   Comparación numérica rápida:")
    print(f"   {'n':>5} | {'Trapecio':>14} {'Err Trap':>12} | {'Pto Medio':>14} {'Err PM':>12} | {'Riem. Izq':>14} {'Err RI':>12}")
    for n in n_test:
        tr = trapecio(f, 0, 2, n)
        pm = punto_medio(f, 0, 2, n)
        ri = riemann_izquierda(f, 0, 2, n)
        print(f"   {n:5d} | {tr:14.8f} {abs(tr-I_exacto):12.4e} | {pm:14.8f} {abs(pm-I_exacto):12.4e} | {ri:14.8f} {abs(ri-I_exacto):12.4e}")

    # 3. Subintervalos necesarios para alcanzar tolerancia 10^-6
    tol = 1e-6
    print(f"\n3. Determinación del número de subintervalos para tolerancia tol = {tol:.0e}:")

    # A) Cotas teóricas conservadoras a priori:
    # Max |f''(x)| en [0, 2] = e^2
    M2 = np.exp(2.0)
    # Trapecio: |E_T| <= (b-a)^3 / (12 n^2) * M2 = 8 / (12 n^2) * e^2 = 2 e^2 / (3 n^2) <= 10^-6
    n_teorico_tr = int(np.ceil(np.sqrt(2.0 * M2 / (3.0 * tol))))
    # Punto medio: |E_M| <= (b-a)^3 / (24 n^2) * M2 = e^2 / (3 n^2) <= 10^-6
    n_teorico_pm = int(np.ceil(np.sqrt(M2 / (3.0 * tol))))
    # Rectángulos: |E_R| <= (b-a)^2 / (2 n) * M1 = 4 / (2 n) * e^2 = 2 e^2 / n <= 10^-6
    n_teorico_ri = int(np.ceil(2.0 * M2 / tol))

    print(f"   A) Cotas teóricas conservadoras (a priori):")
    print(f"      - Trapecio:   n >= sqrt(2*e^2 / (3*tol))  => n = {n_teorico_tr}")
    print(f"      - Punto Medio: n >= sqrt(e^2 / (3*tol))    => n = {n_teorico_pm}")
    print(f"      - Rectángulos: n >= 2*e^2 / tol           => n = {n_teorico_ri:,}")

    # B) Búsqueda computacional exacta del menor n:
    # Trapecio exacto
    # Usamos búsqueda binaria o salto acelerado
    n_real_tr = int(np.sqrt((np.exp(2)-1) / (3.0 * tol))) - 10
    while abs(trapecio(f, 0, 2, n_real_tr) - I_exacto) >= tol:
        n_real_tr += 1
    err_real_tr = abs(trapecio(f, 0, 2, n_real_tr) - I_exacto)

    n_real_pm = int(np.sqrt((np.exp(2)-1) / (6.0 * tol))) - 10
    while abs(punto_medio(f, 0, 2, n_real_pm) - I_exacto) >= tol:
        n_real_pm += 1
    err_real_pm = abs(punto_medio(f, 0, 2, n_real_pm) - I_exacto)

    # Rectángulo asintótico
    n_real_ri = int(np.ceil((np.exp(2) - 1.0) / tol))

    print(f"\n   B) Mínimo número de subintervalos real computacional:")
    print(f"      - Trapecio:   n = {n_real_tr} (Error real = {err_real_tr:.4e} < 10^-6)")
    print(f"      - Punto Medio: n = {n_real_pm} (Error real = {err_real_pm:.4e} < 10^-6)")
    print(f"      - Rectángulos: n = {n_real_ri:,} (~6.39 millones de subintervalos)")

    # Gráfica
    fig, ax = plt.subplots(figsize=(8, 5))
    n_sweep = np.logspace(1, 4, 60, dtype=int)
    errs_sweep_tr = [abs(trapecio(f, 0, 2, n) - I_exacto) for n in n_sweep]
    errs_sweep_pm = [abs(punto_medio(f, 0, 2, n) - I_exacto) for n in n_sweep]
    errs_sweep_ri = [abs(riemann_izquierda(f, 0, 2, n) - I_exacto) for n in n_sweep]

    ax.loglog(n_sweep, errs_sweep_ri, color=ACCENT_ORANGE, label='Rectángulo Izquierdo $O(n^{-1})$')
    ax.loglog(n_sweep, errs_sweep_tr, color=PUJ_BLUE, label='Trapecio $O(n^{-2})$')
    ax.loglog(n_sweep, errs_sweep_pm, color=ACCENT_GREEN, label='Punto Medio $O(n^{-2})$')
    ax.axhline(tol, color='red', linestyle=':', lw=2, label=r'Tolerancia requerida $10^{-6}$')
    
    ax.plot([n_real_tr], [err_real_tr], 'ro', markersize=8)
    ax.annotate(f'Trapecio: $n={n_real_tr}$', xy=(n_real_tr, err_real_tr), xytext=(n_real_tr*1.3, err_real_tr*3),
                arrowprops=dict(facecolor='black', arrowstyle='->'), fontweight='bold')

    ax.set_title(r'Decaimiento del Error hacia Tolerancia $10^{-6}$ para $I = \int_0^2 e^x dx$', fontweight='bold')
    ax.set_xlabel('Número de subintervalos ($n$)')
    ax.set_ylabel('Error absoluto')
    ax.grid(True, which='both')
    ax.legend(loc='upper right')

    ruta_fig = os.path.join(DIR_ACTUAL, 'fig_ejercicio3_exponencial.png')
    fig.savefig(ruta_fig, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Gráfica guardada: {ruta_fig}")


# ==============================================================================
# EJERCICIO 4: DATOS EXPERIMENTALES
# ==============================================================================

def ejecutar_ejercicio_4():
    section_header("Ejercicio 4: Integración de datos experimentales")

    x_datos = np.array([0.0, 1.0, 2.0, 3.0])
    y_datos = np.array([1.0, 1.8, 2.4, 4.1])
    h = 1.0
    n = 3

    print("Datos experimentales dados:")
    print("  x   :", x_datos)
    print("  f(x):", y_datos)
    print(f"Paso uniforme h = {h}, número de subintervalos n = {n}.")

    # 1. Regla del trapecio y rectángulos
    # Trapecio compuesto: T = h/2 [y0 + 2 y1 + 2 y2 + y3]
    I_trapecio = (h / 2.0) * (y_datos[0] + 2.0 * y_datos[1] + 2.0 * y_datos[2] + y_datos[3])
    
    # Rectángulos por izquierda: R_L = h [y0 + y1 + y2]
    I_riem_izq = h * np.sum(y_datos[:-1])

    # Rectángulos por derecha: R_R = h [y1 + y2 + y3]
    I_riem_der = h * np.sum(y_datos[1:])

    # Simpson 3/8 (aplicable para 3 subintervalos):
    I_simpson_38 = (3.0 * h / 8.0) * (y_datos[0] + 3.0 * y_datos[1] + 3.0 * y_datos[2] + y_datos[3])

    print("\n1. Resultados numéricos:")
    print(f"   * Regla del Trapecio:      T = {I_trapecio:.4f}")
    print(f"   * Rectángulos (Izquierda): R_L = {I_riem_izq:.4f}")
    print(f"   * Rectángulos (Derecha):   R_R = {I_riem_der:.4f}")
    print(f"   * (Bonus) Simpson 3/8:    S_3/8 = {I_simpson_38:.4f}")
    print(f"   * Promedio R_L y R_R:      (R_L + R_R)/2 = {(I_riem_izq + I_riem_der)/2.0:.4f} (coincide exactamente con Trapecio)")

    # 2. Limitaciones de integrar datos experimentales
    print("\n2. Discusión de limitaciones de integrar datos experimentales:")
    print("   a) Malla fija no refinable: no es posible disminuir h para reducir el error de truncamiento.")
    print("   b) Propagación de ruido e incertidumbre: el error de medición sigma_i se traslada a la integral.")
    print("   c) Información ausente entre nodos: métodos como el punto medio o cuadraturas gaussianas")
    print("      son inaplicables sin interpolación previa.")
    print("   d) Imposibilidad de acotar el error analítico: no se dispone de f''(xi) analítica.")
    print("   e) Riesgo de oscilaciones espurias (fenómeno de Runge) al sobreajustar con polinomios de grado alto.")

    # Gráfica ilustrativa
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Graficar rectángulos izquierda
    for i in range(n):
        ax.bar(x_datos[i], y_datos[i], width=h, align='edge', alpha=0.15,
               edgecolor=ACCENT_ORANGE, color=ACCENT_ORANGE, linestyle='--',
               label='Rectángulos Izquierda ($R_L=5.20$)' if i == 0 else "")

    # Graficar trapecios
    for i in range(n):
        xs = [x_datos[i], x_datos[i], x_datos[i+1], x_datos[i+1]]
        ys = [0, y_datos[i], y_datos[i+1], 0]
        ax.fill(xs, ys, color=PUJ_BLUE, alpha=0.2, edgecolor=PUJ_BLUE,
                label='Trapecios ($T=6.75$)' if i == 0 else "")

    # Spline cúbico como aproximación suave continua
    from scipy.interpolate import CubicSpline
    cs = CubicSpline(x_datos, y_datos, bc_type='natural')
    x_smooth = np.linspace(0, 3, 200)
    ax.plot(x_smooth, cs(x_smooth), 'g-', alpha=0.8, lw=1.8, label=f'Spline cúbico suave (Área $\\approx {cs.integrate(0, 3):.4f}$)')

    # Puntos experimentales
    ax.plot(x_datos, y_datos, 'ro', markersize=8, label='Datos experimentales $(x_i, f(x_i))$')
    for xi, yi in zip(x_datos, y_datos):
        ax.annotate(f'({xi:.0f}, {yi:.1f})', (xi, yi), textcoords="offset points", xytext=(0, 10), ha='center', fontweight='bold')

    ax.set_title('Aproximación de la Integral con Datos Experimentales', fontweight='bold')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$f(x)$')
    ax.set_xlim(-0.2, 3.5)
    ax.set_ylim(0, 5.0)
    ax.grid(True)
    ax.legend(loc='upper left')

    ruta_fig = os.path.join(DIR_ACTUAL, 'fig_ejercicio4_datos_experimentales.png')
    fig.savefig(ruta_fig, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Gráfica guardada: {ruta_fig}")


# ==============================================================================
# EJERCICIO 5: OSCILACIONES DE ALTA FRECUENCIA
# ==============================================================================

def ejecutar_ejercicio_5():
    section_header("Ejercicio 5: Oscilaciones de alta frecuencia I = int_0^1 e^-x sin(40x^2 + 5x) dx")

    f = lambda x: np.exp(-x) * np.sin(40.0 * x**2 + 5.0 * x)
    
    # Referencia de alta precisión mediante cuadratura adaptativa SciPy
    I_ref, err_ref = spi.quad(f, 0, 1, epsabs=1e-14, epsrel=1e-14, limit=500)
    print(f"Referencia numérica de alta precisión:")
    print(f"   I_ref = {I_ref:.12f} (estimado error quad: {err_ref:.2e})")

    # 1. Regiones de mayor oscilación
    print("\n1. Análisis analítico de frecuencia instantánea:")
    print("   Fase: theta(x) = 40 x^2 + 5 x")
    print("   Frecuencia angular instantánea: omega(x) = d theta / dx = 80 x + 5")
    print("   * En x = 0: omega(0) = 5 rad/s, longitud de onda lambda_0 = 2 pi / 5 ~= 1.26")
    print("   * En x = 1: omega(1) = 85 rad/s, longitud de onda lambda_1 = 2 pi / 85 ~= 0.0739")
    print("   Por tanto, la frecuencia se multiplica por 17 hacia el extremo derecho x -> 1.")

    # 2, 3, 4. Evaluación para n = 20, 40, 80, 160, 320
    n_valores = [20, 40, 80, 160, 320]
    
    print("\n2, 3, 4. Comparación de Trapecio y Simpson 1/3 compuesto:")
    encabezado = f"{'n':>5} | {'Trapecio':>14} {'Err Trap':>12} | {'Simpson 1/3':>14} {'Err Simp':>12}"
    print("-" * len(encabezado))
    print(encabezado)
    print("-" * len(encabezado))

    errs_tr, errs_sim = [], []
    for n in n_valores:
        tr = trapecio(f, 0, 1, n)
        sim = simpson_13(f, 0, 1, n)
        err_t = abs(tr - I_ref)
        err_s = abs(sim - I_ref)
        errs_tr.append(err_t)
        errs_sim.append(err_s)
        print(f"{n:5d} | {tr:14.8f} {err_t:12.4e} | {sim:14.8f} {err_s:12.4e}")
    print("-" * len(encabezado))

    # 5. Orden experimental de convergencia
    print("\n5. Estimación del orden experimental de convergencia p = log2(E_n / E_{2n}):")
    for i in range(len(n_valores) - 1):
        p_t = np.log2(errs_tr[i] / errs_tr[i+1])
        p_s = np.log2(errs_sim[i] / errs_sim[i+1])
        print(f"   Intervalo n = [{n_valores[i]:3d}, {n_valores[i+1]:3d}]: p_Trapecio = {p_t:.2f} (teórico 2), p_Simpson = {p_s:.2f} (teórico 4)")

    # 6. Menor número de subintervalos para tolerancia 10^-6
    tol = 1e-6
    n_min_tr = 20
    while abs(trapecio(f, 0, 1, n_min_tr) - I_ref) >= tol:
        n_min_tr += 1
    err_min_tr = abs(trapecio(f, 0, 1, n_min_tr) - I_ref)

    n_min_sim = 20
    while abs(simpson_13(f, 0, 1, n_min_sim) - I_ref) >= tol:
        n_min_sim += 2
    err_min_sim = abs(simpson_13(f, 0, 1, n_min_sim) - I_ref)

    print(f"\n6. Menor número de subintervalos para tol = {tol:.0e}:")
    print(f"   * Trapecio:   n = {n_min_tr}  (Error = {err_min_tr:.4e} < 10^-6)")
    print(f"   * Simpson 1/3: n = {n_min_sim}  (Error = {err_min_sim:.4e} < 10^-6)")
    print(f"   Simpson 1/3 alcanza la tolerancia con un factor de {n_min_tr / n_min_sim:.1f}x menos intervalos que Trapecio.")

    # Gráficas
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Subplot 1: Función oscilatoria con envolventes
    x_plot = np.linspace(0, 1, 1000)
    y_plot = f(x_plot)
    env_sup = np.exp(-x_plot)
    env_inf = -np.exp(-x_plot)

    axes[0].plot(x_plot, y_plot, color=PUJ_BLUE, lw=1.5, label=r'$f(x) = e^{-x} \sin(40x^2 + 5x)$')
    axes[0].plot(x_plot, env_sup, 'k--', alpha=0.5, label=r'Envolvente $\pm e^{-x}$')
    axes[0].plot(x_plot, env_inf, 'k--', alpha=0.5)
    axes[0].set_title('Función Oscilatoria con Frecuencia Creciente', fontweight='bold')
    axes[0].set_xlabel('$x$')
    axes[0].set_ylabel('$f(x)$')
    axes[0].grid(True)
    axes[0].legend(loc='lower left')

    # Subplot 2: Gráfica Log-Log de convergencia
    axes[1].loglog(n_valores, errs_tr, 'o-', color=PUJ_BLUE, lw=2, label='Trapecio Compuesto')
    axes[1].loglog(n_valores, errs_sim, 's-', color=ACCENT_RED, lw=2, label='Simpson 1/3 Compuesto')
    axes[1].axhline(tol, color='gray', linestyle=':', label=r'Tolerancia $10^{-6}$')

    # Guías asintóticas
    n_arr = np.array(n_valores)
    axes[1].loglog(n_arr, errs_tr[-1] * (n_arr[-1] / n_arr)**2, 'b--', alpha=0.5, label='Pendiente teórica $p=2$')
    axes[1].loglog(n_arr, errs_sim[-1] * (n_arr[-1] / n_arr)**4, 'r--', alpha=0.5, label='Pendiente teórica $p=4$')

    axes[1].set_title('Convergencia y Orden Experimental (Escala Log-Log)', fontweight='bold')
    axes[1].set_xlabel('Número de subintervalos ($n$)')
    axes[1].set_ylabel('Error absoluto $|I - I_n|$')
    axes[1].grid(True, which='both')
    axes[1].legend()

    ruta_fig = os.path.join(DIR_ACTUAL, 'fig_ejercicio5_oscilaciones.png')
    fig.savefig(ruta_fig, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Gráfica guardada: {ruta_fig}")


# ==============================================================================
# EJERCICIO 6: INTEGRAL ALTAMENTE OSCILATORIA
# ==============================================================================

def ejecutar_ejercicio_6():
    section_header("Ejercicio 6: Integral altamente oscilatoria I = int_0^1 sin(1/(x + 0.01)) dx")

    f = lambda x: np.sin(1.0 / (x + 0.01))

    # Referencia de alta precisión
    I_ref, err_ref = spi.quad(f, 0, 1, epsabs=1e-14, epsrel=1e-14, limit=1000)
    print(f"Referencia numérica de alta precisión:")
    print(f"   I_ref = {I_ref:.12f} (estimado error quad: {err_ref:.2e})")

    # 1. Frecuencia de oscilación a lo largo del intervalo
    print("\n1. Análisis de frecuencia de oscilación a lo largo del intervalo:")
    print("   Fase: theta(x) = 1 / (x + 0.01)")
    print("   Frecuencia angular instantánea: omega(x) = |d theta/dx| = 1 / (x + 0.01)^2")
    print("   * En x = 0.0:  omega(0) = 1 / (0.01)^2 = 10,000 rad/s (periodo T ~= 6.28 x 10^-4)")
    print("   * En x = 0.05: omega(0.05) = 1 / (0.06)^2 ~= 278 rad/s (periodo T ~= 0.0226)")
    print("   * En x = 0.1:  omega(0.1) = 1 / (0.11)^2 ~= 82.6 rad/s (periodo T ~= 0.076)")
    print("   * En x = 1.0:  omega(1) = 1 / (1.01)^2 ~= 0.98 rad/s (periodo T ~= 6.41)")
    print("   La frecuencia cae en 4 órdenes de magnitud desde x = 0 hasta x = 1.")

    # 2, 3, 4. Evaluación para n = 50, 100, 200, 400, 800, 1600
    n_valores = [50, 100, 200, 400, 800, 1600]
    
    print("\n2, 3, 4. Aproximación con Trapecio y Simpson 1/3 (Malla Uniforme):")
    encabezado = f"{'n':>5} | {'Trapecio':>14} {'Err Trap':>12} | {'Simpson 1/3':>14} {'Err Simp':>12}"
    print("-" * len(encabezado))
    print(encabezado)
    print("-" * len(encabezado))

    errs_tr, errs_sim = [], []
    for n in n_valores:
        tr = trapecio(f, 0, 1, n)
        sim = simpson_13(f, 0, 1, n)
        err_t = abs(tr - I_ref)
        err_s = abs(sim - I_ref)
        errs_tr.append(err_t)
        errs_sim.append(err_s)
        print(f"{n:5d} | {tr:14.8f} {err_t:12.4e} | {sim:14.8f} {err_s:12.4e}")
    print("-" * len(encabezado))

    # Órdenes experimentales erráticos
    print("\nÓrdenes experimentales aparentes:")
    for i in range(len(n_valores) - 1):
        p_t = np.log2(errs_tr[i] / errs_tr[i+1])
        p_s = np.log2(errs_sim[i] / errs_sim[i+1])
        print(f"   Intervalo n = [{n_valores[i]:4d}, {n_valores[i+1]:4d}]: p_Trapecio = {p_t:6.2f}, p_Simpson = {p_s:6.2f}")

    # 5. Justificación de la ineficiencia de mallas uniformes
    print("\n5. Justificación matemática de la ineficiencia de la malla uniforme:")
    print("   * Criterio de Nyquist-Shannon: se requieren al menos 2 puntos por periodo: h < pi / omega_max.")
    print("     Para omega_max = 10,000 rad/s, se requiere h < 3.14 x 10^-4  =>  n > 3,183.")
    print("   * Con n <= 1600, los nodos cerca de x = 0 sufren submuestreo severo (aliasing),")
    print("     generando errores oscilatorios no asintóticos.")
    print("   * Por contrapartida, en [0.5, 1.0] (donde omega <= 4 rad/s), un paso h uniforme")
    print("     desperdicia cientos de puntos en una región donde la función es prácticamente cuasi-lineal.")

    # 6. Estrategia de refinamiento local y comparación
    print("\n6. Estrategia de refinamiento local (Cuadratura Adaptativa de Simpson):")
    tol_adapt_list = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
    print(f"   {'Tol':>8} | {'Valor Adapt':>14} {'Error Real':>12} | {'Evaluaciones':>12} | {'% Nodos en [0, 0.1]':>20}")
    print("   " + "-" * 72)
    evals_adapt = []
    errs_adapt = []
    for tol_a in tol_adapt_list:
        val_a, evals_a, nodos_a = cuadratura_adaptativa_simpson(f, 0, 1, tol_a)
        err_a = abs(val_a - I_ref)
        pct_zona = np.mean(nodos_a <= 0.1) * 100.0
        evals_adapt.append(evals_a)
        errs_adapt.append(err_a)
        print(f"   {tol_a:8.0e} | {val_a:14.8f} {err_a:12.4e} | {evals_a:12d} | {pct_zona:19.1f}%")
    print("   " + "-" * 72)

    # Gráfica
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel (0, 0): Función global y zoom
    x_dense = np.linspace(0, 1, 4000)
    y_dense = f(x_dense)
    axes[0, 0].plot(x_dense, y_dense, color=PUJ_BLUE, lw=1.2)
    axes[0, 0].set_title(r'(a) Función Global $f(x) = \sin(1/(x + 0.01))$ en $[0, 1]$', fontweight='bold')
    axes[0, 0].set_xlabel('$x$')
    axes[0, 0].set_ylabel('$f(x)$')
    axes[0, 0].grid(True)

    # Panel (0, 1): Zoom en zona hiper-oscilatoria [0, 0.05]
    x_zoom = np.linspace(0, 0.05, 3000)
    y_zoom = f(x_zoom)
    axes[0, 1].plot(x_zoom, y_zoom, color=ACCENT_RED, lw=1.2)
    axes[0, 1].set_title(r'(b) Zoom en Zona Crítica $[0, 0.05]$ ($\omega \approx 10{,}000$ rad/s)', fontweight='bold')
    axes[0, 1].set_xlabel('$x$')
    axes[0, 1].set_ylabel('$f(x)$')
    axes[0, 1].grid(True)

    # Panel (1, 0): Frecuencia instantánea omega(x) en escala semilogarítmica
    x_freq = np.linspace(0, 1, 500)
    omega_vals = 1.0 / (x_freq + 0.01)**2
    axes[1, 0].semilogy(x_freq, omega_vals, color=ACCENT_PURPLE, lw=2)
    axes[1, 0].set_title(r'(c) Frecuencia Instantánea $\omega(x) = (x+0.01)^{-2}$ (Escala Log)', fontweight='bold')
    axes[1, 0].set_xlabel('$x$')
    axes[1, 0].set_ylabel(r'$\omega(x)$ [rad/s]')
    axes[1, 0].grid(True, which='both')

    # Panel (1, 1): Comparación de convergencia: Malla Uniforme vs Adaptativa
    evals_uniformes = [n + 1 for n in n_valores]
    axes[1, 1].loglog(evals_uniformes, errs_tr, 'o--', color=PUJ_BLUE, label='Trapecio Uniforme')
    axes[1, 1].loglog(evals_uniformes, errs_sim, 's--', color=ACCENT_ORANGE, label='Simpson 1/3 Uniforme')
    axes[1, 1].loglog(evals_adapt, errs_adapt, 'D-', color=ACCENT_GREEN, lw=2.2, label='Simpson Adaptativo (Local)')
    axes[1, 1].axhline(1e-6, color='red', linestyle=':', label=r'Tolerancia $10^{-6}$')
    axes[1, 1].set_title('(d) Desempeño: Malla Uniforme vs Refinamiento Adaptativo', fontweight='bold')
    axes[1, 1].set_xlabel('Número total de evaluaciones de la función')
    axes[1, 1].set_ylabel('Error absoluto')
    axes[1, 1].legend()
    axes[1, 1].grid(True, which='both')

    ruta_fig = os.path.join(DIR_ACTUAL, 'fig_ejercicio6_altamente_oscilatoria.png')
    fig.savefig(ruta_fig, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Gráfica guardada: {ruta_fig}")


# ==============================================================================
# FUNCIÓN PRINCIPAL DE EJECUCIÓN
# ==============================================================================

def main():
    print("=" * 80)
    print(" INICIO DE LA SOLUCIÓN COMPUTACIONAL - TALLER DE INTEGRACIÓN NUMÉRICA ")
    print("=" * 80)

    ejecutar_ejercicio_1()
    ejecutar_ejercicio_2()
    ejecutar_ejercicio_3()
    ejecutar_ejercicio_4()
    ejecutar_ejercicio_5()
    ejecutar_ejercicio_6()

    print("\n" + "=" * 80)
    print(" TODOS LOS EJERCICIOS FUERON EJECUTADOS CON ÉXITO. ")
    print(" Figuras generadas y listas para su inclusión en el reporte LaTeX. ")
    print("=" * 80)


if __name__ == '__main__':
    main()
