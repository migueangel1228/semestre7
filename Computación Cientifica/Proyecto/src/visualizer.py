"""
Módulo de visualización científica para generar figuras de alta calidad
sobre el análisis del Exponente de Lyapunov y bifurcaciones.
"""

import os
from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from .maps import LogisticMap, TentMap
from .lyapunov import (
    lyapunov_logistic_spectrum,
    lyapunov_tent_spectrum,
    analytical_lyapunov_tent,
    analytical_lyapunov_logistic_fixed_point,
    calculate_lyapunov_exponent,
)
from .bifurcation import calculate_bifurcation_data
from .sensitivity import simulate_trajectory_divergence
from .cobweb import generate_cobweb_data


def setup_style():
    """Configura parámetros estéticos modernos para matplotlib."""
    plt.rcParams.update({
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 14,
        'axes.grid': True,
        'grid.alpha': 0.35,
        'grid.linestyle': '--',
        'lines.linewidth': 1.6,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
    })


def plot_logistic_complete(save_path: str = "figures/logistic_bifurcation_lyapunov.png"):
    """
    Genera el diagrama combinado de Bifurcación y Exponente de Lyapunov
    para el Mapa Logístico en r in [2.5, 4.0].
    """
    setup_style()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8.5), sharex=True,
                                   gridspec_kw={'height_ratios': [1.2, 1.0], 'hspace': 0.08})

    r_vals = np.linspace(2.5, 4.0, 1500)

    # 1. Diagrama de Bifurcación
    r_bif, x_bif = calculate_bifurcation_data('logistic', r_vals, n_transient=400, n_retained=150)
    ax1.scatter(r_bif, x_bif, s=0.08, c='#1f77b4', alpha=0.3, rasterized=True)
    ax1.set_ylabel(r"Estado en atractor $x^*$")
    ax1.set_ylim(-0.02, 1.02)
    ax1.set_title(r"Mapa Logístico: $x_{n+1} = r x_n (1 - x_n)$ — Bifurcación y Exponente de Lyapunov",
                  fontweight='bold', pad=12)

    # Anotaciones en la bifurcación
    ax1.axvline(3.0, color='gray', linestyle=':', alpha=0.7)
    ax1.text(3.01, 0.1, r"$r_1 = 3.0$" "\n(Periodo 2)", fontsize=9, color='#333333')

    ax1.axvline(1.0 + np.sqrt(6), color='gray', linestyle=':', alpha=0.7)
    ax1.text(1.0 + np.sqrt(6) + 0.01, 0.25, r"$r_2 \approx 3.449$" "\n(Periodo 4)", fontsize=9, color='#333333')

    ax1.axvline(3.5699456, color='#d62728', linestyle='--', alpha=0.8)
    ax1.text(3.575, 0.85, r"$r_\infty \approx 3.5699$" "\n(Inicio del Caos)", fontsize=9, color='#d62728', fontweight='bold')

    ax1.axvspan(3.8284, 3.856, color='#2ca02c', alpha=0.15, label='Ventana Periodo 3')
    ax1.text(3.83, 0.08, "Ventana\nPeriodo 3", fontsize=8.5, color='#1b5e20', ha='center')

    # 2. Exponente de Lyapunov
    lyap_vals = lyapunov_logistic_spectrum(r_vals, n_transient=400, n_iterations=1500)
    
    # Línea cero
    ax2.axhline(0.0, color='black', linestyle='-', linewidth=1.0, alpha=0.8)
    ax2.axvline(3.5699456, color='#d62728', linestyle='--', alpha=0.8)

    # Zonas de estabilidad y caos
    ax2.fill_between(r_vals, lyap_vals, 0, where=(lyap_vals > 0),
                     color='#ff9999', alpha=0.45, label=r'Régimen Caótico ($\lambda > 0$)')
    ax2.fill_between(r_vals, lyap_vals, 0, where=(lyap_vals <= 0),
                     color='#99ff99', alpha=0.4, label=r'Régimen Estable / Periódico ($\lambda \leq 0$)')

    ax2.plot(r_vals, lyap_vals, color='#0d47a1', linewidth=1.2, label=r'$\lambda(r)$ numérico')

    # Solución analítica en punto fijo r in [2.5, 3.0]
    r_analyt = np.linspace(2.5, 3.0, 100)
    lyap_analyt = analytical_lyapunov_logistic_fixed_point(r_analyt)
    ax2.plot(r_analyt, lyap_analyt, 'k--', linewidth=1.5, label=r'$\ln|2 - r|$ (Analítico)')

    # Anotación r = 4.0
    lambda_at_4 = np.log(2.0)
    ax2.scatter([4.0], [lambda_at_4], color='#d62728', s=45, zorder=5)
    ax2.annotate(r"$r=4.0 \rightarrow \lambda = \ln(2) \approx 0.6931$",
                 xy=(4.0, lambda_at_4), xytext=(3.60, 0.75),
                 arrowprops=dict(arrowstyle="->", color='#d62728', lw=1.2),
                 fontsize=9.5, color='#d62728', fontweight='bold')

    ax2.set_xlabel(r"Parámetro de control $r$")
    ax2.set_ylabel(r"Exponente de Lyapunov $\lambda$")
    ax2.set_xlim(2.5, 4.0)
    ax2.set_ylim(-1.6, 0.85)
    ax2.legend(loc='lower left', framealpha=0.9)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f"[OK] Gráfica guardada: {save_path}")


def plot_tent_complete(save_path: str = "figures/tent_bifurcation_lyapunov.png"):
    """
    Genera el diagrama combinado de Bifurcación y Exponente de Lyapunov
    para el Mapa de Tienda en p in [0.0, 2.0].
    """
    setup_style()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8.5), sharex=True,
                                   gridspec_kw={'height_ratios': [1.2, 1.0], 'hspace': 0.08})

    p_vals = np.linspace(0.01, 2.0, 1200)

    # 1. Diagrama de Bifurcación
    p_bif, x_bif = calculate_bifurcation_data('tent', p_vals, n_transient=300, n_retained=150, x0=0.35)
    ax1.scatter(p_bif, x_bif, s=0.08, c='#ff7f0e', alpha=0.3, rasterized=True)
    ax1.set_ylabel(r"Estado en atractor $x^*$")
    ax1.set_ylim(-0.02, 1.02)
    ax1.set_title(r"Modelo de Tienda (Tent Map): $x_{n+1} = f(x_n; p)$ — Bifurcación y $\lambda(p)$",
                  fontweight='bold', pad=12)

    ax1.axvline(1.0, color='gray', linestyle='--', alpha=0.8)
    ax1.text(1.02, 0.8, r"$p = 1.0$" "\n(Frontera de Caos)", fontsize=9.5, color='#333333', fontweight='bold')

    # 2. Exponente de Lyapunov
    p_lyap = np.linspace(0.05, 2.0, 600)
    lyap_num = lyapunov_tent_spectrum(p_lyap, n_transient=200, n_iterations=1200)
    lyap_ana = analytical_lyapunov_tent(p_lyap)

    ax2.axhline(0.0, color='black', linestyle='-', linewidth=1.0, alpha=0.8)
    ax2.axvline(1.0, color='gray', linestyle='--', alpha=0.8)

    ax2.fill_between(p_lyap, lyap_ana, 0, where=(lyap_ana > 0),
                     color='#ff9999', alpha=0.45, label=r'Caótico ($\lambda > 0$, $p > 1$)')
    ax2.fill_between(p_lyap, lyap_ana, 0, where=(lyap_ana <= 0),
                     color='#99ff99', alpha=0.4, label=r'Contractivo ($\lambda \leq 0$, $p \leq 1$)')

    ax2.plot(p_lyap, lyap_ana, color='#b71c1c', linewidth=2.0, label=r'$\lambda(p) = \ln(p)$ (Exacto / Analítico)')
    ax2.plot(p_lyap[::6], lyap_num[::6], 'o', color='#1565c0', markersize=3.5, alpha=0.85,
             label=r'$\lambda(p)$ cálculo numérico de órbita')

    # Marcador en p=2
    lambda_at_2 = np.log(2.0)
    ax2.scatter([2.0], [lambda_at_2], color='#d62728', s=45, zorder=5)
    ax2.annotate(r"$p=2.0 \rightarrow \lambda = \ln(2) \approx 0.6931$",
                 xy=(2.0, lambda_at_2), xytext=(1.45, 0.78),
                 arrowprops=dict(arrowstyle="->", color='#d62728', lw=1.2),
                 fontsize=9.5, color='#d62728', fontweight='bold')

    ax2.set_xlabel(r"Parámetro de control $p$")
    ax2.set_ylabel(r"Exponente de Lyapunov $\lambda$")
    ax2.set_xlim(0.0, 2.02)
    ax2.set_ylim(-2.0, 0.95)
    ax2.legend(loc='lower right', framealpha=0.9)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f"[OK] Gráfica guardada: {save_path}")


def plot_comparison(save_path: str = "figures/lyapunov_comparison_logistic_vs_tent.png"):
    """
    Compara ambos modelos lado a lado y resalta la conjugación topológica
    en los puntos de máximo caos (r=4, p=2).
    """
    setup_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    # Panel 1: Exponentes sobre sus respectivos dominios
    r_vals = np.linspace(2.5, 4.0, 500)
    lyap_log = lyapunov_logistic_spectrum(r_vals, n_transient=400, n_iterations=1500)
    ax1.plot(r_vals, lyap_log, color='#1565c0', lw=1.5, label='Logístico: $\\lambda(r)$')
    ax1.axhline(0, color='black', lw=0.9, ls='--')
    ax1.axhline(np.log(2.0), color='#d62728', lw=1.2, ls=':', label=r'$\lambda = \ln(2) \approx 0.6931$')
    ax1.set_title("Mapa Logístico: $\\lambda(r)$")
    ax1.set_xlabel("Parámetro $r$")
    ax1.set_ylabel("Exponente de Lyapunov $\\lambda$")
    ax1.set_ylim(-1.5, 0.85)
    ax1.legend(loc='lower left')

    # Panel 2: Mapa de Tienda
    p_vals = np.linspace(0.1, 2.0, 500)
    lyap_tent = analytical_lyapunov_tent(p_vals)
    ax2.plot(p_vals, lyap_tent, color='#e65100', lw=2.0, label='Tienda: $\\lambda(p) = \\ln(p)$')
    ax2.axhline(0, color='black', lw=0.9, ls='--')
    ax2.axhline(np.log(2.0), color='#d62728', lw=1.2, ls=':', label=r'$\lambda = \ln(2) \approx 0.6931$')
    ax2.set_title("Modelo de Tienda: $\\lambda(p)$")
    ax2.set_xlabel("Parámetro $p$")
    ax2.set_ylabel("Exponente de Lyapunov $\\lambda$")
    ax2.set_ylim(-1.5, 0.85)
    ax2.legend(loc='lower right')

    fig.suptitle(r"Comparación de Exponentes de Lyapunov y Teorema de Conjugación Topológica: $\lambda(r=4) = \lambda(p=2) = \ln(2)$",
                 fontweight='bold', y=0.98)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f"[OK] Gráfica guardada: {save_path}")


def plot_sensitivity_analysis(save_path: str = "figures/sensitivity_initial_conditions.png"):
    """
    Demuestra la divergencia exponencial de dos trayectorias con separación delta0 = 10^-8.
    """
    setup_style()
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5), gridspec_kw={'hspace': 0.28, 'wspace': 0.22})

    # 1. Logístico r=4.0
    log_map = LogisticMap(r=4.0)
    data_log = simulate_trajectory_divergence(log_map, x0=0.314159, delta0=1e-8, n_steps=45)

    # Subplot 1: Trayectorias
    axes[0, 0].plot(data_log['steps'], data_log['x1'], 'b-o', markersize=3.5, label=r'$x_0 = 0.314159$')
    axes[0, 0].plot(data_log['steps'], data_log['x2'], 'r--s', markersize=3.0, alpha=0.7,
                    label=r'$x_0 + 10^{-8}$')
    axes[0, 0].set_title(r"Mapa Logístico ($r=4$): Sensibilidad Temporal", fontweight='bold')
    axes[0, 0].set_xlabel("Iteración $n$")
    axes[0, 0].set_ylabel("Estado $x_n$")
    axes[0, 0].legend(loc='upper right')

    # Subplot 2: Divergencia logarítmica
    axes[0, 1].plot(data_log['steps'], data_log['log_delta'], 'g-d', markersize=3.5, label=r'$\ln|\delta_n|$ empírico')
    # Pendiente teórica: ln(delta0) + lambda * n
    n_lin = np.arange(0, 25)
    theory_line = np.log(1e-8) + np.log(2.0) * n_lin
    axes[0, 1].plot(n_lin, theory_line, 'k--', label=r'Pendiente teórica: $\lambda = \ln(2) \approx 0.693$')
    axes[0, 1].axhline(np.log(0.5), color='gray', linestyle=':', label='Saturación de atractor')
    axes[0, 1].set_title(r"Divergencia Exponencial $|\delta_n| \approx \delta_0 e^{\lambda n}$", fontweight='bold')
    axes[0, 1].set_xlabel("Iteración $n$")
    axes[0, 1].set_ylabel(r"$\ln|\delta_n|$")
    axes[0, 1].legend(loc='lower right')

    # 2. Tienda p=2.0
    tent_map = TentMap(p=2.0)
    data_tent = simulate_trajectory_divergence(tent_map, x0=0.314159, delta0=1e-8, n_steps=45)

    axes[1, 0].plot(data_tent['steps'], data_tent['x1'], 'b-o', markersize=3.5, label=r'$x_0 = 0.314159$')
    axes[1, 0].plot(data_tent['steps'], data_tent['x2'], 'r--s', markersize=3.0, alpha=0.7,
                    label=r'$x_0 + 10^{-8}$')
    axes[1, 0].set_title(r"Mapa de Tienda ($p=2$): Sensibilidad Temporal", fontweight='bold')
    axes[1, 0].set_xlabel("Iteración $n$")
    axes[1, 0].set_ylabel("Estado $x_n$")
    axes[1, 0].legend(loc='upper right')

    axes[1, 1].plot(data_tent['steps'], data_tent['log_delta'], 'm-d', markersize=3.5, label=r'$\ln|\delta_n|$ empírico')
    theory_line_tent = np.log(1e-8) + np.log(2.0) * n_lin
    axes[1, 1].plot(n_lin, theory_line_tent, 'k--', label=r'Pendiente teórica: $\lambda = \ln(2) \approx 0.693$')
    axes[1, 1].axhline(np.log(0.5), color='gray', linestyle=':', label='Saturación de atractor')
    axes[1, 1].set_title(r"Divergencia Exponencial $|\delta_n| \approx \delta_0 e^{\lambda n}$", fontweight='bold')
    axes[1, 1].set_xlabel("Iteración $n$")
    axes[1, 1].set_ylabel(r"$\ln|\delta_n|$")
    axes[1, 1].legend(loc='lower right')

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f"[OK] Gráfica guardada: {save_path}")


def plot_cobweb_showcase(save_path: str = "figures/cobweb_diagrams.png"):
    """
    Genera 4 diagramas de telaraña comparativos en diferentes regímenes dinámicos.
    """
    setup_style()
    fig, axes = plt.subplots(2, 2, figsize=(11, 10), gridspec_kw={'hspace': 0.25, 'wspace': 0.2})

    cases = [
        (LogisticMap(2.8), 0.2, 35, r"Logístico $r=2.8$ (Punto Fijo Estable)", axes[0, 0]),
        (LogisticMap(3.5), 0.2, 40, r"Logístico $r=3.5$ (Ciclo de Periodo 4)", axes[0, 1]),
        (LogisticMap(3.9), 0.2, 60, r"Logístico $r=3.9$ (Caos Determinista)", axes[1, 0]),
        (TentMap(1.9), 0.2, 60, r"Tienda $p=1.9$ (Caos Determinista)", axes[1, 1]),
    ]

    for model, x0, steps, title, ax in cases:
        xc, yc, cwx, cwy = generate_cobweb_data(model, x0=x0, n_steps=steps)
        ax.plot(xc, yc, 'b-', lw=1.8, label=r'$y = f(x)$')
        ax.plot(xc, xc, 'k--', lw=1.0, alpha=0.7, label=r'$y = x$')
        ax.plot(cwx, cwy, 'r-', lw=0.9, alpha=0.75, label='Órbita')
        ax.set_title(title, fontweight='bold', fontsize=11)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel(r"$x_n$")
        ax.set_ylabel(r"$x_{n+1}$")
        ax.legend(loc='upper left', fontsize=8.5)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f"[OK] Gráfica guardada: {save_path}")


def plot_topological_conjugacy(save_path: str = "figures/conjugacy_logistic_tent.png"):
    """
    Visualiza la conjugación topológica entre el Mapa de Tienda (p=2) y el Mapa Logístico (r=4):
        x = h(y) = sin^2(pi * y / 2)
    """
    setup_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.2))

    y = np.linspace(0, 1, 500)
    h_y = np.sin(np.pi * y / 2.0) ** 2

    ax1.plot(y, h_y, color='#6a1b9a', lw=2.2, label=r'$h(y) = \sin^2\left(\frac{\pi y}{2}\right)$')
    ax1.plot(y, y, 'k--', lw=1.0, alpha=0.5, label=r'$y = x$')
    ax1.set_title(r"Difeomorfismo de Conjugación: $h:[0, 1] \to [0, 1]$", fontweight='bold')
    ax1.set_xlabel(r"Variable Tienda $y \in [0, 1]$")
    ax1.set_ylabel(r"Variable Logístico $x = h(y)$")
    ax1.legend(loc='upper left')

    # Histogramas de medidas invariantes
    tent = TentMap(2.0)
    logm = LogisticMap(4.0)

    n_samples = 80000
    traj_tent = tent.trajectory(0.314159, n_samples)[1000:]
    traj_logm = logm.trajectory(0.314159, n_samples)[1000:]

    ax2.hist(traj_tent, bins=60, density=True, alpha=0.45, color='#ff7f0e',
             label=r'Tienda $p=2$: Medida Uniforme $\rho(y) = 1$')
    ax2.hist(traj_logm, bins=60, density=True, alpha=0.45, color='#1f77b4',
             label=r'Logístico $r=4$: $\rho(x) = \frac{1}{\pi \sqrt{x(1-x)}}$')

    # Curva analítica para el logístico
    x_grid = np.linspace(0.005, 0.995, 300)
    density_log = 1.0 / (np.pi * np.sqrt(x_grid * (1.0 - x_grid)))
    ax2.plot(x_grid, density_log, 'b-', lw=1.5, label='Densidad Invariante Exacta')

    ax2.set_title(r"Densidades de Probabilidad Invariantes", fontweight='bold')
    ax2.set_xlabel("Valor en el Atractor")
    ax2.set_ylabel(r"Densidad $\rho$")
    ax2.set_xlim(0.0, 1.0)
    ax2.set_ylim(0, 4.0)
    ax2.legend(loc='upper center', fontsize=9)

    fig.suptitle(r"Conjugación Topológica: $L(h(y)) = h(T(y)) \rightarrow \lambda_{\mathrm{Log}}(4) = \lambda_{\mathrm{Tienda}}(2) = \ln(2)$",
                 fontweight='bold', y=0.98)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f"[OK] Gráfica guardada: {save_path}")
