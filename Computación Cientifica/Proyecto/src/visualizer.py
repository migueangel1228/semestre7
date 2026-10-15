"""
Módulo de visualización científica para generar figuras de alta calidad
sobre el análisis del Exponente de Lyapunov y bifurcaciones.
"""

import os
from typing import Optional
import numpy as np
import matplotlib.pyplot as plt

from .maps import LogisticMap, TentMap
from .lyapunov import (
    lyapunov_logistic_spectrum,
    lyapunov_tent_spectrum,
    analytical_lyapunov_tent,
    analytical_lyapunov_logistic_fixed_point,
    calculate_lyapunov_exponent,
)
from .bifurcation import calculate_bifurcation_data


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
