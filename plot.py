"""Plotting functions for solver profiles and convergence diagnostics."""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def plot_eq3_residual_convergence(residual_history, cell_index=None, x_position_m=None):
    """Display/save local nonlinear Eq. 3 residual history, including failures."""
    values = np.asarray(residual_history, dtype=float)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] != 2:
        return
    evaluations = np.arange(1, values.shape[0] + 1)
    location = ""
    if cell_index is not None:
        location += f"cell {cell_index}"
    if x_position_m is not None:
        location += f", x={x_position_m:.6g} m"
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(8, 7))
    axes[0].semilogy(evaluations, np.maximum(np.abs(values[:, 0]), 1e-16))
    axes[0].set_ylabel("|gas-wall residual| / scale")
    axes[0].set_title(f"Eq. 3 local solver residuals ({location})")
    axes[0].grid(True, which="both")
    axes[1].semilogy(evaluations, np.maximum(np.abs(values[:, 1]), 1e-16))
    axes[1].set_ylabel("|wall-coolant residual| / scale")
    axes[1].set_xlabel("Residual function evaluation")
    axes[1].grid(True, which="both")
    fig.tight_layout()
    output = Path(__file__).with_name("eq3_residual_convergence.png")
    fig.savefig(output, dpi=150)
    print(f"Eq. 3 residual graph saved to: {output}")
    plt.show()

def plot_axial_eq3_residuals(x_cell, gas_wall_residual, wall_coolant_residual):
    """Display/save absolute Eq. 3 heat-path residuals along the mesh."""
    fig, ax = plt.subplots()
    ax.plot(x_cell, np.abs(gas_wall_residual), label="|gas-side − wall conduction|")
    ax.plot(x_cell, np.abs(wall_coolant_residual), label="|wall conduction − coolant-side|")
    ax.set_xlabel("Axial position, x [m]")
    ax.set_ylabel("Absolute heat-flux residual [W/m²]")
    ax.set_title("Eq. 3 heat-balance residuals by axial cell")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(Path(__file__).with_name("eq3_axial_residuals.png"), dpi=150)

def plot_outer_temperature_residuals(iterations, residuals_K):
    """Display/save the outer Eq. 7/property-coupling convergence history."""
    fig, ax = plt.subplots()
    ax.semilogy(iterations, np.maximum(residuals_K, 1e-16), marker="o")
    ax.axhline(0.1, color="black", linestyle="--", label="0.1 K tolerance")
    ax.set_xlabel("Outer iteration")
    ax.set_ylabel("Maximum coolant-temperature change [K]")
    ax.set_title("Coolant profile convergence residual")
    ax.grid(True, which="both")
    ax.legend()
    fig.tight_layout()
    fig.savefig(Path(__file__).with_name("thermal_outer_residual.png"), dpi=150)

def plot_pressure_profiles(x, p_coolant, gas_pressure, target_outlet_pressure):
    """Create the combined coolant/gas pressure figure."""
    fig, ax = plt.subplots()
    ax.plot(x, p_coolant / 1e6, label="Coolant static pressure")
    ax.plot(
        x,
        gas_pressure / 1e6,
        label="Gas static pressure",
    )
    ax.axhline(
        target_outlet_pressure / 1e6,
        color="black",
        linestyle="--",
        label="Target coolant outlet pressure",
    )
    ax.set_xlabel("Axial position, x [m]")
    ax.set_ylabel("Static pressure [MPa]")
    ax.set_title("Axial pressure profiles")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(Path(__file__).with_name("pressure_profiles.png"), dpi=150)

def plot_temperature_profiles(
    x, x_cell, gas_temperature, T_aw, T_coolant_nodes, T_wall_hot=None, T_wall_coolant=None,
    coolant_label="Coolant bulk temperature", T_saturation=None,
    T_metal=None,
):
    """Create one labeled figure with every available axial temperature."""
    fig, ax = plt.subplots()
    ax.plot(
        x,
        gas_temperature,
        label="Gas static temperature",
    )
    ax.plot(x, T_aw, label="Adiabatic wall temperature, Eqs. 5–6")
    ax.plot(x, T_coolant_nodes, linestyle="--", label=coolant_label)
    if T_wall_hot is not None:
        ax.plot(x_cell, T_wall_hot, label="Hot-side wall temperature")
    if T_wall_coolant is not None:
        ax.plot(x_cell, T_wall_coolant, label="Coolant-side wall temperature")
    if T_metal is not None:
        ax.plot(
            x_cell,
            T_metal,
            linestyle="-.",
            label="Mean metal temperature for conductivity, (T_wg + T_wc)/2",
        )
    if T_saturation is not None:
        ax.plot(x_cell, T_saturation, linestyle=":", label="Coolant saturation temperature")
    ax.set_xlabel("Axial position, x [m]")
    ax.set_ylabel("Temperature [K]")
    ax.set_title("Axial temperature profiles")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(Path(__file__).with_name("temperature_profiles.png"), dpi=150)

def show_plots():
    """Display all figures created during the current solver run."""
    plt.show()
