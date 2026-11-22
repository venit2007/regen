"""Plotting functions for solver profiles and convergence diagnostics."""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


PROFILE_WINDOW = "Cooling solver profiles"


def _profile_panel(index, filename):
    fig = plt.figure(PROFILE_WINDOW, figsize=(13, 9), layout="constrained")
    if not fig.axes:
        for ax in fig.subplots(3, 2).flat:
            ax.set_visible(False)
    ax = fig.axes[index]
    ax.clear()
    ax.set_visible(True)
    ax.set_gid(filename)
    return ax


def save_plot_grid(fig, path):
    """Save the full figure and export the labeled panels from the same render."""
    fig.savefig(path, dpi=fig.dpi)
    with Image.open(path) as image:
        for ax in fig.axes:
            filename = ax.get_gid()
            if not ax.get_visible() or not filename:
                continue
            x0, y0, x1, y1 = ax.get_tightbbox().padded(4).extents
            bounds = (
                max(0, int(np.floor(x0))),
                max(0, image.height - int(np.ceil(y1))),
                min(image.width, int(np.ceil(x1))),
                min(image.height, image.height - int(np.floor(y0))),
            )
            image.crop(bounds).save(path.with_name(filename))


def plot_eq3_residual_convergence(residual_history, cell_index=None, x_position_m=None):
    """Plot local Eq. 3 residuals on the solver figure."""
    values = np.asarray(residual_history, dtype=float)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] != 2:
        return
    evaluations = np.arange(1, values.shape[0] + 1)
    location = f" at cell {cell_index}" if cell_index is not None else ""
    if x_position_m is not None:
        location += f", x={x_position_m:.6g} m"
    ax = _profile_panel(4, "eq3_residual_convergence.png")
    for column, label in enumerate(("Gas − wall", "Wall − coolant")):
        ax.semilogy(evaluations, np.maximum(np.abs(values[:, column]), 1e-16), label=label)
    ax.set(
        xlabel="Residual function evaluation",
        ylabel="Absolute normalized residual",
        title=f"Local Eq. 3 residuals{location}",
    )
    ax.grid(True, which="both")
    ax.legend(fontsize="small")


def plot_axial_eq3_residuals(x_cell, gas_wall_residual, wall_coolant_residual):
    """Display/save absolute Eq. 3 heat-path residuals along the mesh."""
    ax = _profile_panel(4, "eq3_axial_residuals.png")
    ax.plot(x_cell, np.abs(gas_wall_residual), label="|gas-side − wall conduction|")
    ax.plot(x_cell, np.abs(wall_coolant_residual), label="|wall conduction − coolant-side|")
    ax.set_xlabel("Axial position, x [m]")
    ax.set_ylabel("Absolute heat-flux residual [W/m²]")
    ax.set_title("Eq. 3 heat-balance residuals by axial cell")
    ax.grid(True)
    ax.legend(fontsize="small")

def plot_outer_temperature_residuals(iterations, residuals_K):
    """Display/save the outer Eq. 7/property-coupling convergence history."""
    ax = _profile_panel(5, "thermal_outer_residual.png")
    ax.semilogy(iterations, np.maximum(residuals_K, 1e-16), marker="o")
    ax.axhline(0.1, color="black", linestyle="--", label="0.1 K tolerance")
    ax.set_xlabel("Outer iteration")
    ax.set_ylabel("Maximum coolant-temperature change [K]")
    ax.set_title("Coolant profile convergence residual")
    ax.grid(True, which="both")
    ax.legend(fontsize="small")

def plot_pressure_profiles(x, p_coolant, gas_pressure, target_outlet_pressure, *, progress=None):
    """Create the combined coolant/gas pressure figure."""
    ax = _profile_panel(0, "pressure_profiles.png")
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
    ax.figure.suptitle(progress or PROFILE_WINDOW)
    ax.grid(True)
    ax.legend(fontsize="small")

def plot_temperature_profiles(
    x, x_cell, gas_temperature, T_aw, T_coolant_nodes, T_wall_hot=None, T_wall_coolant=None,
    coolant_label="Coolant bulk temperature", T_saturation=None,
    T_metal=None,
):
    """Create one labeled figure with every available axial temperature."""
    ax = _profile_panel(1, "temperature_profiles.png")
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
            label="Mean metal temperature",
        )
    if T_saturation is not None:
        ax.plot(x_cell, T_saturation, linestyle=":", label="Coolant saturation temperature")
    ax.set_xlabel("Axial position, x [m]")
    ax.set_ylabel("Temperature [K]")
    ax.set_title("Axial temperature profiles")
    ax.grid(True)
    ax.legend(fontsize="small")

def plot_channel_width(x, width_m):
    """Show the width of one coolant passage at each axial station."""
    ax = _profile_panel(2, "channel_width.png")
    ax.plot(x, width_m * 1e3)
    ax.set_xlabel("Axial position, x [m]")
    ax.set_ylabel("Channel width [mm]")
    ax.set_title("Coolant channel width")
    ax.margins(x=0, y=0.06)
    ax.grid(True)


def plot_coolant_velocity(x_cell, velocity):
    """Plot the mean coolant speed through a channel at each cell."""
    ax = _profile_panel(3, "coolant_velocity.png")
    ax.plot(x_cell, velocity)
    ax.set(xlabel="Axial position, x [m]", ylabel="Speed [m/s]", title="Coolant velocity")
    ax.grid(True)


def refresh_live_plots():
    """Save the current profiles and repaint the solver window on GUI backends."""
    if plt.fignum_exists(PROFILE_WINDOW):
        fig = plt.figure(PROFILE_WINDOW)
        save_plot_grid(fig, Path(__file__).with_name("solver_profiles.png"))
        if fig.canvas.required_interactive_framework is not None:
            fig.show()
            fig.canvas.draw_idle()
            fig.canvas.flush_events()


def show_plots():
    """Display all figures created during the current solver run."""
    refresh_live_plots()
    plt.show()
