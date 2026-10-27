"""Executable setup, coupled iteration, reporting, CSV output, and plots."""

import csv

import json

import sys

from pathlib import Path

import numpy as np

import plot as plotting

from CoolProp.CoolProp import PropsSI

import channel as physics

from channel import *  # Reuse the physics functions and low-level solver kernels.

def load_config(path):
    """Load and validate solver inputs from a JSON configuration file."""
    with path.open(encoding="utf-8") as config_file:
        config = json.load(config_file)

    if not isinstance(config, dict):
        raise ValueError("Configuration must be a JSON object.")

    required = {
        "fluid", "T_coolant", "p_in", "p_target_out", "m_dot",
        "N_ch", "delta", "w_b", "t_wall", "relative_roughness",
        "gas_p0_Pa", "gas_c_star_m_s", "gas_D_star_m", "gas_r_c_m",
        "gas_mu0_Pa_s", "gas_cp0_J_kgK", "gas_Pr0", "gas_rho0_kg_m3",
        "gas_mu_ref_power_exponent",
    }
    missing = sorted(required - config.keys())
    if missing:
        raise ValueError(f"Missing configuration keys: {', '.join(missing)}")

    if not isinstance(config["fluid"], str) or not config["fluid"].strip():
        raise ValueError("'fluid' must be a non-empty string.")

    numeric_keys = required - {"fluid", "N_ch"}
    for key in numeric_keys:
        value = config[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
            raise ValueError(f"'{key}' must be a finite number.")

    if isinstance(config["N_ch"], bool) or not isinstance(config["N_ch"], int) or config["N_ch"] <= 0:
        raise ValueError("'N_ch' must be a positive integer channel count.")

    for key in (
        "T_coolant", "p_in", "p_target_out", "m_dot", "delta", "w_b",
        "t_wall", "gas_p0_Pa", "gas_c_star_m_s", "gas_D_star_m",
        "gas_r_c_m", "gas_mu0_Pa_s", "gas_cp0_J_kgK", "gas_Pr0",
        "gas_rho0_kg_m3", "gas_mu_ref_power_exponent",
    ):
        if config[key] <= 0:
            raise ValueError(f"'{key}' must be positive.")
    if config["relative_roughness"] < 0:
        raise ValueError("'relative_roughness' must be non-negative.")
    if config["gas_mu_ref_power_exponent"] <= 0:
        raise ValueError("'gas_mu_ref_power_exponent' must be positive.")

    return config

def solve_coupled_thermal_profile(p_in, max_outer_iterations=60, tolerance_K=0.1):
    """Solve cell Eq. 3, march coolant with Eq. 7, and refresh pressure/properties."""
    if q_rad_profile is None:
        raise RuntimeError("Thermal solve requires finite H2O and CO2 radiation inputs.")
    T_nodes = np.full(n_nodes, T_coolant, dtype=float)
    q_rad_cell = np.interp(x_cell, x, q_rad_profile)
    coolant_cp_cell = np.empty(n_cells)
    q_cell = np.empty(n_cells)
    T_wall_hot_cell = np.empty(n_cells)
    T_wall_coolant_cell = np.empty(n_cells)
    k_wall_cell = np.empty(n_cells)
    h_c_cell = np.empty(n_cells)
    iteration_counts = np.empty(n_cells, dtype=int)
    saturation_temperature_cell = np.empty(n_cells)
    outer_temperature_residuals = []

    def raise_partial_failure(i, exc, p_nodes, T_used, velocity_cells,
                              rho_cells, Re_cells, f_cells, dp_cells):
        # Eq. 7 is advanced only across cells whose Eq. 3 wall balance solved.
        T_partial = np.full(n_nodes, np.nan)
        T_partial[0] = T_coolant
        for j in range(i):
            T_partial[j + 1] = T_partial[j] + (
                q_cell[j] * A_heating_cell[j] / (m_dot * coolant_cp_cell[j])
            )
        partial_cells = {}
        for key, data in (("q", q_cell), ("T_wall_hot", T_wall_hot_cell),
                          ("T_wall_coolant", T_wall_coolant_cell),
                          ("k_wall", k_wall_cell), ("h_c", h_c_cell),
                          ("cp", coolant_cp_cell)):
            values = np.full(n_cells, np.nan)
            values[:i] = data[:i]
            partial_cells[key] = values
        T_sat_partial = np.full(n_cells, np.nan)
        T_sat_partial[:i + 1] = saturation_temperature_cell[:i + 1]
        partial = {
            "p": p_nodes, "T": T_partial, "velocity": velocity_cells,
            "rho": rho_cells, "Re": Re_cells, "f": f_cells, "dp": dp_cells,
            "q_rad": q_rad_cell, "T_sat": T_sat_partial,
            "n_solved_cells": i, "failed_cell": i, "failed_x": x_cell[i],
            **partial_cells,
        }
        raise IncompleteThermalSolve(
            f"Thermal march incomplete at cell {i}, x={x_cell[i]:.8g} m: {exc}",
            partial,
        ) from exc

    for outer_iteration in range(1, max_outer_iterations + 1):
        p_nodes, T_used, velocity_cells, rho_cells, Re_cells, f_cells, dp_cells = (
            march_pressure(p_in, T_nodes)
        )
        for i in range(n_cells):
            T_cell_bulk = T_used[i]
            p_cell = p_nodes[i]
            rho, mu, cp, _, k_fluid = properties(T_cell_bulk, p_cell)
            try:
                saturation_temperature_cell[i] = PropsSI(
                    "T", "P", p_cell, "Q", 0, fluid
                )
            except Exception:
                saturation_temperature_cell[i] = np.inf
            coolant_cp_cell[i] = cp
            try:
                (q_cell[i], T_wall_hot_cell[i], T_wall_coolant_cell[i],
                 k_wall_cell[i], h_c_cell[i], iteration_counts[i]) = (
                    solve_cell_thermal_equilibrium_iterative(
                        T_aw_cell_K[i], T_used[i], p_nodes[i], h_g_profile[i],
                        q_rad_cell[i], Re_cells[i], Dh[i], rho, mu, cp, k_fluid,
                        width[i], t_wall, w_b, delta,
                        cell_index=i, x_position_m=x_cell[i],
                    )
                )
            except Exception as exc:
                raise_partial_failure(
                    i, exc, p_nodes, T_used, velocity_cells, rho_cells,
                    Re_cells, f_cells, dp_cells,
                )

        T_updated = march_coolant_energy(
            T_coolant, q_cell, coolant_cp_cell, A_heating_cell, m_dot
        )
        temperature_residual = float(np.max(np.abs(T_updated - T_nodes)))
        outer_temperature_residuals.append(temperature_residual)
        if temperature_residual <= tolerance_K:
            # Update pressure/properties once with the converged coolant profile.
            T_nodes = T_updated
            p_nodes, T_used, velocity_cells, rho_cells, Re_cells, f_cells, dp_cells = (
                march_pressure(p_in, T_nodes)
            )
            for i in range(n_cells):
                T_cell_bulk = T_used[i]
                p_cell = p_nodes[i]
                rho, mu, cp, _, k_fluid = properties(T_cell_bulk, p_cell)
                try:
                    saturation_temperature_cell[i] = PropsSI(
                        "T", "P", p_cell, "Q", 0, fluid
                    )
                except Exception:
                    saturation_temperature_cell[i] = np.inf
                coolant_cp_cell[i] = cp
                try:
                    (q_cell[i], T_wall_hot_cell[i], T_wall_coolant_cell[i],
                     k_wall_cell[i], h_c_cell[i], iteration_counts[i]) = (
                        solve_cell_thermal_equilibrium_iterative(
                            T_aw_cell_K[i], T_used[i], p_nodes[i], h_g_profile[i],
                            q_rad_cell[i], Re_cells[i], Dh[i], rho, mu, cp, k_fluid,
                            width[i], t_wall, w_b, delta,
                            cell_index=i, x_position_m=x_cell[i],
                        )
                    )
                except Exception as exc:
                    raise_partial_failure(
                        i, exc, p_nodes, T_used, velocity_cells, rho_cells,
                        Re_cells, f_cells, dp_cells,
                    )
            T_check = march_coolant_energy(
                T_coolant, q_cell, coolant_cp_cell, A_heating_cell, m_dot
            )
            temperature_residual = float(np.max(np.abs(T_check - T_nodes)))
            outer_temperature_residuals.append(temperature_residual)
            if temperature_residual <= tolerance_K:
                q_gas_cells = h_g_profile * (T_aw_cell_K - T_wall_hot_cell) + q_rad_cell
                q_wall_cells = k_wall_cell / t_wall * (
                    T_wall_hot_cell - T_wall_coolant_cell
                )
                q_coolant_cells = h_c_cell * (
                    T_wall_coolant_cell - T_used[:-1]
                )
                plotting.plot_axial_eq3_residuals(
                    x_cell, q_gas_cells - q_wall_cells,
                    q_wall_cells - q_coolant_cells,
                )
                plotting.plot_outer_temperature_residuals(
                    np.arange(1, len(outer_temperature_residuals) + 1),
                    np.asarray(outer_temperature_residuals),
                )
                return {
                    "p": p_nodes, "T": T_check, "velocity": velocity_cells,
                    "rho": rho_cells, "Re": Re_cells, "f": f_cells, "dp": dp_cells,
                    "q": q_cell, "q_rad": q_rad_cell,
                    "T_wall_hot": T_wall_hot_cell,
                    "T_wall_coolant": T_wall_coolant_cell,
                    "k_wall": k_wall_cell, "h_c": h_c_cell,
                    "cp": coolant_cp_cell, "outer_iterations": outer_iteration,
                    "temperature_residual_K": temperature_residual,
                    "wall_iterations": iteration_counts,
                    "T_sat": saturation_temperature_cell,
                }
        # Under-relax the axial coolant energy update for stable property coupling.
        T_nodes = 0.5 * T_nodes + 0.5 * T_updated

    raise RuntimeError(
        "Coupled coolant temperature/pressure iteration failed to converge "
        f"within {max_outer_iterations} iterations. Last temperature residual "
        f"was {temperature_residual:.6g} K."
    )


def run_paper_validation():
    """Run the paper's straight methane channel cases on the supplied mesh."""
    root = Path(__file__).resolve().parent
    mesh_path = root / "EUCASS2025_validation_mesh_3000_cells.csv"
    cases_path = root / "EUCASS2025_validation_cases.csv"
    output_path = root / "validation.csv"
    with mesh_path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        mesh_fields = reader.fieldnames
        rows = list(reader)
    if not rows or mesh_fields is None:
        raise ValueError(f"Validation mesh is empty or has no header: {mesh_path}")
    with cases_path.open(newline="", encoding="utf-8") as source:
        validation_cases = {
            row["case_id"]: row for row in csv.DictReader(source)
        }

    global fluid, T_coolant, m_dot_channel, relative_roughness
    global n_nodes, n_cells, area, Dh, dx
    fluid = next(iter(validation_cases.values()))["coolant_fluid"]
    T_coolant = float(next(iter(validation_cases.values()))["inlet_bulk_temperature_K"])
    roughness = 0.0  # The reference straight channel is treated as smooth.
    x0 = np.array([float(row["x_start_m"]) for row in rows])
    x1 = np.array([float(row["x_end_m"]) for row in rows])
    dx = np.array([float(row["dx_m"]) for row in rows])
    x_cell = np.array([float(row["x_center_m"]) for row in rows])
    area = np.array([float(row["flow_area_square_m2"]) for row in rows])
    Dh = np.array([float(row["hydraulic_diameter_m"]) for row in rows])
    heated_flag = np.array([int(row["heated_section_flag"]) for row in rows], dtype=bool)
    heating_area = np.array([
        float(row["Eq32_heated_area_per_cell_m2"]) for row in rows
    ])
    expected_heating_area = np.where(heated_flag, np.pi * Dh * dx, 0.0)
    if not np.allclose(heating_area, expected_heating_area, rtol=1e-9, atol=1e-15):
        raise ValueError("Validation Eq. 32 area must be pi * Dh * dx in heated cells.")
    n_cells = len(rows)
    n_nodes = n_cells + 1
    relative_roughness = roughness
    mass_flux = float(next(iter(validation_cases.values()))["mass_flux_kg_m2_s"])
    channel_mass_flow = mass_flux * area
    m_dot_channel = float(channel_mass_flow[0])
    if not np.allclose(channel_mass_flow, m_dot_channel):
        raise ValueError("Validation mesh flow area must be constant along the channel.")
    physics.configure_solver_context(
        fluid=fluid, T_coolant=T_coolant, m_dot_channel=m_dot_channel,
        relative_roughness=relative_roughness, n_nodes=n_nodes,
        n_cells=n_cells, area=area, Dh=Dh, dx=dx,
    )

    case_inputs = {
        "q0MW": ("q_case_0MW_W_m2", float(validation_cases["q0MW"]["CFD_case_inlet_total_pressure_Pa"])),
        "q1MW": ("q_case_1MW_W_m2", float(validation_cases["q1MW"]["CFD_case_inlet_total_pressure_Pa"])),
        "q3MW": ("q_case_3MW_W_m2", float(validation_cases["q3MW"]["CFD_case_inlet_total_pressure_Pa"])),
    }
    profiles = {}
    for case, (q_column, inlet_pressure) in case_inputs.items():
        q_cell = np.array([float(row[q_column]) for row in rows])
        T_nodes = np.full(n_nodes, T_coolant)
        for outer_iteration in range(1, 21):
            p_nodes, T_used, velocity, rho, Re, friction, dp = physics.march_pressure(
                inlet_pressure, T_nodes
            )
            cp_cell = np.empty(n_cells)
            h_cell = np.full(n_cells, np.nan)
            wall_cell = np.full(n_cells, np.nan)
            for i in range(n_cells):
                _, mu, cp_cell[i], _, k_fluid = physics.properties(T_used[i], p_nodes[i])
                if q_cell[i] == 0.0:
                    continue
                wall_temperature = T_used[i] + q_cell[i] * Dh[i] / k_fluid
                for _ in range(100):
                    h = physics.coolant_forced_convection_coefficient(
                        Re[i], T_used[i], wall_temperature, p_nodes[i], Dh[i],
                        rho[i], mu, cp_cell[i], k_fluid,
                    )
                    next_wall_temperature = T_used[i] + q_cell[i] / h
                    if abs(next_wall_temperature - wall_temperature) < 1e-8:
                        wall_temperature = next_wall_temperature
                        break
                    wall_temperature = 0.5 * (wall_temperature + next_wall_temperature)
                else:
                    raise RuntimeError(f"Validation wall iteration failed in cell {i}.")
                h_cell[i] = h
                wall_cell[i] = wall_temperature

            T_updated = physics.march_coolant_energy(
                T_coolant, q_cell, cp_cell, heating_area, m_dot_channel
            )
            residual = float(np.max(np.abs(T_updated - T_nodes)))
            if residual <= 0.1:
                T_nodes = T_updated
                break
            # Imposed heat flux makes the energy march explicit; a direct
            # profile update converges faster for this validation case.
            T_nodes = T_updated
        else:
            raise RuntimeError(f"{case} validation did not converge; residual={residual:g} K")

        p_nodes, T_used, velocity, rho, Re, friction, dp = physics.march_pressure(
            inlet_pressure, T_nodes
        )
        # Recompute reported wall quantities on the converged state.
        cp_cell = np.array([physics.properties(T_used[i], p_nodes[i])[2]
                            for i in range(n_cells)])
        for i in range(n_cells):
            if q_cell[i] == 0.0:
                h_cell[i] = np.nan
                wall_cell[i] = np.nan
                continue
            _, mu, cp, _, k_fluid = physics.properties(T_used[i], p_nodes[i])
            wall_temperature = T_used[i] + q_cell[i] * Dh[i] / k_fluid
            for _ in range(100):
                h = physics.coolant_forced_convection_coefficient(
                    Re[i], T_used[i], wall_temperature, p_nodes[i], Dh[i],
                    rho[i], mu, cp, k_fluid,
                )
                next_wall_temperature = T_used[i] + q_cell[i] / h
                if abs(next_wall_temperature - wall_temperature) < 1e-8:
                    wall_temperature = next_wall_temperature
                    break
                wall_temperature = 0.5 * (wall_temperature + next_wall_temperature)
            h_cell[i], wall_cell[i] = h, wall_temperature
        profiles[case] = {
            "q": q_cell, "T_nodes": T_nodes, "p_nodes": p_nodes,
            "T_bulk": T_used[:-1], "p_cell": p_nodes[:-1],
            "velocity": velocity, "rho": rho, "Re": Re,
            "friction": friction, "dp": dp, "cp": cp_cell,
            "h": h_cell, "T_wall": wall_cell,
        }
        print(
            f"{case}: inlet pressure={inlet_pressure / 1e6:.4f} MPa, "
            f"outlet temperature={T_nodes[-1]:.3f} K, "
            f"outlet pressure={p_nodes[-1] / 1e6:.5f} MPa, "
            f"temperature residual={residual:.4g} K"
        )

    # Retain all input mesh/source columns and append one result set per case.
    output_fields = list(mesh_fields)
    result_columns = ("coolant_bulk_T_K", "coolant_wall_T_K", "pressure_Pa",
                      "h_c_W_m2K", "Re", "velocity_m_s", "dp_Pa")
    for case in case_inputs:
        output_fields.extend(f"{case}_{name}" for name in result_columns)
    with output_path.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=output_fields)
        writer.writeheader()
        for i, original in enumerate(rows):
            row = dict(original)
            for case, result in profiles.items():
                values = {
                    "coolant_bulk_T_K": result["T_bulk"][i],
                    "coolant_wall_T_K": result["T_wall"][i],
                    "pressure_Pa": result["p_cell"][i],
                    "h_c_W_m2K": result["h"][i],
                    "Re": result["Re"][i],
                    "velocity_m_s": result["velocity"][i],
                    "dp_Pa": result["dp"][i],
                }
                for name, value in values.items():
                    row[f"{case}_{name}"] = f"{value:.10g}" if np.isfinite(value) else ""
            writer.writerow(row)

    x_nodes = np.append(x0, x1[-1])
    for quantity, ylabel, title, filename in (
        ("T_nodes", "Coolant bulk temperature [K]", "Validation coolant temperature", "validation_temperature.png"),
        ("p_nodes", "Coolant pressure [MPa]", "Validation coolant pressure", "validation_pressure.png"),
    ):
        fig, ax = plotting.plt.subplots(figsize=(9, 5))
        for case, result in profiles.items():
            values = result[quantity] / 1e6 if quantity == "p_nodes" else result[quantity]
            ax.plot(x_nodes, values, label=case)
        ax.axvspan(0.4, 1.2, color="grey", alpha=0.12, label="Heated section")
        ax.set(xlabel="Axial position [m]", ylabel=ylabel, title=title)
        ax.grid(True)
        ax.legend()
        fig.tight_layout()
        fig.savefig(root / filename, dpi=160)
    fig, ax = plotting.plt.subplots(figsize=(9, 5))
    for case, result in profiles.items():
        mask = np.isfinite(result["T_wall"])
        ax.plot(x_cell[mask], result["T_wall"][mask], label=case)
    ax.axvspan(0.4, 1.2, color="grey", alpha=0.12, label="Heated section")
    ax.set(xlabel="Axial position [m]", ylabel="Coolant-side wall temperature [K]",
           title="Validation coolant-side wall temperature")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(root / "validation_wall_temperature.png", dpi=160)
    print(f"Circular heated area per case: {heating_area.sum():.9f} m^2")
    q0_reference = float(validation_cases["q0MW"]["published_CFD_T_bulk_out_K"])
    q3_reference = float(validation_cases["q3MW"]["published_CFD_T_bulk_out_K"])
    print(
        f"Published CFD references (not direct 1D acceptance values): "
        f"q0MW solver={profiles['q0MW']['T_nodes'][-1]:.3f} K, "
        f"CFD mesh table={q0_reference:.3f} K; "
        f"q3MW solver={profiles['q3MW']['T_nodes'][-1]:.3f} K, "
        f"CFD mesh table={q3_reference:.2f} K "
        f"({100 * (profiles['q3MW']['T_nodes'][-1] / q3_reference - 1):+.2f}%)."
    )
    for x_target, reference in (
        (0.7, float(validation_cases["q3MW"]["published_Twc_at_x_0p7m_K"])),
        (1.0, float(validation_cases["q3MW"]["published_Twc_at_x_1p0m_K"])),
    ):
        cell = int(np.argmin(np.abs(x_cell - x_target)))
        calculated = profiles["q3MW"]["T_wall"][cell]
        print(
            f"Published q3MW coolant-wall temperature at x={x_target:.1f} m: "
            f"solver={calculated:.2f} K, table={reference:.2f} K."
        )
    print(f"Validation profiles written to {output_path}")
    print(
        "Heat-transfer limitation: the paper uses CFD-averaged h_c values for "
        "its validation tool; those numerical values are blank in the case "
        "file. This run uses the existing coolant correlation from channel.py."
    )
    plotting.show_plots()


if "--validation" in sys.argv:
    run_paper_validation()
    raise SystemExit(0)

script_dir = Path(__file__).resolve().parent

config = load_config(script_dir / "config.json")

fluid = config["fluid"]

T_coolant = config["T_coolant"]

p_in = config["p_in"]

p_target_out = config["p_target_out"]

m_dot = config["m_dot"]

N_ch = config["N_ch"]

m_dot_channel = m_dot / N_ch

delta = config["delta"]

w_b = config["w_b"]

t_wall = config["t_wall"]

relative_roughness = config["relative_roughness"]

gas_p0_Pa = config["gas_p0_Pa"]

gas_c_star_m_s = config["gas_c_star_m_s"]

gas_D_star_m = config["gas_D_star_m"]

gas_r_c_m = config["gas_r_c_m"]

gas_mu0_Pa_s = config["gas_mu0_Pa_s"]

gas_cp0_J_kgK = config["gas_cp0_J_kgK"]

gas_Pr0 = config["gas_Pr0"]

gas_rho0_kg_m3 = config["gas_rho0_kg_m3"]

gas_mu_ref_power_exponent = config["gas_mu_ref_power_exponent"]

axial_input_path = script_dir / "HORUS_regen_cooling_axial_inputs.csv"

required_axial_columns = (
    "x_m",
    "radius_m_approx",
    "area_ratio_A_At",
    "gas_static_temperature_K",
    "gas_static_pressure_Pa",
    "gas_mach_approx",
    "gas_density_kg_m3",
    "gas_cp_J_kgK",
    "gas_gamma",
    "gas_viscosity_Pa_s",
    "gas_conductivity_W_mK",
    "gas_prandtl",
)

if not axial_input_path.is_file():
    raise FileNotFoundError(
        f"Required axial inputs file not found: {axial_input_path}"
    )

try:
    axial_data = np.genfromtxt(
        axial_input_path,
        delimiter=",",
        names=True,
        dtype=float,
        encoding="utf-8",
        autostrip=True,
        filling_values=np.nan,
    )
except Exception as exc:
    raise ValueError(f"Could not read axial inputs CSV {axial_input_path}: {exc}") from exc

if axial_data.dtype.names is None:
    raise ValueError(f"Axial inputs CSV has no readable header: {axial_input_path}")

missing_axial_columns = [
    column for column in required_axial_columns if column not in axial_data.dtype.names
]

if missing_axial_columns:
    raise ValueError(
        "Axial inputs CSV is missing required unit-bearing columns: "
        + ", ".join(missing_axial_columns)
    )

axial_profiles = {
    column: np.atleast_1d(np.asarray(axial_data[column], dtype=float))
    for column in required_axial_columns
}

profile_lengths = {values.size for values in axial_profiles.values()}

if len(profile_lengths) != 1:
    raise ValueError("Axial geometry and gas profile columns have inconsistent lengths.")

if next(iter(profile_lengths), 0) < 2:
    raise ValueError("Axial inputs CSV must contain at least two rows.")

for column, values in axial_profiles.items():
    if not np.all(np.isfinite(values)):
        raise ValueError(
            f"Column '{column}' contains NaN or infinite values required by "
            "the axial mesh, engine contour, or imported gas-profile inputs."
        )

x = axial_profiles["x_m"]

r_engine = axial_profiles["radius_m_approx"]

if not np.all(np.diff(x) > 0.0):
    raise ValueError("Axial input column 'x_m' must be strictly increasing (m).")

if np.any(r_engine <= 0.0):
    raise ValueError("Axial input column 'radius_m_approx' must be positive (m).")

if np.any(axial_profiles["area_ratio_A_At"] <= 0.0):
    raise ValueError("Axial input column 'area_ratio_A_At' must be positive.")

for column in (
    "gas_static_temperature_K",
    "gas_static_pressure_Pa",
    "gas_density_kg_m3",
    "gas_cp_J_kgK",
    "gas_viscosity_Pa_s",
    "gas_conductivity_W_mK",
    "gas_prandtl",
):
    if np.any(axial_profiles[column] <= 0.0):
        raise ValueError(f"Axial input column '{column}' must be positive in its named units.")

if np.any(axial_profiles["gas_mach_approx"] < 0.0):
    raise ValueError("Axial input column 'gas_mach_approx' must be non-negative.")

if np.any(axial_profiles["gas_gamma"] <= 1.0):
    raise ValueError("Axial input column 'gas_gamma' must be greater than 1.")

length = x[-1] - x[0]

n_nodes = x.size

n_cells = n_nodes - 1

dx = np.diff(x)

x_cell = 0.5 * (x[:-1] + x[1:])

radius_cell = np.interp(x_cell, x, r_engine)

gas_profiles_cell = {
    column: np.interp(x_cell, x, values)
    for column, values in axial_profiles.items()
    if column not in ("x_m", "radius_m_approx")
}

T_aw_profile_K = adiabatic_wall_temperature(
    axial_profiles["gas_static_temperature_K"],
    axial_profiles["gas_prandtl"],
    axial_profiles["gas_gamma"],
    axial_profiles["gas_mach_approx"],
)

T_g = axial_profiles["gas_static_temperature_K"]

T_ref = T_aw_profile_K

rho_g = axial_profiles["gas_density_kg_m3"]

mu_g = axial_profiles["gas_viscosity_Pa_s"]

if (
    not np.isfinite(T_g).all()
    or not np.isfinite(T_ref).all()
    or not np.isfinite(rho_g).all()
    or not np.isfinite(mu_g).all()
    or np.any(T_g <= 0.0)
    or np.any(T_ref <= 0.0)
    or np.any(rho_g <= 0.0)
    or np.any(mu_g <= 0.0)
):
    raise ValueError("Invalid gas properties or reference temperature.")

rho_ref = rho_g * T_g / T_ref

mu_ref = mu_g * (T_ref / T_g) ** gas_mu_ref_power_exponent

if not np.isfinite(rho_ref).all() or np.any(rho_ref <= 0.0):
    raise ValueError("Provisional rho_ref estimate is invalid.")

if not np.isfinite(mu_ref).all() or np.any(mu_ref <= 0.0):
    raise ValueError("Provisional mu_ref estimate is invalid.")

print(
    "PROVISIONAL MODEL ASSUMPTION: rho_ref and mu_ref at T_ref=T_aw are "
    "estimated from RPA static gas properties, assuming unchanged local "
    "pressure and composition, ideal-gas density scaling, and a gas-viscosity "
    f"power law with exponent {gas_mu_ref_power_exponent:g}. These are "
    "approximations, not paper-provided or CEA-calculated reference properties."
)

T_aw_cell_K = np.interp(x_cell, x, T_aw_profile_K)

rho_ref_cell = np.interp(x_cell, x, rho_ref)

mu_ref_cell = np.interp(x_cell, x, mu_ref)

h_g_profile, phi_g_profile = bartz_gas_heat_transfer_coefficient(
    gas_p0_Pa,
    gas_c_star_m_s,
    gas_D_star_m,
    gas_r_c_m,
    gas_mu0_Pa_s,
    gas_cp0_J_kgK,
    gas_Pr0,
    gas_profiles_cell["area_ratio_A_At"],
    rho_ref_cell,
    gas_profiles_cell["gas_density_kg_m3"],
    mu_ref_cell,
)

for radiation_column in ("p_H2O_Pa", "p_CO2_Pa"):
    if radiation_column not in axial_data.dtype.names:
        raise ValueError(
            f"Missing '{radiation_column}' required for paper radiation equations."
        )

radiation_pressures = {
    column: np.atleast_1d(np.asarray(axial_data[column], dtype=float))
    for column in ("p_H2O_Pa", "p_CO2_Pa")
}

if all(np.all(np.isfinite(values)) for values in radiation_pressures.values()):
    q_rad_h2o_profile = q_rad_h2o(
        T_aw_profile_K, radiation_pressures["p_H2O_Pa"], r_engine
    )
    q_rad_co2_profile = q_rad_co2(
        T_aw_profile_K, radiation_pressures["p_CO2_Pa"], r_engine
    )
    q_rad_profile = q_rad_h2o_profile + q_rad_co2_profile
else:
    q_rad_h2o_profile = None
    q_rad_co2_profile = None
    q_rad_profile = None

assert np.all(np.diff(x) > 0.0), "Solver axial coordinates must increase."

assert np.all(dx > 0.0), "Every solver cell must have positive length."

engine_area_node = np.pi * r_engine**2

width_node, area_node, Dh_node = channel_geometry(
    r_engine, N_ch, w_b, delta
)

engine_area_cell = np.pi * radius_cell**2

width, area, Dh = channel_geometry(
    radius_cell, N_ch, w_b, delta
)

A_heating_cell = N_ch * width * dx

if not np.all(np.isfinite(A_heating_cell)) or np.any(A_heating_cell <= 0.0):
    raise ValueError("Eq. 8 produced invalid heated area values.")

physics.configure_solver_context(
    fluid=fluid, T_coolant=T_coolant, m_dot_channel=m_dot_channel,
    relative_roughness=relative_roughness, n_nodes=n_nodes, n_cells=n_cells,
    area=area, Dh=Dh, dx=dx,
)

physics.plot_eq3_residual_convergence = plotting.plot_eq3_residual_convergence

print(
    "WALL-CONDUCTIVITY EVALUATION ASSUMPTION: the supplied Inconel k(T) "
    "correlation is evaluated at the mean through-wall temperature "
    "(T_wall_hot + T_wall_coolant)/2."
)

try:
    thermal = solve_coupled_thermal_profile(p_in)
except IncompleteThermalSolve as exc:
    partial = exc.partial_results
    n_done = partial["n_solved_cells"]

    def partial_cells_to_nodes(values):
        values = np.asarray(values, dtype=float)
        nodes = np.full(n_nodes, np.nan)
        if n_done:
            nodes[:n_done + 1] = np.interp(
                x[:n_done + 1], x_cell[:n_done], values[:n_done],
                left=values[0], right=values[n_done - 1],
            )
        return nodes

    h_g_nodes = np.interp(x, x_cell, h_g_profile, left=h_g_profile[0], right=h_g_profile[-1])
    q_gas_partial = h_g_profile * (T_aw_cell_K - partial["T_wall_hot"]) + partial["q_rad"]
    write_thermal_outputs_to_csv(axial_input_path, {
        "T_aw_K": T_aw_profile_K,
        "h_g_W_m2K": h_g_nodes,
        "q_gas_W_m2": partial_cells_to_nodes(q_gas_partial),
        "wall_k_W_mK": partial_cells_to_nodes(partial["k_wall"]),
        "T_wall_hot_K": partial_cells_to_nodes(partial["T_wall_hot"]),
        "T_wall_coolant_K": partial_cells_to_nodes(partial["T_wall_coolant"]),
        "q_to_coolant_W_m2": partial_cells_to_nodes(partial["q"]),
    })
    plotting.plot_pressure_profiles(x, partial["p"], axial_profiles["gas_static_pressure_Pa"], p_target_out)
    plotting.plot_temperature_profiles(
        x, x_cell, axial_profiles["gas_static_temperature_K"], T_aw_profile_K, partial["T"], partial["T_wall_hot"], partial["T_wall_coolant"],
        coolant_label="Coolant bulk temperature (partial Eq. 7 march)",
        T_saturation=partial["T_sat"],
        T_metal=0.5 * (partial["T_wall_hot"] + partial["T_wall_coolant"]),
    )
    plotting.show_plots()
    print(
        f"Displayed and saved incomplete thermal march: {n_done} of {n_cells} "
        f"cells completed; failure at cell {partial['failed_cell']}, "
        f"x={partial['failed_x']:.8g} m. Derived CSV values downstream of "
        "the completed prefix are blank."
    )
    raise
except Exception:
    # Keep the requested gas/coolant operating-profile figures available even
    # when no valid coupled wall solution exists. Wall traces require a solution.
    p_fallback, T_fallback, *_ = march_pressure(p_in)
    try:
        T_sat_fallback_nodes = np.asarray([
            PropsSI("T", "P", pressure, "Q", 0, fluid) for pressure in p_fallback
        ])
        T_sat_fallback = np.interp(x_cell, x, T_sat_fallback_nodes)
    except Exception:
        T_sat_fallback = None
    plotting.plot_pressure_profiles(x, p_fallback, axial_profiles["gas_static_pressure_Pa"], p_target_out)
    plotting.plot_temperature_profiles(
        x, x_cell, axial_profiles["gas_static_temperature_K"], T_aw_profile_K, T_fallback,
        coolant_label="Coolant inlet temperature (thermal solve failed)",
        T_saturation=T_sat_fallback,
    )
    plotting.show_plots()
    raise

p = thermal["p"]

T = thermal["T"]

velocity = thermal["velocity"]

rho = thermal["rho"]

Re = thermal["Re"]

f = thermal["f"]

dp = thermal["dp"]

q_cell = thermal["q"]

q_rad_cell = thermal["q_rad"]

T_wall_hot_cell = thermal["T_wall_hot"]

T_wall_coolant_cell = thermal["T_wall_coolant"]

k_wall_cell = thermal["k_wall"]

h_c_cell = thermal["h_c"]

T_sat_cell = thermal["T_sat"]

Taw_node_output = T_aw_profile_K

cell_to_node = lambda values: np.interp(
    x, x_cell, np.asarray(values, dtype=float), left=values[0], right=values[-1]
)

q_gas_cell = h_g_profile * (T_aw_cell_K - T_wall_hot_cell) + q_rad_cell

thermal_node_outputs = {
    "T_aw_K": Taw_node_output,
    "h_g_W_m2K": cell_to_node(h_g_profile),
    "q_gas_W_m2": cell_to_node(q_gas_cell),
    "wall_k_W_mK": cell_to_node(k_wall_cell),
    "T_wall_hot_K": cell_to_node(T_wall_hot_cell),
    "T_wall_coolant_K": cell_to_node(T_wall_coolant_cell),
    "q_to_coolant_W_m2": cell_to_node(q_cell),
}

write_thermal_outputs_to_csv(axial_input_path, thermal_node_outputs)

assert np.all(np.isfinite(p)), "Non-finite pressure"

assert np.all(p > 0), "Non-positive absolute pressure"

assert np.all(np.isfinite(velocity)), "Non-finite velocity"

assert np.all(np.isfinite(Re)), "Non-finite Reynolds number"

assert np.all(np.isfinite(f)), "Non-finite friction factor"

assert np.all(dp >= 0), "Negative friction pressure loss"

assert np.all(np.diff(p) <= 0), "Pressure increases downstream"

drop_from_profile = p[0] - p[-1]

drop_from_cells = np.sum(dp)

print(f"Profile pressure drop: {drop_from_profile:.6f} Pa")

print(f"Sum of cell losses:    {drop_from_cells:.6f} Pa")

print(f"Balance error:         {drop_from_profile - drop_from_cells:.3e} Pa")

print(f"Reynolds range: {Re.min():.1f} - {Re.max():.1f}")

print(f"Friction factor range: {f.min():.5f} - {f.max():.5f}")

print(f"Maximum cell pressure drop: {dp.max():.3f} Pa")

print(f"Minimum cell pressure drop: {dp.min():.3f} Pa")

print(f"Inlet pressure:  {p[0] / 1e6:.5f} MPa")

print(f"Outlet pressure: {p[-1] / 1e6:.5f} MPa")

print(f"Target outlet:   {p_target_out / 1e6:.5f} MPa")

print(f"Pressure drop:   {(p[0] - p[-1]) / 1e3:.5f} kPa")

print(f"Outlet residual: {(p[-1] - p_target_out) / 1e3:.5f} kPa")

print(f"Maximum velocity: {np.max(velocity):.5f} m/s")

print(f"Reynolds range: {np.min(Re):.1f} to {np.max(Re):.1f}")

print(
    f"Coupled Eq. 3 / Eq. 7 iteration converged in "
    f"{thermal['outer_iterations']} outer iterations; coolant-temperature "
    f"residual {thermal['temperature_residual_K']:.4f} K."
)

print(
    f"Bartz Eq. 12 gas-side coefficient h_g range: "
    f"{h_g_profile.min():.3f} - {h_g_profile.max():.3f} W/(m^2 K)"
)

print(
    f"Eq. 13 property factor phi range: "
    f"{phi_g_profile.min():.5f} - {phi_g_profile.max():.5f}"
)

print(
    f"Single-phase coolant heat flux range: {q_cell.min():.3f} - "
    f"{q_cell.max():.3f} W/m^2"
)

print(
    f"Hot-wall temperature range: {T_wall_hot_cell.min():.3f} - "
    f"{T_wall_hot_cell.max():.3f} K"
)

print(
    f"Coolant-side wall temperature range: {T_wall_coolant_cell.min():.3f} - "
    f"{T_wall_coolant_cell.max():.3f} K"
)

print(
    f"Corrected coolant h_c range: {h_c_cell.min():.3f} - "
    f"{h_c_cell.max():.3f} W/(m^2 K)"
)

if np.any(T_wall_coolant_cell >= T_sat_cell):
    print(
        "SINGLE-PHASE LIMITATION: coolant-side wall temperature reaches or "
        "exceeds ethanol saturation temperature in at least one cell; boiling "
        "is not modeled, so those local h_c/q predictions are outside the "
        "single-phase physical regime."
    )
else:
    print("Single-phase check: coolant-side wall remains below saturation.")

print(f"Thermal results written to: {axial_input_path}")

if q_rad_profile is None:
    print(
        "Radiation profiles not calculated: CSV columns p_H2O_Pa and/or "
        "p_CO2_Pa contain missing values."
    )
else:
    print(
        f"Radiation Eq. 9 H2O range: {q_rad_h2o_profile.min():.3f} - "
        f"{q_rad_h2o_profile.max():.3f} W/m^2"
    )
    print(
        f"Radiation Eq. 10 CO2 range: {q_rad_co2_profile.min():.3f} - "
        f"{q_rad_co2_profile.max():.3f} W/m^2"
    )
    print(
        f"Combined radiative heat-flux range: {q_rad_profile.min():.3f} - "
        f"{q_rad_profile.max():.3f} W/m^2"
    )

plotting.plot_pressure_profiles(x, p, axial_profiles["gas_static_pressure_Pa"], p_target_out)

plotting.plot_temperature_profiles(
    x, x_cell, axial_profiles["gas_static_temperature_K"], T_aw_profile_K, T,
    T_wall_hot_cell,
    T_wall_coolant_cell,
    T_saturation=T_sat_cell,
    T_metal=0.5 * (T_wall_hot_cell + T_wall_coolant_cell),
)

plotting.show_plots()

