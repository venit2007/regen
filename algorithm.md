# Solver algorithms

This document gives language-independent pseudocode for the main calculations in `solver.py` and `channel.py`. The active thermal solver uses the configured coolant as a single phase. Two-phase Chen correlations are present as helper functions but are not called by the active solver.

## Load and validate configuration

```text
FUNCTION load_config(file):
    parse JSON from file
    require the top-level value to be an object
    require all operating, geometry, gas, and coolant input keys
    require fluid to be a non-empty string
    require numeric inputs to be finite numbers (not booleans)
    require channel count to be a positive integer
    require temperatures, pressures, flow, dimensions, and gas reference inputs to be positive
    require relative roughness to be non-negative
    RETURN configuration
```

## Build axial mesh and cell inputs

```text
READ required axial columns from CSV
REQUIRE at least two rows, finite values, and strictly increasing axial positions
REQUIRE physically valid positive radii, area ratios, gas properties, and gamma > 1

FOR each adjacent pair of axial nodes:
    cell length = downstream x - upstream x
    cell center = average of the two x positions
    interpolate radius and gas inputs at the cell center

AT each axial node:
    recovery factor = Prandtl^(1/3)                         # Eq. 5
    adiabatic wall temperature = gas static temperature
        * (1 + recovery factor * (gamma - 1) * Mach^2 / 2) # Eq. 6

Estimate gas reference density and viscosity at adiabatic wall temperature
using the configured ideal-gas density scaling and viscosity power law.
Interpolate these estimates to cell centers.

FOR each cell:
    calculate gas coefficient h_g and property factor phi using Bartz Eqs. 12-13
    calculate H2O and CO2 radiative heat fluxes using Eqs. 9-10
    sum radiation contributions
    calculate channel width, flow area, and hydraulic diameter from radius,
        channel count, rib thickness, and channel height
    heated area = channel count * channel width * cell length             # Eq. 8
```

Radiation must have finite H2O and CO2 partial pressure inputs. Missing or invalid input is not replaced with an assumed value.

## March coolant pressure downstream

```text
FUNCTION march_pressure(inlet_pressure, coolant_temperature_at_nodes):
    set pressure at first node to inlet pressure

    FOR cells from inlet to outlet:
        use upstream node temperature and pressure to get coolant properties
        per-channel velocity = (total mass flow / channel count)
                              / (density * channel flow area)
        Reynolds number = Reynolds(velocity, hydraulic diameter, density, viscosity)
        friction factor = friction_factor(Reynolds number, relative roughness)
        pressure loss = friction factor * density * velocity^2 / 2
                        * (cell length / hydraulic diameter)
        downstream pressure = upstream pressure - pressure loss
        fail if downstream pressure is non-finite or non-positive

    RETURN node pressures and cell velocity, density, Reynolds number,
        friction factor, and pressure loss
```

## Solve a cell's wall heat balance

The unknowns are hot-side and coolant-side wall temperatures. For each trial, wall conductivity is evaluated at the mean through-wall temperature. The coolant coefficient depends on the coolant-side wall temperature, so the wall equations are solved iteratively as a bounded nonlinear least-squares problem.

```text
FUNCTION solve_cell(T_aw, T_bulk, pressure, h_g, radiation, cell geometry/properties):
    determine coolant saturation temperature at local pressure
    set upper wall-temperature bound to the lower of:
        maximum supported Inconel temperature, saturation temperature - 0.1 K
    require bulk coolant temperature to be below this bound

    define residuals(hot_wall, coolant_wall):
        mean_wall = (hot_wall + coolant_wall) / 2
        wall_k = Inconel_conductivity(mean_wall)

        Pr = cp * bulk_viscosity / fluid_conductivity
        Nu = 0.027 * Re^0.8 * Pr^0.4 * (bulk_viscosity / wall_viscosity)^0.14  # Eq. 15
        forced_h = Nu * fluid_conductivity / hydraulic_diameter                 # Eq. 16
        single_phase_h = forced_h                                               # Eqs. 14, 17; h_nb=0, F=1
        eta_fin = tanh(m * channel_height) / (m * channel_height)                # Eq. 23
        corrected_h = single_phase_h * (width + 2 * eta_fin * channel_height)
                        / (width + rib_thickness)                               # Eq. 24

        gas_heat = h_g * (T_aw - hot_wall) + radiation
        wall_heat = wall_k / wall_thickness * (hot_wall - coolant_wall)
        coolant_heat = corrected_h * (coolant_wall - T_bulk)

        RETURN [(gas_heat - wall_heat) / heat_scale,
                (wall_heat - coolant_heat) / heat_scale]

    run bounded least-squares on the two wall temperatures
    require optimizer success and normalized residuals <= 1e-8
    recompute wall conductivity and corrected coolant coefficient at solution
    calculate heat flux and wall temperatures from the closed-form Eq. 4
    require the closed-form temperatures to agree with the iterative solution
        within 0.1 K and the three heat-flow paths to balance
    RETURN heat flux, both wall temperatures, wall conductivity,
        corrected coolant coefficient, and evaluation count
```

`heat_scale` is the larger of the estimated gas heat flux and 100,000 W/m²; it scales the optimizer residuals only. The coolant-wall upper bound prevents the active single-phase solution from reaching saturation.

## March coolant energy downstream

```text
FUNCTION march_coolant_energy(inlet_temperature, cell_heat_fluxes,
                              cell_heat_capacities, total_heated_areas,
                              total_mass_flow):
    set first node temperature to inlet temperature
    FOR cells from inlet to outlet:
        temperature rise = heat_flux[cell] * heated_area[cell]
                           / (total_mass_flow * cp[cell])                # Eq. 7
        next node temperature = current node temperature + temperature rise
    verify total coolant enthalpy rise equals sum(heat_flux * heated_area)
    RETURN node temperatures
```

Total mass flow is paired with heated area summed across all channels. The per-channel flow split is used in pressure and velocity calculations, not applied a second time in this energy balance.

## Coupled pressure and thermal iteration

```text
FUNCTION solve_coupled_thermal_profile(inlet_pressure,
                                       maximum_outer_iterations = 60,
                                       temperature_tolerance = 0.1 K):
    require valid radiation inputs
    initialize coolant temperature at every node to inlet temperature

    FOR outer iteration = 1 through maximum_outer_iterations:
        pressure and coolant properties = march_pressure(inlet_pressure, temperatures)

        FOR each cell from inlet to outlet:
            evaluate coolant properties at local bulk temperature and pressure
            solve cell wall balance with solve_cell(...)
            if a cell fails:
                raise an incomplete-solve error carrying the valid upstream results

        updated_temperatures = march_coolant_energy(...)
        residual = maximum absolute difference between updated and used temperatures

        IF residual <= temperature_tolerance:
            replace temperatures with updated profile
            recalculate pressure, properties, and all cell wall solutions
            check coolant energy march again
            IF final maximum temperature difference <= tolerance:
                RETURN converged pressure, temperature, heat flux, wall,
                    property, and iteration results

        temperatures = 0.5 * temperatures + 0.5 * updated_temperatures

    fail with a non-convergence error and last residual
```

The 50/50 update under-relaxation stabilizes the coupling between temperature-dependent fluid properties, pressure drop, and wall heat transfer. The outlet pressure target is reported after solving; this algorithm does not tune inlet pressure or mass flow to force that target.

## Write calculated profiles to CSV

```text
FUNCTION write_thermal_outputs(file, node_profiles):
    read existing CSV rows and headers
    require every requested output column to already exist
    require row count and each profile length to match the axial node count
    replace only requested output-column values (non-finite values become blank)
    write complete CSV to a temporary sibling file
    atomically replace the original CSV with the temporary file
```

For an incomplete thermal solve, only the successfully solved upstream cells are reported as calculated; downstream thermal outputs are left blank.

## Supporting correlation helpers

- **Radiation input validation:** broadcast temperatures, partial pressures, and radius; check finite values and physical signs; convert partial pressure from Pa to bar; evaluate Eq. 9 (H2O) or Eq. 10 (CO2).
- **Channel geometry:** pitch = `2*pi*radius/channel_count`; width = pitch minus rib thickness; area = width times channel height; hydraulic diameter = `2*width*height/(width+height)`. Reject non-positive width.
- **Closed-form Eq. 3/Eq. 4:** total thermal resistance is `1/h_g + wall_thickness/wall_k + 1/h_c`. Heat flux is `(T_aw - T_bulk + radiation/h_g) / resistance`; recover both wall temperatures and verify gas, conduction, and coolant heat fluxes agree.
- **Two-phase helpers (not used by this run path):** calculate Martinelli parameter (Eq. 19), enhancement factor F (Eq. 18), nucleate-boiling terms (Eqs. 20-21), suppression factor S (Eq. 22), combine them with Eq. 14, then apply fin corrections (Eqs. 23-24). These functions require supplied two-phase quality and properties; the active solver does not infer them or invoke boiling.
