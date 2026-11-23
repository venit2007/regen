# Project status and handoff notes

This file gives a working map of the repository for the next AI agent or developer. The main codebase is a regenerative-cooling model for the HORUS liquid rocket engine. The repository also contains an EUCASS straight-channel validation case and a newer N₂O property-table effort.

## Current state

- The main analysis is in `solver.py` and `channel.py`. It reads operating conditions from `config.json`, gas/geometry profiles from `data/HORUS_regen_cooling_axial_inputs.csv`, and writes thermal result columns back into that CSV.
- `config.json` is currently set to `NitrousOxide`, 310 K inlet coolant temperature, 7.5 MPa inlet pressure, 7.0 MPa target outlet pressure, 2.1 kg/s total flow, and 60 channels. These values differ from the Ethanol configuration described in parts of `README.md` and `config.md`.
- The current main cell solver is a single-phase wall-balance model. It uses CoolProp, and it limits the coolant-side wall temperature using a saturation-temperature query. Nitrous oxide has a critical point near 309.5 K and 7.245 MPa ([NIST data](https://www.nist.gov/pml/sensor-science/fluid-metrology/database-thermophysical-properties-gases-used-semiconductor-15)). At the configured state, saturation handling is therefore a likely incompatibility: above critical pressure, a liquid saturation temperature is not defined, and at nearby subcritical pressures the configured bulk temperature is already near or above the saturation boundary. The N₂O tables described below are not currently used by `channel.properties()` or the wall solver.
- A prior standard run reported an infeasible wall balance at cell 247, approximately `x=0.28265 m`, with the hot-side wall trial at the 1375 K upper limit of the Inconel correlation. That report was produced before the current `config.json` switch to N₂O, so it is not a verified failure result for the current configuration.
- The validation pathway is available with `python solver.py --validation`. It uses the supplied EUCASS mesh and case metadata, Eq. 32 circular heated area (`N_ch*pi*Dh*dx`), and square flow area for mass flow. The existing coolant heat-transfer correlation is used because the supplied paper metadata does not provide the CFD-averaged `h_c` values used in the paper's 1-D comparisons. Treat the saved validation outputs as a correlation-based reproduction, not an exact reproduction of the paper's averaged-coefficient run.
- The saved validation output contains three cases. The last recorded run reported outlet bulk temperatures of 118.000 K, 139.490 K, and 179.551 K for 0, 1, and 3 MW/m² respectively. The 3 MW/m² result is 7.46% below the published 194.02 K CFD value; the q0 case remains at its 118 K inlet temperature because the imposed heat flux is zero.
- The standard solver runs substantial setup and calculation code at module import time. Run it as a script; do not import `solver.py` as a passive library module without first separating the top-level execution.

## Running the solver

```bash
python solver.py
```

The main run reads `config.json` and `data/HORUS_regen_cooling_axial_inputs.csv`; successful calculations update existing result columns in the input CSV and save profile/residual plots beside the scripts. A failed cell may still write the valid upstream thermal prefix and leave downstream derived values blank.

Run the paper validation mesh with:

```bash
python solver.py --validation
```

This reads `data/EUCASS2025_validation_mesh_3000_cells.csv` and `data/EUCASS2025_validation_cases.csv`, then writes `data/validation.csv` and the three `validation_*.png` plots beside the scripts. The published benchmark table is in `data/EUCASS2025_published_benchmarks.csv`.

## Code files

### Regenerative-cooling solver

- **`solver.py`** — Main executable and coupled iteration. Validates JSON configuration, loads and checks the axial CSV, builds cell-centred gas/geometry inputs, calculates adiabatic-wall temperature, gas-side Bartz coefficients and radiation, then iterates cell wall balances, coolant energy, and pressure. It also contains the `--validation` entry path, CSV output reporting, and partial-result handling.
- **`channel.py`** — Physics and numerical kernels. Implements radiation Eqs. 9–10, recovery temperature Eqs. 5–6, Bartz Eqs. 12–13, Inconel 718 conductivity, wall balance Eqs. 3–4, channel geometry, CoolProp fluid properties, coolant correlations Eqs. 14–24, Eq. 7 coolant energy march, Darcy-Weisbach pressure loss, local bounded least-squares wall solving, and thermal CSV writing. The energy march permits zero heated area in adiabatic cells. The two-phase Chen helpers exist, but the main engine path is single phase.
- **`plot.py`** — Saves pressure, temperature, Eq. 3 residual, and outer temperature-convergence figures. `show_plots()` calls Matplotlib `show`; in headless environments plots may save successfully without opening a GUI.
- **`regen.py`** — Older standalone pressure-loss prototype. It hard-codes Ethanol and older global inputs and is separate from the current `solver.py` workflow; use `solver.py` for current analysis.
- **`README.md`** — Main solver overview and coupled iteration description. Some of its example/config context predates the current NitrousOxide configuration.
- **`config.md`** — Describes the required JSON keys, units, and original RPA sources. Check it against `config.json` when changing fluids or operating cases.
- **`algorithm.md`** — Language-independent pseudocode for the equations and solver iterations.

### N₂O property data

- **`get_nox_properties.py`** — Fetches isobaric property sweeps from the NIST WebBook CGI for N₂O. Its name says “nox,” but the CAS identifier is nitrous oxide (N₂O). Current grid: 310–400 K in 1 K steps and 7.1–7.5 MPa across 31 isobars. It converts listed units to SI and writes the tables below. It uses `requests` and `pandas` and needs network access to regenerate the files.
- **`n2o_property_tables/density_kg_m3.csv`**, **`cp_j_kgk.csv`**, **`cv_j_kgk.csv`**, **`viscosity_pa_s.csv`**, **`conductivity_w_mk.csv`**, **`enthalpy_j_kg.csv`**, and **`sound_speed_m_s.csv`** — Generated NIST grids with temperature rows and pressure columns. These are data artifacts only; the solver currently queries CoolProp directly and does not interpolate these tables.

### Inputs, references, and outputs

- **`config.json`** — Active operating inputs; currently NitrousOxide as summarized above.
- **`data/HORUS_regen_cooling_axial_inputs.csv`** — Main engine axial mesh, contour, gas profiles, H₂O/CO₂ partial pressures, and existing derived output columns. It is modified in place by the standard run; preserve a copy when input provenance matters.
- **`data/lre_radius_vs_axial_distance.csv`** — Auxiliary engine radius/axial profile data; the current solver reads the contour from the HORUS CSV instead.
- **`data/EUCASS2025_validation_mesh_3000_cells.csv`** — 3000-cell straight-channel validation mesh, including the adiabatic entrance, heated section, square flow area, and circular Eq. 32 heated areas.
- **`data/EUCASS2025_validation_cases.csv`** — Per-case validation conditions and published reference values, with notes on which values are CFD mesh results and which are unavailable for direct comparison.
- **`data/EUCASS2025_published_benchmarks.csv`** — Extracted paper benchmark values and provenance notes. Some outlet-pressure numbers have ambiguous units and are explicitly marked as unsuitable acceptance targets.
- **`data/validation.csv`** — Cell profiles generated by the `--validation` run.
- **`pressure_profiles.png`, `temperature_profiles.png`, `eq3_axial_residuals.png`, `eq3_residual_convergence.png`, `thermal_outer_residual.png`** — Main solver plots (some are failure/convergence diagnostics and may reflect a previous run).
- **`validation_pressure.png`, `validation_temperature.png`, `validation_wall_temperature.png`** — Saved validation plots.
- **`data/RPA-4KN-R2S-Engine.pdf`** and **`data/RPA-4KN-R2S-Engine (1).pdf`** — RPA report copies used as engine/gas input sources; compare before editing or replacing either copy.
- **`data/Assignment-6.pdf`** — Analog/digital electronics assignment, unrelated to the cooling solver.
- **`compliant_parallel_jaw_gripper_concept.svg`** — Standalone gripper concept graphic, unrelated to the cooling solver.

## Important equations and bookkeeping

- Per-channel flow for the engine model is `m_dot / N_ch`; it is used for velocity and cell pressure loss.
- The engine energy march uses total coolant flow with total heated area across all channels, avoiding a second division by channel count.
- The validation case is one equivalent channel. Flow uses the square cross-section area, while heating uses the circular perimeter `pi*Dh` times axial cell length.
- Pressure is marched using the configured inlet pressure. The target outlet pressure is reported as a comparison; the solver does not tune inlet pressure or mass flow to force it.
- The local wall solve checks residuals after optimizer termination. An optimizer message such as `gtol termination condition is satisfied` does not by itself mean the heat balance converged; the residual and temperature bounds must also pass.

## Worktree and next-agent cautions

At the time this file was written, `config.json`, `solver.py`, `channel.py`, and the move into `data/` were worktree changes. The N₂O property tables remain in `n2o_property_tables/` as requested. Python imports/runs also changed the tracked `__pycache__/channel.cpython-314.pyc` artifact. Preserve the existing work unless the owner asks for cleanup; inspect `git status` before reverting or replacing files.

There is no dependency manifest in the repository. The solver imports NumPy, SciPy, Matplotlib, CoolProp, and `fluids`; the N₂O table generator additionally imports pandas and requests.
