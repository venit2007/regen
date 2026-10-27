# Regenerative cooling solver

## Running the solver

Run the executable from this directory with:

```bash
python solver.py
```

The solver reads fixed operating conditions and channel dimensions from
`config.json`, and reads the axial mesh, engine contour, and gas-property
profiles from `HORUS_regen_cooling_axial_inputs.csv`. Paths are resolved from
the script directory, so the command can also be run from another working
directory.

`channel.py` contains reusable physics and local solver kernels. `solver.py`
loads inputs and runs the pressure/thermal iteration. `plot.py` contains the
plotting code.

## Coupled solver workflow

1. The CSV axial nodes define the non-uniform mesh. Cell-centred inputs are
   interpolated from those nodes, and each cell retains its own axial length.
2. Coolant temperature is initially set to the configured inlet temperature
   at every node.
3. For the current coolant temperature profile, the solver marches pressure
   forward from the configured inlet pressure. Local fluid properties are
   recalculated from the current temperature and pressure. The total coolant
   flow is divided by the number of channels once; this per-channel flow is
   used for channel velocity and pressure-loss calculations.
4. At each cell, the inner thermal iteration adjusts the hot-side and
   coolant-side wall-temperature estimates until the heat-flow paths close.
   Local wall conductivity and coolant-side heat-transfer properties are
   recalculated as the trial wall temperatures change.
5. Once the local cell heat fluxes are available, the solver marches the
   coolant temperature downstream. The energy balance uses total mass flow
   with total heated area across all channels, so it does not apply the
   parallel-channel flow split a second time.
6. The updated coolant-temperature profile is compared with the profile used
   at the start of the outer iteration. If it has not converged, the solver
   under-relaxes the update and repeats pressure and thermal calculations.
7. After convergence, pressure and local wall solutions are recalculated
   using the converged coolant profile, followed by a final temperature
   consistency check.

Pressure is therefore coupled to temperature through the local coolant
properties and is recalculated during the thermal iterations. The outlet
pressure target is reported as a comparison; the solver does not adjust the
inlet pressure or mass flow to force that target.

## Residuals and stopping criteria

### Local wall iteration

The local nonlinear solver has two normalized residuals:

- gas-side heat flow minus heat conducted through the wall;
- wall conduction minus heat transferred to the coolant.

Each heat-flux difference is divided by `heat_scale`, the greater of an
estimated gas-side heat flux and `100,000 W/m²`. This normalization helps the
optimizer work with residuals of a useful numerical size; it does not change
the physical heat flux.

`least_squares` varies the two wall temperatures within their physical/model
bounds. Its `xtol`, `ftol`, and `gtol` are each `1e-12`, and its maximum number
of residual evaluations is `max_iterations * 10`. The solver also checks the
residual magnitude independently and verifies that the resulting wall
temperatures agree with the coupled heat-balance solution within `0.1 K`.
Optimizer termination alone is not considered sufficient for a valid cell
solution.

### Axial thermal iteration

The outer residual is the maximum absolute temperature difference, in kelvin,
between the newly marched coolant profile and the profile used for that
iteration. The default acceptance tolerance is `0.1 K`; at most 60 outer
iterations are allowed. Updates are under-relaxed by averaging the old and new
profiles to stabilize the temperature/property/pressure coupling.

### Energy and pressure consistency

The coolant energy march checks that the total heat added from the cell heat
fluxes and heated areas matches the coolant enthalpy rise computed with the
cell heat capacities. The pressure report also compares the inlet-to-outlet
pressure difference with the sum of cell pressure losses. These checks detect
internal accounting inconsistencies; they do not establish that the input
profiles or model assumptions represent the physical engine accurately.

## Failed or incomplete thermal runs

If a cell cannot produce a valid local single-phase wall solution, the solver
raises an error identifying the failed cell and axial position. It preserves
and reports the successfully solved upstream prefix where available; values
downstream of the failed cell are not treated as calculated results. The
solver does not continue by inventing a heat flux or coolant-temperature
profile.

## Outputs

On a successful run, calculated thermal profiles are written to the existing
output columns in `HORUS_regen_cooling_axial_inputs.csv`, preserving the input
mesh and source columns. Pressure, temperature, and convergence diagnostics
are saved as figures in the script directory and displayed at runtime.
