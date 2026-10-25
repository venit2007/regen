# Solver configuration

`channel.py` reads `config.json` from the same directory. The file must contain
all of the following keys:

| Key | Meaning | Unit / valid values |
| --- | --- | --- |
| `fluid` | CoolProp fluid name | Non-empty string |
| `T_coolant` | Coolant inlet temperature | K, positive |
| `p_in` | Trial coolant inlet static pressure | Pa, positive |
| `p_target_out` | Target outlet pressure used in the report | Pa, positive |
| `m_dot` | Total mass flow shared by all parallel channels | kg/s, positive |
| `N_ch` | Number of parallel cooling channels | Positive integer |
| `delta` | Channel height | m, positive |
| `w_b` | Rib thickness | m, positive |
| `t_wall` | Hot-wall thickness | m, positive |
| `relative_roughness` | Relative roughness, `ε / D` | Dimensionless, non-negative |
| `gas_p0_Pa` | Chamber stagnation pressure from the RPA report | Pa, positive |
| `gas_c_star_m_s` | Ideal characteristic velocity from RPA Table 4 | m/s, positive |
| `gas_D_star_m` | Throat diameter from RPA Table 8 | m, positive |
| `gas_r_c_m` | Throat curvature radius `Rn` from RPA Table 8 | m, positive |
| `gas_mu0_Pa_s` | Stagnation gas viscosity from RPA Table 2 | Pa·s, positive |
| `gas_cp0_J_kgK` | Stagnation gas specific heat at constant pressure from RPA Table 2 | J/(kg·K), positive |
| `gas_Pr0` | Stagnation gas Prandtl number from RPA Table 2 | Dimensionless, positive |
| `gas_rho0_kg_m3` | Stagnation gas density from RPA Table 2 | kg/m³, positive |
| `gas_mu_ref_power_exponent` | Exponent used by the provisional gas-viscosity power-law estimate for `mu_ref` | Dimensionless, positive; approximation, not specified by the paper |

The solver computes per-channel flow once as
`m_dot_channel = m_dot / N_ch` and uses that value in the channel flow
calculations. Axial coordinates and engine radii come from
`HORUS_regen_cooling_axial_inputs.csv` in metres. The gas stagnation and throat
scalars above come from `RPA-4KN-R2S-Engine.pdf`; `gas_c_star_m_s` uses the
ideal-performance value (the report also lists a delivered value).
