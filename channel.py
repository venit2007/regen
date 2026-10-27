"""Physics functions and local solver kernels for regenerative cooling."""

import csv
import numpy as np
from CoolProp.CoolProp import PropsSI
from fluids.core import Reynolds
from fluids.friction import friction_factor
from scipy.optimize import least_squares


def configure_solver_context(**values):
    """Provide operating state and mesh arrays to legacy solver kernels."""
    globals().update(values)


class IncompleteThermalSolve(RuntimeError):
    """A local thermal failure carrying the converged upstream cell prefix."""

    def __init__(self, message, partial_results):
        super().__init__(message)
        self.partial_results = partial_results

def _radiation_inputs(T_aw_K, partial_pressure_Pa, radius_m):
    """Broadcast and validate inputs for the paper's radiation equations."""
    try:
        T_aw_K, partial_pressure_Pa, radius_m = np.broadcast_arrays(
            np.asarray(T_aw_K, dtype=float),
            np.asarray(partial_pressure_Pa, dtype=float),
            np.asarray(radius_m, dtype=float),
        )
    except ValueError as exc:
        raise ValueError("Radiation inputs must be scalars or broadcast-compatible arrays.") from exc
    if not all(
        np.all(np.isfinite(values))
        for values in (T_aw_K, partial_pressure_Pa, radius_m)
    ):
        raise ValueError("Radiation inputs must be finite; missing CSV values cannot be used.")
    if np.any(T_aw_K <= 0.0) or np.any(partial_pressure_Pa < 0.0) or np.any(radius_m <= 0.0):
        raise ValueError("Radiation requires T_aw > 0 K, partial pressure >= 0 Pa, and radius > 0 m.")
    return T_aw_K, partial_pressure_Pa, radius_m

def q_rad_h2o(T_aw_K, p_h2o_Pa, radius_m):
    """H2O radiative heat flux using the paper's Eq. 9 (output W/m²).

    The empirical correlation uses partial pressure in bar and engine radius
    in metres; this function accepts Pa and converts Pa to bar.
    """
    T_aw_K, p_h2o_Pa, radius_m = _radiation_inputs(T_aw_K, p_h2o_Pa, radius_m)
    p_h2o_bar = p_h2o_Pa / 1.0e5
    # Paper Eq. 9.
    result = 5.74 * (p_h2o_bar * radius_m)**0.3 * (T_aw_K / 100.0)**3.5
    if not np.all(np.isfinite(result)) or np.any(result < 0.0):
        raise ValueError("Eq. 9 produced invalid H2O radiative heat flux.")
    return float(result) if result.ndim == 0 else result

def q_rad_co2(T_aw_K, p_co2_Pa, radius_m):
    """CO2 radiative heat flux using the paper's Eq. 10 (output W/m²).

    The empirical correlation uses partial pressure in bar and engine radius
    in metres; this function accepts Pa and converts Pa to bar.
    """
    T_aw_K, p_co2_Pa, radius_m = _radiation_inputs(T_aw_K, p_co2_Pa, radius_m)
    p_co2_bar = p_co2_Pa / 1.0e5
    # Paper Eq. 10.
    result = 4.0 * (p_co2_bar * radius_m)**0.3 * (T_aw_K / 100.0)**3.5
    if not np.all(np.isfinite(result)) or np.any(result < 0.0):
        raise ValueError("Eq. 10 produced invalid CO2 radiative heat flux.")
    return float(result) if result.ndim == 0 else result

def adiabatic_wall_temperature(T_gas_K, prandtl, gamma, mach):
    """Calculate T_aw using the paper's recovery-factor Eqs. 5 and 6."""
    T_gas_K, prandtl, gamma, mach = np.broadcast_arrays(
        np.asarray(T_gas_K, dtype=float),
        np.asarray(prandtl, dtype=float),
        np.asarray(gamma, dtype=float),
        np.asarray(mach, dtype=float),
    )
    if not all(np.all(np.isfinite(v)) for v in (T_gas_K, prandtl, gamma, mach)):
        raise ValueError("Eq. 5-6 inputs must be finite.")
    if np.any(T_gas_K <= 0.0) or np.any(prandtl <= 0.0) or np.any(gamma <= 1.0) or np.any(mach < 0.0):
        raise ValueError("Eq. 5-6 require T_g > 0, Pr > 0, gamma > 1, and Mach >= 0.")
    # Paper Eq. 5: recovery factor r = Pr^(1/3).
    recovery_factor = prandtl**(1.0 / 3.0)
    # Paper Eq. 6: adiabatic wall temperature from the local gas state.
    T_aw = T_gas_K * (1.0 + recovery_factor * (gamma - 1.0) * mach**2 / 2.0)
    if not np.all(np.isfinite(T_aw)) or np.any(T_aw <= 0.0):
        raise ValueError("Eqs. 5-6 produced invalid adiabatic wall temperatures.")
    return T_aw

def bartz_gas_heat_transfer_coefficient(
    p0_g, c_star, D_star, r_c, mu0, cp0, Pr0,
    A_over_At, rho_ref, rho, mu_ref,
):
    """Calculate the paper's Bartz gas-side coefficient and Eq. 13 factor.

    Returns ``(h_g, phi)`` in W/(m² K) and dimensionless units. ``A_over_At``
    is the local area ratio A/A*, and all inputs may be broadcast-compatible
    arrays. ``rho_ref`` and ``mu_ref`` are evaluated at T_ref=T_aw.
    """
    try:
        arrays = np.broadcast_arrays(*[
            np.asarray(value, dtype=float)
            for value in (
                p0_g, c_star, D_star, r_c, mu0, cp0, Pr0, A_over_At,
                rho_ref, rho, mu_ref,
            )
        ])
    except ValueError as exc:
        raise ValueError("Paper Eqs. 12-13 inputs must be broadcast-compatible.") from exc
    names = (
        "p0_g", "c_star", "D_star", "r_c", "mu0", "cp0", "Pr0",
        "A_over_At", "rho_ref", "rho", "mu_ref",
    )
    for name, values in zip(names, arrays):
        if not np.all(np.isfinite(values)):
            raise ValueError(f"Paper Eqs. 12-13 input '{name}' must be finite.")
        if np.any(values <= 0.0):
            raise ValueError(f"Paper Eqs. 12-13 input '{name}' must be positive.")

    (p0_g, c_star, D_star, r_c, mu0, cp0, Pr0, A_over_At,
     rho_ref, rho, mu_ref) = arrays
    # Paper Eq. 13: correction for reference-to-local gas property changes.
    phi = (rho_ref / rho)**0.8 * (mu_ref / mu0)**0.2
    # Paper Eq. 12: Bartz correlation, with local A/A* from the geometry CSV.
    h_g = (
        0.026 / D_star**0.2
        * (mu0**0.2 * cp0 / Pr0**0.6)
        * (p0_g / c_star)**0.8
        * (D_star / r_c)**0.1
        * (1.0 / A_over_At)**0.9
        * phi
    )
    if not np.all(np.isfinite(phi)) or np.any(phi <= 0.0):
        raise ValueError("Paper Eq. 13 produced an invalid property factor.")
    if not np.all(np.isfinite(h_g)) or np.any(h_g <= 0.0):
        raise ValueError("Paper Eq. 12 produced an invalid gas-side coefficient.")
    return h_g, phi

def inconel718_conductivity(T):
    """Thermal conductivity of Inconel 718 [W/(m K)], T in kelvin."""
    if not np.isfinite(T):
        raise ValueError("Inconel 718 conductivity requires a finite temperature.")
    if 298.0 <= T <= 800.0:
        return 5.291 + 0.0152 * T + 1.382e-6 * T**2
    elif 800.0 < T < 1173.0:
        # Piecewise-linear interpolation of published recommended data.
        T_data = np.array([800.0, 1100.0, 1200.0])
        k_data = np.array([18.33548, 22.72, 23.61])
        return float(np.interp(T, T_data, k_data))
    elif 1173.0 <= T <= 1375.0:
        return 11.75 + 0.011 * T - 9.327e-7 * T**2
    else:
        raise ValueError(
            f"Inconel 718 conductivity correlation not validated at T={T} K"
        )

def solve_thermal_equilibrium_eq3(T_aw, T_coolant, h_g, q_rad, k_wall, t_wall, h_c):
    """Solve the local coupled wall balance in paper Eq. 3.

    Return ``(q, T_wall_hot, T_wall_coolant)`` in SI units. The equivalent
    closed form (paper Eq. 4) retains radiation as a separate gas-side input.
    This function does not estimate its inputs: callers must provide the
    coefficients and material/radiation values from valid source data.
    """
    values = np.asarray(
        [T_aw, T_coolant, h_g, q_rad, k_wall, t_wall, h_c], dtype=float
    )
    if not np.all(np.isfinite(values)):
        raise ValueError("Paper Eq. 3 inputs must be finite.")
    if T_aw <= 0.0 or T_coolant <= 0.0:
        raise ValueError("Paper Eq. 3 requires positive absolute temperatures.")
    if h_g <= 0.0 or k_wall <= 0.0 or t_wall <= 0.0 or h_c <= 0.0:
        raise ValueError("Paper Eq. 3 requires positive h_g, k_wall, t_wall and h_c.")
    if q_rad < 0.0:
        raise ValueError("Paper Eq. 3 requires non-negative q_rad into the wall.")

    # Paper Eq. 4: eliminate T_wg and T_wc from the coupled balance in Eq. 3.
    resistance = 1.0 / h_g + t_wall / k_wall + 1.0 / h_c
    q = ((T_aw - T_coolant) + q_rad / h_g) / resistance
    # Recover the coupled wall-temperature unknowns using paper Eq. 3.
    T_wall_hot = T_aw - (q - q_rad) / h_g
    T_wall_coolant = T_coolant + q / h_c

    # Check gas-to-wall, wall-conduction, and wall-to-coolant fluxes (Eq. 3).
    q_gas = h_g * (T_aw - T_wall_hot) + q_rad
    q_wall = (k_wall / t_wall) * (T_wall_hot - T_wall_coolant)
    q_coolant = h_c * (T_wall_coolant - T_coolant)
    results = np.asarray([q, T_wall_hot, T_wall_coolant, q_gas, q_wall, q_coolant])
    if not np.all(np.isfinite(results)):
        raise RuntimeError("Paper Eq. 3 produced non-finite heat-transfer results.")
    if q < 0.0:
        raise ValueError("Paper Eq. 3 predicts negative heat flow into the coolant.")
    if not np.allclose([q_gas, q_wall, q_coolant], q, rtol=1e-10, atol=1e-7):
        raise RuntimeError("Paper Eq. 3 heat-flux paths do not balance.")
    return q, T_wall_hot, T_wall_coolant

def channel_geometry(radius, N_ch, w_b, delta):
    """Return local channel width, area and hydraulic diameter."""

    pitch = 2.0 * np.pi * radius / N_ch
    width = pitch - w_b

    if np.any(width <= 0.0):
        raise ValueError(
            "Non-positive channel width: check radius, "
            "channel count and rib thickness."
        )

    area = width * delta
    Dh = 2.0 * width * delta / (width + delta)

    return width, area, Dh

def properties(T, p):
    """CoolProp properties at one local state."""
    try:
        rho = PropsSI("D", "T", T, "P", p, fluid)
        mu = PropsSI("V", "T", T, "P", p, fluid)
        cp = PropsSI("C", "T", T, "P", p, fluid)
        h = PropsSI("H", "T", T, "P", p, fluid)
        k = PropsSI("L", "T", T, "P", p, fluid)
    except Exception as exc:
        raise RuntimeError(
            f"CoolProp failed for {fluid} at T={T} K, p={p} Pa."
        ) from exc

    values = np.asarray([rho, mu, cp, h, k], dtype=float)
    if not np.all(np.isfinite(values)):
        raise RuntimeError(f"CoolProp returned non-finite properties at T={T} K, p={p} Pa.")
    if rho <= 0.0 or mu <= 0.0 or cp <= 0.0 or k <= 0.0:
        raise RuntimeError(f"CoolProp returned non-positive transport properties at T={T} K, p={p} Pa.")

    return rho, mu, cp, h, k

def coolant_forced_convection_coefficient(
    Re, T_bulk, T_wall, p, Dh, rho, mu_bulk, cp, k
):
    """Return the paper's single-phase forced-convection coefficient (Eq. 17).

    The paper describes this Sieder-Tate correlation for turbulent flow but
    does not provide numerical Reynolds-number bounds; no Re cutoff is
    applied here. Reynolds values depend on the configured mass flow, and the
    correlation remains available when that flow is a dummy/test value. Its
    wall-viscosity input requires a wall temperature from the coupled model.
    """
    inputs = np.asarray(
        [Re, T_bulk, T_wall, p, Dh, rho, mu_bulk, cp, k], dtype=float
    )
    if not np.all(np.isfinite(inputs)):
        raise ValueError("Eq. 17 inputs must all be finite.")
    if Re <= 0.0 or T_bulk <= 0.0 or T_wall <= 0.0 or p <= 0.0:
        raise ValueError("Eq. 17 requires positive Re, temperatures, and pressure.")
    if Dh <= 0.0 or rho <= 0.0 or mu_bulk <= 0.0 or cp <= 0.0 or k <= 0.0:
        raise ValueError("Eq. 17 requires positive geometry and bulk properties.")

    try:
        mu_wall = PropsSI("V", "T", T_wall, "P", p, fluid)
    except Exception as exc:
        raise RuntimeError(
            f"CoolProp could not evaluate liquid wall viscosity at T={T_wall} K, p={p} Pa."
        ) from exc
    if not np.isfinite(mu_wall) or mu_wall <= 0.0:
        raise RuntimeError("CoolProp returned invalid wall viscosity for Eq. 17.")

    Pr = cp * mu_bulk / k
    if not np.isfinite(Pr) or Pr <= 0.0:
        raise ValueError("Eq. 17 requires a finite positive Prandtl number.")

    # Paper Eq. 15: Sieder-Tate Nusselt number, including the viscosity ratio.
    Nu = sieder_tate_nusselt_eq15(Re, Pr, mu_bulk, mu_wall)
    # Paper Eq. 16: convert Nusselt number to the forced-convection coefficient.
    h_fc = heat_transfer_coefficient_from_nusselt_eq16(Nu, k, Dh)
    if not np.isfinite(h_fc) or h_fc <= 0.0:
        raise ValueError("Eq. 17 produced an invalid heat-transfer coefficient.")
    return h_fc

def sieder_tate_nusselt_eq15(Re, Pr, mu_bulk, mu_wall):
    """Paper Eq. 15: Sieder-Tate Nusselt number for coolant flow."""
    values = np.asarray([Re, Pr, mu_bulk, mu_wall], dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError("Eq. 15 inputs must be finite.")
    if Re <= 0.0 or Pr <= 0.0 or mu_bulk <= 0.0 or mu_wall <= 0.0:
        raise ValueError("Eq. 15 requires positive Re, Pr, and viscosities.")
    Nu = 0.027 * Re**0.8 * Pr**0.4 * (mu_bulk / mu_wall)**0.14
    if not np.isfinite(Nu) or Nu <= 0.0:
        raise ValueError("Eq. 15 produced an invalid Nusselt number.")
    return Nu

def heat_transfer_coefficient_from_nusselt_eq16(Nu, k_fluid, Dh):
    """Paper Eq. 16: convert Nusselt number to h using hydraulic diameter."""
    values = np.asarray([Nu, k_fluid, Dh], dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError("Eq. 16 inputs must be finite.")
    if Nu <= 0.0 or k_fluid <= 0.0 or Dh <= 0.0:
        raise ValueError("Eq. 16 requires positive Nu, conductivity, and Dh.")
    h = Nu * k_fluid / Dh
    if not np.isfinite(h) or h <= 0.0:
        raise ValueError("Eq. 16 produced an invalid heat-transfer coefficient.")
    return h

def chen_heat_transfer_coefficient_eq14(h_nb, h_fc, F, S):
    """Paper Eq. 14: combine nucleate-boiling and forced-convection terms."""
    try:
        h_nb, h_fc, F, S = np.broadcast_arrays(*[
            np.asarray(v, dtype=float) for v in (h_nb, h_fc, F, S)
        ])
    except ValueError as exc:
        raise ValueError("Eq. 14 inputs must be broadcast-compatible.") from exc
    if not all(np.all(np.isfinite(v)) for v in (h_nb, h_fc, F, S)):
        raise ValueError("Eq. 14 inputs must be finite.")
    if np.any(h_nb < 0.0) or np.any(h_fc <= 0.0) or np.any(F < 0.0) or np.any(S < 0.0):
        raise ValueError("Eq. 14 requires h_nb,F,S >= 0 and h_fc > 0.")
    h_c = S * h_nb + F * h_fc
    if not np.all(np.isfinite(h_c)) or np.any(h_c <= 0.0):
        raise ValueError("Eq. 14 produced an invalid coolant coefficient.")
    return float(h_c) if h_c.ndim == 0 else h_c

def single_phase_coolant_coefficient_eq14_to17(
    Re, T_bulk, T_wall, p, Dh, rho, mu_bulk, cp, k
):
    """Paper Eq. 14 single-phase reduction using Eqs. 15-17.

    With the configured coolant treated as single phase, nucleate boiling is
    absent (h_nb=0); the forced-convection term is used without a two-phase
    enhancement multiplier (F=1). This helper does not evaluate wall
    temperature; it requires T_wall from the coupled Eq. 3 solve.
    """
    h_fc = coolant_forced_convection_coefficient(
        Re, T_bulk, T_wall, p, Dh, rho, mu_bulk, cp, k
    )
    # Eq. 14 with h_nb=0 and F=1 for the stated single-phase coolant model.
    return chen_heat_transfer_coefficient_eq14(h_nb=0.0, h_fc=h_fc, F=1.0, S=0.0)

def chen_reynolds_factor_eq18(X_tt):
    """Paper Eq. 18: Chen Reynolds-number enhancement factor F."""
    X_tt = np.asarray(X_tt, dtype=float)
    if not np.all(np.isfinite(X_tt)) or np.any(X_tt <= 0.0):
        raise ValueError("Eq. 18 requires finite positive Martinelli parameter X_tt.")
    inv_X = 1.0 / X_tt
    F = np.where(inv_X <= 0.1, 1.0, 2.35 * (inv_X + 0.213)**0.736)
    if not np.all(np.isfinite(F)) or np.any(F <= 0.0):
        raise ValueError("Eq. 18 produced an invalid enhancement factor F.")
    return float(F) if F.ndim == 0 else F

def martinelli_parameter_eq19(vapor_quality, rho_l, rho_v, mu_l, mu_v):
    """Paper Eq. 19: Martinelli parameter X_tt for two-phase flow.

    ``vapor_quality`` is the vapor mass fraction x; the liquid mass fraction
    z is 1-x. The correlation requires an actual two-phase quality, 0<x<1.
    """
    try:
        x, rho_l, rho_v, mu_l, mu_v = np.broadcast_arrays(*[
            np.asarray(v, dtype=float)
            for v in (vapor_quality, rho_l, rho_v, mu_l, mu_v)
        ])
    except ValueError as exc:
        raise ValueError("Eq. 19 inputs must be broadcast-compatible.") from exc
    arrays = (x, rho_l, rho_v, mu_l, mu_v)
    if not all(np.all(np.isfinite(v)) for v in arrays):
        raise ValueError("Eq. 19 inputs must be finite.")
    if np.any((x <= 0.0) | (x >= 1.0)):
        raise ValueError("Eq. 19 requires two-phase vapor quality 0 < x < 1.")
    if any(np.any(v <= 0.0) for v in (rho_l, rho_v, mu_l, mu_v)):
        raise ValueError("Eq. 19 phase densities and viscosities must be positive.")
    z = 1.0 - x
    # Paper Eq. 19.
    X_tt = (z / x)**0.9 * (rho_v / rho_l)**0.5 * (mu_l / mu_v)**0.1
    if not np.all(np.isfinite(X_tt)) or np.any(X_tt <= 0.0):
        raise ValueError("Eq. 19 produced an invalid Martinelli parameter.")
    return float(X_tt) if X_tt.ndim == 0 else X_tt

def forster_zuber_nusselt_eq20(Re, Pr_l):
    """Paper Eq. 20: Forster-Zuber nucleate-boiling Nusselt correlation."""
    try:
        Re, Pr_l = np.broadcast_arrays(
            np.asarray(Re, dtype=float), np.asarray(Pr_l, dtype=float)
        )
    except ValueError as exc:
        raise ValueError("Eq. 20 inputs must be broadcast-compatible.") from exc
    if not np.all(np.isfinite(Re)) or not np.all(np.isfinite(Pr_l)):
        raise ValueError("Eq. 20 inputs must be finite.")
    if np.any(Re <= 0.0) or np.any(Pr_l <= 0.0):
        raise ValueError("Eq. 20 requires positive Re and liquid Prandtl number.")
    # Paper Eq. 20.
    Nu_nb = 0.0015 * Re**0.62 * Pr_l**0.88
    if not np.all(np.isfinite(Nu_nb)) or np.any(Nu_nb <= 0.0):
        raise ValueError("Eq. 20 produced an invalid nucleate-boiling Nusselt number.")
    return float(Nu_nb) if Nu_nb.ndim == 0 else Nu_nb

def forster_zuber_h_nb_eq21(
    k_l, cp_l, rho_l, sigma, mu_l, H_s_lv, rho_v, delta_T, delta_p
):
    """Paper Eq. 21: Forster-Zuber nucleate-boiling coefficient.

    SI inputs: W/(m K), J/(kg K), kg/m³, N/m, Pa s, J/kg, kg/m³, K, Pa.
    """
    try:
        arrays = np.broadcast_arrays(*[
            np.asarray(v, dtype=float)
            for v in (k_l, cp_l, rho_l, sigma, mu_l, H_s_lv, rho_v, delta_T, delta_p)
        ])
    except ValueError as exc:
        raise ValueError("Eq. 21 inputs must be broadcast-compatible.") from exc
    if not all(np.all(np.isfinite(v)) for v in arrays):
        raise ValueError("Eq. 21 inputs must be finite.")
    if any(np.any(v <= 0.0) for v in arrays):
        raise ValueError("Eq. 21 requires positive properties, superheat and pressure difference.")
    k_l, cp_l, rho_l, sigma, mu_l, H_s_lv, rho_v, delta_T, delta_p = arrays
    # Paper Eq. 21, with the paper's empirical coefficient and exponents.
    h_nb = (
        0.00122
        * k_l**0.79 * cp_l**0.45 * rho_l**0.49
        / (sigma**0.5 * mu_l**0.29 * H_s_lv**0.24 * rho_v**0.24)
        * delta_T**0.24 * delta_p**0.75
    )
    if not np.all(np.isfinite(h_nb)) or np.any(h_nb <= 0.0):
        raise ValueError("Eq. 21 produced an invalid nucleate-boiling coefficient.")
    return float(h_nb) if h_nb.ndim == 0 else h_nb

def chen_suppression_factor_eq22(Re_l, F):
    """Paper Eq. 22: Chen nucleate-boiling suppression factor S."""
    try:
        Re_l, F = np.broadcast_arrays(
            np.asarray(Re_l, dtype=float), np.asarray(F, dtype=float)
        )
    except ValueError as exc:
        raise ValueError("Eq. 22 inputs must be broadcast-compatible.") from exc
    if not np.all(np.isfinite(Re_l)) or not np.all(np.isfinite(F)):
        raise ValueError("Eq. 22 inputs must be finite.")
    if np.any(Re_l <= 0.0) or np.any(F <= 0.0):
        raise ValueError("Eq. 22 requires positive liquid Reynolds number and F.")
    # Paper Eq. 22.
    S = 1.0 / (1.0 + 2.53e-6 * (Re_l * F**1.25)**1.17)
    if not np.all(np.isfinite(S)) or np.any((S <= 0.0) | (S > 1.0)):
        raise ValueError("Eq. 22 produced an invalid suppression factor S.")
    return float(S) if S.ndim == 0 else S

def fin_efficiency_eq23(h_c, w_b, k_wall, delta):
    """Paper Eq. 23: fin efficiency of the channel rib/wall."""
    try:
        h_c, w_b, k_wall, delta = np.broadcast_arrays(*[
            np.asarray(v, dtype=float) for v in (h_c, w_b, k_wall, delta)
        ])
    except ValueError as exc:
        raise ValueError("Eq. 23 inputs must be broadcast-compatible.") from exc
    arrays = (h_c, w_b, k_wall, delta)
    if not all(np.all(np.isfinite(v)) for v in arrays):
        raise ValueError("Eq. 23 inputs must be finite.")
    if any(np.any(v <= 0.0) for v in arrays):
        raise ValueError("Eq. 23 requires positive h_c, rib thickness, wall k and channel height.")
    m_delta = np.sqrt(2.0 * h_c * w_b / k_wall) * delta / w_b
    # Paper Eq. 23: tanh(m*delta)/ (m*delta).
    eta_f = np.tanh(m_delta) / m_delta
    if not np.all(np.isfinite(eta_f)) or np.any((eta_f <= 0.0) | (eta_f > 1.0)):
        raise ValueError("Eq. 23 produced an invalid fin efficiency.")
    return float(eta_f) if eta_f.ndim == 0 else eta_f

def corrected_coolant_coefficient_eq24(h_c, width, eta_f, delta, w_b):
    """Paper Eq. 24: apply fin efficiency to the coolant-side coefficient."""
    try:
        h_c, width, eta_f, delta, w_b = np.broadcast_arrays(*[
            np.asarray(v, dtype=float) for v in (h_c, width, eta_f, delta, w_b)
        ])
    except ValueError as exc:
        raise ValueError("Eq. 24 inputs must be broadcast-compatible.") from exc
    arrays = (h_c, width, eta_f, delta, w_b)
    if not all(np.all(np.isfinite(v)) for v in arrays):
        raise ValueError("Eq. 24 inputs must be finite.")
    if any(np.any(v <= 0.0) for v in (h_c, width, delta, w_b)):
        raise ValueError("Eq. 24 requires positive h_c and channel dimensions.")
    if np.any((eta_f <= 0.0) | (eta_f > 1.0)):
        raise ValueError("Eq. 24 requires fin efficiency 0 < eta_f <= 1.")
    # Paper Eq. 24.
    h_c_corrected = h_c * (width + 2.0 * eta_f * delta) / (width + w_b)
    if not np.all(np.isfinite(h_c_corrected)) or np.any(h_c_corrected <= 0.0):
        raise ValueError("Eq. 24 produced an invalid corrected coefficient.")
    return float(h_c_corrected) if h_c_corrected.ndim == 0 else h_c_corrected

def chen_corrected_coefficient_eq14_to24(
    h_nb, h_fc, X_tt, Re_l, width, w_b, k_wall, delta
):
    """Apply Chen Eqs. 14, 18, 22 and fin corrections Eqs. 23-24.

    The caller must supply a valid two-phase ``h_nb`` (Eq. 21) and X_tt
    (Eq. 19). This function deliberately does not infer quality or saturation
    properties.
    """
    F = chen_reynolds_factor_eq18(X_tt)
    S = chen_suppression_factor_eq22(Re_l, F)
    h_c = chen_heat_transfer_coefficient_eq14(h_nb, h_fc, F, S)
    eta_f = fin_efficiency_eq23(h_c, w_b, k_wall, delta)
    h_c_corrected = corrected_coolant_coefficient_eq24(
        h_c, width, eta_f, delta, w_b
    )
    return h_c_corrected, h_c, F, S, eta_f

def single_phase_corrected_coefficient_eq14_to24(
    Re, T_bulk, T_wall, p, Dh, rho, mu_bulk, cp, k_fluid,
    width, w_b, k_wall, delta,
):
    """Calculate single-phase h_c using paper Eqs. 14-17 and 23-24.

    Eqs. 18-22 are two-phase boiling terms and therefore are not activated
    for the configured single-phase coolant. Returns corrected h_c plus raw
    h_c and fin efficiency. T_wall is still an input from the Eq. 3 solve.
    """
    h_c = single_phase_coolant_coefficient_eq14_to17(
        Re, T_bulk, T_wall, p, Dh, rho, mu_bulk, cp, k_fluid
    )
    eta_f = fin_efficiency_eq23(h_c, w_b, k_wall, delta)
    h_c_corrected = corrected_coolant_coefficient_eq24(
        h_c, width, eta_f, delta, w_b
    )
    return h_c_corrected, h_c, eta_f

def march_coolant_energy(T_in, q_cell, cp_cell, A_heating_cell, m_dot_total):
    """March coolant bulk temperature using the paper's Eq. 7 per cell.

    q_cell is supplied in W/m². Areas are the total heated areas from Eq. 8,
    so the matching denominator is total mass flow (not per-channel flow).
    """
    q_cell = np.asarray(q_cell, dtype=float)
    cp_cell = np.asarray(cp_cell, dtype=float)
    A_heating_cell = np.asarray(A_heating_cell, dtype=float)
    if q_cell.ndim != 1 or q_cell.size == 0:
        raise ValueError("q_cell must be a non-empty one-dimensional array.")
    if cp_cell.shape != q_cell.shape or A_heating_cell.shape != q_cell.shape:
        raise ValueError("q_cell, cp_cell, and heated-area arrays must have equal lengths.")
    if not np.all(np.isfinite(q_cell)) or not np.all(np.isfinite(cp_cell)) or not np.all(np.isfinite(A_heating_cell)):
        raise ValueError("Eq. 7 inputs must be finite.")
    if not np.isfinite(T_in) or T_in <= 0.0 or not np.isfinite(m_dot_total) or m_dot_total <= 0.0:
        raise ValueError("Eq. 7 requires positive finite inlet temperature and total flow.")
    if np.any(q_cell < 0.0) or np.any(cp_cell <= 0.0) or np.any(A_heating_cell <= 0.0):
        raise ValueError("Eq. 7 requires non-negative heat flux and positive cp and area.")

    T = np.empty(q_cell.size + 1, dtype=float)
    T[0] = T_in
    for i in range(q_cell.size):
        # Paper Eq. 7: ΔTc = q A_heating / (m_dot cp), discretized per cell.
        T[i + 1] = T[i] + q_cell[i] * A_heating_cell[i] / (m_dot_total * cp_cell[i])
    if not np.all(np.isfinite(T)):
        raise RuntimeError("Eq. 7 produced a non-finite coolant temperature.")

    energy_in = m_dot_total * np.sum(cp_cell * np.diff(T))
    energy_from_flux = np.sum(q_cell * A_heating_cell)
    if not np.isclose(energy_in, energy_from_flux, rtol=1e-12, atol=1e-9):
        raise RuntimeError("Coolant energy march does not balance the supplied heat input.")
    return T

def pressure_loss(f, rho, velocity, dx, Dh):
    return f * (rho * velocity**2 / 2.0) * (dx / Dh)

def march_pressure(p_in, coolant_temperature_nodes=None):
    p = np.zeros(n_nodes)
    if coolant_temperature_nodes is None:
        T = np.full(n_nodes, T_coolant)
    else:
        T = np.asarray(coolant_temperature_nodes, dtype=float)
        if T.shape != (n_nodes,) or not np.all(np.isfinite(T)) or np.any(T <= 0.0):
            raise ValueError(
                "Coolant temperature profile must be finite, positive, and have one value per axial node."
            )

    velocity = np.zeros(n_cells)
    rho_cell = np.zeros(n_cells)
    mu_cell = np.zeros(n_cells)
    Re = np.zeros(n_cells)
    f = np.zeros(n_cells)
    dp = np.zeros(n_cells)

    p[0] = p_in

    for i in range(n_cells):
        # Evaluate properties at the current upstream state.
        rho, mu, cp, h, k = properties(T[i], p[i])

        # The configured flow is total across all parallel channels.
        velocity[i] = m_dot_channel / (rho * area[i])
        rho_cell[i] = rho
        mu_cell[i] = mu

        Re[i] = Reynolds(
            V=velocity[i],
            D=Dh[i],
            rho=rho,
            mu=mu
        )

        f[i] = friction_factor(
            Re=Re[i],
            eD=relative_roughness
        )

        dp[i] = pressure_loss(
            f[i], rho, velocity[i], dx[i], Dh[i]
        )

        # Downstream pressure.
        p[i + 1] = p[i] - dp[i]

        if not np.isfinite(p[i + 1]) or p[i + 1] <= 0.0:
            raise RuntimeError(
                f"Invalid pressure at node {i + 1}: {p[i + 1]}"
            )

    return p, T, velocity, rho_cell, Re, f, dp

def solve_cell_thermal_equilibrium_iterative(
    T_aw, T_coolant, p, h_g, q_rad, Re, Dh, rho, mu_bulk, cp, k_fluid,
    width, t_wall, w_b, delta, tolerance_K=0.1, max_iterations=250,
    cell_index=None, x_position_m=None,
):
    """Iterate Eq. 3 with single-phase coolant Eqs. 14-17 and 23-24.

    The temperature-dependent wall conductivity is evaluated at the mean
    through-wall temperature, an explicit representative-temperature choice.
    Returns q, both wall temperatures, k_wall, corrected h_c, and iterations.
    """
    inputs = np.asarray(
        [T_aw, T_coolant, p, h_g, q_rad, Re, Dh, rho, mu_bulk, cp, k_fluid,
         width, t_wall, w_b, delta, tolerance_K], dtype=float
    )
    if not np.all(np.isfinite(inputs)):
        raise ValueError("Local thermal-iteration inputs must be finite.")
    if np.any(inputs[:4] <= 0.0) or np.any(inputs[5:] <= 0.0):
        raise ValueError("Local thermal-iteration temperatures and properties must be positive.")
    if q_rad < 0.0 or tolerance_K <= 0.0 or max_iterations <= 0:
        raise ValueError("Local thermal iteration has invalid radiation or iteration controls.")
    if T_aw <= T_coolant:
        raise ValueError("Eq. 3 requires adiabatic wall temperature above coolant bulk temperature.")

    # Solve the two independent Eq. 3 balances for the unknown wall temperatures.
    # Scaling residuals keeps least-squares conditioning useful across the nozzle.
    heat_scale = max(h_g * (T_aw - T_coolant) + q_rad, 1.0e5)
    residual_history = []

    def residuals(wall_temperatures):
        T_wall_hot, T_wall_coolant = wall_temperatures
        T_wall_mean = 0.5 * (T_wall_hot + T_wall_coolant)
        k_wall = inconel718_conductivity(T_wall_mean)
        h_c, _, _ = single_phase_corrected_coefficient_eq14_to24(
            Re, T_coolant, T_wall_coolant, p, Dh, rho, mu_bulk, cp, k_fluid,
            width, w_b, k_wall, delta,
        )
        q_gas = h_g * (T_aw - T_wall_hot) + q_rad
        q_wall = k_wall / t_wall * (T_wall_hot - T_wall_coolant)
        q_coolant = h_c * (T_wall_coolant - T_coolant)
        normalized = np.asarray([(q_gas - q_wall) / heat_scale,
                                 (q_wall - q_coolant) / heat_scale])
        residual_history.append(normalized.copy())
        return normalized

    upper_wall_temperature = 1375.0  # Upper validated limit of supplied k_wall(T).
    try:
        T_sat = PropsSI("T", "P", p, "Q", 0, fluid)
    except Exception as exc:
        raise RuntimeError(
            f"Could not determine {fluid} saturation temperature at p={p} Pa."
        ) from exc
    upper_wall_coolant = min(upper_wall_temperature, T_sat - 0.1)
    if T_coolant >= upper_wall_coolant:
        raise RuntimeError(
            f"Bulk coolant T={T_coolant:.3f} K is not below the subcooled wall limit "
            f"{upper_wall_coolant:.3f} K at p={p:.3f} Pa."
        )
    wall_headroom = upper_wall_temperature - T_coolant
    if wall_headroom <= 0.0:
        raise ValueError(
            "Coolant temperature is at or above the supported Inconel conductivity range."
        )
    initial = np.asarray([
        T_coolant + min(0.65 * (T_aw - T_coolant), 0.90 * wall_headroom),
        T_coolant + min(0.10 * (T_aw - T_coolant), 0.45 * (upper_wall_coolant - T_coolant)),
    ])
    upper_bounds = np.asarray([upper_wall_temperature, upper_wall_coolant])
    initial = np.minimum(initial, upper_bounds - 1e-6)
    solution = least_squares(
        residuals,
        initial,
        bounds=(np.asarray([T_coolant, T_coolant]), upper_bounds),
        xtol=1e-12,
        ftol=1e-12,
        gtol=1e-12,
        max_nfev=max_iterations * 10,
    )
    if not solution.success or np.max(np.abs(residuals(solution.x))) > 1e-8:
        plot_eq3_residual_convergence(
            residual_history, cell_index=cell_index, x_position_m=x_position_m
        )
        raise RuntimeError(
            "No converged single-phase Eq. 3 wall balance below coolant "
            "saturation; the requested no-boiling solution is not supported "
            "by the current inputs. "
            f"{solution.message}; wall trial={solution.x} K, "
            f"bounds={upper_bounds} K, normalized residuals={residuals(solution.x)}."
        )

    T_wall_hot, T_wall_coolant = solution.x
    if T_wall_hot <= T_wall_coolant:
        raise RuntimeError("Eq. 3 solution has hot-wall temperature below coolant-side wall temperature.")
    T_wall_mean = 0.5 * (T_wall_hot + T_wall_coolant)
    k_wall = inconel718_conductivity(T_wall_mean)
    h_c, _, _ = single_phase_corrected_coefficient_eq14_to24(
        Re, T_coolant, T_wall_coolant, p, Dh, rho, mu_bulk, cp, k_fluid,
        width, w_b, k_wall, delta,
    )
    q, T_wall_hot_check, T_wall_coolant_check = solve_thermal_equilibrium_eq3(
        T_aw, T_coolant, h_g, q_rad, k_wall, t_wall, h_c
    )
    consistency = max(
        abs(T_wall_hot_check - T_wall_hot),
        abs(T_wall_coolant_check - T_wall_coolant),
    )
    if consistency > tolerance_K:
        plot_eq3_residual_convergence(
            residual_history, cell_index=cell_index, x_position_m=x_position_m
        )
        raise RuntimeError(
            f"Eq. 3 wall solution missed the paper's {tolerance_K:g} K "
            f"convergence tolerance by {consistency:.6g} K."
        )
    return q, T_wall_hot_check, T_wall_coolant_check, k_wall, h_c, solution.nfev

def write_thermal_outputs_to_csv(path, node_profiles):
    """Write derived thermal profiles while preserving source CSV columns/data."""
    with path.open("r", newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        fieldnames = reader.fieldnames
        rows = list(reader)
    if fieldnames is None:
        raise ValueError(f"Cannot write thermal results: no CSV header in {path}.")
    missing = sorted(set(node_profiles) - set(fieldnames))
    if missing:
        raise ValueError(
            "Cannot write thermal results; output columns are missing: "
            + ", ".join(missing)
        )
    if len(rows) != n_nodes:
        raise ValueError(
            f"Thermal output has {len(rows)} rows but axial mesh has {n_nodes} nodes."
        )
    if any(np.asarray(values).shape != (n_nodes,) for values in node_profiles.values()):
        raise ValueError("Every thermal output profile must have one value per CSV node.")
    for i, row in enumerate(rows):
        for column, values in node_profiles.items():
            value = float(values[i])
            row[column] = f"{value:.10g}" if np.isfinite(value) else ""
    temporary_path = path.with_name(path.name + ".tmp")
    with temporary_path.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary_path.replace(path)
