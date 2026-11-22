import io
import os
import time
import numpy as np
import pandas as pd
import requests

# 1. Define your solver's temperature and pressure grid bounds (in SI units)
T_MIN_K = 310.0
T_MAX_K = 710.0
T_STEP_K = 1.0          # 1 K resolution

P_MIN_PA = 6.5e6        # Minimum expected channel pressure (e.g., p_target_out)
P_MAX_PA = 8.1e6        # Maximum expected channel pressure (e.g., p_in)
P_NUM_POINTS = 31       # Number of pressure grid points (e.g., every 0.1 MPa)

OUTPUT_DIR = "n2o_property_tables"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def fetch_nist_isobar(p_pa: float, t_low: float, t_high: float, t_inc: float) -> pd.DataFrame:
    """Fetches a single isobaric temperature sweep from NIST WebBook CGI."""
    url = "https://webbook.nist.gov/cgi/fluid.cgi"
    params = {
        "Action": "Data",
        "Wide": "on",
        "ID": "C10024972",       # CAS ID for Nitrous Oxide (N2O)
        "Type": "IsoBar",
        "Digits": "6",
        "P": f"{p_pa / 1e6:.6f}",  # Convert Pa -> MPa for NIST
        "THigh": str(t_high),
        "TLow": str(t_low),
        "TInc": str(t_inc),
        "RefState": "DEF",
        "TUnit": "K",
        "PUnit": "MPa",
        "DUnit": "kg/m3",
        "HUnit": "kJ/kg",
        "WUnit": "m/s",
        "VisUnit": "Pa*s",
        "STUnit": "N/m",
    }
    resp = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()
    return pd.read_csv(io.StringIO(resp.text), sep="\t")


# 2. Define mapping from output CSV filenames to NIST columns and SI unit multipliers
PROPERTY_MAP = {
    "density_kg_m3.csv": ("Density (kg/m3)", 1.0),
    "cp_j_kgk.csv": ("Cp (J/g*K)", 1000.0),                # J/(g*K) -> J/(kg*K)
    "cv_j_kgk.csv": ("Cv (J/g*K)", 1000.0),                # J/(g*K) -> J/(kg*K)
    "viscosity_pa_s.csv": ("Viscosity (Pa*s)", 1.0),
    "conductivity_w_mk.csv": ("Therm. Cond. (W/m*K)", 1.0),
    "enthalpy_j_kg.csv": ("Enthalpy (kJ/kg)", 1000.0),     # kJ/kg -> J/kg
    "sound_speed_m_s.csv": ("Sound Spd. (m/s)", 1.0),
}

# Initialize empty dictionary of DataFrames (indexed by Temperature in K)
t_grid = np.arange(T_MIN_K, T_MAX_K + T_STEP_K / 2, T_STEP_K)
p_grid_pa = np.linspace(P_MIN_PA, P_MAX_PA, P_NUM_POINTS)

tables = {
    fname: pd.DataFrame(index=pd.Index(t_grid, name="T_K"))
    for fname in PROPERTY_MAP
}

# 3. Query each isobar and populate columns (named by pressure in Pa)
# 3. Query each isobar and populate columns (named by pressure in Pa)
for p_pa in p_grid_pa:
    print(f"Fetching N2O isobar at P = {p_pa / 1e6:.3f} MPa...")
    df_iso = fetch_nist_isobar(p_pa, T_MIN_K, T_MAX_K, T_STEP_K)

    # Index by Temperature and drop any duplicate saturation boundary rows
    df_iso["Temperature (K)"] = pd.to_numeric(df_iso["Temperature (K)"], errors="coerce")
    df_iso = df_iso.dropna(subset=["Temperature (K)"]).set_index("Temperature (K)")
    df_iso = df_iso[~df_iso.index.duplicated(keep="last")]

    col_label = f"{p_pa:.1f}"  # Column header is exact pressure in Pa
    for fname, (nist_col, scale) in PROPERTY_MAP.items():
        # Convert strings like 'undefined' or '*' to NaN, interpolate across them, and scale
        numeric_col = pd.to_numeric(df_iso[nist_col], errors="coerce")
        series = (
            numeric_col.reindex(t_grid)
            .interpolate(method="linear", limit_direction="both")
            * scale
        )
        tables[fname][col_label] = series

    time.sleep(0.3)  # Brief pause to avoid rate-limiting on NIST servers

# 4. Write each property to its own standalone CSV file
for fname, df_prop in tables.items():
    out_path = os.path.join(OUTPUT_DIR, fname)
    df_prop.to_csv(out_path)
    print(f"Saved {out_path} ({df_prop.shape[0]} temps x {df_prop.shape[1]} pressures)")
