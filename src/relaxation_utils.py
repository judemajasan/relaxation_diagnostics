import numpy as np
import pandas as pd
import pybamm


# -----------------------------
# 1. Experiment and model setup
# -----------------------------
def make_experiment(discharge_min=20, rest_min=10):
    return pybamm.Experiment(
        [
            pybamm.step.string(
                f"Discharge at 1C for {discharge_min} minutes",
                period="1 second",
            ),
            pybamm.step.string(
                f"Rest for {rest_min} minutes",
                period="0.1 second",
            ),
        ]
    )


def make_model():
    return pybamm.lithium_ion.DFN({"SEI": "constant"})


def make_base_params():
    return pybamm.ParameterValues("Chen2020")


# -----------------------------
# 2. Storage-loss / LLI-like perturbation
# -----------------------------
def make_storage_loss_params(delta_q_ah):
    param = make_base_params()

    c_init_n = param["Initial concentration in negative electrode [mol.m-3]"]
    L_SEI_0 = param["Initial SEI thickness [m]"]
    eps_neg = param["Negative electrode porosity"]

    F = 96485.3
    L_n = param["Negative electrode thickness [m]"]
    L_y = param["Electrode width [m]"]
    L_z = param["Electrode height [m]"]
    V_n = L_n * L_y * L_z

    R_n = param["Negative particle radius [m]"]
    eps_act_n = param["Negative electrode active material volume fraction"]
    a_n = 3 * eps_act_n / R_n

    V_bar_SEI = param["SEI partial molar volume [m3.mol-1]"]
    z_SEI = param["Ratio of lithium moles to SEI moles"]

    c_SEI = delta_q_ah * 3600 / (F * z_SEI * V_n)
    delta_c_n = c_SEI * z_SEI / eps_act_n

    c_init_degraded = c_init_n - delta_c_n
    L_SEI_0_degraded = L_SEI_0 + c_SEI * V_bar_SEI / a_n
    eps_neg_degraded = eps_neg - (L_SEI_0_degraded - L_SEI_0) * a_n

    param_degraded = param.copy()
    param_degraded.update(
        {
            "Initial concentration in negative electrode [mol.m-3]": c_init_degraded,
            "Initial SEI thickness [m]": L_SEI_0_degraded,
            "Negative electrode porosity": eps_neg_degraded,
        }
    )

    return param_degraded


# -----------------------------
# 3. Rest-window extraction
# -----------------------------
def detect_rest_start(solution, current_tol=1e-10):
    t = solution["Time [s]"].entries
    I = solution["Current [A]"].entries
    zero_idx = np.where(np.abs(I) < current_tol)[0]
    if len(zero_idx) == 0:
        raise RuntimeError("Could not detect start of rest step from current trace.")
    return t[zero_idx[0]]


def extract_rest_window(solution, rest_window_s=120):
    t = solution["Time [s]"].entries
    V = solution["Voltage [V]"].entries
    rest_start = detect_rest_start(solution)

    mask = (t >= rest_start) & (t <= rest_start + rest_window_s)
    rest_t = t[mask] - rest_start
    rest_V = V[mask]

    return rest_t, rest_V, rest_start


# -----------------------------
# 4. Relaxation features
# -----------------------------
def interp_at(x, y, xq):
    return np.interp(xq, x, y)


def safe_linear_slope(x, y):
    if len(x) < 2:
        return np.nan
    return np.polyfit(x, y, 1)[0]


def safe_log_time_slope(t, V):
    if len(t) < 2:
        return np.nan
    x = np.log1p(t)
    return np.polyfit(x, V, 1)[0]


def compute_features(rest_t, rest_V, window_s, tol=1e-9):
    mask = rest_t <= window_s + tol
    t = rest_t[mask]
    V = rest_V[mask]

    if len(t) < 2:
        return {
            "window_s": window_s,
            "fast_recovery_5s_V": np.nan,
            "initial_slope_0_5_V_per_s": np.nan,
            "log_time_slope_5_30_V_per_log_s": np.nan,
            "relaxation_area_Vs": np.nan,
        }

    # Fast feature
    t_fast_end = min(5.0, t[-1])
    fast_recovery = interp_at(t, V, t_fast_end) - V[0]

    mask_fast = t <= t_fast_end + tol
    slope_0_5 = safe_linear_slope(t[mask_fast], V[mask_fast])

    # Intermediate feature: 5–30 s
    if t[-1] >= 30.0 - tol:
        mask_mid = (t >= 5.0 - tol) & (t <= 30.0 + tol)
        log_slope_5_30 = safe_log_time_slope(t[mask_mid], V[mask_mid])
    else:
        log_slope_5_30 = np.nan

    # Cumulative feature
    area = np.trapezoid(V - V[0], t)

    return {
        "window_s": window_s,
        "fast_recovery_5s_V": fast_recovery,
        "initial_slope_0_5_V_per_s": slope_0_5,
        "log_time_slope_5_30_V_per_log_s": log_slope_5_30,
        "relaxation_area_Vs": area,
    }


def compute_all_windows(rest_t, rest_V, windows_s=[5, 10, 30, 60, 120]):
    return pd.DataFrame([compute_features(rest_t, rest_V, w) for w in windows_s])


# -----------------------------
# 5. Voltage-component extraction
# -----------------------------
COMPONENT_KEYS = {
    "Battery OCV [V]": "Battery open-circuit voltage [V]",
    "Reaction overpotential [V]": "X-averaged battery reaction overpotential [V]",
    "Solid ohmic losses [V]": "X-averaged battery solid phase ohmic losses [V]",
    "Electrolyte ohmic losses [V]": "X-averaged battery electrolyte ohmic losses [V]",
    "Concentration overpotential [V]": "X-averaged battery concentration overpotential [V]",
    "Battery voltage [V]": "Battery voltage [V]",
}


def extract_component_rest_window(solution, rest_window_s=120):
    t = solution["Time [s]"].entries
    rest_start = detect_rest_start(solution)
    mask = (t >= rest_start) & (t <= rest_start + rest_window_s)
    rest_t = t[mask] - rest_start

    out = {"time_s": rest_t}
    for pretty_name, key in COMPONENT_KEYS.items():
        out[pretty_name] = solution[key].entries[mask]
    return out


def component_summaries(comp_df):
    t = comp_df["time_s"].values

    summaries = {}

    for key in [
        "Reaction overpotential [V]",
        "Solid ohmic losses [V]",
        "Electrolyte ohmic losses [V]",
        "Concentration overpotential [V]",
    ]:
        y = comp_df[key].values
        summaries[f"{key}__start"] = y[0]
        summaries[f"{key}__end_5s"] = np.interp(5.0, t, y)
        summaries[f"{key}__end_30s"] = np.interp(30.0, t, y)
        summaries[f"{key}__drop_0_5s"] = y[0] - np.interp(5.0, t, y)
        summaries[f"{key}__drop_5_30s"] = np.interp(5.0, t, y) - np.interp(30.0, t, y)

    return pd.Series(summaries)