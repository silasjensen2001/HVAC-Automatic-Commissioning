import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from models import HVAC
from controller import StateFeedbackControllerDisturbanceRejection, StateFeedbackController


# ── Main run configuration ────────────────────────────────────────────────────
model_mode = "nonlinear"  # "linear" or "nonlinear"

RUN_MODE = "compare"  # "compare", "lqr_only", "lmi_only"

LQR_TUNING_MODE = "bryson"  # "manual", "bryson", "bryson_sweep"
LMI_TUNING_MODE = "bryson"  # "manual", "bryson", "bryson_sweep"

# Options: "sinusoid", "weather_profile", "step", "stochastic", "constant"
TEST_CASE = "weather_profile"


# ── Sweep and simulation settings ─────────────────────────────────────────────
USE_PARALLEL_SWEEP = True
MAX_PARALLEL_WORKERS = 2

deadband_margin = 0.0005        # [°C], acceptable temperature margin around reference
metrics_ignore_fraction = 0.001 # Ignore first 0.1% of simulation when computing metrics


# ── References and disturbance base values ────────────────────────────────────
T1_ref = 10.0 + 273.15   # Cooler air outlet setpoint [K]
T2_ref = 20.0 + 273.15   # Heater air outlet setpoint [K]
r      = np.array([T1_ref, T2_ref])

T_in  = 23 + 273.15
t_day = 24 * 3600


# ── Bryson tuning limits ──────────────────────────────────────────────────────
air_temp_max_error = 1.0     # [K]
water_temp_max_error = 50.0  # [K]
wanted_settling_time = 5.0   # [s]


# ── Manual Q/R settings ───────────────────────────────────────────────────────
# Used only when LQR_TUNING_MODE or LMI_TUNING_MODE is set to "manual".
LQR_MANUAL_Q_SCALE = 10000
LQR_MANUAL_R_SCALE = 8

LMI_MANUAL_Q_SCALE = 100
LMI_MANUAL_R_SCALE = 10


# ── Sweep factor grid ─────────────────────────────────────────────────────────
Qx_factors = [0.25, 0.5, 1.0, 2.0, 4.0]
Qi_factors = [0.25, 0.5, 1.0, 2.0, 4.0]
R_factors  = [0.25, 0.5, 1.0, 2.0, 4.0]


# ── Plant parameters ──────────────────────────────────────────────────────────
params_cooler = dict(
    type                  = "cooler",
    num_segments          = 5,
    num_pipes             = 10,
    gamma                 = 951.87,
    cross_area_water      = 0.000201*2,
    heat_exchanger_depth  = 0.06*2,
    heat_exchanger_width  = 0.5,
    heat_exchanger_height = 0.5,
    volume_flow_wet_air   = 0.72634,
    water_supply_T        = 4.0 + 273.15,
    Kvs                   = 1.6471,
)

params_heater = dict(
    type                  = "heater",
    num_segments          = 5,
    num_pipes             = 10,
    gamma                 = 951.87,
    cross_area_water      = 0.000201,
    heat_exchanger_depth  = 0.06,
    heat_exchanger_width  = 0.5,
    heat_exchanger_height = 0.5,
    volume_flow_wet_air   = 0.72634,
    water_supply_T        = 66.9 + 273.15,
    Kvs                   = 1.6471,
)


# ── Instantiate plant ─────────────────────────────────────────────────────────
#hvac = HVAC(configs=[params_cooler, params_heater], mode=model_mode, const_disturbance=28 + 273.15)
hvac = HVAC(configs=[params_cooler, params_heater], mode=model_mode, const_disturbance=None)


# ── Export state-space model ──────────────────────────────────────────────────
data_dir = Path(__file__).resolve().parent.parent / "models/linear"
data_dir.mkdir(parents=True, exist_ok=True)
hvac._export_state_space(data_dir / "HVAC_model.mat")


# ── Dimensions ────────────────────────────────────────────────────────────────
K = hvac._lin_components[0].K
N = hvac.total_states   # 4K = 20


# ── Bryson bounds ─────────────────────────────────────────────────────────────
# State ordering in the simulation:
#   0:K       cooler air
#   K:2K      cooler water
#   2K:3K     heater air
#   3K:4K     heater water
#
# Therefore the correct limit ordering is:
#   [air, water, air, water]
x_max = np.concatenate([
    np.full(K, air_temp_max_error),
    np.full(K, water_temp_max_error),
    np.full(K, air_temp_max_error),
    np.full(K, water_temp_max_error),
])

# Integral states: air temperature error accumulated over a desired time scale
xI_max = np.full(2, air_temp_max_error * wanted_settling_time)

# Input weighting limit
u_max = np.array([1.0, 1.0])


# ── Initial conditions ────────────────────────────────────────────────────────
x0 = np.concatenate([
    np.full(K, 23 + 273.15),          # cooler air
    np.full(K, 23 + 273.15),          # cooler water
    np.full(K, 9.9 + 273.15),         # heater air
    np.full(K, 9.9 + 273.15),         # heater water
])


# ── Bryson Q/R helper ─────────────────────────────────────────────────────────
def bryson_weight_values(
    x_max_dev: np.ndarray,
    xI_max: np.ndarray,
    u_max: np.ndarray,
    Qx_factor: float = 1.0,
    Qi_factor: float = 1.0,
    R_factor: float = 1.0,
):
    """
    Returns the actual diagonal Qx, Qi, and R weights implied by Bryson-style
    normalization with optional dimensionless tuning factors.

        Qx_i = Qx_factor / x_max_dev_i²
        Qi_i = Qi_factor / xI_max_i²
        R_i  = R_factor  / u_max_i²

    Pure Bryson corresponds to all factors being 1.
    """
    x_max_dev = np.asarray(x_max_dev, dtype=float)
    xI_max = np.asarray(xI_max, dtype=float)
    u_max = np.asarray(u_max, dtype=float)

    Qx_weights = Qx_factor / x_max_dev**2
    Qi_weights = Qi_factor / xI_max**2
    R_weights  = R_factor  / u_max**2

    return Qx_weights, Qi_weights, R_weights


def bryson_cost_matrices(
    plant,
    x_max_dev: np.ndarray,
    xI_max: np.ndarray,
    u_max: np.ndarray,
    Qx_factor: float = 1.0,
    Qi_factor: float = 1.0,
    R_factor: float = 1.0,
):
    """
    Creates diagonal Q and R matrices using Bryson-style normalized limits.

    Augmented state:
        x_aug = [x, x_I]

    Cost structure:
        Q = diag(Qx_1, ..., Qx_n, Qi_1, ..., Qi_p)
        R = diag(R_1, ..., R_m)

    where:
        Qx_i = Qx_factor / x_max_dev_i²
        Qi_i = Qi_factor / xI_max_i²
        R_i  = R_factor  / u_max_i²
    """
    n = plant.A.shape[0]
    m = plant.B_u.shape[1]
    p = plant.C.shape[0]

    Qx_weights, Qi_weights, R_weights = bryson_weight_values(
        x_max_dev=x_max_dev,
        xI_max=xI_max,
        u_max=u_max,
        Qx_factor=Qx_factor,
        Qi_factor=Qi_factor,
        R_factor=R_factor,
    )

    if len(Qx_weights) != n:
        raise ValueError(f"x_max_dev must have length {n}, got {len(Qx_weights)}.")
    if len(Qi_weights) != p:
        raise ValueError(f"xI_max must have length {p}, got {len(Qi_weights)}.")
    if len(R_weights) != m:
        raise ValueError(f"u_max must have length {m}, got {len(R_weights)}.")

    Q = np.diag(np.concatenate([Qx_weights, Qi_weights]))
    R = np.diag(R_weights)

    return Q, R


# ── Disturbance definitions ───────────────────────────────────────────────────
def make_disturbance(test_case: str):
    """
    Returns:
        d:      callable d(t) returning disturbance vector in Kelvin
        t_end:  simulation end time [s]
        n_eval: number of evaluation points
        label:  text label for plots
    """

    if test_case == "sinusoid":
        Amp = 5.0
        T_period = 60 #t_day

        def d(t):
            T_in_sys = T_in + Amp * np.sin(2*np.pi*t/T_period)
            return np.array([T_in_sys])

        t_end = 50 #t_day
        n_eval = 200
        label = f"Sinusoidal inlet disturbance, A={Amp:.1f} °C, period={T_period/3600:.1f} h"
        return d, t_end, n_eval, label
    
    elif test_case == "weather_profile":
        def d(t):
            shift_t = t - 54000

            term1 = 18.0
            term2 = 6.0 * np.cos((2 * np.pi * shift_t) / (86400))
            term3 = 1.6 * np.cos((2 * np.pi * shift_t) / (43200))
            term4 = 0.5 * np.cos((2 * np.pi * shift_t) / (28800))
            term5 = 2.0 * np.sin((2 * np.pi * t) / (259200))

            T_in_sys = term1 + term2 + term3 + term4 + term5 + 273.15
            return np.array([T_in_sys])

        t_end = 2 * t_day
        n_eval = 10 * t_day
        label = "Weather-like inlet disturbance with daily and multi-day harmonics"
        return d, t_end, n_eval, label

    elif test_case == "step":
        step_amp = 5.0
        step_time = 20.0

        def d(t):
            T_in_sys = T_in + (step_amp if t >= step_time else 0.0)
            return np.array([T_in_sys])

        t_end = 50
        n_eval = 200
        label = f"Step inlet disturbance, ΔT={step_amp:.1f} °C at t={step_time:.0f} s"
        return d, t_end, n_eval, label

    elif test_case == "stochastic":
        sigma = 3.0
        sample_time = 30.0
        seed = 1

        t_end = 600 #t_day
        n_eval = 500

        rng = np.random.default_rng(seed)
        noise_t = np.arange(0.0, t_end + sample_time, sample_time)
        noise_values = rng.normal(loc=0.0, scale=sigma, size=len(noise_t))

        def d(t):
            noise = np.interp(t, noise_t, noise_values)
            T_in_sys = T_in + noise
            return np.array([T_in_sys])

        label = f"Stochastic inlet disturbance, σ={sigma:.1f} °C, sample time={sample_time:.0f} s"
        return d, t_end, n_eval, label

    elif test_case == "constant":
        def d(t):
            return np.array([T_in])

        t_end = 30
        n_eval = 3000
        label = "Constant inlet temperature"
        return d, t_end, n_eval, label

    else:
        raise ValueError(
            f"Unknown TEST_CASE='{test_case}'. Use 'sinusoid', 'weather_profile', 'step', 'stochastic', or 'constant'."
        )


d, t_end, n_eval, disturbance_label = make_disturbance(TEST_CASE)
t_eval = np.linspace(0, t_end, n_eval)


# ── Performance metrics ───────────────────────────────────────────────────────
def step_metrics(t, y, target, step_time=0.0, tol=0.02):
    t = np.asarray(t)
    y = np.asarray(y)

    idx = t >= step_time
    t2 = t[idx] - step_time
    y2 = y[idx]

    if len(t2) < 2:
        return dict(rise_time=np.nan, settling_time=np.nan, overshoot=np.nan, peak_error=np.nan)

    y_initial = y2[0]
    y_final = target
    delta = y_final - y_initial

    error = y2 - y_final
    peak_error = np.max(np.abs(error))

    if abs(delta) > 1e-9:
        if delta > 0:
            overshoot = max(0.0, np.max(y2) - y_final) / abs(delta) * 100.0
        else:
            overshoot = max(0.0, y_final - np.min(y2)) / abs(delta) * 100.0
    else:
        overshoot = np.nan

    rise_time = np.nan
    if abs(delta) > 1e-9:
        y10 = y_initial + 0.1 * delta
        y90 = y_initial + 0.9 * delta

        if delta > 0:
            idx10 = np.where(y2 >= y10)[0]
            idx90 = np.where(y2 >= y90)[0]
        else:
            idx10 = np.where(y2 <= y10)[0]
            idx90 = np.where(y2 <= y90)[0]

        if len(idx10) > 0 and len(idx90) > 0:
            rise_time = t2[idx90[0]] - t2[idx10[0]]

    band = tol * max(1.0, abs(y_final))
    outside = np.where(np.abs(error) > band)[0]

    if len(outside) == 0:
        settling_time = 0.0
    elif outside[-1] < len(t2) - 1:
        settling_time = t2[outside[-1] + 1]
    else:
        settling_time = np.nan

    return dict(
        rise_time=rise_time,
        settling_time=settling_time,
        overshoot=overshoot,
        peak_error=peak_error,
    )


def deadband_error_metrics(t, y, reference, margin=0.02):
    """
    Measures only the part of the output error that exceeds the allowed margin.

    If |y - reference| <= margin:
        contribution = 0

    If |y - reference| > margin:
        contribution = |y - reference| - margin
    """
    t = np.asarray(t)
    y = np.asarray(y)

    error = y - reference
    excess_error = np.maximum(np.abs(error) - margin, 0.0)

    deadband_iae = np.trapz(excess_error, t)
    deadband_rms = np.sqrt(np.mean(excess_error**2))
    deadband_peak = np.max(excess_error)

    outside = excess_error > 0
    time_outside = np.trapz(outside.astype(float), t)
    fraction_outside = time_outside / (t[-1] - t[0])

    return dict(
        deadband_iae=deadband_iae,
        deadband_rms=deadband_rms,
        deadband_peak=deadband_peak,
        time_outside=time_outside,
        fraction_outside=fraction_outside,
    )


def deadband_excursion_metrics(t, y, reference, margin=0.02):
    """
    Measures how often and how long the output leaves the acceptable band.
    """
    t = np.asarray(t)
    y = np.asarray(y)

    error = y - reference
    outside = np.abs(error) > margin

    excursions = []
    start_time = None

    for i in range(len(t)):
        if outside[i] and start_time is None:
            start_time = t[i]

        if not outside[i] and start_time is not None:
            excursions.append(t[i] - start_time)
            start_time = None

    if start_time is not None:
        excursions.append(t[-1] - start_time)

    excursions = np.array(excursions)

    if len(excursions) == 0:
        return dict(
            n_excursions=0,
            mean_excursion_time=0.0,
            max_excursion_time=0.0,
        )

    return dict(
        n_excursions=len(excursions),
        mean_excursion_time=np.mean(excursions),
        max_excursion_time=np.max(excursions),
    )


def actuator_metrics(t, u, u_min=0.0, u_max=1.0, tol=1e-6):
    """
    Measures valve usage, saturation, and movement.

    total_variation is especially useful as a valve-wear/control-effort metric:
        sum |u[k+1] - u[k]|

    integrated_absolute_rate is the continuous-time equivalent:
        integral |du/dt| dt
    """
    t = np.asarray(t)
    u = np.asarray(u)

    sat_low = u <= u_min + tol
    sat_high = u >= u_max - tol
    saturated = sat_low | sat_high

    saturation_time = np.trapz(saturated.astype(float), t)
    saturation_fraction = saturation_time / (t[-1] - t[0])

    high_saturation_time = np.trapz(sat_high.astype(float), t)
    low_saturation_time = np.trapz(sat_low.astype(float), t)

    high_saturation_fraction = high_saturation_time / (t[-1] - t[0])
    low_saturation_fraction = low_saturation_time / (t[-1] - t[0])

    du = np.diff(u)
    dt = np.diff(t)
    du_dt = du / dt

    total_variation = np.sum(np.abs(du))
    integrated_absolute_rate = np.trapz(np.abs(du_dt), t[:-1])
    rms_du_dt = np.sqrt(np.mean(du_dt**2))
    peak_du_dt = np.max(np.abs(du_dt))

    mean_u = np.mean(u)
    rms_u = np.sqrt(np.mean(u**2))

    return dict(
        mean_u=mean_u,
        rms_u=rms_u,
        min_u=np.min(u),
        max_u=np.max(u),
        total_variation=total_variation,
        integrated_absolute_rate=integrated_absolute_rate,
        rms_du_dt=rms_du_dt,
        peak_du_dt=peak_du_dt,
        saturation_time=saturation_time,
        saturation_fraction=saturation_fraction,
        high_saturation_time=high_saturation_time,
        high_saturation_fraction=high_saturation_fraction,
        low_saturation_time=low_saturation_time,
        low_saturation_fraction=low_saturation_fraction,
    )


def disturbance_rejection_metrics(t, y, reference, inlet, u, ignore_fraction=metrics_ignore_fraction):
    """
    Metrics that are useful for all disturbance cases.
    """
    t = np.asarray(t)
    y = np.asarray(y)
    inlet = np.asarray(inlet)
    u = np.asarray(u)

    idx = t >= ignore_fraction * t[-1]

    y_dev = y[idx] - reference

    peak_output_deviation = np.max(np.abs(y_dev))
    rms_output_deviation = np.sqrt(np.mean(y_dev**2))
    iae = np.trapz(np.abs(y_dev), t[idx])

    inlet_peak_to_peak = np.ptp(inlet[idx])
    output_peak_to_peak = np.ptp(y[idx])

    if inlet_peak_to_peak > 1e-9:
        attenuation_ratio = output_peak_to_peak / inlet_peak_to_peak
        attenuation_db = 20*np.log10(attenuation_ratio) if attenuation_ratio > 1e-12 else -np.inf
    else:
        attenuation_ratio = np.nan
        attenuation_db = np.nan

    deadband = deadband_error_metrics(t[idx], y[idx], reference, margin=deadband_margin)
    excursions = deadband_excursion_metrics(t[idx], y[idx], reference, margin=deadband_margin)
    actuator = actuator_metrics(t[idx], u[idx])

    return dict(
        peak_output_deviation=peak_output_deviation,
        rms_output_deviation=rms_output_deviation,
        iae=iae,
        inlet_peak_to_peak=inlet_peak_to_peak,
        output_peak_to_peak=output_peak_to_peak,
        attenuation_ratio=attenuation_ratio,
        attenuation_db=attenuation_db,

        deadband_iae=deadband["deadband_iae"],
        deadband_rms=deadband["deadband_rms"],
        deadband_peak=deadband["deadband_peak"],
        time_outside=deadband["time_outside"],
        fraction_outside=deadband["fraction_outside"],

        n_excursions=excursions["n_excursions"],
        mean_excursion_time=excursions["mean_excursion_time"],
        max_excursion_time=excursions["max_excursion_time"],

        mean_u=actuator["mean_u"],
        rms_u=actuator["rms_u"],
        min_u=actuator["min_u"],
        max_u=actuator["max_u"],
        total_variation=actuator["total_variation"],
        integrated_absolute_rate=actuator["integrated_absolute_rate"],
        rms_du_dt=actuator["rms_du_dt"],
        peak_du_dt=actuator["peak_du_dt"],
        saturation_time=actuator["saturation_time"],
        saturation_fraction=actuator["saturation_fraction"],
        high_saturation_time=actuator["high_saturation_time"],
        high_saturation_fraction=actuator["high_saturation_fraction"],
        low_saturation_time=actuator["low_saturation_time"],
        low_saturation_fraction=actuator["low_saturation_fraction"],
    )


def combined_metrics(result):
    """
    Combines cooler/heater metrics into one worst-case set.
    This is used for ranking Q/R tuning candidates.
    """
    mc = result["metrics_cooler"]
    mh = result["metrics_heater"]

    peak_output_deviation = max(mc["peak_output_deviation"], mh["peak_output_deviation"])
    rms_output_deviation = max(mc["rms_output_deviation"], mh["rms_output_deviation"])
    attenuation_db = max(mc["attenuation_db"], mh["attenuation_db"])

    deadband_iae = max(mc["deadband_iae"], mh["deadband_iae"])
    deadband_rms = max(mc["deadband_rms"], mh["deadband_rms"])
    deadband_peak = max(mc["deadband_peak"], mh["deadband_peak"])
    fraction_outside = max(mc["fraction_outside"], mh["fraction_outside"])
    max_excursion_time = max(mc["max_excursion_time"], mh["max_excursion_time"])

    saturation_fraction = max(mc["saturation_fraction"], mh["saturation_fraction"])
    high_saturation_fraction = max(mc["high_saturation_fraction"], mh["high_saturation_fraction"])

    total_variation = max(mc["total_variation"], mh["total_variation"])
    integrated_absolute_rate = max(mc["integrated_absolute_rate"], mh["integrated_absolute_rate"])
    rms_du_dt = max(mc["rms_du_dt"], mh["rms_du_dt"])
    peak_du_dt = max(mc["peak_du_dt"], mh["peak_du_dt"])

    # Score philosophy:
    #   1. stay inside the acceptable temperature band
    #   2. avoid actuator saturation
    #   3. avoid unnecessary valve motion
    #
    # The score is normalized by allowed metric values, so each term is dimensionless.
    allowed_deadband_rms = 0.05          # [°C]
    allowed_saturation_fraction = 0.05   # [-]
    allowed_total_variation = 0.5        # [-]
    allowed_rms_du_dt = 0.05             # [1/s]

    score = (
        1.0 * deadband_rms / allowed_deadband_rms
        + 1.0 * saturation_fraction / allowed_saturation_fraction
        + 1.0 * total_variation / allowed_total_variation
        + 1.0 * rms_du_dt / allowed_rms_du_dt
    )

    return dict(
        peak_output_deviation=peak_output_deviation,
        rms_output_deviation=rms_output_deviation,
        attenuation_db=attenuation_db,

        deadband_iae=deadband_iae,
        deadband_rms=deadband_rms,
        deadband_peak=deadband_peak,
        fraction_outside=fraction_outside,
        max_excursion_time=max_excursion_time,

        saturation_fraction=saturation_fraction,
        high_saturation_fraction=high_saturation_fraction,

        total_variation=total_variation,
        integrated_absolute_rate=integrated_absolute_rate,
        rms_du_dt=rms_du_dt,
        peak_du_dt=peak_du_dt,

        score=score,
    )


# ── Simulation helper ─────────────────────────────────────────────────────────
def simulate_controller(controller, name):
    augmented_state0 = np.concatenate([x0, np.zeros(controller.n_outputs)])

    sol = solve_ivp(
        controller.controller_derivatives(r=r, d=d),
        (0, t_end), augmented_state0, t_eval=t_eval,
        method="Radau", rtol=1e-6, atol=1e-8,
    )

    if not sol.success:
        raise RuntimeError(f"Simulation failed for {name}: {sol.message}")

    T_inlet = np.array([d(t)[0] for t in sol.t]) - 273.15

    T_air_cooler   = sol.y[0:K]     - 273.15
    T_water_cooler = sol.y[K:2*K]   - 273.15
    T_air_heater   = sol.y[2*K:3*K] - 273.15
    T_water_heater = sol.y[3*K:4*K] - 273.15
    x_I_hist       = sol.y[N:]

    u_hist = np.array([
        controller.compute_input(sol.y[:N, i], sol.y[N:, i], r)[0]
        for i in range(sol.y.shape[1])
    ]).T

    Kx_x_hist = np.array([
        controller.K_x @ controller.plant._to_shifted_frame(sol.y[:N, i])
        for i in range(sol.y.shape[1])
    ]).T

    KI_xI_hist = np.array([
        controller.K_I @ sol.y[N:, i]
        for i in range(sol.y.shape[1])
    ]).T

    y_cooler = sol.y[K-1]    - 273.15 #T_air_cooler.mean(axis=0)
    y_heater = sol.y[3*K-1]  - 273.15 #T_air_heater.mean(axis=0) 

    e_cooler = (T1_ref - 273.15) - y_cooler
    e_heater = (T2_ref - 273.15) - y_heater

    cooler_ref_C = T1_ref - 273.15
    heater_ref_C = T2_ref - 273.15

    metrics_cooler = disturbance_rejection_metrics(
        sol.t, y_cooler, cooler_ref_C, T_inlet, u_hist[0]
    )

    metrics_heater = disturbance_rejection_metrics(
        sol.t, y_heater, heater_ref_C, T_inlet, u_hist[1]
    )

    if TEST_CASE == "step":
        step_time_for_metrics = 20.0

        step_metrics_cooler = step_metrics(
            sol.t, y_cooler, cooler_ref_C, step_time=step_time_for_metrics, tol=0.02
        )

        step_metrics_heater = step_metrics(
            sol.t, y_heater, heater_ref_C, step_time=step_time_for_metrics, tol=0.02
        )
    else:
        step_metrics_cooler = None
        step_metrics_heater = None

    return dict(
        name=name,
        controller=controller,
        sol=sol,
        T_inlet=T_inlet,
        T_air_cooler=T_air_cooler,
        T_water_cooler=T_water_cooler,
        T_air_heater=T_air_heater,
        T_water_heater=T_water_heater,
        x_I_hist=x_I_hist,
        u_hist=u_hist,
        Kx_x_hist=Kx_x_hist,
        KI_xI_hist=KI_xI_hist,
        y_cooler=y_cooler,
        y_heater=y_heater,
        e_cooler=e_cooler,
        e_heater=e_heater,
        metrics_cooler=metrics_cooler,
        metrics_heater=metrics_heater,
        step_metrics_cooler=step_metrics_cooler,
        step_metrics_heater=step_metrics_heater,
    )


# ── Controller synthesis helpers ──────────────────────────────────────────────
def selected_controller_names(run_mode: str):
    if run_mode == "compare":
        return ["Disturbance rejection", "LQR"]
    if run_mode == "lqr_only":
        return ["LQR"]
    if run_mode == "lmi_only":
        return ["Disturbance rejection"]

    raise ValueError("RUN_MODE must be 'compare', 'lqr_only', or 'lmi_only'.")


def controller_tuning_mode(controller_name: str):
    if controller_name == "LQR":
        return LQR_TUNING_MODE
    if controller_name == "Disturbance rejection":
        return LMI_TUNING_MODE

    raise ValueError(f"Unknown controller name: {controller_name}")


def validate_tuning_mode(tuning_mode: str):
    valid_modes = ["manual", "bryson", "bryson_sweep"]
    if tuning_mode not in valid_modes:
        raise ValueError(f"Tuning mode must be one of {valid_modes}, got '{tuning_mode}'.")


def build_controller(controller_name: str, Q: np.ndarray, R: np.ndarray):
    if controller_name == "LQR":
        return StateFeedbackController.find_controller_gains(hvac, Q=Q, R=R)

    if controller_name == "Disturbance rejection":
        return StateFeedbackControllerDisturbanceRejection.find_controller_gains(hvac, Q=Q, R=R)

    raise ValueError(f"Unknown controller name: {controller_name}")


def make_cost_matrices_for_controller(
    controller_name: str,
    tuning_mode: str,
    Qx_factor: float = 1.0,
    Qi_factor: float = 1.0,
    R_factor: float = 1.0,
):
    if tuning_mode == "manual":
        if controller_name == "LQR":
            return StateFeedbackController.cost_matrices(
                hvac,
                Q_scale=LQR_MANUAL_Q_SCALE,
                R_scale=LQR_MANUAL_R_SCALE,
            )

        if controller_name == "Disturbance rejection":
            return StateFeedbackControllerDisturbanceRejection.cost_matrices(
                hvac,
                Q_scale=LMI_MANUAL_Q_SCALE,
                R_scale=LMI_MANUAL_R_SCALE,
            )

    if tuning_mode in ["bryson", "bryson_sweep"]:
        return bryson_cost_matrices(
            hvac,
            x_max_dev=x_max,
            xI_max=xI_max,
            u_max=u_max,
            Qx_factor=Qx_factor,
            Qi_factor=Qi_factor,
            R_factor=R_factor,
        )

    raise ValueError(f"Unknown tuning mode: {tuning_mode}")


def tuning_label(
    controller_name: str,
    tuning_mode: str,
    Qx_factor: float = 1.0,
    Qi_factor: float = 1.0,
    R_factor: float = 1.0,
    score: float | None = None,
):
    if tuning_mode == "manual":
        if controller_name == "LQR":
            return f"LQR manual: Qscale={LQR_MANUAL_Q_SCALE:g}, Rscale={LQR_MANUAL_R_SCALE:g}"

        if controller_name == "Disturbance rejection":
            return f"LMI manual: Qscale={LMI_MANUAL_Q_SCALE:g}, Rscale={LMI_MANUAL_R_SCALE:g}"

    if tuning_mode == "bryson":
        return (
            f"{controller_name} Bryson: "
            f"air_max={air_temp_max_error:g}, "
            f"water_max={water_temp_max_error:g}, "
            f"xI_max={xI_max[0]:g}, "
            f"u_max=[{u_max[0]:g}, {u_max[1]:g}], "
            f"factors=[1, 1, 1]"
        )

    if tuning_mode == "bryson_sweep":
        score_text = "" if score is None else f", score={score:.6g}"
        return (
            f"{controller_name} Bryson sweep: "
            f"air_max={air_temp_max_error:g}, "
            f"water_max={water_temp_max_error:g}, "
            f"xI_max={xI_max[0]:g}, "
            f"u_max=[{u_max[0]:g}, {u_max[1]:g}], "
            f"best_factors=[{Qx_factor:g}, {Qi_factor:g}, {R_factor:g}]"
            f"{score_text}"
        )

    raise ValueError(f"Unknown tuning mode: {tuning_mode}")


def run_sweep_candidate(controller_name: str, candidate_idx: int, Qx_factor: float, Qi_factor: float, R_factor: float):
    tuning_mode = "bryson_sweep"

    Qx_weights, Qi_weights, R_weights = bryson_weight_values(
        x_max_dev=x_max,
        xI_max=xI_max,
        u_max=u_max,
        Qx_factor=Qx_factor,
        Qi_factor=Qi_factor,
        R_factor=R_factor,
    )

    candidate_name = (
        f"{controller_name} sweep {candidate_idx}: "
        f"factors: Qx={Qx_factor}, Qi={Qi_factor}, R={R_factor}"
    )

    Q_sweep, R_sweep = make_cost_matrices_for_controller(
        controller_name,
        tuning_mode,
        Qx_factor=Qx_factor,
        Qi_factor=Qi_factor,
        R_factor=R_factor,
    )

    controller_sweep = build_controller(controller_name, Q_sweep, R_sweep)
    result_sweep = simulate_controller(controller_sweep, candidate_name)
    combined = combined_metrics(result_sweep)

    return dict(
        controller_name=controller_name,
        candidate_idx=candidate_idx,
        Qx_factor=Qx_factor,
        Qi_factor=Qi_factor,
        R_factor=R_factor,
        Q=Q_sweep,
        R=R_sweep,
        Qx_air_weight=Qx_weights[0],
        Qx_water_weight=Qx_weights[K],
        Qi_weight=Qi_weights[0],
        R_weight=R_weights[0],
        name=candidate_name,
        result=result_sweep,
        **combined,
    )


def run_bryson_sweep(controller_name: str):
    candidates = []
    candidate_idx = 0

    for Qx_factor in Qx_factors:
        for Qi_factor in Qi_factors:
            for R_factor in R_factors:
                candidate_idx += 1
                candidates.append((controller_name, candidate_idx, Qx_factor, Qi_factor, R_factor))

    print(f"\n=== Starting Bryson-normalized factor sweep for {controller_name} ===")
    print(f"Number of candidates: {len(candidates)}")
    print(
        f"Base Bryson limits: "
        f"air_max={air_temp_max_error:g}, "
        f"water_max={water_temp_max_error:g}, "
        f"xI_max={xI_max[0]:g}, "
        f"u_max=[{u_max[0]:g}, {u_max[1]:g}]"
    )

    base_Qx_weights, base_Qi_weights, base_R_weights = bryson_weight_values(
        x_max_dev=x_max,
        xI_max=xI_max,
        u_max=u_max,
    )

    print(
        f"Base weights: "
        f"Q_air={base_Qx_weights[0]:.6g}, "
        f"Q_water={base_Qx_weights[K]:.6g}, "
        f"Qi={base_Qi_weights[0]:.6g}, "
        f"R={base_R_weights[0]:.6g}"
    )

    sweep_records = []

    if USE_PARALLEL_SWEEP:
        print(f"Running sweep in parallel with max_workers={MAX_PARALLEL_WORKERS}")

        with ThreadPoolExecutor(max_workers=MAX_PARALLEL_WORKERS) as executor:
            future_to_candidate = {
                executor.submit(run_sweep_candidate, *candidate): candidate
                for candidate in candidates
            }

            for future in as_completed(future_to_candidate):
                candidate = future_to_candidate[future]
                _, idx, Qx_factor, Qi_factor, R_factor = candidate

                try:
                    record = future.result()
                    sweep_records.append(record)
                    print(
                        f"Finished {controller_name} candidate {idx}: "
                        f"Qx_fac={Qx_factor}, Qi_fac={Qi_factor}, R_fac={R_factor}, "
                        f"score={record['score']:.6g}"
                    )
                except Exception as exc:
                    print(
                        f"{controller_name} candidate failed: idx={idx}, "
                        f"Qx_fac={Qx_factor}, Qi_fac={Qi_factor}, R_fac={R_factor}"
                    )
                    print(f"Reason: {exc}")

    else:
        for candidate in candidates:
            _, idx, Qx_factor, Qi_factor, R_factor = candidate
            print(
                f"Synthesizing and simulating {controller_name} candidate {idx}: "
                f"Qx_fac={Qx_factor}, Qi_fac={Qi_factor}, R_fac={R_factor}"
            )

            try:
                record = run_sweep_candidate(*candidate)
                sweep_records.append(record)
            except Exception as exc:
                print(
                    f"{controller_name} candidate failed: idx={idx}, "
                    f"Qx_fac={Qx_factor}, Qi_fac={Qi_factor}, R_fac={R_factor}"
                )
                print(f"Reason: {exc}")

    if len(sweep_records) == 0:
        raise RuntimeError(f"No sweep candidates were successfully simulated for {controller_name}.")

    sweep_records_sorted = sorted(sweep_records, key=lambda item: item["score"])
    best_sweep = sweep_records_sorted[0]

    print(f"\n=== Best {controller_name} Bryson-normalized sweep candidates by score ===")
    for item in sweep_records_sorted[:10]:
        print(
            f"idx={item['candidate_idx']:>3}, "
            f"fac=[{item['Qx_factor']:g}, {item['Qi_factor']:g}, {item['R_factor']:g}], "
            f"Q_air={item['Qx_air_weight']:.6g}, "
            f"Q_water={item['Qx_water_weight']:.6g}, "
            f"Qi={item['Qi_weight']:.6g}, "
            f"R={item['R_weight']:.6g}, "
            f"score={item['score']:.6g}, "
            f"deadband_rms={item['deadband_rms']:.6g}, "
            f"frac_out={item['fraction_outside']:.6g}, "
            f"sat={item['saturation_fraction']:.6g}, "
            f"TV={item['total_variation']:.6g}, "
            f"rms_du_dt={item['rms_du_dt']:.6g}"
        )

    plot_sweep_summary(controller_name, sweep_records_sorted)

    return best_sweep, sweep_records_sorted


def plot_sweep_summary(controller_name: str, sweep_records_sorted: list[dict]):
    sweep_records_plot = sorted(sweep_records_sorted, key=lambda item: item["candidate_idx"])
    x_sweep = np.array([item["candidate_idx"] for item in sweep_records_plot])

    deadband_rms_vals = np.array([item["deadband_rms"] for item in sweep_records_plot])
    fraction_outside_vals = np.array([item["fraction_outside"] for item in sweep_records_plot])
    saturation_vals = np.array([item["saturation_fraction"] for item in sweep_records_plot])
    total_variation_vals = np.array([item["total_variation"] for item in sweep_records_plot])
    rms_du_dt_vals = np.array([item["rms_du_dt"] for item in sweep_records_plot])
    score_vals = np.array([item["score"] for item in sweep_records_plot])

    best_idx = sweep_records_sorted[0]["candidate_idx"]

    fig_sweep, ax_sweep = plt.subplots(6, 1, figsize=(12, 14), sharex=True)

    ax_sweep[0].plot(x_sweep, deadband_rms_vals, marker="o", linewidth=1.5)
    ax_sweep[0].axvline(best_idx, color="black", linestyle="--", linewidth=1)
    ax_sweep[0].set_ylabel("Deadband RMS [°C]")
    ax_sweep[0].grid(True, alpha=0.35)

    ax_sweep[1].plot(x_sweep, fraction_outside_vals, marker="o", linewidth=1.5)
    ax_sweep[1].axvline(best_idx, color="black", linestyle="--", linewidth=1)
    ax_sweep[1].set_ylabel("Fraction outside [-]")
    ax_sweep[1].grid(True, alpha=0.35)

    ax_sweep[2].plot(x_sweep, saturation_vals, marker="o", linewidth=1.5)
    ax_sweep[2].axvline(best_idx, color="black", linestyle="--", linewidth=1)
    ax_sweep[2].set_ylabel("Saturation fraction [-]")
    ax_sweep[2].grid(True, alpha=0.35)

    ax_sweep[3].plot(x_sweep, total_variation_vals, marker="o", linewidth=1.5)
    ax_sweep[3].axvline(best_idx, color="black", linestyle="--", linewidth=1)
    ax_sweep[3].set_ylabel("Total variation [-]")
    ax_sweep[3].grid(True, alpha=0.35)

    ax_sweep[4].plot(x_sweep, rms_du_dt_vals, marker="o", linewidth=1.5)
    ax_sweep[4].axvline(best_idx, color="black", linestyle="--", linewidth=1)
    ax_sweep[4].set_ylabel("RMS du/dt [1/s]")
    ax_sweep[4].grid(True, alpha=0.35)

    ax_sweep[5].plot(x_sweep, score_vals, marker="o", linewidth=1.5)
    ax_sweep[5].axvline(best_idx, color="black", linestyle="--", linewidth=1)
    ax_sweep[5].set_ylabel("Score [-]")
    ax_sweep[5].set_xlabel("Candidate index")
    ax_sweep[5].grid(True, alpha=0.35)

    fig_sweep.suptitle(
        f"Bryson-normalized Q/R factor sweep — {controller_name}\n"
        f"Deadband margin = ±{deadband_margin:.3f} °C. Dashed line marks lowest-score candidate.",
        fontweight="bold"
    )

    plt.tight_layout()


def build_and_simulate_controller(controller_name: str):
    tuning_mode = controller_tuning_mode(controller_name)
    validate_tuning_mode(tuning_mode)

    if tuning_mode in ["manual", "bryson"]:
        Q, R = make_cost_matrices_for_controller(controller_name, tuning_mode)
        controller = build_controller(controller_name, Q, R)
        label = tuning_label(controller_name, tuning_mode)
        result = simulate_controller(controller, controller_name)

        return result, label, None, Q, R

    if tuning_mode == "bryson_sweep":
        best_sweep, sweep_records_sorted = run_bryson_sweep(controller_name)
        best_result = best_sweep["result"]

        label = tuning_label(
            controller_name,
            tuning_mode,
            Qx_factor=best_sweep["Qx_factor"],
            Qi_factor=best_sweep["Qi_factor"],
            R_factor=best_sweep["R_factor"],
            score=best_sweep["score"],
        )

        # Rename the result so later plots use the clean controller name.
        best_result["name"] = controller_name

        return best_result, label, sweep_records_sorted, best_sweep["Q"], best_sweep["R"]

    raise ValueError(f"Unknown tuning mode: {tuning_mode}")


# ── Build and simulate selected controller(s) ─────────────────────────────────
controllers = {}
results = {}
controller_qr_labels = {}
sweep_records_by_controller = {}
final_QR_matrices = {}

for controller_name in selected_controller_names(RUN_MODE):
    print(f"\n\n============================================================")
    print(f"Preparing controller: {controller_name}")
    print(f"Tuning mode: {controller_tuning_mode(controller_name)}")
    print(f"============================================================")

    result, label, sweep_records, Q_used, R_used = build_and_simulate_controller(controller_name)

    controllers[controller_name] = result["controller"]
    results[controller_name] = result
    controller_qr_labels[controller_name] = label
    final_QR_matrices[controller_name] = dict(Q=Q_used, R=R_used)

    if sweep_records is not None:
        sweep_records_by_controller[controller_name] = sweep_records

qr_info_text = " | ".join(controller_qr_labels.values())


# This is the result used for the original detailed diagnostic plot.
if RUN_MODE == "compare" and "Disturbance rejection" in results:
    active_name = "Disturbance rejection"
else:
    active_name = next(iter(results.keys()))

active = results[active_name]

sol            = active["sol"]
T_inlet        = active["T_inlet"]
T_air_cooler   = active["T_air_cooler"]
T_water_cooler = active["T_water_cooler"]
T_air_heater   = active["T_air_heater"]
T_water_heater = active["T_water_heater"]
x_I_hist       = active["x_I_hist"]
u_hist         = active["u_hist"]
Kx_x_hist      = active["Kx_x_hist"]
KI_xI_hist     = active["KI_xI_hist"]
y_cooler       = active["y_cooler"]
y_heater       = active["y_heater"]
e_cooler       = active["e_cooler"]
e_heater       = active["e_heater"]

metrics_cooler = active["metrics_cooler"]
metrics_heater = active["metrics_heater"]

step_metrics_cooler = active["step_metrics_cooler"]
step_metrics_heater = active["step_metrics_heater"]

mid = K // 2

cooler_ref_C = T1_ref - 273.15
heater_ref_C = T2_ref - 273.15


# ── Terminal summary ──────────────────────────────────────────────────────────
np.set_printoptions(precision=8, suppress=False, linewidth=200)

print(f"\n=== Test case ===")
print(f"  {TEST_CASE}")
print(f"  {disturbance_label}")

print(f"\n=== Run configuration ===")
print(f"  RUN_MODE: {RUN_MODE}")
print(f"  LQR_TUNING_MODE: {LQR_TUNING_MODE}")
print(f"  LMI_TUNING_MODE: {LMI_TUNING_MODE}")
print(f"  USE_PARALLEL_SWEEP: {USE_PARALLEL_SWEEP}")
print(f"  MAX_PARALLEL_WORKERS: {MAX_PARALLEL_WORKERS}")

print(f"\n=== Controller Q/R settings ===")
for name, label in controller_qr_labels.items():
    print(f"  {name}: {label}")

print(f"\n=== Bryson limits ===")
print(f"  Air temperature max error:   {air_temp_max_error:g} K")
print(f"  Water temperature max error: {water_temp_max_error:g} K")
print(f"  Wanted settling time:        {wanted_settling_time:g} s")
print(f"  Integrator max:              {xI_max[0]:g} K·s")
print(f"  Input max:                   [{u_max[0]:g}, {u_max[1]:g}]")

base_Qx_weights, base_Qi_weights, base_R_weights = bryson_weight_values(
    x_max_dev=x_max,
    xI_max=xI_max,
    u_max=u_max,
)

print(f"\n=== Base Bryson weight values ===")
print(f"  Q air weight:   {base_Qx_weights[0]:.8g}")
print(f"  Q water weight: {base_Qx_weights[K]:.8g}")
print(f"  Qi weights:     {base_Qi_weights}")
print(f"  R weights:      {base_R_weights}")

print(f"\n=== Final Q/R matrices actually used ===")
for name, qr in final_QR_matrices.items():
    Q_used = qr["Q"]
    R_used = qr["R"]

    print(f"\n--- {name} ---")
    print(f"Q shape: {Q_used.shape}")
    print(f"R shape: {R_used.shape}")

    print("Q diagonal:")
    print(np.diag(Q_used))

    print("R matrix:")
    print(R_used)

if "Disturbance rejection" in final_QR_matrices and "LQR" in final_QR_matrices:
    Q_lmi = final_QR_matrices["Disturbance rejection"]["Q"]
    R_lmi = final_QR_matrices["Disturbance rejection"]["R"]
    Q_lqr = final_QR_matrices["LQR"]["Q"]
    R_lqr = final_QR_matrices["LQR"]["R"]

    print(f"\n=== Q/R equality check: LMI vs LQR ===")
    print(f"  Q same shape: {Q_lmi.shape == Q_lqr.shape}")
    print(f"  R same shape: {R_lmi.shape == R_lqr.shape}")
    print(f"  max |Q_LMI - Q_LQR|: {np.max(np.abs(Q_lmi - Q_lqr)):.12g}")
    print(f"  max |R_LMI - R_LQR|: {np.max(np.abs(R_lmi - R_lqr)):.12g}")
    print(f"  np.allclose(Q_LMI, Q_LQR): {np.allclose(Q_lmi, Q_lqr)}")
    print(f"  np.allclose(R_LMI, R_LQR): {np.allclose(R_lmi, R_lqr)}")

for name, result in results.items():
    print(f"\n\n============================================================")
    print(f"Controller: {name}")
    print(f"============================================================")

    T_air_cooler_i   = result["T_air_cooler"]
    T_water_cooler_i = result["T_water_cooler"]
    T_air_heater_i   = result["T_air_heater"]
    T_water_heater_i = result["T_water_heater"]
    u_hist_i         = result["u_hist"]
    x_I_hist_i       = result["x_I_hist"]

    print(f"\n=== Final temperatures ===")
    print(f"  {'Seg':<6} {'Cooler Air':>12} {'Cooler Water':>14} {'Heater Air':>12} {'Heater Water':>14}")
    print(f"  {'-'*60}")
    for k in range(K):
        print(f"  {k+1:<6} {T_air_cooler_i[k,-1]:>12.3f} {T_water_cooler_i[k,-1]:>14.3f}"
              f" {T_air_heater_i[k,-1]:>12.3f} {T_water_heater_i[k,-1]:>14.3f}")

    print(f"\n=== Final valve openings ===")
    print(f"  Cooler valve: {u_hist_i[0,-1]:.4f}")
    print(f"  Heater valve: {u_hist_i[1,-1]:.4f}")

    print(f"\n=== Final integrator states ===")
    print(f"  x_I[0] (cooler): {x_I_hist_i[0,-1]:.4f}")
    print(f"  x_I[1] (heater): {x_I_hist_i[1,-1]:.4f}")

    print(f"\n=== Disturbance rejection metrics, ignoring first {100*metrics_ignore_fraction:.1f}% of simulation ===")
    print(f"Deadband margin: ±{deadband_margin:.3f} °C")

    print(f"Cooler output:")
    for key, value in result["metrics_cooler"].items():
        print(f"  {key:<30}: {value:.6g}")

    print(f"Heater output:")
    for key, value in result["metrics_heater"].items():
        print(f"  {key:<30}: {value:.6g}")

    if TEST_CASE == "step":
        print(f"\n=== Step-like response metrics after disturbance step ===")
        print("These are measured relative to returning to the temperature reference after the inlet step.")

        print(f"Cooler output:")
        for key, value in result["step_metrics_cooler"].items():
            print(f"  {key:<30}: {value:.6g}")

        print(f"Heater output:")
        for key, value in result["step_metrics_heater"].items():
            print(f"  {key:<30}: {value:.6g}")


# ── Compact controller comparison table ───────────────────────────────────────
if RUN_MODE == "compare" and "Disturbance rejection" in results and "LQR" in results:
    print("\n\n============================================================")
    print("Compact controller comparison")
    print("============================================================")
    print(f"Deadband margin: ±{deadband_margin:.3f} °C")

    comparison_keys = [
        "deadband_rms",
        "deadband_iae",
        "fraction_outside",
        "max_excursion_time",
        "saturation_fraction",
        "high_saturation_fraction",
        "total_variation",
        "integrated_absolute_rate",
        "rms_du_dt",
        "peak_du_dt",
        "rms_output_deviation",
        "peak_output_deviation",
        "attenuation_db",
    ]

    for output_name, metric_name in [
        ("Cooler output", "metrics_cooler"),
        ("Heater output", "metrics_heater"),
    ]:
        print(f"\n--- {output_name} ---")
        print(f"{'Metric':<32} {'LMI / H-inf':>16} {'LQR':>16} {'Better':>16}")
        print("-" * 84)

        for key in comparison_keys:
            val_lmi = results["Disturbance rejection"][metric_name][key]
            val_lqr = results["LQR"][metric_name][key]

            # For attenuation_db, more negative is better because it means stronger attenuation.
            # For the other listed metrics, smaller is better.
            if key == "attenuation_db":
                if val_lmi < val_lqr:
                    better = "LMI / H-inf"
                elif val_lqr < val_lmi:
                    better = "LQR"
                else:
                    better = "Equal"
            else:
                if val_lmi < val_lqr:
                    better = "LMI / H-inf"
                elif val_lqr < val_lmi:
                    better = "LQR"
                else:
                    better = "Equal"

            print(f"{key:<32} {val_lmi:>16.6g} {val_lqr:>16.6g} {better:>16}")


# ── Comparison report plot: temperatures and valve inputs ─────────────────────
if len(results) > 1:
    fig_compare, ax_compare = plt.subplots(2, 1, figsize=(11, 6), sharex=True)

    ax_compare[0].plot(sol.t, T_inlet, linestyle=":", linewidth=2, color="black", label="Inlet air disturbance")

    for name, result in results.items():
        ax_compare[0].plot(
            result["sol"].t,
            result["T_air_cooler"].mean(axis=0),
            linewidth=2,
            label=f"Avg air — Cooler ({name})"
        )

        ax_compare[0].plot(
            result["sol"].t,
            result["T_air_heater"].mean(axis=0),
            linewidth=2,
            label=f"Avg air — Heater ({name})"
        )

    ax_compare[0].axhline(cooler_ref_C, linestyle="--", linewidth=1.5, label=f"Ref Cooler ({cooler_ref_C:.1f} °C)")
    ax_compare[0].axhline(heater_ref_C, linestyle="--", linewidth=1.5, label=f"Ref Heater ({heater_ref_C:.1f} °C)")
    ax_compare[0].set_title(f"Air Temperature — Controller Comparison\n{disturbance_label}", fontweight="bold")
    ax_compare[0].set_ylabel("Temperature [°C]", fontweight="bold")
    ax_compare[0].grid(True, alpha=0.35)
    ax_compare[0].legend(loc="best")

    for name, result in results.items():
        ax_compare[1].plot(
            result["sol"].t,
            result["u_hist"][0],
            linewidth=2,
            label=f"u_sat — Cooler ({name})"
        )

        ax_compare[1].plot(
            result["sol"].t,
            result["u_hist"][1],
            linewidth=2,
            label=f"u_sat — Heater ({name})"
        )

    ax_compare[1].set_title("Valve Inputs — Controller Comparison", fontweight="bold")
    ax_compare[1].set_xlabel("Time [s]", fontweight="bold")
    ax_compare[1].set_ylabel("Valve units [-]", fontweight="bold")
    ax_compare[1].set_ylim(-0.05, 1.05)
    ax_compare[1].grid(True, alpha=0.35)
    ax_compare[1].legend(loc="best")

    fig_compare.text(
        0.5,
        0.01,
        qr_info_text,
        ha="center",
        fontsize=9
    )

    plt.tight_layout(rect=[0, 0.04, 1, 1])


# ── Report plot: active controller only ────────────────────────────────────────
fig_report, ax_report = plt.subplots(2, 1, figsize=(11, 6), sharex=True)

ax_report[0].plot(sol.t, T_air_cooler.mean(axis=0), linewidth=2, label=f"Avg air — Cooler ({active_name})")
ax_report[0].plot(sol.t, T_air_heater.mean(axis=0), linewidth=2, label=f"Avg air — Heater ({active_name})")
ax_report[0].plot(sol.t, T_inlet, linestyle=":", linewidth=2, color="black", label="Inlet air disturbance")
ax_report[0].axhline(cooler_ref_C, linestyle="--", linewidth=1.5, label=f"Ref Cooler ({cooler_ref_C:.1f} °C)")
ax_report[0].axhline(heater_ref_C, linestyle="--", linewidth=1.5, label=f"Ref Heater ({heater_ref_C:.1f} °C)")
ax_report[0].set_title(f"Air Temperature — Cooler -> Heater\n{disturbance_label}", fontweight="bold")
ax_report[0].set_ylabel("Temperature [°C]", fontweight="bold")
ax_report[0].grid(True, alpha=0.35)
ax_report[0].legend(loc="best")

ax_report[1].plot(sol.t, u_hist[0], linewidth=2, label=f"u_sat — Cooler ({active_name})")
ax_report[1].plot(sol.t, u_hist[1], linewidth=2, label=f"u_sat — Heater ({active_name})")
ax_report[1].set_title("Valve Inputs", fontweight="bold")
ax_report[1].set_xlabel("Time [s]", fontweight="bold")
ax_report[1].set_ylabel("Valve units [-]", fontweight="bold")
ax_report[1].set_ylim(-0.05, 1.05)
ax_report[1].grid(True, alpha=0.35)
ax_report[1].legend(loc="best")

fig_report.text(
    0.5,
    0.01,
    controller_qr_labels[active_name],
    ha="center",
    fontsize=9
)

plt.tight_layout(rect=[0, 0.04, 1, 1])


# ── Diagnostic plot ───────────────────────────────────────────────────────────
fig, axes = plt.subplots(6, 2, figsize=(14, 24), sharex=True)

axes[0, 0].plot(sol.t, T_air_cooler.mean(axis=0), color="tomato", linewidth=2, label="Avg air")
axes[0, 0].plot(sol.t, T_inlet, color="black", linewidth=1.5, linestyle=":", label="Inlet (actual)")
axes[0, 0].axhline(cooler_ref_C,  color="green", linestyle="--", label=f"Ref ({cooler_ref_C:.1f} °C)")
axes[0, 0].set_title("Cooler — Air Temperature")
axes[0, 0].set_ylabel("Temperature [°C]")
axes[0, 0].legend(fontsize=8)
axes[0, 0].grid(True, alpha=0.35)

axes[1, 0].plot(sol.t, T_water_cooler[mid], color="steelblue", linewidth=2, label=f"Seg {mid+1}")
axes[1, 0].set_title("Cooler — Water Temperature")
axes[1, 0].set_ylabel("Temperature [°C]")
axes[1, 0].legend(fontsize=8)
axes[1, 0].grid(True, alpha=0.35)

axes[0, 1].plot(sol.t, T_air_heater.mean(axis=0), color="tomato", linewidth=2, label="Avg air")
axes[0, 1].axhline(heater_ref_C, color="green", linestyle="--", label=f"Ref ({heater_ref_C:.1f} °C)")
axes[0, 1].set_title("Heater — Air Temperature")
axes[0, 1].set_ylabel("Temperature [°C]")
axes[0, 1].legend(fontsize=8)
axes[0, 1].grid(True, alpha=0.35)

axes[1, 1].plot(sol.t, T_water_heater[mid], color="steelblue", linewidth=2, label=f"Seg {mid+1}")
axes[1, 1].set_title("Heater — Water Temperature")
axes[1, 1].set_ylabel("Temperature [°C]")
axes[1, 1].legend(fontsize=8)
axes[1, 1].grid(True, alpha=0.35)

axes[2, 0].plot(sol.t, x_I_hist[0], color="purple", linewidth=2)
axes[2, 0].set_title("Integrator State — Cooler")
axes[2, 0].set_ylabel("x_I [K·s]")
axes[2, 0].set_xlabel("Time [s]")
axes[2, 0].grid(True, alpha=0.35)

axes[2, 1].plot(sol.t, x_I_hist[1], color="purple", linewidth=2)
axes[2, 1].set_title("Integrator State — Heater")
axes[2, 1].set_ylabel("x_I [K·s]")
axes[2, 1].set_xlabel("Time [s]")
axes[2, 1].grid(True, alpha=0.35)

for col, label in enumerate(["Cooler", "Heater"]):
    axes[3, col].plot(sol.t, Kx_x_hist[col], color="teal", linewidth=2)
    axes[3, col].set_title(f"Kx·x — {label}")
    axes[3, col].set_ylabel("Kx·x [valve units]")
    axes[3, col].grid(True, alpha=0.35)

for col, label in enumerate(["Cooler", "Heater"]):
    axes[4, col].plot(sol.t, KI_xI_hist[col], color="mediumorchid", linewidth=2)
    axes[4, col].set_title(f"KI·xI — {label}")
    axes[4, col].set_ylabel("KI·xI [valve units]")
    axes[4, col].set_xlabel("Time [s]")
    axes[4, col].grid(True, alpha=0.35)

axes[5, 0].plot(sol.t, u_hist[0], color="darkorange", linewidth=2)
axes[5, 0].set_title("Valve Opening — Cooler")
axes[5, 0].set_ylabel("Opening [-]")
axes[5, 0].set_ylim(-0.05, 1.05)
axes[5, 0].set_xlabel("Time [s]")
axes[5, 0].grid(True, alpha=0.35)

axes[5, 1].plot(sol.t, u_hist[1], color="darkorange", linewidth=2)
axes[5, 1].set_title("Valve Opening — Heater")
axes[5, 1].set_ylabel("Opening [-]")
axes[5, 1].set_ylim(-0.05, 1.05)
axes[5, 1].set_xlabel("Time [s]")
axes[5, 1].grid(True, alpha=0.35)

plt.suptitle(f"HVAC cascade ({model_mode}): Cooler → Heater — {active_name}\n{controller_qr_labels[active_name]}", fontsize=13)
plt.tight_layout()
plt.show()