import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from pathlib import Path
from models import HVAC
from controller import StateFeedbackControllerDisturbanceRejection, StateFeedbackController


# ── Parameters ────────────────────────────────────────────────────────────────
params_cooler = dict(
    type                  = "cooler",
    num_segments          = 5,
    num_pipes             = 10,
    gamma                 = 951.87,
    cross_area_water      = 0.000201,
    heat_exchanger_depth  = 0.06,
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

model_mode = "nonlinear"  # "linear" or "nonlinear"

# If True, both controllers are simulated using the exact same disturbance.
# If False, only the controller selected by use_lqr is simulated.
compare_controllers = True

use_lqr = True  # Used only when compare_controllers = False


# ── Instantiate plant ─────────────────────────────────────────────────────────
#hvac = HVAC(configs=[params_cooler, params_heater], mode=model_mode, const_disturbance=28 + 273.15)
hvac = HVAC(configs=[params_cooler, params_heater], mode=model_mode, const_disturbance=None)


# ── Export state-space model ──────────────────────────────────────────────────
data_dir = Path(__file__).resolve().parent.parent / "models/linear"
data_dir.mkdir(parents=True, exist_ok=True)
hvac._export_state_space(data_dir / "HVAC_model.mat")


# ── Instantiate controller(s) ─────────────────────────────────────────────────
controllers = {}
Q_test_scale = 1000000
R_test_scale = 1000

if compare_controllers:
    Q_dr, R_dr = StateFeedbackControllerDisturbanceRejection.cost_matrices(hvac, Q_scale=Q_test_scale, R_scale=R_test_scale)
    controllers["Disturbance rejection"] = StateFeedbackControllerDisturbanceRejection.find_controller_gains(
        hvac, Q=Q_dr, R=R_dr
    )

    Q_lqr, R_lqr = StateFeedbackController.cost_matrices(hvac, Q_scale=5, R_scale=10)
    controllers["LQR"] = StateFeedbackController.find_controller_gains(
        hvac, Q=Q_lqr, R=R_lqr
    )

else:
    if not use_lqr:
        Q, R = StateFeedbackControllerDisturbanceRejection.cost_matrices(hvac, Q_scale=Q_test_scale, R_scale=R_test_scale)
        controllers["Disturbance rejection"] = StateFeedbackControllerDisturbanceRejection.find_controller_gains(
            hvac, Q=Q, R=R
        )
    else:
        Q, R = StateFeedbackController.cost_matrices(hvac, Q_scale=5, R_scale=10)
        controllers["LQR"] = StateFeedbackController.find_controller_gains(
            hvac, Q=Q, R=R
        )


# ── Dimensions ────────────────────────────────────────────────────────────────
K = hvac._lin_components[0].K
N = hvac.total_states   # 4K = 20


# ── Initial conditions ────────────────────────────────────────────────────────
x0 = np.concatenate([
    np.full(K, 23 + 273.15),          # cooler air
    np.full(K, 23 + 273.15),          # cooler water
    np.full(K, 9.9 + 273.15),         # heater air
    np.full(K, 9.9 + 273.15),         # heater water
])


# ── References ────────────────────────────────────────────────────────────────
T1_ref = 10.0 + 273.15   # Cooler air outlet setpoint [K]
T2_ref = 20.0 + 273.15   # Heater air outlet setpoint [K]
r      = np.array([T1_ref, T2_ref])

T_in = 23 + 273.15       # Nominal air inlet to cooler [K]
t_day = 24 * 3600


# ── Test case selector ────────────────────────────────────────────────────────
# Options: "sinusoid", "step", "stochastic", "constant"
TEST_CASE = "stochastic"


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

    elif test_case == "step":
        step_amp = 5.0        # [°C]
        step_time = 20.0      # [s]

        def d(t):
            T_in_sys = T_in + (step_amp if t >= step_time else 0.0)
            return np.array([T_in_sys])

        t_end = 50
        n_eval = 200
        label = f"Step inlet disturbance, ΔT={step_amp:.1f} °C at t={step_time:.0f} s"
        return d, t_end, n_eval, label

    elif test_case == "stochastic":
        sigma = 3.0           # [°C], random inlet fluctuation size
        sample_time = 30.0    # [s], new random value every sample_time seconds
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

        t_end = 1000
        n_eval = 300
        label = "Constant inlet temperature"
        return d, t_end, n_eval, label

    else:
        raise ValueError(
            f"Unknown TEST_CASE='{test_case}'. Use 'sinusoid', 'step', 'stochastic', or 'constant'."
        )


d, t_end, n_eval, disturbance_label = make_disturbance(TEST_CASE)
t_eval = np.linspace(0, t_end, n_eval)


# ── Performance metrics ───────────────────────────────────────────────────────
def step_metrics(t, y, target, step_time=0.0, tol=0.02):
    """
    Basic step-response-like metrics.

    For normal reference tracking:
        target = desired setpoint.

    For disturbance rejection:
        target = the value the output should return to after the disturbance.
    """
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

    # Overshoot relative to final value
    if abs(delta) > 1e-9:
        if delta > 0:
            overshoot = max(0.0, np.max(y2) - y_final) / abs(delta) * 100.0
        else:
            overshoot = max(0.0, y_final - np.min(y2)) / abs(delta) * 100.0
    else:
        overshoot = np.nan

    # Rise time: 10% to 90% of the movement toward final value
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

    # Settling time: time after which signal stays within tol band around target
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


def disturbance_rejection_metrics(t, y, reference, inlet, u, ignore_fraction=0.1):
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

    du_dt = np.diff(u) / np.diff(t)
    peak_du_dt = np.max(np.abs(du_dt))

    saturation_fraction = np.mean((u <= 1e-6) | (u >= 1.0 - 1e-6))

    return dict(
        peak_output_deviation=peak_output_deviation,
        rms_output_deviation=rms_output_deviation,
        iae=iae,
        inlet_peak_to_peak=inlet_peak_to_peak,
        output_peak_to_peak=output_peak_to_peak,
        attenuation_ratio=attenuation_ratio,
        attenuation_db=attenuation_db,
        peak_du_dt=peak_du_dt,
        saturation_fraction=saturation_fraction,
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
    x_I_hist       = sol.y[N:]      # shape (n_outputs, n_timesteps)

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

    y_cooler = sol.y[K-1]    - 273.15   # cooler air outlet, last segment
    y_heater = sol.y[3*K-1]  - 273.15   # heater air outlet, last segment

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


# ── Simulate selected controller(s) ───────────────────────────────────────────
results = {}

for name, controller in controllers.items():
    print(f"Simulating controller: {name}")
    results[name] = simulate_controller(controller, name)

# This is the result used for the original detailed diagnostic plot.
if compare_controllers:
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


# ── Comparison report plot: temperatures and valve inputs ─────────────────────
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

plt.tight_layout()


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

plt.tight_layout()


# ── Plot ──────────────────────────────────────────────────────────────────────
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

# Row 3 — Kx·x contribution
for col, label in enumerate(["Cooler", "Heater"]):
    axes[3, col].plot(sol.t, Kx_x_hist[col], color="teal", linewidth=2)
    axes[3, col].set_title(f"Kx·x — {label}")
    axes[3, col].set_ylabel("Kx·x [valve units]")
    axes[3, col].grid(True, alpha=0.35)

# Row 4 — KI·xI contribution
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

plt.suptitle(f"HVAC cascade ({model_mode}): Cooler → Heater — {active_name}", fontsize=13)
plt.tight_layout()
plt.show()


# ── Terminal summary ──────────────────────────────────────────────────────────
print(f"\n=== Test case ===")
print(f"  {TEST_CASE}")
print(f"  {disturbance_label}")

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

    print(f"\n=== Disturbance rejection metrics, ignoring first 10% of simulation ===")
    print(f"Cooler output:")
    for key, value in result["metrics_cooler"].items():
        print(f"  {key:<24}: {value:.6g}")

    print(f"Heater output:")
    for key, value in result["metrics_heater"].items():
        print(f"  {key:<24}: {value:.6g}")

    if TEST_CASE == "step":
        print(f"\n=== Step-like response metrics after disturbance step ===")
        print("These are measured relative to returning to the temperature reference after the inlet step.")

        print(f"Cooler output:")
        for key, value in result["step_metrics_cooler"].items():
            print(f"  {key:<24}: {value:.6g}")

        print(f"Heater output:")
        for key, value in result["step_metrics_heater"].items():
            print(f"  {key:<24}: {value:.6g}")