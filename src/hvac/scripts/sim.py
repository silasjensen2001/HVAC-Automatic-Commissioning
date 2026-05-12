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

model_mode = "nonlinear"  # "linear" or "nonlinear"

# ── Instantiate plant ─────────────────────────────────────────────────────────
hvac = HVAC(configs=[params_cooler, params_heater], mode=model_mode, const_disturbance=None)

# ── Export state-space model ──────────────────────────────────────────────────
data_dir = Path(__file__).resolve().parent.parent / "models/linear"
data_dir.mkdir(parents=True, exist_ok=True)
hvac._export_state_space(data_dir / "HVAC_model.mat")

# ── Toggles ───────────────────────────────────────────────────────────────────
COMPARE_CONTROLLERS    = True  # True: simulate both and plot side by side
USE_DISTURBANCE_REJECTION = True   # Used when COMPARE_CONTROLLERS = False
USE_BRYSON             = False  # Only used for LQR (not compatible with LMI design)
CASE_DISTURBANCE       = 4      # 1: Temp, 2: RH, 3: Flow, 4: All combined

# ── Bryson bounds (physical / absolute frame) ─────────────────────────────────
x_max  = np.full(20, 20 + 273.15)
u_max  = np.array([0.5, 0.5])
xI_max = np.array([10.0, 10.0])

# ── Dimensions ────────────────────────────────────────────────────────────────
K = hvac._lin_components[0].K
N = hvac.total_states   # 4K = 20

# ── Time ──────────────────────────────────────────────────────────────────────
t_day          = 24 * 3600
points_per_day = t_day * 3
t_end          = t_day * 2
t_eval         = np.linspace(0, t_end, points_per_day)

# ── Initial conditions ────────────────────────────────────────────────────────
x0 = np.concatenate([
    np.full(K, 23 + 273.15),   # cooler air
    np.full(K, 23 + 273.15),   # cooler water
    np.full(K, 9.9 + 273.15),  # heater air
    np.full(K, 9.9 + 273.15),  # heater water
])

# ── References ────────────────────────────────────────────────────────────────
T1_ref = 10.0 + 273.15
T2_ref = 20.0 + 273.15
r      = np.array([T1_ref, T2_ref])

# ── Disturbance function ──────────────────────────────────────────────────────
def d(t):
    T_day = 24 * 3600
    match CASE_DISTURBANCE:
        case 1:  # Temperature disturbance only
            T_in = 23 + 273.15 + 5 * np.cos(2 * np.pi * t / t_day)
            rh_in = 0.832
            volume_flow_wet_air = (params_cooler["volume_flow_wet_air"]
                                   / (params_cooler["num_segments"] * params_cooler["num_pipes"]))

        case 2:  # RH disturbance only
            T_in = 23 + 273.15
            rh_in = 0.75 + 0.25 * np.sin(2 * np.pi * t / T_day + np.pi / 3)
            rh_in = np.clip(rh_in, 0.0, 1.0)
            volume_flow_wet_air = (params_cooler["volume_flow_wet_air"]
                                   / (params_cooler["num_segments"] * params_cooler["num_pipes"]))

        case 3:  # Flow disturbance only
            T_in = 23 + 273.15
            rh_in = 0.832
            flow_base = hvac._lin_components[0].volume_flow_wet_air
            volume_flow_wet_air = flow_base + 0.5 * flow_base * np.cos(2 * np.pi * t / (t_day / 2))

        case 4:  # All disturbances combined
            shift_t = t - 54000
            T_in = (23
                    + 6   * np.cos(2 * np.pi * shift_t / 86400)
                    + 1.6 * np.cos(2 * np.pi * shift_t / 43200)
                    + 0.5 * np.cos(2 * np.pi * shift_t / 28800)
                    + 2   * np.sin(2 * np.pi * t / 259200)
                    + 273.15)
            rh_in = 0.75 + 0.25 * np.sin(2 * np.pi * t / T_day + 2 * np.pi / 3)
            rh_in = np.clip(rh_in, 0.0, 1.0)
            flow_base = hvac._lin_components[0].volume_flow_wet_air
            volume_flow_wet_air = flow_base + 0.5 * flow_base * np.cos(2 * np.pi * t / (t_day / 2))

    return np.array([T_in, rh_in, volume_flow_wet_air])

# ── Build controller and simulate ─────────────────────────────────────────────
def build_and_simulate(use_disturbance_rejection):
    ControllerCls = (
        StateFeedbackControllerDisturbanceRejection if use_disturbance_rejection
        else StateFeedbackController
    )
    if use_disturbance_rejection:
        Q, R = ControllerCls.cost_matrices(hvac, Q_scale=100.0, R_scale=3.0)
    else:
        if USE_BRYSON:
            Q, R = ControllerCls.cost_bryson(hvac, x_max=x_max, u_max=u_max, x_I_max=xI_max)
        else:
            Q, R = ControllerCls.cost_matrices(hvac, Q_scale=10.0, R_scale=800.0)

    ctrl = ControllerCls.find_controller_gains(hvac, Q=Q, R=R)
    aug0 = np.concatenate([x0, np.zeros(ctrl.n_outputs)])
    sol  = solve_ivp(
        ctrl.controller_derivatives(r=r, d=d),
        (0, t_end), aug0, t_eval=t_eval,
        method="Radau", rtol=1e-6, atol=1e-8,
    )
    return ctrl, sol

if COMPARE_CONTROLLERS:
    ctrl_dr,  sol_dr  = build_and_simulate(use_disturbance_rejection=True)
    ctrl_lqr, sol_lqr = build_and_simulate(use_disturbance_rejection=False)
    solutions = [
        (sol_dr,  ctrl_dr,  "LMI disturbance rejection"),
        (sol_lqr, ctrl_lqr, "LQR"),
    ]
else:
    ctrl, sol = build_and_simulate(use_disturbance_rejection=USE_DISTURBANCE_REJECTION)
    label     = "LMI disturbance rejection" if USE_DISTURBANCE_REJECTION else "LQR"
    solutions = [(sol, ctrl, label)]

# ── Unpack helper ─────────────────────────────────────────────────────────────
def unpack(sol, ctrl):
    T_inlet      = np.array([d(t)[0] for t in sol.t]) - 273.15
    T_air_cooler = sol.y[0:K]     - 273.15
    T_air_heater = sol.y[2*K:3*K] - 273.15
    u_hist       = np.array([
        ctrl.compute_input(sol.y[:N, i], sol.y[N:, i], r)[0]
        for i in range(sol.y.shape[1])
    ]).T
    return T_inlet, T_air_cooler, T_air_heater, u_hist

# ── Disturbance signals (same for all controllers) ────────────────────────────
t_ref       = solutions[0][0].t
RH_in       = np.array([d(t)[1] for t in t_ref])
V_flow_hist = np.array([d(t)[2] for t in t_ref])

# ── Plot setup ────────────────────────────────────────────────────────────────
case_titles = {
    1: "Temperature Disturbance Only",
    2: "Relative Humidity Disturbance Only",
    3: "Volumetric Flow Disturbance Only",
    4: "All Disturbances Combined",
}

ctrl_colors = {
    "LMI disturbance rejection": ("tomato",    "firebrick"),
    "LQR":                        ("royalblue", "navy"),
}

if COMPARE_CONTROLLERS:
    fig, axes  = plt.subplots(2, 2, figsize=(16, 9), sharex=True, sharey="row")
    temp_axes  = [axes[0, 0], axes[0, 1]]
    valve_axes = [axes[1, 0], axes[1, 1]]
else:
    fig, axes  = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    temp_axes  = [axes[0]]
    valve_axes = [axes[1]]

for idx, (sol_i, ctrl_i, lbl) in enumerate(solutions):
    T_inlet, T_air_cooler, T_air_heater, u_hist = unpack(sol_i, ctrl_i)
    c_cooler, c_heater = ctrl_colors[lbl]

    # ── Temperature subplot ───────────────────────────────────────────────────
    ax_temp = temp_axes[idx]

    ax_temp.plot(sol_i.t, T_air_cooler.mean(axis=0), color=c_cooler,  linewidth=2,   label="Cooler avg air")
    ax_temp.plot(sol_i.t, T_air_heater.mean(axis=0), color=c_heater,  linewidth=2,   label="Heater avg air")
    ax_temp.axhline(T1_ref - 273.15, color="green",     linestyle="--", label=f"Cooler ref ({T1_ref-273.15:.1f} °C)")
    ax_temp.axhline(T2_ref - 273.15, color="limegreen", linestyle="--", label=f"Heater ref ({T2_ref-273.15:.1f} °C)")
    ax_temp.set_ylabel("Temperature [°C]")
    ax_temp.set_title(f"{lbl} — {case_titles[CASE_DISTURBANCE]}")
    ax_temp.grid(True, alpha=0.35)

    if CASE_DISTURBANCE in (1, 4):
        ax_temp.plot(sol_i.t, T_inlet, color="black", linewidth=1.5, linestyle=":", label="Inlet (actual)")

    lines_r, labels_r = [], []
    if CASE_DISTURBANCE in (2, 4):
        ax_rh = ax_temp.twinx()
        ax_rh.plot(sol_i.t, RH_in, color="steelblue", linewidth=1.5, linestyle="-.", alpha=0.8, label="RH in")
        ax_rh.set_ylabel("RH [-]", color="steelblue")
        ax_rh.set_ylim(-0.05, 1.05)
        ax_rh.tick_params(axis="y", labelcolor="steelblue")
        lines_r, labels_r = ax_rh.get_legend_handles_labels()

    lines_f, labels_f = [], []
    if CASE_DISTURBANCE in (3, 4):
        ax_flow = ax_temp.twinx()
        if CASE_DISTURBANCE == 4:
            ax_flow.spines["right"].set_position(("axes", 1.12))
            ax_flow.spines["right"].set_visible(True)
        ax_flow.plot(sol_i.t, V_flow_hist, color="mediumpurple", linewidth=1.5, linestyle="--", alpha=0.8, label="Vol. flow")
        ax_flow.set_ylabel("Vol. flow [m³/s]", color="mediumpurple")
        ax_flow.tick_params(axis="y", labelcolor="mediumpurple")
        v_min, v_max = V_flow_hist.min(), V_flow_hist.max()
        v_range = v_max - v_min
        ax_flow.set_ylim(v_min - 2 * v_range, v_max + 0.1 * v_range)
        lines_f, labels_f = ax_flow.get_legend_handles_labels()

    lines_t, labels_t = ax_temp.get_legend_handles_labels()
    ax_temp.legend(lines_t + lines_r + lines_f, labels_t + labels_r + labels_f, fontsize=8)

    # ── Valve subplot ─────────────────────────────────────────────────────────
    ax_v = valve_axes[idx]
    ax_v.plot(sol_i.t, u_hist[0], color=c_cooler, linewidth=2, label="Cooler")
    ax_v.plot(sol_i.t, u_hist[1], color=c_heater,  linewidth=2, label="Heater")
    ax_v.set_title(f"Valve Openings — {lbl}")
    ax_v.set_ylabel("Opening [-]")
    ax_v.set_ylim(-0.05, 1.05)
    ax_v.set_xlabel("Time [s]")
    ax_v.legend(fontsize=8)
    ax_v.grid(True, alpha=0.35)

plt.suptitle(
    f"HVAC ({model_mode}): Cooler → Heater — Case {CASE_DISTURBANCE}"
    + (" — Controller Comparison" if COMPARE_CONTROLLERS else ""),
    fontsize=13,
)
fig.subplots_adjust(right=0.83 if CASE_DISTURBANCE == 4 else 0.93)
plt.tight_layout()
plt.show()