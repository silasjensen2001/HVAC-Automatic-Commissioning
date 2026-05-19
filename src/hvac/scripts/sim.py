import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
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
COMPARE_CONTROLLERS       = True    # True: overlay both in one 2×1 layout
USE_DISTURBANCE_REJECTION = False   # Used when COMPARE_CONTROLLERS = False
USE_BRYSON                = True    # Only used for LQR (not compatible with LMI design)
CASE_DISTURBANCE          = 5       # 0: All constant, 1: Temp, 2: RH, 3: Flow, 4: All combined, 5: Step change in T_in

# ── Dimensions ────────────────────────────────────────────────────────────────
K = hvac._lin_components[0].K
N = hvac.total_states

# ── Bryson bounds ─────────────────────────────────────────────────────────────
air_temp_max_error = 1.0    # [K]
water_temp_max_error = 50.0 # [K]
wanted_settling_time = 2.0 # [s]

# Air states (first K and K+1:2K): air_temp_max_error
# Water states (2K:3K and 3K:4K): water_temp_max_error
x_max = np.concatenate([
    np.full(K, air_temp_max_error),
    np.full(K, water_temp_max_error),
    np.full(K, air_temp_max_error),
    np.full(K, water_temp_max_error),
])

# Integral states: air_temp_max_error * wanted_settling_time
xI_max = np.full(2, air_temp_max_error * wanted_settling_time)

u_max  = np.array([1.0, 1.0])

# ── Time ──────────────────────────────────────────────────────────────────────
t_day          = 24 * 3600
points_per_day = 3000
t_end          = 100
t_start = 58 if CASE_DISTURBANCE == 5 else 0
t_eval  = np.linspace(t_start, t_end, points_per_day)

# ── Initial conditions ────────────────────────────────────────────────────────
x0 = np.concatenate([
    np.full(K, 15 + 273.15),
    np.full(K, 15 + 273.15),
    np.full(K, 9.9 + 273.15),
    np.full(K, 9.9 + 273.15),
])

# ── References ────────────────────────────────────────────────────────────────
T1_ref = 10.0 + 273.15
T2_ref = 20.0 + 273.15
r      = np.array([T1_ref, T2_ref])

# ── Disturbance function ──────────────────────────────────────────────────────
def d(t):
    T_day = 24 * 3600
    match CASE_DISTURBANCE:
        case 0:
            T_in  = 23 + 273.15
            rh_in = 0.832
            volume_flow_wet_air = (params_cooler["volume_flow_wet_air"]
                                   / (params_cooler["num_segments"] * params_cooler["num_pipes"]))
        case 1:
            shift_t = t - 54000
            T_in = (23
                    + 6   * np.cos(2 * np.pi * shift_t / 86400)
                    + 1.6 * np.cos(2 * np.pi * shift_t / 43200)
                    + 0.5 * np.cos(2 * np.pi * shift_t / 28800)
                    + 2   * np.sin(2 * np.pi * t / 259200)
                    + 273.15)
            rh_in = 0.832
            volume_flow_wet_air = (params_cooler["volume_flow_wet_air"]
                                   / (params_cooler["num_segments"] * params_cooler["num_pipes"]))
        case 2:
            T_in  = 23 + 273.15
            rh_in = np.clip(0.70 + 0.25 * np.sin(2 * np.pi * t / T_day + np.pi / 3), 0.0, 1.0)
            volume_flow_wet_air = (params_cooler["volume_flow_wet_air"]
                                   / (params_cooler["num_segments"] * params_cooler["num_pipes"]))
        case 3:
            T_in  = 23 + 273.15
            rh_in = 0.832
            flow_base           = hvac._lin_components[0].volume_flow_wet_air
            volume_flow_wet_air = flow_base + 0.5 * flow_base * np.cos(2 * np.pi * t / (t_day / 2))
        case 4:
            shift_t = t - 54000
            T_in = (23
                    + 6   * np.cos(2 * np.pi * shift_t / 86400)
                    + 1.6 * np.cos(2 * np.pi * shift_t / 43200)
                    + 0.5 * np.cos(2 * np.pi * shift_t / 28800)
                    + 2   * np.sin(2 * np.pi * t / 259200)
                    + 273.15)
            rh_in               = np.clip(0.70 + 0.25 * np.sin(2 * np.pi * t / T_day + 2 * np.pi / 3), 0.0, 1.0)
            flow_base           = hvac._lin_components[0].volume_flow_wet_air
            volume_flow_wet_air = flow_base + 0.5 * flow_base * np.cos(2 * np.pi * t / (t_day / 2))
        case 5:
            # All constant, but T_in steps to 28°C after 60 seconds
            T_in = 15 + 273.15 if t < 60 else 23 + 273.15
            rh_in = 0.832
            volume_flow_wet_air = (
                params_cooler["volume_flow_wet_air"]
                / (params_cooler["num_segments"] * params_cooler["num_pipes"])
            )
    return np.array([T_in, rh_in, volume_flow_wet_air])

# ── Build controller and simulate ─────────────────────────────────────────────
def build_and_simulate(use_disturbance_rejection):
    ControllerCls = (
        StateFeedbackControllerDisturbanceRejection if use_disturbance_rejection
        else StateFeedbackController
    )
    if use_disturbance_rejection:
        Q, R = ControllerCls.cost_matrices(hvac, Q_scale=10.0, Qi_scale=5.0, R_scale=5.0, use_disturbance_rejection=True)
    else:
        Q, R = (ControllerCls.cost_bryson(hvac, x_max=x_max, u_max=u_max, x_I_max=xI_max, shifted=True)
                if USE_BRYSON else
                ControllerCls.cost_matrices(hvac, Q_scale=10000.0, R_scale=8.0))
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
        (sol_dr,  ctrl_dr,  "LMI"),
        (sol_lqr, ctrl_lqr, "LQR"),
    ]
else:
    ctrl, sol = build_and_simulate(use_disturbance_rejection=USE_DISTURBANCE_REJECTION)
    solutions  = [(sol, ctrl, "LMI" if USE_DISTURBANCE_REJECTION else "LQR")]

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

# ── Disturbance signals ───────────────────────────────────────────────────────
t_ref       = solutions[0][0].t
RH_in       = np.array([d(t)[1] for t in t_ref])
V_flow_hist = np.array([d(t)[2] for t in t_ref])

# ── Viridis colour scheme ─────────────────────────────────────────────────────
_vir = plt.cm.viridis
ctrl_colors = {
    "LMI": (_vir(0.15), _vir(0.45)),   # (cooler, heater)
    "LQR": (_vir(0.65), _vir(0.90)),
}

case_titles = {
    0: "Constant Disturbances",
    1: "Temperature Disturbance Only",
    2: "Relative Humidity Disturbance Only",
    3: "Volumetric Flow Disturbance Only",
    4: "All Disturbances Combined",
    5: "Step Change in T_in at t=60s",
}

# ── Figure — always 2×1 ───────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True)
ax_temp  = axes[0]
ax_valve = axes[1]

# ── Shared disturbance overlays (drawn once) ──────────────────────────────────
lines_r, labels_r = [], []
lines_f, labels_f = [], []

if CASE_DISTURBANCE in (2, 4):
    ax_rh = ax_temp.twinx()
    ax_rh.plot(t_ref, RH_in, color="steelblue", linewidth=1.5,
               linestyle="-.", alpha=0.7, label="RH in")
    ax_rh.set_ylabel("RH [-]", color="steelblue")
    ax_rh.set_ylim(-0.05, 1.05)
    ax_rh.tick_params(axis="y", labelcolor="steelblue")
    lines_r, labels_r = ax_rh.get_legend_handles_labels()

if CASE_DISTURBANCE in (3, 4):
    ax_flow = ax_temp.twinx()
    if CASE_DISTURBANCE == 4:
        ax_flow.spines["right"].set_position(("axes", 1.08))  # moved closer
        ax_flow.spines["right"].set_visible(True)
    ax_flow.plot(t_ref, V_flow_hist, color="mediumpurple", linewidth=1.5,
                 linestyle="--", alpha=0.7, label="Air flow")  # renamed
    ax_flow.set_ylabel("Air flow [m³/s]", color="mediumpurple")
    ax_flow.tick_params(axis="y", labelcolor="mediumpurple")
    v_min, v_max = V_flow_hist.min(), V_flow_hist.max()
    v_range = v_max - v_min
    ax_flow.set_ylim(v_min - 3 * v_range, v_max + 0.1 * v_range)
    ax_flow.yaxis.set_major_locator(ticker.LinearLocator(numticks=4))
    ax_flow.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.4f"))
    lines_f, labels_f = ax_flow.get_legend_handles_labels()

# ── Per-controller traces ─────────────────────────────────────────────────────
for sol_i, ctrl_i, lbl in solutions:
    T_inlet, T_air_cooler, T_air_heater, u_hist = unpack(sol_i, ctrl_i)
    c_cooler, c_heater = ctrl_colors[lbl]

    ax_temp.plot(sol_i.t, T_air_cooler.mean(axis=0), color=c_cooler,
                 linewidth=2, label=f"Cooler air ({lbl})")
    ax_temp.plot(sol_i.t, T_air_heater.mean(axis=0), color=c_heater,
                 linewidth=2, label=f"Heater air ({lbl})")

    ax_valve.plot(sol_i.t, u_hist[0], color=c_cooler, linewidth=2,
                  label=f"Cooler ({lbl})")
    ax_valve.plot(sol_i.t, u_hist[1], color=c_heater, linewidth=2,
                  linestyle="--", label=f"Heater ({lbl})")

# ── Reference lines — black, drawn once on top ────────────────────────────────
ax_temp.axhline(T1_ref - 273.15, color="black", linestyle="--",
                linewidth=1.2, label=f"Cooler ref ({T1_ref-273.15:.1f} °C)")
ax_temp.axhline(T2_ref - 273.15, color="black", linestyle="-.",
                linewidth=1.2, label=f"Heater ref ({T2_ref-273.15:.1f} °C)")

# ── Inlet temp — plotted last so it appears below refs in legend ──────────────
if CASE_DISTURBANCE in (1, 4, 5):
    ax_temp.plot(t_ref, np.array([d(t)[0] for t in t_ref]) - 273.15,
                 color="dimgray", linewidth=1.5, linestyle=":",
                 label="Inlet temp")
    
# ── Axes labels, limits, legends ─────────────────────────────────────────────
ax_temp.set_ylabel("Temperature [°C]", fontweight="bold")
ax_temp.set_title(f"Air Temperatures & Disturbances — {case_titles[CASE_DISTURBANCE]}", fontweight="bold")
ax_temp.grid(True, alpha=0.35)
lines_t, labels_t = ax_temp.get_legend_handles_labels()
leg_temp = ax_temp.legend(lines_t + lines_r + lines_f,
                           labels_t + labels_r + labels_f,
                           fontsize=10, loc="upper right",
                           borderpad=1.2, labelspacing=0.6,
                           handlelength=2.5, handletextpad=0.8)
ax_rh.set_ylabel("RH [-]", color="steelblue", fontweight="bold") if CASE_DISTURBANCE in (2, 4) else None
ax_flow.set_ylabel("Air flow [m³/s]", color="mediumpurple", fontweight="bold") if CASE_DISTURBANCE in (3, 4) else None

ax_valve.set_ylabel("Opening [-]", fontweight="bold")
ax_valve.set_xlabel("Time [s]", fontweight="bold")
ax_valve.set_title("Valve Openings", fontweight="bold")
leg_valve = ax_valve.legend(fontsize=10, loc="upper right",
                             borderpad=1.2, labelspacing=0.6,
                             handlelength=2.5, handletextpad=0.8)
ax_valve.grid(True, alpha=0.35)

fig.subplots_adjust(right=0.86 if CASE_DISTURBANCE == 4 else 0.93)
plt.tight_layout()
case_filenames = {
    0: "Constant_disturbances",
    1: "T_in_disturbance",
    2: "RH_in_disturbance",
    3: "Air_flow_disturbance",
    4: "All_disturbances",
    5: "Step_T_in",
}

ctrl_suffix = "normal_LMI_and_LQR" if COMPARE_CONTROLLERS else ("normal_LMI" if USE_DISTURBANCE_REJECTION else "normal_LQR")

plt.savefig(f"{case_filenames[CASE_DISTURBANCE]}_{ctrl_suffix}.png", dpi=300, bbox_inches="tight")
plt.show()