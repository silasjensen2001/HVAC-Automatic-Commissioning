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
COMPARE_CONTROLLERS       = True   # True: overlay both in one 2×1 layout
USE_DISTURBANCE_REJECTION = False   # Used when COMPARE_CONTROLLERS = False
USE_BRYSON                = False    # Only used for LQR (not compatible with LMI design)
CASE_DISTURBANCE          = 5       # 0: All constant, 1: Temp, 2: RH, 3: Flow, 4: All combined, 5: Step change in T_in
PLOT_OMEGA                = True   # True: add a third subplot with omega per controller

# ── Dimensions ────────────────────────────────────────────────────────────────
K = hvac._lin_components[0].K
N = hvac.total_states

# ── Bryson bounds ─────────────────────────────────────────────────────────────
air_temp_max_error   = 1.0
water_temp_max_error = 50.0
wanted_settling_time = 2.0

x_max = np.concatenate([
    np.full(K, air_temp_max_error),
    np.full(K, water_temp_max_error),
    np.full(K, air_temp_max_error),
    np.full(K, water_temp_max_error),
])
xI_max = np.full(2, air_temp_max_error * wanted_settling_time)
u_max  = np.array([1.0, 1.0])

# ── Time ──────────────────────────────────────────────────────────────────────
t_day          = 24 * 3600
points_per_day = 3000
t_end          = 80 if CASE_DISTURBANCE == 5 else 2*t_day
t_start        = 59 if CASE_DISTURBANCE == 5 else 0
t_eval         = np.linspace(t_start, t_end, points_per_day)

# ── References ────────────────────────────────────────────────────────────────
T1_ref = 10.0 + 273.15
T2_ref = 20.0 + 273.15
r      = np.array([T1_ref, T2_ref])

# ── Disturbance function ──────────────────────────────────────────────────────
STEP_TIME = 60.0

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
            T_in = (28
                    + 6   * np.cos(2 * np.pi * shift_t / 86400)
                    + 1.6 * np.cos(2 * np.pi * shift_t / 43200)
                    + 0.5 * np.cos(2 * np.pi * shift_t / 28800)
                    + 2   * np.sin(2 * np.pi * t / 259200)
                    + 273.15)
            rh_in               = np.clip(0.70 + 0.25 * np.sin(2 * np.pi * t / T_day + 2 * np.pi / 3), 0.0, 1.0)
            flow_base           = hvac._lin_components[0].volume_flow_wet_air
            volume_flow_wet_air = flow_base + 0.5 * flow_base * np.cos(2 * np.pi * t / (t_day / 2))
        case 5:
            T_in = 15 + 273.15 if t < STEP_TIME else 23 + 273.15
            rh_in = 0.832
            volume_flow_wet_air = (
                params_cooler["volume_flow_wet_air"]
                / (params_cooler["num_segments"] * params_cooler["num_pipes"])
            )
    return np.array([T_in, rh_in, volume_flow_wet_air])

# ── Initial conditions ────────────────────────────────────────────────────────
_T_in_0 = d(t_start)[0]
print(f"Initial inlet temperature: {_T_in_0 - 273.15:.2f} °C")
x0 = np.concatenate([
    np.full(K, _T_in_0),
    np.full(K, _T_in_0),
    np.full(K, 9.9 + 273.15),
    np.full(K, 9.9 + 273.15),
])

# ── Build controller and simulate ─────────────────────────────────────────────
def build_and_simulate(use_disturbance_rejection):
    ControllerCls = (
        StateFeedbackControllerDisturbanceRejection if use_disturbance_rejection
        else StateFeedbackController
    )
    if use_disturbance_rejection:
        Q, R = ControllerCls.cost_matrices(hvac, Q_air_temp=0.25, Q_water_temp=1e-4, Q_i=0.08, R_u=2.0)
    else:
        Q, R = (ControllerCls.cost_bryson(hvac, x_max=x_max, u_max=u_max, x_I_max=xI_max, shifted=False)
                if USE_BRYSON else
                ControllerCls.cost_matrices(hvac, Q_air_temp=0.25, Q_water_temp=1e-4, Q_i=0.16, R_u=0.5))
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
    solutions = [(sol_dr, ctrl_dr, "LMI"), (sol_lqr, ctrl_lqr, "LQR")]
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
    if PLOT_OMEGA:
        ode_with_omega = ctrl.controller_derivatives(r=r, d=d, return_omega=True)
        omega_hist = np.array([
            ode_with_omega(sol.t[i], sol.y[:, i])[1]
            for i in range(sol.y.shape[1])
        ])
    else:
        omega_hist = None
    return T_inlet, T_air_cooler, T_air_heater, u_hist, omega_hist


# ── Step-response metrics (disturbance rejection) ─────────────────────────────
def compute_step_metrics(t, signal, ref_degC, step_time=STEP_TIME, settling_band=0.05):
    mask   = t >= step_time
    t_post = t[mask]
    y_post = signal[mask]

    if len(t_post) < 4:
        return None

    error = y_post - ref_degC

    # ── Peak ──────────────────────────────────────────────────────────────
    peak_idx       = int(np.argmax(np.abs(error)))
    peak_deviation = float(error[peak_idx])
    t_peak         = float(t_post[peak_idx])

    if abs(peak_deviation) < 1e-6:
        return None

    # ── Settling time ──────────────────────────────────────────────────────
    band    = settling_band
    outside = np.where(np.abs(error) > band)[0]
    if len(outside) == 0:
        settling_time = 0.0
        t_settled     = step_time
    else:
        last_out      = int(outside[-1])
        t_settled     = float(t_post[last_out])
        settling_time = t_settled - step_time

    return dict(
        peak_deviation = peak_deviation,
        t_peak         = t_peak,
        peak_value     = float(y_post[peak_idx]),
        settling_time  = settling_time,
        t_settled      = t_settled,
        band           = band,
        ref            = ref_degC,
    )


# ── Annotation layout config ──────────────────────────────────────────────────
def annotate_metrics(ax, metrics, color, lane_offset, x_settle_offset=0.3):
    """
    Draw metrics annotations in a fixed vertical lane.
    Settling-time label is placed LEFT of the dashed line so it never
    falls off the right edge of the axes.
    """
    if metrics is None:
        return

    ymin, ymax = ax.get_ylim()
    y_range    = ymax - ymin
    ref        = metrics["ref"]
    sign       = np.sign(metrics["peak_deviation"])

    y_lane = ref + lane_offset

    # ── ±2 % settling band ────────────────────────────────────────────────
    ax.axhspan(ref - metrics["band"], ref + metrics["band"],
               color=color, alpha=0.07, zorder=0)

    # ── Settling-time vertical dashed line + label LEFT of line ───────────
    t_s = metrics["t_settled"]
    ax.axvline(t_s, color=color, linestyle=":", lw=1.5, alpha=0.9)

    # Determine whether the label fits to the right; if not, go left.
    x_data_range = ax.get_xlim()[1] - ax.get_xlim()[0]
    right_room   = ax.get_xlim()[1] - t_s          # data units to right edge

    if right_room < 0.12 * x_data_range:
        # Not enough room on right → place label to the LEFT
        ax.text(t_s - 0.2, y_lane,
                f"$t_s$ = {metrics['settling_time']:.2f} s",
                ha="right", va="center", fontsize=10, color=color,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=color,
                          alpha=0.92, lw=0.8))
    else:
        ax.text(t_s + 0.2, y_lane,
                f"$t_s$ = {metrics['settling_time']:.2f} s",
                ha="left", va="center", fontsize=10, color=color,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=color,
                          alpha=0.92, lw=0.8))

    # ── Peak-deviation annotation ─────────────────────────────────────────
    ax.annotate(
        f"$\\Delta T_{{peak}}$ = {metrics['peak_deviation']:+.2f} °C",
        xy=(metrics["t_peak"], metrics["peak_value"]),
        xytext=(metrics["t_peak"] + 0.2, y_lane - sign * 0.05 * y_range + 5.0),
        arrowprops=dict(arrowstyle="->", color=color, lw=1.2,
                        connectionstyle="arc3,rad=0.15"),
        ha="left", va="center", fontsize=10, color=color,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=color,
                  alpha=0.92, lw=0.8),
        annotation_clip=False)

    ax.set_ylim(ymin, ymax)


# ── Disturbance signals ───────────────────────────────────────────────────────
t_ref       = solutions[0][0].t
RH_in       = np.array([d(t)[1] for t in t_ref])
V_flow_hist = np.array([d(t)[2] for t in t_ref])

# ── Viridis colour scheme ─────────────────────────────────────────────────────
_vir = plt.cm.viridis
ctrl_colors = {
    "LMI": (_vir(0.15), _vir(0.45)),
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

# ── Figure ────────────────────────────────────────────────────────────────────
n_rows     = 3 if PLOT_OMEGA else 2
fig_height = 12 if PLOT_OMEGA else 9
fig, axes  = plt.subplots(n_rows, 1, figsize=(13, fig_height), sharex=True)
ax_temp  = axes[0]
ax_valve = axes[1]
ax_omega = axes[2] if PLOT_OMEGA else None

# ── Shared disturbance overlays ───────────────────────────────────────────────
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
        ax_flow.spines["right"].set_position(("axes", 1.08))
        ax_flow.spines["right"].set_visible(True)
    ax_flow.plot(t_ref, V_flow_hist, color="mediumpurple", linewidth=1.5,
                 linestyle="--", alpha=0.7, label="Air flow")
    ax_flow.set_ylabel("Air flow [m³/s]", color="mediumpurple", fontweight="bold")
    ax_flow.tick_params(axis="y", labelcolor="mediumpurple")
    v_min, v_max = V_flow_hist.min(), V_flow_hist.max()
    v_range = v_max - v_min
    ax_flow.set_ylim(v_min - 3 * v_range, v_max + 0.1 * v_range)
    ax_flow.yaxis.set_major_locator(ticker.LinearLocator(numticks=4))
    ax_flow.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.4f"))
    lines_f, labels_f = ax_flow.get_legend_handles_labels()

# ── Per-controller traces ─────────────────────────────────────────────────────
unpacked_data = []

for sol_i, ctrl_i, lbl in solutions:
    T_inlet, T_air_cooler, T_air_heater, u_hist, omega_hist = unpack(sol_i, ctrl_i)
    unpacked_data.append((sol_i, lbl, T_air_cooler, T_air_heater))

    c_cooler, c_heater = ctrl_colors[lbl]

    ax_temp.plot(sol_i.t, T_air_cooler.mean(axis=0), color=c_cooler,
                 linewidth=2, label=f"Cooler air ({lbl})")
    ax_temp.plot(sol_i.t, T_air_heater.mean(axis=0), color=c_heater,
                 linewidth=2, label=f"Heater air ({lbl})")

    ax_valve.plot(sol_i.t, u_hist[0], color=c_cooler, linewidth=2,
                  label=f"Cooler ({lbl})")
    ax_valve.plot(sol_i.t, u_hist[1], color=c_heater, linewidth=2,
                  linestyle="--", label=f"Heater ({lbl})")

    if PLOT_OMEGA:
        ax_omega.plot(sol_i.t, omega_hist[:, :K].mean(axis=1), color=c_cooler,
                      linewidth=2, label=f"Cooler ω ({lbl})")
        ax_omega.plot(sol_i.t, omega_hist[:, K:].mean(axis=1), color=c_heater,
                      linewidth=2, linestyle="--", label=f"Heater ω ({lbl})")

# ── Reference lines ────────────────────────────────────────────────────────────
ax_temp.axhline(T1_ref - 273.15, color="black", linestyle="--",
                linewidth=1.2, label=f"Cooler ref ({T1_ref-273.15:.1f} °C)")
ax_temp.axhline(T2_ref - 273.15, color="black", linestyle="-.",
                linewidth=1.2, label=f"Heater ref ({T2_ref-273.15:.1f} °C)")

if CASE_DISTURBANCE in (1, 4, 5):
    ax_temp.plot(t_ref, np.array([d(t)[0] for t in t_ref]) - 273.15,
                 color="dimgray", linewidth=1.5, linestyle=":",
                 label="Inlet temp")

# ── Axis labels / titles / legends ───────────────────────────────────────────
ax_temp.set_ylabel("Temperature [°C]", fontweight="bold")
ax_temp.set_title(f"Air Temperatures & Disturbances — {case_titles[CASE_DISTURBANCE]}",
                  fontweight="bold")
ax_temp.grid(True, alpha=0.35)

lines_t, labels_t = ax_temp.get_legend_handles_labels()
ax_temp.legend(lines_t + lines_r + lines_f,
               labels_t + labels_r + labels_f,
               fontsize=10, loc="upper right",
               borderpad=1.2, labelspacing=0.6,
               handlelength=2.5, handletextpad=0.8)

if CASE_DISTURBANCE in (2, 4):
    ax_rh.set_ylabel("RH [-]", color="steelblue", fontweight="bold")
if CASE_DISTURBANCE in (3, 4):
    ax_flow.set_ylabel("Air flow [m³/s]", color="mediumpurple", fontweight="bold")

ax_valve.set_ylabel("Opening [-]", fontweight="bold")
ax_valve.set_title("Valve Openings", fontweight="bold")
ax_valve.grid(True, alpha=0.35)
ax_valve.legend(fontsize=10, loc="upper right",
                borderpad=1.2, labelspacing=0.6,
                handlelength=2.5, handletextpad=0.8)

if PLOT_OMEGA:
    ax_omega.set_ylabel("ω [-]", fontweight="bold")
    ax_omega.set_title("Omega", fontweight="bold")
    ax_omega.grid(True, alpha=0.35)
    ax_omega.legend(fontsize=10, loc="upper right",
                    borderpad=1.2, labelspacing=0.6,
                    handlelength=2.5, handletextpad=0.8)
    ax_omega.set_xlabel("Time [s]", fontweight="bold")
else:
    ax_valve.set_xlabel("Time [s]", fontweight="bold")

fig.subplots_adjust(right=0.86 if CASE_DISTURBANCE == 4 else 0.93)

# ── Finalise layout and lock ylim before annotating ──────────────────────────
plt.tight_layout()
fig.canvas.draw()

if CASE_DISTURBANCE == 5:
    # Expand ylim to give clean headroom for annotation lanes
    ymin_cur, ymax_cur = ax_temp.get_ylim()
    y_span = ymax_cur - ymin_cur
    ax_temp.set_ylim(ymin_cur - 0.05 * y_span, ymax_cur + 0.45 * y_span)
    fig.canvas.draw()

    # ── Lane assignments (fraction of y_range relative to each ref) ──────
    # Cooler (ref=10°C, signal goes UP): put its lane above the peak,
    #   well clear of the heater ref line at 20°C.
    # Heater (ref=20°C, signal barely moves): put its lane above heater ref.
    # For COMPARE_CONTROLLERS the second controller gets a slightly higher lane.
    #
    # lane_cooler[i] and lane_heater[i] are for the i-th entry in solutions.
    n_ctrl = len(solutions)
    if n_ctrl == 1:
        lanes_cooler = [5.0]    # fraction of y_range above cooler ref (10°C)
        lanes_heater = [5.0]    # fraction above heater ref (20°C)
        x_settle_offsets = [0.0]
    else:  # two controllers side-by-side
        lanes_cooler = [2.0, 0.0]
        lanes_heater = [2.0, 0.0]
        x_settle_offsets = [0.0, 0.0]

    print(f"\n── Step-response metrics ({case_titles[5]}) ──")
    for idx, (sol_i, lbl, T_air_cooler, T_air_heater) in enumerate(unpacked_data):
        c_cooler, c_heater = ctrl_colors[lbl]
        x_off = x_settle_offsets[idx]

        # Cooler
        m_cooler = compute_step_metrics(sol_i.t, T_air_cooler.mean(axis=0),
                                        ref_degC=T1_ref - 273.15)
        annotate_metrics(ax_temp, m_cooler, color=c_cooler,
                         lane_offset=lanes_cooler[idx], x_settle_offset=x_off)
        if m_cooler:
            print(f"  Cooler ({lbl}):")
            print(f"    Peak deviation : {m_cooler['peak_deviation']:+.4f} °C")
            print(f"    Settling time  : {m_cooler['settling_time']:.4f} s")

        # Heater
        m_heater = compute_step_metrics(sol_i.t, T_air_heater.mean(axis=0),
                                        ref_degC=T2_ref - 273.15)
        annotate_metrics(ax_temp, m_heater, color=c_heater,
                         lane_offset=lanes_heater[idx], x_settle_offset=x_off)
        if m_heater:
            print(f"  Heater ({lbl}):")
            print(f"    Peak deviation : {m_heater['peak_deviation']:+.4f} °C")
            print(f"    Settling time  : {m_heater['settling_time']:.4f} s")

# ── Save ──────────────────────────────────────────────────────────────────────
case_filenames = {
    0: "Constant_disturbances",
    1: "T_in_disturbance",
    2: "RH_in_disturbance",
    3: "Air_flow_disturbance",
    4: "All_disturbances",
    5: "Step_T_in",
}
ctrl_suffix = ("normal_LMI_and_LQR" if COMPARE_CONTROLLERS
               else ("normal_LMI" if USE_DISTURBANCE_REJECTION else "normal_LQR"))

plt.savefig(f"{case_filenames[CASE_DISTURBANCE]}_{ctrl_suffix}.png", dpi=300, bbox_inches="tight")
plt.show()