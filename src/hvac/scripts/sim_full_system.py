import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from pathlib import Path
from models import HVAC
from controller import StateFeedbackControllerDisturbanceRejection, StateFeedbackController


# ── Flow rates ────────────────────────────────────────────────────────────────
q_fresh  = 2615 / 3600    # fresh air intake through pre-treatment path [m³/s]
q_main   = 8638 / 3600 * 0.5   # total flow in main conditioning loop [m³/s]
q_return = q_main - q_fresh   # recirculated return air [m³/s]

# ── Hardware parameters — pre-treatment path (carries q_fresh) ────────────────
params_cooler_pre = dict(
    type                  = "cooler",
    num_segments          = 5,
    num_pipes             = 10,
    gamma                 = 951.87,
    cross_area_water      = 0.000201*2,
    heat_exchanger_depth  = 0.06*2,
    heat_exchanger_width  = 0.5,
    heat_exchanger_height = 0.5,
    volume_flow_wet_air   = q_fresh,
    water_supply_T        = 4.0 + 273.15,
    Kvs                   = 1.6471,
)
params_heater_pre = dict(
    type                  = "heater",
    num_segments          = 5,
    num_pipes             = 10,
    gamma                 = 951.87,
    cross_area_water      = 0.000201,
    heat_exchanger_depth  = 0.06,
    heat_exchanger_width  = 0.5,
    heat_exchanger_height = 0.5,
    volume_flow_wet_air   = q_fresh,
    water_supply_T        = 66.9 + 273.15,
    Kvs                   = 1.6471,
)

# ── Hardware parameters — main conditioning path (carries q_main) ─────────────
params_cooler_main = dict(
    type                  = "cooler",
    num_segments          = 5,
    num_pipes             = 10,
    gamma                 = 951.87,
    cross_area_water      = 0.000201*2,
    heat_exchanger_depth  = 0.06*2,
    heat_exchanger_width  = 0.5,
    heat_exchanger_height = 0.5,
    volume_flow_wet_air   = q_main,
    water_supply_T        = 4.0 + 273.15,
    Kvs                   = 1.6471,
)
params_heater_main = dict(
    type                  = "heater",
    num_segments          = 5,
    num_pipes             = 10,
    gamma                 = 951.87,
    cross_area_water      = 0.000201,
    heat_exchanger_depth  = 0.06,
    heat_exchanger_width  = 0.5,
    heat_exchanger_height = 0.5,
    volume_flow_wet_air   = q_main,
    water_supply_T        = 66.9 + 273.15,
    Kvs                   = 1.6471,
)

# ── Duct parameters ────────────────────────────────────────────────────────────
duct_params_fresh = dict(   # duct_1: fresh air path
    volume_flow_rate   = q_fresh,
    cross_section_area = 0.25,
    duct_length        = 5.0,
    num_segments       = 5,
)
duct_params_main = dict(    # duct_2, duct_return: main loop
    volume_flow_rate   = q_main,
    cross_section_area = 0.25,
    duct_length        = 5.0,
    num_segments       = 5,
)
room_params = dict(
    volume_flow_rate   = q_main,
    cross_section_area = 20.0,
    duct_length        = 8.0,
    num_segments       = 5,
)

# ── Operating conditions ───────────────────────────────────────────────────────
model_mode = "nonlinear"      # "linear" or "nonlinear"
T_fresh    = 23.0 + 273.15   # outdoor air temperature [K]

# ── Topology ──────────────────────────────────────────────────────────────────
#
#   fresh_air ─► pre_cooler ─► pre_heater ─► duct_1 ─►┐
#                                                       junction_1
#                                          duct_return ─►┘
#                                               ▲
#   junction_1 ─► duct_2 ─► cooler_main ─► heater_main ─► cooler_trim ─► room ─► duct_return
#
nodes = [
    # Pre-treatment path (fresh air only, q_fresh)
    {
        "id":    "pre_cooler",
        "type":  "cooler",
        "config": {
            "T_out_target": 10.0 + 273.15,
            **params_cooler_pre,
        },
    },
    {
        "id":    "pre_heater",
        "type":  "heater",
        "input": "pre_cooler",
        "config": {
            "T_out_target": 19.0 + 273.15,
            **params_heater_pre,
        },
    },
    {
        "id":    "duct_1",
        "type":  "airduct",
        "input": "pre_heater",
        "config": duct_params_fresh,
    },
    # Junction mixes pre-treated fresh air with recirculated return air
    {
        "id":   "junction_1",
        "type": "junction",
        "config": {
            "inputs": [("duct_1", q_fresh), ("duct_return", q_return)],
        },
    },
    # Main conditioning path (q_main)
    {
        "id":    "duct_2",
        "type":  "airduct",
        "input": "junction_1",
        "config": duct_params_main,
    },
    {
        "id":    "cooler_main",
        "type":  "cooler",
        "input": "duct_2",
        "config": {
            "T_out_target": 10.0 + 273.15,
            **params_cooler_main,
        },
    },
    {
        "id":    "heater_main",
        "type":  "heater",
        "input": "cooler_main",
        "config": {
            "T_out_target": 22.0 + 273.15,
            **params_heater_main,
        },
    },
    {
        "id":    "room",
        "type":  "airduct",
        "input": "heater_main",
        "config": room_params,
    },
    {
        "id":    "duct_return",
        "type":  "airduct",
        "input": "room",
        "config": duct_params_main,
    },
]

# ── Instantiate plant ─────────────────────────────────────────────────────────
hvac = HVAC(nodes=nodes, T_fresh=T_fresh, mode=model_mode, const_disturbance=23.0 + 273.15)

# ── Export state-space model ──────────────────────────────────────────────────
data_dir = Path(__file__).resolve().parent.parent / "models/linear"
data_dir.mkdir(parents=True, exist_ok=True)
hvac._export_state_space(data_dir / "HVAC_model.mat")

# ── Instantiate controller ────────────────────────────────────────────────────
USE_DISTURBANCE_REJECTION = False
USE_BRYSON                = False

ControllerCls = (
    StateFeedbackControllerDisturbanceRejection if USE_DISTURBANCE_REJECTION
    else StateFeedbackController
)

if USE_BRYSON:
    x_max  = np.full(hvac.total_states, 20 + 273.15)
    u_max  = np.ones(5) * 0.5
    xI_max = np.ones(5) * 10.0
    Q, R = ControllerCls.cost_bryson(hvac, x_max=x_max, u_max=u_max, x_I_max=xI_max)
else:
    Q, R = ControllerCls.cost_matrices(hvac, Q_scale=5.0, R_scale=800.0)

controller = ControllerCls.find_controller_gains(hvac, Q=Q, R=R)

# ── Dimensions ────────────────────────────────────────────────────────────────
K      = hvac._lin_components[0].K   # HX segments (5)
K_duct = duct_params_fresh["num_segments"]  # duct segments
K_room = room_params["num_segments"]  # room segments (5)
N      = hvac.total_states            # total plant states

# Node ordering (state-bearing only, no junctions):
#   pre_cooler  : 2K  (air + water)
#   pre_heater  : 2K
#   duct_1      : K_duct
#   duct_2      : K_duct
#   cooler_main : 2K
#   heater_main : 2K
#   room        : K_room
#   duct_return : K_duct
# Total = 4*2*5 + 2*3 + 5 = 40 + 6 + 5 + 3 = 54 (+ duct_1/duct_2 = 60)

# ── Time ──────────────────────────────────────────────────────────────────────
t_end  = 200
t_eval = np.linspace(0, t_end, t_end * 10)

# ── Initial conditions (all air at T_fresh, water at supply temps) ─────────────
x0 = np.concatenate([
    np.full(K,      T_fresh),                          # pre_cooler air
    np.full(K,      T_fresh),  # pre_cooler water
    np.full(K,      T_fresh),                          # pre_heater air
    np.full(K,      T_fresh),  # pre_heater water
    np.full(K_duct, T_fresh),                          # duct_1
    np.full(K_duct, T_fresh),                          # duct_2
    np.full(K,      T_fresh),                          # cooler_main air
    np.full(K,      T_fresh),  # cooler_main water
    np.full(K,      T_fresh),                          # heater_main air
    np.full(K,      T_fresh),  # heater_main water
    np.full(K_room, 30 + 273.15),                          # room
    np.full(K_duct, T_fresh),                          # duct_return
])

# ── References ────────────────────────────────────────────────────────────────
# One reference per actuated HX in node order: pre_cooler, pre_heater, cooler_main, heater_main
T_refs = np.array([10.0, 19.0, 10.0, 22.0]) + 273.15
r      = T_refs

def d(t):
    Amp   = 3.0
    T_day = 24 * 3600
    #return np.array([T_fresh + Amp * np.sin(2 * np.pi * t / T_day)])
    return np.array([T_fresh])

# ── Simulate ──────────────────────────────────────────────────────────────────
augmented_state0 = np.concatenate([x0, np.zeros(controller.n_outputs)])
sol = solve_ivp(
    controller.controller_derivatives(r=r, d=d),
    (0, t_end), augmented_state0, t_eval=t_eval,
    method="Radau", rtol=1e-6, atol=1e-8,
)

# ── Unpack solution ────────────────────────────────────────────────────────────
off = 0
T_air_pre_cooler   = sol.y[off:off+K]      - 273.15; off += K
T_water_pre_cooler = sol.y[off:off+K]      - 273.15; off += K
T_air_pre_heater   = sol.y[off:off+K]      - 273.15; off += K
T_water_pre_heater = sol.y[off:off+K]      - 273.15; off += K
T_duct_1           = sol.y[off:off+K_duct] - 273.15; off += K_duct
T_duct_2           = sol.y[off:off+K_duct] - 273.15; off += K_duct
T_air_cooler_main  = sol.y[off:off+K]      - 273.15; off += K
T_water_cooler_main= sol.y[off:off+K]      - 273.15; off += K
T_air_heater_main  = sol.y[off:off+K]      - 273.15; off += K
T_water_heater_main= sol.y[off:off+K]      - 273.15; off += K
T_room             = sol.y[off:off+K_room] - 273.15; off += K_room
T_duct_return      = sol.y[off:off+K_duct] - 273.15; off += K_duct
x_I_hist           = sol.y[N:]

T_fresh_sig = np.array([d(t)[0] for t in sol.t]) - 273.15

# Reconstructed junction mixed temperature
T_junction = (
    q_fresh * T_duct_1[-1] + q_return * T_duct_return[-1]
) / (q_fresh + q_return)

# ── Control history ───────────────────────────────────────────────────────────
u_hist = np.array([
    controller.compute_input(sol.y[:N, i], sol.y[N:, i], r)[0]
    for i in range(sol.y.shape[1])
]).T

# ── Output signals ────────────────────────────────────────────────────────────
y_pre_cooler  = T_air_pre_cooler.mean(axis=0)
y_pre_heater  = T_air_pre_heater.mean(axis=0)
y_cooler_main = T_air_cooler_main.mean(axis=0)
y_heater_main = T_air_heater_main.mean(axis=0)
y_room        = T_room.mean(axis=0)

# ── Plot ──────────────────────────────────────────────────────────────────────
# Layout (4 rows × 2 cols):
#   [0,0] Pre-treatment temps   [0,1] Main conditioning temps
#   [1,0] Pre-treatment valves  [1,1] Main path valves
#   [2,0] Junction mixing       [2,1] Room
hx_names = ["Pre-Cooler", "Pre-Heater", "Cooler Main", "Heater Main"]
fig, axes = plt.subplots(3, 2, figsize=(14, 13), sharex=True)

# [0,0] — Pre-treatment temperatures
ax = axes[0, 0]
ax.plot(sol.t, T_fresh_sig,   color="gray",      lw=1.5, linestyle=":", label="Fresh air in")
ax.plot(sol.t, y_pre_cooler,  color="steelblue", lw=2,   label=f"After pre-cooler (ref {T_refs[0]-273.15:.0f} °C)")
ax.plot(sol.t, y_pre_heater,  color="tomato",    lw=2,   label=f"After pre-heater (ref {T_refs[1]-273.15:.0f} °C)")
ax.axhline(T_refs[0]-273.15,  color="steelblue", lw=1,   linestyle="--", alpha=0.5)
ax.axhline(T_refs[1]-273.15,  color="tomato",    lw=1,   linestyle="--", alpha=0.5)
ax.set_title("Pre-treatment — Temperatures")
ax.set_ylabel("°C"); ax.legend(fontsize=8); ax.grid(True, alpha=0.35)

# [0,1] — Main conditioning temperatures
ax = axes[0, 1]
ax.plot(sol.t, T_junction,    color="black",     lw=1.5, linestyle=":", label="Junction mix in")
ax.plot(sol.t, y_cooler_main, color="steelblue", lw=2,   label=f"After cooler main (ref {T_refs[2]-273.15:.0f} °C)")
ax.plot(sol.t, y_heater_main, color="tomato",    lw=2,   label=f"After heater main (ref {T_refs[3]-273.15:.0f} °C)")
ax.axhline(T_refs[2]-273.15,  color="steelblue", lw=1,   linestyle="--", alpha=0.5)
ax.axhline(T_refs[3]-273.15,  color="tomato",    lw=1,   linestyle="--", alpha=0.5)
ax.set_title("Main conditioning — Temperatures")
ax.set_ylabel("°C"); ax.legend(fontsize=8); ax.grid(True, alpha=0.35)

# [1,0] — Pre-treatment valve openings
ax = axes[1, 0]
for i in range(2):
    ax.plot(sol.t, u_hist[i], lw=2, label=hx_names[i])
ax.set_title("Pre-treatment — Valve Openings")
ax.set_ylabel("Opening [-]"); ax.set_ylim(-0.05, 1.05)
ax.legend(fontsize=8); ax.grid(True, alpha=0.35)

# [1,1] — Main path valve openings
ax = axes[1, 1]
for i in range(2, 4):
    ax.plot(sol.t, u_hist[i], lw=2, label=hx_names[i])
ax.set_title("Main conditioning — Valve Openings")
ax.set_ylabel("Opening [-]"); ax.set_ylim(-0.05, 1.05)
ax.legend(fontsize=8); ax.grid(True, alpha=0.35)

# [2,0] — Junction mixing
ax = axes[2, 0]
ax.plot(sol.t, y_pre_heater,      color="tomato",    lw=1.5, linestyle=":", label="Fresh path in")
ax.plot(sol.t, T_duct_return[-1], color="slateblue", lw=1.5, linestyle=":", label="Return air in")
ax.plot(sol.t, T_junction,        color="black",     lw=2,   label="Junction mix out")
ax.set_title(f"Junction 1 — fresh:{q_fresh/(q_fresh+q_return)*100:.0f}%  return:{q_return/(q_fresh+q_return)*100:.0f}%")
ax.set_ylabel("°C"); ax.set_xlabel("Time [s]")
ax.legend(fontsize=8); ax.grid(True, alpha=0.35)

# [2,1] — Room
ax = axes[2, 1]
ax.plot(sol.t, y_heater_main,     color="tomato",    lw=1.5, linestyle=":", label="Supply air in")
ax.plot(sol.t, y_room,            color="darkorchid",lw=2,   label="Room avg")
ax.plot(sol.t, T_duct_return[-1], color="gray",      lw=1.5, linestyle=":", label="Return air out")
ax.set_title("Room")
ax.set_ylabel("°C"); ax.set_xlabel("Time [s]")
ax.legend(fontsize=8); ax.grid(True, alpha=0.35)

plt.tight_layout()
plt.show()

# ── Terminal summary ──────────────────────────────────────────────────────────
print(f"\n=== Final temperatures ===")
print(f"  Pre-cooler air out  : {y_pre_cooler[-1]:+.2f} °C  (ref {T_refs[0]-273.15:.1f} °C)")
print(f"  Pre-heater air out  : {y_pre_heater[-1]:+.2f} °C  (ref {T_refs[1]-273.15:.1f} °C)")
print(f"  Junction mix        : {T_junction[-1]:+.2f} °C")
print(f"  Cooler main air out : {y_cooler_main[-1]:+.2f} °C  (ref {T_refs[2]-273.15:.1f} °C)")
print(f"  Heater main air out : {y_heater_main[-1]:+.2f} °C  (ref {T_refs[3]-273.15:.1f} °C)")
print(f"  Room avg            : {y_room[-1]:+.2f} °C")
print(f"\n=== Final valve openings ===")
for i, name in enumerate(hx_names):
    print(f"  {name:<16}: {u_hist[i,-1]:.4f}")
