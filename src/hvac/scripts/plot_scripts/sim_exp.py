import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from pathlib import Path
from models import HVAC
from controller import StateFeedbackController


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

params_cooler2 = dict(
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

params_heater2 = dict(
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
hvac = HVAC(
    configs=[params_cooler, params_heater, params_cooler2, params_heater2],
    mode=model_mode,
    disturbance=28 + 273.15,
)

# ── Export state-space model ──────────────────────────────────────────────────
data_dir = Path(__file__).resolve().parent.parent / "models/linear"
data_dir.mkdir(parents=True, exist_ok=True)
hvac._export_state_space(data_dir / "HVAC_model.mat")

# ── Instantiate controller ────────────────────────────────────────────────────
Q, R = StateFeedbackController.cost_matrices(hvac, Q_scale=1, R_scale=1)
controller = StateFeedbackController.find_controller_gains(hvac, Q=Q, R=R)

# ── Dimensions ────────────────────────────────────────────────────────────────
K = hvac._lin_components[0].K   # segments per heat exchanger
N = hvac.total_states            # 4 * 2K = 8K = 40

# ── Time ──────────────────────────────────────────────────────────────────────
t_end  = 15
t_eval = np.linspace(0, t_end, 2000)

# ── Initial conditions ────────────────────────────────────────────────────────
# State order: [cooler1_air, cooler1_water, heater1_air, heater1_water,
#               cooler2_air, cooler2_water, heater2_air, heater2_water]
x0 = np.concatenate([
    np.full(K, 28.0 + 273.15),    # cooler1 air  — starts at inlet disturbance
    np.full(K, 28.0 + 273.15),    # cooler1 water
    np.full(K,  9.9 + 273.15),    # heater1 air
    np.full(K,  9.9 + 273.15),    # heater1 water
    np.full(K, 20.0 + 273.15),    # cooler2 air
    np.full(K, 20.0 + 273.15),    # cooler2 water
    np.full(K, 15.0 + 273.15),    # heater2 air
    np.full(K, 15.0 + 273.15),    # heater2 water
])

# ── References and disturbances ───────────────────────────────────────────────
T1_ref = 10.0 + 273.15   # Cooler1 air outlet setpoint [K]
T2_ref = 20.0 + 273.15   # Heater1 air outlet setpoint [K]
T3_ref = 15.0 + 273.15   # Cooler2 air outlet setpoint [K]
T4_ref = 25.0 + 273.15   # Heater2 air outlet setpoint [K]
r = np.array([T1_ref, T2_ref, T3_ref, T4_ref])

T_in = 28 + 273.15        # Air inlet to cooler1 [K]

def d(t):
    return np.array([T_in])

# ── Simulate ──────────────────────────────────────────────────────────────────
augmented_state0 = np.concatenate([x0, np.zeros(controller.n_outputs)])
sol = solve_ivp(
    controller.controller_derivatives(r=r, d=d),
    (0, t_end), augmented_state0, t_eval=t_eval,
    method="Radau", rtol=1e-6, atol=1e-8,
)

# ── Unpack solution ───────────────────────────────────────────────────────────
T_air_cooler1   = sol.y[0*K : 1*K] - 273.15
T_water_cooler1 = sol.y[1*K : 2*K] - 273.15
T_air_heater1   = sol.y[2*K : 3*K] - 273.15
T_water_heater1 = sol.y[3*K : 4*K] - 273.15
T_air_cooler2   = sol.y[4*K : 5*K] - 273.15
T_water_cooler2 = sol.y[5*K : 6*K] - 273.15
T_air_heater2   = sol.y[6*K : 7*K] - 273.15
T_water_heater2 = sol.y[7*K : 8*K] - 273.15

# ── Control inputs ────────────────────────────────────────────────────────────
u_hist = np.array([
    controller.compute_input(sol.y[:N, i], sol.y[N:, i], r)
    for i in range(sol.y.shape[1])
]).T  # shape (n_inputs=4, n_timesteps)

mid = K // 2

# ── Plot: 3 rows × 4 columns ──────────────────────────────────────────────────
# Row 0: Air temperature   | Row 1: Water temperature  | Row 2: Valve opening
# Col 0: Cooler 1          | Col 1: Heater 1           | Col 2: Cooler 2  | Col 3: Heater 2
elements = [
    {
        "label":    "Cooler 1",
        "T_air":    T_air_cooler1,
        "T_water":  T_water_cooler1,
        "ref":      T1_ref - 273.15,
        "u":        u_hist[0],
        "T_in":     T_in - 273.15,
        "air_color": "tomato",
        "water_color": "steelblue",
    },
    {
        "label":    "Heater 1",
        "T_air":    T_air_heater1,
        "T_water":  T_water_heater1,
        "ref":      T2_ref - 273.15,
        "u":        u_hist[1],
        "T_in":     None,
        "air_color": "tomato",
        "water_color": "steelblue",
    },
    {
        "label":    "Cooler 2",
        "T_air":    T_air_cooler2,
        "T_water":  T_water_cooler2,
        "ref":      T3_ref - 273.15,
        "u":        u_hist[2],
        "T_in":     None,
        "air_color": "tomato",
        "water_color": "steelblue",
    },
    {
        "label":    "Heater 2",
        "T_air":    T_air_heater2,
        "T_water":  T_water_heater2,
        "ref":      T4_ref - 273.15,
        "u":        u_hist[3],
        "T_in":     None,
        "air_color": "tomato",
        "water_color": "steelblue",
    },
]

fig, axes = plt.subplots(3, 4, figsize=(20, 12), sharex=True)

for col, el in enumerate(elements):
    # ── Row 0: Air temperature ────────────────────────────────────────────────
    ax = axes[0, col]
    ax.plot(sol.t, el["T_air"].mean(axis=0),
            color=el["air_color"], linewidth=2, label="Avg air")
    if el["T_in"] is not None:
        ax.axhline(el["T_in"], color="black", linestyle=":",
                   label=f"Inlet ({el['T_in']:.1f} °C)")
    ax.axhline(el["ref"], color="green", linestyle="--",
               label=f"Ref ({el['ref']:.1f} °C)")
    ax.set_title(f"{el['label']} — Air Temperature")
    ax.set_ylabel("Temperature [°C]")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.35)

    # ── Row 1: Water temperature ──────────────────────────────────────────────
    ax = axes[1, col]
    ax.plot(sol.t, el["T_water"][mid],
            color=el["water_color"], linewidth=2, label=f"Seg {mid+1}")
    ax.set_title(f"{el['label']} — Water Temperature")
    ax.set_ylabel("Temperature [°C]")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.35)

    # ── Row 2: Valve opening ──────────────────────────────────────────────────
    ax = axes[2, col]
    ax.plot(sol.t, el["u"], color="darkorange", linewidth=2)
    ax.set_title(f"{el['label']} — Valve Opening")
    ax.set_ylabel("Opening [-]")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Time [s]")
    ax.grid(True, alpha=0.35)

plt.suptitle(
    f"HVAC cascade ({model_mode}): Cooler 1 → Heater 1 → Cooler 2 → Heater 2",
    fontsize=13,
)
plt.tight_layout()
plt.show()

# ── Terminal summary ──────────────────────────────────────────────────────────
print(f"\n=== Final temperatures ===")
headers = ("Seg", "C1 Air", "C1 Water", "H1 Air", "H1 Water",
           "C2 Air", "C2 Water", "H2 Air", "H2 Water")
print(f"  {headers[0]:<6}" + "".join(f"{h:>12}" for h in headers[1:]))
print(f"  {'-'*102}")
for k in range(K):
    print(
        f"  {k+1:<6}"
        f" {T_air_cooler1[k,-1]:>12.3f} {T_water_cooler1[k,-1]:>12.3f}"
        f" {T_air_heater1[k,-1]:>12.3f} {T_water_heater1[k,-1]:>12.3f}"
        f" {T_air_cooler2[k,-1]:>12.3f} {T_water_cooler2[k,-1]:>12.3f}"
        f" {T_air_heater2[k,-1]:>12.3f} {T_water_heater2[k,-1]:>12.3f}"
    )

print(f"\n=== Final valve openings ===")
for i, label in enumerate(["Cooler 1", "Heater 1", "Cooler 2", "Heater 2"]):
    print(f"  {label}: {u_hist[i,-1]:.4f}")