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
#hvac = HVAC(configs=[params_cooler, params_heater], mode=model_mode, const_disturbance=28 + 273.15)
hvac = HVAC(configs=[params_cooler, params_heater], mode=model_mode, const_disturbance=None)

# ── Export state-space model ──────────────────────────────────────────────────
data_dir = Path(__file__).resolve().parent.parent / "models/linear"
data_dir.mkdir(parents=True, exist_ok=True)
hvac._export_state_space(data_dir / "HVAC_model.mat")

# ── Instantiate controller ────────────────────────────────────────────────────
USE_DISTURBANCE_REJECTION = False   # Toggle between LQR and H∞
USE_BRYSON = False                  # Toggle between Bryson and uniform scaling
    
# ── Bryson bounds (physical / absolute frame) ─────────────────────────────────
x_max   = np.full(20, 20 + 273.15)  # absolute temp ceiling per state [K] — converted to deviation inside cost_bryson
u_max   = np.array([0.5, 0.5])      # max acceptable valve command [-]
xI_max  = np.array([10.0, 10.0])    # max acceptable integrator state [K·s]

# ── Cost matrices ─────────────────────────────────────────────────────────────
ControllerCls = (
    StateFeedbackControllerDisturbanceRejection if USE_DISTURBANCE_REJECTION
    else StateFeedbackController
)

if USE_BRYSON:
    Q, R = ControllerCls.cost_bryson(hvac, x_max=x_max, u_max=u_max, x_I_max=xI_max)
else:
    Q, R = ControllerCls.cost_matrices(hvac, Q_scale=5.0, R_scale=800.0)

# ── Build controller ──────────────────────────────────────────────────────────
controller = ControllerCls.find_controller_gains(hvac, Q=Q, R=R)
# ── Dimensions ────────────────────────────────────────────────────────────────
K = hvac._lin_components[0].K
N = hvac.total_states   # 4K = 20

# ── Time ──────────────────────────────────────────────────────────────────────
t_day = 24*3600
points_per_day = t_day * 3

t_end  = 30 #t_day
t_eval = np.linspace(0, t_end, points_per_day)

# ── Initial conditions ────────────────────────────────────────────────────────
x0 = np.concatenate([
    np.full(K, 23 + 273.15),          # cooler air
    np.full(K, 23 + 273.15),          # cooler water
    np.full(K, 9.9 + 273.15),     # heater air
    np.full(K, 9.9 + 273.15),         # heater water
])

# ── References and disturbances ───────────────────────────────────────────────
T1_ref = 10.0 + 273.15   # Cooler air outlet setpoint [K]
T2_ref = 20.0 + 273.15   # Heater air outlet setpoint [K]
r      = np.array([T1_ref, T2_ref])

T_in = 23 + 273.15        # Air inlet to cooler [K]

def d(t):
    Amp = 5
    T_day = 24*3600
    T_in_sys = T_in + Amp * np.sin(2*np.pi*t/T_day)
    return np.array([T_in_sys])
# ── Simulate ──────────────────────────────────────────────────────────────────
augmented_state0  = np.concatenate([x0, np.zeros(controller.n_outputs)])
sol = solve_ivp(
    controller.controller_derivatives(r=r, d=d),
    (0, t_end), augmented_state0, t_eval=t_eval,
    method="Radau", rtol=1e-6, atol=1e-8,
)

# ── Unpack solution ───────────────────────────────────────────────────────────
T_inlet = np.array([d(t)[0] for t in sol.t]) - 273.15
T_air_cooler   = sol.y[0:K]     - 273.15
T_water_cooler = sol.y[K:2*K]   - 273.15
T_air_heater   = sol.y[2*K:3*K] - 273.15
T_water_heater = sol.y[3*K:4*K] - 273.15
x_I_hist       = sol.y[N:]      # shape (n_outputs, n_timesteps)

# ── Decompose control contributions ──────────────────────────────────────────
u_hist = np.array([
    controller.compute_input(sol.y[:N, i], sol.y[N:, i], r)[0]   # add r
    for i in range(sol.y.shape[1])
]).T

Kx_x_hist = np.array([
    controller.K_x @ controller.plant._to_shifted_frame(sol.y[:N, i])  # use shifted frame
    for i in range(sol.y.shape[1])
]).T# shape (n_inputs, n_timesteps)

KI_xI_hist = np.array([
    controller.K_I @ sol.y[N:, i]
    for i in range(sol.y.shape[1])
]).T   # shape (n_inputs, n_timesteps)

# Nr · r is constant (r is fixed), so broadcast across time
r_shift = r - controller.plant.C @ controller.plant.coordinate_shift
Nr_contribution = controller.N @ r_shift          # shape (n_inputs,)
Nr_hist = Nr_contribution[:, np.newaxis] * np.ones((1, len(sol.t)))  # (n_inputs, n_timesteps)

mid = K // 2
# ── Output error (non-integrated) ────────────────────────────────────────────
y_cooler = sol.y[K-1]    - 273.15   # cooler air outlet (last segment)
y_heater = sol.y[3*K-1]  - 273.15   # heater air outlet (last segment)
e_cooler = (T1_ref - 273.15) - y_cooler
e_heater = (T2_ref - 273.15) - y_heater

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(5, 2, figsize=(14, 20), sharex=True)

axes[0, 0].plot(sol.t, T_air_cooler.mean(axis=0), color="tomato", linewidth=2, label="Avg air")
axes[0, 0].plot(sol.t, T_inlet, color="black", linewidth=1.5, linestyle=":", label="Inlet (actual)")
axes[0, 0].axhline(T1_ref - 273.15,  color="green", linestyle="--", label=f"Ref ({T1_ref-273.15:.1f} °C)")
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
axes[0, 1].axhline(T2_ref - 273.15, color="green", linestyle="--", label=f"Ref ({T2_ref-273.15:.1f} °C)")
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
axes[2, 0].grid(True, alpha=0.35)

axes[2, 1].plot(sol.t, x_I_hist[1], color="purple", linewidth=2)
axes[2, 1].set_title("Integrator State — Heater")
axes[2, 1].set_ylabel("x_I [K·s]")
axes[2, 1].grid(True, alpha=0.35)

# Row 3 — Control contributions + sum
for col, label in enumerate(["Cooler", "Heater"]):
    combined = Kx_x_hist[col] + KI_xI_hist[col]
    axes[3, col].plot(sol.t, Kx_x_hist[col],  color="teal",         linewidth=1.5, linestyle="--", label="Kx·x")
    axes[3, col].plot(sol.t, KI_xI_hist[col], color="mediumorchid", linewidth=1.5, linestyle="--", label="KI·xI")
    axes[3, col].plot(sol.t, Nr_hist[col],     color="goldenrod",    linewidth=1.5, linestyle="--", label="Nr·r")
    axes[3, col].plot(sol.t, combined,         color="black",        linewidth=2,                   label="Sum")
    axes[3, col].set_title(f"Control contributions — {label}")
    axes[3, col].set_ylabel("Valve units [-]")
    axes[3, col].legend(fontsize=8)
    axes[3, col].grid(True, alpha=0.35)

# Row 4 — Valve openings
axes[4, 0].plot(sol.t, u_hist[0], color="darkorange", linewidth=2)
axes[4, 0].set_title("Valve Opening — Cooler")
axes[4, 0].set_ylabel("Opening [-]")
axes[4, 0].set_ylim(-0.05, 1.05)
axes[4, 0].set_xlabel("Time [s]")
axes[4, 0].grid(True, alpha=0.35)

axes[4, 1].plot(sol.t, u_hist[1], color="darkorange", linewidth=2)
axes[4, 1].set_title("Valve Opening — Heater")
axes[4, 1].set_ylabel("Opening [-]")
axes[4, 1].set_ylim(-0.05, 1.05)
axes[4, 1].set_xlabel("Time [s]")
axes[4, 1].grid(True, alpha=0.35)

plt.suptitle(f"HVAC cascade ({model_mode}): Cooler → Heater", fontsize=13)
plt.tight_layout()
plt.show()

# ── Terminal summary ──────────────────────────────────────────────────────────
print(f"\n=== Final temperatures ===")
print(f"  {'Seg':<6} {'Cooler Air':>12} {'Cooler Water':>14} {'Heater Air':>12} {'Heater Water':>14}")
print(f"  {'-'*60}")
for k in range(K):
    print(f"  {k+1:<6} {T_air_cooler[k,-1]:>12.3f} {T_water_cooler[k,-1]:>14.3f}"
          f" {T_air_heater[k,-1]:>12.3f} {T_water_heater[k,-1]:>14.3f}")

print(f"\n=== Final valve openings ===")
print(f"  Cooler valve: {u_hist[0,-1]:.4f}")
print(f"  Heater valve: {u_hist[1,-1]:.4f}")

print(f"\n=== Final integrator states ===")
print(f"  x_I[0] (cooler): {x_I_hist[0,-1]:.4f}")
print(f"  x_I[1] (heater): {x_I_hist[1,-1]:.4f}")