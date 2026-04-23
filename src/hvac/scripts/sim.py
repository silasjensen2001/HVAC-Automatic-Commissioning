import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from pathlib import Path
from models import LinearHeatExchanger, HVAC
from controller import StateFeedbackController


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

# ── Instantiate plant ─────────────────────────────────────────────────────────
lin_cooler = LinearHeatExchanger(**params_cooler)
lin_heater = LinearHeatExchanger(**params_heater)
hvac       = HVAC(components=[lin_cooler, lin_heater])

# ── Export state-space model ──────────────────────────────────────────────────
data_dir = Path(__file__).resolve().parent.parent / "models/linear"
data_dir.mkdir(parents=True, exist_ok=True)
hvac._export_state_space(data_dir / "HVAC_model.mat")

# ── Instantiate controller ────────────────────────────────────────────────────
controller_dir = Path(__file__).resolve().parent.parent / "models/controller"
controller = StateFeedbackController.from_mat_files(
    plant    = hvac,
    K_x_path = controller_dir / "K_x.mat",
    K_I_path = controller_dir / "K_I.mat",
)

# ── Dimensions ────────────────────────────────────────────────────────────────
K = lin_cooler.K
N = hvac.total_states   # 4K = 20

# ── Time ──────────────────────────────────────────────────────────────────────
t_end  = 2000
t_eval = np.linspace(0, t_end, 2000)

# ── Initial conditions ────────────────────────────────────────────────────────
x0 = np.concatenate([
    np.full(K, 28 + 273.15),          # cooler air
    np.full(K, 28 + 273.15),          # cooler water
    np.full(K, 9.9 + 273.15),     # heater air
    np.full(K, 9.9 + 273.15),         # heater water
])

# ── References and disturbances ───────────────────────────────────────────────
T1_ref = 15.0 + 273.15   # Cooler air outlet setpoint [K]
T2_ref = 20.0 + 273.15   # Heater air outlet setpoint [K]
r      = np.array([T1_ref, T2_ref])

T_in = 28 + 273.15        # Air inlet to cooler [K]

def d(t):
    return np.array([T_in])

# ── Simulate ──────────────────────────────────────────────────────────────────
augmented_state0  = np.concatenate([x0, np.zeros(controller.n_outputs)])
sol = solve_ivp(
    controller.controller_derivatives(r=r, d=d),
    (0, t_end), augmented_state0, t_eval=t_eval,
    method="Radau", rtol=1e-6, atol=1e-8,
)

# ── Unpack solution ───────────────────────────────────────────────────────────
T_air_cooler   = sol.y[0:K]     - 273.15
T_water_cooler = sol.y[K:2*K]   - 273.15
T_air_heater   = sol.y[2*K:3*K] - 273.15
T_water_heater = sol.y[3*K:4*K] - 273.15
x_I_hist       = sol.y[N:]      # shape (n_outputs, n_timesteps)

u_hist = np.array([
    controller.compute_input(sol.y[:N, i], sol.y[N:, i])
    for i in range(sol.y.shape[1])
]).T   # shape (n_inputs, n_timesteps)

mid = K // 2

# ── Decompose control contributions ──────────────────────────────────────────
Kx_x_hist = np.array([
    controller.K_x @ sol.y[:N, i]
    for i in range(sol.y.shape[1])
]).T   # shape (n_inputs, n_timesteps)

KI_xI_hist = np.array([
    controller.K_I @ sol.y[N:, i]
    for i in range(sol.y.shape[1])
]).T   # shape (n_inputs, n_timesteps)

# ── Output error (non-integrated) ────────────────────────────────────────────
y_cooler = sol.y[K-1]    - 273.15   # cooler air outlet (last segment)
y_heater = sol.y[3*K-1]  - 273.15   # heater air outlet (last segment)
e_cooler = (T1_ref - 273.15) - y_cooler
e_heater = (T2_ref - 273.15) - y_heater

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(6, 2, figsize=(14, 24), sharex=True)

axes[0, 0].plot(sol.t, T_air_cooler.mean(axis=0), color="tomato", linewidth=2, label="Avg air")
axes[0, 0].axhline(T_in - 273.15,    color="black", linestyle=":",  label=f"Inlet ({T_in-273.15:.1f} °C)")
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

"""
axes[2, 0].plot(sol.t, u_hist[0], color="darkorange", linewidth=2)
axes[2, 0].set_title("Valve Opening — Cooler")
axes[2, 0].set_ylabel("Opening [-]")
axes[2, 0].set_ylim(-0.05, 1.05)
axes[2, 0].set_xlabel("Time [s]")
axes[2, 0].grid(True, alpha=0.35)

axes[2, 1].plot(sol.t, u_hist[1], color="darkorange", linewidth=2)
axes[2, 1].set_title("Valve Opening — Heater")
axes[2, 1].set_ylabel("Opening [-]")
axes[2, 1].set_ylim(-0.05, 1.05)
axes[2, 1].set_xlabel("Time [s]")
axes[2, 1].grid(True, alpha=0.35)
"""
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

# Row 5 — Raw error
axes[5, 0].plot(sol.t, e_cooler, color="crimson", linewidth=2)
axes[5, 0].axhline(0, color="black", linestyle="--", linewidth=0.8)
axes[5, 0].set_title("Error — Cooler")
axes[5, 0].set_ylabel("e [°C]")
axes[5, 0].set_xlabel("Time [s]")
axes[5, 0].grid(True, alpha=0.35)

axes[5, 1].plot(sol.t, e_heater, color="crimson", linewidth=2)
axes[5, 1].axhline(0, color="black", linestyle="--", linewidth=0.8)
axes[5, 1].set_title("Error — Heater")
axes[5, 1].set_ylabel("e [°C]")
axes[5, 1].set_xlabel("Time [s]")
axes[5, 1].grid(True, alpha=0.35)

plt.suptitle("HVAC cascade (Linear): Cooler → Heater", fontsize=13)
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