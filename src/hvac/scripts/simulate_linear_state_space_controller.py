import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from pathlib import Path
from models import LinearHeatExchanger, HVAC
from scipy.io import loadmat


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

# ── Instantiate components and HVAC ──────────────────────────────────────────
lin_cooler = LinearHeatExchanger(**params_cooler)
lin_heater = LinearHeatExchanger(**params_heater)

hvac = HVAC(components=[lin_cooler, lin_heater])

# ── Export model ──────────────────────────────────────────────────────────────
data_dir = Path(__file__).resolve().parent.parent / "models/linear"
data_dir.mkdir(parents=True, exist_ok=True)
type_label = "HVAC"
data_path = data_dir / f"{type_label}_model.mat"
hvac._export_state_space(data_path)

# ── Load gain matrices ────────────────────────────────────────────────────────
controller_dir = Path(__file__).resolve().parent.parent / "models/controller"
_K_x = loadmat(controller_dir / "K_x.mat")
_K_I = loadmat(controller_dir / "K_I.mat")

K_x = _K_x["Kx"]   # shape (2, N)
K_I = _K_I["Ki"]   # shape (2, 2)

# ── Dimensions ────────────────────────────────────────────────────────────────
K = lin_cooler.K
N = hvac.total_states  # 4K = 20

# ── Time ──────────────────────────────────────────────────────────────────────
t_end  = 500
t_eval = np.linspace(0, t_end, 10000)

# ── Initial conditions ────────────────────────────────────────────────────────
x0 = np.concatenate([
    np.full(K, 28 + 273.15),          # cooler air
    np.full(K, 28 + 273.15),          # cooler water
    np.full(K, 11.3235 + 273.15),     # heater air
    np.full(K, 9.9 + 273.15),         # heater water
])

# ── References ────────────────────────────────────────────────────────────────
T1_ref = 12.0 + 273.15   # Cooler air outlet setpoint [K]
T2_ref = 20.0 + 273.15   # Heater air outlet setpoint [K]

# ── Inputs ────────────────────────────────────────────────────────────────────
T_in = 28 + 273.15   # Air inlet to cooler [K]

def d(t):
    return np.array([T_in])

# ── Output function: [T1, T2] ─────────────────────────────────────────────────
def outputs(x):
    T1 = x[0:K].mean()         # avg cooler-air  ← matches C matrix
    T2 = x[2*K:3*K].mean()     # avg heater-air  ← matches C matrix
    return np.array([T1, T2])

# ── Control law ───────────────────────────────────────────────────────────────
def u(x, x_I):
    z = hvac._to_shifted_frame(x)       
    val = K_x @ z + K_I @ x_I          # standard LQI in deviation coordinates
    #val = K_I @ x_I
    return np.clip(val, 0.0, 1.0)

# ── Augmented ODE: z = [x (N), x_I (2)] ──────────────────────────────────────
def augmented_derivatives(t, z):
    x   = z[:N]
    x_I = z[N:]

    y_ref   = np.array([T1_ref, T2_ref])
    error   = y_ref - outputs(x)

    # Raw (unclipped) control signal
    u_raw = K_x @ hvac._to_shifted_frame(x) + K_I @ x_I

    # Anti-windup: freeze integrator when saturated AND error pushes deeper into saturation
    anti_windup = np.ones(2)
    for i in range(2):
        if (u_raw[i] >= 1.0 and error[i] > 0) or (u_raw[i] <= 0.0 and error[i] < 0):
            anti_windup[i] = 0.0

    dx_I = anti_windup * error

    dx = hvac.derivatives(x, u(x, x_I), d(t))
    return np.concatenate([dx, dx_I])

# ── Augmented initial conditions ──────────────────────────────────────────────
x_I0 = np.zeros(2)
z0   = np.concatenate([x0, x_I0])

# ── Simulate ──────────────────────────────────────────────────────────────────
sol = solve_ivp(
    augmented_derivatives,
    (0, t_end), z0, t_eval=t_eval,
    method="Radau", rtol=1e-6, atol=1e-8,
)

# ── Unpack ────────────────────────────────────────────────────────────────────
T_air_cooler   = sol.y[0:K]     - 273.15
T_water_cooler = sol.y[K:2*K]   - 273.15
T_air_heater   = sol.y[2*K:3*K] - 273.15
T_water_heater = sol.y[3*K:4*K] - 273.15
x_I_hist       = sol.y[N:]      # shape (2, n_timesteps)

# ── Reconstruct valve signals ─────────────────────────────────────────────────
u_hist = np.array([
    u(sol.y[:N, i], sol.y[N:, i]) for i in range(sol.y.shape[1])
]).T   # shape (2, n_timesteps)

mid = K // 2

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 2, figsize=(14, 12), sharex=True)

axes[0, 0].plot(sol.t, T_air_cooler.mean(axis=0), color="tomato", linewidth=2, label="Avg air")
axes[0, 0].axhline(T_in - 273.15, color="black", linestyle=":", label=f"Inlet ({T_in-273.15:.1f} °C)")
axes[0, 0].axhline(T1_ref - 273.15, color="green", linestyle="--", label=f"Ref ({T1_ref-273.15:.1f} °C)")
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
axes[2, 0].axhline(0, color="black", linestyle=":", linewidth=1)
axes[2, 0].set_title("Integral State — Cooler (x_I[0])")
axes[2, 0].set_ylabel("Integrated error [K·s]")
axes[2, 0].set_xlabel("Time [s]")
axes[2, 0].grid(True, alpha=0.35)

axes[2, 1].plot(sol.t, x_I_hist[1], color="purple", linewidth=2)
axes[2, 1].axhline(0, color="black", linestyle=":", linewidth=1)
axes[2, 1].set_title("Integral State — Heater (x_I[1])")
axes[2, 1].set_ylabel("Integrated error [K·s]")
axes[2, 1].set_xlabel("Time [s]")
axes[2, 1].grid(True, alpha=0.35)


plt.suptitle("HVAC cascade (Linear): Cooler → Heater", fontsize=13)
plt.tight_layout()
plt.show()

# ── Terminal summary ──────────────────────────────────────────────────────────
print("K_I =", K_I)
print("K_x =", K_x)
print("coordinate_shift =", hvac.coordinate_shift)

print(f"\n=== Final temperatures ===")
print(f"  {'Seg':<6} {'Cooler Air':>12} {'Cooler Water':>14} {'Heater Air':>12} {'Heater Water':>14}")
print(f"  {'-'*60}")
for k in range(K):
    print(f"  {k+1:<6} {T_air_cooler[k,-1]:>12.3f} {T_water_cooler[k,-1]:>14.3f}"
          f" {T_air_heater[k,-1]:>12.3f} {T_water_heater[k,-1]:>14.3f}")

print(f"\n=== Final valve openings ===")
print(f"  Cooler valve: {u_hist[0,-1]:.4f}")
print(f"  Heater valve: {u_hist[1,-1]:.4f}")