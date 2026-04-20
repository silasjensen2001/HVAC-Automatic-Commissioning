import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from pathlib import Path
from models import LinearHeatExchanger, HVAC


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

# export model

data_dir = Path(__file__).resolve().parent.parent / "models/linear"
data_dir.mkdir(parents=True, exist_ok=True)
type_label = "HVAC"
data_path = data_dir / f"{type_label}_model.mat"

hvac._export_state_space(data_path)

K = lin_cooler.K
N = hvac.total_states  # 2 * 2K = 20

# ── Time ──────────────────────────────────────────────────────────────────────
t_end  = 500
t_eval = np.linspace(0, t_end, 10000)

# ── Initial conditions ────────────────────────────────────────────────────────
x0 = np.concatenate([
    np.full(K, 28 + 273.15),   # cooler air
    np.full(K, 28 + 273.15),   # cooler water
    np.full(K, 11.3235 + 273.15),  # heater air
    np.full(K, 9.9 + 273.15),  # heater water
])

# ── Inputs ────────────────────────────────────────────────────────────────────
T_in          = 28 + 273.15   # Air inlet to cooler [K]
valve_cooler  = 0.95
valve_heater  = 0.02

def u(t):
    return np.array([valve_cooler, valve_heater])  # one per component

def d(t):
    return np.array([T_in])  # only external disturbance: cooler air inlet

# ── Simulate ──────────────────────────────────────────────────────────────────
sol = solve_ivp(
    lambda t, x: hvac.derivatives(x, u(t), d(t)),
    (0, t_end), x0, t_eval=t_eval,
    method="Radau", rtol=1e-6, atol=1e-8,
)

# ── Unpack ────────────────────────────────────────────────────────────────────
T_air_cooler   = sol.y[0:K]       - 273.15
T_water_cooler = sol.y[K:2*K]     - 273.15
T_air_heater   = sol.y[2*K:3*K]   - 273.15
T_water_heater = sol.y[3*K:4*K]   - 273.15

mid = K // 2

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)

axes[0, 0].plot(sol.t, T_air_cooler.mean(axis=0), color="tomato", linewidth=2, label="Avg air")
axes[0, 0].axhline(T_in - 273.15, color="black", linestyle=":", label=f"Inlet ({T_in-273.15:.1f} °C)")
axes[0, 0].set_title("Cooler — Air Temperature")
axes[0, 0].set_ylabel("Temperature [°C]")
axes[0, 0].legend(fontsize=8)
axes[0, 0].grid(True, alpha=0.35)

axes[1, 0].plot(sol.t, T_water_cooler[mid], color="steelblue", linewidth=2, label=f"Seg {mid+1}")
axes[1, 0].set_title("Cooler — Water Temperature")
axes[1, 0].set_ylabel("Temperature [°C]")
axes[1, 0].set_xlabel("Time [s]")
axes[1, 0].legend(fontsize=8)
axes[1, 0].grid(True, alpha=0.35)

axes[0, 1].plot(sol.t, T_air_heater.mean(axis=0), color="tomato", linewidth=2, label="Avg air")
axes[0, 1].set_title("Heater — Air Temperature")
axes[0, 1].set_ylabel("Temperature [°C]")
axes[0, 1].legend(fontsize=8)
axes[0, 1].grid(True, alpha=0.35)

axes[1, 1].plot(sol.t, T_water_heater[mid], color="steelblue", linewidth=2, label=f"Seg {mid+1}")
axes[1, 1].set_title("Heater — Water Temperature")
axes[1, 1].set_ylabel("Temperature [°C]")
axes[1, 1].set_xlabel("Time [s]")
axes[1, 1].legend(fontsize=8)
axes[1, 1].grid(True, alpha=0.35)

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