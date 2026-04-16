import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from pathlib import Path
from models import LinearHeatExchanger, NonlinearHeatExchanger

# ── Shared parameters ─────────────────────────────────────────────────────────
params = dict(
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

linear_model    = LinearHeatExchanger(**params)
nonlinear_model = NonlinearHeatExchanger(**params)

data_dir = Path(__file__).resolve().parent.parent / "data"
data_dir.mkdir(parents=True, exist_ok=True)
type_label = params["type"].capitalize()
data_path = data_dir / f"Heatexchanger_{type_label}_both_linear_model.mat"

linear_model._export_state_space(data_path)

print(f"\nSaved linear model data to: {data_path}")


K = linear_model.K

# ── Initial conditions ────────────────────────────────────────────────────────
T_init     = np.full(K, 9.9 + 273.15)
theta_init = np.full(K, 9.9 + 273.15)
x0 = np.concatenate([T_init, theta_init])

# ── Inputs ────────────────────────────────────────────────────────────────────
T_in           = 9.9 + 273.15
valve_position = 0.02

def u_fn(t):
    return np.array([valve_position])

def d_fn(t):
    return np.array([T_in])

# ── Integrate both ────────────────────────────────────────────────────────────
t_end  = 500
t_eval = np.linspace(0, t_end, 10000)

sol_lin = solve_ivp(
    lambda t, x: linear_model.derivatives(x, u_fn(t), d_fn(t)),
    (0, t_end), x0, t_eval=t_eval,
    method="Radau", rtol=1e-6, atol=1e-8,
)

sol_nl = solve_ivp(
    lambda t, x: nonlinear_model.derivatives(x, u_fn(t), d_fn(t)),
    (0, t_end), x0, t_eval=t_eval,
    method="Radau", rtol=1e-6, atol=1e-8,
)

# ── Unpack ────────────────────────────────────────────────────────────────────
T_air_lin   = sol_lin.y[:K] - 273.15
T_water_lin = sol_lin.y[K:] - 273.15
T_air_nl    = sol_nl.y[:K]  - 273.15
T_water_nl  = sol_nl.y[K:]  - 273.15

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
colors = plt.cm.tab10(np.linspace(0, 0.5, K))

mid = K // 2

# — Air temperature — average only
axes[0].plot(sol_lin.t, T_air_lin.mean(axis=0), color="tomato",  linewidth=2,
             linestyle="-",  label="Lin avg")
axes[0].plot(sol_nl.t,  T_air_nl.mean(axis=0),  color="tomato",  linewidth=2,
             linestyle="--", label="NL  avg")

# — Water temperature — middle segment only
axes[1].plot(sol_lin.t, T_water_lin[mid], color="steelblue", linewidth=2,
             linestyle="-",  label=f"Lin seg {mid+1}")
axes[1].plot(sol_nl.t,  T_water_nl[mid],  color="steelblue", linewidth=2,
             linestyle="--", label=f"NL  seg {mid+1}")

axes[0].axhline(T_in - 273.15, color="black", linestyle=":", linewidth=1,
                label=f"Air inlet ({T_in - 273.15:.1f} °C)")
axes[0].set_ylabel("Air temperature [°C]", fontsize=12)
axes[0].set_title("Linear (—) vs Nonlinear (--) — Air Temperatures", fontsize=13)
axes[0].legend(fontsize=8, loc="center right", ncol=2)
axes[0].grid(True, alpha=0.35)

axes[1].set_xlabel("Time [s]", fontsize=12)
axes[1].set_ylabel("Water temperature [°C]", fontsize=12)
axes[1].set_title("Linear (—) vs Nonlinear (--) — Water Temperatures", fontsize=13)
axes[1].legend(fontsize=8, loc="center right", ncol=2)
axes[1].grid(True, alpha=0.35)

plt.tight_layout()
plt.show()

# ── Terminal summary ──────────────────────────────────────────────────────────
print("\n=== Final time temperatures ===")
print(f"  {'Segment':<10} {'Lin Air':>10} {'NL Air':>10} {'Lin Water':>12} {'NL Water':>12}")
print(f"  {'-'*56}")
for k in range(K):
    print(f"  {k+1:<10} {T_air_lin[k,-1]:>10.3f} {T_air_nl[k,-1]:>10.3f}"
          f" {T_water_lin[k,-1]:>12.3f} {T_water_nl[k,-1]:>12.3f}")