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
t_end  = 250
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

# ── Unpack air temperatures only ──────────────────────────────────────────────
T_air_lin = sol_lin.y[:K] - 273.15
T_air_nl  = sol_nl.y[:K]  - 273.15

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 5))

ax.plot(sol_lin.t, T_air_lin.mean(axis=0), color="#4CAF50", linewidth=2,
        label="Linear (avg)")
ax.plot(sol_nl.t,  T_air_nl.mean(axis=0),  color="#7B2D8B", linewidth=2,
        label="Nonlinear (avg)")
ax.axhline(T_in - 273.15, color="gray", linestyle=":", linewidth=1.5,
           label=f"Air inlet ({T_in - 273.15:.1f} °C)")

ax.set_xlabel("Time [s]", fontsize=12)
ax.set_ylabel("Air temperature [°C]", fontsize=12)
ax.set_title("Heater: Linear vs Nonlinear — Air Temperature", fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.35)

plt.tight_layout()
plt.savefig(Path(__file__).with_suffix(".png"), dpi=300)
plt.show()

# ── Terminal summary ──────────────────────────────────────────────────────────
print("\n=== Final time air temperatures ===")
print(f"  {'Segment':<10} {'Lin Air':>10} {'NL Air':>10}")
print(f"  {'-'*32}")
for k in range(K):
    print(f"  {k+1:<10} {T_air_lin[k,-1]:>10.3f} {T_air_nl[k,-1]:>10.3f}")

# ── Error metrics: RMSE + MAE ─────────────────────────────────────────────────
err_air = T_air_lin - T_air_nl

rmse_air = np.sqrt(np.mean(err_air**2))
mae_air  = np.mean(np.abs(err_air))

print(f"\n=== Linear vs Nonlinear error (over full trajectory) ===")
print(f"  {'Metric':<12} {'Air':>12}")
print(f"  {'-'*26}")
print(f"  {'RMSE [°C]':<12} {rmse_air:>12.6f}")
print(f"  {'MAE  [°C]':<12} {mae_air:>12.6f}")

# ── Per-segment RMSE and MAE ──────────────────────────────────────────────────
print(f"\n=== Per-segment RMSE and MAE ===")
print(f"  {'Segment':<10} {'RMSE [°C]':>12} {'MAE [°C]':>12}")
print(f"  {'-'*36}")
for k in range(K):
    rmse_k = np.sqrt(np.mean(err_air[k]**2))
    mae_k  = np.mean(np.abs(err_air[k]))
    print(f"  {k+1:<10} {rmse_k:>12.6f} {mae_k:>12.6f}")