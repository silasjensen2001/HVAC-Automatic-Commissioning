import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from models import NonlinearHeatExchanger

# ── Instantiate cooler ────────────────────────────────────────────────────────
model = NonlinearHeatExchanger(
    type                       = "cooler",
    num_segments               = 5,
    num_pipes                  = 10,
    gamma                      = 951.87, # [W/K] product of heat transfer coefficient and area radiator
    cross_area_water           = 0.000201*2, # [m²] cross-sectional area for water flow
    heat_exchanger_depth       = 0.06*2,
    heat_exchanger_width       = 0.5,
    heat_exchanger_height      = 0.5,
    volume_flow_wet_air        = 0.72634,   # [m³/s]
    water_supply_T             = 4.0 + 273.15, # [K]
    Kvs                        = 1.6471 # [m³/h] valve flow coefficient at fully open
)

K = model.K   # number of segments

# ── Initial conditions ────────────────────────────────────────────────────────
T_init     = np.full(K, 23.0 + 273.15)   # air   [K]
theta_init = np.full(K, 23.0 + 273.15)   # water [K]
x0 = np.concatenate([T_init, theta_init])

# ── Inputs: constant throughout ──────────────────────────────────────────────
T_in           = 23.0 + 273.15   # [K]  warm humid air inlet
valve_position = 0.353              # [0–1]  valve opening for water flow

def u_fn(t):
    return np.array([valve_position])

def d_fn(t):
    return np.array([T_in])

# ── Integrate ─────────────────────────────────────────────────────────────────
t_end  = 30
t_eval = np.linspace(0, t_end, 1000)

def ode(t, x):
    return model.derivatives(x, u_fn(t), d_fn(t))

sol = solve_ivp(
    ode, (0, t_end), x0,
    t_eval=t_eval,
    method="Radau",
    rtol=1e-6,
    atol=1e-8,
)

# ── Unpack & convert K → °C ───────────────────────────────────────────────────
T_air   = sol.y[:K] - 273.15   # shape (K, n_time)
T_water = sol.y[K:] - 273.15   # shape (K, n_time)

# Reconstruct dynamic water inlet temperature theta_in(t)
theta_return = sol.y[2 * K - 1, :]  # last water segment [K]
theta_in = np.array([model._valve_model(valve_position, tr) for tr in theta_return])
theta_in_c = theta_in - 273.15

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

colors_air   = plt.cm.viridis(np.linspace(0.4, 0.9, K))
colors_water = plt.cm.viridis(np.linspace(0.4, 0.9, K))

# — Air temperature —
ax = axes[0]
for k in range(K):
    ax.plot(sol.t, T_air[k], color=colors_air[k],
            linewidth=2, label=f"Air seg {k+1}")
ax.plot(sol.t, T_air.mean(axis=0), color="black", linestyle="--",
        linewidth=1.5, label="Air avg")
ax.axhline(T_in - 273.15, color="black", linestyle=":", linewidth=1.0,
           label=f"Air inlet ({T_in - 273.15:.0f} °C)")
ax.set_ylabel("Air temperature  [°C]", fontsize=12)
ax.set_title("Nonlinear Cooler — Air Segment Temperatures", fontsize=13)
ax.legend(fontsize=9, loc="center right", ncol=2)
ax.grid(True, alpha=0.35)

# — Water temperature —
ax = axes[1]
for k in range(K):
    ax.plot(sol.t, T_water[k], color=colors_water[k],
            linewidth=2, label=f"Water seg {k+1}")
ax.plot(sol.t, theta_in_c, color="black", linestyle=":", linewidth=2.0,
        label="Water inlet θ_in(t)")
ax.set_xlabel("Time  [s]", fontsize=12)
ax.set_ylabel("Water temperature  [°C]", fontsize=12)
ax.set_title("Nonlinear Cooler — Water Segment Temperatures", fontsize=13)
ax.legend(fontsize=9, loc="center right", ncol=2)
ax.grid(True, alpha=0.35)

plt.tight_layout()
plt.show()

#fig.savefig("cooler_simulation.png", dpi=300)

# ── Terminal summary ──────────────────────────────────────────────────────────
print("\n=== Final segment temperatures ===")
print(f"  {'Segment':<10} {'Air [°C]':>12} {'Water [°C]':>12}")
print(f"  {'-'*36}")
for k in range(K):
    print(f"  {k+1:<10} {T_air[k, -1]:>12.8f} {T_water[k, -1]:>12.8f}")
print(f"\n  Air inlet (const)    : {T_in - 273.15:.1f} °C")
print(f"  Water supply (const) : {model.water_supply_T - 273.15:.1f} °C")
print(f"  Water inlet final    : {theta_in_c[-1]:.3f} °C")
print(f"  Water inlet range    : {theta_in_c.min():.3f} to {theta_in_c.max():.3f} °C")