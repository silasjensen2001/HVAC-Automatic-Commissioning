import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from pathlib import Path
from models import HVAC


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
    water_supply_T        = 8.0 + 273.15,
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

# ── Static valve openings ─────────────────────────────────────────────────────
u_cooler = 0.95
u_heater = 0.02
u_static = np.array([u_cooler, u_heater])

# ── Instantiate both plant models ─────────────────────────────────────────────
T_in = 28 + 273.15   # air inlet temperature [K]
hvac_nl  = HVAC(configs=[params_cooler, params_heater], mode="nonlinear", disturbance=T_in)
hvac_lin = HVAC(configs=[params_cooler, params_heater], mode="linear", disturbance=T_in)

# ── Dimensions ────────────────────────────────────────────────────────────────
K = hvac_nl._lin_components[0].K     # segments per heat exchanger (5)
N = hvac_nl.total_states              # 4K = 20

# ── Time ──────────────────────────────────────────────────────────────────────
t_end  = 300
t_eval = np.linspace(0, t_end, 2000)

# ── Initial conditions ────────────────────────────────────────────────────────
x0 = np.concatenate([
    np.full(K, 28 + 273.15),   # cooler air
    np.full(K, 28 + 273.15),   # cooler water
    np.full(K, 9.9 + 273.15),  # heater air
    np.full(K, 9.9 + 273.15),  # heater water
])

def d(t):
    return np.array([T_in])

# ── ODE wrappers with static input ───────────────────────────────────────────
# NOTE: adjust the method name below to match your HVAC class API,
#       e.g. hvac.f / hvac.rhs / hvac.derivatives / hvac.ode
def ode_nl(t, x):
    return hvac_nl.derivatives(x, u_static, d(t))

def ode_lin(t, x):
    return hvac_lin.derivatives(x, u_static, d(t))

# ── Simulate ──────────────────────────────────────────────────────────────────
common_ivp = dict(t_eval=t_eval, method="Radau", rtol=1e-6, atol=1e-8)

print("Simulating nonlinear model...")
sol_nl  = solve_ivp(ode_nl,  (0, t_end), x0, **common_ivp)

print("Simulating linear model...")
sol_lin = solve_ivp(ode_lin, (0, t_end), x0, **common_ivp)

# ── Unpack helper ─────────────────────────────────────────────────────────────
def unpack(sol):
    return dict(
        T_air_cooler   = sol.y[0:K]     - 273.15,
        T_water_cooler = sol.y[K:2*K]   - 273.15,
        T_air_heater   = sol.y[2*K:3*K] - 273.15,
        T_water_heater = sol.y[3*K:4*K] - 273.15,
    )

nl  = unpack(sol_nl)
lin = unpack(sol_lin)

mid = K // 2

# ── Plot helpers ──────────────────────────────────────────────────────────────
NL_COLOR  = "tomato"
LIN_COLOR = "steelblue"

def plot_pair(ax, t_nl, nl_data, t_lin, lin_data, title, ylabel, hline=None):
    ax.plot(t_nl,  nl_data,  color=NL_COLOR,  linewidth=2,   label="Nonlinear")
    ax.plot(t_lin, lin_data, color=LIN_COLOR, linewidth=2,
            linestyle="--", label="Linear")
    if hline is not None:
        val, label = hline
        ax.axhline(val, color="black", linestyle=":", linewidth=1, label=label)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.35)

# ── Figure ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(4, 2, figsize=(14, 18), sharex=True)

# Row 0 — Air outlet temperature (average across segments)
plot_pair(
    axes[0, 0], sol_nl.t, nl["T_air_cooler"].mean(axis=0),
    sol_lin.t, lin["T_air_cooler"].mean(axis=0),
    "Cooler — Air temperature (mean)", "Temperature [°C]",
    hline=(T_in - 273.15, f"Inlet ({T_in-273.15:.1f} °C)"),
)
plot_pair(
    axes[0, 1], sol_nl.t, nl["T_air_heater"].mean(axis=0),
    sol_lin.t, lin["T_air_heater"].mean(axis=0),
    "Heater — Air temperature (mean)", "Temperature [°C]",
)

# Row 1 — Water temperature at mid segment
plot_pair(
    axes[1, 0], sol_nl.t, nl["T_water_cooler"][mid],
    sol_lin.t, lin["T_water_cooler"][mid],
    f"Cooler — Water temperature (seg {mid+1})", "Temperature [°C]",
)
plot_pair(
    axes[1, 1], sol_nl.t, nl["T_water_heater"][mid],
    sol_lin.t, lin["T_water_heater"][mid],
    f"Heater — Water temperature (seg {mid+1})", "Temperature [°C]",
)

# Row 2 — Outlet temperature (last segment air)
plot_pair(
    axes[2, 0], sol_nl.t, nl["T_air_cooler"][-1],
    sol_lin.t, lin["T_air_cooler"][-1],
    "Cooler — Air outlet (seg K)", "Temperature [°C]",
)
plot_pair(
    axes[2, 1], sol_nl.t, nl["T_air_heater"][-1],
    sol_lin.t, lin["T_air_heater"][-1],
    "Heater — Air outlet (seg K)", "Temperature [°C]",
)

# Row 3 — Nonlinear vs linear error (absolute difference)
axes[3, 0].plot(sol_nl.t,
                np.abs(nl["T_air_cooler"].mean(axis=0) - lin["T_air_cooler"].mean(axis=0)),
                color="mediumpurple", linewidth=2)
axes[3, 0].set_title("Cooler — |NL − Lin| air temperature")
axes[3, 0].set_ylabel("Δ Temperature [°C]")
axes[3, 0].set_xlabel("Time [s]")
axes[3, 0].grid(True, alpha=0.35)

axes[3, 1].plot(sol_nl.t,
                np.abs(nl["T_air_heater"].mean(axis=0) - lin["T_air_heater"].mean(axis=0)),
                color="mediumpurple", linewidth=2)
axes[3, 1].set_title("Heater — |NL − Lin| air temperature")
axes[3, 1].set_ylabel("Δ Temperature [°C]")
axes[3, 1].set_xlabel("Time [s]")
axes[3, 1].grid(True, alpha=0.35)

plt.suptitle(
    f"HVAC open-loop: nonlinear vs. linear\n"
    f"u_cooler = {u_cooler:.2f}  |  u_heater = {u_heater:.2f}  |  T_in = {T_in-273.15:.1f} °C",
    fontsize=13,
)
plt.tight_layout()
plt.show()

# ── Terminal summary ──────────────────────────────────────────────────────────
print(f"\n=== Final temperatures — Nonlinear ===")
print(f"  {'Seg':<6} {'Cooler Air':>12} {'Cooler Water':>14} {'Heater Air':>12} {'Heater Water':>14}")
print(f"  {'-'*60}")
for k in range(K):
    print(f"  {k+1:<6}"
          f" {nl['T_air_cooler'][k,-1]:>12.3f}"
          f" {nl['T_water_cooler'][k,-1]:>14.3f}"
          f" {nl['T_air_heater'][k,-1]:>12.3f}"
          f" {nl['T_water_heater'][k,-1]:>14.3f}")

print(f"\n=== Final temperatures — Linear ===")
print(f"  {'Seg':<6} {'Cooler Air':>12} {'Cooler Water':>14} {'Heater Air':>12} {'Heater Water':>14}")
print(f"  {'-'*60}")
for k in range(K):
    print(f"  {k+1:<6}"
          f" {lin['T_air_cooler'][k,-1]:>12.3f}"
          f" {lin['T_water_cooler'][k,-1]:>14.3f}"
          f" {lin['T_air_heater'][k,-1]:>12.3f}"
          f" {lin['T_water_heater'][k,-1]:>14.3f}")

print(f"\n=== Max absolute deviation at t_end ===")
print(f"  Cooler air (mean):  {np.abs(nl['T_air_cooler'].mean(axis=0)[-1] - lin['T_air_cooler'].mean(axis=0)[-1]):.4f} °C")
print(f"  Heater air (mean):  {np.abs(nl['T_air_heater'].mean(axis=0)[-1] - lin['T_air_heater'].mean(axis=0)[-1]):.4f} °C")