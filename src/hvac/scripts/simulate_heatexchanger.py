import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from models_nonlinear import NonlinearHeatExchanger

# ── Instantiate cooler ────────────────────────────────────────────────────────

model = NonlinearHeatExchanger(
    type                       = "heater",
    num_segments               = 5,
    num_pipes                  = 10,
    heat_transfer_coefficient  = 10.0,
    Area_radiator              = 13.2 /40,
    cross_area_water           = 0.000201,
    heat_exchanger_depth       = 0.06,
    heat_exchanger_width       = 0.5,
    heat_exchanger_height      = 0.5,
    volume_flow_wet_air        = 0.72634,   # [m³/s]
    volume_flow_water          = 0.000027,  # [m³/s]
)

K = model.K   # number of segments

# ── Initial conditions ────────────────────────────────────────────────────────
T_init     = np.full(K, 10.0 + 273.15)   # air   [K]
theta_init = np.full(K,  10.0 + 273.15)   # water [K]
x0 = np.concatenate([T_init, theta_init])

# ── Inputs: constant throughout ──────────────────────────────────────────────
T_in     = 10.0 + 273.15   # [K]  warm humid air inlet
theta_in =  66.0 + 273.15   # [K]  chilled water inlet

def u_fn(t):
    return np.array([T_in, theta_in])

# ── Integrate ─────────────────────────────────────────────────────────────────
t_end  = 80
t_eval = np.linspace(0, t_end, 1000)

def ode(t, x):
    return model.derivatives(x, u_fn(t))

sol = solve_ivp(
    ode, (0, t_end), x0,
    t_eval=t_eval,
    method="RK45",
    rtol=1e-6,
    atol=1e-8,
)

# ── Unpack & convert K → °C ───────────────────────────────────────────────────
T_air   = sol.y[:K] - 273.15   # shape (K, n_time)
T_water = sol.y[K:] - 273.15   # shape (K, n_time)

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

colors_air   = plt.cm.Oranges(np.linspace(0.4, 0.9, K))
colors_water = plt.cm.Blues  (np.linspace(0.4, 0.9, K))

# — Air temperature —
ax = axes[0]
for k in range(K):
    ax.plot(sol.t, T_air[k], color=colors_air[k],
            linewidth=2, label=f"Air seg {k+1}")
ax.axhline(T_in - 273.15, color="sienna", linestyle=":", linewidth=1.0,
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
ax.axhline(theta_in - 273.15, color="navy", linestyle=":", linewidth=1.0,
           label=f"Water inlet ({theta_in - 273.15:.0f} °C)")
ax.set_xlabel("Time  [s]", fontsize=12)
ax.set_ylabel("Water temperature  [°C]", fontsize=12)
ax.set_title("Nonlinear Cooler — Water Segment Temperatures", fontsize=13)
ax.legend(fontsize=9, loc="center right", ncol=2)
ax.grid(True, alpha=0.35)

plt.tight_layout()
plt.show()

# ── Terminal summary ──────────────────────────────────────────────────────────
print("\n=== Final segment temperatures ===")
print(f"  {'Segment':<10} {'Air [°C]':>12} {'Water [°C]':>12}")
print(f"  {'-'*36}")
for k in range(K):
    print(f"  {k+1:<10} {T_air[k, -1]:>12.3f} {T_water[k, -1]:>12.3f}")
print(f"\n  Air inlet (const)   : {T_in     - 273.15:.1f} °C")
print(f"  Water inlet (const) : {theta_in  - 273.15:.1f} °C")