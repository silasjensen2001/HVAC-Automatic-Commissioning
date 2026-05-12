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
    gamma                      = 951.87,
    cross_area_water           = 0.000201*2,
    heat_exchanger_depth       = 0.06*2,
    heat_exchanger_width       = 0.5,
    heat_exchanger_height      = 0.5,
    volume_flow_wet_air        = 0.72634,
    water_supply_T             = 4.0 + 273.15,
    Kvs                        = 1.6471
)

K = model.K

# ── Initial conditions ────────────────────────────────────────────────────────
T_init     = np.full(K, 23.0 + 273.15)
theta_init = np.full(K, 23.0 + 273.15)
x0 = np.concatenate([T_init, theta_init])

T_in = 29.0 + 273.15

t_end  = 80
t_eval = np.linspace(0, t_end, 10000)

# ── Valve cases ───────────────────────────────────────────────────────────────
valve_values = [0.0, 0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.0]
valve_cases  = [{"valve": vp, "label": f"{vp:.2f}"} for vp in valve_values]
colors       = plt.cm.viridis(np.linspace(0.10, 0.90, len(valve_values)))

# ── Run simulations & plot ────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))

for case, color in zip(valve_cases, colors):
    vp = case["valve"]

    def ode(t, x, vp=vp):
        return model.derivatives(x, np.array([vp]), np.array([T_in]))

    sol = solve_ivp(
        ode, (0, t_end), x0,
        t_eval=t_eval,
        method="Radau",
        rtol=1e-6,
        atol=1e-8,
    )

    T_air_avg = sol.y[:K].mean(axis=0) - 273.15
    ax.plot(sol.t, T_air_avg, color=color, linewidth=2.0, label=f"Valve {case['label']}")

ax.axhline(T_in - 273.15, color="dimgrey", linestyle=":", linewidth=1.2,
           label=f"Air inlet ({T_in - 273.15:.0f} °C)")

ax.set_xlabel("Time  [s]", fontsize=12)
ax.set_ylabel("Avg air temperature  [°C]", fontsize=12)
ax.set_title("Cooler — Average Air Temperature by Valve Position", fontsize=13)
ax.legend(fontsize=9, ncol=2, loc="lower right", framealpha=1.0)
ax.grid(True, alpha=0.35)
ax.set_xlim(0, t_end)

plt.tight_layout()
plt.savefig("cooler_valve_comparison.png", dpi=300, bbox_inches='tight')
plt.show()