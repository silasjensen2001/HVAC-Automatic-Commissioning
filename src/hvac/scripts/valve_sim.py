import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from models_nonlinear import NonlinearHeatExchanger

# ── Instantiate heater ────────────────────────────────────────────────────────
model = NonlinearHeatExchanger(
    type                       = "heater",
    num_segments               = 5,
    num_pipes                  = 10,
    gamma                      = 715.44, # [W/K] product of heat transfer coefficient and area radiator
    cross_area_water           = 0.000201,
    heat_exchanger_depth       = 0.06,
    heat_exchanger_width       = 0.5,
    heat_exchanger_height      = 0.5,
    volume_flow_wet_air        = 0.72634,   # [m³/s]
    volume_flow_water          = 0.00096,  # [m³/s]
    water_supply_T             = 65 + 273.15, # [K]
    Kvs                        = 3.44 # [m³/h] valve flow coefficient at fully open
)

K    = model.K
T_in = 10.0 + 273.15  # [K]

t_end  = 60
t_eval = np.linspace(0, t_end, 5000)

valve_positions = np.arange(0.1, 1.01, 0.1)
colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(valve_positions)))

fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

for vp, color in zip(valve_positions, colors):
    print(f"Simulating valve position: {vp:.1f}")

    x0 = np.concatenate([
        np.full(K, 10.0 + 273.15),
        np.full(K, 10.0 + 273.15),
    ])

    sol = solve_ivp(
        lambda t, x: model.derivatives(x, np.array([T_in, vp])),
        (0, t_end), x0,
        t_eval=t_eval,
        method="RK45",
        rtol=1e-6,
        atol=1e-8,
    )

    avg_air   = (sol.y[:K]  - 273.15).mean(axis=0)
    avg_water = (sol.y[K:]  - 273.15).mean(axis=0)

    axes[0].plot(sol.t, avg_air,   color=color, linewidth=1.8, label=f"Vp = {vp:.1f}")
    axes[1].plot(sol.t, avg_water, color=color, linewidth=1.8, label=f"Vp = {vp:.1f}")

# — Air plot —
axes[0].axhline(T_in - 273.15, color="sienna", linestyle=":", linewidth=1.2,
                label=f"Air inlet ({T_in - 273.15:.0f} °C)")
axes[0].set_ylabel("Avg air temperature  [°C]", fontsize=12)
axes[0].set_title("Nonlinear Heater — Average Air Temperature per Valve Position", fontsize=13)
axes[0].legend(fontsize=9, loc="center right", ncol=2)
axes[0].grid(True, alpha=0.35)

# — Water plot —
axes[1].axhline(model.water_supply_T - 273.15, color="royalblue", linestyle=":",
                linewidth=1.2, label=f"Water supply ({model.water_supply_T - 273.15:.0f} °C)")
axes[1].set_ylabel("Avg water temperature  [°C]", fontsize=12)
axes[1].set_title("Nonlinear Heater — Average Water Temperature per Valve Position", fontsize=13)
axes[1].set_xlabel("Time  [s]", fontsize=12)
axes[1].legend(fontsize=9, loc="center right", ncol=2)
axes[1].grid(True, alpha=0.35)

plt.tight_layout()
plt.show()