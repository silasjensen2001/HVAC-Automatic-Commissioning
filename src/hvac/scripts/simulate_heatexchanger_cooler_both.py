import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from pathlib import Path
import csv
from models import LinearHeatExchanger, NonlinearHeatExchanger

# ── Shared parameters ─────────────────────────────────────────────────────────
params = dict(
    type                       = "cooler",
    num_segments               = 5,
    num_pipes                  = 10,
    gamma                      = 951.87, # [W/K] product of heat transfer coefficient and area radiator
    cross_area_water           = 0.000201, # [m²] cross-sectional area for water flow
    heat_exchanger_depth       = 0.06,
    heat_exchanger_width       = 0.5,
    heat_exchanger_height      = 0.5,
    volume_flow_wet_air        = 0.72634,   # [m³/s]
    water_supply_T             = 4.0 + 273.15, # [K]
    Kvs                        = 1.6471 # [m³/h] valve flow coefficient at fully open
)

linear_model    = LinearHeatExchanger(**params)
nonlinear_model = NonlinearHeatExchanger(**params)

# Extract A, B, and offset matrix
A = linear_model.A
B = linear_model.B
offset = linear_model.Offset

print("=== Linear model matrices ===")
print("A matrix:")
print(A)
print("\nB matrix:")
print(B)
print("\nOffset vector:")
print(offset)

data_dir = Path(__file__).resolve().parent.parent / "data"
data_dir.mkdir(parents=True, exist_ok=True)
csv_path = data_dir / "heater_both_linear_model.csv"

with csv_path.open("w", newline="") as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(["matrix", "row", "col", "value"])

    for row_index in range(A.shape[0]):
        for col_index in range(A.shape[1]):
            writer.writerow(["A", row_index, col_index, A[row_index, col_index]])

    for row_index in range(B.shape[0]):
        for col_index in range(B.shape[1]):
            writer.writerow(["B", row_index, col_index, B[row_index, col_index]])

    for row_index, value in enumerate(np.ravel(offset)):
        writer.writerow(["offset", row_index, "", value])

print(f"\nSaved linear model data to: {csv_path}")

K = linear_model.K

# ── Initial conditions ────────────────────────────────────────────────────────
T_init     = np.full(K, 28 + 273.15)
theta_init = np.full(K, 28 + 273.15)
x0 = np.concatenate([T_init, theta_init])

# ── Inputs ────────────────────────────────────────────────────────────────────
T_in           = 28 + 273.15
valve_position = 0.95

def u_fn(t):
    return np.array([T_in, valve_position])

# ── Integrate both ────────────────────────────────────────────────────────────
t_end  = 30
t_eval = np.linspace(0, t_end, 10000)

sol_lin = solve_ivp(
    lambda t, x: linear_model.derivatives(x, u_fn(t)),
    (0, t_end), x0, t_eval=t_eval,
    method="Radau", rtol=1e-6, atol=1e-8,
)

sol_nl = solve_ivp(
    lambda t, x: nonlinear_model.derivatives(x, u_fn(t)),
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