import numpy as np
import matplotlib.pyplot as plt
from models_nonlinear import NonlinearHeatExchanger

# ── Instantiate with realistic HVAC values ──────────────────────────────────
model = NonlinearHeatExchanger(
    num_segments               = 5,
    num_pipes                  = 10,
    heat_transfer_coefficient  = 70.0, #!
    Area_radiator              = 13.2,
    cross_area_water           = 0.000201,
    heat_exchanger_depth       = 0.06,
    heat_exchanger_width       = 0.5,
    heat_exchanger_height      = 0.5,
    volume_flow_wet_air        = 0.72634, # [m^3/s]
    volume_flow_water          = 0.000027, # [m^3/s]
)

# ── Known inputs (all in Kelvin) ─────────────────────────────────────────────
T_in  = 28.0 + 273.15
T_out = 10.0 + 273.15

theta_values_c = [5.0, 8.0, 17.73]  # °C

# ── Run checks ───────────────────────────────────────────────────────────────
print("=== Cooler segment derivative sanity check ===")
print(f"  T_in  : {T_in  - 273.15:.1f} °C")
print(f"  T_out : {T_out - 273.15:.1f} °C")
print(f"  {'theta (°C)':<12} {'dT/dt (K/s)':<16} {'sign':<8} {'monotonicity'}")
print(f"  {'-'*55}")

results = []
for theta_c in theta_values_c:
    theta = theta_c + 273.15
    dTdt  = model._air_cooler_segment_derivative(T_in, T_out, theta)
    results.append(dTdt)

    sign_ok = "✓" if dTdt < 0 else "✗ FAIL"

    mono_str = ""
    if len(results) > 1:
        mono_str = "✓" if results[-1] > results[-2] else "✗ FAIL"

    print(f"  {theta_c:<12.1f} {dTdt:<16.6f} {sign_ok:<8} {mono_str}")

# ── Assert physically consistent sign behavior and monotonicity ───────────────
# dT/dt is expected to increase with theta and can change sign at an equilibrium theta.
# Estimate that equilibrium from two nearby points.
theta_values_k = [t + 273.15 for t in theta_values_c]
dtheta = theta_values_k[1] - theta_values_k[0]
slope = (results[1] - results[0]) / dtheta
assert slope > 0, "Expected dT/dt to increase with theta"



theta_eq = theta_values_k[0] - results[0] / slope

for theta_k, dTdt in zip(theta_values_k, results):
    if theta_k < theta_eq:
        assert dTdt < 0, "Sign check failed below equilibrium: expected dT/dt < 0"
    elif theta_k > theta_eq:
        assert dTdt > 0, "Sign check failed above equilibrium: expected dT/dt > 0"

    # ── Plot dT/dt vs theta ───────────────────────────────────────────────────────
    theta_sweep_c = np.linspace(5.0, 28.0, 200)
    dTdt_sweep = [
        model._air_cooler_segment_derivative(T_in, T_out, t + 273.15)
        for t in theta_sweep_c
    ]

    fig, ax = plt.subplots(figsize=(8, 4))

    ax.plot(theta_sweep_c, dTdt_sweep, color="steelblue", linewidth=2, label="dT/dt")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.axvline(theta_eq - 273.15, color="tomato", linewidth=1.2, linestyle="--",
            label=f"Equilibrium θ ≈ {theta_eq - 273.15:.2f} °C")

    # Mark the three test points
    for theta_c, dTdt in zip(theta_values_c, results):
        ax.scatter(theta_c, dTdt, zorder=5, color="steelblue")

    ax.set_xlabel("Water temperature θ [°C]")
    ax.set_ylabel("dT/dt  [K/s]")
    ax.set_title(f"Cooler segment derivative  (T_in={T_in-273.15:.1f}°C, T_out={T_out-273.15:.1f}°C)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

assert all(results[i] > results[i-1] for i in range(1, len(results))), \
    "Monotonicity check failed: expected dT/dt to increase (less negative) as theta increases"



print(f"\n✓ Sign checks passed relative to equilibrium theta ≈ {theta_eq - 273.15:.2f} °C")
print("✓ Monotonicity check passed: warmer water increases dT/dt")