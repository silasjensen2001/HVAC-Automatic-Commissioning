import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

"""
linearization_comparison.py
────────────────────────────────────────────────────────────────────────────────
Perturbation study: nonlinear vs linearised air-segment derivatives.

For each heat-exchanger type (cooler / heater) and each of the three scalar
inputs (T_in, T_out, θ), we sweep a perturbation δ ∈ [-5, +5] °C around the
operating point while keeping the other two inputs fixed.  We then compare:

    f_nonlinear(op + δ)          — true derivative
    f_linear(op + δ) = f0        — constant (Taylor zeroth order)
                      + a·δT_in  — (only when T_in is perturbed)
                      + b·δT_out — (only when T_out is perturbed)
                      + c·δθ     — (only when θ is perturbed)

and plot the approximation error  ε = f_nonlinear − f_linear.

Run from the *same directory* as models.py:
    python linearization_comparison.py
────────────────────────────────────────────────────────────────────────────────
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root

# ── Import your existing model classes ───────────────────────────────────────
from models import NonlinearHeatExchanger

# ─────────────────────────────────────────────────────────────────────────────
#  Shared parameters — keep in sync with your main simulation
# ─────────────────────────────────────────────────────────────────────────────
PARAMS_COOLER = dict(
    type                  = "cooler",
    num_segments          = 5,
    num_pipes             = 10,
    gamma                 = 951.87,
    cross_area_water      = 0.000201 * 2,
    heat_exchanger_depth  = 0.06 * 2,
    heat_exchanger_width  = 0.5,
    heat_exchanger_height = 0.5,
    volume_flow_wet_air   = 0.72634,
    water_supply_T        = 4.0 + 273.15,
    Kvs                   = 1.6471,
)

PARAMS_HEATER = dict(
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

# Valve operating points — must match LinearHeatExchanger defaults
VALVE_OP = {"cooler": 0.353, "heater": 0.02}

# ─────────────────────────────────────────────────────────────────────────────
#  Helper: find equilibrium for a single NonlinearHeatExchanger
# ─────────────────────────────────────────────────────────────────────────────
def find_equilibrium(nl: NonlinearHeatExchanger, u_op: float, T_in_op: float) -> np.ndarray:
    x0 = np.full(nl.N, 15 + 273.15)
    res = root(lambda x: nl.derivatives(x, np.array([u_op]), np.array([T_in_op])), x0)
    if not res.success:
        raise RuntimeError(f"Equilibrium not found: {res.message}")
    residual = np.abs(nl.derivatives(res.x, np.array([u_op]), np.array([T_in_op]))).max()
    print(f"  [{nl.type}] Equilibrium OK  (max residual {residual:.2e})")
    return res.x


# ─────────────────────────────────────────────────────────────────────────────
#  Core analysis: perturb one variable, measure linearisation error
# ─────────────────────────────────────────────────────────────────────────────
def perturbation_study(params: dict, n_points: int = 200, delta_range: float = 5.0):
    """
    Returns a dict with keys:
        delta       — perturbation array [K]
        segments    — list of segment indices analysed (middle segment by default)
        T_in_op, T_out_op, theta_op — operating point values [K]
        errors      — dict with keys 'T_in', 'T_out', 'theta'
                      each is shape (n_segments, n_points) of ε values [K/s]
        derivs_nl   — same structure, nonlinear derivative at perturbed point
        derivs_lin  — same structure, linear approximation
        coeffs      — dict with 'a', 'b', 'c' per segment [K/s per K]
    """
    nl     = NonlinearHeatExchanger(**params)
    hx_type = params["type"]
    u_op   = VALVE_OP[hx_type]

    T_in_op = (nl.T_operational_in_cooler if hx_type == "cooler"
               else nl.T_operational_in_heater)

    x_eq   = find_equilibrium(nl, u_op, T_in_op)
    T_eq   = x_eq[:nl.K]      # air temperatures at eq
    theta_eq = x_eq[nl.K:]    # water temperatures at eq

    seg_fn = (nl._air_cooler_segment_derivative if hx_type == "cooler"
              else nl._air_heater_segment_derivative)

    delta   = np.linspace(-delta_range, delta_range, n_points)
    eps     = 1e-5

    # Analyse every segment so we can pick any for plotting
    all_errors  = {k: {} for k in range(nl.K)}
    all_nl      = {k: {} for k in range(nl.K)}
    all_lin     = {k: {} for k in range(nl.K)}
    all_coeffs  = {}

    for k in range(nl.K):
        T_in_k  = T_in_op
        T_out_k = T_eq[k]
        theta_k = theta_eq[k]

        f0 = seg_fn(T_in_k, T_out_k, theta_k)

        # Central-difference linearisation coefficients (identical to LinearHeatExchanger)
        a = (seg_fn(T_in_k + eps, T_out_k, theta_k) -
             seg_fn(T_in_k - eps, T_out_k, theta_k)) / (2 * eps)
        b = (seg_fn(T_in_k, T_out_k + eps, theta_k) -
             seg_fn(T_in_k, T_out_k - eps, theta_k)) / (2 * eps)
        c = (seg_fn(T_in_k, T_out_k, theta_k + eps) -
             seg_fn(T_in_k, T_out_k, theta_k - eps)) / (2 * eps)

        all_coeffs[k] = dict(a=a, b=b, c=c, f0=f0,
                             T_in=T_in_k, T_out=T_out_k, theta=theta_k)

        for var, label in [("T_in", "T_in"), ("T_out", "T_out"), ("theta", "theta")]:
            f_nl  = np.zeros(n_points)
            f_lin = np.zeros(n_points)

            for i, d in enumerate(delta):
                if var == "T_in":
                    f_nl[i]  = seg_fn(T_in_k + d, T_out_k, theta_k)
                    f_lin[i] = f0 + a * d
                elif var == "T_out":
                    f_nl[i]  = seg_fn(T_in_k, T_out_k + d, theta_k)
                    f_lin[i] = f0 + b * d
                else:  # theta
                    f_nl[i]  = seg_fn(T_in_k, T_out_k, theta_k + d)
                    f_lin[i] = f0 + c * d

            all_nl[k][var]     = f_nl
            all_lin[k][var]    = f_lin
            all_errors[k][var] = f_nl - f_lin

    return dict(
        delta=delta,
        T_in_op=T_in_op, T_out_op=T_eq, theta_op=theta_eq,
        errors=all_errors, derivs_nl=all_nl, derivs_lin=all_lin,
        coeffs=all_coeffs,
        K=nl.K,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Plotting
# ─────────────────────────────────────────────────────────────────────────────

# Three viridis colours, evenly spaced and perceptually distinct
_VIRIDIS  = plt.cm.viridis
VAR_COLORS = {
    "T_in":  _VIRIDIS(0.15),
    "T_out": _VIRIDIS(0.55),
    "theta": _VIRIDIS(0.88),
}
VAR_LABELS = {
    "T_in":  r"$\delta T_{\mathrm{in}}$",
    "T_out": r"$\delta T_{\mathrm{out}}$",
    "theta": r"$\delta\,\theta$",
}
VAR_LINESTYLE = {
    "T_in":  "-",    # solid
    "T_out": "-",    # solid
    "theta": ":",    # dotted
}
VAR_LINEWIDTH = {
    "T_in":  2.2,
    "T_out": 2.2,
    "theta": 2.8,   # thicker so dots stay visible
}
VAR_XLABEL = r"Perturbation $\delta$ [°C]"


def _style_ax(ax):
    """Apply clean white-background styling to an axes."""
    ax.set_facecolor("white")
    ax.tick_params(colors="#333333", labelsize=9)
    for sp in ax.spines.values():
        sp.set_edgecolor("#cccccc")
        sp.set_linewidth(0.8)
    ax.grid(True, color="#e5e5e5", linewidth=0.7, linestyle="--")


def _avg_error(res: dict, var: str) -> np.ndarray:
    """Return the segment-averaged linearisation error for one perturbation variable."""
    K = res["K"]
    return np.mean([res["errors"][k][var] for k in range(K)], axis=0)


def plot_comparison(res_cooler: dict, res_heater: dict) -> plt.Figure:
    """
    Single figure, two columns (Cooler | Heater).

    Each column shows three curves — one per perturbation variable
    (T_in, T_out, θ) — where each curve is the segment-averaged
    linearisation error  ε = f_nonlinear − f_linear.

    The y-scales are independent so the relative magnitudes within each
    heat exchanger are clearly readable.
    """
    fig, axes = plt.subplots(
        1, 2,
        figsize=(13, 5),
        sharey=False,          # independent y-scales
    )
    fig.patch.set_facecolor("white")

    datasets = [
        ("Cooler", res_cooler, axes[0]),
        ("Heater", res_heater, axes[1]),
    ]

    for title, res, ax in datasets:
        _style_ax(ax)
        delta = res["delta"]

        for var in ("T_in", "T_out", "theta"):
            err_avg = _avg_error(res, var)
            ax.plot(
                delta, err_avg,
                color=VAR_COLORS[var],
                linewidth=VAR_LINEWIDTH[var],
                linestyle=VAR_LINESTYLE[var],
                label=VAR_LABELS[var],
            )

        ax.axhline(0,  color="#aaaaaa", linewidth=0.9, linestyle="--", zorder=0)
        ax.axvline(0,  color="#aaaaaa", linewidth=0.9, linestyle="--", zorder=0)

        # Operating-point summary in subtitle
        # Use segment-average op-point temperatures
        K = res["K"]
        T_out_avg   = np.mean([res["coeffs"][k]["T_out"] for k in range(K)]) - 273.15
        theta_avg   = np.mean([res["coeffs"][k]["theta"] for k in range(K)]) - 273.15
        T_in_val    = res["coeffs"][0]["T_in"] - 273.15
        max_errs = {var: np.abs(_avg_error(res, var)).max() for var in ("T_in", "T_out", "theta")}
        subtitle = (
            f"op-point avg:  "
            f"$T_{{in}}$={T_in_val:.1f}°C  "
            f"$\\langle T_{{out}}\\rangle$={T_out_avg:.1f}°C  "
            f"$\\langle\\theta\\rangle$={theta_avg:.1f}°C"
        )
        ax.set_title(
            f"{title}\n" + subtitle,
            fontsize=11, color="#222222", pad=8, linespacing=1.6,
        )
        ax.set_xlabel(VAR_XLABEL, fontsize=10, color="#333333")
        ax.set_ylabel(
            r"Avg. segment error  $\varepsilon$ [K/s]",
            fontsize=10, color="#333333",
        )
        ax.legend(
            fontsize=9.5, framealpha=0.9,
            edgecolor="#cccccc", facecolor="white",
            loc="best",
        )

    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== Running perturbation study ===")
    print("\n[Cooler]")
    res_cooler = perturbation_study(PARAMS_COOLER, n_points=300, delta_range=10.0)

    print("\n[Heater]")
    res_heater = perturbation_study(PARAMS_HEATER, n_points=300, delta_range=10.0)

    print("\n=== Linearisation coefficients (middle segment) ===")
    for label, res in [("Cooler", res_cooler), ("Heater", res_heater)]:
        seg = res["K"] // 2
        c   = res["coeffs"][seg]
        print(f"\n  {label} (seg {seg+1})")
        print(f"    T_in_op  = {c['T_in']  - 273.15:.2f} °C")
        print(f"    T_out_op = {c['T_out'] - 273.15:.2f} °C")
        print(f"    theta_op = {c['theta'] - 273.15:.2f} °C")
        print(f"    a (∂f/∂T_in)  = {c['a']:+.6f} K/s per K")
        print(f"    b (∂f/∂T_out) = {c['b']:+.6f} K/s per K")
        print(f"    c (∂f/∂θ)     = {c['c']:+.6f} K/s per K")

    fig = plot_comparison(res_cooler, res_heater)

    print("\n=== Max absolute avg-segment error per variable ===")
    for label, res in [("Cooler", res_cooler), ("Heater", res_heater)]:
        print(f"\n  {label}")
        for var in ("T_in", "T_out", "theta"):
            err = _avg_error(res, var)
            print(f"    {var:6s}  max|ε| = {np.abs(err).max():.4e} K/s")
    fig.savefig("linearization_comparison.png", dpi=300,
                bbox_inches="tight", facecolor="white")
    print("\nSaved → linearization_comparison.png")
    plt.show()
