import io
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


_vir  = plt.cm.viridis
_plas = plt.cm.plasma
_cw   = plt.cm.coolwarm

COLORMAPS = {
    "viridis":  _vir,
    "plasma":   _plas,
    "coolwarm": _cw,
}

DEFAULT_COOLER = "#3b82f6"
DEFAULT_HEATER = "#ef4444"
DEFAULT_OTHER  = ["#10b981", "#f59e0b", "#8b5cf6", "#ec4899"]


TYPE_RANGES = {
    "viridis":  {"cooler": (0.05, 0.18), "heater": (0.80, 0.95), "other": (0.42, 0.58)},
    "plasma":   {"cooler": (0.05, 0.18), "heater": (0.80, 0.95), "other": (0.42, 0.58)},
    "coolwarm": {"cooler": (0.05, 0.20), "heater": (0.80, 0.95), "other": (0.45, 0.55)},
}


def _detect_type(label: str) -> str:
    l = label.lower()
    if "cool" in l: return "cooler"
    if "heat" in l: return "heater"
    return "other"


def _assign_colors(labels: list[str], scheme: str) -> list:
    if scheme == "default":
        other_idx = 0
        colors = []
        for label in labels:
            t = _detect_type(label)
            if t == "cooler":
                colors.append(DEFAULT_COOLER)
            elif t == "heater":
                colors.append(DEFAULT_HEATER)
            else:
                colors.append(DEFAULT_OTHER[other_idx % len(DEFAULT_OTHER)])
                other_idx += 1
        return colors

    cmap   = COLORMAPS.get(scheme, _vir)
    ranges = TYPE_RANGES.get(scheme, TYPE_RANGES["viridis"])

    # Group indices by type
    groups: dict[str, list[int]] = {"cooler": [], "heater": [], "other": []}
    for i, label in enumerate(labels):
        groups[_detect_type(label)].append(i)

    colors = [None] * len(labels)
    for typ, indices in groups.items():
        lo, hi = ranges[typ]
        n = len(indices)
        ts = [((lo + hi) / 2)] if n == 1 else list(np.linspace(lo, hi, n))
        for idx, t in zip(indices, ts):
            colors[idx] = cmap(t)
    return colors


DPI = 200


def _apply_style(ax):
    ax.set_facecolor("white")
    ax.grid(True, color="#dddddd", linewidth=0.8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#333333")
        spine.set_linewidth(0.8)
    ax.tick_params(colors="#333333", labelsize=10)
    ax.xaxis.label.set_color("#333333")
    ax.yaxis.label.set_color("#333333")
    ax.title.set_color("#333333")


def _tight_ylim(ax, pad: float = 0.04):
    """Set y limits from actual data lines only (ignores axhline/axvline spans)."""
    ys = []
    for line in ax.lines:
        ydata = np.asarray(line.get_ydata())
        # axhline stores exactly 2 identical values; skip those
        if ydata.size > 2:
            ys.append(ydata)
    if not ys:
        return
    all_y = np.concatenate(ys)
    lo, hi = np.nanmin(all_y), np.nanmax(all_y)
    span = hi - lo if hi > lo else 1.0
    ax.set_ylim(lo - pad * span, hi + pad * span)


def render_temperatures(data: dict, scheme: str, title_suffix: str = "") -> bytes:
    t         = np.array(data["t"])
    outputs   = data["outputs"]
    d_signal  = data.get("d_signal", [])

    labels = [s["label"] for s in outputs.values()]
    colors = _assign_colors(labels, scheme)

    fig, ax = plt.subplots(figsize=(11, 4), dpi=DPI)
    fig.patch.set_facecolor("white")

    if d_signal:
        ax.plot(t, d_signal, color="#94a3b8", linewidth=1.2, linestyle=":",
                label="Outdoor air (T_fresh)")

    for color, (_, series) in zip(colors, outputs.items()):
        ax.plot(t, series["y"], color=color, linewidth=2, label=series["label"])
        ax.axhline(series["ref"], color=color, linewidth=1.2, linestyle="--",
                   label=f'{series["label"]} ref ({series["ref"]:.1f} °C)')

    _apply_style(ax)
    _tight_ylim(ax)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Temperature (°C)")
    ttl = "Air Temperatures"
    if title_suffix:
        ttl += f" — {title_suffix}"
    ax.set_title(ttl, fontweight="bold")
    ax.legend(framealpha=0.9, fontsize=9, loc="upper right")
    fig.tight_layout()
    return _to_png(fig)


def render_valves(data: dict, scheme: str, title_suffix: str = "") -> bytes:
    t      = np.array(data["t"])
    valves = data["valves"]

    labels = [s["label"] for s in valves.values()]
    colors = _assign_colors(labels, scheme)

    fig, ax = plt.subplots(figsize=(11, 4), dpi=DPI)
    fig.patch.set_facecolor("white")

    for color, (_, series) in zip(colors, valves.items()):
        ax.plot(t, series["y"], color=color, linewidth=2, label=series["label"])

    ax.axhline(1.0, color="#ef4444", linewidth=1, linestyle=":", label="Max (1.0)")
    ax.axhline(0.0, color="#ef4444", linewidth=1, linestyle=":", label="Min (0.0)")
    ax.set_ylim(-0.05, 1.05)

    _apply_style(ax)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Opening (0–1)")
    ttl = "Valve Openings"
    if title_suffix:
        ttl += f" — {title_suffix}"
    ax.set_title(ttl, fontweight="bold")
    ax.legend(framealpha=0.9, fontsize=9, loc="upper right")
    fig.tight_layout()
    return _to_png(fig)


def render_humidity(data: dict, scheme: str, title_suffix: str = "") -> bytes:
    t        = np.array(data["t"])
    humidity = data.get("humidity", {})

    if not humidity:
        raise ValueError("Humidity data not available (nonlinear mode only)")

    labels = [s["label"] for s in humidity.values()]
    colors = _assign_colors(labels, scheme)

    fig, ax = plt.subplots(figsize=(11, 4), dpi=DPI)
    fig.patch.set_facecolor("white")

    for color, (_, series) in zip(colors, humidity.items()):
        ax.plot(t, series["y"], color=color, linewidth=2,
                label=f'{series["label"]} inlet')

    _apply_style(ax)
    _tight_ylim(ax)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Specific humidity (kg/kg dry air)")
    ttl = "Specific Humidity at Heat Exchanger Inlets"
    if title_suffix:
        ttl += f" — {title_suffix}"
    ax.set_title(ttl, fontweight="bold")
    ax.legend(framealpha=0.9, fontsize=9, loc="upper right")
    fig.tight_layout()
    return _to_png(fig)


def render_junctions(data: dict, scheme: str, title_suffix: str = "") -> bytes:
    t         = np.array(data["t"])
    junctions = data.get("junctions", {})

    if not junctions:
        raise ValueError("Junction data not available")

    n_junctions = len(junctions)
    fig, axes = plt.subplots(n_junctions, 2,
                             figsize=(13, 3.5 * n_junctions), dpi=DPI,
                             squeeze=False)
    fig.patch.set_facecolor("white")

    for row, (_, jd) in enumerate(junctions.items()):
        inlet_colors = _assign_colors(jd["inlet_labels"], scheme)
        outlet_color = "#f59e0b"

        ax_t, ax_h = axes[row]

        for color, src, lbl in zip(inlet_colors, jd["inlet_ids"], jd["inlet_labels"]):
            ax_t.plot(t, jd["inlet_temperatures"][src], color=color, linewidth=2, label=lbl)
            ax_h.plot(t, jd["inlet_specific_humidities"][src], color=color, linewidth=2)

        ax_t.plot(t, jd["outlet_temperatures"],       color=outlet_color, linewidth=2,
                  linestyle="--", label="Mixed outlet")
        ax_h.plot(t, jd["outlet_specific_humidities"], color=outlet_color, linewidth=2,
                  linestyle="--")

        _apply_style(ax_t); _tight_ylim(ax_t)
        _apply_style(ax_h); _tight_ylim(ax_h)
        ax_t.set_title(f'Junction: {jd["label"]} — Temperature', fontweight="bold")
        ax_h.set_title(f'Junction: {jd["label"]} — Humidity',    fontweight="bold")
        ax_t.set_xlabel("Time (s)"); ax_t.set_ylabel("Temperature (°C)")
        ax_h.set_xlabel("Time (s)"); ax_h.set_ylabel("Specific humidity (kg/kg)")
        ax_t.legend(framealpha=0.9, fontsize=9)

    fig.tight_layout()
    return _to_png(fig)


def _to_png(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=DPI)
    plt.close(fig)
    buf.seek(0)
    return buf.read()
