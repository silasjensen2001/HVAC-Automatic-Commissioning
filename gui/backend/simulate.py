import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src/hvac/scripts")))

import numpy as np
from scipy.integrate import solve_ivp
from collections import defaultdict

from models import HVAC
from controller import StateFeedbackController, StateFeedbackControllerDisturbanceRejection


def _order_nodes(rf_nodes: list, rf_edges: list) -> list:
    """BFS from inlet nodes; any cycle residue is appended at the end."""
    in_degree: dict[str, int] = defaultdict(int)
    out_adj: dict[str, list] = defaultdict(list)
    for e in rf_edges:
        in_degree[e["target"]] += 1
        out_adj[e["source"]].append(e["target"])

    node_by_id = {n["id"]: n for n in rf_nodes}
    queue = [n for n in rf_nodes if in_degree[n["id"]] == 0]
    visited: set[str] = set()
    ordered: list = []

    while queue:
        node = queue.pop(0)
        if node["id"] in visited:
            continue
        visited.add(node["id"])
        ordered.append(node)
        for nid in out_adj[node["id"]]:
            in_degree[nid] -= 1
            if in_degree[nid] <= 0 and nid not in visited:
                queue.append(node_by_id[nid])

    for n in rf_nodes:
        if n["id"] not in visited:
            ordered.append(n)

    return ordered


def _propagate_fan_flows(rf_nodes: list, rf_edges: list) -> dict:
    """BFS from each fan; returns {node_id: flow_m3s} for downstream HVAC nodes.

    Stops propagation at:
    - junctions (they handle mixing themselves)
    - nodes whose outgoing edges lead to a junction (those nodes sit at branch
      endpoints and their flow rate should match the edge's flow_rate attribute,
      which the user sets explicitly — e.g. the return duct before the mixing point)
    """
    out_adj: dict[str, list] = defaultdict(list)
    for e in rf_edges:
        out_adj[e["source"]].append(e["target"])

    node_by_id = {n["id"]: n for n in rf_nodes}
    override_flows: dict[str, float] = {}

    def feeds_junction(nid: str) -> bool:
        return any(
            node_by_id.get(tid, {}).get("type") == "junction"
            for tid in out_adj[nid]
        )

    for n in rf_nodes:
        if n["type"] != "fan":
            continue
        flow = float(n["data"].get("volume_flow_rate", 1.0))
        queue = list(out_adj[n["id"]])
        visited: set[str] = set()
        while queue:
            nid = queue.pop(0)
            if nid in visited:
                continue
            visited.add(nid)
            tnode = node_by_id.get(nid)
            if not tnode:
                continue
            ttype = tnode["type"]
            if ttype in ("junction", "fan", "outdoor_air", "exhaust"):
                continue
            # Don't override nodes that directly feed a junction — their flow
            # rate is set by the edge's flow_rate attribute used for mixing.
            if feeds_junction(nid):
                continue
            override_flows[nid] = flow
            queue.extend(out_adj[nid])

    return override_flows


def _react_flow_to_hvac_nodes(rf_nodes: list, rf_edges: list) -> list:
    in_edges: dict[str, list] = defaultdict(list)
    for e in rf_edges:
        in_edges[e["target"]].append(e)

    fan_flows = _propagate_fan_flows(rf_nodes, rf_edges)
    ordered = _order_nodes(rf_nodes, rf_edges)
    hvac_nodes = []

    for rf_node in ordered:
        nid   = rf_node["id"]
        ntype = rf_node["type"]
        data  = rf_node["data"]
        ins   = in_edges[nid]

        # Skip non-HVAC nodes
        if ntype in ("outdoor_air", "fan", "exhaust"):
            continue

        if ntype == "junction":
            inputs = [
                (e["source"], float((e.get("data") or {}).get("flow_rate", 1.0)))
                for e in ins
                # skip edges from fan/outdoor_air sources (they were skipped from hvac_nodes)
                if next((n for n in rf_nodes if n["id"] == e["source"]), {}).get("type") not in ("outdoor_air", "fan")
            ]
            # Fall back to all edges if filtering removed everything
            if not inputs:
                inputs = [
                    (e["source"], float((e.get("data") or {}).get("flow_rate", 1.0)))
                    for e in ins
                ]
            hvac_nodes.append({
                "id": nid,
                "type": "junction",
                "config": {"inputs": inputs},
            })

        elif ntype in ("cooler", "heater"):
            vfr = fan_flows.get(nid, float(data["volume_flow_rate"]))
            config = {
                "type":                  ntype,
                "num_segments":          int(data["num_segments"]),
                "num_pipes":             int(data["num_pipes"]),
                "gamma":                 float(data["gamma"]),
                "cross_area_water":      float(data["cross_area_water"]),
                "heat_exchanger_depth":  float(data["heat_exchanger_depth"]),
                "heat_exchanger_width":  float(data["heat_exchanger_width"]),
                "heat_exchanger_height": float(data["heat_exchanger_height"]),
                "volume_flow_wet_air":   vfr,
                "water_supply_T":        float(data["water_supply_T"]) + 273.15,
                "Kvs":                   float(data["Kvs"]),
                "T_out_target":          float(data["T_out_target"]) + 273.15,
            }
            node = {"id": nid, "type": ntype, "config": config}
            # Find first non-fan/outdoor_air predecessor
            for e in ins:
                src_node = next((n for n in rf_nodes if n["id"] == e["source"]), None)
                if src_node and src_node["type"] not in ("outdoor_air", "fan"):
                    node["input"] = e["source"]
                    break
                elif src_node and src_node["type"] == "fan":
                    # Use the fan's predecessor as input
                    fan_ins = [fe for fe in (in_edges.get(src_node["id"]) or [])
                               if next((n for n in rf_nodes if n["id"] == fe["source"]), {}).get("type")
                               not in ("outdoor_air", "fan")]
                    if fan_ins:
                        node["input"] = fan_ins[0]["source"]
            hvac_nodes.append(node)

        elif ntype == "airduct":
            vfr = fan_flows.get(nid, float(data["volume_flow_rate"]))
            config = {
                "volume_flow_rate":   vfr,
                "cross_section_area": float(data["cross_section_area"]),
                "duct_length":        float(data["duct_length"]),
                "num_segments":       int(data["num_segments"]),
            }
            node = {"id": nid, "type": "airduct", "config": config}
            for e in ins:
                src_node = next((n for n in rf_nodes if n["id"] == e["source"]), None)
                if src_node and src_node["type"] not in ("outdoor_air", "fan"):
                    node["input"] = e["source"]
                    break
                elif src_node and src_node["type"] == "fan":
                    fan_ins = [fe for fe in (in_edges.get(src_node["id"]) or [])
                               if next((n for n in rf_nodes if n["id"] == fe["source"]), {}).get("type")
                               not in ("outdoor_air", "fan")]
                    if fan_ins:
                        node["input"] = fan_ins[0]["source"]
            hvac_nodes.append(node)

    return hvac_nodes


def get_system_info(rf_nodes: list, rf_edges: list, sim_params: dict) -> dict:
    outdoor_node = next((n for n in rf_nodes if n["type"] == "outdoor_air"), None)
    if outdoor_node:
        T_fresh_C = float(outdoor_node["data"].get("T_fresh", 23.0))
    else:
        T_fresh_C = float(sim_params.get("T_fresh", 23.0))
    T_fresh_K = T_fresh_C + 273.15

    hvac_nodes = _react_flow_to_hvac_nodes(rf_nodes, rf_edges)
    hvac = HVAC(nodes=hvac_nodes, T_fresh=T_fresh_K, mode="linear", const_disturbance=T_fresh_K)

    n = hvac.A.shape[0]
    m = hvac.B_u.shape[1]
    p = hvac.C.shape[0]
    Q_scale = float(sim_params.get("Q_scale", 5.0))
    R_scale = float(sim_params.get("R_scale", 800.0))

    return {
        "n_states": n,
        "n_inputs": m,
        "n_outputs": p,
        "q_size": n + p,
        "r_size": m,
        "q_diag_default": [Q_scale] * (n + p),
        "r_diag_default": [R_scale] * m,
    }


def run_simulation(rf_nodes: list, rf_edges: list, sim_params: dict) -> dict:
    # Extract outdoor air settings from the outdoor_air canvas node if present
    outdoor_node = next((n for n in rf_nodes if n["type"] == "outdoor_air"), None)
    if outdoor_node:
        od = outdoor_node["data"]
        T_fresh_C          = float(od.get("T_fresh", 23.0))
        disturbance_type   = str(od.get("disturbance_type", "constant"))
        disturbance_amp    = float(od.get("disturbance_amplitude", 3.0))
        disturbance_period = float(od.get("disturbance_period", 86400.0))
    else:
        T_fresh_C          = float(sim_params.get("T_fresh", 23.0))
        disturbance_type   = str(sim_params.get("disturbance_type", "constant"))
        disturbance_amp    = float(sim_params.get("disturbance_amplitude", 3.0))
        disturbance_period = float(sim_params.get("disturbance_period", 86400.0))

    T_fresh_K       = T_fresh_C + 273.15
    t_end           = float(sim_params.get("t_end", 200.0))
    Q_scale         = float(sim_params.get("Q_scale", 5.0))
    R_scale         = float(sim_params.get("R_scale", 800.0))
    model_mode      = str(sim_params.get("model_mode", "nonlinear"))
    controller_type = str(sim_params.get("controller_type", "lqr"))

    hvac_nodes = _react_flow_to_hvac_nodes(rf_nodes, rf_edges)

    hvac = HVAC(
        nodes=hvac_nodes,
        T_fresh=T_fresh_K,
        mode=model_mode,
        const_disturbance=T_fresh_K,
    )

    # Build Q and R matrices — advanced (per-element) or simple (scaled identity)
    use_advanced = bool(sim_params.get("use_advanced_qr", False))
    n_aug = hvac.A.shape[0] + hvac.C.shape[0]
    m_sys = hvac.B_u.shape[1]

    Q_diag_lqr = sim_params.get("Q_diag_lqr")
    R_diag_lqr = sim_params.get("R_diag_lqr")
    Q_diag_dr  = sim_params.get("Q_diag_dr")
    R_diag_dr  = sim_params.get("R_diag_dr")

    if use_advanced and Q_diag_lqr and R_diag_lqr:
        if len(Q_diag_lqr) != n_aug:
            raise ValueError(
                f"Advanced Q (LQR) has {len(Q_diag_lqr)} values but system needs {n_aug} "
                f"(n_states={hvac.A.shape[0]}, n_outputs={hvac.C.shape[0]}). "
                f"Click 'Get dimensions' to refresh the correct size."
            )
        if len(R_diag_lqr) != m_sys:
            raise ValueError(
                f"Advanced R (LQR) has {len(R_diag_lqr)} values but system needs {m_sys} inputs."
            )
        Q_lqr = np.diag(Q_diag_lqr)
        R_lqr = np.diag(R_diag_lqr)
    else:
        Q_lqr, R_lqr = StateFeedbackController.cost_matrices(hvac, Q_scale=Q_scale, R_scale=R_scale)

    if use_advanced and Q_diag_dr and R_diag_dr:
        if len(Q_diag_dr) != n_aug:
            raise ValueError(
                f"Advanced Q (DR) has {len(Q_diag_dr)} values but system needs {n_aug} "
                f"(n_states={hvac.A.shape[0]}, n_outputs={hvac.C.shape[0]}). "
                f"Click 'Get dimensions' to refresh the correct size."
            )
        if len(R_diag_dr) != m_sys:
            raise ValueError(
                f"Advanced R (DR) has {len(R_diag_dr)} values but system needs {m_sys} inputs."
            )
        Q_dr = np.diag(Q_diag_dr)
        R_dr = np.diag(R_diag_dr)
    else:
        Q_dr, R_dr = StateFeedbackController.cost_matrices(hvac, Q_scale=Q_scale, R_scale=R_scale)

    Q, R = (Q_dr, R_dr) if controller_type == "lmi" else (Q_lqr, R_lqr)

    if controller_type == "lmi":
        # When const_disturbance is set, _assemble_system folds B_d into the
        # operating-point offset and does NOT store it on the hvac object.
        # Build a second linear-only instance without const_disturbance so that
        # B_d is returned and stored, then attach it to the main hvac instance.
        hvac_lin = HVAC(
            nodes=hvac_nodes,
            T_fresh=T_fresh_K,
            mode="linear",
            const_disturbance=None,
        )
        hvac.B_d = hvac_lin.B_d  # A, B_u, C are identical; only B_d was missing
        ctrl = StateFeedbackControllerDisturbanceRejection.find_controller_gains(
            hvac, Q=Q, R=R
        )
    else:
        ctrl = StateFeedbackController.find_controller_gains(hvac, Q=Q, R=R)

    actuated_ids = [n["id"] for n in hvac_nodes if n["type"] in ("cooler", "heater")]
    id_to_label  = {n["id"]: n["data"].get("label", n["id"]) for n in rf_nodes}

    r = np.array([
        next(n["config"]["T_out_target"] for n in hvac_nodes if n["id"] == aid)
        for aid in actuated_ids
    ])

    N  = hvac.total_states
    x0 = np.full(N, T_fresh_K)
    aug = np.concatenate([x0, np.zeros(ctrl.n_outputs)])

    n_pts  = max(500, int(t_end * 5))
    t_eval = np.linspace(0, t_end, n_pts)

    if disturbance_type == "sinusoidal":
        def d_func(t):
            return np.array([T_fresh_K + disturbance_amp * np.sin(2 * np.pi * t / disturbance_period)])
    else:
        def d_func(t):
            return np.array([T_fresh_K])

    sol = solve_ivp(
        ctrl.controller_derivatives(r=r, d=d_func),
        (0, t_end), aug, t_eval=t_eval,
        method="Radau", rtol=1e-6, atol=1e-8,
    )

    if not sol.success:
        raise RuntimeError(f"ODE solver failed: {sol.message}")

    y_all  = hvac.C @ sol.y[:N, :]
    u_hist = np.array([
        ctrl.compute_input(sol.y[:N, k], sol.y[N:, k], r)[0]
        for k in range(sol.y.shape[1])
    ]).T

    outputs, valves = {}, {}
    for i, aid in enumerate(actuated_ids):
        label = id_to_label.get(aid, aid)
        outputs[aid] = {
            "label": label,
            "y":     (y_all[i] - 273.15).tolist(),
            "ref":   float(r[i] - 273.15),
        }
        valves[aid] = {
            "label": label,
            "y":     u_hist[i].tolist(),
        }

    cl_eigs = np.linalg.eigvals(hvac.A + hvac.B_u @ ctrl.K_x)
    cl_eigs_sorted = sorted(cl_eigs, key=lambda e: e.real)

    steady_state = {
        aid: {
            "label":  id_to_label.get(aid, aid),
            "temp_C": float(y_all[i, -1] - 273.15),
            "ref_C":  float(r[i] - 273.15),
            "valve":  float(u_hist[i, -1]),
        }
        for i, aid in enumerate(actuated_ids)
    }

    d_signal = np.array([d_func(t_)[0] - 273.15 for t_ in sol.t])

    # Compute inlet specific humidity and junction mixing time series (nonlinear only)
    humidity  = {}
    junctions = {}
    if model_mode == "nonlinear":
        hx_ids      = [n["id"] for n in hvac_nodes if n["type"] in ("cooler", "heater")]
        junction_ids = [n["id"] for n in hvac_nodes if n["type"] == "junction"]

        for aid in hx_ids:
            humidity[aid] = {"label": id_to_label.get(aid, aid), "y": []}

        # Initialise junction time series structure from first timestep
        first_jdata = hvac.compute_junction_states(sol.y[:N, 0], float(d_func(sol.t[0])[0]))
        for jid in junction_ids:
            jd = first_jdata.get(jid, {})
            inlets = jd.get("inputs", [])
            junctions[jid] = {
                "label":                    id_to_label.get(jid, jid),
                "inlet_ids":                [src for src, _ in inlets],
                "inlet_labels":             [id_to_label.get(src, src) for src, _ in inlets],
                "flows":                    [float(flow) for _, flow in inlets],
                "inlet_temperatures":       {src: [] for src, _ in inlets},
                "inlet_specific_humidities":{src: [] for src, _ in inlets},
                "outlet_temperatures":      [],
                "outlet_specific_humidities": [],
            }

        for k in range(sol.y.shape[1]):
            x_k   = sol.y[:N, k]
            T_ext = float(d_func(sol.t[k])[0])

            omegas = hvac.compute_inlet_specific_humidities(x_k, T_ext)
            for aid in hx_ids:
                humidity[aid]["y"].append(float(omegas.get(aid, 0.0)))

            jdata = hvac.compute_junction_states(x_k, T_ext)
            for jid in junction_ids:
                jd  = jdata.get(jid, {})
                jts = junctions[jid]
                jts["outlet_temperatures"].append(float(jd.get("outlet_temperature", 0.0)) - 273.15)
                jts["outlet_specific_humidities"].append(float(jd.get("outlet_specific_humidity", 0.0)))
                for src in jts["inlet_ids"]:
                    jts["inlet_temperatures"][src].append(float(jd.get("inlet_temperatures", {}).get(src, 0.0)) - 273.15)
                    jts["inlet_specific_humidities"][src].append(float(jd.get("inlet_specific_humidities", {}).get(src, 0.0)))

    return {
        "t":        sol.t.tolist(),
        "d_signal": d_signal.tolist(),
        "outputs":  outputs,
        "valves":   valves,
        "humidity":  humidity,
        "junctions": junctions,
        "metrics": {
            "K_I":              ctrl.K_I.tolist(),
            "K_x":              ctrl.K_x.tolist(),
            "N":                ctrl.N.tolist(),
            "M":                ctrl.M.tolist(),
            "cl_eigenvalues":   [{"re": float(e.real), "im": float(e.imag)} for e in cl_eigs_sorted],
            "condition_number": float(np.linalg.cond(hvac.A)),
            "steady_state":     steady_state,
            "actuated_ids":     actuated_ids,
        },
    }
