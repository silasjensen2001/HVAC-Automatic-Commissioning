from abc import ABC, abstractmethod
import time
import numpy as np
import time
import scipy.io as sio
from scipy.optimize import root

# - - - - - - - - - - - - - - - - HVAC - - - - - - - - - - - - - - - -
class HVAC:
    def __init__(self, nodes: list, T_fresh: float, mode: str = "linear", const_disturbance: float = None):
        """
        Args:
            nodes:   Ordered list of node dicts describing the air-flow graph.
                     Each dict must contain:
                       "id"           — unique string identifier
                       "type"         — "cooler" | "heater" | "airduct"
                       "config"       - configuration parameters
                       "input"        — id of upstream node (optional; defaults to serial order)       
            mode:    "linear"    — simulate with linearised state-space matrices.
                     "nonlinear" — simulate with full nonlinear ODEs.
                     The linear model (A, B_u, B_d, C, coordinate_shift) is always
                     built so it is available for controller design and .mat export.
        """
        if mode not in ("linear", "nonlinear"):
            raise ValueError(f"mode must be 'linear' or 'nonlinear', got {mode!r}")

        self.nodes             = nodes
        self.T_fresh           = T_fresh
        self.mode              = mode
        self.const_disturbance = const_disturbance

        T_in_map = self._propagate_operating_temps()

        self._lin_components = []
        for node in nodes:
            ntype  = node["type"]
            config = node.get("config", {})
            T_in   = T_in_map[node["id"]]
            if ntype in ("cooler", "heater"):
                T_out_target = config["T_out_target"]
                params = {k: v for k, v in config.items() if k != "T_out_target"}
                self._lin_components.append(
                    LinearHeatExchanger(T_in_op=T_in, T_out_target=T_out_target, **params)
                )
            elif ntype == "airduct":
                self._lin_components.append(AirDuctModel(**config))

        self.total_states     = sum(c.num_states for c in self._lin_components)
        self.coordinate_shift = np.zeros(self.total_states)

        # Must be set before _assemble_system, which uses both
        self._state_node_ids = [n["id"] for n in nodes if n["type"] != "junction"]
        self._inlet_sources  = self._build_inlet_sources()

        if self.const_disturbance is not None:
            self.A, self.B_u, self.C = self._assemble_system()
        else:
            self.A, self.B_u, self.B_d, self.C = self._assemble_system()

        self._check_controllability(self.A, self.B_u)

        # Nonlinear components — only built when needed
        if mode == "nonlinear":
            self._nl_components = []
            for node in nodes:
                ntype  = node["type"]
                config = node.get("config", {})
                if ntype in ("cooler", "heater"):
                    params = {k: v for k, v in config.items() if k != "T_out_target"}
                    self._nl_components.append(NonlinearHeatExchanger(**params))
                elif ntype == "airduct":
                    self._nl_components.append(AirDuctModel(**config))

    def _export_state_space(self, file_path: str):
        data = {
            "A": self.A,
            "B_u": self.B_u,
            "x_shift": self.coordinate_shift,
            "C": self.C,
        }

        if hasattr(self, "B_d"):
            data["B_d"] = self.B_d

        sio.savemat(file_path, data)

    def _build_inlet_sources(self) -> list:
        """
        For every state-bearing node (non-junction), describe where its air inlet comes from.
        Returns one descriptor per entry in _lin_components / _nl_components:
          ("external",)                     — d[0] (fresh air)
          ("node", node_id)                 — outlet of another state-bearing node
          ("junction", [(src_id, q), ...])  — flow-weighted mix; src_id is "external" or a node id
        """
        node_by_id  = {n["id"]: n for n in self.nodes}
        serial_pred = {self.nodes[i]["id"]: self.nodes[i-1]["id"]
                       for i in range(1, len(self.nodes))}

        sources = []
        for node in self.nodes:
            if node["type"] == "junction":
                continue

            input_id = node.get("input") or serial_pred.get(node["id"])

            if not input_id:
                sources.append(("external",))
            elif input_id == "external":
                sources.append(("external",))
            else:
                upstream = node_by_id[input_id]
                if upstream["type"] == "junction":
                    inputs = upstream.get("config", {}).get("inputs", [])
                    sources.append(("junction", inputs))
                else:
                    sources.append(("node", input_id))

        return sources

    def _propagate_operating_temps(self) -> dict:
        """
        Compute the air inlet temperature at the operating point for every node.

        Two-pass strategy:
          Pass 1 — HX T_outs are fixed (= T_out_target, independent of T_in).
          Pass 2 — Duct and junction T_outs are resolved recursively from their
                   upstream nodes.  Junctions mix multiple inputs by flow weight.
                   Cycle detection guards against invalid topologies.

        Returns {node_id: T_in_op}.
        """
        node_by_id  = {n["id"]: n for n in self.nodes}
        serial_pred = {self.nodes[i]["id"]: self.nodes[i-1]["id"]
                       for i in range(1, len(self.nodes))}

        # Pass 1: HX T_outs are known immediately from their setpoints.
        T_out = {"external": self.T_fresh}
        for node in self.nodes:
            if node["type"] in ("cooler", "heater"):
                T_out[node["id"]] = node["config"]["T_out_target"]

        # Pass 2: resolve ducts and junctions via memoised recursion.
        def resolve(nid, visiting=None):
            if nid in T_out:
                return T_out[nid]
            visiting = visiting or set()
            if nid in visiting:
                raise ValueError(f"Cycle detected resolving T_out for node '{nid}'.")
            visiting.add(nid)

            node   = node_by_id[nid]
            ntype  = node["type"]
            config = node.get("config", {})

            if ntype == "junction":
                inputs  = config.get("inputs", [])
                total_q = sum(q for _, q in inputs)
                T_out[nid] = sum(q * resolve(src, visiting) for src, q in inputs) / total_q
            else:  # airduct — transparent at steady state
                upstream   = node.get("input") or serial_pred.get(nid) or "external"
                T_out[nid] = resolve(upstream, visiting)

            return T_out[nid]

        for node in self.nodes:
            resolve(node["id"])

        # Compute T_in for each state-bearing node.
        T_in_map  = {}
        T_current = self.T_fresh
        for node in self.nodes:
            nid      = node["id"]
            input_id = node.get("input") or serial_pred.get(nid)

            if node["type"] == "junction":
                T_in_map[nid] = T_out[nid]   # junction has no state; T_in = mixed output
            else:
                tin = resolve(input_id) if input_id else T_current
                T_in_map[nid] = tin

            T_current = T_out[nid]

        return T_in_map

    def _assemble_system(self):
        """
        Assemble full block state-space using inlet_sources for air routing.

        Inlet types handled:
          "external"  → B_d contribution (fresh air disturbance)
          "node"      → off-diagonal A block (serial coupling from upstream outlet)
          "junction"  → weighted mix: external fraction → B_d,
                        recirculated fraction → off-diagonal A block (feedback)

        Components with is_actuated=False contribute no B_u column.
        Components with is_output=False contribute no C row.
        """
        n_actuated = sum(1 for c in self._lin_components if getattr(c, "is_actuated", True))
        n_outputs  = sum(1 for c in self._lin_components if getattr(c, "is_output",   True))

        A      = np.zeros((self.total_states, self.total_states))
        B_u    = np.zeros((self.total_states, n_actuated))
        B_d    = np.zeros((self.total_states, 1))
        C      = np.zeros((n_outputs, self.total_states))
        Offset = np.zeros((self.total_states, 1))

        # Build (offset, num_states, component) lookup keyed by node id
        node_offsets = {}
        off = 0
        for nid, comp in zip(self._state_node_ids, self._lin_components):
            node_offsets[nid] = (off, comp.num_states, comp)
            off += comp.num_states

        u_idx = 0
        y_idx = 0

        for nid, comp, source in zip(self._state_node_ids, self._lin_components, self._inlet_sources):
            off, n, _ = node_offsets[nid]

            A[off:off+n, off:off+n] = comp.A

            if source[0] == "external":
                B_d[off:off+n, :] += comp.B_d

            elif source[0] == "node":
                src_off, src_n, src_comp = node_offsets[source[1]]
                A[off:off+n, src_off:src_off+src_n] += comp.B_d @ src_comp.C

            elif source[0] == "junction":
                inputs  = source[1]
                total_q = sum(q for _, q in inputs)
                for src, q in inputs:
                    alpha = q / total_q
                    if src == "external":
                        B_d[off:off+n, :] += alpha * comp.B_d
                    else:
                        src_off, src_n, src_comp = node_offsets[src]
                        A[off:off+n, src_off:src_off+src_n] += alpha * comp.B_d @ src_comp.C

            if getattr(comp, "is_actuated", True):
                B_u[off:off+n, u_idx:u_idx+1] += comp.B_u
                u_idx += 1

            if getattr(comp, "is_output", True):
                C[y_idx, off:off+n] = comp.C[0, :]
                y_idx += 1

            Offset[off:off+n, :] += comp.Offset

        if self.const_disturbance is not None:
            Offset += B_d * self.const_disturbance
            self._compute_frame_shift(A, Offset)
            return A, B_u, C
        else:
            self._compute_frame_shift(A, Offset)
            return A, B_u, B_d, C

    def _check_stability_and_rank(self, A: np.ndarray):
        eigenvalues = np.linalg.eigvals(A)
        if np.any(eigenvalues.real >= 0):
            raise ValueError("System is not stable. Eigenvalues:\n", eigenvalues)
        if np.linalg.matrix_rank(A) < A.shape[0]:
            raise ValueError("System matrix A is not full rank. Rank:", np.linalg.matrix_rank(A))

    def _check_controllability(self, A: np.ndarray, B_u: np.ndarray):
        # PBH test: rank([λI - A, B]) = n for every eigenvalue λ of A.
        # Numerically stable for stiff systems unlike the matrix-power method.
        n = A.shape[0]
        unstable_uncontrollable = []
        stable_uncontrollable   = []
        for lam in np.linalg.eigvals(A):
            pbh = np.hstack([lam * np.eye(n) - A, B_u])
            if np.linalg.matrix_rank(pbh) < n:
                if lam.real >= 0:
                    unstable_uncontrollable.append(lam)
                else:
                    stable_uncontrollable.append(lam)
        if stable_uncontrollable:
            print(f"  [controllability] {len(stable_uncontrollable)} stable uncontrollable mode(s) "
                  f"(will decay naturally): {[f'{l.real:.3f}' for l in stable_uncontrollable]}")
        if unstable_uncontrollable:
            raise ValueError(
                f"System has {len(unstable_uncontrollable)} UNSTABLE uncontrollable mode(s) — "
                f"cannot stabilise: {unstable_uncontrollable}"
            )

    def _compute_frame_shift(self, A: np.ndarray, Offset: np.ndarray):
        self._check_stability_and_rank(A)
        self.coordinate_shift = np.linalg.solve(-A, Offset).flatten()

    def _to_original_frame(self, x_shifted: np.ndarray) -> np.ndarray:
        return x_shifted + self.coordinate_shift

    def _to_shifted_frame(self, x: np.ndarray) -> np.ndarray:
        return x - self.coordinate_shift

    def _nonlinear_derivatives(self, x: np.ndarray, u: np.ndarray, d: np.ndarray) -> np.ndarray:
        T_external = d[0]

        # Snapshot current outlet temperature for every state-bearing node
        outlet_T = {}
        off = 0
        for nid, comp in zip(self._state_node_ids, self._nl_components):
            n   = comp.num_states
            x_k = x[off:off+n]
            outlet_T[nid] = float(x_k[-1]) if isinstance(comp, AirDuctModel) \
                            else float(np.mean(x_k[:comp.K]))
            off += n

        def get_T_in(source):
            if source[0] == "external":
                return T_external
            elif source[0] == "node":
                return outlet_T[source[1]]
            else:  # junction
                inputs  = source[1]
                total_q = sum(q for _, q in inputs)
                return sum(
                    q * (T_external if src == "external" else outlet_T[src])
                    for src, q in inputs
                ) / total_q

        off   = 0
        parts = []
        u_idx = 0
        for comp, source in zip(self._nl_components, self._inlet_sources):
            n        = comp.num_states
            x_k      = x[off:off+n]
            T_air_in = get_T_in(source)
            if isinstance(comp, AirDuctModel):
                dxdt = comp.derivatives(x_k, T_air_in)
            else:
                dxdt  = comp.derivatives(x_k, u[u_idx:u_idx+1], np.array([T_air_in]))
                u_idx += 1
            parts.append(dxdt)
            off += n
        return np.concatenate(parts)
    
    def derivatives(self, x: np.ndarray, u: np.ndarray, d: np.ndarray) -> np.ndarray:
        if self.mode == "linear":
            z = self._to_shifted_frame(x)

            # If constant disturbance, fold it into the linear dynamics as an offset
            if self.const_disturbance is not None:
                return self.A @ z + self.B_u @ u
            else:
                return self.A @ z + self.B_u @ u + self.B_d @ d
        else:
            return self._nonlinear_derivatives(x, u, d)


# - - - - - - - - - - - - - - - - Heat Exchanger - - - - - - - - - - - - - - - -
class BaseHeatExchanger(ABC):
    """
    Abstract base class for heat exchanger models.
    Subclasses implement either a linear (matrix) or nonlinear (ODE) formulation.
    Both expose the same derivatives(x, u, d) interface so the simulator
    does not need to know which one it is running.

    State vector: x = [T_1, ..., T_K, θ_1, ..., θ_K]
        T_k   : air temperature in segment k   [K]
        θ_k   : water temperature in segment k [K]

    Control vector: u = [valve_position]
        valve_position : water valve opening    [0..1]

    Disturbance vector: d = [T_in]
        T_in  : air inlet temperature           [K]
    """
    is_actuated = True   # contributes a column to the global B_u
    is_output   = True   # contributes a row to the global C

    def __init__(
        self,
        type: str,
        num_segments: int,
        num_pipes: int,
        gamma: float,
        cross_area_water: float,
        heat_exchanger_depth: float,
        heat_exchanger_width: float,
        heat_exchanger_height: float,
        volume_flow_wet_air: float,
        water_supply_T: float,
        Kvs: float,
    ):
        # type of heat exchanger (e.g. "cooler" or "heater")
        self.type         = type
        
        self.K            = num_segments
        self.N            = 2 * self.K

        # Define segment dimensions
        segment_width = heat_exchanger_width / num_segments
        segment_height = heat_exchanger_height / num_pipes
        segment_depth = heat_exchanger_depth

        # Physical parameters
        self.cross_area_water = cross_area_water
        self.cross_area_wet_air = (segment_depth * segment_height) - cross_area_water
        self.delta_x = segment_width

        self.T_operational_in_cooler = 23 + 273.15 # Was 28
        self.T_operational_in_heater = 9.9 + 273.15
        self.relative_humidity_in_system = 0.832

        self.p = 101325 # [Pa] - Atmospheric pressure

        self.volume_flow_wet_air = volume_flow_wet_air / (num_segments * num_pipes)
        self.volume_flow_water = (Kvs / 3600) / num_pipes

        self.gamma = gamma / (num_segments * num_pipes) # product of heat_transfer_coefficient and area radiator 
        
        self.water_density = 1000 # [kg/m³] - Density of water at room temperature

        # Specific heat capacities - Assumed constant pressure at 26.85 degrees celsius [J/(kg * K)]
        self.c_pc = 4.184 * 1000 # Condensate
        self.c_pa = 1.005 * 1000 # Dry air 
        self.c_pv = 1.864 * 1000 # Vapor

        # Gas constant
        self.gas_constant = 287.056 # [m^2/(K * s^2)]

        # Valve parameters
        self.Kvs = Kvs # max flow rate at fully open valve [m³/h]
        self.water_inlet_flow = Kvs #Flow rate in [m³/h]
        self.Valve_position_max = 1
        self.pressure_differential = 1.0 # [Bar]
        self.water_supply_T = water_supply_T # [K] - Water supply temperature

        # Shared coupling coefficients
        self.Newton_coeff   = self._newton_coupling_coeff()


        print(f"Initialized {self.type} with {self.K} segments and {num_pipes} pipes.")
        
    def _check_valve_model(self):
        # Does valve model make sense?
        coefficient = (self.Kvs/self.water_inlet_flow) * np.sqrt(self.pressure_differential)
        if coefficient > 1:
            # Throw error
            raise ValueError(f"Valve model coefficient {coefficient:.2f} > 1. Cannot supply more flow, than the inlet is set to take")
         
    def _newton_coupling_coeff(self) -> float:
        return self.gamma /self.delta_x

    # Water segment is linear in both cooler and heater, so we can reuse the same function for both models
    def _water_segment_derivative(self, T: float, theta_in: float, theta_out: float) -> float:
        nc       = self.Newton_coeff * (1/(self.c_pc * self.water_density * self.cross_area_water))
        ac       = self.volume_flow_water / (self.cross_area_water * self.delta_x)

        return nc * T - (nc + ac) * theta_out + ac * theta_in


    @property
    def num_states(self) -> int:
        """Total number of states — 2 per segment (1 air + 1 water)."""
        return self.N

    @abstractmethod
    def derivatives(self, x: np.ndarray, u: np.ndarray, d: np.ndarray) -> np.ndarray:
        """
        Compute dx/dt given current state x, control input u, and disturbance d.

        Args:
            x (np.ndarray): State vector [T_1..T_K, θ_1..θ_K], shape (2K,)
            u (np.ndarray): Input vector [valve_position],          shape (1,)
            d (np.ndarray): Disturbance vector [T_in],              shape (1,)

        Returns:
            dxdt (np.ndarray): State derivatives,                 shape (2K,)
        """

class LinearHeatExchanger(BaseHeatExchanger):
    def __init__(self, T_in_op: float, T_out_target: float, **kwargs):
        super().__init__(**kwargs)
        self._kwargs = kwargs  # stored so we can spin up NonlinearHeatExchanger internally

        self.theta_return_operation_point = None  # set by _find_valve_for_setpoint
        self.valve_operation_point        = 0.3   # initial guess; overwritten below
        self.eps = 1e-5

        self.A, self.B_u, self.B_d, self.Offset, self.C = \
            self._construct_matrix_state_space(T_in_op, T_out_target)

    def _export_state_space(self, file_path: str):
        sio.savemat(file_path, {"A": self.A, "B_u": self.B_u, "B_d": self.B_d, "Offset": self.Offset, "C": self.C})

    def _check_valve_model(self):
        # Does valve model make sense?
        coefficient = (self.Kvs/self.water_inlet_flow) * np.sqrt(self.pressure_differential)
        if coefficient > 1:
            # Throw error
            raise ValueError(f"Valve model coefficient {coefficient:.2f} > 1. Cannot supply more flow, than the inlet is set to take")
    
    def _check_stability_and_rank(self, A: np.ndarray):
        # Check if all eigenvalues of A have negative real part
        eigenvalues = np.linalg.eigvals(A)
        if np.any(eigenvalues.real >= 0):
            raise ValueError("System is not stable. Eigenvalues:\n", eigenvalues)
        # Check if A is full rank
        if np.linalg.matrix_rank(A) < A.shape[0]:
            raise ValueError("System matrix A is not full rank. Rank:", np.linalg.matrix_rank(A))

    def _valve_model_linear(self, Valve_position: float, theta_return: float) -> float:
        valve_model_operation_point = self._valve_model(Valve_position=self.valve_operation_point, theta_return=self.theta_return_operation_point)
        k = (self.Kvs * np.sqrt(self.pressure_differential)) / (self.water_inlet_flow * self.Valve_position_max)

        return valve_model_operation_point + k * (self.water_supply_T - self.theta_return_operation_point) * (Valve_position - self.valve_operation_point) + (1-k * self.valve_operation_point) * (theta_return - self.theta_return_operation_point)

    def _valve_model(self, Valve_position: float, theta_return: float) -> float:
        # Valve model
        theta_inlet = (self.Kvs/self.water_inlet_flow) * (Valve_position/self.Valve_position_max) * np.sqrt(self.pressure_differential) * (self.water_supply_T - theta_return) + theta_return
        return theta_inlet
    
    def _construct_water_state_block(self):
        nc       = self.Newton_coeff * (1/(self.c_pc * self.water_density * self.cross_area_water))
        ac       = self.volume_flow_water / (self.cross_area_water * self.delta_x)

        # Valve model - Given as: valve_model_operation_point + valve_coefficient_1 * (Valve_position - self.valve_operation_point) + valve_coefficient_2 * (theta_return - self.theta_return_operation_point)
        valve_model_operation_point = self._valve_model(Valve_position=self.valve_operation_point, theta_return=self.theta_return_operation_point)
        k = (self.Kvs * np.sqrt(self.pressure_differential)) / (self.water_inlet_flow * self.Valve_position_max)
        valve_coefficient_1 = k * (self.water_supply_T - self.theta_return_operation_point)
        valve_coefficient_2 = (1-k * self.valve_operation_point)

        valve_offset = ac * (valve_model_operation_point - valve_coefficient_1 * self.valve_operation_point - valve_coefficient_2 * self.theta_return_operation_point)
        theta_1_return_coupling = ac * valve_coefficient_2
        valve_position_coefficient = ac * valve_coefficient_1

        # Shape (K, 2K), gives water dynamics in terms of both air and water states
        A_water = np.zeros((self.K, self.K * 2))
        for k in range(self.K):
            # Air to water coupling
            A_water[k, k] = nc

            # Self term
            A_water[k, self.K + k] = -(nc + ac)
            # Upwind advection
            if k > 0:
                A_water[k, self.K + k - 1] = ac

        # Direct couple theta_return to the first water segment through the valve model
        A_water[0, self.N - 1] = theta_1_return_coupling
        
        # B_water: inlet boundary condition θ_0_in drives the first water segment
        B_water = np.zeros((self.K, 1))
        B_water[0, 0] = valve_position_coefficient

        # No constant offset in water dynamics
        offset_water = np.zeros((self.K, 1))
        offset_water[0, 0] = valve_offset

        return A_water, B_water, offset_water
    
    def _find_valve_for_setpoint(self, nonlinear: "NonlinearHeatExchanger",
                                 T_in: float, T_out_target: float):
        """
        Given inlet temperature T_in and desired outlet temperature T_out_target,
        find the valve position u and full equilibrium state x such that:
            f(x, u, T_in) = 0   and   mean(x[:K]) = T_out_target   (mean of parallel air segments)
        """
        N, K = nonlinear.N, nonlinear.K

        def residual(xu):
            x, u = xu[:N], xu[N:N+1]
            dxdt = nonlinear.derivatives(x, u, np.array([T_in]))
            y    = np.mean(x[:K]) - T_out_target   # mean of parallel air segments matches C matrix
            return np.concatenate([dxdt, [y]])

        x0 = np.concatenate([
            np.linspace(T_in, T_out_target, K),   # air: ramp from inlet to target
            np.full(K, nonlinear.water_supply_T),  # water: start at supply temp
        ])
        result = root(residual, np.concatenate([x0, [0.3]]))
        if not result.success:
            raise ValueError(
                f"Valve search failed for T_in={T_in-273.15:.1f}°C → "
                f"T_out={T_out_target-273.15:.1f}°C: {result.message}"
            )
        x_eq = result.x[:N]
        u_eq = float(np.clip(result.x[N], 0.0, 1.0))
        print(f"  Valve position: {u_eq:.4f}  "
              f"(residual: {np.abs(residual(result.x)).max():.2e})")
        return x_eq, u_eq

    def _construct_air_state_block(self, T_in_op: float, T_out_target: float):
        nonlinear = NonlinearHeatExchanger(**self._kwargs)

        x_eq, u_eq = self._find_valve_for_setpoint(nonlinear, T_in_op, T_out_target)
        self.valve_operation_point        = u_eq
        T_eq     = x_eq[:self.K]
        theta_eq = x_eq[self.K:]
        self.theta_return_operation_point = theta_eq[-1]

        seg_deriv = (nonlinear._air_cooler_segment_derivative if self.type == "cooler"
                     else nonlinear._air_heater_segment_derivative)

        A_air      = np.zeros((self.K, self.K * 2))
        B_air      = np.zeros((self.K, 1))
        offset_air = np.zeros((self.K, 1))

        for k in range(self.K):
            T_out_op = T_eq[k]
            theta_op = theta_eq[k]

            f0 = seg_deriv(T_in_op, T_out_op, theta_op)
            a  = (seg_deriv(T_in_op + self.eps, T_out_op, theta_op) - seg_deriv(T_in_op - self.eps, T_out_op, theta_op)) / (2*self.eps)
            b  = (seg_deriv(T_in_op, T_out_op + self.eps, theta_op) - seg_deriv(T_in_op, T_out_op - self.eps, theta_op)) / (2*self.eps)
            c  = (seg_deriv(T_in_op, T_out_op, theta_op + self.eps) - seg_deriv(T_in_op, T_out_op, theta_op - self.eps)) / (2*self.eps)
            d_const = f0 - a*T_in_op - b*T_out_op - c*theta_op

            A_air[k, k]          = b   # ∂f/∂T_out — self damping
            A_air[k, self.K + k] = c   # ∂f/∂theta — water coupling
            B_air[k, 0]          = a   # ∂f/∂T_in  — disturbance gain
            offset_air[k, 0]     = d_const

            print(f"  Seg {k}: a={a:.6f}  b={b:.6f}  c={c:.6f}  d={d_const:.6f}")

        return A_air, B_air, offset_air

    def _construct_matrix_state_space(self, T_in_op: float, T_out_target: float):
        A      = np.zeros((self.N, self.N))
        B_u    = np.zeros((self.N, 1))
        B_d    = np.zeros((self.N, 1))
        C      = np.zeros((1, self.N))
        Offset = np.zeros((self.N, 1))

        A_air, B_air, offset_air     = self._construct_air_state_block(T_in_op, T_out_target)
        A_water, B_water, offset_water = self._construct_water_state_block()

        A[:self.K, :] = A_air
        A[self.K:, :] = A_water
        B_u[self.K:, 0]    = B_water.flatten()
        B_d[0:self.K, 0]   = B_air.flatten()
        Offset = np.vstack([offset_air, offset_water])
        C[0, :self.K] = 1 / self.K   # output = mean of parallel air segments

        return A, B_u, B_d, Offset, C
    
    def derivatives(self, x: np.ndarray, u: np.ndarray, d: np.ndarray) -> np.ndarray:
        return self.A @ x + self.B_u @ u + self.B_d @ d + self.Offset.flatten()

class NonlinearHeatExchanger(BaseHeatExchanger):
    """
    Nonlinear heat exchanger model for simulation.
    Water dynamics remain linear (exact).
    Air dynamics use the full nonlinear ODE including moisture terms.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Mass and mass flows of dry air
        self.mass_dry_air = self._mass_dry_air(self.T_operational_in_cooler, self.relative_humidity_in_system)
        self.mass_flow_dry_air = self._mass_flow_dry_air(self.T_operational_in_cooler, self.relative_humidity_in_system)

        self._check_valve_model()

    def _saturation_pressure(self, T: float) -> float:
        T_celsius = T - 273.15
        p_sat_kPa = 0.61121 * np.exp((18.678 - T_celsius/234.5) * (T_celsius / (257.14 + T_celsius)))
        p_sat = p_sat_kPa * 1000 #Convert to [Pa]
        return p_sat
    
    def _d_saturation_pressure_dT(self, T:float) -> float:
        T_celsius = T - 273.15
        g = (18.678-T_celsius/234.5) * (T_celsius / (257.14 + T_celsius))
        dg_dT = (-1/234.5) * (T_celsius/(257.14 + T_celsius)) + (18.678-T_celsius/234.5) * (257.14 /(257.14 + T_celsius)**2)
        d_saturation_pressure_dT_kPa = 0.61121 * np.exp(g) * dg_dT
        d_saturation_pressure_dT = d_saturation_pressure_dT_kPa * 1000 #Convert to [Pa]
        return d_saturation_pressure_dT
        
    def _partial_pressure_vapor(self, T: float, relative_humidity:float) -> float:
        saturation_pressure = self._saturation_pressure(T) # [Pa]
        partial_pressure_vapor = relative_humidity * saturation_pressure
        return partial_pressure_vapor

    def _domega_dT_out(self, T: float, relative_humidity: float):
        domega_dT_out = (0.622 * relative_humidity * self._d_saturation_pressure_dT(T) * (self.p - 2 * relative_humidity * self._saturation_pressure(T))) / (self.p - relative_humidity * self._saturation_pressure(T))**2
        return domega_dT_out

    def _omega(self, T:float, relative_humidity:float) -> float:
        return  0.622 * (self._partial_pressure_vapor(T, relative_humidity)/(self.p-self._partial_pressure_vapor(T, relative_humidity)))

    def _mass_dry_air(self, T: float, relative_humidity:float) -> float:
        partial_pressure_vapor = self._partial_pressure_vapor(T, relative_humidity)
        partial_pressure_dry_air = self.p - partial_pressure_vapor
        omega = self._omega(T, relative_humidity)
        mass_dry_air = (partial_pressure_dry_air * (self.delta_x * self.cross_area_wet_air * (1-omega)))/(self.gas_constant * T)
        return mass_dry_air

    def _mass_flow_dry_air(self, T: float, relative_humidity:float) -> float:

        partial_pressure_dry_air = self.p - self._partial_pressure_vapor(T, relative_humidity)
        omega = self._omega(T, relative_humidity)
        mass_flow_dry_air = (partial_pressure_dry_air * self.volume_flow_wet_air * (1-omega)) / (self.gas_constant * T)

        return mass_flow_dry_air

    def _air_cooler_segment_derivative(self, T_in: float, T_out: float, theta: float) -> float:
        # Cooler specific assumptions 
        relative_humidity = 1
        L = 2500.9 * 1000 # [J/kg]
        T_ref = 273.15 # [K]
                
        # omega
        omega_in = self._omega(T_in, self.relative_humidity_in_system)
        omega_out = self._omega(T_out, relative_humidity) # [Kg/Kg]
        domega_dT_out = self._domega_dT_out(T_out, relative_humidity)
        
        # Mass flows
        mass_flow_vapor_in = omega_in * self.mass_flow_dry_air
        mass_flow_vapor_out = omega_out * self.mass_flow_dry_air

        # Numerator terms
        newtons_cooling_term = self.Newton_coeff * (T_out - theta)
        advective_term_dry_air = self.mass_flow_dry_air * self.c_pa * (T_in - T_out)
        heat_vapor_in = mass_flow_vapor_in * (self.c_pv * (T_in - T_ref) + L)
        heat_vapor_out = mass_flow_vapor_out * (self.c_pv * (T_out - T_ref) + L)
        heat_condensate = (mass_flow_vapor_in - mass_flow_vapor_out) * self.c_pc * (T_out - T_ref)
        
        numerator = -newtons_cooling_term + advective_term_dry_air + heat_vapor_in - heat_vapor_out - heat_condensate
        
        # Denominator terms
        denominator = self.mass_dry_air * (self.c_pa + omega_out * self.c_pv + (self.c_pv * (T_out - T_ref) + L) * domega_dT_out - domega_dT_out * self.c_pc * (T_out - T_ref))
        return numerator / denominator

    def _air_heater_segment_derivative(self, T_in: float, T_out: float, theta: float) -> float:
        # Heater specific assumptions 
        relative_humidity = self._saturation_pressure(T_in) / self._saturation_pressure(T_out)
        
        # omega
        omega = self._omega(T_out, relative_humidity)

        # Mass flows
        mass_flow_vapor = omega * self.mass_flow_dry_air
  
        # Numerator terms
        newtons_cooling_term = self.Newton_coeff * (T_out - theta)
        advective_term_dry_air = self.mass_flow_dry_air * self.c_pa * (T_in - T_out)
        advective_term_vapor = mass_flow_vapor * self.c_pv * (T_in - T_out)

        numerator = -newtons_cooling_term + advective_term_dry_air + advective_term_vapor

        # Denominator terms
        denominator = self.mass_dry_air * (self.c_pa + omega * self.c_pv)

        return numerator / denominator

    def _valve_model(self, Valve_position: float, theta_return: float) -> float:
        # Valve model
        theta_inlet = (self.Kvs/self.water_inlet_flow) * (Valve_position/self.Valve_position_max) * np.sqrt(self.pressure_differential) * (self.water_supply_T - theta_return) + theta_return
        return theta_inlet

    def derivatives(self, x: np.ndarray, u: np.ndarray, d: np.ndarray) -> np.ndarray:
        """
        Compute dx/dt for the full nonlinear system.

        Args:
            x (np.ndarray): [T_1..T_K, θ_1..θ_K], shape (2K,)
            u (np.ndarray): [valve_position],           shape (1,)
            d (np.ndarray): [T_in],                     shape (1,)
        """
        T     = x[:self.K]
        theta = x[self.K:]
        T_in = d[0] # Air inlet temperature is treated as a disturbance input
        valve_position = u[0] # Valve position is the control input

        theta_in = self._valve_model(Valve_position=valve_position, theta_return=theta[-1])

        if self.type == "cooler":
            dT_dt = np.array([self._air_cooler_segment_derivative(T_in, T[k], theta[k]) for k in range(self.K)])
        else:
            dT_dt = np.array([self._air_heater_segment_derivative(T_in, T[k], theta[k]) for k in range(self.K)])

        dtheta_dt = np.array([self._water_segment_derivative(T[k],
                theta_in if k == 0 else theta[k - 1],
                theta[k],
            )
            for k in range(self.K)
        ])

        return np.concatenate([dT_dt, dtheta_dt])
    
# - - - - - - - - - - - - - - - - Airduct - - - - - - - - - - - - - - - -
class AirDuctModel:
    """
    Linear advection model for an air duct.
    Also serves as the assembly component for HVAC (both linear and nonlinear paths).

    No valve → is_actuated = False  (no column added to global B_u).
    No control output → is_output = False  (no row added to global C).
    C is still used internally for inter-component air coupling (last segment = outlet).
    """
    is_actuated = False
    is_output   = False

    def __init__(self, volume_flow_rate=1, cross_section_area=2,
                 duct_length=10, num_segments=10):
        self.q_a   = volume_flow_rate
        self.A_a   = cross_section_area
        self.L     = duct_length
        self.K     = num_segments
        self.N     = self.K
        self.dz    = self.L / self.K
        self.alpha = self.q_a / (self.A_a * self.dz)

        self.A, self.B_d = self._construct_state_space()
        self.B_u         = np.zeros((self.N, 1))
        self.C           = np.zeros((1, self.N))
        self.C[0, -1]    = 1.0          # last segment = air outlet
        self.Offset      = np.zeros((self.N, 1))

    @property
    def num_states(self):
        return self.N

    def _construct_state_space(self):
        A = np.zeros((self.K, self.K))
        for k in range(self.K):
            A[k, k] = -self.alpha
            if k > 0:
                A[k, k - 1] = self.alpha
        B_d = np.zeros((self.K, 1))
        B_d[0, 0] = self.alpha
        return A, B_d

    def derivatives(self, T, u):
        return self.A @ T + self.B_d.flatten() * u

# - - - - - - - - - - - - - - - - Junction - - - - - - - - - - - - - - - -
class Junction:
    """
    Stateless N-input mixing junction.

    Computes a mass-flow-weighted outlet temperature from multiple inlets.
    No states — purely algebraic at each timestep.
    """
    num_states = 0
    K          = 0

    def mix(self, inputs):
        """
        Compute mixed outlet temperature.

        Args:
            inputs (list of (float, float)): (flow_rate [m³/s], temperature [K]) pairs

        Returns:
            T_mixed (float): Mass-flow-weighted mixed temperature [K]
        """
        total_q = sum(q for q, _ in inputs)
        if total_q == 0:
            raise ValueError("Total flow rate into junction is zero.")
        return sum(q * T for q, T in inputs) / total_q

# - - - - - - - - - - - - - - - - Sink - - - - - - - - - - - - - - - -
class Sink:
    """
    Terminal flow sink. Absorbs a fixed flow rate. No states.
    Used as a bookkeeping node to confirm global flow conservation.
    """
    num_states = 0
    K          = 0

    def __init__(self, flow_rate):
        """
        Args:
            flow_rate (float): Flow rate absorbed by this sink [m³/s]
        """
        self.q = flow_rate