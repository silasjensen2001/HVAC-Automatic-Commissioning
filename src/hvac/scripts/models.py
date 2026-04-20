from abc import ABC, abstractmethod
import time
import numpy as np
import time
import scipy.io as sio
from scipy.optimize import root

# - - - - - - - - - - - - - - - - HVAC - - - - - - - - - - - - - - - -
class HVAC:
    def __init__(self, components: list):
        """
        Args:
            components: Ordered list of LinearHeatExchanger (or compatible) instances.
                        Air flows through them in series: components[0] → components[1] → ...
        """
  
        self.components = components
        self.total_states = sum(c.num_states for c in components)

        self.coordinate_shift = np.zeros((self.total_states,)) 

        self.A, self.B_u, self.B_d, self.C = self._assemble_system()

    def _export_state_space(self, file_path: str):
        sio.savemat(file_path, {"A": self.A, "B_u": self.B_u, "B_d": self.B_d, "x_shift": self.coordinate_shift, "C": self.C})

    def _assemble_system(self):
        """
        Assemble full block state-space from component instances.
        Each component must expose: A, B_u, B_d, C, Offset
        """
        
        A = np.zeros((self.total_states, self.total_states))
        B_u = np.zeros((self.total_states, len(self.components))) # Single input per component 
        B_d = np.zeros((self.total_states, 1))
        C = np.zeros((len(self.components), self.total_states))
        Offset = np.zeros((self.total_states, 1))

        # Local offset and last output for coupling between components
        offset = 0
        prev_C = None
        prev_offset = None
        prev_n = None

        for comp_idx, component in enumerate(self.components):
            n = component.num_states
            
            # Block diagonal self-dynamics
            A[offset:offset+n, offset:offset+n] = component.A
            
            # Series air coupling from previous component
            if prev_C is not None:
                coupling = component.B_d @ prev_C  # (n x n_prev)
                A[offset:offset+n, prev_offset:prev_offset+prev_n] = coupling
            
            # Input matrix
            B_u[offset:offset+n, comp_idx:comp_idx+1] += component.B_u
            
            # Disturbance matrix
            if offset == 0:
                B_d[offset:offset+n, :] += component.B_d  # only used for first component
            
            # Output matrix
            C[comp_idx, offset:offset+n] = component.C[0, :]

            # Offset
            Offset[offset:offset+n, :] += component.Offset
            
            prev_C = component.C
            prev_offset = offset
            prev_n = n
            offset += n

        self._compute_frame_shift(A, Offset)

        return A, B_u, B_d, C

    def _check_stability_and_rank(self, A: np.ndarray):
        # Check if all eigenvalues of A have negative real part
        eigenvalues = np.linalg.eigvals(A)
        if np.any(eigenvalues.real >= 0):
            raise ValueError("System is not stable. Eigenvalues:\n", eigenvalues)
        # Check if A is full rank
        if np.linalg.matrix_rank(A) < A.shape[0]:
            raise ValueError("System matrix A is not full rank. Rank:", np.linalg.matrix_rank(A))

    def _compute_frame_shift(self, A: np.ndarray, Offset: np.ndarray):
        self._check_stability_and_rank(A)
        self.coordinate_shift = np.linalg.solve(-A, Offset).flatten()

    def _to_original_frame(self, x_shifted: np.ndarray) -> np.ndarray:
        return x_shifted + self.coordinate_shift

    def _to_shifted_frame(self, x: np.ndarray) -> np.ndarray:
        return x - self.coordinate_shift

    def derivatives(self, x: np.ndarray, u: np.ndarray, d: np.ndarray) -> np.ndarray:
        z = self._to_shifted_frame(x)
        return self.A @ z + self.B_u @ u + self.B_d @ d

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

        self.T_operational_in_cooler = 28 + 273.15 
        self.T_operational_in_heater = 9.9 + 273.15
        #self.relative_humidity_in_system = 0.5
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
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._kwargs = kwargs  # store so we can spin up NonlinearHeatExchanger internally

        if self.type == "heater":
            self.valve_operation_point = 0.02 # [0-1] valve opening for water flow
            self.theta_return_operation_point = 26.611 + 273.15 # [K] return water temperature at operation point
        else:
            self.valve_operation_point = 0.95 # [0-1] valve opening for water flow
            self.theta_return_operation_point = 5.56611657 + 273.15 # [K] return water temperature at operation point
        
        # Central difference parameters for linearization of air dynamics
        self.eps = 1e-5
        self.equilibrium_initial_guess = 15 + 273.15

        self.A, self.B_u, self.B_d, self.Offset, self.C  = self._construct_matrix_state_space()

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
    
    def _find_equilibrium(self, nonlinear: "NonlinearHeatExchanger", u_op: np.ndarray, d_op: np.ndarray) -> np.ndarray:
        x0 = np.full(nonlinear.N, self.equilibrium_initial_guess)
        result = root(lambda x: nonlinear.derivatives(x, u_op, d_op), x0)
        if not result.success:
            raise ValueError(f"Equilibrium search failed: {result.message}")
        print(f"  Equilibrium found. Max residual: {np.abs(nonlinear.derivatives(result.x, u_op, d_op)).max():.2e}")
        return result.x

    def _construct_air_state_block(self):
        # Spin up nonlinear version with identical parameters
        nonlinear = NonlinearHeatExchanger(**self._kwargs)
                
        if self.type == "cooler":
            T_in_op = self.T_operational_in_cooler
        else:
            # Heater 
            T_in_op = self.T_operational_in_heater
            
        d_op = np.array([T_in_op])
        u_op = np.array([self.valve_operation_point])

        # Find equilibrium
        x_eq      = self._find_equilibrium(nonlinear, u_op, d_op)
        T_eq     = x_eq[:self.K]   # air temperatures at equilibrium, one per segment
        theta_eq = x_eq[self.K:]   # water temperatures at equilibrium, one per segment

        seg_deriv = (nonlinear._air_cooler_segment_derivative if self.type == "cooler"
                     else nonlinear._air_heater_segment_derivative)

        A_air      = np.zeros((self.K, self.K * 2))
        B_air      = np.zeros((self.K, 1))
        offset_air = np.zeros((self.K, 1))

        for k in range(self.K):
            T_out_op = T_eq[k]
            theta_op = theta_eq[k]

            f0 = seg_deriv(T_in_op, T_out_op, theta_op)

            # Central finite differences
            a = (seg_deriv(T_in_op + self.eps, T_out_op, theta_op) - seg_deriv(T_in_op - self.eps, T_out_op, theta_op)) / (2*self.eps)
            b = (seg_deriv(T_in_op, T_out_op + self.eps, theta_op) - seg_deriv(T_in_op, T_out_op - self.eps, theta_op)) / (2*self.eps)
            c = (seg_deriv(T_in_op, T_out_op, theta_op + self.eps) - seg_deriv(T_in_op, T_out_op, theta_op - self.eps)) / (2*self.eps)

            # Affine offset
            d_const = f0 - a*T_in_op - b*T_out_op - c*theta_op

            A_air[k, k]          = b   # ∂f/∂T_out — self damping
            A_air[k, self.K + k] = c   # ∂f/∂theta — water coupling
            B_air[k, 0]          = a   # ∂f/∂T_in  — disturbance gain
            offset_air[k, 0]     = d_const

            print(f"  Seg {k}: a={a:.6f}  b={b:.6f}  c={c:.6f}  d={d_const:.6f}")

        return A_air, B_air, offset_air

    def _construct_matrix_state_space(self):
        
        A = np.zeros((self.N, self.N)) # State matrix
        B_u = np.zeros((self.N, 1)) # Input matrix for valve position - Controllable input
        B_d = np.zeros((self.N, 1)) # Input matrix for air inlet temperature - Disturbance input
        C = np.zeros((1, self.N)) # Output matrix
        Offset = np.zeros((self.N, 1))

        A_air, B_air, offset_air = self._construct_air_state_block()
        A_water, B_water, offset_water = self._construct_water_state_block()

        # A
        ## Air dynamics
        A[:self.K, :] = A_air

        ## Water dynamics
        A[self.K:, :] = A_water
        
        # B 
        B_u[self.K:,0] = B_water.flatten()
        B_d[0:self.K, 0] = B_air.flatten()
        
        # Constant offset
        Offset = np.vstack([offset_air, offset_water])

        C[0, :self.K] = 1 / self.K # Average air temperature across segments as output

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
        heat_vapor_in = mass_flow_vapor_in * (self.c_pv * T_in + L)
        heat_vapor_out = mass_flow_vapor_out * (self.c_pv * T_out + L)
        heat_condensate = (mass_flow_vapor_in - mass_flow_vapor_out) * self.c_pc * T_out
        
        numerator = -newtons_cooling_term + advective_term_dry_air + heat_vapor_in - heat_vapor_out - heat_condensate
        
        # Denominator terms
        denominator = self.mass_dry_air * (self.c_pa + omega_out * self.c_pv + (self.c_pv * T_out + L) * domega_dT_out - domega_dT_out * self.c_pc * T_out)

        return numerator / denominator

    def _air_heater_segment_derivative(self, T_in: float, T_out: float, theta: float) -> float:
        # Heater specific assumptions 
        relative_humidity = self._saturation_pressure(T_in) / self._saturation_pressure(T_out)
        
        # omega
        omega = self._omega(T_out, relative_humidity)

        # Mass flows
        mass_flow_vapor = omega * self.mass_flow_dry_air
  
        # Numerator terms
        newtons_cooling_term = (self.gamma/self.delta_x) * (T_out - theta)
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
        else: # heater
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
    def __init__(self, volume_flow_rate=1, cross_section_area=2,
                 duct_length=10, num_segments=10):
        """
        Initialize the air duct model.

        Args:
            volume_flow_rate (float):   Volumetric flow rate [m³/s]
            cross_section_area (float): Cross-sectional area of the duct [m²]
            duct_length (float):        Length of the duct [m]
            num_segments (int):         Number of segments to discretize into
        """
        self.q_a   = volume_flow_rate
        self.A_a   = cross_section_area
        self.L     = duct_length
        self.K     = num_segments
        self.N     = self.K
        self.dz    = self.L / self.K
        self.alpha = self.q_a / (self.A_a * self.dz)

        self.initial_state = np.zeros(self.N)
        self.A, self.B = self._construct_state_space()

    @property
    def num_states(self):
        """Total number of states — 1 per segment."""
        return self.N

    def _construct_state_space(self):
        """
        Construct state-space matrices for dT/dt = A·T + B·u.

        Returns:
            A (ndarray): State matrix [N x N]
            B (ndarray): Input matrix [N x 1]
        """
        A = np.zeros((self.K, self.K))
        for k in range(self.K):
            A[k, k] = -self.alpha
            if k > 0:
                A[k, k - 1] = self.alpha

        B = np.zeros((self.K, 1))
        B[0, 0] = self.alpha
        return A, B

    def set_initial_temperature(self, temperature_C):
        """
        Set uniform initial temperature for all segments.

        Args:
            temperature_C (float): Initial temperature [°C]
        """
        self.initial_state = np.full(self.N, temperature_C + 273.15)

    def derivatives(self, T, u):
        """
        Compute dT/dt given current state and input.

        Args:
            T (ndarray): Current segment temperatures [K]
            u (float):   Inlet temperature [K]

        Returns:
            dTdt (ndarray): Temperature derivatives [K/s]
        """
        return self.A @ T + self.B.flatten() * u

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