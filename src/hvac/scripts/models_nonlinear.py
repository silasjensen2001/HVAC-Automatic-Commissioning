from abc import ABC, abstractmethod
import numpy as np



class BaseHeatExchanger(ABC):
    """
    Abstract base class for heat exchanger models.
    Subclasses implement either a linear (matrix) or nonlinear (ODE) formulation.
    Both expose the same derivatives(x, u) interface so the simulator
    does not need to know which one it is running.

    State vector: x = [T_1, ..., T_K, θ_1, ..., θ_K]
        T_k   : air temperature in segment k   [K]
        θ_k   : water temperature in segment k [K]

    Input vector: u = [T_in, θ_in]
        T_in  : air inlet temperature           [K]
        θ_in  : water inlet temperature         [K]
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

        self.T_operational_out_cooler = 9.9 + 273.15 #!!! 
        self.T_operational_in_cooler = 28 + 273.15 #!!!! Carsten said 28 degrees outside??????
        self.T_operational_out_heater = 20 + 273.15 #!!!
        self.T_operational_in_heater = self.T_operational_out_cooler #!!!
        self.relative_humidity_in_system = 0.6 #!!! Assumed constant relative humidity in to the system

        self.p = 101325 # [Pa] - Atmospheric pressure

        self.volume_flow_wet_air = volume_flow_wet_air / (num_segments * num_pipes)
        self.volume_flow_water = (Kvs / 3600) / num_pipes

        self.gamma = gamma / num_segments # product of heat_transfer_coefficient and area radiator
        
        self.water_density = 1000 # [kg/m³] - Density of water at room temperature

        # Specific heat capacities - Assumed constant pressure at 26.85 degrees celsius [J/(kg * K)]
        self.c_pc = 4.180 * 1000 # Condensate
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
    def derivatives(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """
        Compute dx/dt given current state x and input u.

        Args:
            x (np.ndarray): State vector [T_1..T_K, θ_1..θ_K], shape (2K,)
            u (np.ndarray): Input vector [T_in, θ_in],           shape (2,)

        Returns:
            dxdt (np.ndarray): State derivatives,                 shape (2K,)
        """

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

class LinearHeatExchanger(BaseHeatExchanger):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.valve_operation_point = 0.02 # [0-1] valve opening for water flow
        self.theta_return_operation_point = 26.611 + 273.15 # [K] return water temperature at operation point

        # Construct the system
        self.A, self.B, self.Offset = self._construct_state_space()

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
        k = (self.Kvs * np.sqrt(self.pressure_differential)) / self.water_inlet_flow
        return k * (self.water_supply_T - self.theta_return_operation_point) * Valve_position + (1-k * self.valve_operation_point) * theta_return
    
    def _valve_model(self, Valve_position: float, theta_return: float) -> float:
        # Valve model
        theta_inlet = (self.Kvs/self.water_inlet_flow) * (Valve_position/self.Valve_position_max) * np.sqrt(self.pressure_differential) * (self.water_supply_T - theta_return) + theta_return
        return theta_inlet

    def _air_cooler_segment_derivative(self, T_in: float, T_out: float, theta: float) -> float:
        return 0

    def _air_heater_segment_derivative(self, T_in: float, T_out: float, theta: float, segment: int) -> float:
        coeffs = {0: (47.445533, -29594.354278, 29542.454314, 1260.826680),
                1: (47.483811, -29594.354278, 29542.454314, 1249.992126),
                2: (47.521761, -29594.354278, 29542.454314, 1239.250538),
                3: (47.559384, -29594.354278, 29542.454314, 1228.601363),
                4: (47.596683, -29594.354278, 29542.454314, 1218.043659),}
        a, b, c, d = coeffs[segment]
        return a * T_in + b * T_out + c * theta + d
    
    def _construct_water_state_block(self):
        nc       = self.Newton_coeff * (1/(self.c_pc * self.water_density * self.cross_area_water))
        ac       = self.volume_flow_water / (self.cross_area_water * self.delta_x)

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

        # B_water: inlet boundary condition θ_0_in drives the first water segment
        B_water = np.zeros((self.N, 1))
        B_water[self.K, 0] = ac

        # No constant offset in water dynamics
        offset_water = np.zeros((self.K, 1))

        return A_water, B_water, offset_water
    
    def _construct_air_state_block(self):
        """
        Construct the air sub-block of the state-space A matrix.

        Returns:
            A_air : np.ndarray, shape (K, 2K)
                Sub-block for air dynamics. Columns 0..K-1 are air-to-air coupling,
                columns K..2K-1 are air-to-water coupling.
                To assemble the full A matrix:
                    A[:K, :] = A_air
        """

        coeffs = {0: (47.445533, -29594.354278, 29542.454314, 1260.826680),
                1: (47.483811, -29594.354278, 29542.454314, 1249.992126),
                2: (47.521761, -29594.354278, 29542.454314, 1239.250538),
                3: (47.559384, -29594.354278, 29542.454314, 1228.601363),
                4: (47.596683, -29594.354278, 29542.454314, 1218.043659),}

        #Construct the air sub-block of the A matrix and offset vector for constant terms in the air dynamics
        A_air    = np.zeros((self.K, self.K * 2))
        B_air    = np.zeros((self.N, 1))
        offset_air = np.zeros((self.K, 1))

        for k in range(self.K):
            a, b, c, d = coeffs[k]
            A_air[k, k]          =  b
            A_air[k, self.K + k] =  c
            B_air[k, 0]          =  a
            offset_air[k, 0]     =  d

        return A_air, B_air, offset_air

    def _construct_state_space(self):
        
        A = np.zeros((self.N, self.N))
        B = np.zeros((self.N, 2))
        Offset = np.zeros((self.N, 2))


        A_air, B_air, offset_air = self._construct_air_state_block()
        A_water, B_water, offset_water = self._construct_water_state_block()

        # A
        ## Air dynamics
        A[:self.K, :] = A_air

        ## Water dynamics
        A[self.K:, :] = A_water
        
        # B 
        B = np.hstack([B_air, B_water])

        # Constant offset
        Offset = np.vstack([offset_air, offset_water])

        return A, B, Offset
    
    def derivatives(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
    
        # Valve model — get water inlet from valve position and return temperature
        theta_in = self._valve_model(Valve_position=u[1], theta_return=x[self.N - 1])
        u_linear = np.array([u[0], theta_in])  # [T_in, theta_in]
    
        return self.A @ x + self.B @ u_linear + self.Offset.flatten()
    
    def derivatives_old(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """
        Compute dx/dt for the full nonlinear system.

        Args:
            x (np.ndarray): [T_1..T_K, θ_1..θ_K], shape (2K,)
            u (np.ndarray): [T_in, valve_position],           shape (2,)
        """
        T     = x[:self.K]
        theta = x[self.K:]
        T_in, valve_position = u[0], u[1]

        theta_in = self._valve_model(Valve_position=valve_position, theta_return=theta[-1])

        if self.type == "cooler":
            dT_dt = np.array([self._air_cooler_segment_derivative(T_in, T[k], theta[k]) for k in range(self.K)])
        else: # heater
            dT_dt = np.array([self._air_heater_segment_derivative(T_in, T[k], theta[k], k) for k in range(self.K)])

        dtheta_dt = np.array([self._water_segment_derivative(T[k],
                theta_in if k == 0 else theta[k - 1],
                theta[k],
            )
            for k in range(self.K)
        ])

        return np.concatenate([dT_dt, dtheta_dt])

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

        print(f"diff evaluated in point {self._air_heater_segment_derivative(self.T_operational_in_heater, 27.53079433, 27.68566257)}")


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
        mass_flow_dry_air = (partial_pressure_dry_air * self.volume_flow_wet_air * (1-omega)) / (self.gas_constant * self.T_operational_in_cooler)

        return mass_flow_dry_air

    def _air_cooler_segment_derivative(self, T_in: float, T_out: float, theta: float) -> float:
        # Cooler specific assumptions 
        relative_humidity = 1
        L = 2500.9 * 1000 # [J/kg]
                
        # omega
        omega_in = self._omega(T_in, relative_humidity)
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

    def derivatives(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """
        Compute dx/dt for the full nonlinear system.

        Args:
            x (np.ndarray): [T_1..T_K, θ_1..θ_K], shape (2K,)
            u (np.ndarray): [T_in, valve_position],           shape (2,)
        """
        T     = x[:self.K]
        theta = x[self.K:]
        T_in, valve_position = u[0], u[1]

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