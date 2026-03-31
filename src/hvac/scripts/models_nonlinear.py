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
        heat_transfer_coefficient: float,
        Area_radiator: float,
        cross_area_water: float,
        heat_exchanger_depth: float,
        heat_exchanger_width: float,
        heat_exchanger_height: float,
        volume_flow_wet_air: float,
        volume_flow_water: float,
        water_supply_T: float = 5 + 273.15
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
        
        self.T_operational_out_cooler = 10 + 273.15 #!!! 
        self.T_operational_in_cooler = 28 + 273.15 #!!!! Carsten said 28 degrees outside??????
        self.T_operational_out_heater = 20 + 273.15 #!!!
        self.T_operational_in_heater = self.T_operational_out_cooler #!!!

        self.p = 101325 # [Pa] - Atmospheric pressure

        self.volume_flow_wet_air = volume_flow_wet_air / (num_segments * num_pipes)
        self.volume_flow_water = volume_flow_water / num_pipes

        self.alpha        = heat_transfer_coefficient
        self.A_r          = Area_radiator / num_segments
        
        self.water_density = 1000 # [kg/m³] - Density of water at room temperature

        # Specific heat capacities - Assumed constant pressure at 26.85 degrees celsius [J/(kg * K)]
        self.c_pc = 4.180 * 1000 # Condensate
        self.c_pa = 1.005 * 1000 # Dry air 
        self.c_pv = 1.864 * 1000 # Vapor

        # Gas constant
        self.gas_constant = 287.056 # [m^2/(K * s^2)]

        # Valve parameters
        self.Kvs = 3 # max flow rate at fully open valve [m³/h]
        self.water_inlet_flow = volume_flow_water  * 3600 #Flow rate in [m³/h]
        self.Valve_position_max = 1
        self.pressure_differential = 0.75 # [Bar]
        self.water_supply_T = water_supply_T # [K] - Water supply temperature

        # Shared coupling coefficients
        self.Newton_coeff   = self._newton_coupling_coeff()


        print(f"Initialized {self.type} with {self.K} segments and {num_pipes} pipes.")
        

    def _newton_coupling_coeff(self) -> float:
        return self.alpha * self.A_r /self.delta_x

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
    """
    Linear state-space heat exchanger model.
    Intended for control design (LQR, MPC, pole placement, etc.)

        ẋ = A·x + B·u
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Build sub-blocks then assemble
        A_air,   B_air   = self._construct_air_state_block()
        A_water, B_water = self._construct_water_state_block()
        self.A, self.B = self._assemble(A_air, B_air, A_water, B_water)

    def _construct_air_state_block(self):
        """
        Air dynamics sub-block.

        Returns:
            A_air  : (K, 2K)
            B_air  : (2K, 1) — air inlet T_in at row 0
        """
        nc = self.Newton_coeff
        ac = self.volume_flow_water / (self.cross_area_water * self.delta_x)

        A_air = np.zeros((self.K, self.K * 2))
        for k in range(self.K):
            # Self term
            A_air[k, k]  = -(nc + ac)
            # Water to air coupling
            A_air[k, self.K + k] = nc
            # Upwind advection from previous segment
            if k > 0:
                A_air[k, k - 1] = ac

        B_air = np.zeros((self.K * 2, 1))
        B_air[0, 0] = ac

        return A_air, B_air

    def _construct_water_state_block(self):
        """
        Water dynamics sub-block.

        Returns:
            A_water  : (K, 2K)
            B_water  : (2K, 1) — water inlet θ_in at row K
        """
        nc = self.Newton_coeff
        ac = self.volume_flow_water / (self.cross_area_water * self.delta_x)

        A_water = np.zeros((self.K, self.K * 2))
        for k in range(self.K):
            # Air to water coupling
            A_water[k, k] = nc
            # Self term
            A_water[k, self.K + k] = -(nc + ac)
            # Upwind advection from previous segment
            if k > 0:
                A_water[k, self.K + k - 1] = ac

        B_water = np.zeros((self.K * 2, 1))
        B_water[self.K, 0] = ac

        return A_water, B_water

    def _assemble(self, A_air, B_air, A_water, B_water):
        """Assemble full (2K, 2K) A and (2K, 2) B from sub-blocks."""
        A = np.zeros((self.N, self.N))
        A[:self.K, :] = A_air
        A[self.K:, :] = A_water

        B = np.hstack([B_air, B_water])
        return A, B

    def derivatives(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        return self.A @ x + self.B @ u

class NonlinearHeatExchanger(BaseHeatExchanger):
    """
    Nonlinear heat exchanger model for simulation.
    Water dynamics remain linear (exact).
    Air dynamics use the full nonlinear ODE including moisture terms.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
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

    def _mass_flow_dry_air(self, T: float, relative_humidity:float) -> float:

        partial_pressure_dry_air = self.p - self._partial_pressure_vapor(T, relative_humidity)
        omega = self._omega(T, relative_humidity)
        mass_flow_dry_air = (partial_pressure_dry_air * self.volume_flow_wet_air * (1-omega)) / (self.gas_constant * self.T_operational_in_cooler)

        return mass_flow_dry_air

    def _air_cooler_segment_derivative(self, T_in: float, T_out: float, theta: float) -> float:
        # Cooler specific assumptions 
        relative_humidity = 1
        L = 2500.9 * 1000 # [J/kg]
        
        partial_pressure_vapor = self._partial_pressure_vapor(self.T_operational_out_cooler, relative_humidity)
        partial_pressure_dry_air = self.p - partial_pressure_vapor
        
        # omega
        omega_in = self._omega(T_in, relative_humidity)
        omega_out = self._omega(T_out, relative_humidity) # [Kg/Kg]
        domega_dT_out = self._domega_dT_out(T_out, relative_humidity)

        # Mass
        mass_dry_air = (partial_pressure_dry_air * (self.delta_x * self.cross_area_wet_air * (1-omega_out)))/(self.gas_constant * self.T_operational_out_cooler)

        # Mass flows
        mass_flow_dry_air = self._mass_flow_dry_air(self.T_operational_in_cooler, relative_humidity)
        mass_flow_vapor_in = omega_in * mass_flow_dry_air
        mass_flow_vapor_out = omega_out * mass_flow_dry_air

        # Numerator terms
        newtons_cooling_term = self.Newton_coeff * (T_out - theta)
        advective_term_dry_air = mass_flow_dry_air * self.c_pa * (T_in - T_out)
        heat_vapor_in = mass_flow_vapor_in * (self.c_pv * T_in + L)
        heat_vapor_out = mass_flow_vapor_out * (self.c_pv * T_out + L)
        heat_condensate = (mass_flow_vapor_in - mass_flow_vapor_out) * self.c_pc * T_out
        
        numerator = -newtons_cooling_term + advective_term_dry_air + heat_vapor_in - heat_vapor_out - heat_condensate
        
        # Denominator terms
        denominator = mass_dry_air * (self.c_pa + omega_out * self.c_pv + (self.c_pv * T_out + L) * domega_dT_out - domega_dT_out * self.c_pc * T_out)

        return numerator / denominator

    def _air_heater_segment_derivative(self, T_in: float, T_out: float, theta: float) -> float:
        # Heater specific assumptions 
        relative_humidity = self._saturation_pressure(T_in) / self._saturation_pressure(T_out)
        partial_pressure_vapor_out = self._partial_pressure_vapor(self.T_operational_out_heater, relative_humidity)
        partial_pressure_dry_air_out = self.p - partial_pressure_vapor_out
        
        # omega
        omega = self._omega(T_in, relative_humidity)

        # Mass
        mass_dry_air = (partial_pressure_dry_air_out * (self.delta_x * self.cross_area_wet_air * (1-omega)))/(self.gas_constant * self.T_operational_out_heater)

        # Mass flows
        mass_flow_dry_air = self._mass_flow_dry_air(self.T_operational_in_cooler, relative_humidity)
        mass_flow_vapor = omega * mass_flow_dry_air
        # Numerator terms
        newtons_cooling_term = ((self.alpha * self.A_r)/self.delta_x) * (T_out - theta)
        advective_term_dry_air = mass_flow_dry_air * self.c_pa * (T_in - T_out)
        advective_term_vapor = mass_flow_vapor * self.c_pv * (T_in - T_out)

        numerator = -newtons_cooling_term + advective_term_dry_air + advective_term_vapor

        # Denominator terms
        denominator = mass_dry_air * (self.c_pa + omega * self.c_pv)

        return numerator / denominator

    def _water_segment_derivative(self, T: float, theta_in: float, theta_out: float) -> float:
        nc       = self.Newton_coeff * (1/(self.c_pc * self.water_density * self.cross_area_water))
        ac       = self.volume_flow_water / (self.cross_area_water * self.delta_x)

        return nc * T - (nc + ac) * theta_out + ac * theta_in

    def _valve_model(self, Valve_position: float, theta_return: float) -> float:
        # Valve model
        theta_inlet = (self.Kvs/self.water_inlet_flow) * (Valve_position/self.Valve_position_max) * np.sqrt(self.pressure_differential) * (self.water_supply_T - theta_return) + theta_return
        return theta_inlet

    def _check_valve_model(self):
        # Does valve model make sense?
        coefficient = (self.Kvs/self.water_inlet_flow) * np.sqrt(self.pressure_differential)
        if coefficient > 1:
            # Throw error
            raise ValueError(f"Valve model coefficient {coefficient:.2f} > 1. Cannot supply more flow, than the inlet is set to take")
            
    
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