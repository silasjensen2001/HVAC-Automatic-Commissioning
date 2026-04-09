import numpy as np

class HeatExchangerModel:
    def __init__(
        self,
        model: str = "heater",
        volume_flow_rate: float = 1.0,
        cross_section_area: float = 2.0,
        num_segments: int = 5,
        heat_transfer_coefficient: float = 25.0,
    ):
        """
        Initialize a heat exchanger model.

        Args:
            model (str):                        Type of heat exchanger ("heater" or "cooler")
            volume_flow_rate (float):           Volumetric flow rate [m³/s]
            cross_section_area (float):         Cross-sectional area of the duct [m²]
            num_segments (int):                 Number of segments to discretize into
            heat_transfer_coefficient (float):  Convective heat transfer coefficient α [W/(m²·K)]
        """
        self.model = model
        self.K = num_segments

        # State vector layout: [T_1, ..., T_K, θ_1, ..., θ_K]
        # T_k: air temperature in segment k
        # θ_k: water temperature in segment k
        # Total states: 2K (K air + K water)
        self.N = 2 * self.K

        # Newton's law coupling coefficient

        self.heat_transfer_coefficient = heat_transfer_coefficient 
        self.Area_radiator = 1.0  # m², effective area for heat transfer per segment
        self.specific_heat_capacity = 1005.0  # J/(kg·K), for air
        self.density = 1.225  # kg/m³, for air
        self.Area_water = 0.5  # m², cross-sectional area for water flow
        self.segment_length = 1.0  # m, length of each segment


       # Find the coupling coefficient for Newton's law
        self.Newton_coeff = self.newtons_law_coupling()

        self.A_water, self.B_water = self._construct_water_state_block(
            Area_water=self.Area_water,
            segment_length=1.0,
            volume_flow_rate=volume_flow_rate,
            newton_coeff=self.Newton_coeff,
        )

        self.A_air, self.B_air = self._construct_air_state_block(
            newton_coeff=self.Newton_coeff,
        )

        self.A, self.B = self._construct_state_space(
            A_air=self.A_air,
            B_air=self.B_air,
            A_water=self.A_water,
            B_water=self.B_water,
        )

        print("A matrix:\n", self.A)
        print("B matrix:\n", self.B)


    @property
    def num_states(self) -> int:
        """Total number of states — 2 per segment (1 air + 1 water)."""
        return self.N

    def newtons_law_coupling(self) -> float:
        """
        Compute the heat transfer between air and water in each segment using Newton's law.

        Args:
            T_air (np.ndarray): Air temperatures in each segment [K], shape (K,)
            θ_water (np.ndarray): Water temperatures in each segment [K], shape (K,)
        """
        return (self.heat_transfer_coefficient * self.Area_radiator / (self.specific_heat_capacity * self.density * self.Area_water * self.segment_length))

    def _construct_water_state_block(
        self,
        Area_water: float,
        segment_length: float,
        volume_flow_rate: float,
        newton_coeff: float,
    ):
        """
        Construct the water sub-block of the state-space A matrix and B vector.

        Returns:
            A_water : np.ndarray, shape (K, 2K)
                Sub-block for water dynamics. Columns 0..K-1 are air coupling,
                columns K..2K-1 are water-to-water coupling.
            B_water : np.ndarray, shape (2K, 1)
                Input vector; non-zero only at the water inlet (row K).
        """
        
        Advection_coeff = volume_flow_rate / (Area_water * segment_length)

        # Shape (K, 2K), gives water dynamics in terms of both air and water states
        A_water = np.zeros((self.K, self.K * 2))
        for k in range(self.K):
            # Air to water coupling
            A_water[k, k] = newton_coeff

            # Self term
            A_water[k, self.K + k] = -(newton_coeff + Advection_coeff)
            # Upwind advection
            if k > 0:
                A_water[k, self.K + k - 1] = Advection_coeff

        # B_water: inlet boundary condition θ_0_in drives the first water segment
        B_water = np.zeros((self.K * 2, 1))
        B_water[self.K, 0] = Advection_coeff

        return A_water, B_water

    def _construct_air_state_block(self, newton_coeff: float):
        """
        Construct the air sub-block of the state-space A matrix.

        Returns:
            A_air : np.ndarray, shape (K, 2K)
                Sub-block for air dynamics. Columns 0..K-1 are air-to-air coupling,
                columns K..2K-1 are air-to-water coupling.
                To assemble the full A matrix:
                    A[:K, :] = A_air
        """

        #Construct the air sub-block of the A matrix
        A_air = np.zeros((self.K, self.K * 2))
        for k in range(self.K):
            # Air to water coupling
            A_air[k, k] = -newton_coeff
            A_air[k, self.K + k] = newton_coeff

        # Construct the input vector for the air inlet temperature
        B_air = np.zeros((self.K * 2, 1))
        B_air[0, 0] = 1.0  # Air inlet temperature drives the first air segment
        return A_air, B_air

    def _construct_state_space(self, A_air: np.ndarray, B_air: np.ndarray, A_water: np.ndarray, B_water: np.ndarray):
        """
        Construct the full state-space A matrix and B vector.

        Returns:
            A (np.ndarray): Full state matrix [2K x 2K]
            B (np.ndarray): Full input matrix [2K x 2]

        """
        A = np.zeros((self.N, self.N))
        B = np.zeros((self.N, 2))

        # A
        ## Air dynamics
        A[:self.K, :] = A_air

        ## Water dynamics
        A[self.K:, :] = A_water
        
        # B 
        B = np.hstack([B_air, B_water])

        return A, B
        

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
        self.A, self.B     = self._construct_state_space()

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
        return self.A @ T + self.B.flatten() * u +

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