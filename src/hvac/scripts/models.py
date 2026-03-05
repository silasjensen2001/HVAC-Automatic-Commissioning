import numpy as np


class AirDuctModel:
    def __init__(self, volume_flow_rate=1, cross_section_area=2,
                 duct_length=10, num_segments=10):
        """
        Initialize the air duct model.
        Args:
            volume_flow_rate (float): Volumetric flow rate [m³/s]
            cross_section_area (float): Cross-sectional area of the duct [m²]
            duct_length (float): Length of the duct [m]
            num_segments (int): Number of segments to discretize the duct into
        """
        self.q_a = volume_flow_rate
        self.A_a = cross_section_area
        self.L = duct_length
        self.K = num_segments           # number of segments
        self.N = self.K                 # number of states (1 per segment for pure air duct)
        self.dz = self.L / self.K
        self.alpha = self.q_a / (self.A_a * self.dz)

        self.initial_state = np.zeros(self.N)
        self.A, self.B = self._construct_state_space()

    @property
    def num_states(self):
        """Total number of states. 1 per segment for a pure air duct."""
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
                A[k, k-1] = self.alpha

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





