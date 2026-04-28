import numpy as np
from scipy.io import loadmat
from pathlib import Path


class StateFeedbackController:
    """
    Args:
        plant:  HVAC instance — used for C matrix and coordinate_shift.
        K_x:    State feedback gain matrix,    shape (n_inputs, n_states).
        K_I:    Integral feedback gain matrix, shape (n_inputs, n_outputs).
        u_min:  Lower saturation limit for valve positions (default 0.0).
        u_max:  Upper saturation limit for valve positions (default 1.0).
    """

    def __init__(self, plant, K_x: np.ndarray, K_I: np.ndarray, N: np.ndarray, u_min: float = 0.0, u_max: float = 1.0):
        self.plant     = plant
        self.K_x       = K_x
        self.K_I       = K_I
        self.N         = N # Reference matrix
        self.u_min     = u_min
        self.u_max     = u_max
        self.n_outputs = plant.C.shape[0]
        self.x_I       = np.zeros(self.n_outputs)

    @classmethod
    def from_mat_files(cls, plant, K_x_path: Path, K_I_path: Path, N_path: Path, K_x_key: str = "Kx", K_I_key: str = "Ki", N_key: str = "N", **kwargs) -> "StateFeedbackController":
        K_x = loadmat(K_x_path)[K_x_key]
        K_I = loadmat(K_I_path)[K_I_key]
        N = loadmat(N_path)[N_key]
        return cls(plant, K_x, K_I, N, **kwargs)

    def reset(self):
        """Reset integrator state to zero."""
        self.x_I = np.zeros(self.n_outputs)

    def outputs(self, x: np.ndarray) -> np.ndarray:
        """Compute plant outputs y = C @ x."""
        return self.plant.C @ x

    def _raw_input(self, x: np.ndarray, x_I: np.ndarray, r: np.ndarray) -> np.ndarray:
        # Controller is designed around the shifted coordinates
        z = self.plant._to_shifted_frame(x) # Shift state
        r_shift = r - self.plant.C @ (self.plant.coordinate_shift)  # shift reference
        return self.K_x @ z + self.K_I @ x_I + self.N @ r_shift

    def compute_input(self, x: np.ndarray, x_I: np.ndarray, r: np.ndarray) -> np.ndarray:
        """
        Compute saturated control input.

        Args:
            x:   Physical state vector, shape (n_states,).
            x_I: Integrator state,      shape (n_outputs,).
            r:   Reference/setpoint,   shape (n_outputs,).

        Returns:
            u: Clipped valve positions, shape (n_inputs,).
        """
        return np.clip(self._raw_input(x, x_I, r), self.u_min, self.u_max)

    def integrator_derivative(self, x: np.ndarray, x_I: np.ndarray, r: np.ndarray) -> np.ndarray:
        """
        Compute integrator derivative with conditional anti-windup.

        Args:
            x:   Physical state vector,    shape (n_states,).
            x_I: Current integrator state, shape (n_outputs,).
            r:   Reference/setpoint,       shape (n_outputs,).

        Returns:
            dx_I: Integrator derivative, shape (n_outputs,).
        """
        error = self.outputs(x)- r
        #error[0] = -error[0]  # Invert error for cooler outlet temperature (we want it to be below the setpoint)
        u_raw = self._raw_input(x, x_I, r)

        anti_windup = np.ones(self.n_outputs)

        for i in range(self.n_outputs):
            saturated_high = u_raw[i] >= self.u_max and error[i] > 0
            saturated_low  = u_raw[i] <= self.u_min and error[i] < 0
            if saturated_high or saturated_low:
                anti_windup[i] = 0.0

        return anti_windup * error

    def controller_derivatives(self, r: np.ndarray, d) -> callable:
        """
        The augmented state vector is z = [x (n_states), x_I (n_outputs)].

        Args:
            r: Reference/setpoint vector, shape (n_outputs,).
            d: Callable d(t) returning disturbance vector.

        Returns:
            Callable ode(t, z) -> dz/dt.

        Example:
            z0  = np.concatenate([x0, np.zeros(controller.n_outputs)])
            sol = solve_ivp(controller.controller_derivatives(r=r, d=d), (0, t_end), z0, ...)
        """
        N = self.plant.total_states

        def ode(t, augmented_state):
            x   = augmented_state[:N] # physical state
            x_I = augmented_state[N:] # integrator state
            u    = self.compute_input(x, x_I, r) # saturated control input
            
            # Compute plant derivatives and integrator derivatives, then concatenate
            dx   = self.plant.derivatives(x, u, d(t))
            dx_I = self.integrator_derivative(x, x_I, r)
            return np.concatenate([dx, dx_I])

        return ode