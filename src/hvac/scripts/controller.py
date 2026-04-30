import numpy as np
from scipy.io import loadmat
from pathlib import Path
import scipy.linalg as la
import scipy.signal as signal

class StateFeedbackController:
    """
    Args:
        plant:  HVAC instance — used for C matrix and coordinate_shift.
        K_x:    State feedback gain matrix,    shape (n_inputs, n_states).
        K_I:    Integral feedback gain matrix, shape (n_inputs, n_outputs).
        u_min:  Lower saturation limit for valve positions (default 0.0).
        u_max:  Upper saturation limit for valve positions (default 1.0).
    """

    def __init__(self, plant, K_x: np.ndarray, K_I: np.ndarray, N: np.ndarray, M: np.ndarray, u_min: float = 0.0, u_max: float = 1.0):
        self.plant     = plant
        self.K_x       = K_x
        self.K_I       = K_I
        self.N         = N # Reference matrix
        self.M         = M
        self.u_min     = u_min
        self.u_max     = u_max
        self.n_outputs = plant.C.shape[0]
        self.x_I       = np.zeros(self.n_outputs)

    @classmethod
    def from_mat_files(cls, plant, K_x_path: Path, K_I_path: Path, N_path: Path, M_path: Path, K_x_key: str = "Kx", K_I_key: str = "Ki", N_key: str = "N", M_key: str = "M", **kwargs) -> "StateFeedbackController":
        K_x = loadmat(K_x_path)[K_x_key]
        K_I = loadmat(K_I_path)[K_I_key]
        N = loadmat(N_path)[N_key]
        M = loadmat(M_path)[M_key] if 'M_path' in kwargs else None
        return cls(plant, K_x, K_I, N, M, **kwargs)

    @classmethod
    def find_controller_gains(
        cls,
        plant,
        Q: np.ndarray | None = None,
        R: np.ndarray | None = None,
        u_min: float = 0.0,
        u_max: float = 1.0,
    ) -> "StateFeedbackController":
        A   = plant.A
        B_u = plant.B_u
        C   = plant.C

        n, m, p = A.shape[0], B_u.shape[1], C.shape[0]

        # ── Augmented system ──────────────────────────────────────────────────
        A_aug = np.block([[A,               np.zeros((n, p))],
                        [C,               np.zeros((p, p))]])
        B_aug = np.block([[B_u             ],
                        [np.zeros((p, m))]])

        n_aug = A_aug.shape[0]

        # ── LQR ───────────────────────────────────────────────────────────────
        Q = Q if Q is not None else np.eye(n_aug) * 1
        R = R if R is not None else np.eye(m)     * 8

        X   = la.solve_continuous_are(A_aug, B_aug, Q, R)
        K_aug = -la.solve(R, B_aug.T @ X)   # –R⁻¹ Bᵀ X

        K_x = K_aug[:, :n]   # (m × n)
        K_I = K_aug[:, n:]   # (m × p)

        F = A_aug + B_aug @ K_aug
        # ── Stability check ───────────────────────────────────────────────────
        eigs = np.linalg.eigvals(F)
        assert np.all(eigs.real < 0), (
            f"Closed-loop NOT stable! max Re(λ) = {np.max(eigs.real):+.6f}"
        )

        # Steady-state reference gains 
        SS_lhs = np.block([[A,  B_u             ],
                        [C,  np.zeros((p, m))]])
        SS_rhs = np.block([[np.zeros((n, p))],
                        [np.eye(p)        ]])

        N_xu = la.solve(SS_lhs, SS_rhs)
        N    = N_xu[n:, :] - K_x @ N_xu[:n, :]   # (m × p)

        # Anti windup gain
        alpha = 3.0
        desired_poles = np.sort(eigs.real)[-p:] * alpha
       
        result = signal.place_poles(
            np.zeros((p, p)),   # A of integrator subsystem (square!)
            -K_I.T,              # B: (p × m)  [note: K_I is (m×p), so K_I.T is (p×m)]
            desired_poles,            # p desired poles
        )

        M = result.gain_matrix.T   # (p × m)

        return cls(plant, K_x, K_I, N, M, u_min=u_min, u_max=u_max)

    @classmethod
    def cost_matrices(cls, plant, Q_scale: float = 1.0, R_scale: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
        n   = plant.A.shape[0]
        m   = plant.B_u.shape[1]
        p   = plant.C.shape[0]

        Q = np.eye(n + p) * Q_scale
        R = np.eye(m)     * R_scale
        return Q, R

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
        error = self.outputs(x) - r
        #error[0] = -error[0]  # Invert error for cooler outlet temperature (we want it to be below the setpoint)
        u_raw = self._raw_input(x, x_I, r)
        u_sat = np.clip(u_raw, self.u_min, self.u_max)

        # Anti-windup correction
        anti_windup = self.M @ (u_sat - u_raw)

        return error - anti_windup

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