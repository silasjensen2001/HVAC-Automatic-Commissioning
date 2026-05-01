import numpy as np
from scipy.io import loadmat
from pathlib import Path
import scipy.linalg as la
import scipy.signal as signal
import cvxpy as cp

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

    def raw_input(self, x: np.ndarray, x_I: np.ndarray, r: np.ndarray) -> np.ndarray:
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
        u_raw = self.raw_input(x, x_I, r)
        return np.clip(u_raw, self.u_min, self.u_max), u_raw

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
        u_raw = self.raw_input(x, x_I, r)
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
            u_sat, _ = self.compute_input(x, x_I, r) # saturated control input
            
            # Compute plant derivatives and integrator derivatives, then concatenate
            dx   = self.plant.derivatives(x, u_sat, d(t))
            dx_I = self.integrator_derivative(x, x_I, r)
            return np.concatenate([dx, dx_I])

        return ode

  
    
class StateFeedbackControllerDisturbanceRejection:
    def __init__(
        self,
        plant,
        K_x: np.ndarray,
        K_I: np.ndarray,
        N: np.ndarray,
        M: np.ndarray,
        u_min: float = 0.0,
        u_max: float = 1.0,
    ):
        self.plant     = plant
        self.K_x       = K_x
        self.K_I       = K_I
        self.N         = N  # intentionally unused for now
        self.M         = M
        self.u_min     = u_min
        self.u_max     = u_max
        self.u_offset  = 0.5 * (u_min + u_max)
        self.n_outputs = plant.C.shape[0]
        self.x_I       = np.zeros(self.n_outputs)

    @classmethod
    def cost_matrices(cls, plant, Q_scale: float = 1.0, R_scale: float = 1.0):
        n = plant.A.shape[0]
        m = plant.B_u.shape[1]
        p = plant.C.shape[0]

        Q = np.eye(n + p) * Q_scale
        R = np.eye(m) * R_scale
        return Q, R

    @classmethod
    def find_controller_gains(
        cls,
        plant,
        Q: np.ndarray | None = None,
        R: np.ndarray | None = None,
        u_min: float = 0.0,
        u_max: float = 1.0,
    ) -> "StateFeedbackControllerDisturbanceRejection":

        A   = plant.A
        B_u = plant.B_u
        B_d = plant.B_d
        C   = plant.C

        n = A.shape[0]
        m = B_u.shape[1]
        p = C.shape[0]
        nw = B_d.shape[1]

        # ── Augmented system ──────────────────────────────────────────────────
        A_aug = np.block([
            [A, np.zeros((n, p))],
            [C, np.zeros((p, p))]
        ])

        B_aug = np.vstack([
            B_u,
            np.zeros((p, m))
        ])

        Bd_aug = np.vstack([
            B_d,
            np.zeros((p, nw))
        ])

        na = n + p

        Qz = Q if Q is not None else np.eye(na)
        Ru = R if R is not None else np.eye(m)

        Cz = np.vstack([
            la.sqrtm(Qz),
            np.zeros((m, na))
        ])

        Dz = np.vstack([
            np.zeros((na, m)),
            la.sqrtm(Ru)
        ])

        nz = Cz.shape[0]

        ubar = 0.5 * (u_max - u_min) * np.ones(m)
        
        eps_lmi = 1e-6

        # ── Define LMI variables ──────────────────────────────────────────────

        P = cp.Variable((na, na), symmetric=True)
        Y = cp.Variable((m, na))
        X = cp.Variable((m, m), symmetric=True)
        gamma = cp.Variable(nonneg=True)

        M11 = A_aug @ P + P @ A_aug.T + B_aug @ Y + Y.T @ B_aug.T
        M12 = Bd_aug
        M13 = P @ Cz.T + Y.T @ Dz.T

        LMI_Hinf = cp.bmat([
            [M11,    M12,                         M13],
            [M12.T, -gamma * np.eye(nw),          np.zeros((nw, nz))],
            [M13.T, np.zeros((nz, nw)),          -gamma * np.eye(nz)],
        ])

        LMI_input = cp.bmat([
            [X,   Y],
            [Y.T, P],
        ])

        # ── Constrains on system ──────────────────────────────────────────────────
        constraints = [
            P >> eps_lmi * np.eye(na),
            LMI_Hinf << -eps_lmi * np.eye(na + nw + nz),
            LMI_input >> eps_lmi * np.eye(m + na),
            cp.diag(X) <= ubar**2,
        ]

        # Solve LMI problem
        problem = cp.Problem(cp.Minimize(gamma), constraints)
        problem.solve(solver=cp.CLARABEL, verbose=False)

        # Check if the problem was solved successfully
        if problem.status not in ["optimal", "optimal_inaccurate"]:
            raise RuntimeError(
                f"Bounded-input H-infinity LMI failed. Status: {problem.status}"
            )

        # Extract controller gains
        K_aug = Y.value @ np.linalg.inv(P.value)
        K_x = K_aug[:, :n]
        K_I = K_aug[:, n:]

        

        # ── Steady-state reference gains  ──────────────────────────────────────────────────
        SS_lhs = np.block([[A,  B_u             ],
                        [C,  np.zeros((p, m))]])
        SS_rhs = np.block([[np.zeros((n, p))],
                        [np.eye(p)        ]])

        N_xu = la.solve(SS_lhs, SS_rhs)
        N    = N_xu[n:, :] - K_x @ N_xu[:n, :]   # (m × p)
        #N = np.zeros_like(N)  #! Disable reference feedforward for now, since it can cause issues with the disturbance rejection formulation

        # ── Find anti-windup gain M ──────────────────────────────────────────────────
        F = A_aug + B_aug @ K_aug
        eigs = np.linalg.eigvals(F)

        alpha = 3.0
        desired_poles = np.sort(eigs.real)[-p:] * alpha

        result = signal.place_poles(
            np.zeros((p, p)),
            -K_I.T,
            desired_poles,
        )

        M = result.gain_matrix.T

        return cls(plant=plant, K_x=K_x, K_I=K_I, N=N, M=M, u_min=u_min, u_max=u_max)

    def reset(self):
        self.x_I = np.zeros(self.n_outputs)

    def outputs(self, x: np.ndarray) -> np.ndarray:
        return self.plant.C @ x

    def raw_input(self, x: np.ndarray, x_I: np.ndarray, r: np.ndarray) -> np.ndarray:
        z = self.plant._to_shifted_frame(x)
        r_shift = r - self.plant.C @ (self.plant.coordinate_shift)  # shift reference
        return self.K_x @ z + self.K_I @ x_I + self.N @ r_shift

    def compute_input(self, x: np.ndarray, x_I: np.ndarray, r: np.ndarray) -> np.ndarray:
        u_shifted = self.raw_input(x, x_I, r)
        u_valve = self.u_offset + u_shifted
        return np.clip(u_valve, self.u_min, self.u_max), u_valve

    def integrator_derivative(self, x: np.ndarray, x_I: np.ndarray, r: np.ndarray) -> np.ndarray:
        error = self.outputs(x) - r

        # Calculate unsaturated and saturated control inputs
        u_valve_sat, u_valve = self.compute_input(x, x_I, r)

        # Anti-windup correction
        anti_windup = self.M @ (u_valve_sat - u_valve)

        return error - anti_windup

    def controller_derivatives(self, r: np.ndarray, d) -> callable:
        N = self.plant.total_states

        def ode(t, augmented_state):
            x   = augmented_state[:N]
            x_I = augmented_state[N:]

            u_sat, _ = self.compute_input(x, x_I, r)

            dx   = self.plant.derivatives(x, u_sat, d(t))
            dx_I = self.integrator_derivative(x, x_I, r)

            return np.concatenate([dx, dx_I])

        return ode