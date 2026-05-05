from abc import ABC, abstractmethod
import numpy as np
import scipy.linalg as la
import scipy.signal as signal
import cvxpy as cp
from scipy.io import loadmat
from pathlib import Path

class BaseStateFeedbackController(ABC):
    """Abstract base for state-feedback."""

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
        self.N         = N
        self.M         = M
        self.u_min     = u_min
        self.u_max     = u_max
        self.n_outputs = plant.C.shape[0]
        self.x_I       = np.zeros(self.n_outputs)

    # ── Shared concrete methods ──────────────────────────────────────────────

    def outputs(self, x: np.ndarray) -> np.ndarray:
        return self.plant.C @ x

    def raw_input(self, x: np.ndarray, x_I: np.ndarray, r: np.ndarray) -> np.ndarray:
        z       = self.plant._to_shifted_frame(x)
        r_shift = r - self.plant.C @ self.plant.coordinate_shift
        return self.K_x @ z + self.K_I @ x_I + self.N @ r_shift

    def integrator_derivative(
        self, x: np.ndarray, x_I: np.ndarray, r: np.ndarray
    ) -> np.ndarray:
        error             = self.outputs(x) - r
        u_sat, u_raw      = self.compute_input(x, x_I, r)
        anti_windup       = self.M @ (u_sat - u_raw)
        return error + anti_windup

    def controller_derivatives(self, r: np.ndarray, d) -> callable:
        """
        Returns an ODE callable for use with solve_ivp.
        Augmented state: z = [x (n_states), x_I (n_outputs)].
        """
        N = self.plant.total_states

        def ode(t, augmented_state):
            x, x_I  = augmented_state[:N], augmented_state[N:]
            u_sat, _ = self.compute_input(x, x_I, r)
            dx       = self.plant.derivatives(x, u_sat, d(t))
            dx_I     = self.integrator_derivative(x, x_I, r)
            return np.concatenate([dx, dx_I])

        return ode

    @classmethod
    def cost_matrices(
        cls, plant, Q_scale: float = 1.0, R_scale: float = 1.0
    ) -> tuple[np.ndarray, np.ndarray]:
        n = plant.A.shape[0]
        m = plant.B_u.shape[1]
        p = plant.C.shape[0]
        return np.eye(n + p) * Q_scale, np.eye(m) * R_scale

    @staticmethod
    def _compute_anti_windup_gain(
        A_aug: np.ndarray, B_aug: np.ndarray, K_aug: np.ndarray, K_I: np.ndarray, alpha: float = 3.0
    ) -> np.ndarray:
        """Shared pole-placement anti-windup gain computation."""
        p = K_I.shape[0]
        eigs = np.linalg.eigvals(A_aug + B_aug @ K_aug)
        desired_poles = np.sort(eigs.real)[-p:] * alpha
        result = signal.place_poles(np.zeros((p, p)), K_I.T, desired_poles)
        return result.gain_matrix.T

    @staticmethod
    def _steady_state_reference_gain(
        A: np.ndarray, B_u: np.ndarray, C: np.ndarray, K_x: np.ndarray
    ) -> np.ndarray:
        """Shared steady-state reference pre-gain N."""
        n, m, p = A.shape[0], B_u.shape[1], C.shape[0]
        SS_lhs = np.block([[A, B_u], [C, np.zeros((p, m))]])
        SS_rhs = np.block([[np.zeros((n, p))], [np.eye(p)]])
        N_xu   = la.solve(SS_lhs, SS_rhs)
        return N_xu[n:, :] - K_x @ N_xu[:n, :]

    # ── Subclass ────────────────────────────────────────────────────

    @abstractmethod
    def compute_input(
        self, x: np.ndarray, x_I: np.ndarray, r: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (u_saturated, u_unsaturated)."""

    @classmethod
    @abstractmethod
    def find_controller_gains(cls, plant, **kwargs) -> "BaseStateFeedbackController":
        """Compute and return a fully constructed controller instance."""


class StateFeedbackController(BaseStateFeedbackController):
    """LQR-based state-feedback controller."""

    def compute_input(self, x, x_I, r):
        u_raw = self.raw_input(x, x_I, r)
        return np.clip(u_raw, self.u_min, self.u_max), u_raw

    @classmethod
    def find_controller_gains(cls, plant, Q=None, R=None, u_min=0.0, u_max=1.0):
        # Extract system matrices and dimensions
        A, B_u, C   = plant.A, plant.B_u, plant.C
        n, m, p     = A.shape[0], B_u.shape[1], C.shape[0]

        # Form augmented system for integral action
        A_aug = np.block([[A, np.zeros((n, p))], [C, np.zeros((p, p))]])
        B_aug = np.vstack([B_u, np.zeros((p, m))])

        Q = Q if Q is not None else np.eye(n + p)
        R = R if R is not None else np.eye(m) * 8

        # Solve continuous-time algebraic Riccati equation for augmented system
        X     = la.solve_continuous_are(A_aug, B_aug, Q, R)
        K_aug = -la.solve(R, B_aug.T @ X)

        # Check closed-loop stability
        eigs = np.linalg.eigvals(A_aug + B_aug @ K_aug)
        assert np.all(eigs.real < 0), f"Closed-loop NOT stable! max Re(λ) = {np.max(eigs.real):+.6f}"

        # Extract state-feedback and integral gains
        K_x, K_I = K_aug[:, :n], K_aug[:, n:]
        N = cls._steady_state_reference_gain(A, B_u, C, K_x)
        M = cls._compute_anti_windup_gain(A_aug, B_aug, K_aug, K_I)

        # Return the constructed controller instance
        return cls(plant, K_x, K_I, N, M, u_min=u_min, u_max=u_max)

class StateFeedbackControllerDisturbanceRejection(BaseStateFeedbackController):
    """H∞ LMI-based state-feedback controller with disturbance rejection."""

    def __init__(self, plant, K_x, K_I, N, M, u_min=0.0, u_max=1.0):
        super().__init__(plant, K_x, K_I, N, M, u_min, u_max)
        self.u_offset = 0.5 * (u_min + u_max)

    def compute_input(self, x, x_I, r):
        u_valve = self.u_offset + self.raw_input(x, x_I, r)
        return np.clip(u_valve, self.u_min, self.u_max), u_valve

    @classmethod
    def find_controller_gains(cls, plant, Q=None, R=None, u_min=0.0, u_max=1.0):
        # Extract system matrices and dimensions
        A, B_u, B_d, C = plant.A, plant.B_u, plant.B_d, plant.C
        n, m, p, nw    = A.shape[0], B_u.shape[1], C.shape[0], B_d.shape[1]

        # Form augmented system for integral action
        A_aug  = np.block([[A, np.zeros((n, p))], [C, np.zeros((p, p))]])
        B_aug  = np.vstack([B_u, np.zeros((p, m))])
        Bd_aug = np.vstack([B_d, np.zeros((p, nw))])
        na     = n + p

        # Define performance weighting matrices
        Qz = Q if Q is not None else np.eye(na)
        Ru = R if R is not None else np.eye(m)

        # Construct LMI matrices for H∞ synthesis
        Cz = np.vstack([la.sqrtm(Qz), np.zeros((m, na))])
        Dz = np.vstack([np.zeros((na, m)), la.sqrtm(Ru)])
        nz = Cz.shape[0]

        ubar = 0.5 * (u_max - u_min) * np.ones(m)
        eps  = 1e-6

        # Define CVXPY variables for LMI optimization
        P, Y, X, gamma = (
            cp.Variable((na, na), symmetric=True),
            cp.Variable((m, na)),
            cp.Variable((m, m), symmetric=True),
            cp.Variable(nonneg=True),
        )

        M11 = A_aug @ P + P @ A_aug.T + B_aug @ Y + Y.T @ B_aug.T
        M12 = Bd_aug
        M13 = P @ Cz.T + Y.T @ Dz.T

        # Define LMI constraints for bounded-input H∞ synthesis
        constraints = [
            P >> eps * np.eye(na),
            cp.bmat([[M11, M12, M13],
                     [M12.T, -gamma * np.eye(nw), np.zeros((nw, nz))],
                     [M13.T, np.zeros((nz, nw)), -gamma * np.eye(nz)]]) << -eps * np.eye(na + nw + nz),
            cp.bmat([[X, Y], [Y.T, P]]) >> eps * np.eye(m + na),
            cp.diag(X) <= ubar**2,
        ]

        # Solve the LMI optimization problem
        cp.Problem(cp.Minimize(gamma), constraints).solve(solver=cp.CLARABEL, verbose=False)

        if P.value is None:
            raise RuntimeError("Bounded-input H∞ LMI failed.")

        # Extract controller gains from optimization results
        K_aug = Y.value @ np.linalg.inv(P.value)
        K_x, K_I = K_aug[:, :n], K_aug[:, n:]

        N = cls._steady_state_reference_gain(A, B_u, C, K_x)
        M = cls._compute_anti_windup_gain(A_aug, B_aug, K_aug, K_I)

        return cls(plant=plant, K_x=K_x, K_I=K_I, N=N, M=M, u_min=u_min, u_max=u_max)