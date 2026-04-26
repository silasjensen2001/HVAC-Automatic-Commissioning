"""
LMI-based full-state-feedback controller with integral action for the HVAC system.

Augmented state: z_aug = [z (shifted plant state, n), x_I (integrator states, p)]
Control law:     u = -(K_x @ z + K_I @ x_I)

Augmented open-loop system (eq. 89/90 in LTI notes):
    A_aug = [[A,  0 ],     B_aug = [[B_u],
             [-C, 0 ]]               [ 0 ]]

The system has a very wide eigenvalue spread (-651 to -0.027), which makes the
Lyapunov matrix Q ill-conditioned. To fix this, we pre-balance the augmented
system using a diagonal similarity transformation T (Osborne scaling) so that
the LMI is solved in better-conditioned coordinates, then transform K back.

Balanced state: z̃ = inv(T) @ z_aug
    A_bal = inv(T) @ A_aug @ T
    B_bal = inv(T) @ B_aug

LMI in balanced coordinates (eq. 90, adapted for u = -K @ z convention):
    A_bal @ Q + Q @ A_bal^T - B_bal @ Y - Y^T @ B_bal^T + 2*ALPHA*Q < 0

Note on signs: the notes write +BY for u = Kx. Our controller uses u = -K_phys @ z
(K_phys > 0, minus sign in controller.py), so the closed-loop is A - B @ K_phys.
Substituting Y = K_phys @ Q and using congruence with P = Q^{-1} yields the -BY form.

Gains recovered as: K_full = K_bal @ inv(T),  K_bal = solve(Q, Y^T)^T
"""

import numpy as np
import scipy.io as sio
import scipy.linalg as la
import cvxpy as cp
from pathlib import Path

# ── Tuning ────────────────────────────────────────────────────────────────────
# Minimum closed-loop decay rate: ||x(t)|| <= C*exp(-ALPHA*t)*||x(0)||
ALPHA = 0.000001

# ── Load linearised model ─────────────────────────────────────────────────────
model_path = Path(__file__).resolve().parent.parent / "models/linear/HVAC_model.mat"
data = sio.loadmat(model_path)

A       = data["A"]
B_u     = data["B_u"]
C       = data["C"]
x_shift = data["x_shift"].flatten()

n = A.shape[0]    # plant states   (20)
m = B_u.shape[1]  # inputs         (2)
p = C.shape[0]    # outputs        (2)

print(f"Plant: n={n} states, m={m} inputs, p={p} outputs")
eigs_ol = np.linalg.eigvals(A)
print(f"Open-loop eigenvalue range: [{eigs_ol.real.min():.4f}, {eigs_ol.real.max():.4f}]")

# ── Augmented system (plant + integrators) ────────────────────────────────────
# dz/dt   = A @ z + B_u @ u
# dx_I/dt = -C @ z          (constant reference absorbed at operating point)
n_aug = n + p

A_aug = np.zeros((n_aug, n_aug))
A_aug[:n, :n] = A
A_aug[n:, :n] = -C

B_aug = np.zeros((n_aug, m))
B_aug[:n, :] = B_u

# ── Balance augmented system for numerical conditioning ───────────────────────
# Ill-conditioning of Q is caused by the wide eigenvalue spread.
# We apply a diagonal similarity T so that A_bal = inv(T) @ A_aug @ T
# has more evenly distributed row/column norms.
#
# matrix_balance returns (A_bal, T) such that A_aug = T @ A_bal @ inv(T),
# i.e. A_bal = inv(T) @ A_aug @ T.  permute=False gives pure diagonal scaling.
A_bal, T = la.matrix_balance(A_aug, permute=False, separate=False)
T_diag     = np.diag(T)
T_inv_diag = 1.0 / T_diag
T_inv      = np.diag(T_inv_diag)
B_aug_bal  = T_inv @ B_aug

print(f"\nBalancing T diagonal range: [{T_diag.min():.3e}, {T_diag.max():.3e}]")
print(f"A_aug cond before balancing: {np.linalg.cond(A_aug):.2e}")
print(f"A_bal cond after  balancing: {np.linalg.cond(A_bal):.2e}")

# ── LMI (eq. 90 from LTI notes, solved in balanced coordinates) ───────────────
# Variables in balanced coordinates
Q = cp.Variable((n_aug, n_aug), symmetric=True)
Y = cp.Variable((m, n_aug))

lmi_lhs = (A_bal @ Q + Q @ A_bal.T
           - B_aug_bal @ Y - Y.T @ B_aug_bal.T
           + 2 * ALPHA * Q)

constraints = [
    Q >> np.eye(n_aug) * 1e-3,         # Q > 0
    lmi_lhs << -np.eye(n_aug) * 1e-4,  # stability LMI < 0
]

# Minimise trace(Q) to obtain a well-conditioned Lyapunov matrix
prob = cp.Problem(cp.Minimize(cp.trace(Q)), constraints)

for solver in [cp.MOSEK, cp.CLARABEL, cp.SCS]:
    try:
        prob.solve(solver=solver, verbose=False)
        if prob.status in ("optimal", "optimal_inaccurate"):
            print(f"\nSolved with {solver} (status: {prob.status})")
            break
    except cp.error.SolverError:
        continue
else:
    raise RuntimeError(f"All solvers failed. Last status: {prob.status}")

Q_val = Q.value
Y_val = Y.value

if Q_val is None or Y_val is None:
    raise RuntimeError("Solver returned None — try reducing ALPHA or loosening constraints.")

print(f"Q condition number: {np.linalg.cond(Q_val):.2e}")

# ── Recover gains ─────────────────────────────────────────────────────────────
# K_bal @ Q = Y  →  Q^T @ K_bal^T = Y^T  →  K_bal = solve(Q, Y^T)^T
# (using la.solve for numerical stability instead of explicit Q^{-1})
K_bal  = la.solve(Q_val, Y_val.T, assume_a="sym").T   # gains in balanced coords
K_full = K_bal @ T_inv                                 # transform to original coords
K_x    = K_full[:, :n]                                 # (m, n)
K_i    = K_full[:, n:]                                 # (m, p)

# ── Verify closed-loop stability ──────────────────────────────────────────────
A_cl  = A_aug - B_aug @ K_full
eigs  = np.linalg.eigvals(A_cl)

print(f"\nClosed-loop eigenvalues (real part, sorted):")
for eig in sorted(eigs, key=lambda e: e.real, reverse=True):
    print(f"  {eig.real:+.5f}  {eig.imag:+.5f}j")

if not np.all(eigs.real < 0):
    raise RuntimeError("Closed-loop is not stable! Try reducing ALPHA or checking the model.")

print(f"\nAll {n_aug} eigenvalues stable. Slowest: {eigs.real.max():.5f}")

# ── Save ──────────────────────────────────────────────────────────────────────
out_dir = Path(__file__).resolve().parent.parent / "models/controller"
out_dir.mkdir(parents=True, exist_ok=True)

sio.savemat(out_dir / "K_x.mat", {"Kx": K_x})
sio.savemat(out_dir / "K_I.mat", {"Ki": K_i})

print(f"\nK_x  shape: {K_x.shape}  →  {out_dir / 'K_x.mat'}")
print(f"K_I  shape: {K_i.shape}  →  {out_dir / 'K_I.mat'}")
print(f"\nK_x:\n{K_x}")
print(f"\nK_I:\n{K_i}")
