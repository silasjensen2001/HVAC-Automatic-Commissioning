"""
Simple LMI-based full-state-feedback controller with integral action
for the HVAC system.

This matches the MATLAB CVX script:
    A_a = [A   0
          -C   0]
    B_a = [B_u
            0 ]

LMI:
    P > 0
    (A_a P - B_a Y) + (A_a P - B_a Y)' < 0

Controller recovery:
    K_a = Y P^{-1}

Control law:
    u_dev = -K_a @ [z; x_I]

Split:
    K_x = K_a[:, :n]
    K_I = K_a[:, n:]

where z is the shifted plant state and x_I is the integral state.
"""

import numpy as np
import scipy.io as sio
import scipy.linalg as la
import cvxpy as cp
from pathlib import Path


# ── Paths ─────────────────────────────────────────────────────────────────────
root = Path(__file__).resolve().parent.parent

model_path = root / "models/linear/HVAC_model.mat"
out_dir = root / "models/controller"
out_dir.mkdir(parents=True, exist_ok=True)

K_x_path = out_dir / "K_x.mat"
K_I_path = out_dir / "K_I.mat"


# ── Load linearised model ─────────────────────────────────────────────────────
data = sio.loadmat(model_path)

A = data["A"]
B_u = data["B_u"]
C = data["C"]

n = A.shape[0]
m = B_u.shape[1]
p = C.shape[0]

print("System dimensions:")
print(f"  States  n = {n}")
print(f"  Inputs  m = {m}")
print(f"  Outputs p = {p}")

eigs_ol = np.linalg.eigvals(A)
print("\nOpen-loop eigenvalues:")
print(f"  Max real part: {np.max(eigs_ol.real):+.6f}")
print(f"  Min real part: {np.min(eigs_ol.real):+.6f}")


# ── Augmented PI system ───────────────────────────────────────────────────────
# z_dot   = A z + B_u u
# xI_dot  = -C z
#
# z_aug = [z; xI]

n_aug = n + p

A_aug = np.zeros((n_aug, n_aug))
A_aug[:n, :n] = A
A_aug[n:, :n] = -C

B_aug = np.zeros((n_aug, m))
B_aug[:n, :] = B_u

print("\nAugmented system:")
print(f"  A_aug shape = {A_aug.shape}")
print(f"  B_aug shape = {B_aug.shape}")


# ── Optional controllability checks ───────────────────────────────────────────
ctrb_original = np.hstack([
    np.linalg.matrix_power(A, i) @ B_u
    for i in range(n)
])
rank_original = np.linalg.matrix_rank(ctrb_original)

ctrb_aug = np.hstack([
    np.linalg.matrix_power(A_aug, i) @ B_aug
    for i in range(n_aug)
])
rank_aug = np.linalg.matrix_rank(ctrb_aug)

print("\nControllability:")
print(f"  Original rank  = {rank_original} / {n}")
print(f"  Augmented rank = {rank_aug} / {n_aug}")

if rank_original < n:
    print("  WARNING: Original system may not be fully controllable.")

if rank_aug < n_aug:
    print("  WARNING: Augmented system may not be fully controllable.")


# ── LMI solve ─────────────────────────────────────────────────────────────────
eps = 1e-6

P = cp.Variable((n_aug, n_aug), symmetric=True)
Y = cp.Variable((m, n_aug))

LMI = A_aug @ P - B_aug @ Y
LMI = LMI + LMI.T

constraints = [
    P >> eps * np.eye(n_aug),
    LMI << -eps * np.eye(n_aug),
]

# This is the direct feasibility problem from MATLAB.
# A tiny objective is added only to help numerical solvers pick a bounded solution.
objective = cp.Minimize(cp.trace(P) + 1e-8 * cp.sum_squares(Y))

prob = cp.Problem(objective, constraints)

solvers = [cp.MOSEK, cp.CLARABEL, cp.SCS]

for solver in solvers:
    try:
        print(f"\nTrying solver: {solver}")
        prob.solve(solver=solver, verbose=False)

        if prob.status in ("optimal", "optimal_inaccurate"):
            print(f"  Solved with {solver}")
            print(f"  Status: {prob.status}")
            break

    except cp.error.SolverError:
        print(f"  Solver {solver} failed.")

else:
    raise RuntimeError(f"LMI problem not solved. Last status: {prob.status}")


P_val = P.value
Y_val = Y.value

if P_val is None or Y_val is None:
    raise RuntimeError("Solver returned None for P or Y.")


# ── Recover controller ────────────────────────────────────────────────────────
# MATLAB:
#   K_a = Y / P
#
# Python equivalent:
#   K_a = Y @ inv(P)
#
# Numerically safer:
#   K_a.T = solve(P.T, Y.T)

K_a = la.solve(P_val.T, Y_val.T).T

K_x = K_a[:, :n]
K_I = K_a[:, n:]


# ── Verify closed-loop stability ──────────────────────────────────────────────
A_cl = A_aug - B_aug @ K_a
eig_Acl = np.linalg.eigvals(A_cl)

print("\nClosed-loop eigenvalues:")
for eig in sorted(eig_Acl, key=lambda e: e.real, reverse=True):
    print(f"  {eig.real:+.6f} {eig.imag:+.6f}j")

max_real = np.max(eig_Acl.real)

print("\nStability check:")
print(f"  Max real part = {max_real:+.6e}")

if np.all(eig_Acl.real < 0):
    print("  System is stable.")
else:
    raise RuntimeError("System is NOT stable.")


# ── Save gains ────────────────────────────────────────────────────────────────
sio.savemat(K_x_path, {"Kx": K_x})
sio.savemat(K_I_path, {"Ki": K_I})

print("\nSaved controller gains:")
print(f"  K_x shape {K_x.shape} -> {K_x_path}")
print(f"  K_I shape {K_I.shape} -> {K_I_path}")

print("\nK_x:")
print(K_x)

print("\nK_I:")
print(K_I)