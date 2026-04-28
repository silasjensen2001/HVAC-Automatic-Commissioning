"""
HVAC Controller Gain Design  —  LQR
=====================================
Replaces LMI with LQR, which naturally penalises large gains via R.
Tune Q and R to adjust performance vs control effort.
"""

import numpy as np
import scipy.io as sio
import scipy.linalg as la
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
root       = Path(__file__).resolve().parent.parent
model_path = root / "models/linear/HVAC_model.mat"
out_dir    = root / "models/controller"
out_dir.mkdir(parents=True, exist_ok=True)

# ── Load model ─────────────────────────────────────────────────────────────────
data = sio.loadmat(model_path)
A   = data["A"]
B_u = data["B_u"]
C   = data["C"]

n = A.shape[0]
m = B_u.shape[1]
p = C.shape[0]

print(f"System: n={n}, m={m}, p={p}")

# ── Augmented system (plant + integrator) ──────────────────────────────────────
#
#   A_aug = [ A   0 ]     B_aug = [ B_u ]
#           [ C   0 ]             [  0  ]

A_aug = np.block([[A, np.zeros((n, p))],
                  [C, np.zeros((p, p))]])
B_aug = np.block([[B_u              ],
                  [np.zeros((p, m)) ]])

# ── Phase 1: LQR → K_x, K_I ───────────────────────────────────────────────────
#
# Minimises:  J = ∫ (x'Qx + u'Ru) dt
#
# Q penalises state error   — larger Q = faster convergence, larger gains
# R penalises control effort — larger R = smaller gains, slower response
#
# Tune R first: if gains are too large, increase R (e.g. 10, 100, 1000)

# Maximum acceptable deviations (tune these)
max_temp_dev  = 100.0    # K — how much temperature error is "acceptable"
max_integral  = 500.0   # K·s — max integrator state
max_valve     = 1.0     # valve is [0, 1]

# Bryson: Q_ii = 1/max_i²,  R_ii = 1/max_i²
Q = np.diag(
    [1/max_temp_dev**2] * n +       # plant states
    [1/max_integral**2] * p         # integrator states
)
R = np.diag([1/max_valve**2] * m)   # = np.eye(m) since max_valve=1


# Solve continuous-time algebraic Riccati equation
X   = la.solve_continuous_are(A_aug, B_aug, Q, R)
K_a = -la.solve(R, B_aug.T @ X)   # K_a = R^{-1} B' X

K_x = K_a[:, :n]   # m × n  — state feedback
K_I = K_a[:, n:]   # m × p  — integral feedback

# Verify stability
F    = A_aug + B_aug @ K_a
eigs = np.linalg.eigvals(F)
assert np.all(eigs.real < 0), "Closed-loop NOT stable!"
print(f"Stable ✓  max Re(λ) = {np.max(eigs.real):+.6f}")
print(f"\nGain magnitudes:")
print(f"  max|K_x| = {np.max(np.abs(K_x)):.4f}")
print(f"  max|K_I| = {np.max(np.abs(K_I)):.4f}")

# ── Phase 2: Steady-state → N ──────────────────────────────────────────────────
#
# Solve: [A   B_u] [N_x]   [0]
#        [C    0 ] [N_u] = [I]
#
# N = N_u - K_x @ N_x

SS_lhs = np.block([[A,  B_u             ],
                   [C,  np.zeros((p, m))]])
SS_rhs = np.block([[np.zeros((n, p))],
                   [np.eye(p)        ]])

N_xu = la.solve(SS_lhs, SS_rhs)
N_x  = N_xu[:n, :]
N_u  = N_xu[n:, :]
N    = N_u - K_x @ N_x   # m × p

# ── Save & print ───────────────────────────────────────────────────────────────
sio.savemat(str(out_dir / "K_x.mat"), {"Kx": K_x})
sio.savemat(str(out_dir / "K_I.mat"), {"Ki": K_I})
sio.savemat(str(out_dir / "N.mat"),   {"N":  N  })

print(f"\nK_x {K_x.shape}:\n{K_x}")
print(f"\nK_I {K_I.shape}:\n{K_I}")
print(f"\nN   {N.shape}:\n{N}")