import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# ── models.py (inline) ────────────────────────────────────────────────────────

class AirDuctModel:
    def __init__(self, volume_flow_rate=1, cross_section_area=2,
                 duct_length=10, num_segments=10):
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
        return self.N

    def _construct_state_space(self):
        A = np.zeros((self.K, self.K))
        for k in range(self.K):
            A[k, k] = -self.alpha
            if k > 0:
                A[k, k - 1] = self.alpha
        B = np.zeros((self.K, 1))
        B[0, 0] = self.alpha
        return A, B

    def set_initial_temperature(self, temperature_C):
        self.initial_state = np.full(self.N, temperature_C + 273.15)

    def derivatives(self, T, u):
        return self.A @ T + self.B.flatten() * u


# ── Simulation ────────────────────────────────────────────────────────────────

q_fresh    = 0.5   # [m³/s]
cross_area = 0.5   # [m²]

duct = AirDuctModel(
    volume_flow_rate  = q_fresh,
    cross_section_area= cross_area,
    duct_length       = 5.0,
    num_segments      = 5,
)
duct.set_initial_temperature(20.0)   # start at 20 °C

# Inlet: step from 20 °C → 40 °C at t = 5 s
def inlet_fn(t):
    return 20.0 + 273.15 if t < 5.0 else 40.0 + 273.15   # [K]

t_end  = 20.0
t_eval = np.linspace(0, t_end, 600)

def ode(t, T):
    u = inlet_fn(t)
    return duct.derivatives(T, u)

sol = solve_ivp(ode, (0, t_end), duct.initial_state,
                t_eval=t_eval, method='RK45', rtol=1e-6, atol=1e-8)

T_segments = sol.y - 273.15   # convert K → °C

# ── Plot ──────────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(10, 5))

colors = plt.cm.viridis(np.linspace(0, 0.85, duct.K))
for k in range(duct.K):
    ax.plot(sol.t, T_segments[k], color=colors[k],
            linewidth=2, label=f'Segment {k+1}  (z={((k+1.0)*duct.dz):.1f} m)')

# Mark the step change
ax.axvline(5, color='red', linestyle='--', linewidth=1.2, label='Inlet step (20→40 °C)')

ax.set_xlabel('Time  [s]', fontsize=12)
ax.set_ylabel('Temperature  [°C]', fontsize=12)
ax.set_title('Air Duct — Segment Temperatures After Inlet Step Change', fontsize=13)
ax.legend(fontsize=9, loc='upper left')
ax.grid(True, alpha=0.4)
plt.tight_layout()
plt.show()
