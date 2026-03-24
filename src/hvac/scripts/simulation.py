import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from models import AirDuctModel, Junction, Sink


class RecirculatingSystem:
    """
    Models the following topology:

        q_fresh (T_inlet_fn)
              ↓
          [Duct 0]   ← inlet duct / transport delay
              ↓
        [Junction 1] ←─────────────────────────┐
              ↓  q_total = q_fresh + q_recirc   │
          [Duct 1]                              │
              ↓                                 │
           [Room]    ← large volume (e.g. 5x5x5)│
              ↓                                 │
          [Duct 2]                              │
              ↓                                 │
        [Junction 2] ──→ [Sink]  (q_fresh)     │
              │                                 │
              └─────────────────────────────────┘
                            q_recirc

    Global conservation: q_fresh leaves at sink, q_recirc loops back.
    States: segments of Duct 0, Duct 1, Room, Duct 2 — junctions are algebraic.
    """

    def __init__(self, duct0, duct1, room, duct2, q_fresh, q_recirc):
        """
        Args:
            duct0 (AirDuctModel):  Inlet duct / transport delay before Junction 1
            duct1 (AirDuctModel):  Duct after Junction 1
            room  (AirDuctModel):  Large-volume room between Duct 1 and Duct 2
            duct2 (AirDuctModel):  Duct after room, feeds Junction 2
            q_fresh (float):       Fresh air volumetric flow rate [m³/s]
            q_recirc (float):      Recirculated air flow rate [m³/s]

        Note:
            duct0.q_a              should equal q_fresh.
            duct1, room, duct2.q_a should all equal q_fresh + q_recirc.
        """
        self.duct0    = duct0
        self.duct1    = duct1
        self.room     = room
        self.duct2    = duct2
        self.q_fresh  = q_fresh
        self.q_recirc = q_recirc
        self.q_total  = q_fresh + q_recirc
        self.junction = Junction()
        self.sink     = Sink(flow_rate=q_fresh)

        self._validate_flow_rates()

        # State layout: [duct0 | duct1 | room | duct2]
        n0 = duct0.num_states
        n1 = duct1.num_states
        nr = room.num_states
        n2 = duct2.num_states
        self._sl0 = slice(0,              n0)
        self._sl1 = slice(n0,             n0 + n1)
        self._slr = slice(n0 + n1,        n0 + n1 + nr)
        self._sl2 = slice(n0 + n1 + nr,   n0 + n1 + nr + n2)

    def _validate_flow_rates(self):
        """Warn if duct flow rates are inconsistent with declared flows."""
        if not np.isclose(self.duct0.q_a, self.q_fresh, rtol=1e-3):
            print(f"Warning: duct0.q_a={self.duct0.q_a} does not match "
                  f"q_fresh={self.q_fresh:.4f}")
        for name, duct in [('duct1', self.duct1), ('room', self.room), ('duct2', self.duct2)]:
            if not np.isclose(duct.q_a, self.q_total, rtol=1e-3):
                print(f"Warning: {name}.q_a={duct.q_a} does not match "
                      f"q_fresh + q_recirc={self.q_total:.4f}")

    def simulate(self, t_end, inlet_fn, t_eval=None):
        """
        Simulate the recirculating system.

        Args:
            t_end (float):       Simulation end time [s]
            inlet_fn (callable): Fresh air inlet temperature as function of time [°C]
            t_eval (ndarray):    Optional time points to evaluate at [s]

        Returns:
            t (ndarray):        Time vector [s]
            T_duct0 (ndarray):  Duct 0 temperatures [°C]
            T_duct1 (ndarray):  Duct 1 temperatures [°C]
            T_room  (ndarray):  Room temperatures   [°C]
            T_duct2 (ndarray):  Duct 2 temperatures [°C]
        """
        T0_init = np.concatenate([
            self.duct0.initial_state,
            self.duct1.initial_state,
            self.room.initial_state,
            self.duct2.initial_state,
        ])

        def ode(t, T_all):
            T0 = T_all[self._sl0]
            T1 = T_all[self._sl1]
            Tr = T_all[self._slr]
            T2 = T_all[self._sl2]

            # Duct 0: raw fresh air inlet
            T_fresh   = inlet_fn(t) + 273.15
            dT0 = self.duct0.derivatives(T0, T_fresh)

            # Junction 1: delayed fresh air + recirculated air
            T_delayed = T0[-1]
            T_recirc  = T2[-1]
            T_j1 = self.junction.mix([
                (self.q_fresh,  T_delayed),
                (self.q_recirc, T_recirc),
            ])

            # Duct 1: inlet = Junction 1 output
            dT1 = self.duct1.derivatives(T1, T_j1)

            # Room: inlet = Duct 1 outlet
            # Large cross-section → small alpha → slow temperature change
            dTr = self.room.derivatives(Tr, T1[-1])

            # Duct 2: inlet = Room outlet
            dT2 = self.duct2.derivatives(T2, Tr[-1])

            # Junction 2 is algebraic — no states needed.
            return np.concatenate([dT0, dT1, dTr, dT2])

        if t_eval is None:
            t_eval = np.linspace(0, t_end, 500)

        sol = solve_ivp(ode, (0, t_end), T0_init, t_eval=t_eval,
                        method='RK45', rtol=1e-6, atol=1e-8)

        return (
            sol.t,
            sol.y[self._sl0] - 273.15,
            sol.y[self._sl1] - 273.15,
            sol.y[self._slr] - 273.15,
            sol.y[self._sl2] - 273.15,
        )

    def plot(self, t, T_duct0, T_duct1, T_room, T_duct2):
        """
        Plot segment temperatures over time for all components.
        """
        components = [
            (self.duct0, T_duct0, 'Duct 0  (inlet delay)'),
            (self.duct1, T_duct1, 'Duct 1  (after Junction 1)'),
            (self.room,  T_room,  'Room    (5×5×5 m)'),
            (self.duct2, T_duct2, 'Duct 2  (→ Junction 2 → sink / recirculation)'),
        ]

        fig, axes = plt.subplots(len(components), 1,
                                 figsize=(10, 4 * len(components)),
                                 sharex=True)

        for ax, (comp, T, title) in zip(axes, components):
            for k in range(comp.K):
                ax.plot(t, T[k], label=f'Segment {k+1}')
            ax.set_ylabel('Temperature [°C]')
            ax.set_title(title)
            ax.legend(fontsize=7)
            ax.grid(True)

        axes[-1].set_xlabel('Time [s]')
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    q_fresh  = 0.1   # [m³/s]
    q_recirc = 0.2   # [m³/s]
    q_total  = q_fresh + q_recirc

    # Duct 0: narrow inlet pipe, only carries fresh air
    duct0 = AirDuctModel(volume_flow_rate=q_fresh, cross_section_area=0.5,
                         duct_length=5.0, num_segments=5)
    duct0.set_initial_temperature(25)

    # Duct 1: normal duct after junction
    duct1 = AirDuctModel(volume_flow_rate=q_total, cross_section_area=2.0,
                         duct_length=10.0, num_segments=5)
    duct1.set_initial_temperature(25)

    # Room: 5x5 cross-section, 5m long → very large volume, slow dynamics
    # Small alpha = q_total / (A * dz) → air moves slowly through the room
    room = AirDuctModel(volume_flow_rate=q_total, cross_section_area=25.0,
                        duct_length=5.0, num_segments=5)
    room.set_initial_temperature(40)
    
    # Duct 2: normal duct after room
    duct2 = AirDuctModel(volume_flow_rate=q_total, cross_section_area=2.0,
                         duct_length=10.0, num_segments=5)
    duct2.set_initial_temperature(25)

    system = RecirculatingSystem(duct0=duct0, duct1=duct1, room=room, duct2=duct2,
                                 q_fresh=q_fresh, q_recirc=q_recirc)

    t, T0, T1, Tr, T2 = system.simulate(t_end=600, inlet_fn=lambda t: 20)
    system.plot(t, T0, T1, Tr, T2)