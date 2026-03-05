import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Custom models
from models import AirDuctModel


class System:
    def __init__(self, components):
        """
        Initialize the system with an ordered list of components.
        Args:
            components (list): Component instances, where the outlet of
                               each feeds the inlet of the next
        """
        self.components = components

    def simulate(self, t_end, inlet_fn, t_eval=None):
        """
        Simulate the full system.
        Args:
            t_end (float):       Simulation end time [s]
            inlet_fn (callable): Inlet temperature as a function of time [°C]
            t_eval (ndarray):    Optional array of time points to evaluate at [s]
        Returns:
            t (ndarray):            Time vector [s]
            states (list[ndarray]): Temperature trajectories per component [°C]
        """
        initial_state_combined = np.concatenate([c.initial_state for c in self.components])

        component_state_counts = [c.num_states for c in self.components]
        component_slices, current_index = [], 0
        for state_count in component_state_counts:
            component_slices.append(slice(current_index, current_index + state_count))
            current_index += state_count

        def ode(t, T_all):
            dT_all = np.empty_like(T_all)
            u = inlet_fn(t) + 273.15

            for comp, sl in zip(self.components, component_slices):
                dT_all[sl] = comp.derivatives(T_all[sl], u)
                u = T_all[sl][-1]

            return dT_all

        if t_eval is None:
            t_eval = np.linspace(0, t_end, 500)

        sol = solve_ivp(ode, (0, t_end), initial_state_combined, t_eval=t_eval,
                        method='RK45', rtol=1e-6, atol=1e-8)
        return sol.t, [sol.y[sl] - 273.15 for sl in component_slices]

    def plot(self, t, states):
        """
        Plot segment temperatures over time for each component.
        Args:
            t (ndarray):            Time vector [s]
            states (list[ndarray]): Temperature trajectories per component [°C]
        """
        fig, axes = plt.subplots(len(self.components), 1,
                                 figsize=(10, 4 * len(self.components)),
                                 sharex=True)
        if len(self.components) == 1:
            axes = [axes]

        for ax, comp, state in zip(axes, self.components, states):
            for k in range(comp.K):
                ax.plot(t, state[k], label=f'Segment {k+1}')
            ax.set_ylabel('Temperature [°C]')
            ax.set_title(type(comp).__name__)
            ax.legend(fontsize=7)
            ax.grid(True)

        axes[-1].set_xlabel('Time [s]')
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    duct = AirDuctModel(volume_flow_rate=0.3, cross_section_area=2.0,
                        duct_length=10.0, num_segments=10)
    duct.set_initial_temperature(25)

    duct2 = AirDuctModel(volume_flow_rate=0.3, cross_section_area=2.0,
                         duct_length=10.0, num_segments=10)
    duct2.set_initial_temperature(30)

    system = System(components=[duct, duct2])
    t, states = system.simulate(t_end=200, inlet_fn=lambda t: 20)
    system.plot(t, states)