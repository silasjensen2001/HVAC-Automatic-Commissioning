# HVAC Automatic Commissioning

A tool for modelling, simulating, and tuning the control of HVAC air-handling
systems. Build an air-flow network visually — coolers, heaters, ducts, fans,
junctions — then simulate the closed-loop response of a state-feedback
controller and inspect temperatures, valve positions, humidity, and mixing
behaviour.

The project has two parts:

- **A physics & control core** (`src/hvac/`) — linear and nonlinear models of
  HVAC components plus state-feedback controllers with integral action.
- **A web GUI** (`gui/`) — a drag-and-drop node-graph editor (React) backed by
  a FastAPI server that runs the simulations.

The backend translates the node graph from the GUI into an ordered list of
component models, builds the plant (`models.HVAC`), designs a controller
(`controller.py`), integrates the closed loop with `scipy.integrate.solve_ivp`,
and returns the time series as JSON.

---

## Prerequisites

- **Python** 3.10+ (developed on 3.12)
- **Node.js / npm** (Node 18+ recommended)

The launch scripts install the rest for you.

---

## Quick start

### Linux / macOS

```bash
./run_gui.sh
```

### Windows

```powershell
powershell -ExecutionPolicy Bypass -File .\run_gui.ps1
```

The script will, on first run:

1. Create a Python virtual environment at `gui/backend/.venv` and install the
   backend dependencies.
2. Install the frontend npm dependencies.
3. Start the FastAPI backend on <http://localhost:8000>.
4. Start the Vite dev server on <http://localhost:5173> and open it in your
   browser.

Press **Ctrl+C** in the terminal to stop both the frontend and the backend.

---

## Using the tool

1. **Build a system** — drag components from the left palette onto the canvas
   and connect their ports. A typical loop has an outdoor-air source → fan →
   coolers/heaters → junction (mixing return air) → exhaust.
2. **Set parameters** — click a node or edge to edit its properties in the
   right-hand panel (target temperatures, geometry, flow rates, valve sizing,
   disturbance settings, etc.).
3. **Configure the simulation** — in the palette, set the duration, model mode
   (linear/nonlinear), controller type (LQR/LMI), and Q/R tuning.
4. **Run** — click *Simulate*. The graph is validated first; errors block the
   run and warnings are shown but don't.
5. **Inspect results** — the results panel shows temperatures, valve positions,
   humidity, and junction mixing, plus controller metrics. Export any plot as
   PNG.

### Simulation parameters

| Parameter         | Meaning                                                        |
| ----------------- | ------------------------------------------------------------- |
| `t_end`           | Simulation horizon (seconds)                                  |
| `model_mode`      | `nonlinear` (full ODEs) or `linear` (linearised state space)  |
| `controller_type` | `lqr` or `lmi` (disturbance-rejection)                        |
| `Q_scale`         | Global scaling of the state-error penalty                     |
| `R_scale`         | Global scaling of the control-effort penalty                  |
| advanced Q/R      | Per-state / per-input diagonal weights (optional)             |

### Running the core directly

The model and controller code can be used without the GUI. The standalone
scripts in `src/hvac/scripts/` (e.g. `sim_full_system.py`) build a system,
design a controller, simulate, and plot with Matplotlib:

```bash
cd src/hvac/scripts
python sim_full_system.py
```