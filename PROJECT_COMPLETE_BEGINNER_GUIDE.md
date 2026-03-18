# Complete Beginner Guide: Entire DeePC Project (All Files)

## 1) Scope

This guide documents the full repository, file by file, with code excerpts and beginner-level explanations.

Repository root covered in this guide:

- [__init__.py](__init__.py)
- [main.py](main.py)
- [system.py](system.py)
- [data.py](data.py)
- [simulation.py](simulation.py)
- [plotting.py](plotting.py)
- [README.md](README.md)
- [requirements.txt](requirements.txt)
- [controllers/__init__.py](controllers/__init__.py)
- [controllers/mpc.py](controllers/mpc.py)
- [controllers/deepc.py](controllers/deepc.py)
- [controllers/select_dpc.py](controllers/select_dpc.py)
- [graphs/](graphs)
- [__pycache__/](__pycache__)
- [controllers/__pycache__/](controllers/__pycache__)
- [.venv/](.venv)
- [.git/](.git)

## 2) What the Project Does

The project compares two predictive control approaches on a nonlinear system:

- MPC: model-based predictive control using local linearization and quadratic programming.
- DeePC: data-enabled predictive control using trajectory data and behavioral constraints.

Main runtime behavior:

1. Create nonlinear plant.
2. Generate random dataset trajectories.
3. Build controllers.
4. Run closed-loop simulation.
5. Compute RMSE.
6. Save figures in [graphs/](graphs).

## 3) Root Package File

### [__init__.py](__init__.py)

Current content: empty file.

Purpose:

- Marks the project root as a Python package when needed.

## 4) Main Entry Point

### [main.py](main.py)

This file orchestrates the full experiment.

Core imports:

```python
from controllers.deepc import DeePCConfig, DeePCController
from controllers.mpc import MPCConfig, MPCController
from data import generate_dataset, make_reference, summarize_dataset_ranges
from plotting import plot_results
from simulation import run_all_simulations_with_diagnostics
from system import NonlinearSystem
```

Configuration object:

```python
@dataclass
class SimulationConfig:
	n_data: int = 120
	t_data: int = 100
	data_seed: int = 7
	t_ini: int = 12
	t_f: int = 12
	t_sim: int = 120
	x0: float = 0.2
	u_min: float = -2.0
	u_max: float = 2.0
	u_weight: float = 0.1
```

Controller creation block:

```python
mpc = MPCController(
	system=system,
	config=MPCConfig(
		t_f=cfg.t_f,
		u_min=cfg.u_min,
		u_max=cfg.u_max,
		u_weight=cfg.u_weight,
	),
)

deepc = DeePCController(
	trajectories=dataset,
	config=DeePCConfig(
		t_ini=cfg.t_ini,
		t_f=cfg.t_f,
		u_min=cfg.u_min,
		u_max=cfg.u_max,
		lambda_u=1e-4,
		lambda_g=1e-4,
		lambda_ini_u=1e2,
		lambda_ini_y=1e2,
		use_soft_ini=True,
		convex_g=True,
		max_columns=600,
		debug=False,
	),
)
```

Simulation and plotting block:

```python
r = make_reference(cfg.t_sim)
results, deepc_diagnostics = run_all_simulations_with_diagnostics(...)

print("RMSE summary")
for name, res in results.items():
	print(f"- {name:10s}: {res.rmse:.6f}")

graphs_dir = plot_results(...)
print(f"Saved plots to: {graphs_dir.resolve()}")
```

## 5) Nonlinear Plant Model

### [system.py](system.py)

Plant code:

```python
class NonlinearSystem:
	def __init__(self, u_min: float = -1.0, u_max: float = 1.0) -> None:
		self.u_min = u_min
		self.u_max = u_max

	def step(self, x: float, u: float) -> float:
		u_clipped = float(np.clip(u, self.u_min, self.u_max))
		return float(x + 0.5 * np.sin(x) + u_clipped)

	def output(self, x: float) -> float:
		return float(x)
```

Model equations:

$$x_{t+1} = x_t + 0.5\sin(x_t) + u_t$$
$$y_t = x_t$$

Notes:

- Input clipping is enforced in `step`.
- Measured output is the full state.

## 6) Data and Hankel Utilities

### [data.py](data.py)

Data containers:

```python
@dataclass
class Trajectory:
	x: np.ndarray
	y: np.ndarray
	u: np.ndarray
```

Rollout function:

```python
def rollout(system: NonlinearSystem, x0: float, u_seq: np.ndarray) -> Trajectory:
	t_data = int(u_seq.shape[0])
	x = np.zeros(t_data + 1, dtype=float)
	y = np.zeros(t_data + 1, dtype=float)

	x[0] = x0
	y[0] = system.output(x0)

	for t in range(t_data):
		x[t + 1] = system.step(x[t], float(u_seq[t]))
		y[t + 1] = system.output(x[t + 1])

	return Trajectory(x=x, y=y, u=u_seq.astype(float))
```

Random dataset generation:

```python
def generate_dataset(...):
	rng = np.random.default_rng(seed)
	trajectories: List[Trajectory] = []
	for _ in range(n_data):
		u_seq = rng.uniform(u_min, u_max, size=t_data)
		x0 = float(rng.uniform(x0_min, x0_max))
		trajectories.append(rollout(system=system, x0=x0, u_seq=u_seq))
	return trajectories
```

Hankel tools used by DeePC:

```python
def build_hankel(signal: np.ndarray, window: int) -> np.ndarray:
	n = signal.shape[0]
	if window > n:
		raise ValueError(...)
	cols = n - window + 1
	return np.vstack([signal[i : i + cols] for i in range(window)])


def build_hankel_from_trajectories(trajectories, signal_name, window):
	blocks = []
	for tr in trajectories:
		sig = getattr(tr, signal_name)
		if signal_name == "y":
			sig = sig[1:]
		if sig.shape[0] >= window:
			blocks.append(build_hankel(sig, window))
	if not blocks:
		raise ValueError(...)
	return np.concatenate(blocks, axis=1)
```

Notes:

- Output alignment uses `y[1:]` so outputs pair with corresponding inputs in DeePC indexing.
- `make_reference` generates a sinusoidal target: $r_t = \sin(0.1t)$.

## 7) Controller Package Exports

### [controllers/__init__.py](controllers/__init__.py)

```python
from controllers.deepc import DeePCController
from controllers.mpc import MPCController

__all__ = ["MPCController", "DeePCController"]
```

Purpose:

- Provides top-level controller imports.

## 8) MPC Controller File

### [controllers/mpc.py](controllers/mpc.py)

Config dataclass:

```python
@dataclass
class MPCConfig:
	t_f: int
	u_min: float
	u_max: float
	u_weight: float = 0.1
```

Nominal rollout:

```python
def _nominal_rollout(self, x0: float) -> np.ndarray:
	x_bar = np.zeros(self.config.t_f + 1, dtype=float)
	x_bar[0] = x0
	for k in range(self.config.t_f):
		x_bar[k + 1] = self.system.step(x_bar[k], self._last_u_plan[k])
	return x_bar
```

Local linearization and optimization:

```python
for k in range(t_f):
	a[k] = 1.0 + 0.5 * np.cos(x_bar[k])
	f_bar = x_bar[k] + 0.5 * np.sin(x_bar[k])
	c[k] = f_bar - a[k] * x_bar[k]

x = cp.Variable(t_f + 1)
u = cp.Variable(t_f)
...
problem = cp.Problem(cp.Minimize(cost), constraints)
problem.solve(solver=cp.OSQP, warm_start=True, verbose=False)
```

Control output logic:

- Uses first element of optimized input plan.
- Shifts stored plan for warm-start-like behavior on next step.

## 9) DeePC Controller File

### [controllers/deepc.py](controllers/deepc.py)

Config dataclass:

```python
@dataclass
class DeePCConfig:
	t_ini: int
	t_f: int
	u_min: float
	u_max: float
	lambda_u: float = 1e-4
	lambda_g: float = 1e-4
	lambda_ini_u: float = 1e2
	lambda_ini_y: float = 1e2
	use_soft_ini: bool = True
	convex_g: bool = True
	max_columns: int = 600
	debug: bool = True
```

Diagnostic step container:

```python
@dataclass
class DeePCStepInfo:
	status: str
	objective: float
	u_applied: float
	u_plan: np.ndarray
	y_plan: np.ndarray
	r_plan: np.ndarray
	g: np.ndarray
	u_ini_residual_norm: float
	y_ini_residual_norm: float
	g_l2_norm: float
	g_l1_norm: float
	had_solution: bool
```

Hankel split:

```python
up = hu[:t_ini, :]
uf = hu[t_ini:, :]
yp = hy[:t_ini, :]
yf = hy[t_ini:, :]
```

Core optimization setup:

```python
g = cp.Variable(n_col)
u_f = cp.Variable(t_f)
y_f = cp.Variable(t_f)
s_u = cp.Variable(t_ini)
s_y = cp.Variable(t_ini)

constraints = [
	u_f == uf @ g,
	y_f == yf @ g,
	u_f >= self.config.u_min,
	u_f <= self.config.u_max,
]
if self.config.convex_g:
	constraints += [g >= 0.0, cp.sum(g) == 1.0]
```

Soft initial matching:

```python
constraints += [
	up @ g + s_u == u_ini,
	yp @ g + s_y == y_ini,
]
```

Cost:

```python
cost += cp.sum_squares(y_f - r)
cost += self.config.lambda_u * cp.sum_squares(u_f)
cost += self.config.lambda_g * cp.sum_squares(g)
cost += self.config.lambda_ini_u * cp.sum_squares(s_u)
cost += self.config.lambda_ini_y * cp.sum_squares(s_y)
```

Solver block:

```python
problem.solve(solver=cp.OSQP, warm_start=True, verbose=False, max_iter=10000)
```

Fallback path:

```python
problem.solve(solver=cp.SCS, warm_start=True, verbose=False, max_iters=2000)
```

Output selection:

- Applies first control from `uf @ g` and clips to bounds.
- Stores detailed norms and status for diagnostics.

## 10) Optional Selective DeePC Variant

### [controllers/select_dpc.py](controllers/select_dpc.py)

This file defines a DeePC variant that chooses a subset of trajectories nearest to the current state.

Config extension:

```python
@dataclass
class SelectDPCConfig(DeePCConfig):
	n_closest: int = 10
```

Trajectory selection logic:

```python
def _select_trajectories(self, x_now: float) -> List[Trajectory]:
	distances = []
	for tr in self.trajectories:
		dist = float(np.min(np.abs(tr.x - x_now)))
		distances.append(dist)

	idx = np.argsort(np.asarray(distances))
	n_pick = int(max(1, min(self.select_config.n_closest, len(self.trajectories))))
	selected = [self.trajectories[i] for i in idx[:n_pick]]
	return selected
```

Purpose:

- Reduce data library online and bias predictions toward locally relevant behavior.

## 11) Simulation Engine and Diagnostics

### [simulation.py](simulation.py)

Result containers:

```python
@dataclass
class SimulationResult:
	x: np.ndarray
	y: np.ndarray
	u: np.ndarray
	rmse: float
```

```python
@dataclass
class DeePCDiagnostics:
	t: np.ndarray
	status: list[str]
	objective: np.ndarray
	had_solution: np.ndarray
	...
```

MPC simulation loop:

```python
for t in range(t_sim):
	y[t] = system.output(x[t])
	r_future = r[t : t + controller.config.t_f]
	u[t] = controller.compute_control(x_now=x[t], r_future=r_future)
	x[t + 1] = system.step(x[t], u[t])
```

DeePC simulation loop highlights:

```python
y_for_input_times = y_hist[1:] if y_hist.shape[0] > 1 else np.array([], dtype=float)
r_future = np.sin(0.1 * (t + np.arange(controller.config.t_f, dtype=float)))
u_t, step_info = controller.compute_control_with_info(...)
diag_steps.append(step_info)
```

Important note:

- `r_future` inside DeePC simulation is regenerated sinusoidally in code, matching the same sinusoid pattern used by `make_reference`.

## 12) Plot Generation

### [plotting.py](plotting.py)

Main public function:

```python
def plot_results(
	results: Dict[str, SimulationResult],
	r: np.ndarray,
	output_dir: str = "graphs",
	deepc_diagnostics: DeePCDiagnostics | None = None,
	dataset: list[Trajectory] | None = None,
	top_g_to_plot: int = 12,
) -> Path:
```

Always generated figures:

- outputs_vs_reference.png
- control_inputs.png
- tracking_error.png

Extra DeePC diagnostic figures when diagnostics exist:

- deepc_internal_metrics.png
- deepc_g_heatmap.png
- deepc_prediction_alignment.png
- deepc_prediction_mismatch.png
- deepc_vs_dataset_trajectories.png

The helper `_plot_deepc_diagnostics` produces residual, norm, coefficient, and prediction-alignment visualizations.

## 13) Dependency List

### [requirements.txt](requirements.txt)

```text
numpy>=1.24
cvxpy>=1.4
matplotlib>=3.7
```

Dependency roles:

- NumPy: arrays and numeric operations.
- CVXPY: optimization modeling layer and solver calls.
- Matplotlib: result visualization.

## 14) README Overview

### [README.md](README.md)

The README provides:

- project purpose,
- system equations,
- install command,
- run command,
- generated plot names,
- key tunable parameters.

Basic commands documented there:

```bash
pip install -r requirements.txt
python main.py
```

## 15) Runtime Output Directory

### [graphs/](graphs)

Purpose:

- Stores PNG plots produced by [plotting.py](plotting.py).

The folder is generated/updated by running [main.py](main.py).

## 16) Cache and Environment Directories

### [__pycache__/](__pycache__)
### [controllers/__pycache__/](controllers/__pycache__)

Purpose:

- Python bytecode caches for faster imports.

### [.venv/](.venv)

Purpose:

- Local virtual environment containing interpreter and installed packages.

### [.git/](.git)

Purpose:

- Git repository metadata and version history.

## 17) End-to-End Execution Summary

Putting all files together:

1. [main.py](main.py) reads [SimulationConfig](main.py).
2. [system.py](system.py) provides plant dynamics.
3. [data.py](data.py) builds dataset and Hankel-ready trajectory data.
4. [controllers/mpc.py](controllers/mpc.py) and [controllers/deepc.py](controllers/deepc.py) compute control actions.
5. [simulation.py](simulation.py) runs closed-loop loops and collects diagnostics.
6. [plotting.py](plotting.py) saves all plots in [graphs/](graphs).
7. [README.md](README.md) and [requirements.txt](requirements.txt) document usage and dependencies.

## 18) Mathematical Summary of Both Controllers

Plant:

$$x_{k+1} = x_k + 0.5\sin(x_k) + u_k, \quad y_k = x_k$$

MPC local linear model at step $k$:

$$x_{k+1} \approx a_k x_k + u_k + c_k$$
$$a_k = 1 + 0.5\cos(\bar{x}_k), \quad c_k = \bar{f}_k - a_k\bar{x}_k$$

MPC stage cost:

$$\ell_k = (x_{k+1} - r_k)^2 + w_u u_k^2$$

DeePC behavior constraints:

$$u_f = U_f g, \quad y_f = Y_f g$$
$$U_p g + s_u = u_{ini}, \quad Y_p g + s_y = y_{ini}$$

DeePC objective:

$$\|y_f-r\|_2^2 + \lambda_u\|u_f\|_2^2 + \lambda_g\|g\|_2^2 + \lambda_{ini,u}\|s_u\|_2^2 + \lambda_{ini,y}\|s_y\|_2^2$$

## 19) Glossary

- Horizon (`t_f`): number of future steps optimized each control move.
- Initial window (`t_ini`): number of past samples enforced in DeePC matching.
- Hankel matrix: stacked shifted windows from time-series data.
- RMSE: root mean square error, a tracking-performance metric.
- Warm start: using previous solve information to help the next optimization.

## 20) Conclusion

This repository is a complete comparison framework for nonlinear tracking with model-based and data-driven predictive control. Every source file contributes a specific stage of the workflow: model, data, optimization, simulation, and visualization. The structure is compact and suitable for experimentation with horizons, penalties, dataset size, and solver settings.
