# Complete Beginner Guide: Entire Project Explained (All Files)

## 1) Goal of This Document

This guide explains all files and folders in the repository in beginner-friendly language.

The focus is explanation first:

- what each file does,
- why it exists,
- how data moves through it,
- what non-basic logic means,
- how all modules connect end to end.

Imports are intentionally omitted from explanations and code snippets.

## 2) Repository Map and Meaning

Covered items:

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

## 3) High-Level Project Workflow

The project compares two predictive controllers on the same nonlinear system.

Execution pipeline:

1. Build plant model from [system.py](system.py).
2. Generate many trajectories in [data.py](data.py).
3. Build one MPC controller and one DeePC controller.
4. Simulate both controllers in [simulation.py](simulation.py).
5. Compute RMSE for each method.
6. Produce plots in [plotting.py](plotting.py) and save to [graphs/](graphs).

This structure is modular: each file owns one stage of the pipeline.

## 4) Root Package File

### [__init__.py](__init__.py)

Current state: empty.

Why it exists:

- It marks the repository root as a Python package context when needed.
- It can later hold package-level exports or metadata.

No runtime logic is currently implemented here.

## 5) Main Orchestration

### [main.py](main.py)

This is the execution entry point.

Main responsibilities:

- define experiment parameters,
- build model and dataset,
- create controllers,
- run closed-loop simulation,
- print metrics,
- save figures.

### 5.1 Configuration object

The SimulationConfig dataclass stores all high-impact knobs:

- data volume: n_data, t_data,
- controller memory and horizon: t_ini, t_f,
- closed-loop length: t_sim,
- initial condition: x0,
- actuator limits: u_min, u_max,
- MPC effort penalty: u_weight.

Why this matters:

- Centralized parameters make experiments reproducible.
- Changing one field modifies behavior across the pipeline.

### 5.2 Runtime sequence in plain language

Inside main:

1. A NonlinearSystem object is created with input bounds.
2. Random dataset trajectories are generated.
3. Dataset min and max ranges are printed.
4. MPC and DeePC controllers are configured.
5. A sinusoidal reference is created.
6. Both controllers are simulated.
7. RMSE values are printed.
8. Plot files are generated in [graphs/](graphs).

This file is the only script needed for normal execution.

## 6) Plant Model

### [system.py](system.py)

This file defines the physical or process model used in simulation.

State update equation:

$$x_{t+1} = x_t + 0.5\sin(x_t) + u_t$$

Output equation:

$$y_t = x_t$$

### 6.1 Important method behavior

- step(x, u):
  - clips u to configured bounds,
  - applies nonlinear update,
  - returns next state.

- output(x):
  - returns the measured output,
  - here it is the state itself.

Why clipping is important:

- It enforces actuator limits consistently,
- both data generation and simulation obey the same physical constraints.

## 7) Data Generation and Hankel Utilities

### [data.py](data.py)

This file is foundational for DeePC because DeePC learns behavior from trajectory data.

### 7.1 Trajectory container

Trajectory stores one rollout with aligned arrays:

- x with length T+1,
- y with length T+1,
- u with length T.

The extra state/output sample exists because each control action moves the state forward one step.

### 7.2 Rollout logic

rollout(system, x0, u_seq):

- starts from x0,
- applies each control in u_seq,
- stores state and output at each time,
- returns one complete Trajectory.

Why this function is reused:

- It guarantees consistent simulation logic for every generated trajectory.

### 7.3 Random dataset creation

generate_dataset(...):

- samples random control sequences in [u_min, u_max],
- samples random initial states in [x0_min, x0_max],
- calls rollout repeatedly,
- returns a list of trajectories.

Why multiple trajectories are needed:

- DeePC needs broad behavioral coverage, not one single path.

### 7.4 Range summary

summarize_dataset_ranges(...) computes global min and max of:

- all control values,
- all aligned outputs.

The output uses y[1:] so outputs align with input timing in DeePC matrix construction.

### 7.5 Hankel matrix construction

Two helper functions build the data blocks used by DeePC.

Concept:

- A Hankel matrix stacks shifted windows of a time-series.
- Each column is one local trajectory fragment.

build_hankel(signal, window):

- verifies window length,
- constructs all sliding windows,
- stacks them into a matrix.

build_hankel_from_trajectories(...):

- loops over trajectories,
- extracts either input or output signal,
- aligns output by dropping first sample,
- builds one Hankel per trajectory,
- concatenates across trajectories.

Why this is core to DeePC:

- DeePC prediction is expressed as linear combinations of these data columns.

### 7.6 Reference generator

make_reference(t_sim) returns:

$$r_t = \sin(0.1t)$$

This is the target signal tracked by both controllers.

## 8) Controller Package Export File

### [controllers/__init__.py](controllers/__init__.py)

Purpose:

- Re-exports controller classes for cleaner package imports.
- Defines the public names exposed by the controllers package.

No control logic is implemented here.

## 9) MPC Controller Explained

### [controllers/mpc.py](controllers/mpc.py)

This file implements model-based predictive control.

### 9.1 Config object

MPCConfig stores:

- horizon length t_f,
- actuator bounds,
- control penalty weight u_weight.

### 9.2 Core idea

MPC algorithm in this file:

1. Build a nominal state rollout using the previous control plan.
2. Linearize nonlinear dynamics around that nominal trajectory.
3. Solve a quadratic optimization problem over horizon t_f.
4. Apply first control move only (receding horizon).

### 9.3 Why nominal rollout exists

The system is nonlinear, but the QP solver needs a linear model.

Nominal rollout provides expansion points x_bar[k] so local linear parameters can be computed.

### 9.4 Linearization details

At each horizon step k, the model is approximated by:

$$x_{k+1} \approx a_k x_k + u_k + c_k$$

with

$$a_k = 1 + 0.5\cos(\bar{x}_k)$$

and an offset c_k that keeps the approximation consistent at the nominal point.

Meaning:

- a_k is local slope of nonlinear dynamics,
- c_k corrects bias from linearization.

### 9.5 Optimization objective and constraints

The QP minimizes tracking error plus input effort:

$$\sum_k (x_{k+1}-r_k)^2 + w_u u_k^2$$

subject to:

- dynamic constraints from local linear model,
- input lower and upper bounds,
- initial state equality.

### 9.6 Returned control and warm behavior

After solving:

- first action u_plan[0] is applied,
- remaining plan is shifted and reused as previous guess next step.

This gives temporal smoothness and speed improvements in repeated solves.

## 10) DeePC Controller Explained

### [controllers/deepc.py](controllers/deepc.py)

This file implements data-enabled predictive control using trajectory data directly.

### 10.1 Config object

DeePCConfig includes:

- memory window t_ini,
- prediction horizon t_f,
- input limits,
- regularization weights,
- option for soft initial consistency,
- option to force convex coefficients,
- max Hankel columns,
- debug logging toggle.

### 10.2 Diagnostic object

DeePCStepInfo stores per-step internals such as:

- solver status,
- objective value,
- applied input,
- planned trajectories,
- coefficient vector g,
- residual norms,
- solution existence flag.

This object powers post-run diagnostics and plots.

### 10.3 Hankel preparation

During initialization, Hankel matrices are built and split into:

- Up, Uf for past/future inputs,
- Yp, Yf for past/future outputs.

Why split is needed:

- past blocks enforce consistency with recent history,
- future blocks generate planned control/output trajectories.

### 10.4 Online DeePC optimization variables

Each control step optimizes:

- g: mixing weights over data columns,
- u_f: future inputs,
- y_f: future outputs,
- s_u, s_y: slack variables for soft history matching.

### 10.5 DeePC constraints

Behavior constraints:

$$u_f = U_f g, \quad y_f = Y_f g$$

Input bounds:

$$u_{min} \le u_f \le u_{max}$$

Optional convexity constraints:

$$g \ge 0, \quad \sum g = 1$$

Soft past matching:

$$U_p g + s_u = u_{ini}, \quad Y_p g + s_y = y_{ini}$$

Interpretation:

- predicted trajectories must lie in the data-driven behavior space,
- slacks absorb mismatch when exact consistency is impossible.

### 10.6 DeePC objective function

The cost combines:

- tracking error,
- input effort,
- coefficient regularization,
- penalties on slack magnitudes.

Form:

$$\|y_f-r\|_2^2 + \lambda_u\|u_f\|_2^2 + \lambda_g\|g\|_2^2 + \lambda_{ini,u}\|s_u\|_2^2 + \lambda_{ini,y}\|s_y\|_2^2$$

Role of each term:

- tracking term drives performance,
- input penalty avoids overly aggressive actions,
- g penalty discourages extreme coefficient values,
- slack penalties enforce history consistency strength.

### 10.7 Solver execution and fallback

Primary solver call uses OSQP with warm start and iteration cap.

Fallback solver SCS is attempted only if the first solve throws an exception.

Important nuance:

- status values like user_limit can still produce numeric candidate solutions,
- had_solution checks whether variable values were actually returned.

### 10.8 Output applied to plant

If solve is successful:

- compute u from Uf g,
- apply first element after clipping,
- cache diagnostics.

If no numeric solution is available:

- reuse last applied control within bounds,
- fill diagnostic fields with safe defaults.

## 11) Selective DeePC Variant

### [controllers/select_dpc.py](controllers/select_dpc.py)

This file introduces a local-data DeePC variation.

Key idea:

- do not use all trajectories each step,
- choose only trajectories close to current state.

### 11.1 Selection metric

For each stored trajectory:

- compute minimum absolute distance between current state and any trajectory state sample,
- rank trajectories by this distance,
- keep n_closest trajectories.

### 11.2 Solve path

- build Hankel blocks from selected trajectories,
- if selection fails (for example insufficient window), fall back to full Hankel set,
- solve using inherited DeePC logic.

Potential benefit:

- local data can improve relevance and reduce problem size.

Potential risk:

- too few selected trajectories can reduce excitation diversity.

## 12) Simulation Engine

### [simulation.py](simulation.py)

This file executes both controllers against the same plant and computes metrics.

### 12.1 Result classes

SimulationResult stores:

- state trajectory x,
- output trajectory y,
- control sequence u,
- RMSE metric.

DeePCDiagnostics stores rich per-step internals for analysis and plotting.

### 12.2 RMSE helper

_rmse(y, r) computes

$$\sqrt{\text{mean}((y-r)^2)}$$

This gives one scalar tracking score per controller.

### 12.3 MPC simulation loop

At each time step:

1. measure current output,
2. slice future reference,
3. request control from MPC,
4. step plant forward.

### 12.4 DeePC simulation loop

At each time step:

1. measure current output,
2. align output history with input history,
3. build future reference segment,
4. request DeePC control (with or without diagnostics),
5. step plant,
6. append history arrays.

### 12.5 Post-processing diagnostics

When diagnostics are enabled:

- collect status/objective/norm trends,
- stack all g vectors into a matrix,
- stack predicted plans,
- roll out predicted control plan through true nonlinear plant,
- compute prediction-vs-rollout mismatch norms.

This creates insight beyond final RMSE.

### 12.6 Public simulation wrappers

- run_all_simulations: returns results only.
- run_all_simulations_with_diagnostics: returns results plus DeePCDiagnostics.

## 13) Plotting and Visualization

### [plotting.py](plotting.py)

This file converts numeric results into figures.

### 13.1 Core comparison figures

Generated for both controllers:

- output tracking against reference,
- control input trajectories,
- tracking error trajectories.

### 13.2 DeePC internal figures

Generated when diagnostics are provided:

- applied input and objective trends,
- residual and norm trends,
- heatmap of strongest g coefficients,
- predicted vs measured one-step alignment,
- prediction mismatch norms,
- closed-loop trajectories over dataset trajectories.

### 13.3 Top-g feature

Averages absolute value of each g component over time,
then plots only top-ranked coefficients for readability.

## 14) Project Documentation File

### [README.md](README.md)

This file provides:

- project purpose,
- system equations,
- install/run commands,
- output plot names,
- key configurable parameters.

It is concise and oriented for quick start.

## 15) Dependency File

### [requirements.txt](requirements.txt)

Dependencies:

- numpy,
- cvxpy,
- matplotlib.

Why each exists:

- numerical arrays and vectorized math,
- convex optimization modeling and solving,
- plotting and figure export.

## 16) Output and Support Folders

### [graphs/](graphs)

Stores generated PNG result files.

### [__pycache__/](__pycache__) and [controllers/__pycache__/](controllers/__pycache__)

Python bytecode caches automatically produced by interpreter.

### [.venv/](.venv)

Local virtual environment containing project-specific packages and interpreter binaries.

### [.git/](.git)

Git metadata for version control (commits, branches, history).

## 17) Full End-to-End Data Flow

Step-by-step data movement:

1. [main.py](main.py) sets parameters and builds objects.
2. [data.py](data.py) creates trajectory library from random controls.
3. [controllers/deepc.py](controllers/deepc.py) transforms trajectories into Hankel blocks.
4. [simulation.py](simulation.py) repeatedly asks each controller for control action.
5. [system.py](system.py) advances nonlinear plant state.
6. [simulation.py](simulation.py) accumulates trajectories and metrics.
7. [plotting.py](plotting.py) renders and saves visual summaries.

This loop is the core experiment architecture.

## 18) Beginner Interpretation Notes

Key conceptual differences:

- MPC uses explicit model equations each step.
- DeePC uses historical data structure instead of explicit model inside optimization.

Practical consequences:

- MPC quality depends on model quality and linearization quality.
- DeePC quality depends strongly on data richness, scaling, and regularization.

Both methods solve optimization repeatedly in a receding-horizon pattern.

## 19) Glossary

- Horizon: number of future steps optimized each control step.
- Initial window: number of recent samples used to anchor DeePC prediction.
- Hankel matrix: matrix made of stacked shifted windows of time-series data.
- Receding horizon: apply first planned control, then re-optimize at next step.
- RMSE: root mean square tracking error.
- Warm start: reusing previous optimization information to speed repeated solves.

## 20) Final Summary

The repository is a compact but complete control comparison framework.

- [system.py](system.py) defines the nonlinear plant.
- [data.py](data.py) provides trajectory generation and Hankel utilities.
- [controllers/mpc.py](controllers/mpc.py) implements model-based predictive control.
- [controllers/deepc.py](controllers/deepc.py) implements data-enabled predictive control with diagnostics.
- [controllers/select_dpc.py](controllers/select_dpc.py) provides a local trajectory selection variant.
- [simulation.py](simulation.py) runs closed-loop experiments and computes metrics.
- [plotting.py](plotting.py) saves comparison and diagnostic figures.
- [main.py](main.py) orchestrates the entire run.

All files and folders in the workspace now have explicit explanation in this guide.
