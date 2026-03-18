# Beginner Report: Why You See "Solution May Be Inaccurate"

## 1) Executive Summary

Your code is working, but most DeePC optimization steps are ending with a "good-enough" numerical answer instead of a fully converged one.

That is why CVXPY prints:

"Solution may be inaccurate. Try another solver, adjusting the solver settings..."

From your run, the solver status counts were:

- optimal: 4
- user_limit: 116

So DeePC still produced controls on every step, but in 116 steps OSQP stopped because it hit its iteration budget before reaching strict tolerances.

This can hurt control quality, and that matches your RMSE:

- MPC: 0.071866
- DeePC: 0.903015

## 2) What This Project Is Doing (High Level)

You compare two controllers on a nonlinear system:

- MPC (model-based)
- DeePC (data-driven)

Main flow is in [main.py](main.py):

1. Build nonlinear system.
2. Generate random input-output dataset.
3. Build MPC and DeePC controllers.
4. Simulate closed loop.
5. Print RMSE and save plots.

## 3) The System You Control

Defined in [system.py](system.py), the dynamics are:

$$x_{t+1} = x_t + 0.5\sin(x_t) + u_t$$
$$y_t = x_t$$

Important beginner point:

- This is nonlinear because of $\sin(x_t)$.
- Nonlinearity makes data-driven prediction and optimization numerically harder.

## 4) How Data Is Generated

In [data.py](data.py), you create many random trajectories:

- Inputs sampled in about [-2, 2]
- Different random initial states
- Outputs can become much larger (your printed range was about [-25, 29])

Why this matters:

- Inputs and outputs are on different scales.
- Big scale differences can make optimization less numerically friendly.

## 5) How DeePC Works in This Code

Main DeePC implementation is in [controllers/deepc.py](controllers/deepc.py).

### 5.1 Hankel Matrices

DeePC builds past/future block matrices from dataset windows:

- $U_p, U_f, Y_p, Y_f$

These are created in [_build_hankels](controllers/deepc.py#L54) and used to represent trajectories as a combination of dataset columns.

### 5.2 Decision Variables

Inside [_solve_with_hankels](controllers/deepc.py#L82), optimization solves for:

- $g$: coefficients over data columns
- $u_f$: future control sequence
- $y_f$: future output sequence
- $s_u, s_y$: slack variables for soft initial matching

### 5.3 Constraints

Key constraints:

- $u_f = U_f g$
- $y_f = Y_f g$
- input bounds on $u_f$
- if convex mode: $g \ge 0$ and $\sum g = 1$
- soft matching of initial window via slacks

### 5.4 Cost Function

You minimize:

$$\|y_f-r\|_2^2 + \lambda_u\|u_f\|_2^2 + \lambda_g\|g\|_2^2 + \lambda_{ini,u}\|s_u\|_2^2 + \lambda_{ini,y}\|s_y\|_2^2$$

with weights from [main.py](main.py#L64).

## 6) Why the Warning Appears

The warning is triggered at solve time in [controllers/deepc.py](controllers/deepc.py#L148):

- OSQP is called with max_iter=10000.
- Many steps end with status user_limit.

What "user_limit" means (beginner version):

- "I found a candidate solution, but stopped because iteration/time limits were reached before full accuracy target."

So the warning does not mean immediate failure. It means numerical confidence is lower than ideal.

## 7) Why This Happens in Your Specific Setup

Several factors combine:

1. Problem size
- You allow up to 600 Hankel columns.
- That makes the optimization larger each control step.

2. Scaling mismatch
- Input is around [-2, 2], output around [-25, 29].
- Different magnitudes can slow ADMM-based convergence.

3. Weight magnitudes differ a lot
- Some penalties are tiny (1e-4), others large (1e2).
- Mixed scales can make conditioning harder.

4. Nonlinear plant vs linear behavioral approximation
- DeePC predicts from data combinations, but true plant is nonlinear.
- This can increase tension between constraints and tracking quality.

## 8) Interpreting Your RMSE Numbers

From your run:

- MPC RMSE is low (good tracking).
- DeePC RMSE is much higher.

This means DeePC currently underperforms, and the frequent low-accuracy solver terminations are likely one contributor.

Other contributors can also exist (data richness, horizon choices, regularization, column selection), but solver quality is clearly part of the story because 116 of 120 solves are user_limit.

## 9) Is the Warning Dangerous?

Short answer: not always, but you should treat it as a reliability warning.

- If occasional: often acceptable.
- If frequent (your case): performance can drift, and comparisons become less trustworthy.

## 10) Beginner-Friendly Fix Strategy (Ordered)

1. Improve solver robustness first
- Increase max_iter further.
- Set explicit tolerances (eps_abs, eps_rel).
- Enable polish.
- If status is user_limit or optimal_inaccurate, retry with another solver.

2. Normalize signals for DeePC
- Scale dataset u and y to similar ranges before building Hankel matrices.
- Unscale control/output when applying/reporting.

3. Reduce problem size
- Lower max_columns (for example 300-400) and compare.
- Smaller QP often converges more reliably.

4. Tune regularization
- Adjust lambda_g, lambda_ini_u, lambda_ini_y to avoid ill-conditioning and over-constraining.

5. Re-check data coverage
- Ensure trajectories excite relevant dynamics near operating region.

## 11) What You Can Trust Right Now

You can trust these facts from current evidence:

- The warning is real and frequent.
- DeePC is returning numeric solutions every step (had_solution_rate = 1.0).
- Most solutions are not fully converged to strict tolerance under current OSQP settings.
- Current DeePC performance is significantly worse than MPC in this benchmark.

## 12) File Map (Where to Look)

- Main experiment setup: [main.py](main.py)
- DeePC optimization and solve call: [controllers/deepc.py](controllers/deepc.py)
- Simulation and diagnostics collection: [simulation.py](simulation.py)
- Data generation and Hankel construction: [data.py](data.py)
- Plots and diagnostics figures: [plotting.py](plotting.py)

## 13) Glossary (Beginner)

- QP: Quadratic Program, an optimization problem with quadratic cost and linear constraints.
- OSQP: A fast QP solver using ADMM iterations.
- CVXPY: Python modeling layer that sends your QP to solvers.
- ADMM: Iterative optimization method; may need many iterations for hard/scaled problems.
- user_limit: Solver stopped at configured iteration/resource limit.
- optimal: Solver reached requested convergence criteria.
- DeePC: Data-enabled predictive control, control directly from measured trajectories.
- RMSE: Root Mean Square Error, average tracking error magnitude.

## 14) Final Beginner Takeaway

Your project is functioning correctly, but the DeePC optimization is usually terminating early in numerical terms. That is exactly why CVXPY warns about potential inaccuracy. The warning is not a crash; it is a quality flag. In your current experiment, this aligns with weaker DeePC tracking performance.
