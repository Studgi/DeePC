from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cvxpy as cp
import numpy as np
from cvxpy.constraints.constraint import Constraint


@dataclass
class MPCConfig:
    """Container for all MPC tuning and actuator-limit settings.

    Attributes:
        t_f: Prediction horizon length (number of future control moves optimized).
        u_min: Lower bound of admissible control input.
        u_max: Upper bound of admissible control input.
        u_weight: Penalty weight on input effort in the stage cost.
    """

    # Number of steps in the finite prediction horizon.
    t_f: int
    # Hard lower input constraint used in the optimization problem.
    u_min: float
    # Hard upper input constraint used in the optimization problem.
    u_max: float
    # Regularization on control magnitude (bigger -> smoother/smaller inputs).
    u_weight: float = 0.1


class MPCController:
    def __init__(self, system: Any, config: MPCConfig) -> None:
        # Nonlinear plant model that provides x_{k+1} = f(x_k) + u_k behavior via step().
        self.system = system
        # User-selected MPC settings (horizon, limits, cost weight).
        self.config = config
        # Previous optimal input plan, reused to build the nominal trajectory.
        # This acts like a warm-start trajectory for successive linearization.
        self._last_u_plan = np.zeros(self.config.t_f, dtype=float)

    def _nominal_rollout(self, x0: float, current_v: float) -> np.ndarray:
        """Roll out a nominal state trajectory using the previous input plan.

        The MPC problem linearizes the nonlinear dynamics around this nominal
        trajectory, so we first simulate the model forward from current state x0.
        """

        # Nominal states x_bar[0], ..., x_bar[t_f].
        x_bar = np.zeros(self.config.t_f + 1, dtype=float)
        # Anchor the rollout at the currently measured state.
        x_bar[0] = x0
        # Forward-simulate one step at a time using last cycle's planned inputs.
        for k in range(self.config.t_f):
            x_bar[k + 1] = self.system.step(x_bar[k], self._last_u_plan[k], v=current_v)
        # Return the trajectory used as linearization reference.
        return x_bar

    def compute_control(self, x_now: float, r_future: np.ndarray, current_v: float = None) -> float:
        """Compute the first MPC action for the current state.

        Args:
            x_now: Current measured state.
            r_future: Desired future reference trajectory.
            current_v: The currently measured velocity (used as constant over horizon).

        Returns:
            The first control move from the optimized input sequence.
        """
        
        # If no velocity provided, fall back to the system's baseline.
        if current_v is None:
            current_v = getattr(self.system, "v", 2.0)

        # Horizon shorthand for readability.
        t_f = self.config.t_f
        if r_future.shape[0] < t_f:
            pad_val = r_future[-1] if r_future.size > 0 else 0.0
            r = np.pad(r_future, (0, t_f - r_future.shape[0]), constant_values=pad_val)
        else:
            r = r_future[:t_f]

        x_bar = self._nominal_rollout(x_now, current_v)

        dt = getattr(self.system, "dt", 0.1)
        L = getattr(self.system, "L", 2.5)
        # We assume Velocity stays constant at current_v across the predicted LTV horizon
        alpha = dt * (current_v / L)

        x = cp.Variable(t_f + 1)
        u = cp.Variable(t_f)

        cost = 0.0
        constraints: list[Constraint] = [x[0] == x_now]

        for k in range(t_f):
            u_bar_k = self._last_u_plan[k]
            # Linearize tan(u) around u_bar_k: tan(u) approx tan(u_bar) + sec^2(u_bar)*(u - u_bar)
            tan_u_bar = np.tan(u_bar_k)
            sec2_u_bar = 1.0 / (np.cos(u_bar_k) ** 2 + 1e-6)
            
            # x_{k+1} = x_k + alpha * (tan_u_bar + sec2_u_bar * (u_k - u_bar_k))
            c_k = alpha * (tan_u_bar - sec2_u_bar * u_bar_k)
            B_k = alpha * sec2_u_bar
            
            constraints += [x[k + 1] == x[k] + B_k * u[k] + c_k]
            constraints += [u[k] >= self.config.u_min, u[k] <= self.config.u_max]
            cost += cp.square(x[k + 1] - r[k]) + self.config.u_weight * cp.square(u[k])

        # Build and solve the convex quadratic program.
        problem = cp.Problem(cp.Minimize(cost), constraints)
        try:
            # OSQP is efficient for QP; warm_start reuses prior solution information.
            problem.solve(solver=cp.OSQP, warm_start=True, verbose=False)
        except Exception:
            # Fallback: let CVXPY choose another available solver automatically.
            problem.solve(warm_start=True, verbose=False)

        # If the solver fails, return a safe neutral command.
        if u.value is None:
            return 0.0

        # Convert optimized input vector to a flat NumPy array.
        u_plan = np.asarray(u.value).reshape(-1)
        # Receding-horizon update:
        # shift left so next call starts from "the rest" of this optimal plan.
        self._last_u_plan = np.roll(u_plan, -1)
        # Fill the new tail entry by repeating the previous tail value.
        # This avoids introducing an abrupt artificial zero at the end.
        self._last_u_plan[-1] = self._last_u_plan[-2] if t_f > 1 else self._last_u_plan[-1]

        # Apply only the first input now, clipped again for extra safety.
        return float(np.clip(u_plan[0], self.config.u_min, self.config.u_max))
