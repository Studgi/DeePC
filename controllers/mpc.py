from __future__ import annotations

from dataclasses import dataclass

import cvxpy as cp
import numpy as np

from system import NonlinearSystem


@dataclass
class MPCConfig:
    t_f: int
    u_min: float
    u_max: float
    u_weight: float = 0.1


class MPCController:
    def __init__(self, system: NonlinearSystem, config: MPCConfig) -> None:
        self.system = system
        self.config = config
        self._last_u_plan = np.zeros(self.config.t_f, dtype=float)

    def _nominal_rollout(self, x0: float) -> np.ndarray:
        x_bar = np.zeros(self.config.t_f + 1, dtype=float)
        x_bar[0] = x0
        for k in range(self.config.t_f):
            x_bar[k + 1] = self.system.step(x_bar[k], self._last_u_plan[k])
        return x_bar

    def compute_control(self, x_now: float, r_future: np.ndarray) -> float:
        t_f = self.config.t_f
        if r_future.shape[0] < t_f:
            pad_val = r_future[-1] if r_future.size > 0 else 0.0
            r = np.pad(r_future, (0, t_f - r_future.shape[0]), constant_values=pad_val)
        else:
            r = r_future[:t_f]

        x_bar = self._nominal_rollout(x_now)

        a = np.zeros(t_f, dtype=float)
        c = np.zeros(t_f, dtype=float)
        for k in range(t_f):
            # First-order local linearization around x_bar[k].
            a[k] = 1.0 + 0.5 * np.cos(x_bar[k])
            f_bar = x_bar[k] + 0.5 * np.sin(x_bar[k])
            c[k] = f_bar - a[k] * x_bar[k]

        x = cp.Variable(t_f + 1)
        u = cp.Variable(t_f)

        cost = 0.0
        constraints = [x[0] == x_now]

        for k in range(t_f):
            constraints += [x[k + 1] == a[k] * x[k] + u[k] + c[k]]
            constraints += [u[k] >= self.config.u_min, u[k] <= self.config.u_max]
            cost += cp.square(x[k + 1] - r[k]) + self.config.u_weight * cp.square(u[k])

        problem = cp.Problem(cp.Minimize(cost), constraints)
        try:
            problem.solve(solver=cp.OSQP, warm_start=True, verbose=False)
        except Exception:
            problem.solve(warm_start=True, verbose=False)

        if u.value is None:
            return 0.0

        u_plan = np.asarray(u.value).reshape(-1)
        self._last_u_plan = np.roll(u_plan, -1)
        self._last_u_plan[-1] = self._last_u_plan[-2] if t_f > 1 else self._last_u_plan[-1]

        return float(np.clip(u_plan[0], self.config.u_min, self.config.u_max))
