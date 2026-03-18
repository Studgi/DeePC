from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import cvxpy as cp
import numpy as np

from data import Trajectory, build_hankel_from_trajectories


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


class DeePCController:
    def __init__(self, trajectories: List[Trajectory], config: DeePCConfig) -> None:
        self.trajectories = trajectories
        self.config = config
        self._last_u = 0.0
        self.up, self.uf, self.yp, self.yf = self._build_hankels(trajectories)
        self.h_u = np.vstack([self.up, self.uf])
        self.h_y = np.vstack([self.yp, self.yf])
        self.h_u_rank = int(np.linalg.matrix_rank(self.h_u))
        self.h_y_rank = int(np.linalg.matrix_rank(self.h_y))

        if self.config.debug:
            print(f"[DeePC] Hankel rank Hu: {self.h_u_rank}/{self.h_u.shape[0]}")
            print(f"[DeePC] Hankel rank Hy: {self.h_y_rank}/{self.h_y.shape[0]}")
            print(f"[DeePC] Hankel columns used: {self.up.shape[1]}")

    def _build_hankels(self, trajectories: List[Trajectory]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        t_ini = self.config.t_ini
        t_f = self.config.t_f
        total = t_ini + t_f

        hu = build_hankel_from_trajectories(trajectories, signal_name="u", window=total)
        hy = build_hankel_from_trajectories(trajectories, signal_name="y", window=total)

        n_col = hu.shape[1]
        if self.config.max_columns > 0 and n_col > self.config.max_columns:
            idx = np.linspace(0, n_col - 1, num=self.config.max_columns, dtype=int)
            hu = hu[:, idx]
            hy = hy[:, idx]

        up = hu[:t_ini, :]
        uf = hu[t_ini:, :]
        yp = hy[:t_ini, :]
        yf = hy[t_ini:, :]
        return up, uf, yp, yf

    def _prepare_ini(self, seq: np.ndarray, target_len: int, pad_value: float = 0.0) -> np.ndarray:
        if seq.shape[0] >= target_len:
            return seq[-target_len:].astype(float)
        pad_len = target_len - seq.shape[0]
        return np.concatenate([np.full(pad_len, pad_value, dtype=float), seq.astype(float)])

    def _solve_with_hankels(
        self,
        up: np.ndarray,
        uf: np.ndarray,
        yp: np.ndarray,
        yf: np.ndarray,
        u_hist: np.ndarray,
        y_hist_for_input_times: np.ndarray,
        y_now: float,
        r_future: np.ndarray,
        return_info: bool = False,
    ) -> float | tuple[float, DeePCStepInfo]:
        t_ini = self.config.t_ini
        t_f = self.config.t_f

        if r_future.shape[0] < t_f:
            pad_val = r_future[-1] if r_future.size > 0 else 0.0
            r = np.pad(r_future, (0, t_f - r_future.shape[0]), constant_values=pad_val)
        else:
            r = r_future[:t_f]

        u_ini = self._prepare_ini(u_hist, t_ini, pad_value=0.0)
        y_ini = self._prepare_ini(y_hist_for_input_times, t_ini, pad_value=y_now)

        n_col = up.shape[1]
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

        if self.config.use_soft_ini:
            constraints += [
                up @ g + s_u == u_ini,
                yp @ g + s_y == y_ini,
            ]
        else:
            constraints += [
                up @ g == u_ini,
                yp @ g == y_ini,
                s_u == 0.0,
                s_y == 0.0,
            ]

        cost = 0.0
        cost += cp.sum_squares(y_f - r)
        cost += self.config.lambda_u * cp.sum_squares(u_f)
        cost += self.config.lambda_g * cp.sum_squares(g)
        cost += self.config.lambda_ini_u * cp.sum_squares(s_u)
        cost += self.config.lambda_ini_y * cp.sum_squares(s_y)

        problem = cp.Problem(cp.Minimize(cost), constraints)
        try:
            problem.solve(solver=cp.OSQP, warm_start=True, verbose=False, max_iter=10000)
        except Exception:
            try:
                problem.solve(solver=cp.SCS, warm_start=True, verbose=False, max_iters=2000)
            except Exception:
                pass

        had_solution = u_f.value is not None and y_f.value is not None and g.value is not None
        status = str(problem.status)
        objective = float(problem.value) if problem.value is not None else float("nan")

        if had_solution:
            u_plan = np.asarray(u_f.value).reshape(-1)
            y_plan = np.asarray(y_f.value).reshape(-1)
            g_vec = np.asarray(g.value).reshape(-1)
            u_from_g = np.asarray(uf @ g_vec).reshape(-1)
            u_applied = float(np.clip(u_from_g[0], self.config.u_min, self.config.u_max))
            self._last_u = u_applied
            u_ini_residual_norm = float(np.linalg.norm(up @ g_vec - u_ini))
            y_ini_residual_norm = float(np.linalg.norm(yp @ g_vec - y_ini))
            g_l2_norm = float(np.linalg.norm(g_vec, 2))
            g_l1_norm = float(np.linalg.norm(g_vec, 1))
            tracking_norm = float(np.linalg.norm(yf @ g_vec - r))

            if self.config.debug:
                print(
                    "[DeePC] "
                    f"status={status} "
                    f"||Yf g - r||={tracking_norm:.6e} "
                    f"||g||={g_l2_norm:.6e} "
                    f"u0={u_applied:.6e} "
                    f"||Upg-u||={u_ini_residual_norm:.3e} "
                    f"||Ypg-y||={y_ini_residual_norm:.3e}"
                )
        else:
            u_plan = np.zeros(t_f, dtype=float)
            y_plan = np.zeros(t_f, dtype=float)
            g_vec = np.zeros(n_col, dtype=float)
            u_applied = float(np.clip(self._last_u, self.config.u_min, self.config.u_max))
            u_ini_residual_norm = float("nan")
            y_ini_residual_norm = float("nan")
            g_l2_norm = float("nan")
            g_l1_norm = float("nan")

        if not return_info:
            return u_applied

        info = DeePCStepInfo(
            status=status,
            objective=objective,
            u_applied=u_applied,
            u_plan=u_plan,
            y_plan=y_plan,
            r_plan=r,
            g=g_vec,
            u_ini_residual_norm=u_ini_residual_norm,
            y_ini_residual_norm=y_ini_residual_norm,
            g_l2_norm=g_l2_norm,
            g_l1_norm=g_l1_norm,
            had_solution=had_solution,
        )
        return u_applied, info

    def compute_control(
        self,
        u_hist: np.ndarray,
        y_hist_for_input_times: np.ndarray,
        y_now: float,
        r_future: np.ndarray,
    ) -> float:
        return self._solve_with_hankels(
            up=self.up,
            uf=self.uf,
            yp=self.yp,
            yf=self.yf,
            u_hist=u_hist,
            y_hist_for_input_times=y_hist_for_input_times,
            y_now=y_now,
            r_future=r_future,
            return_info=False,
        )

    def compute_control_with_info(
        self,
        u_hist: np.ndarray,
        y_hist_for_input_times: np.ndarray,
        y_now: float,
        r_future: np.ndarray,
    ) -> tuple[float, DeePCStepInfo]:
        return self._solve_with_hankels(
            up=self.up,
            uf=self.uf,
            yp=self.yp,
            yf=self.yf,
            u_hist=u_hist,
            y_hist_for_input_times=y_hist_for_input_times,
            y_now=y_now,
            r_future=r_future,
            return_info=True,
        )
