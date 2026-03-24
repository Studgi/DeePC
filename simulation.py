from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np

from controllers.deepc import DeePCController, DeePCStepInfo
from controllers.mpc import MPCController


@dataclass
class SimulationResult:
    x: np.ndarray
    y: np.ndarray
    u: np.ndarray
    rmse: float


@dataclass
class DeePCDiagnostics:
    t: np.ndarray
    status: list[str]
    objective: np.ndarray
    had_solution: np.ndarray
    u_applied: np.ndarray
    y_now: np.ndarray
    r_now: np.ndarray
    u_ini_residual_norm: np.ndarray
    y_ini_residual_norm: np.ndarray
    g_l2_norm: np.ndarray
    g_l1_norm: np.ndarray
    g_matrix: np.ndarray
    u_plan_matrix: np.ndarray
    y_plan_matrix: np.ndarray
    y_actual_plan_matrix: np.ndarray
    r_plan_matrix: np.ndarray
    pred_actual_norm: np.ndarray


def _rmse(y: np.ndarray, r: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y - r) ** 2)))


def simulate_mpc(
    system: Any,
    controller: MPCController,
    x0: float,
    r: np.ndarray,
    t_sim: int,
) -> SimulationResult:
    x = np.zeros(t_sim + 1, dtype=float)
    y = np.zeros(t_sim, dtype=float)
    u = np.zeros(t_sim, dtype=float)

    x[0] = x0

    for t in range(t_sim):
        y[t] = system.output(x[t])
        r_future = r[t : t + controller.config.t_f]
        u[t] = controller.compute_control(x_now=x[t], r_future=r_future)
        x[t + 1] = system.step(x[t], u[t])

    return SimulationResult(x=x, y=y, u=u, rmse=_rmse(y, r))


def simulate_deepc(
    system: Any,
    controller: DeePCController,
    x0: float,
    r: np.ndarray,
    t_sim: int,
    collect_diagnostics: bool = False,
    lifted_input: bool = False,
) -> tuple[SimulationResult, DeePCDiagnostics | None]:
    x = np.zeros(t_sim + 1, dtype=float)
    y = np.zeros(t_sim, dtype=float)
    u = np.zeros(t_sim, dtype=float)

    x[0] = x0

    u_hist = np.array([], dtype=float)
    y_hist = np.array([system.output(x0)], dtype=float)

    diag_steps: list[DeePCStepInfo] = []

    for t in range(t_sim):
        y_now = system.output(x[t])
        y[t] = y_now

        # y_hist contains y_0..y_t. For DeePC with shifted output indexing,
        # align with u_0..u_{t-1} using y_1..y_t.
        y_for_input_times = y_hist[1:] if y_hist.shape[0] > 1 else np.array([], dtype=float)

        r_future = np.sin(0.1 * (t + np.arange(controller.config.t_f, dtype=float)))
        if collect_diagnostics:
            u_t, step_info = controller.compute_control_with_info(
                u_hist=u_hist,
                y_hist_for_input_times=y_for_input_times,
                y_now=y_now,
                r_future=r_future,
            )
            diag_steps.append(step_info)
        else:
            u_t = controller.compute_control(
                u_hist=u_hist,
                y_hist_for_input_times=y_for_input_times,
                y_now=y_now,
                r_future=r_future,
            )

        u[t] = u_t
        x[t + 1] = system.step(x[t], u_t)

        u_hist = np.append(u_hist, u_t)
        y_hist = np.append(y_hist, system.output(x[t + 1]))

    result = SimulationResult(x=x, y=y, u=u, rmse=_rmse(y, r))

    if not collect_diagnostics:
        return result, None

    n_col = controller.up.shape[1]
    t_f = controller.config.t_f
    status = [s.status for s in diag_steps]
    objective = np.asarray([s.objective for s in diag_steps], dtype=float)
    had_solution = np.asarray([s.had_solution for s in diag_steps], dtype=bool)
    u_applied = np.asarray([s.u_applied for s in diag_steps], dtype=float)
    y_now_all = y.copy()
    r_now_all = r.copy()
    u_ini_residual = np.asarray([s.u_ini_residual_norm for s in diag_steps], dtype=float)
    y_ini_residual = np.asarray([s.y_ini_residual_norm for s in diag_steps], dtype=float)
    g_l2 = np.asarray([s.g_l2_norm for s in diag_steps], dtype=float)
    g_l1 = np.asarray([s.g_l1_norm for s in diag_steps], dtype=float)
    g_matrix = np.vstack([s.g for s in diag_steps]) if diag_steps else np.zeros((t_sim, n_col), dtype=float)
    u_plan_matrix = np.vstack([s.u_plan for s in diag_steps]) if diag_steps else np.zeros((t_sim, t_f), dtype=float)
    y_plan_matrix = np.vstack([s.y_plan for s in diag_steps]) if diag_steps else np.zeros((t_sim, t_f), dtype=float)
    y_actual_plan_matrix = np.zeros((t_sim, t_f), dtype=float)
    pred_actual_norm = np.full(t_sim, np.nan, dtype=float)

    for i, step in enumerate(diag_steps):
        if not step.had_solution:
            continue

        x_roll = x[i]
        y_roll = np.zeros(t_f, dtype=float)
        for k in range(t_f):
            x_roll = system.step(x_roll, float(step.u_plan[k]))
            y_roll[k] = system.output(x_roll)

        y_actual_plan_matrix[i, :] = y_roll
        pred_actual_norm[i] = float(np.linalg.norm(step.y_plan - y_roll))

        if controller.config.debug:
            print(f"[DeePC-Validation] t={i} ||y_pred - y_actual||={pred_actual_norm[i]:.6e}")

    r_plan_matrix = np.vstack([s.r_plan for s in diag_steps]) if diag_steps else np.zeros((t_sim, t_f), dtype=float)

    diagnostics = DeePCDiagnostics(
        t=np.arange(t_sim, dtype=int),
        status=status,
        objective=objective,
        had_solution=had_solution,
        u_applied=u_applied,
        y_now=y_now_all,
        r_now=r_now_all,
        u_ini_residual_norm=u_ini_residual,
        y_ini_residual_norm=y_ini_residual,
        g_l2_norm=g_l2,
        g_l1_norm=g_l1,
        g_matrix=g_matrix,
        u_plan_matrix=u_plan_matrix,
        y_plan_matrix=y_plan_matrix,
        y_actual_plan_matrix=y_actual_plan_matrix,
        r_plan_matrix=r_plan_matrix,
        pred_actual_norm=pred_actual_norm,
    )
    return result, diagnostics


def run_all_simulations(
    system: Any,
    mpc: MPCController,
    deepc: DeePCController,
    x0: float,
    r: np.ndarray,
    t_sim: int,
) -> Dict[str, SimulationResult]:
    deepc_result, _ = simulate_deepc(system=system, controller=deepc, x0=x0, r=r, t_sim=t_sim, collect_diagnostics=False)
    return {
        "MPC": simulate_mpc(system=system, controller=mpc, x0=x0, r=r, t_sim=t_sim),
        "DeePC": deepc_result,
    }


def run_all_simulations_with_diagnostics(
    system: Any,
    mpc: MPCController,
    deepc: DeePCController,
    x0: float,
    r: np.ndarray,
    t_sim: int,
) -> tuple[Dict[str, SimulationResult], DeePCDiagnostics]:
    mpc_result = simulate_mpc(system=system, controller=mpc, x0=x0, r=r, t_sim=t_sim)
    deepc_result, diagnostics = simulate_deepc(
        system=system,
        controller=deepc,
        x0=x0,
        r=r,
        t_sim=t_sim,
        collect_diagnostics=True,
    )
    if diagnostics is None:
        raise RuntimeError("Expected DeePC diagnostics, but none were collected.")

    return {"MPC": mpc_result, "DeePC": deepc_result}, diagnostics
