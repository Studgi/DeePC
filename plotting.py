from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np

from data import Trajectory
from simulation import DeePCDiagnostics, SimulationResult


def _plot_deepc_diagnostics(
    out_dir: Path,
    results: Dict[str, SimulationResult],
    deepc_diag: DeePCDiagnostics,
    dataset: list[Trajectory] | None,
    top_idx: np.ndarray,
) -> None:
    t = deepc_diag.t.astype(float)

    fig1, axes = plt.subplots(4, 1, figsize=(11, 10), sharex=True)
    axes[0].plot(t, deepc_diag.u_applied, label="u_deepc", linewidth=1.6)
    axes[0].set_ylabel("u")
    axes[0].set_title("DeePC Applied Input")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(t, deepc_diag.objective, label="objective", linewidth=1.4)
    axes[1].set_ylabel("cost")
    axes[1].set_title("Optimization Objective")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    axes[2].plot(t, deepc_diag.u_ini_residual_norm, label="||Up g - u_ini||", linewidth=1.4)
    axes[2].plot(t, deepc_diag.y_ini_residual_norm, label="||Yp g - y_ini||", linewidth=1.4)
    axes[2].set_ylabel("residual")
    axes[2].set_title("Initial Condition Matching Residuals")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    axes[3].plot(t, deepc_diag.g_l2_norm, label="||g||_2", linewidth=1.4)
    axes[3].plot(t, deepc_diag.g_l1_norm, label="||g||_1", linewidth=1.4)
    axes[3].set_xlabel("time step")
    axes[3].set_ylabel("norm")
    axes[3].set_title("Coefficient Norms")
    axes[3].grid(True, alpha=0.3)
    axes[3].legend()

    fig1.tight_layout()
    fig1.savefig(out_dir / "deepc_internal_metrics.png", dpi=150)
    plt.close(fig1)

    g_top = deepc_diag.g_matrix[:, top_idx]
    fig2, ax2 = plt.subplots(figsize=(11, 4))
    im = ax2.imshow(g_top.T, aspect="auto", origin="lower", cmap="coolwarm")
    ax2.set_xlabel("time step")
    ax2.set_ylabel("selected g index rank")
    ax2.set_title("DeePC Top-|g| Coefficients Over Time")
    fig2.colorbar(im, ax=ax2, label="g value")
    fig2.tight_layout()
    fig2.savefig(out_dir / "deepc_g_heatmap.png", dpi=150)
    plt.close(fig2)

    fig3, ax3 = plt.subplots(figsize=(11, 4))
    ax3.plot(t, deepc_diag.y_now, label="y_now", linewidth=1.5)
    ax3.plot(t, deepc_diag.r_now, "k--", label="r_now", linewidth=1.6)
    ax3.plot(t, deepc_diag.y_plan_matrix[:, 0], label="first predicted y", linewidth=1.3)
    ax3.plot(t, deepc_diag.y_actual_plan_matrix[:, 0], label="first actual y (plan rollout)", linewidth=1.3)
    ax3.set_xlabel("time step")
    ax3.set_ylabel("output")
    ax3.set_title("DeePC One-Step Prediction vs Measured Output")
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    fig3.tight_layout()
    fig3.savefig(out_dir / "deepc_prediction_alignment.png", dpi=150)
    plt.close(fig3)

    fig5, ax5 = plt.subplots(figsize=(11, 4))
    ax5.plot(t, deepc_diag.pred_actual_norm, linewidth=1.5, label="||y_pred - y_actual_rollout||")
    ax5.set_xlabel("time step")
    ax5.set_ylabel("norm")
    ax5.set_title("DeePC Prediction vs Actual Rollout Mismatch")
    ax5.grid(True, alpha=0.3)
    ax5.legend()
    fig5.tight_layout()
    fig5.savefig(out_dir / "deepc_prediction_mismatch.png", dpi=150)
    plt.close(fig5)

    if dataset is not None and len(dataset) > 0:
        fig4, ax4 = plt.subplots(figsize=(11, 4))
        n_plot = min(20, len(dataset))
        idx = np.linspace(0, len(dataset) - 1, num=n_plot, dtype=int)
        for i in idx:
            y_seg = dataset[i].y[:-1]
            ax4.plot(np.arange(y_seg.shape[0]), y_seg, color="0.8", linewidth=0.8)

        deepc_y = results["DeePC"].y
        mpc_y = results["MPC"].y
        ax4.plot(np.arange(deepc_y.shape[0]), deepc_y, color="tab:blue", linewidth=2.0, label="DeePC closed-loop")
        ax4.plot(np.arange(mpc_y.shape[0]), mpc_y, color="tab:orange", linewidth=1.8, label="MPC closed-loop")
        ax4.set_xlabel("time step")
        ax4.set_ylabel("y")
        ax4.set_title("Closed-Loop Trajectories vs Data Library")
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        fig4.tight_layout()
        fig4.savefig(out_dir / "deepc_vs_dataset_trajectories.png", dpi=150)
        plt.close(fig4)


def plot_results(
    results: Dict[str, SimulationResult],
    r: np.ndarray,
    output_dir: str = "graphs",
    deepc_diagnostics: DeePCDiagnostics | None = None,
    dataset: list[Trajectory] | None = None,
    top_g_to_plot: int = 12,
) -> Path:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    t = np.arange(r.shape[0])

    fig1, ax1 = plt.subplots(figsize=(10, 4))
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    fig3, ax3 = plt.subplots(figsize=(10, 4))

    # Output tracking.
    ax1.plot(t, r, "k--", linewidth=2.0, label="reference")
    for name, res in results.items():
        ax1.plot(t, res.y, linewidth=1.8, label=f"{name} (RMSE={res.rmse:.3f})")
    ax1.set_ylabel("y")
    ax1.set_xlabel("time step")
    ax1.set_title("Output Tracking")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Control actions.
    for name, res in results.items():
        ax2.step(t, res.u, where="post", linewidth=1.5, label=name)
    ax2.set_ylabel("u")
    ax2.set_xlabel("time step")
    ax2.set_title("Control Inputs")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # Tracking errors.
    for name, res in results.items():
        ax3.plot(t, res.y - r, linewidth=1.5, label=name)
    ax3.set_ylabel("y - r")
    ax3.set_xlabel("time step")
    ax3.set_title("Tracking Error")
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    fig1.tight_layout()
    fig2.tight_layout()
    fig3.tight_layout()

    fig1.savefig(out_dir / "outputs_vs_reference.png", dpi=150)
    fig2.savefig(out_dir / "control_inputs.png", dpi=150)
    fig3.savefig(out_dir / "tracking_error.png", dpi=150)

    plt.close(fig1)
    plt.close(fig2)
    plt.close(fig3)

    if deepc_diagnostics is not None:
        g_abs_mean = np.mean(np.abs(deepc_diagnostics.g_matrix), axis=0)
        top_k = int(max(1, min(top_g_to_plot, g_abs_mean.shape[0])))
        top_idx = np.argsort(g_abs_mean)[-top_k:][::-1]
        _plot_deepc_diagnostics(
            out_dir=out_dir,
            results=results,
            deepc_diag=deepc_diagnostics,
            dataset=dataset,
            top_idx=top_idx,
        )

    return out_dir
