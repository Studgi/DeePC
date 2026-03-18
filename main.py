from __future__ import annotations

from dataclasses import dataclass

from controllers.deepc import DeePCConfig, DeePCController
from controllers.mpc import MPCConfig, MPCController
from data import generate_dataset, make_reference, summarize_dataset_ranges
from plotting import plot_results
from simulation import run_all_simulations_with_diagnostics
from system import NonlinearSystem


@dataclass
class SimulationConfig:
    # Data generation
    n_data: int = 200
    t_data: int = 100
    data_seed: int = 7

    # Controller horizons
    t_ini: int = 20
    t_f: int = 20

    # Closed-loop simulation
    t_sim: int = 200
    x0: float = 0.2

    # Input bounds and cost weight
    u_min: float = -2.0
    u_max: float = 2.0
    u_weight: float = 0.1


def main() -> None:
    cfg = SimulationConfig()

    system = NonlinearSystem(u_min=cfg.u_min, u_max=cfg.u_max)

    dataset = generate_dataset(
        system=system,
        n_data=cfg.n_data,
        t_data=cfg.t_data,
        u_min=cfg.u_min,
        u_max=cfg.u_max,
        seed=cfg.data_seed,
    )
    ranges = summarize_dataset_ranges(dataset)
    print(
        "Dataset ranges: "
        f"u in [{ranges['u_min']:.3f}, {ranges['u_max']:.3f}], "
        f"y in [{ranges['y_min']:.3f}, {ranges['y_max']:.3f}]"
    )

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

    r = make_reference(cfg.t_sim)

    results, deepc_diagnostics = run_all_simulations_with_diagnostics(
        system=system,
        mpc=mpc,
        deepc=deepc,
        x0=cfg.x0,
        r=r,
        t_sim=cfg.t_sim,
    )

    print("RMSE summary")
    for name, res in results.items():
        print(f"- {name:10s}: {res.rmse:.6f}")

    graphs_dir = plot_results(
        results=results,
        r=r,
        output_dir="graphs",
        deepc_diagnostics=deepc_diagnostics,
        dataset=dataset,
        top_g_to_plot=12,
    )
    print(f"Saved plots to: {graphs_dir.resolve()}")


if __name__ == "__main__":
    main()
