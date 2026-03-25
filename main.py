from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from controllers.deepc import DeePCConfig, DeePCController
from controllers.mpc import MPCConfig, MPCController
from data import generate_dataset, make_reference, summarize_dataset_ranges
from evaluation_suite import generate_all_plots
from simulation import (run_all_simulations_with_diagnostics, simulate_deepc,
                        simulate_mpc)
from system import KinematicBicycleYaw, LiftedKinematicBicycleYaw


@dataclass
class SimulationConfig:
    # Data generation
    n_data: int = 100
    t_data: int = 40
    data_seed: int = 7

    # Controller horizons
    t_ini: int = 5
    t_f: int = 10

    # Closed-loop simulation
    t_sim: int = 50
    x0: float = 0.0

    # Input bounds and cost weight
    u_min: float = -np.pi/4
    u_max: float = np.pi/4
    u_weight: float = 0.1



def main() -> None:
    cfg = SimulationConfig()

    # 1. Nominal system (Ground Truth)
    system = KinematicBicycleYaw(u_min=cfg.u_min, u_max=cfg.u_max)
    system_lifted = LiftedKinematicBicycleYaw(u_min=cfg.u_min, u_max=cfg.u_max)

    # 2. Vanilla DPC Dataset
    dataset_vanilla = generate_dataset(
        system=system,
        n_data=cfg.n_data,
        t_data=cfg.t_data,
        u_min=cfg.u_min,
        u_max=cfg.u_max,
        seed=cfg.data_seed,
        lift_input=False
    )

    # 3. Lifted DPC Dataset
    dataset_lifted = generate_dataset(
        system=system,
        n_data=cfg.n_data,
        t_data=cfg.t_data,
        u_min=cfg.u_min,
        u_max=cfg.u_max,
        seed=cfg.data_seed,
        lift_input=True
    )

    print("Vanilla dataset ranges:", summarize_dataset_ranges(dataset_vanilla))
    print("Lifted dataset ranges:", summarize_dataset_ranges(dataset_lifted))

    mpc = MPCController(
        system=system,
        config=MPCConfig(
            t_f=cfg.t_f,
            u_min=cfg.u_min,
            u_max=cfg.u_max,
            u_weight=cfg.u_weight,
        ),
    )

    deepc_vanilla = DeePCController(
        trajectories=dataset_vanilla,
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

    deepc_lifted = DeePCController(
        trajectories=dataset_lifted,
        config=DeePCConfig(
            t_ini=cfg.t_ini,
            t_f=cfg.t_f,
            u_min=np.tan(cfg.u_min),
            u_max=np.tan(cfg.u_max),
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

    # 4. Generate the path to track. Higher frequencies means tighter, more aggressive turns.
    # Try increasing this frequency (e.g. to 0.8) to see Vanilla DPC fail and Lifted DPC succeed!
    path_frequency = 0.5
    r = make_reference(cfg.t_sim, frequency=path_frequency)

    # Run simulations
    res_mpc = simulate_mpc(system=system, controller=mpc, x0=cfg.x0, r=r, t_sim=cfg.t_sim)
    res_deepc_vanilla, diag_vanilla = simulate_deepc(
        system=system, controller=deepc_vanilla, x0=cfg.x0, r=r, t_sim=cfg.t_sim, collect_diagnostics=True
    )
    res_deepc_lifted, diag_lifted = simulate_deepc(
        system=system_lifted, controller=deepc_lifted, x0=cfg.x0, r=r, t_sim=cfg.t_sim, collect_diagnostics=True
    )
    
    # NOTE: The lifted results u vector needs to be arctan'd if we want the actual steering angle metric
    res_deepc_lifted.u = np.arctan(res_deepc_lifted.u)

    results = {
        "MPC": res_mpc,
        "Vanilla DPC": res_deepc_vanilla,
        "Structure-Informed DPC": res_deepc_lifted,
    }

    print("RMSE summary")
    for name, res in results.items():
        print(f"- {name:25s}: {res.rmse:.6f}")

    out_dir = generate_all_plots(results)
    print(f"Saved plots to: {out_dir}")


if __name__ == "__main__":
    main()
