from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

from controllers.deepc import DeePCConfig, DeePCController
from controllers.mpc import MPCConfig, MPCController
from data import generate_dataset, make_reference, summarize_dataset_ranges
from evaluation_suite import generate_all_plots
from simulation import (run_all_simulations_with_diagnostics, simulate_deepc,
                        simulate_mpc)
from system import KinematicBicycleYaw, LiftedKinematicBicycleYaw


def export_debug_logs(results: dict, r: np.ndarray, v_seq: np.ndarray = None, out_dir: str = "debug_logs") -> None:
    """Exports logs of outputs for all models to CSV files for better assessment."""
    import csv
    os.makedirs(out_dir, exist_ok=True)
    
    for name, res in results.items():
        filename = os.path.join(out_dir, f"{name.replace(' ', '_')}_debug_log.csv")
        with open(filename, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Time_Step", "Velocity", "Reference", "State_x", "Output_y", "Input_u"])
            
            # y, u, and r are matched up to length t_sim. x has length t_sim+1.
            for t in range(len(res.y)):
                ref_val = r[t] if t < len(r) else r[-1]
                vel_val = v_seq[t] if v_seq is not None else 2.0
                writer.writerow([t, vel_val, ref_val, res.x[t], res.y[t], res.u[t]])
            
        print(f"Exported debug logs for {name} to {filename}")


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
    u_weight: float = 1e-4



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
            convex_g=False,
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
            convex_g=False,
            max_columns=600,
            debug=False,
        ),
    )

    # 4. Generate the path to track. Higher frequencies means tighter, more aggressive turns.
    # Try increasing this frequency (e.g. to 1.5) to see Vanilla DPC fail and Lifted DPC succeed!
    path_frequency = 0.75
    r = make_reference(cfg.t_sim, frequency=path_frequency)
    
    # 4.b Generate the velocity sequence to stress-test the controllers
    # Let velocity rapidly oscillate between 0.5 m/s and 3.5 m/s 
    # to guarantee MPC's constant horizon assumption (and DeePC's dataset assumption) completely fails!
    # This ensures velocity changes drastically _within_ the prediction horizon (t_f=10).
    t = np.arange(cfg.t_sim)
    v_test = 2.0 + 1.5 * np.sin(2.0 * np.pi * t / 20.0)

    # Run simulations
    res_mpc = simulate_mpc(system=system, controller=mpc, x0=cfg.x0, r=r, t_sim=cfg.t_sim, v_seq=v_test)
    res_deepc_vanilla, diag_vanilla = simulate_deepc(
        system=system, controller=deepc_vanilla, x0=cfg.x0, r=r, t_sim=cfg.t_sim, collect_diagnostics=True, v_seq=v_test
    )
    res_deepc_lifted, diag_lifted = simulate_deepc(
        system=system_lifted, controller=deepc_lifted, x0=cfg.x0, r=r, t_sim=cfg.t_sim, collect_diagnostics=True, v_seq=v_test
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

    # Generate debug documents for assessment
    export_debug_logs(results, r, v_test)

    out_dir = generate_all_plots(results, r, v_test)
    print(f"Saved plots to: {out_dir}")


if __name__ == "__main__":
    main()
