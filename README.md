# MiniMPC Comparison Framework

This project compares two controllers on a simple nonlinear system:

- Classical MPC (model-based, local linearization + QP)
- DeePC (data-enabled predictive control)

Note: for this nonlinear benchmark, DeePC uses a robust regularized formulation
with soft initial-condition matching and convex coefficient constraints to improve
closed-loop behavior.

## System

Discrete-time nonlinear system:

x_{t+1} = x_t + 0.5 * sin(x_t) + u_t

y_t = x_t

## Install

```bash
pip install -r requirements.txt
```

## Run

```bash
python main.py
```

The script prints RMSE for each controller and saves plots for:

- outputs vs reference
- control inputs
- tracking error

The plots are saved to the `graphs/` folder as PNG files:

- `outputs_vs_reference.png`
- `control_inputs.png`
- `tracking_error.png`

Additional DeePC internal visualizations are also saved:

- `deepc_internal_metrics.png`
- `deepc_g_heatmap.png`
- `deepc_prediction_alignment.png`
- `deepc_prediction_mismatch.png`
- `deepc_vs_dataset_trajectories.png`

## Main Parameters

Edit values in `SimulationConfig` in `main.py`:

- `t_sim`: simulation length
- `t_data`: trajectory length in dataset
- `n_data`: number of dataset trajectories
- `t_ini`: DeePC initial window length
- `t_f`: prediction horizon
- `u_min`, `u_max`: input bounds
