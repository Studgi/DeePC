import os
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Ensure we can import the project modules by adding the parent dir to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from controllers.deepc import DeePCConfig, DeePCController
from controllers.mpc import MPCConfig, MPCController
from data import generate_dataset, make_reference
from evaluation_suite import generate_all_plots
from main import SimulationConfig
from simulation import SimulationResult, simulate_deepc
from system import KinematicBicycleYaw, LiftedKinematicBicycleYaw

st.set_page_config(page_title="Control Visualizer", layout="wide")

# ---- STYLING ----
st.markdown("""
<style>
    .stApp {
        background-color: #040b14;
    }
    h1, h2, h3, h4 {
        color: #e0e7ff;
    }
    .metric-row {
        display: flex;
        justify-content: space-between;
        margin-bottom: 20px;
    }
    .metric-card {
        background-color: #0b1b36;
        padding: 20px;
        border-radius: 10px;
        flex: 1;
        margin: 0 10px;
        text-align: center;
        border: 1px solid #1a3c6b;
    }
    .metric-value {
        font-size: 24px;
        font-weight: bold;
        color: #2a75d3;
    }
    .metric-label {
        font-size: 14px;
        color: #8da4c9;
    }
</style>
""", unsafe_allow_html=True)


def generate_custom_reference(t_sim: int, ref_type: str, dt: float = 0.1) -> np.ndarray:
    t = np.arange(t_sim, dtype=float) * dt
    if ref_type == "Sine Wave (Gentle)":
        return np.sin(0.5 * t)
    elif ref_type == "Sine Wave (Aggressive)":
        return 1.5 * np.sin(1.0 * t)
    elif ref_type == "Square Wave (Jumps)":
        period = 4.0  # seconds
        return 1.0 * np.sign(np.sin(2 * np.pi * t / period))
    elif ref_type == "Step Response":
        r = np.zeros(t_sim)
        r[int(t_sim*0.2):] = 1.0
        r[int(t_sim*0.6):] = 0.0
        return r
    else: # "Straight Line"
        return np.zeros(t_sim)

def generate_velocity_sequence(t_sim: int, vel_type: str) -> np.ndarray:
    if vel_type == "Constant (2.0 m/s - Baseline)":
        return np.full(t_sim + 60, 2.0) # pad for horizon
    elif vel_type == "Constant (1.0 m/s)":
        return np.full(t_sim + 60, 1.0)
    elif vel_type == "Constant (3.0 m/s)":
        return np.full(t_sim + 60, 3.0)
    elif vel_type == "Constant (4.0 m/s)":
        return np.full(t_sim + 60, 4.0)
    elif vel_type == "Accelerating (1.5 -> 4.0 m/s)":
        return np.linspace(1.5, 4.0, t_sim + 60)
    elif vel_type == "Braking (4.0 -> 1.0 m/s)":
        return np.linspace(4.0, 1.0, t_sim + 60)
    elif vel_type == "Sine Wave (0.5 -> 3.5 m/s)":
        t = np.arange(t_sim + 60)
        return 2.0 + 1.5 * np.sin(2.0 * np.pi * t / 20.0)
    else:
        return np.full(t_sim + 60, 2.0)

@st.cache_resource(show_spinner="Generating datasets...")
def get_datasets(cfg: SimulationConfig):
    system = KinematicBicycleYaw(u_min=cfg.u_min, u_max=cfg.u_max)
    data_vanilla = generate_dataset(
        system=system, n_data=cfg.n_data, t_data=cfg.t_data, 
        u_min=cfg.u_min, u_max=cfg.u_max, seed=cfg.data_seed, lift_input=False
    )
    data_lifted = generate_dataset(
        system=system, n_data=cfg.n_data, t_data=cfg.t_data, 
        u_min=cfg.u_min, u_max=cfg.u_max, seed=cfg.data_seed, lift_input=True
    )
    return data_vanilla, data_lifted

@st.cache_resource(show_spinner="Running simulations...")
def run_cached_simulations(ref_type: str, vel_type: str):
    cfg = SimulationConfig()
    data_vanilla, data_lifted = get_datasets(cfg)
    r = generate_custom_reference(cfg.t_sim, ref_type)
    v_seq = generate_velocity_sequence(cfg.t_sim, vel_type)
    
    # Run MPC explicitly to capture open-loop plans step-by-step
    system_mpc = KinematicBicycleYaw(u_min=cfg.u_min, u_max=cfg.u_max)
    mpc_ctrl = MPCController(
        system=system_mpc, 
        config=MPCConfig(t_f=cfg.t_f, u_min=cfg.u_min, u_max=cfg.u_max, u_weight=cfg.u_weight)
    )
    
    mpc_x = np.zeros(cfg.t_sim + 1)
    mpc_y = np.zeros(cfg.t_sim)
    mpc_u = np.zeros(cfg.t_sim)
    mpc_plan_y = np.zeros((cfg.t_sim, cfg.t_f))
    mpc_plan_u = np.zeros((cfg.t_sim, cfg.t_f))
    
    mpc_x[0] = cfg.x0
    for t in range(cfg.t_sim):
        current_v = v_seq[t]
        mpc_y[t] = system_mpc.output(mpc_x[t])
        r_f = r[t + 1 : t + 1 + cfg.t_f]
        if r_f.shape[0] < cfg.t_f:
            pad_val = r_f[-1] if r_f.size > 0 else 0.0
            r_f = np.pad(r_f, (0, cfg.t_f - r_f.shape[0]), constant_values=pad_val)
            
        u_t = mpc_ctrl.compute_control(x_now=mpc_x[t], r_future=r_f, current_v=current_v)
        
        # Capture the internal plan
        mpc_plan_u[t, :] = np.clip(np.roll(mpc_ctrl._last_u_plan, 1), cfg.u_min, cfg.u_max)
        mpc_plan_u[t, 0] = u_t
        
        y_roll = np.zeros(cfg.t_f)
        x_roll = mpc_x[t]
        for k in range(cfg.t_f):
            v_roll = v_seq[t+k] if (t+k) < len(v_seq) else v_seq[-1]
            x_roll = system_mpc.step(x_roll, mpc_plan_u[t, k], v=v_roll)
            y_roll[k] = system_mpc.output(x_roll)
        mpc_plan_y[t, :] = y_roll
            
        mpc_u[t] = u_t
        mpc_x[t + 1] = system_mpc.step(mpc_x[t], u_t, v=current_v)


    # Run DeePC Vanilla
    deepc_v = DeePCController(
        trajectories=data_vanilla,
        config=DeePCConfig(t_ini=cfg.t_ini, t_f=cfg.t_f, u_min=cfg.u_min, u_max=cfg.u_max, 
                           lambda_u=1e-4, lambda_g=1e-4, lambda_ini_u=1e2, lambda_ini_y=1e2, 
                           use_soft_ini=True, convex_g=False, max_columns=600, debug=False)
    )
    system_v = KinematicBicycleYaw(u_min=cfg.u_min, u_max=cfg.u_max)
    res_v, diag_v = simulate_deepc(system_v, deepc_v, cfg.x0, r, cfg.t_sim, collect_diagnostics=True, v_seq=v_seq)


    # Run DeePC Lifted
    deepc_l = DeePCController(
        trajectories=data_lifted,
        config=DeePCConfig(t_ini=cfg.t_ini, t_f=cfg.t_f, u_min=np.tan(cfg.u_min), u_max=np.tan(cfg.u_max), 
                           lambda_u=1e-4, lambda_g=1e-4, lambda_ini_u=1e2, lambda_ini_y=1e2, 
                           use_soft_ini=True, convex_g=False, max_columns=600, debug=False)
    )
    system_l = LiftedKinematicBicycleYaw(u_min=cfg.u_min, u_max=cfg.u_max)
    res_l, diag_l = simulate_deepc(system_l, deepc_l, cfg.x0, r, cfg.t_sim, collect_diagnostics=True, lifted_input=True, v_seq=v_seq)
    
    # Fix the lifted u to actual limits
    res_l.u = np.arctan(res_l.u)
    # The planned u in diag also needs arctan for proper visual scaling
    diag_l.u_plan_matrix = np.arctan(diag_l.u_plan_matrix)

    mpc_rmse = float(np.sqrt(np.mean((mpc_y - r[:cfg.t_sim])**2)))
    res_mpc = SimulationResult(x=mpc_x, y=mpc_y, u=mpc_u, rmse=mpc_rmse)

    return cfg, [(res_mpc, mpc_plan_y, mpc_plan_u), (res_v, diag_v), (res_l, diag_l)], r, (deepc_v.yf, deepc_l.yf), v_seq

st.title("🎛️ Control Systems Visualizer")
st.markdown("Interactive step-by-step visualizer for investigating MPC, Vanilla DeePC, and Structure-Informed DeePC.")

st.sidebar.header("Settings")
mode = st.sidebar.radio("Select Controller", ["Standard MPC", "Vanilla DPC", "Structure Lifted DPC"])
ref_type = st.sidebar.selectbox(
    "Select Reference Trajectory", 
    ["Sine Wave (Gentle)", "Sine Wave (Aggressive)", "Square Wave (Jumps)", "Step Response", "Straight Line"],
    index=0
)
vel_type = st.sidebar.selectbox(
    "Select Velocity Profile",
    ["Constant (2.0 m/s - Baseline)", "Constant (1.0 m/s)", "Constant (3.0 m/s)", "Constant (4.0 m/s)", "Accelerating (1.5 -> 4.0 m/s)", "Braking (4.0 -> 1.0 m/s)", "Sine Wave (0.5 -> 3.5 m/s)"],
    index=0
)

cfg, cache_results, r, hankel_futures, v_seq = run_cached_simulations(ref_type, vel_type)
(res_mpc, mpc_plan_y, mpc_plan_u), (res_v, diag_v), (res_l, diag_l) = cache_results

mpc_y = res_mpc.y
mpc_u = res_mpc.u
(yf_v, yf_l) = hankel_futures

tab_interactive, tab_evaluation = st.tabs(["Interactive Simulation", "Academic Evaluation Suite"])

with tab_evaluation:
    st.header("📊 Evaluation Suite Integration")
    st.markdown("These are the exact formal measurement figures exported directly from `evaluation_suite.py`!")
    
    # We construct the results dictionary format that `generate_all_plots` expects
    results_dict = {
        "MPC": res_mpc,
        "Vanilla DPC": res_v,
        "Structure-Informed DPC": res_l
    }
    
    with st.spinner("Generating thesis figures..."):
        out_dir = generate_all_plots(results_dict, r, v_seq)
        
        st.subheader("1. System Performance & Tracking Error")
        st.image(os.path.join(out_dir, "1_Performance_and_Error.png"), use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("2. Open-Loop Prediction")
            st.image(os.path.join(out_dir, "2_Open_Loop_Prediction.png"), use_container_width=True)
            
            st.subheader("4. Region of Attraction Heatmaps")
            st.image(os.path.join(out_dir, "4_RoA_Heatmaps.png"), use_container_width=True)
            
            st.subheader("6. The Pareto Front")
            st.image(os.path.join(out_dir, "6_Pareto_Front.png"), use_container_width=True)
            
        with col2:
            st.subheader("3. Computational Complexity")
            st.image(os.path.join(out_dir, "3_Computational_Complexity.png"), use_container_width=True)
            
            st.subheader("5. Data Efficiency & Noise Robustness")
            st.image(os.path.join(out_dir, "5_Data_and_Noise_Robustness.png"), use_container_width=True)

with tab_interactive:
    t_step = st.sidebar.slider("Current Time Step", 0, cfg.t_sim - 1, 0, 1)

    # Extract Data based on Mode
    if mode == "Standard MPC":
        y_hist, u_hist = mpc_y, mpc_u
        y_plan, u_plan = mpc_plan_y[t_step], mpc_plan_u[t_step]
        status = "Solved (OSQP)"
        g_weights = None
        y_hankel_future = None
    elif mode == "Vanilla DPC":
        y_hist, u_hist = res_v.y, res_v.u
        y_plan, u_plan = diag_v.y_plan_matrix[t_step], diag_v.u_plan_matrix[t_step]
        status = diag_v.status[t_step]
        g_weights = diag_v.g_matrix[t_step]
        y_hankel_future = yf_v
    else:
        y_hist, u_hist = res_l.y, res_l.u
        y_plan, u_plan = diag_l.y_plan_matrix[t_step], diag_l.u_plan_matrix[t_step]
        status = diag_l.status[t_step]
        g_weights = diag_l.g_matrix[t_step]
        y_hankel_future = yf_l

    y_now = y_hist[t_step]
    u_now = u_hist[t_step]
    r_now = r[t_step]
    v_now = v_seq[t_step]
    pred_length = cfg.t_f

    st.markdown(f"""
    <div class="metric-row">
    <div class="metric-card">
        <div class="metric-label">Velocity (v)</div>
        <div class="metric-value">{v_now:.2f} m/s</div>
    </div>
    <div class="metric-card">
        <div class="metric-label">Current State (Yaw)</div>
        <div class="metric-value">{y_now:.4f} rad</div>
    </div>
    <div class="metric-card">
        <div class="metric-label">Reference Target</div>
        <div class="metric-value">{r_now:.4f} rad</div>
    </div>
    <div class="metric-card">
        <div class="metric-label">Solver Status</div>
        <div class="metric-value">{status}</div>
    </div>
    </div>
    """, unsafe_allow_html=True)


    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Trajectory Tracking (Closed-Loop + Plan)")
        fig = go.Figure()
        
        # Plot past trajectory
        past_t = np.arange(t_step + 1)
        fig.add_trace(go.Scatter(x=past_t, y=y_hist[:t_step+1], mode='lines+markers', name='Actual History', marker=dict(color="#1f77b4")))
        
        # Plot full reference
        fig.add_trace(go.Scatter(x=np.arange(cfg.t_sim), y=r, mode='lines', name='Reference', line=dict(dash='dash', color="#aab4c2")))
        
        # Plot Open-Loop Plan
        future_t = np.arange(t_step + 1, t_step + 1 + pred_length)
        if len(future_t) > 0:
            # If DPC, plot a faint cloud of "possible" future trajectories from the Hankel matrix
            if "DPC" in mode and y_hankel_future is not None:
                top_indices = np.argsort(np.abs(g_weights))[-5:][::-1]

                num_cols_to_plot = min(150, y_hankel_future.shape[1])  # sample max 150 to keep UI fast
                sample_idx = set(np.linspace(0, y_hankel_future.shape[1]-1, num_cols_to_plot, dtype=int))
                sample_idx.update(top_indices) # ensure top indices are plotted 

                for i in sample_idx:
                    raw_traj = y_hankel_future[:, i]
                    # Shift the trajectory vertically just for visualization purposes so it spans out from current position
                    shift = y_now - raw_traj[0]
                    vis_traj = raw_traj + shift
                    vis_line = np.insert(vis_traj, 0, y_now)
                    plan_t = np.insert(future_t, 0, t_step)

                    if i in top_indices:
                        fig.add_trace(go.Scatter(
                            x=plan_t, y=vis_line, mode='lines',
                            line=dict(color="rgba(0, 255, 204, 0.6)", width=2),
                            showlegend=False, hoverinfo='skip'
                        ))
                    else:
                        fig.add_trace(go.Scatter(
                            x=plan_t, y=vis_line, mode='lines',
                            line=dict(color="rgba(255, 255, 255, 0.05)", width=1), # Very faint white/gray
                            showlegend=False, hoverinfo='skip'
                        ))

                # Add a dummy trace for legend
                fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines', name='Static Memory Options', line=dict(color="rgba(255, 255, 255, 0.4)", width=1)))
                fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines', name='Dominant Memories Picked', line=dict(color="rgba(0, 255, 204, 0.6)", width=2)))

        plan_line = np.insert(y_plan, 0, y_now)
        plan_t = np.insert(future_t, 0, t_step)
        fig.add_trace(go.Scatter(x=plan_t, y=plan_line, mode='lines', name='Forecast (Chosen weighted Plan)', line=dict(color="#ff9900", width=3.5)))

        fig.update_layout(template="plotly_dark", plot_bgcolor="#040b14", paper_bgcolor="#040b14", margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Control Input (Closed-Loop + Plan)")
        fig2 = go.Figure()
        
        fig2.add_trace(go.Scatter(x=past_t, y=u_hist[:t_step+1], mode='lines+markers', name='Applied Input', marker=dict(color="#2ca02c")))
        
        if len(future_t) > 0:
            fig2.add_trace(go.Scatter(x=future_t, y=u_plan, mode='lines', name='Planned Input', line=dict(color="#ff9900", width=3)))
            
        fig2.update_layout(template="plotly_dark", plot_bgcolor="#040b14", paper_bgcolor="#040b14", margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig2, use_container_width=True)

    # Educational / Under-the-hood Details
    if "DPC" in mode and g_weights is not None:
        st.markdown("---")
        st.header(" How DeePC Works (Beginner's Guide)")     

        st.info('''
        ### Phase 1: Data Collection & Building the Hankel Matrix
        Before this car even started driving, we ran an **Offline Data Collection** phase. We applied random steering angles to the car and recorded its movements.
        We took all those random movements and chopped them up into short overlapping snippets (trajectories). We stacked these snippets side-by-side to create a massive data table called a **Hankel Matrix**.
        Each column in this matrix represents a possible way the car can move based on real historical evidence.
        ''')

        st.success(f'''
        ### Phase 2: Online Control (What's happening at Step {t_step})
        Now the car is driving live. DeePC doesn't use a physics equation to predict the future. Instead, it looks at the Hankel matrix and calculates a mathematical **weight (g)** for each column.
        1. **Matching the Past:** It finds a combination of historical columns that, when added together, exactly match what the car did over the last few seconds.
        2. **Predicting the Future:** Because those columns contain both "past" and "future" data snippets, whatever combination of weights perfectly matched the past will *also* project a valid, physically accurate path into the future!
        ''')

        st.subheader(f"The **g** Vector (Trajectory Weights) at Step {t_step}")

        # Identify the top 5 most important columns (highest absolute weight)
        top_indices = np.argsort(np.abs(g_weights))[-5:][::-1] 

        st.write(f"To follow the reference line right now, the solver decided to blend **{len(g_weights)}** different historical trajectories. Most trajectories are ignored (weight $\\approx 0$), but a few exact trajectories are heavily favored to build the current plan.")
        st.markdown(f"**Notice the active columns:** At this exact time step, the solver is focusing most heavily on memory columns **{', '.join(map(str, top_indices))}** (highlighted in cyan below) because those specific trajectories in the dataset best match the car's current physical situation!")   

        colors = ["#ab62c0"] * len(g_weights)
        for idx in top_indices:
            colors[idx] = "#00ffcc"

        fig_g = go.Figure(go.Bar(x=np.arange(len(g_weights)), y=g_weights, marker_color=colors))

        # Add markers to point them out visually
        fig_g.add_trace(go.Scatter(
            x=top_indices,
            y=[g_weights[i] for i in top_indices],
            mode='markers',
            marker=dict(symbol='star', size=10, color='white'),    
            name="Dominant Trajectories"
        ))

        fig_g.update_layout(
            template="plotly_dark",
            plot_bgcolor="#040b14", paper_bgcolor="#040b14",       
            xaxis_title="Hankel Historical Column Index",
            yaxis_title="Calculated Weight (g)",
            margin=dict(l=20, r=20, t=30, b=20),
            showlegend=False
        )
        st.plotly_chart(fig_g, use_container_width=True)       

        st.info('''
        ** Key Interactive Observations to look for:**       
        1. **Cruising Straight:** When the reference line is flat, you will notice the cyan trajectories tightly group in the *vertical middle* of the cloud. The solver is specifically selecting historical moments where the steering wheel was near zero!
        2. **Taking a Sharp Turn:** Slide to a time step where the reference line jumps up or down. Watch how the cyan lines instantly snap to the extreme top or bottom edges of the cloud. It is searching its memory for the hardest turns it has ever experienced to pull off the maneuver.
        3. **Trajectory Blending:** Notice how the final orange plan almost never perfectly matches *one* single cyan line. The solver is mathematically blending them, putting positive and negative weights to cancel out noise and synthesize a distinctly new, optimal path!
        ''')

        # Show some matrix math behind the scenes
        with st.expander("Show the Math Context", expanded=False):
            st.latex(r''' egin{pmatrix} U_p \ Y_p \end{pmatrix} g = egin{pmatrix} u_{ini} \ y_{ini} \end{pmatrix} \quad (	ext{Match the Past}) ''')
            st.latex(r''' egin{pmatrix} U_f \ Y_f \end{pmatrix} g = egin{pmatrix} u_{future} \ y_{future} \end{pmatrix} \quad (	ext{Predict the Future}) ''')
            st.write(f"In this specific step {t_step}, the solver generated an **Objective Cost Goal** of: `{diag_v.objective[t_step] if 'Vanilla' in mode else diag_l.objective[t_step]:.4f}`.")

            # Give a visual colored indicator of whether the solver is relaxed or stressed
            cost_val = diag_v.objective[t_step] if 'Vanilla' in mode else diag_l.objective[t_step]
            if cost_val < 0.1:
                st.success(" Relaxed State: The car is happily on the line and doing minimal work.")
            elif cost_val < 1.0:
                st.warning(" Active State: A curve/jump occurred. The solver is expending effort to realign.")
            else:
                st.error(" Stressed State: The cost is huge! The solver is making dramatic steering corrections to fix a massive error.")

            u_ini_res = diag_l.u_ini_residual_norm[t_step] if "Lifted" in mode else diag_v.u_ini_residual_norm[t_step]
            y_ini_res = diag_l.y_ini_residual_norm[t_step] if "Lifted" in mode else diag_v.y_ini_residual_norm[t_step]
            st.write(f"- It failed to perfectly match past input history by an error norm of: `{u_ini_res:.4f}`")
            st.write(f"- It failed to perfectly match past output history by an error norm of: `{y_ini_res:.4f}`")

            st.write("### Visualizing the Hankel Data Matrix")     
            st.write("To understand what DeePC is 'thinking', look at the raw data it uses instead of equations!")

            # Generate a conceptual visualization of the Hankel overlapping structure
            hankel_demo_data = np.random.randn(20) * 0.1
            # Adding a wave to simulate steering input
            hankel_demo_data += np.sin(np.linspace(0, 4*np.pi, 20))

            T_ini = 3
            T_f = 6
            T_total = T_ini + T_f
            demo_cols = 10

            H_demo = np.zeros((T_total, demo_cols))
            for i in range(demo_cols):
                H_demo[:, i] = hankel_demo_data[i:i+T_total]

            fig_hankel = go.Figure(data=go.Heatmap(
                z=H_demo,
                colorscale='Viridis',
                text=np.round(H_demo, 2),
                texttemplate="%{text}",
                colorbar=dict(title='Value')
            ))

            # Draw a line separating past from future
            fig_hankel.add_hline(y=T_ini - 0.5, line_dash="dash", line_color="red", line_width=4)
            # Highlight the weights applied below
            fig_hankel.update_layout(
                title="Sample Hankel Matrix (Overlapping Historical Snippets)",
                xaxis_title="Historical Snippet Index (Column)",       
                yaxis_title="Time Steps (Rows: Top are 'Past', Bottom are 'Future')",
                template="plotly_dark", plot_bgcolor="#040b14", paper_bgcolor="#040b14"
            )
            st.plotly_chart(fig_hankel, use_container_width=True)  
            st.write("Notice how the values shift diagonally! Each column is a short trajectory. The red dashed line separates the `past` memory (U_p, Y_p) from the `predicted future` (U_f, Y_f). DeePC mathematically finds combinations of these columns that match what the car just did, and automatically reads the bottom half to know what it will do next!")


        st.markdown("---")
        st.subheader("Model Expectation vs Reality")
        st.write("When DeePC blends historical data, it generates an open-loop plan assuming the vehicle will perfectly follow that orange line. Below, we can see the difference between what the controller *thought* would happen, versus what *actually* happens when those calculated steering inputs are shoved into the real physical simulation.")

        y_actual_plan = diag_l.y_actual_plan_matrix[t_step] if "Lifted" in mode else diag_v.y_actual_plan_matrix[t_step]
        pred_error = diag_l.pred_actual_norm[t_step] if "Lifted" in mode else diag_v.pred_actual_norm[t_step]

        fig_val = go.Figure()
        # What the model expected to happen
        fig_val.add_trace(go.Scatter(x=np.arange(pred_length), y=y_plan, mode='lines', name='Expected Open-Loop Plan', line=dict(color="#ff9900", dash="dash", width=3)))
        # What actual happens when we take the model's planned U sequence and shove it into the true simulation physics
        fig_val.add_trace(go.Scatter(x=np.arange(pred_length), y=y_actual_plan, mode='lines', name='Actual Simulated Physics', line=dict(color="#00ffcc", width=3)))
        fig_val.update_layout(title=f"Open-Loop Prediction Error Norm: {pred_error:.5f}", template="plotly_dark", plot_bgcolor="#040b14", paper_bgcolor="#040b14")
        st.plotly_chart(fig_val, use_container_width=True)     

    elif mode == "Standard MPC":
        st.markdown("---")
        st.header(" How Model Predictive Control Works (Beginner's Guide)")

        st.info('''
        ### Step 1: The N-Step Future Plan
        Unlike DeePC, the Standard MPC completely ignores historical data. It relies purely on the exact mathematical physics equations (`x_{k+1} = x_k + (v/L) * tan(u)`).
        At the current timestep, the MPC looks at the road ahead (the reference line) and calculates an entire sequence of future steering commandscalled an **N-Step Plan**that will keep the car perfectly on track with minimal effort.   
        ''')

        st.success('''
        ### Step 2: The Receding Horizon (Shifting the Plan)
        Even though the MPC just spent heavy computational power generating a massive 10-step plan into the future, **it only applies the very first step** to the actual car.
        Why? Because the real world has disturbances and nonlinearities. Before the next time step begins, the car will move slightly differently than expected.
        So, the MPC discards the rest of its plan, **shifts its viewpoint ahead by dt seconds**, measures the car's actual new position, and generates a brand new N-Step plan all over again! This "look ahead, step once, shift, repeat" process is called **Receding Horizon Control**.
        ''')

        st.write(f"#### Visualizing the Horizon Shift (Step {max(0, t_step - 1)} $\rightarrow$ Step {t_step})")
        fig_rh = go.Figure()
        if t_step > 0:
            # Previous plan (what the MPC thought it was going to do at t-1)   
            prev_plan_y = mpc_plan_y[t_step - 1]
            prev_t = np.arange(t_step, t_step + pred_length)
            fig_rh.add_trace(go.Scatter(x=prev_t, y=prev_plan_y, mode='lines+markers', name=f'Old Plan (from Step {t_step - 1})', line=dict(color="rgba(255, 153, 0, 0.4)", dash="dot", width=3)))

            # Current plan (at t)
            curr_plan_y = mpc_plan_y[t_step]
            curr_t = np.arange(t_step + 1, t_step + 1 + pred_length)
            fig_rh.add_trace(go.Scatter(x=curr_t, y=curr_plan_y, mode='lines+markers', name=f'New Plan (at Step {t_step})', line=dict(color="#00ffcc", width=3)))

            fig_rh.update_layout(title="Comparing Consecutive N-Step Plans", template="plotly_dark", plot_bgcolor="#040b14", paper_bgcolor="#040b14")
            st.plotly_chart(fig_rh, use_container_width=True)
        else:
            st.info("Move the slider to step 1 or beyond to see the old plan shift into the new plan!")


        st.warning('''
        ### Step 3: Successive Linearization (Drawing Tangent Lines for CVXPY) 
        Here is the catch: Mathematical solvers like CVXPY are incredibly fast at solving *linear* problems (Quadratic Programs), but our car's steering physics rely on `tan(u)`, which is curved (nonlinear). If we feed a curve to CVXPY, it crashes or takes forever.

        So how do we trick the solver?
        Whenever the MPC calculates its new plan, it takes the old plan from the previous step and acts as a mathematician: it literally draws straight **tangent lines** at the current points on the curve (this is called taking the Jacobian or Taylor Expansion).
        By feeding CVXPY these straight tangent lines instead of the real curve, the solver can mathematically guarantee an optimal steering response in milliseconds. As the car moves along the curve every `dt` seconds, the MPC constantly redraws these tangent lines to stay accurate!
        ''')

        # Tangent line visualization
        u_prev_val = u_hist[t_step - 1] if t_step > 0 else 0.0
        st.write(f"#### Visualizing the Tangent Line (Linearizing at $u$ = {u_prev_val:.4f} rad for the OSQP Solver)")

        u_range = np.linspace(-np.pi/3, np.pi/3, 100)
        v_L = 10.0 / 2.5 # Assuming average v and L for purely visual representation
        f_u = v_L * np.tan(u_range)

        # Tangent: f(u)  f(up) + f'(up)*(u - up)  where f'(up) = v/L * sec^2(up)
        sec_sq = 1.0 / (np.cos(u_prev_val)**2)
        tangent_u = v_L * np.tan(u_prev_val) + (v_L * sec_sq) * (u_range - u_prev_val)

        fig_tan = go.Figure()
        fig_tan.add_trace(go.Scatter(x=u_range, y=f_u, mode='lines', name='Real Physics: Non-Linear Curve x ~ tan(u)', line=dict(color="#ff9900", width=4)))   
        fig_tan.add_trace(go.Scatter(x=u_range, y=tangent_u, mode='lines', name='Tangent Line given to CVXPY as proxy', line=dict(color="#00ffcc", dash="dash", width=3)))
        fig_tan.add_trace(go.Scatter(x=[u_prev_val], y=[v_L * np.tan(u_prev_val)], mode='markers', name='Current Point (Taylor Expansion)', marker=dict(color="white", size=12)))

        fig_tan.update_layout(template="plotly_dark", title=f"Trick #1: Successive Linearization (Step {t_step})", xaxis_title="Steering Angle u (rad)", yaxis_title="State Space Movement Vector", plot_bgcolor="#040b14", paper_bgcolor="#040b14", yaxis=dict(range=[-10, 10]))
        st.plotly_chart(fig_tan, use_container_width=True)

        with st.expander("Show the Math Context", expanded=False):
            st.latex(r''' \min_{u, x} \sum_{k=0}^{T_f} (x_k - r_k)^2 + R \cdot u_k^2  ''')

            # Calculate MPC cost manually for the UI since it wasn't logged like DeePC.
            # Cost = sum of (y_plan - r_plan)^2 + R * u_plan^2
            r_plan_mpc = r[t_step:t_step+pred_length]
            if len(r_plan_mpc) < pred_length:
                r_plan_mpc = np.pad(r_plan_mpc, (0, pred_length - len(r_plan_mpc)), constant_values=r_plan_mpc[-1] if len(r_plan_mpc)>0 else 0)

            y_err_sq = np.sum((y_plan - r_plan_mpc)**2)
            u_err_sq = cfg.u_weight * np.sum(u_plan**2)
            mpc_cost = y_err_sq + u_err_sq

            st.write(f"In this specific step {t_step}, the solver generated an **Objective Cost Goal** of: `{mpc_cost:.4f}`.")

            if mpc_cost < 0.1:
                st.success(" Relaxed State: The car is happily on the line and doing minimal work.")
            elif mpc_cost < 1.0:
                st.warning(" Active State: A curve/jump occurred. The solver is expending work to realign.")
            else:
                st.error(" Stressed State: The cost is huge! The solver is making dramatic steering corrections to fix a massive error.")

            st.write("**The Nonlinear Physics Equation:**")
            st.latex(r''' x_{k+1} = x_k + rac{v}{L} \cdot 	an(u_k) ''')     
            st.write("**The Linearized 'Tangent Line' Trick given to CVXPY:**")
            st.latex(r''' x_{k+1} pprox x_k + \left( rac{v}{L} \cdot \sec^2(u_{prev}) 
ight) u_k + c_k ''')
            st.write(f"At Step {t_step}, the OSQP solver successfully optimized this Quadratic Program based on those drawn tangent lines.")

    @st.cache_resource(show_spinner="Running Velocity Sweep Experiment...")
    def run_velocity_sweep():
        # Sweep over static velocities
        velocities = [1.0, 2.0, 3.0, 4.0]
        results = {"Velocity (m/s)": velocities, "MPC RMSE": [], "Vanilla DPC RMSE": [], "Lifted DPC RMSE": []}

        sweep_cfg = SimulationConfig()
        r_sweep = generate_custom_reference(sweep_cfg.t_sim, "Sine Wave (Aggressive)") # Aggressive is perfect for exposing bounds
        data_v, data_l = get_datasets(sweep_cfg)

        for v in velocities:
            v_seq_sweep = np.full(sweep_cfg.t_sim + 60, v)

            # MPC
            sys_m = KinematicBicycleYaw(u_min=sweep_cfg.u_min, u_max=sweep_cfg.u_max)    
            ctrl_m = MPCController(system=sys_m, config=MPCConfig(t_f=sweep_cfg.t_f, u_min=sweep_cfg.u_min, u_max=sweep_cfg.u_max))
            x_m = np.zeros(sweep_cfg.t_sim + 1)
            y_m = np.zeros(sweep_cfg.t_sim)
            x_m[0] = sweep_cfg.x0
            for t_i in range(sweep_cfg.t_sim):
                y_m[t_i] = sys_m.output(x_m[t_i])
                rf = r_sweep[t_i + 1 : t_i + 1 + sweep_cfg.t_f]
                rf = np.pad(rf, (0, max(0, sweep_cfg.t_f - len(rf))), constant_values=r_sweep[-1]) if len(rf) < sweep_cfg.t_f else rf[:sweep_cfg.t_f]
                u_m = ctrl_m.compute_control(x_now=x_m[t_i], r_future=rf, current_v=v)
                x_m[t_i+1] = sys_m.step(x_m[t_i], u_m, v=v)
            rms = float(np.sqrt(np.mean((y_m - r_sweep)**2)))
            if rms > 1e3: rms = float('inf')
            results["MPC RMSE"].append(rms)

            # Vanilla DPC
            sys_van = KinematicBicycleYaw(u_min=sweep_cfg.u_min, u_max=sweep_cfg.u_max)      
            ctrl_van = DeePCController(trajectories=data_v, config=DeePCConfig(t_ini=sweep_cfg.t_ini, t_f=sweep_cfg.t_f, u_min=sweep_cfg.u_min, u_max=sweep_cfg.u_max))
            res_v_sweep, _ = simulate_deepc(sys_van, ctrl_van, sweep_cfg.x0, r_sweep, sweep_cfg.t_sim, v_seq=v_seq_sweep)
            if res_v_sweep.rmse > 1e3:
                results["Vanilla DPC RMSE"].append(float('inf'))
            else:
                results["Vanilla DPC RMSE"].append(res_v_sweep.rmse)

            # Lifted DPC
            sys_lift = LiftedKinematicBicycleYaw(u_min=sweep_cfg.u_min, u_max=sweep_cfg.u_max)
            ctrl_lift = DeePCController(trajectories=data_l, config=DeePCConfig(t_ini=sweep_cfg.t_ini, t_f=sweep_cfg.t_f, u_min=np.tan(sweep_cfg.u_min), u_max=np.tan(sweep_cfg.u_max)))
            res_l_sweep, _ = simulate_deepc(sys_lift, ctrl_lift, sweep_cfg.x0, r_sweep, sweep_cfg.t_sim, lifted_input=True, v_seq=v_seq_sweep)
            if res_l_sweep.rmse > 1e3:
                results["Lifted DPC RMSE"].append(float('inf'))
            else:
                results["Lifted DPC RMSE"].append(res_l_sweep.rmse)

        return pd.DataFrame(results)

    st.markdown("---")
    st.header(" Velocity Sweep Experiment (Stress Test)")
    st.write("We sweep the constant physical velocity of the car from 1.0 m/s up to 4.0 m/s. The DeePC Data (Hankel Matrices) were collected exclusively at $v = 2.0$ m/s.")

    sweep_df = run_velocity_sweep()

    fig_sweep = go.Figure()
    fig_sweep.add_trace(go.Scatter(x=sweep_df["Velocity (m/s)"], y=sweep_df["MPC RMSE"], mode='lines+markers', name='LTV-MPC (Knows V)', line=dict(color="#1f77b4", width=3)))
    fig_sweep.add_trace(go.Scatter(x=sweep_df["Velocity (m/s)"], y=sweep_df["Vanilla DPC RMSE"], mode='lines+markers', name='Static Vanilla DPC', line=dict(color="#d62728", width=3)))
    fig_sweep.add_trace(go.Scatter(x=sweep_df["Velocity (m/s)"], y=sweep_df["Lifted DPC RMSE"], mode='lines+markers', name='Static Lifted DPC', line=dict(color="#2ca02c", width=3)))
    fig_sweep.add_vline(x=2.0, line_dash="dash", line_color="rgba(255,255,255,0.5)", annotation_text="Data Collection Velocity")
    fig_sweep.update_layout(template="plotly_dark", plot_bgcolor="#040b14", paper_bgcolor="#040b14", xaxis_title="Simulated Velocity V_now (m/s)", yaxis_title="Tracking RMSE (Lower is Better)")
    st.warning("⚠️ **Vulnerability Exposed!** As the physical velocity deviates further away from the $2.0$ m/s training data, both Static DeePC controllers suffer catastrophic error inflation because their Hankel matrices expect a slow car. In our upcoming 'Residual Learning' update, we will introduce a secondary controller that learns this performance gap ($e = y - \\hat{y}$) and corrects the DPC on-the-fly!")

    st.warning(" **Vulnerability Exposed!** As the physical velocity deviates further away from the $2.0$ m/s training data, both Static DeePC controllers suffer catastrophic error inflation because their Hankel matrices expect a slow car. In our upcoming 'Residual Learning' update, we will introduce a secondary controller that learns this performance gap ($e = y - \hat{y}$) and corrects the DPC on-the-fly!")

