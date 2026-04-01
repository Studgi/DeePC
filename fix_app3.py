import re

with open('website/app.py', 'r', encoding='utf-8') as f:
    content = f.read()

old_text = r'''        # Plot Open-Loop Plan
        future_t = np.arange\(t_step \+ 1, t_step \+ 1 \+ pred_length\)
        if len\(future_t\) > 0:
            # If DPC, plot a faint cloud of "possible" future trajectories from the Hankel matrix(?:.*?)fig\.update_layout'''

new_text = '''        # Plot Open-Loop Plan
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

        fig.update_layout'''

res = re.sub(old_text, new_text, content, flags=re.DOTALL)
with open('website/app.py', 'w', encoding='utf-8') as f:
    f.write(res)
