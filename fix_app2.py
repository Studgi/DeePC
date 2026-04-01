with open('website/app.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

for i, line in enumerate(lines):
    if "y_hankel_future.shape[1]" in line or "sample_idx = set" in line or "sample_idx.update" in line:
        if 310 < i < 330:
            lines[i] = "    " + line
    elif "for i in sample_idx:" in line or "raw_traj = y_hankel_future" in line or "shift = y_now" in line or "vis_traj = " in line or "vis_line = " in line or "plan_t = np.insert" in line or "if i in top_indices:" in line or "else:" in line:
        if 320 < i < 345:
            lines[i] = "    " + line
    elif "fig.add_trace" in line and ("rgba" in line or "Static Memory Options" in line or "Dominant Memories Picked" in line):
        if 330 < i < 350:
            lines[i] = "    " + line
    elif "x=plan_t, y=vis_line, mode='lines'" in line or "line=dict(color" in line or "showlegend=False, hoverinfo='skip'" in line:
        if 330 < i < 350:
            lines[i] = "    " + line
            
# Also fix the closing parenthesis inside the if 'DPC' block
for i, line in enumerate(lines):
    if line.strip() == "))":
        if 330 < i < 350:
            lines[i] = "    " + line

with open('website/app.py', 'w', encoding='utf-8') as f:
    f.writelines(lines)
