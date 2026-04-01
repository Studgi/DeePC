with open('website/app.py', encoding='utf-8') as f:
    lines = f.readlines()
out = []
in_deepc_info = False
for i, line in enumerate(lines):
    if 'if "DPC" in mode and g_weights is not None:' in line:
        in_deepc_info = True
        out.append(line)
        continue
    if in_deepc_info:
        if 'elif mode == "Standard MPC":' in line:
            in_deepc_info = False
            out.append('                ' + line.lstrip())
            continue
        if line.strip() == '':
            out.append('\n')
        elif line.startswith('                    '):
            out.append(line)
        elif line.startswith('                '):
            out.append('    ' + line)
        elif line.startswith('            '):
            out.append('        ' + line)
        elif line.startswith('        '):
            out.append('            ' + line)
        elif line.startswith('    '):
            out.append('                ' + line)
        else:
            out.append('                    ' + line.lstrip())
    else:
        out.append(line)
with open('website/app.py', 'w', encoding='utf-8') as f:
    f.writelines(out)
