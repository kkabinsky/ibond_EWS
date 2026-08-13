import re

with open("presentation.tex", "r", encoding="utf-8") as f:
    text = f.read()

# Replace any & that is not preceeded by \ and not inside tabular environment lines with \tabular or \begin{tabular}
lines = text.split("\n")
new_lines = []

in_tabular = False
for line in lines:
    if "\\begin{tabular}" in line or "\\begin{longtable}" in line:
        in_tabular = True
        new_lines.append(line)
        continue
    if "\\end{tabular}" in line or "\\end{longtable}" in line:
        in_tabular = False
        new_lines.append(line)
        continue
    
    if in_tabular:
        new_lines.append(line)
    else:
        # replace unescaped & with \&
        fixed_line = re.sub(r'(?<!\\)&', r'\\&', line)
        new_lines.append(fixed_line)

fixed_text = "\n".join(new_lines)

with open("presentation.tex", "w", encoding="utf-8") as f:
    f.write(fixed_text)

print("Fixed all unescaped ampersands in presentation.tex successfully!")
