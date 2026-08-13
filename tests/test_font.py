# -*- coding: utf-8 -*-
import subprocess

test_tex = r"""\documentclass[11pt,a4paper]{article}
\usepackage{fontspec}
\XeTeXlinebreaklocale "th"
\XeTeXlinebreakskip=0pt plus 1pt
\defaultfontfeatures{Scale=1.23,Ligatures=TeX}
\setmainfont{Angsana New}[
  BoldFont={Angsana New},
  ItalicFont={Angsana New},
  BoldItalicFont={Angsana New}
]
\setsansfont{Angsana New}[
  BoldFont={Angsana New},
  ItalicFont={Angsana New},
  BoldItalicFont={Angsana New}
]
\setmonofont{Angsana New}
\usepackage[unicode=true]{hyperref}

\begin{document}
ทดสอบภาษาไทย Thai XeLaTeX Test: สถาปัตยกรรมระบบ EWS
\end{document}
"""

with open("test_font.tex", "w", encoding="utf-8") as f:
    f.write(test_tex)

res = subprocess.run(["xelatex", "-interaction=nonstopmode", "test_font.tex"], capture_output=True, text=True)
print("Return code:", res.returncode)
with open("test_font.log", "r", encoding="utf-8", errors="ignore") as f:
    log = f.read()
    if "Error" in log:
        print("Log has errors!")
    else:
        print("Clean font rendering!")
