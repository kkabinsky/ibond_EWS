# -*- coding: utf-8 -*-
import subprocess

test_tex = r"""\documentclass[11pt,a4paper]{article}
\usepackage{fontspec}
\usepackage{xunicode}
\usepackage{xltxtra}
\XeTeXlinebreaklocale "th"
\XeTeXlinebreakskip=0pt plus 1pt
\defaultfontfeatures{Scale=1.2,Ligatures=TeX}
\setmainfont[Path = C:/Windows/Fonts/, Extension = .ttf, BoldFont = angsab, ItalicFont = angsai, BoldItalicFont = angsau]{angsana}

\begin{document}
ทดสอบระบบภาษาไทย Thai XeLaTeX Test
\end{document}
"""

with open("test_thai.tex", "w", encoding="utf-8") as f:
    f.write(test_tex)

res = subprocess.run(["xelatex", "-interaction=nonstopmode", "test_thai.tex"], capture_output=True, text=True)
print("Return code:", res.returncode)
