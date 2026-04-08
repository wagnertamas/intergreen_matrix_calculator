import os
import re

def escape_tex(text):
    text = text.replace('\\', '\\textbackslash{}')
    text = text.replace('{', '\\{').replace('}', '\\}')
    for char in ('_', '%', '$', '#', '&'):
        text = text.replace(char, '\\' + char)
    text = text.replace('^', '\\textasciicircum{}')
    text = text.replace('~', '\\textasciitilde{}')
    text = text.replace('<', '$<$').replace('>', '$>$')
    text = re.sub(r'->', r'$\\rightarrow$', text)
    text = re.sub(r'<-', r'$\\leftarrow$', text)
    text = text.replace('ε', r'$\epsilon$').replace('≈', r'$\approx$').replace('∈', r'$\in$')
    text = text.replace('→', r'$\rightarrow$').replace('←', r'$\leftarrow$').replace('≤', r'$\le$').replace('≥', r'$\ge$')
    return text

def is_verbatim_trigger(line):
    # Tables
    if '|' in line and '--' in line: return True
    if '+---' in line: return True
    # If the line contains a lot of spaces separating values (like data columns)
    if re.search(r'\s{4,}', line.strip()) and ':' in line and len(line) > 20: return True
    # Specific keywords
    if line.strip().startswith('Mean:') or line.strip().startswith('TotalWaitingTime'): return True
    return False

def is_verbatim_end(line, next_line=""):
    # If the line is completely unindented normal text
    if line.strip() != "" and not line.startswith(' '):
        if not is_verbatim_trigger(line):
            return True
    return False

def convert():
    txt_path = 'methodology_and_findings.txt'
    tex_path = 'methodology_and_findings.tex'
    
    with open(txt_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    out = []
    out.append(r"""\documentclass[11pt,a4paper]{article}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage[magyar]{babel}
\usepackage[margin=2.2cm]{geometry}
\usepackage{xcolor}
\usepackage{titlesec}
\usepackage{tcolorbox}
\tcbuselibrary{skins, breakable}
\usepackage{enumitem}
\usepackage{graphicx}
\usepackage{hyperref}
\usepackage{lmodern}
\usepackage{amssymb}

\definecolor{primary}{RGB}{30, 80, 150}
\definecolor{accent}{RGB}{40, 140, 90}
\definecolor{danger}{RGB}{180, 50, 50}

\hypersetup{colorlinks=true, linkcolor=primary, urlcolor=primary}
\titleformat{\section}{\Large\bfseries\color{primary}}{\thesection}{1em}{}[\titlerule]
\titleformat{\subsection}{\large\bfseries\color{accent}}{\thesubsection}{1em}{}

\setlength{\parindent}{0pt}
\setlength{\parskip}{0.5em}

\begin{document}
\begin{center}
    \vspace*{1em}
    {\Huge\bfseries\color{primary} SUMO RL Traffic Light Control}\vspace{0.5em}\\
    {\Large\bfseries Methodology \& Research Findings}\vspace{0.8em}\\
    {\large Comprehensive Documentation for Paper Writing}
    \vspace{1.5em}
\end{center}
\tableofcontents
\newpage
""")

    i = 0
    in_verbatim = False

    while i < len(lines):
        line = lines[i].rstrip('\n')
        line = line.replace('─', '-').replace('│', '|').replace('┌', '+').replace('┐', '+').replace('└', '+').replace('┘', '+')
        line = line.replace('├', '+').replace('┤', '+').replace('┬', '+').replace('┴', '+').replace('┼', '+')
        
        # Section
        if line.startswith('====') and i + 2 < len(lines) and lines[i+2].startswith('===='):
            if in_verbatim:
                out.append(r"\end{verbatim}")
                in_verbatim = False
            title = lines[i+1].strip()
            title = re.sub(r'^([IXV]+)\.\s*', r'\1. ', title) # keep it as I. Title
            out.append(f"\\section*{{{escape_tex(title)}}}")
            out.append(f"\\addcontentsline{{toc}}{{section}}{{{escape_tex(title)}}}")
            i += 3
            continue
            
        # Subsection
        if i + 1 < len(lines) and lines[i+1].startswith('----') and len(lines[i+1]) > 5 and line.strip() != "":
            if in_verbatim:
                out.append(r"\end{verbatim}")
                in_verbatim = False
            title = line.strip()
            out.append(f"\\subsection*{{{escape_tex(title)}}}")
            out.append(f"\\addcontentsline{{toc}}{{subsection}}{{{escape_tex(title)}}}")
            i += 2
            continue

        if is_verbatim_trigger(line):
            if not in_verbatim:
                out.append(r"\vspace{1em}\begin{verbatim}")
                in_verbatim = True
                
        if in_verbatim and is_verbatim_end(line):
            out.append(r"\end{verbatim}\vspace{1em}")
            in_verbatim = False

        if in_verbatim:
            out.append(line)
        else:
            if line.strip() == "":
                out.append("")
            else:
                stripped = line.lstrip()
                prefix = ""
                if stripped.startswith('- ') or stripped.startswith('* '):
                    prefix = r"\par\noindent\hspace*{1em}\textbullet{} "
                    stripped = stripped[2:]
                elif stripped.startswith('□ '):
                    prefix = r"\par\noindent\hspace*{1em}$\square$ "
                    stripped = stripped[2:]
                elif stripped.startswith('✓ '):
                    prefix = r"\par\noindent\hspace*{1em}\checkmark "
                    stripped = stripped[2:]
                
                text = escape_tex(stripped)
                if ':' in text:
                    parts = text.split(':', 1)
                    if (parts[0].isupper() and " " in parts[0]) or sum(1 for c in parts[0] if c.isupper()) > 3:
                        text = r"\textbf{" + parts[0] + ":}" + parts[1]

                text = text.replace('[x]', r'\textbf{[x]}').replace('[ ]', r'\textbf{[ ]}')
                out.append(prefix + text)
        i += 1

    if in_verbatim:
        out.append(r"\end{verbatim}")

    out.append(r"\end{document}")

    with open(tex_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(out))
    print("Conversion finished.")

if __name__ == '__main__':
    convert()
