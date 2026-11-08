"""
A LaTeX dokumentáció forrásának előállítása a Markdown oldalakból (pandoc kell),
és opcionálisan a PDF fordítása (pdflatex kell).

    python docs/sumo_morl/latex/build_latex.py          # .tex fájlok
    python docs/sumo_morl/latex/build_latex.py --pdf    # + PDF
"""
import argparse
import os
import re
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
DOCS = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(DOCS))

PAGES = ["README.md", "01_csomopont_es_jelzesterv.md", "02_megfigyeles.md", "03_celfuggvenyek.md",
         "04_kornyezet_api.md", "05_segedfuggvenyek.md", "06_feladatotletek.md"]

# Kódlista a függelékbe: (cím, útvonal a repó gyökeréhez képest)
CODE = [
    ("Csomag belépési pont", "sumo_morl/__init__.py"),
    ("A környezet", "sumo_morl/env.py"),
    ("Célfüggvények", "sumo_morl/objectives.py"),
    ("Hurokdetektorok", "sumo_morl/detectors.py"),
    ("Jelzésterv a közbenső időkből", "sumo_morl/intergreen.py"),
    ("Forgatókönyv", "sumo_morl/scenario.py"),
    ("Hálózat generálása", "sumo_morl/network.py"),
    ("Forgalmi igény", "sumo_morl/traffic.py"),
    ("Összehasonlító vezérlők", "sumo_morl/policies.py"),
    ("MORL segédfüggvények", "sumo_morl/morl.py"),
    ("Ábrák", "sumo_morl/visualize.py"),
    ("SUMO segédfüggvények", "sumo_morl/sumo_utils.py"),
    ("Közbensőidő-számítás: geometria", "geometry_utils.py"),
    ("Közbensőidő-számítás: hálózat beolvasása", "sumo_parser.py"),
    ("Demó: fix idejű program", "examples/sumo_morl/00_demo_fix_ido.py"),
    ("Példa: ismerkedés", "examples/sumo_morl/01_ismerkedes.py"),
    ("Példa: baseline-ok", "examples/sumo_morl/02_baseline_osszehasonlitas.py"),
    ("Példa: súlyozott tanítás (SB3)", "examples/sumo_morl/03_sb3_sulyozott_tanitas.py"),
    ("Példa: saját célfüggvény", "examples/sumo_morl/04_sajat_celfuggveny.py"),
    ("Tesztek: jelzésterv", "sumo_morl/tests/test_plan.py"),
    ("Tesztek: környezet", "sumo_morl/tests/test_env.py"),
    ("Telepítési leíró", "pyproject.toml"),
]


def _fix_tables(tex: str) -> str:
    """A hosszú szövegű oszlopokat tördelhető (p) oszlopokká alakítja."""
    def repl(m):
        spec, body = m.group(1), m.group(2)
        n = spec.count("l")
        rows = [r for r in re.split(r"\\\\\s*\n", body) if "&" in r]
        lens = [0] * n
        for r in rows:
            cells = r.split("&")
            for i, c in enumerate(cells[:n]):
                txt = re.sub(r"\\[a-zA-Z]+|[{}]|\\passthrough|lstinline!", "", c)
                lens[i] = max(lens[i], len(txt.strip()))
        if sum(lens) <= 75:
            return m.group(0)
        long_cols = [i for i in range(n) if lens[i] > 22]
        short = sum(lens[i] for i in range(n) if i not in long_cols)
        free = max(0.35, 0.96 - short / 95.0)
        tot = sum(lens[i] for i in long_cols)
        cols = []
        for i in range(n):
            if i in long_cols:
                cols.append(r">{\raggedright\arraybackslash}p{%.2f\linewidth}" % (free * lens[i] / tot))
            else:
                cols.append("l")
        return "\\begin{longtable}[]{@{}" + "".join(cols) + "@{}}" + body
    return re.sub(r"\\begin\{longtable\}\[\]\{@\{\}(l+)@\{\}\}(.*?\\end\{longtable\})", repl, tex, flags=re.S)


def md_to_tex(md: str) -> str:
    md = re.sub(r"^## Tartalom\n.*?(?=^## )", "", md, flags=re.S | re.M)   # tartalomjegyzék ki
    md = re.sub(r"\[([^\]]+)\]\((?!http)[^)]*\.md[^)]*\)", r"\1", md)      # belső linkek -> szöveg
    md = re.sub(r"\[([^\]]+)\]\(#[^)]*\)", r"\1", md)
    out = subprocess.run(["pandoc", "-f", "gfm", "-t", "latex", "--listings", "--top-level-division=section", "--columns=40"],
                         input=md, capture_output=True, text=True, check=True).stdout
    out = _fix_tables(out)
    out = out.replace(r"\includegraphics{img/kozbenso", r"\includegraphics[width=0.5\linewidth]{img/kozbenso")
    return out.replace(r"\includegraphics{img/", r"\includegraphics[width=\linewidth]{img/")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", action="store_true")
    args = ap.parse_args()
    body = []
    for p in PAGES:
        with open(os.path.join(DOCS, p), encoding="utf-8") as f:
            body.append(md_to_tex(f.read()))
    with open(os.path.join(HERE, "tartalom.tex"), "w", encoding="utf-8") as f:
        f.write("% Automatikusan generálva a docs/sumo_morl/*.md fájlokból (build_latex.py)\n")
        f.write("\n\n".join(body))
    with open(os.path.join(HERE, "kodok.tex"), "w", encoding="utf-8") as f:
        f.write("% Automatikusan generálva (build_latex.py)\n")
        for title, path in CODE:
            lang = "Python" if path.endswith(".py") else "{}"
            f.write(f"\\subsection{{{title}: \\texttt{{{path.replace('_', chr(92) + '_')}}}}}\n")
            f.write(f"\\lstinputlisting[language={lang}]{{\\rootdir/{path}}}\n\n")
    print("tartalom.tex, kodok.tex kész")
    if args.pdf:
        for _ in range(2):
            subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "sumo_morl_dokumentacio.tex"],
                           cwd=HERE, check=True, stdout=subprocess.DEVNULL)
        print("PDF kész:", os.path.join(HERE, "sumo_morl_dokumentacio.pdf"))


if __name__ == "__main__":
    main()
