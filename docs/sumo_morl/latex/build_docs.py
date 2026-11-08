"""
A dokumentáció előállítása:

* ``kezikonyv.pdf``        – a könyvtár használata (LaTeX),
* ``programfelepites.pdf`` – a belső felépítés és a teljes forráskód (LaTeX),
* ``../api/``              – API-referencia a docstringekből (pdoc, HTML).

Használat (a csomag gyökérkönyvtárából)::

    pip install pdoc
    python docs/sumo_morl/latex/build_docs.py            # minden
    python docs/sumo_morl/latex/build_docs.py --no-pdf   # csak az API-referencia
"""
import argparse
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DOCS = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(DOCS))

MODULES = ["sumo_morl", "sumo_morl.env", "sumo_morl.scenario", "sumo_morl.intergreen",
           "sumo_morl.detectors", "sumo_morl.objectives", "sumo_morl.traffic", "sumo_morl.policies",
           "sumo_morl.morl", "sumo_morl.visualize", "sumo_morl.network", "sumo_morl.sumo_utils"]

# A programfelépítés függelékének kódlistája: (cím, útvonal a gyökérhez képest)
CODE = [
    ("Csomag belépési pont", "sumo_morl/__init__.py"),
    ("A környezet", "sumo_morl/env.py"),
    ("Forgatókönyv", "sumo_morl/scenario.py"),
    ("Jelzésterv a közbenső időkből", "sumo_morl/intergreen.py"),
    ("Hurokdetektorok", "sumo_morl/detectors.py"),
    ("Célfüggvények", "sumo_morl/objectives.py"),
    ("Forgalmi igény", "sumo_morl/traffic.py"),
    ("Összehasonlító vezérlők", "sumo_morl/policies.py"),
    ("MORL segédfüggvények", "sumo_morl/morl.py"),
    ("Ábrák", "sumo_morl/visualize.py"),
    ("Hálózat generálása", "sumo_morl/network.py"),
    ("SUMO segédfüggvények", "sumo_morl/sumo_utils.py"),
    ("Közbensőidő-számítás: geometria", "geometry_utils.py"),
    ("Közbensőidő-számítás: hálózat beolvasása", "sumo_parser.py"),
    ("Demó: fix idejű program", "examples/sumo_morl/00_demo_fix_ido.py"),
    ("Példa: ismerkedés", "examples/sumo_morl/01_ismerkedes.py"),
    ("Példa: összehasonlító vezérlők", "examples/sumo_morl/02_baseline_osszehasonlitas.py"),
    ("Példa: súlyozott tanítás (SB3)", "examples/sumo_morl/03_sb3_sulyozott_tanitas.py"),
    ("Példa: saját célfüggvény", "examples/sumo_morl/04_sajat_celfuggveny.py"),
    ("Tesztek: jelzésterv", "sumo_morl/tests/test_plan.py"),
    ("Tesztek: környezet", "sumo_morl/tests/test_env.py"),
    ("Telepítési leíró", "pyproject.toml"),
]


def write_code_listing() -> None:
    with open(os.path.join(HERE, "kodok.tex"), "w", encoding="utf-8") as f:
        f.write("% Automatikusan generálva (build_docs.py)\n")
        for title, path in CODE:
            lang = "Python" if path.endswith(".py") else "{}"
            f.write(f"\\subsection{{{title}}}\n\\kod{{{path.replace('_', chr(92) + '_')}}}\n")
            f.write(f"\\lstinputlisting[language={lang}]{{\\rootdir/{path}}}\n\n")


def build_pdf(name: str) -> None:
    for _ in range(2):
        subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", f"{name}.tex"],
                       cwd=HERE, check=True, stdout=subprocess.DEVNULL)
    for ext in ("aux", "log", "out", "toc"):
        p = os.path.join(HERE, f"{name}.{ext}")
        if os.path.exists(p):
            os.remove(p)
    print("kész:", os.path.join(HERE, f"{name}.pdf"))


def build_api() -> None:
    out = os.path.join(DOCS, "api")
    shutil.rmtree(out, ignore_errors=True)
    subprocess.run([sys.executable, "-m", "pdoc", "--docformat", "google", "--no-show-source",
                    "-o", out, *MODULES], cwd=ROOT, check=True)
    print("kész:", os.path.join(out, "index.html"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-pdf", action="store_true")
    ap.add_argument("--no-api", action="store_true")
    a = ap.parse_args()
    write_code_listing()
    if not a.no_pdf:
        build_pdf("kezikonyv")
        build_pdf("programfelepites")
    if not a.no_api:
        build_api()


if __name__ == "__main__":
    main()
