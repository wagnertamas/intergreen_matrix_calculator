#!/usr/bin/env python3
"""
StrategyCompareDialog — RL stratégia-összehasonlító ablak a GUI-hoz.

Megnyitható a főablakból: "RL Összehasonlítás" gombbal.
Egy .zip modellt tölt be (ugyanolyan formátum mint a transfer learning),
és összehasonlítja a fix lámpaprogram / SUMO actuated / betöltött NN teljesítményét
R1C1_C kereszteződésen.
"""

import os
import sys
import json
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np

# ── Matplotlib Tkinter backend ──────────────────────────────────────────────
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.cm as cm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


class StrategyCompareDialog:
    """
    Összehasonlító ablak: fixed / actuated / betöltött NN modell.
    Ugyanazt a zip formátumot kezeli mint az IndependentDQNTrainer
    transfer learning funkciója.
    """

    def __init__(self, parent):
        self.top = tk.Toplevel(parent)
        self.top.title("RL Stratégia-összehasonlítás — R1C1_C")
        self.top.geometry("900x780")
        self.top.resizable(True, True)

        self._running = False
        self._model   = None
        self._model_name = ""

        self._build_ui()

    # ─────────────────────────────────────────────────────────────────────────
    # UI ÉPÍTÉS
    # ─────────────────────────────────────────────────────────────────────────
    def _build_ui(self):
        # ── Felső vezérlősor ─────────────────────────────────────────────────
        ctrl = ttk.LabelFrame(self.top, text="Beállítások", padding=8)
        ctrl.pack(side=tk.TOP, fill=tk.X, padx=8, pady=6)

        # Zip fájl sor
        zip_row = ttk.Frame(ctrl)
        zip_row.pack(fill=tk.X, pady=2)
        ttk.Label(zip_row, text="Model (.zip):").pack(side=tk.LEFT)
        self.zip_var = tk.StringVar(value="")
        ttk.Entry(zip_row, textvariable=self.zip_var, width=52).pack(side=tk.LEFT, padx=4)
        ttk.Button(zip_row, text="Tallózás…", command=self._browse_zip).pack(side=tk.LEFT)
        self.load_btn = ttk.Button(zip_row, text="Betöltés", command=self._load_model)
        self.load_btn.pack(side=tk.LEFT, padx=4)
        self.model_lbl = ttk.Label(zip_row, text="— nincs betöltve —",
                                   foreground="#888", width=20)
        self.model_lbl.pack(side=tk.LEFT, padx=6)

        # Paraméter sor
        param_row = ttk.Frame(ctrl)
        param_row.pack(fill=tk.X, pady=4)

        ttk.Label(param_row, text="Flow szintek (veh/h, vesszővel):").pack(side=tk.LEFT)
        self.flow_var = tk.StringVar(value="150, 300, 450, 600, 750, 900")
        ttk.Entry(param_row, textvariable=self.flow_var, width=30).pack(side=tk.LEFT, padx=6)

        ttk.Label(param_row, text="  Epizód/szint:").pack(side=tk.LEFT)
        self.ep_var = tk.IntVar(value=2)
        ttk.Spinbox(param_row, from_=1, to=5, textvariable=self.ep_var, width=4).pack(side=tk.LEFT)

        ttk.Label(param_row, text="  Időtartam (s):").pack(side=tk.LEFT, padx=(10, 0))
        self.dur_var = tk.IntVar(value=1200)
        ttk.Entry(param_row, textvariable=self.dur_var, width=6).pack(side=tk.LEFT, padx=4)

        ttk.Label(param_row, text="  ").pack(side=tk.LEFT)
        self.gui_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(param_row, text="SUMO-GUI", variable=self.gui_var).pack(side=tk.LEFT, padx=4)

        # Gombsor
        btn_row = ttk.Frame(ctrl)
        btn_row.pack(fill=tk.X, pady=2)
        self.run_btn = ttk.Button(btn_row, text="▶  Összehasonlítás indítása",
                                  command=self._start, state=tk.DISABLED)
        self.run_btn.pack(side=tk.LEFT)
        self.stop_btn = ttk.Button(btn_row, text="⏹  Leállítás",
                                   command=self._stop, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=6)
        self.save_btn = ttk.Button(btn_row, text="💾  Ábra mentése…",
                                   command=self._save_figure, state=tk.DISABLED)
        self.save_btn.pack(side=tk.LEFT, padx=6)

        # ── Notebook: Log + Eredmények ────────────────────────────────────────
        nb = ttk.Notebook(self.top)
        nb.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)

        # Log tab
        log_frame = ttk.Frame(nb)
        nb.add(log_frame, text="Napló")
        self.log_text = tk.Text(log_frame, height=12, font=("Courier", 9),
                                state=tk.DISABLED, wrap=tk.WORD)
        log_scroll = ttk.Scrollbar(log_frame, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=log_scroll.set)
        log_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text.pack(fill=tk.BOTH, expand=True)

        # Eredmény tab
        res_frame = ttk.Frame(nb)
        nb.add(res_frame, text="Eredmények")
        self.fig = Figure(figsize=(8.5, 4.5), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=res_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.nb = nb   # megőrizzük a tab-váltáshoz

    # ─────────────────────────────────────────────────────────────────────────
    # ZIP TALLÓZÁS & BETÖLTÉS
    # ─────────────────────────────────────────────────────────────────────────
    def _browse_zip(self):
        path = filedialog.askopenfilename(
            parent=self.top,
            title="RL model zip kiválasztása",
            filetypes=[("SB3 model", "*.zip"), ("Minden fájl", "*.*")],
            initialdir=SCRIPT_DIR,
        )
        if path:
            self.zip_var.set(path)

    def _load_model(self):
        path = self.zip_var.get().strip()
        if not path or not os.path.exists(path):
            messagebox.showerror("Hiba", "Érvénytelen fájlútvonal.", parent=self.top)
            return
        self._log(f"Modell betöltése: {os.path.basename(path)} …")
        self.load_btn.config(state=tk.DISABLED)
        threading.Thread(target=self._load_model_thread, args=(path,), daemon=True).start()

    def _load_model_thread(self, path):
        try:
            model, algo = _load_sb3_zip(path)
            self._model      = model
            self._model_name = f"{os.path.splitext(os.path.basename(path))[0]} [{algo}]"
            self.top.after(0, self._on_model_loaded, self._model_name)
        except Exception as e:
            self.top.after(0, self._on_model_error, str(e))

    def _on_model_loaded(self, name):
        self.model_lbl.config(text=f"✓ {name[:28]}", foreground="#1a7a1a")
        self.run_btn.config(state=tk.NORMAL)
        self.load_btn.config(state=tk.NORMAL)
        self._log(f"✓ Modell betöltve: {name}")

    def _on_model_error(self, msg):
        self.model_lbl.config(text="✗ Hiba", foreground="#cc0000")
        self.load_btn.config(state=tk.NORMAL)
        self._log(f"✗ Betöltési hiba: {msg}")
        messagebox.showerror("Betöltési hiba", msg, parent=self.top)

    # ─────────────────────────────────────────────────────────────────────────
    # FUTTATÁS
    # ─────────────────────────────────────────────────────────────────────────
    def _start(self):
        if self._running:
            return
        try:
            flows = [int(x.strip()) for x in self.flow_var.get().split(",") if x.strip()]
            if not flows:
                raise ValueError("Legalább egy flow szint kell.")
        except ValueError as e:
            messagebox.showerror("Hiba", str(e), parent=self.top)
            return

        self._running = True
        self.run_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.save_btn.config(state=tk.DISABLED)
        self._log("─" * 60)
        self._log(f"Futtatás: flow={flows}, ep/szint={self.ep_var.get()}, "
                  f"idő={self.dur_var.get()}s" +
                  ("  [SUMO-GUI]" if self.gui_var.get() else ""))

        params = {
            "flows":    flows,
            "episodes": self.ep_var.get(),
            "duration": self.dur_var.get(),
            "model":    self._model,
            "name":     self._model_name,
            "use_gui":  self.gui_var.get(),
        }
        threading.Thread(target=self._run_thread, args=(params,), daemon=True).start()

    def _stop(self):
        self._running = False
        self._log("⏹ Leállítás kérve…")

    def _run_thread(self, params):
        try:
            results = _run_comparison(
                flows      = params["flows"],
                episodes   = params["episodes"],
                duration   = params["duration"],
                model      = params["model"],
                model_name = params["name"],
                use_gui    = params.get("use_gui", False),
                log_fn     = lambda msg: self.top.after(0, self._log, msg),
                stop_fn    = lambda: not self._running,
            )
            self.top.after(0, self._show_results, results)
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            self.top.after(0, self._log, f"✗ Hiba: {e}\n{tb}")
        finally:
            self.top.after(0, self._on_done)

    def _on_done(self):
        self._running = False
        self.run_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)

    # ─────────────────────────────────────────────────────────────────────────
    # EREDMÉNYEK MEGJELENÍTÉSE
    # ─────────────────────────────────────────────────────────────────────────
    def _show_results(self, records):
        if not records:
            self._log("Nincs eredmény.")
            return

        import pandas as pd
        df = pd.DataFrame(records)

        self.fig.clear()

        strategies = ["fixed", "actuated"]
        if self._model_name:
            strategies.append(self._model_name)

        colors = {"fixed": "#4A90D9", "actuated": "#27AE60",
                  self._model_name: "#E67E22"}

        metrics = [
            ("avg_speed",   "Átl. sebesség [m/s]",     True),
            ("throughput",  "Throughput [jármű]",       True),
            ("halt_ratio",  "Megállási arány",           False),
            ("total_co2_g", "Össz. CO2 [g]",             False),
        ]

        axes = self.fig.subplots(1, 4)
        self.fig.suptitle("Stratégia-összehasonlítás — R1C1_C", fontsize=11)

        for ax, (col, ylabel, higher_better) in zip(axes, metrics):
            x = np.arange(len(strategies))
            vals  = [df[df["strategy"] == s][col].mean()           for s in strategies]
            errs  = [float(np.nan_to_num(df[df["strategy"] == s][col].std(ddof=0))) if len(df[df["strategy"]==s]) > 1 else 0
                     for s in strategies]
            bar_colors = [colors.get(s, "#999") for s in strategies]
            bars = ax.bar(x, vals, yerr=errs, capsize=5,
                          color=bar_colors, edgecolor="#333", linewidth=0.7)
            # Legjobb kiemelése
            best = max(vals) if higher_better else min(v for v in vals if v > 0)
            for b, v in zip(bars, vals):
                if abs(v - best) < 1e-9:
                    b.set_edgecolor("#c0392b")
                    b.set_linewidth(2.5)
            ax.set_xticks(x)
            short_labels = [s[:18] + "…" if len(s) > 18 else s for s in strategies]
            ax.set_xticklabels(short_labels, rotation=25, ha="right", fontsize=8)
            ax.set_ylabel(ylabel, fontsize=8)
            arrow = "↑ jobb" if higher_better else "↓ jobb"
            ax.set_title(f"{col}  ({arrow})", fontsize=9)
            ax.grid(axis="y", lw=0.5, alpha=0.45)

        self.fig.tight_layout(rect=[0, 0, 1, 0.95])
        self.canvas.draw()
        self.save_btn.config(state=tk.NORMAL)

        # Napló: összesítő táblázat
        self._log("\n── Összesítő (átlag) ──")
        for s in strategies:
            sub = df[df["strategy"] == s]
            if sub.empty:
                continue
            self._log(f"  {s:<35s}  "
                      f"spd={sub['avg_speed'].mean():.2f}  "
                      f"tp={sub['throughput'].mean():.0f}  "
                      f"halt={sub['halt_ratio'].mean():.3f}")

        # Ugrik az Eredmények tabra
        self.nb.select(1)

    def _save_figure(self):
        path = filedialog.asksaveasfilename(
            parent=self.top,
            defaultextension=".png",
            filetypes=[("PNG kép", "*.png"), ("PDF", "*.pdf")],
            initialfile="comparison_result.png",
        )
        if path:
            self.fig.savefig(path, dpi=150, bbox_inches="tight")
            self._log(f"Ábra mentve: {path}")

    # ─────────────────────────────────────────────────────────────────────────
    # NAPLÓ
    # ─────────────────────────────────────────────────────────────────────────
    def _log(self, msg: str):
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, msg + "\n")
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)


# ─────────────────────────────────────────────────────────────────────────────
# SB3 ZIP BETÖLTÉS — ugyanaz mint az IndependentDQNTrainer transfer learning
# ─────────────────────────────────────────────────────────────────────────────
def _load_sb3_zip(path: str):
    """
    Betölti az SB3 modellt a zip-ből (DQN / QRDQN / PPO / A2C sorrendben próbálva).
    Visszatér: (model, algo_name)
    """
    candidates = []
    try:
        from stable_baselines3 import DQN, PPO, A2C
        candidates += [("DQN", DQN), ("PPO", PPO), ("A2C", A2C)]
    except ImportError:
        pass
    try:
        from sb3_contrib import QRDQN
        candidates.insert(1, ("QRDQN", QRDQN))
    except ImportError:
        pass
    if not candidates:
        raise ImportError("A stable_baselines3 csomag nincs telepítve.")

    last_err = None
    for name, Cls in candidates:
        try:
            model = Cls.load(path, device="cpu")
            return model, name
        except Exception as e:
            last_err = e
    raise RuntimeError(f"Nem sikerült betölteni: {last_err}")


# ─────────────────────────────────────────────────────────────────────────────
# ÖSSZEHASONLÍTÁS FUTTATÁSA (a compare_strategies.py logikájával)
# ─────────────────────────────────────────────────────────────────────────────
def _run_comparison(flows, episodes, duration, model, model_name,
                    use_gui=False, log_fn=print, stop_fn=lambda: False):
    """
    Lefuttatja a három stratégiát és visszaadja a rekordok listáját.
    """
    # Importok itt, mert a felhasználó gépen futnak
    sys.path.insert(0, SCRIPT_DIR)
    from compare_strategies import (
        _import_sumo, _start_sumo, _stop_sumo,
        generate_route_file, cleanup_route_files,
        _setup_actuated, _LightAgent, TARGET_JID, LOGIC_FILE,
        WARMUP, DELTA_TIME,
    )
    import random, time

    # DURATION felülírása (a dialog értéke)
    import compare_strategies as cs
    orig_dur = cs.DURATION
    cs.DURATION = duration

    try:
        _import_sumo()
        import traci as _traci
        # a cs.traci és sumolib-ot is frissíteni kell, ha még None
        if cs.traci is None:
            cs.traci = _traci
        import sumolib as _sumolib
        if cs.sumolib is None:
            cs.sumolib = _sumolib

        records = []
        strategies = [("fixed", None), ("actuated", None)]
        if model is not None:
            strategies.append(("nn", model))

        total = len(flows) * episodes * len(strategies)
        done  = 0

        for flow in flows:
            for ep in range(episodes):
                if stop_fn():
                    log_fn("⏹ Leállítva.")
                    return records

                route_file = generate_route_file(flow, ep)

                for strat_key, m in strategies:
                    if stop_fn():
                        return records

                    label = strat_key if strat_key != "nn" else model_name
                    t0 = time.time()

                    try:
                        result = _run_episode(strat_key, route_file, m,
                                              duration, log_fn, use_gui=use_gui)
                        elapsed = time.time() - t0
                        rec = {
                            "strategy":    label,
                            "flow":        flow,
                            "episode":     ep,
                            "avg_speed":   result["avg_speed"],
                            "throughput":  result["throughput"],
                            "halt_ratio":  result["halt_ratio"],
                            "total_co2_g": result.get("total_co2_g", 0.0),
                        }
                        records.append(rec)
                        done += 1
                        log_fn(f"  [{done:3d}/{total}] {label:<35s} "
                               f"spd={result['avg_speed']:.2f}m/s  "
                               f"tp={result['throughput']:.0f}  "
                               f"halt={result['halt_ratio']:.3f}  "
                               f"co2={result.get('total_co2_g', 0):.0f}g  "
                               f"({elapsed:.0f}s)")
                    except Exception as e:
                        done += 1
                        import traceback
                        log_fn(f"  HIBA [{label}]: {e}\n{traceback.format_exc()}")

        _stop_sumo()
        cleanup_route_files()
        return records
    finally:
        cs.DURATION = orig_dur


def _run_episode(strategy, route_file, model, duration, log_fn, use_gui=False):
    """Egy epizód futtatása. use_gui=True esetén sumo-gui-val indul (--start --quit-on-end)."""
    from compare_strategies import (
        _import_sumo, _stop_sumo, _setup_actuated, _LightAgent,
        TARGET_JID, LOGIC_FILE, NET_FILE, DETECTOR_FILE, WARMUP, DELTA_TIME,
    )
    import compare_strategies as cs

    _import_sumo()
    traci = cs.traci

    # SUMO bináris kiválasztása
    sumo_bin = "sumo-gui" if use_gui else "sumo"
    sumo_args = [
        "-n", NET_FILE, "-r", route_file, "-a", DETECTOR_FILE,
        "--no-step-log", "true", "--ignore-route-errors", "true",
        "--no-warnings", "true", "--xml-validation", "never", "--random", "true",
    ]
    if use_gui:
        sumo_args += ["--start", "--quit-on-end", "true"]

    # Szimuláció (újra)indítása: traci már fut → close + start
    if cs._traci_running:
        try:
            traci.close()
        except Exception:
            pass
    traci.start([sumo_bin] + sumo_args)
    cs._traci_running = True
    jid = TARGET_JID

    all_loops = traci.inductionloop.getIDList()
    controlled = traci.trafficlight.getControlledLinks(jid)
    incoming_lanes = set()
    for group in controlled:
        for link in group:
            if link:
                incoming_lanes.add(link[0])
    dets = sorted([l for l in all_loops
                   if traci.inductionloop.getLaneID(l) in incoming_lanes])

    with open(LOGIC_FILE) as f:
        ldata = json.load(f)

    agent = _LightAgent(jid, ldata[jid], dets)

    if strategy == "actuated":
        _setup_actuated(jid)

    # Warmup
    for _ in range(WARMUP):
        traci.simulationStep()

    total_steps = (duration - WARMUP) // DELTA_TIME
    _model_keys = (set(model.policy.observation_space.spaces.keys())
                   if strategy == "nn" and model is not None else set())

    agent._reset_episode()   # Epizód metrikák egyszer, itt

    for _ in range(total_steps):
        agent._reset_obs()   # Obs akkumulátorok minden delta_time ablak elején

        for _ in range(DELTA_TIME):
            if strategy == "nn":
                agent.update()
            traci.simulationStep()
            agent.collect(list(incoming_lanes))

        # Ablak lefutott → obs akkumulált átlagokból (= edzéskori megfigyelés)
        if strategy == "nn" and agent.is_ready() and model is not None:
            obs = agent.get_obs()
            obs_f = {k: v for k, v in obs.items() if k in _model_keys}
            obs_b = {k: v.reshape(1, *v.shape) for k, v in obs_f.items()}
            action, _ = model.predict(obs_b, deterministic=True)
            agent.set_target(int(action[0]))

    return agent.metrics()
