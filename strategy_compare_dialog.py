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
        self._last_strategy_order = []

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

        # Vezérlési módok
        mode_row = ttk.Frame(ctrl)
        mode_row.pack(fill=tk.X, pady=2)
        ttk.Label(mode_row, text="Vezérlési módok:").pack(side=tk.LEFT)
        self.var_mode_fixed = tk.BooleanVar(value=True)
        self.var_mode_actuated = tk.BooleanVar(value=True)
        self.var_mode_nn = tk.BooleanVar(value=True)
        ttk.Checkbutton(mode_row, text="Fixed", variable=self.var_mode_fixed,
                command=self._update_run_button_state).pack(side=tk.LEFT, padx=6)
        ttk.Checkbutton(mode_row, text="Actuated", variable=self.var_mode_actuated,
                command=self._update_run_button_state).pack(side=tk.LEFT, padx=2)
        ttk.Checkbutton(mode_row, text="NN model", variable=self.var_mode_nn,
                command=self._update_run_button_state).pack(side=tk.LEFT, padx=2)

        # Gombsor
        btn_row = ttk.Frame(ctrl)
        btn_row.pack(fill=tk.X, pady=2)
        self.run_btn = ttk.Button(btn_row, text="▶  Összehasonlítás indítása",
                      command=self._start, state=tk.NORMAL)
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

        # Fundamentális diagram tab
        fd_frame = ttk.Frame(nb)
        nb.add(fd_frame, text="Fundamentális")
        self.fd_fig = Figure(figsize=(8.5, 4.5), dpi=100)
        self.fd_canvas = FigureCanvasTkAgg(self.fd_fig, master=fd_frame)
        self.fd_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.nb = nb   # megőrizzük a tab-váltáshoz
        self._update_run_button_state()

    def _get_selected_modes(self):
        modes = []
        if self.var_mode_fixed.get():
            modes.append("fixed")
        if self.var_mode_actuated.get():
            modes.append("actuated")
        if self.var_mode_nn.get():
            modes.append("nn")
        return modes

    def _update_run_button_state(self):
        if self._running:
            self.run_btn.config(state=tk.DISABLED)
            return
        modes = self._get_selected_modes()
        runnable = False
        if "fixed" in modes or "actuated" in modes:
            runnable = True
        if "nn" in modes and self._model is not None:
            runnable = True
        self.run_btn.config(state=(tk.NORMAL if runnable else tk.DISABLED))

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
        self.load_btn.config(state=tk.NORMAL)
        self._update_run_button_state()
        self._log(f"✓ Modell betöltve: {name}")

    def _on_model_error(self, msg):
        self.model_lbl.config(text="✗ Hiba", foreground="#cc0000")
        self.load_btn.config(state=tk.NORMAL)
        self._update_run_button_state()
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

        selected_modes = self._get_selected_modes()
        if not selected_modes:
            messagebox.showerror("Hiba", "Válassz legalább egy vezérlési módot.", parent=self.top)
            return
        if "nn" in selected_modes and self._model is None:
            messagebox.showerror("Hiba", "Az NN módhoz tölts be modellt, vagy vedd ki az NN pipát.", parent=self.top)
            return

        label_map = {"fixed": "fixed", "actuated": "actuated", "nn": self._model_name}
        self._last_strategy_order = [label_map[m] for m in selected_modes if label_map.get(m)]

        self._running = True
        self.run_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.save_btn.config(state=tk.DISABLED)
        self._log("─" * 60)
        self._log(f"Futtatás: flow={flows}, ep/szint={self.ep_var.get()}, "
                  f"idő={self.dur_var.get()}s, módok={selected_modes}" +
                  ("  [SUMO-GUI]" if self.gui_var.get() else ""))

        params = {
            "flows":    flows,
            "episodes": self.ep_var.get(),
            "duration": self.dur_var.get(),
            "model":    self._model,
            "name":     self._model_name,
            "modes":    selected_modes,
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
                selected_modes = params.get("modes"),
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
        self.stop_btn.config(state=tk.DISABLED)
        self._update_run_button_state()

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

        if self._last_strategy_order:
            strategies = [s for s in self._last_strategy_order if s in set(df["strategy"].unique())]
        else:
            strategies = ["fixed", "actuated"]
            if self._model_name and self._model_name in set(df["strategy"].unique()):
                strategies.append(self._model_name)

        colors = {"fixed": "#4A90D9", "actuated": "#27AE60",
                  self._model_name: "#E67E22"}

        metrics = [
            ("avg_speed",   "Átl. sebesség [m/s]",     True),
            ("throughput",  "Throughput [jármű]",       True),
            ("halt_ratio",  "Megállási arány",           False),
            ("total_co2_g", "Össz. CO2 [g]",             False),
        ]

        axes = self.fig.subplots(2, 4)
        self.fig.suptitle("Stratégia-összehasonlítás — R1C1_C (aggregált + flow-szint)", fontsize=11)

        for idx, (col, ylabel, higher_better) in enumerate(metrics):
            ax = axes[0][idx]
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
            ax.set_title(f"Aggregált: {col}  ({arrow})", fontsize=9)
            ax.grid(axis="y", lw=0.5, alpha=0.45)

            ax_f = axes[1][idx]
            flow_stats = df.groupby(["flow", "strategy"])[col].mean().reset_index()
            for s in strategies:
                sub = flow_stats[flow_stats["strategy"] == s].sort_values("flow")
                if sub.empty:
                    continue
                ax_f.plot(sub["flow"].values,
                          sub[col].values,
                          marker="o",
                          linewidth=1.8,
                          markersize=4,
                          color=colors.get(s, "#999"),
                          label=s)
            ax_f.set_xlabel("Flow [veh/h]", fontsize=8)
            ax_f.set_ylabel(ylabel, fontsize=8)
            ax_f.set_title(f"Flow-szint: {col}", fontsize=9)
            ax_f.grid(axis="both", lw=0.5, alpha=0.35)
            if idx == 0:
                ax_f.legend(fontsize=7, frameon=True)

        self.fig.tight_layout(rect=[0, 0, 1, 0.95])
        self.canvas.draw()
        self.save_btn.config(state=tk.NORMAL)

        # 3. fül: fundamentális (proxy) diagramok
        self._show_fundamental_diagrams(df, strategies, colors)

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

        self._log("\n── Flow-szintű átlagok ──")
        for fl in sorted(df["flow"].unique()):
            self._log(f"  Flow={int(fl)}")
            for s in strategies:
                sub = df[(df["strategy"] == s) & (df["flow"] == fl)]
                if sub.empty:
                    continue
                self._log(f"    {s:<31s} spd={sub['avg_speed'].mean():.2f}  "
                          f"tp={sub['throughput'].mean():.0f}  halt={sub['halt_ratio'].mean():.3f}")

        # Ugrik az Eredmények tabra
        self.nb.select(1)

    def _show_fundamental_diagrams(self, df, strategies, colors):
        """Fundamentális (proxy) diagramok kirajzolása.

        Klasszikus alakok occupancy-proxy sűrűséggel:
          - D*-V
          - Q-D*
          - Q-V

        D* = átlagos detektor occupancy (vagy fallback: halt_ratio),
        Q = throughput [veh/h] (epizódidőre normalizálva).
        """
        self.fd_fig.clear()
        axes = self.fd_fig.subplots(1, 3)
        self.fd_fig.suptitle("Fundamentális diagramok (proxy) — R1C1_C", fontsize=11)

        df = df.copy()
        if "avg_occupancy" in df.columns and not np.all(np.isnan(df["avg_occupancy"].values)):
            df["density_proxy"] = df["avg_occupancy"].clip(0, 1)
            density_label = "D* [occupancy]"
        else:
            df["density_proxy"] = df["halt_ratio"].clip(0, 1)
            density_label = "D* [halt_ratio proxy]"

        dur = float(df["duration_s"].iloc[0]) if "duration_s" in df.columns and len(df) > 0 else 1.0
        dur = max(dur, 1.0)
        df["q_vehph"] = df["throughput"] * (3600.0 / dur)

        # 1) D*-V
        ax = axes[0]
        for s in strategies:
            sub = df[df["strategy"] == s]
            if sub.empty:
                continue
            ax.scatter(sub["density_proxy"], sub["avg_speed"], s=24,
                       alpha=0.72, color=colors.get(s, "#888"), label=s)
            g = sub.groupby("flow", as_index=False)[["density_proxy", "avg_speed"]].mean().sort_values("flow")
            ax.plot(g["density_proxy"], g["avg_speed"], marker="o", linewidth=1.5,
                    markersize=4, color=colors.get(s, "#888"))
        ax.set_xlabel(density_label, fontsize=8)
        ax.set_ylabel("V [m/s]", fontsize=8)
        ax.set_title("D*-V", fontsize=9)
        ax.grid(alpha=0.35, lw=0.5)
        ax.legend(fontsize=7, frameon=True)

        # 2) Q-D*
        ax = axes[1]
        for s in strategies:
            sub = df[df["strategy"] == s]
            if sub.empty:
                continue
            ax.scatter(sub["density_proxy"], sub["q_vehph"], s=24,
                       alpha=0.72, color=colors.get(s, "#888"))
            g = sub.groupby("flow", as_index=False)[["density_proxy", "q_vehph"]].mean().sort_values("flow")
            ax.plot(g["density_proxy"], g["q_vehph"], marker="o", linewidth=1.5,
                    markersize=4, color=colors.get(s, "#888"))
        ax.set_xlabel(density_label, fontsize=8)
        ax.set_ylabel("Q [veh/h]", fontsize=8)
        ax.set_title("Q-D*", fontsize=9)
        ax.grid(alpha=0.35, lw=0.5)

        # 3) Q-V
        ax = axes[2]
        for s in strategies:
            sub = df[df["strategy"] == s]
            if sub.empty:
                continue
            ax.scatter(sub["avg_speed"], sub["q_vehph"], s=24,
                       alpha=0.72, color=colors.get(s, "#888"))
            g = sub.groupby("flow", as_index=False)[["avg_speed", "q_vehph"]].mean().sort_values("flow")
            ax.plot(g["avg_speed"], g["q_vehph"], marker="o", linewidth=1.5,
                    markersize=4, color=colors.get(s, "#888"))
        ax.set_xlabel("V [m/s]", fontsize=8)
        ax.set_ylabel("Q [veh/h]", fontsize=8)
        ax.set_title("Q-V", fontsize=9)
        ax.grid(alpha=0.35, lw=0.5)

        self.fd_fig.tight_layout(rect=[0, 0, 1, 0.95])
        self.fd_canvas.draw()

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
                    selected_modes=None,
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
        _build_obs_batch_for_model,
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
        selected_modes = selected_modes or ["fixed", "actuated", "nn"]
        strategies = []
        if "fixed" in selected_modes:
            strategies.append(("fixed", None))
        if "actuated" in selected_modes:
            strategies.append(("actuated", None))
        if "nn" in selected_modes and model is not None:
            strategies.append(("nn", model))

        if not strategies:
            log_fn("[INFO] Nincs futtatható stratégia kiválasztva.")
            return records

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
                            "duration_s":  duration,
                            "avg_speed":   result["avg_speed"],
                            "throughput":  result["throughput"],
                            "halt_ratio":  result["halt_ratio"],
                            "avg_occupancy": result.get("avg_occupancy", np.nan),
                            "total_co2_g": result.get("total_co2_g", 0.0),
                            "agent_id":    result.get("agent_id", TARGET_JID),
                            "total_decisions": result.get("total_decisions", 0),
                            "action_counts": result.get("action_counts", {}),
                        }
                        records.append(rec)
                        done += 1
                        log_fn(f"  [{done:3d}/{total}] {label:<35s} "
                               f"spd={result['avg_speed']:.2f}m/s  "
                               f"tp={result['throughput']:.0f}  "
                               f"halt={result['halt_ratio']:.3f}  "
                               f"co2={result.get('total_co2_g', 0):.0f}g  "
                               f"({elapsed:.0f}s)")
                        if result.get("action_counts"):
                            total_decisions = max(int(result.get("total_decisions", 0)), 1)
                            dist = ", ".join(
                                f"a{int(a)}={int(c)} ({100.0 * int(c) / total_decisions:.0f}%)"
                                for a, c in sorted(result["action_counts"].items(), key=lambda kv: int(kv[0]))
                            )
                            log_fn(f"         ↳ Akció-eloszlás [{result.get('agent_id', TARGET_JID)}]: {dist}")
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
        _import_sumo, _stop_sumo, _LightAgent, _CurveActuatedController,
        _build_obs_batch_for_model,
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

    actuated_ctrl = None
    if strategy == "actuated":
        actuated_ctrl = _CurveActuatedController(jid, ldata[jid], dets)

    # Warmup
    for _ in range(WARMUP):
        traci.simulationStep()

    total_steps = (duration - WARMUP) // DELTA_TIME
    agent._reset_episode()   # Epizód metrikák egyszer, itt
    action_counts = {}

    for _ in range(total_steps):
        agent._reset_obs()   # Obs akkumulátorok minden delta_time ablak elején

        if strategy == "actuated" and actuated_ctrl is not None:
            actuated_ctrl.step()

        for _ in range(DELTA_TIME):
            if strategy == "nn":
                agent.update()
            traci.simulationStep()
            agent.collect(list(incoming_lanes))

        # Ablak lefutott → obs akkumulált átlagokból (= edzéskori megfigyelés)
        if strategy == "nn" and agent.is_ready() and model is not None:
            obs = agent.get_obs()
            obs_b = _build_obs_batch_for_model(model, obs)
            action, _ = model.predict(obs_b, deterministic=True)
            action_int = int(action[0])
            action_counts[action_int] = action_counts.get(action_int, 0) + 1
            agent.set_target(action_int)

    result = agent.metrics()
    if strategy == "nn":
        total_decisions = int(sum(action_counts.values()))
        result["agent_id"] = jid
        result["total_decisions"] = total_decisions
        result["action_counts"] = dict(sorted(action_counts.items()))
    return result
