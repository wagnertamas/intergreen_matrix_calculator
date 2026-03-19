
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import threading
import queue
import json

# Try to import rl_trainer components
try:
    from rl_trainer import IndependentDQNTrainer, HAS_RL_LIBS
except ImportError:
    HAS_RL_LIBS = False
    print("Warning: rl_trainer not found or RL libs missing.")

class TransferLearningDialog:
    def __init__(self, parent, net_file, logic_file, detector_file):
        self.top = tk.Toplevel(parent)
        self.top.title("Transfer Learning (Fine-Tuning)")
        self.top.geometry("600x800")
        
        self.net_file = net_file
        self.logic_file = logic_file
        self.detector_file = detector_file
        
        self.log_queue = queue.Queue()
        self.trainer_thread = None
        self.trainer_instance = None
        
        # Load available junctions
        self.available_junctions = []
        if self.logic_file and os.path.exists(self.logic_file):
            try:
                with open(self.logic_file, 'r') as f:
                    data = json.load(f)
                    self.available_junctions = list(data.keys())
            except Exception as e:
                print(f"Error loading logic file for dropdown: {e}")
        
        self.model_path = tk.StringVar()
        
        self.model_path.trace("w", self.on_model_path_change)
        self.setup_ui()
        self.check_files()
        self.top.after(100, self.process_logs)

    def check_files(self):
        missing = []
        if not self.logic_file or not os.path.exists(self.logic_file): 
            missing.append("traffic_lights.json")
        if not self.detector_file or not os.path.exists(self.detector_file): 
            missing.append("detectors.add.xml")
        
        if missing:
            self.log(f"ERROR: Missing config files: {', '.join(missing)}")
            self.log("Please export SUMO files from the tools menu first!")
            self.btn_start.config(state="disabled")

    def setup_ui(self):
        # --- MODEL SELECTION (New for Transfer Learning) ---
        frame_model = tk.LabelFrame(self.top, text="1. Select Pre-trained Model", padx=10, pady=5)
        frame_model.pack(fill="x", padx=10, pady=5)
        
        tk.Label(frame_model, text="Model File (.zip):").pack(anchor="w")
        
        hbox = tk.Frame(frame_model)
        hbox.pack(fill="x", pady=2)
        
        self.entry_model = tk.Entry(hbox, textvariable=self.model_path)
        self.entry_model.pack(side="left", fill="x", expand=True, padx=(0, 5))
        
        tk.Button(hbox, text="Browse...", command=self.browse_model).pack(side="right")
        
        # --- JUNCTION SELECTION ---
        frame_target = tk.LabelFrame(self.top, text="2. Target Intersection", padx=10, pady=5)
        frame_target.pack(fill="x", padx=10, pady=5)
        
        tk.Label(frame_target, text="Select Junction ID for Fine-Tuning:").pack(anchor="w")
        self.combo_agent = ttk.Combobox(frame_target, values=self.available_junctions, state="readonly")
        if self.available_junctions:
            self.combo_agent.current(0)
        self.combo_agent.pack(fill="x", pady=2)
        
        # --- WANDB ---
        frame_wandb = tk.LabelFrame(self.top, text="3. Logging (WandB)", padx=10, pady=5)
        frame_wandb.pack(fill="x", padx=10, pady=5)
        
        tk.Label(frame_wandb, text="Project Name:").grid(row=0, column=0, sticky="w")
        self.entry_project = tk.Entry(frame_wandb)
        self.entry_project.insert(0, "sumo-rl-finetune")
        self.entry_project.grid(row=0, column=1, padx=5, sticky="ew")

        # --- HYPERPARAMETERS ---
        frame_hyper = tk.LabelFrame(self.top, text="4. Fine-Tuning Parameters", padx=10, pady=5)
        frame_hyper.pack(fill="x", padx=10, pady=5)

        params = [
            ("Total Timesteps:", "50000", "entry_steps"), # Default less actions for fine-tuning
            ("Learning Rate:", "0.00005", "entry_lr"), # Lower LR for fine-tuning
            ("Exploration Fraction:", "0.2", "entry_expl"), # Less exploration
            ("Batch Size:", "32", "entry_batch"),
            ("Buffer Size:", "50000", "entry_buffer"),
            ("Gamma:", "0.99", "entry_gamma"),
            ("Network Layers:", "2", "entry_num_layers"),
            ("Layer Size:", "64", "entry_layer_size"),
        ]

        for i, (label, default, attr_name) in enumerate(params):
            tk.Label(frame_hyper, text=label).grid(row=i, column=0, sticky="w")
            entry = tk.Entry(frame_hyper)
            entry.insert(0, default)
            entry.grid(row=i, column=1, padx=5, sticky="ew")
            setattr(self, attr_name, entry)

        # --- SUMO GUI ---
        self.var_gui_enabled = tk.BooleanVar(value=False)
        chk_gui = tk.Checkbutton(frame_hyper, text="Enable SUMO GUI", variable=self.var_gui_enabled)
        chk_gui.grid(row=len(params), column=0, sticky="w", pady=5)

        # --- FIX FORGALOM ---
        frame_flow = tk.LabelFrame(self.top, text="5. Traffic Generation", padx=10, pady=5)
        frame_flow.pack(fill="x", padx=10, pady=5)

        self.var_fixed_flow = tk.BooleanVar(value=False)
        chk_fixed = tk.Checkbutton(frame_flow, text="Fix forgalom (debug/teszt)",
                                    variable=self.var_fixed_flow, command=self.toggle_fixed_flow)
        chk_fixed.grid(row=0, column=0, sticky="w", columnspan=2)

        tk.Label(frame_flow, text="Target (veh/h/lane):").grid(row=1, column=0, sticky="w")
        self.entry_flow_target = tk.Entry(frame_flow, width=10)
        self.entry_flow_target.insert(0, "500")
        self.entry_flow_target.config(state="disabled")
        self.entry_flow_target.grid(row=1, column=1, padx=5)

        tk.Label(frame_flow, text="Spread (±):").grid(row=1, column=2, sticky="w")
        self.entry_flow_spread = tk.Entry(frame_flow, width=10)
        self.entry_flow_spread.insert(0, "0")
        self.entry_flow_spread.config(state="disabled")
        self.entry_flow_spread.grid(row=1, column=3, padx=5)

        tk.Label(frame_flow, text="(Spread=0 → minden epizód pontosan Target forgalommal fut)",
                 font=("Arial", 8)).grid(row=2, column=0, columnspan=4, sticky="w")

        # --- CONTROLS ---
        frame_btns = tk.Frame(self.top, pady=10)
        frame_btns.pack(fill="x")

        self.btn_start = tk.Button(frame_btns, text="Start Fine-Tuning", command=self.start_training, 
                                   bg="green", fg="white", font=("Arial", 10, "bold"))
        self.btn_start.pack(side="left", padx=20)

        self.btn_stop = tk.Button(frame_btns, text="Stop", command=self.stop_training, state="disabled",
                                  bg="red", fg="white")
        self.btn_stop.pack(side="left", padx=5)

        self.txt_log = tk.Text(self.top, height=10, state="disabled", bg="#f0f0f0")
        self.txt_log.pack(fill="both", expand=True, padx=10, pady=10)

    def on_model_path_change(self, *args):
        path = self.model_path.get()
        # Ensure widgets exist before configuring (trace might fire early)
        if hasattr(self, 'entry_num_layers'):
            if path.lower().endswith(".onnx"):
                self.entry_num_layers.config(state="disabled")
                self.entry_layer_size.config(state="disabled")
            else:
                self.entry_num_layers.config(state="normal")
                self.entry_layer_size.config(state="normal")

    def toggle_fixed_flow(self):
        state = "normal" if self.var_fixed_flow.get() else "disabled"
        self.entry_flow_target.config(state=state)
        self.entry_flow_spread.config(state=state)

    def browse_model(self):
        filename = filedialog.askopenfilename(
            title="Select Pre-trained Model",
            filetypes=[("Model Files", "*.zip *.onnx"), ("Zip Files", "*.zip"), ("ONNX Files", "*.onnx"), ("All Files", "*.*")]
        )
        if filename:
            self.model_path.set(filename)

    def log(self, msg):
        self.log_queue.put(msg)

    def process_logs(self):
        while not self.log_queue.empty():
            msg = self.log_queue.get()
            self.txt_log.config(state="normal")
            self.txt_log.insert("end", str(msg) + "\n")
            self.txt_log.see("end")
            self.txt_log.config(state="disabled")
        self.top.after(100, self.process_logs)

    def get_settings(self):
        return {
            "total_timesteps": int(self.entry_steps.get()),
            "wandb_project": self.entry_project.get(),
            "learning_rate": float(self.entry_lr.get()),
            "batch_size": int(self.entry_batch.get()),
            "buffer_size": int(self.entry_buffer.get()),
            "gamma": float(self.entry_gamma.get()),
            "exploration_fraction": float(self.entry_expl.get()),
            "num_layers": int(self.entry_num_layers.get()),
            "layer_size": int(self.entry_layer_size.get()),
            "single_agent_id": self.combo_agent.get(),
            "sumo_gui": self.var_gui_enabled.get(),
            "load_model_path": self.model_path.get(),
            "fixed_flow": self.var_fixed_flow.get(),
            "flow_target": int(self.entry_flow_target.get()) if self.var_fixed_flow.get() else None,
            "flow_spread": int(self.entry_flow_spread.get()) if self.var_fixed_flow.get() else None,
        }

    def start_training(self):
        if not HAS_RL_LIBS:
            messagebox.showerror("Error", "RL libraries (SB3, WandB) missing!")
            return

        settings = self.get_settings()
        
        if not settings["load_model_path"]:
            messagebox.showwarning("Warning", "Please select a model file for Transfer Learning!")
            return
            
        if not os.path.exists(settings["load_model_path"]):
             messagebox.showerror("Error", f"Model file not found: {settings['load_model_path']}")
             return

        if not settings["single_agent_id"]:
             messagebox.showwarning("Warning", "Please select a target Junction!")
             return

        try:
            # Fix forgalom beállítás
            fixed_flow = None
            if settings.get("fixed_flow") and settings.get("flow_target") is not None:
                fixed_flow = {'target': settings["flow_target"], 'spread': settings.get("flow_spread", 0)}

            # We reuse IndependentDQNTrainer but pass load_model_path
            self.trainer_instance = IndependentDQNTrainer(
                net_file=self.net_file,
                logic_file=self.logic_file,
                detector_file=self.detector_file,
                total_timesteps=settings["total_timesteps"],
                wandb_project=settings["wandb_project"],
                log_queue=self.log_queue,
                hyperparams=settings,
                single_agent_id=settings["single_agent_id"], # ALWAYS Single Agent for Fine-Tuning here
                sumo_gui=settings["sumo_gui"],
                load_model_path=settings["load_model_path"],
                fixed_flow=fixed_flow,
            )

            self.trainer_thread = threading.Thread(target=self.run_trainer_thread)
            self.trainer_thread.daemon = True
            self.trainer_thread.start()

            self.btn_start.config(state="disabled")
            self.btn_stop.config(state="normal")
            
        except ValueError as e:
            messagebox.showerror("Input Error", f"Invalid parameter: {e}")

    def run_trainer_thread(self):
        try:
            self.trainer_instance.run()
        except Exception as e:
            self.log(f"Thread Error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.top.after(0, self.on_training_finished)

    def stop_training(self):
        if self.trainer_instance:
            self.log("Stopping requested...")
            self.trainer_instance.stop_requested = True

    def on_training_finished(self):
        self.log("Training thread exited.")
        self.btn_start.config(state="normal")
        self.btn_stop.config(state="disabled")
