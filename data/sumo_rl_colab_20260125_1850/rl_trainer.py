import tkinter as tk
from tkinter import ttk, messagebox
import gymnasium as gym
import numpy as np
import threading
import queue
import time
import os
import sys

# Opcionális importok ellenőrzése
try:
    from stable_baselines3 import DQN
    from stable_baselines3.common.logger import configure
    from stable_baselines3.common.vec_env import DummyVecEnv
    import wandb
    import torch
    HAS_RL_LIBS = True
except ImportError:
    HAS_RL_LIBS = False
    print("Figyelem: RL könyvtárak (stable-baselines3, wandb, torch) hiányoznak.")

from sumo_rl_environment import SumoRLEnvironment

# Export modul importálása (ha létezik)
try:
    from export_utils import export_to_colab_package
    HAS_EXPORT = True
except ImportError:
    HAS_EXPORT = False

# =============================================================================
# 1. TRÉNER LOGIKA (IndependentDQNTrainer)
# =============================================================================

class IndependentDQNTrainer:
    def __init__(self, 
                 net_file, logic_file, detector_file, 
                 total_timesteps=100000, 
                 wandb_project="sumo-rl",
                 log_queue=None,
                 hyperparams=None,
                 reward_weights=None,
                 n_envs=1):
        
        self.net_file = net_file
        self.logic_file = logic_file
        self.detector_file = detector_file
        self.total_timesteps = total_timesteps
        self.project_name = wandb_project
        self.log_queue = log_queue
        self.hyperparams = hyperparams or {}
        self.reward_weights = reward_weights or {'time': 1.0, 'co2': 1.0}
        
        self.stop_requested = False
        self.agents = {}
        self.env = None
        self.reward_smoothing = {} # Futó átlag tárolása

    def log(self, msg):
        if self.log_queue: self.log_queue.put(msg)
        else: print(msg)

    def run(self):
        self.log("Initializing Environment...")
        
        # 1. WandB init és SWEEP Támogatás
        if HAS_RL_LIBS:
            if wandb.run is None:
                try:
                    wandb.init(project=self.project_name, config=self.hyperparams, sync_tensorboard=False)
                except Exception as e:
                    self.log(f"WandB init failed (skipped): {e}")

        # Config összefésülése (Sweep config felülírja a GUI beállításokat)
        current_config = self.hyperparams.copy()
        if HAS_RL_LIBS and wandb.run:
            for key in wandb.config.keys():
                current_config[key] = wandb.config[key]
            self.log("WandB config applied (Sweep compatible).")

        # 2. Környezet létrehozása
        self.env = SumoRLEnvironment(
            net_file=self.net_file,
            logic_json_file=self.logic_file,
            detector_file=self.detector_file,
            reward_weights=self.reward_weights,
            sumo_gui=False,
            min_green_time=5,
            delta_time=5,
            random_traffic=True
        )

        # 3. KÖRNYEZET INDÍTÁSA
        self.log("Starting SUMO...")
        try:
            obs, infos = self.env.reset()
        except Exception as e:
            self.log(f"CRITICAL ERROR during env.reset(): {e}")
            return

        agent_ids = list(self.env.agents.keys())
        self.log(f"Agents discovered: {agent_ids}")

        if not HAS_RL_LIBS:
            self.log("RL libraries missing. Stopping.")
            self.env.close()
            return

        # 4. Modellek létrehozása
        runs_dir = os.path.join(os.path.dirname(self.net_file), "runs")
        
        lr = float(current_config.get("learning_rate", 1e-4))
        bs = int(current_config.get("batch_size", 32))
        buf = int(current_config.get("buffer_size", 10000))
        
        # PARAMÉTEREK:
        # Gamma: Mostantól konstans (WandB/GUI vezérelt), nem csökken.
        gamma = float(current_config.get("gamma", 0.99))
        
        # Epsilon (Exploration): Ezt az SB3 kezeli, de mi logoljuk majd.
        expl_fraction = float(current_config.get("exploration_fraction", 0.5))
        
        # Architektúra
        num_layers = int(current_config.get("num_layers", 2))
        layer_size = int(current_config.get("layer_size", 64))
        net_arch = [layer_size] * num_layers

        for jid in agent_ids:
            self.reward_smoothing[jid] = 0.0 

            agent_obs_space = self.env.observation_space[jid]
            agent_act_space = self.env.action_space[jid]
            
            def make_dummy_wrapper():
                class SingleAgentWrapper(gym.Env):
                    def __init__(self):
                        self.observation_space = agent_obs_space
                        self.action_space = agent_act_space
                    def reset(self, **kwargs): return self.observation_space.sample(), {}
                    def step(self, a): return self.observation_space.sample(), 0, False, False, {}
                return SingleAgentWrapper()

            model_env = DummyVecEnv([make_dummy_wrapper])
            tb_log = os.path.join(runs_dir, self.project_name, jid)
            
            self.agents[jid] = DQN(
                "MultiInputPolicy",
                model_env,
                learning_rate=lr,
                buffer_size=buf,
                batch_size=bs,
                gamma=gamma, # Fix Gamma
                exploration_fraction=expl_fraction, # SB3 ebből építi az Epsilon ütemezést
                policy_kwargs=dict(net_arch=net_arch),
                verbose=0,
                tensorboard_log=tb_log,
                device="auto"
            )
            self.agents[jid].set_logger(configure(tb_log, ["stdout", "tensorboard"]))

        # 5. Tanítási Ciklus
        self.log(f"Starting Training Loop (Gamma={gamma}, Expl_Fraction={expl_fraction})...")
        global_step = 0
        start_time = time.time()
        
        while global_step < self.total_timesteps and not self.stop_requested:
            
            # Progress update az SB3-nak (ez elengedhetetlen az Epsilon csökkentéshez!)
            progress = global_step / self.total_timesteps
            remaining_progress = 1.0 - progress
            
            for model in self.agents.values():
                model._current_progress_remaining = remaining_progress

            # --- AKCIÓVÁLASZTÁS ---
            actions = {}
            for jid, model in self.agents.items():
                agent_obs = {k: v.reshape(1, *v.shape) for k, v in obs[jid].items()}
                action, _ = model.predict(agent_obs, deterministic=False)
                actions[jid] = int(action[0])

            # --- LÉPÉS ---
            next_obs, rewards, global_done, _, infos = self.env.step(actions)
            global_step += 1

            # --- BUFFER & TRAIN ---
            for jid, model in self.agents.items():
                o = {k: v.reshape(1, *v.shape) for k, v in obs[jid].items()}
                no = {k: v.reshape(1, *v.shape) for k, v in next_obs[jid].items()}
                a = np.array([[actions[jid]]])
                r = np.array([rewards[jid]])
                d = np.array([global_done])
                
                model.replay_buffer.add(o, no, a, r, d, [infos[jid]])

                self.reward_smoothing[jid] = 0.95 * self.reward_smoothing[jid] + 0.05 * rewards[jid]

                if global_step > 100 and global_step % 4 == 0:
                    model.train(gradient_steps=1, batch_size=bs)
                
                model.num_timesteps += 1

            obs = next_obs
            
            # --- CUSTOM WANDB LOGGING ---
            if global_step % 100 == 0:
                elapsed = time.time() - start_time
                fps = int(global_step / (elapsed + 1e-5))
                self.log(f"Step: {global_step}/{self.total_timesteps} | FPS: {fps}")
                
                if wandb.run:
                    log_dict = {
                        "global_step": global_step, 
                        "fps": fps,
                        "train/gamma": gamma, # Konstans Gamma logolása
                    }
                    
                    for jid, model in self.agents.items():
                        # Metrikák kinyerése
                        curr_lr = model.policy.optimizer.param_groups[0]["lr"]
                        curr_loss = model.logger.name_to_value.get("train/loss", 0.0)
                        
                        # EPSILON (Exploration Rate) kinyerése
                        # Az SB3 exploration_schedule függvénye adja vissza az aktuális értéket
                        curr_epsilon = model.exploration_schedule(remaining_progress)
                        
                        log_dict[f"{jid}/global_step"] = global_step
                        log_dict[f"{jid}/train/learning_rate"] = curr_lr
                        log_dict[f"{jid}/train/loss"] = curr_loss
                        log_dict[f"{jid}/train/epsilon"] = curr_epsilon # ITT AZ ÚJ EPSILON LOG
                        log_dict[f"reward_smooth/{jid}"] = self.reward_smoothing[jid]

                    wandb.log(log_dict, commit=True)

            if global_done:
                obs, _ = self.env.reset()

        # 6. ONNX EXPORT
        self.log("Training Finished. Exporting ONNX models...")
        self.export_onnx_models(runs_dir)

        self.env.close()
        if wandb.run:
            wandb.finish()
        self.log("Done.")

    def export_onnx_models(self, output_dir):
        if not HAS_RL_LIBS: return

        save_path = os.path.join(output_dir, "onnx_models")
        os.makedirs(save_path, exist_ok=True)

        for jid, model in self.agents.items():
            try:
                # Egyszerű wrapper az exportáláshoz
                class OnnxablePolicy(torch.nn.Module):
                    def __init__(self, policy):
                        super().__init__()
                        self.policy = policy
                    
                    def forward(self, phase, occupancy, flow):
                        observation = {
                            "phase": phase,
                            "occupancy": occupancy,
                            "flow": flow
                        }
                        return self.policy.q_net(self.policy.extract_features(observation, self.policy.features_extractor))

                onnx_policy = OnnxablePolicy(model.policy).to("cpu")
                
                obs_space = self.env.observation_space[jid]
                s_phase = torch.zeros((1, *obs_space["phase"].shape), dtype=torch.float32)
                s_occ = torch.zeros((1, *obs_space["occupancy"].shape), dtype=torch.float32)
                s_flow = torch.zeros((1, *obs_space["flow"].shape), dtype=torch.float32)
                
                model_path = os.path.join(save_path, f"{jid}.onnx")
                
                torch.onnx.export(
                    onnx_policy,
                    (s_phase, s_occ, s_flow),
                    model_path,
                    opset_version=11,
                    input_names=["input_phase", "input_occupancy", "input_flow"],
                    output_names=["output_q_values"]
                )
                
                if wandb.run:
                    wandb.save(model_path)
                
                self.log(f"Exported: {model_path}")

            except Exception as e:
                self.log(f"ONNX Export Failed for {jid}: {e}")


# =============================================================================
# 2. GUI DIALOG
# =============================================================================

class TrainingDialog:
    def __init__(self, parent, net_file, logic_file, detector_file):
        self.top = tk.Toplevel(parent)
        self.top.title("Reinforcement Learning Trainer")
        self.top.geometry("600x750")
        
        self.net_file = net_file
        self.logic_file = logic_file
        self.detector_file = detector_file
        
        self.log_queue = queue.Queue()
        self.trainer_thread = None
        self.trainer_instance = None
        
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
            self.log(f"HIBA: Hiányzó fájlok: {', '.join(missing)}")
            self.log("Kérlek először exportáld a SUMO fájlokat a főablakban (SUMO Export)!")
            self.btn_start.config(state="disabled")

    def setup_ui(self):
        frame_wandb = tk.LabelFrame(self.top, text="WandB Logging", padx=10, pady=5)
        frame_wandb.pack(fill="x", padx=10, pady=5)
        
        tk.Label(frame_wandb, text="Project Name:").grid(row=0, column=0, sticky="w")
        self.entry_project = tk.Entry(frame_wandb)
        self.entry_project.insert(0, "sumo-rl-single")
        self.entry_project.grid(row=0, column=1, padx=5, sticky="ew")

        tk.Label(frame_wandb, text="API Key (Optional):").grid(row=1, column=0, sticky="w")
        self.entry_apikey = tk.Entry(frame_wandb, show="*")
        self.entry_apikey.grid(row=1, column=1, padx=5, sticky="ew")

        frame_hyper = tk.LabelFrame(self.top, text="Hyperparameters", padx=10, pady=5)
        frame_hyper.pack(fill="x", padx=10, pady=5)

        params = [
            ("Total Timesteps:", "100000", "entry_steps"),
            ("Learning Rate:", "0.0001", "entry_lr"),
            ("Batch Size:", "32", "entry_batch"),
            ("Buffer Size:", "50000", "entry_buffer"),
            ("Gamma:", "0.99", "entry_gamma"),
            ("Exploration Fraction:", "0.5", "entry_expl"),
            ("Network Layers:", "2", "entry_num_layers"),
            ("Layer Size:", "64", "entry_layer_size"),
        ]

        for i, (label, default, attr_name) in enumerate(params):
            tk.Label(frame_hyper, text=label).grid(row=i, column=0, sticky="w")
            entry = tk.Entry(frame_hyper)
            entry.insert(0, default)
            entry.grid(row=i, column=1, padx=5, sticky="ew")
            setattr(self, attr_name, entry)

        frame_reward = tk.LabelFrame(self.top, text="Reward Weights", padx=10, pady=5)
        frame_reward.pack(fill="x", padx=10, pady=5)

        tk.Label(frame_reward, text="Time Weight:").grid(row=0, column=0)
        self.entry_w_time = tk.Entry(frame_reward, width=10)
        self.entry_w_time.insert(0, "1.0")
        self.entry_w_time.grid(row=0, column=1)

        tk.Label(frame_reward, text="CO2 Weight:").grid(row=0, column=2)
        self.entry_w_co2 = tk.Entry(frame_reward, width=10)
        self.entry_w_co2.insert(0, "1.0")
        self.entry_w_co2.grid(row=0, column=3)

        frame_btns = tk.Frame(self.top, pady=10)
        frame_btns.pack(fill="x")

        self.btn_start = tk.Button(frame_btns, text="Start Training", command=self.start_training, 
                                   bg="green", fg="white", font=("Arial", 10, "bold"))
        self.btn_start.pack(side="left", padx=20)

        self.btn_stop = tk.Button(frame_btns, text="Stop", command=self.stop_training, state="disabled",
                                  bg="red", fg="white")
        self.btn_stop.pack(side="left", padx=5)

        if HAS_EXPORT:
            self.btn_export = tk.Button(frame_btns, text="Export for Colab", command=self.export_config,
                                        bg="blue", fg="white")
            self.btn_export.pack(side="right", padx=20)

        self.txt_log = tk.Text(self.top, height=15, state="disabled", bg="#f0f0f0")
        self.txt_log.pack(fill="both", expand=True, padx=10, pady=10)

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
            "wandb_api_key": self.entry_apikey.get(),
            "learning_rate": float(self.entry_lr.get()),
            "batch_size": int(self.entry_batch.get()),
            "buffer_size": int(self.entry_buffer.get()),
            "gamma": float(self.entry_gamma.get()),
            "exploration_fraction": float(self.entry_expl.get()),
            "w_time": float(self.entry_w_time.get()),
            "w_co2": float(self.entry_w_co2.get()),
            "num_layers": int(self.entry_num_layers.get()),
            "layer_size": int(self.entry_layer_size.get()),
        }

    def start_training(self):
        if not HAS_RL_LIBS:
            messagebox.showerror("Hiba", "RL könyvtárak (SB3, WandB) hiányoznak!")
            return

        try:
            settings = self.get_settings()
            if settings["wandb_api_key"]:
                os.environ["WANDB_API_KEY"] = settings["wandb_api_key"]

            self.trainer_instance = IndependentDQNTrainer(
                net_file=self.net_file,
                logic_file=self.logic_file,
                detector_file=self.detector_file,
                total_timesteps=settings["total_timesteps"],
                wandb_project=settings["wandb_project"],
                log_queue=self.log_queue,
                hyperparams=settings,
                reward_weights={'time': settings["w_time"], 'co2': settings["w_co2"]}
            )

            self.trainer_thread = threading.Thread(target=self.run_trainer_thread)
            self.trainer_thread.daemon = True
            self.trainer_thread.start()

            self.btn_start.config(state="disabled")
            self.btn_stop.config(state="normal")
            
        except ValueError as e:
            messagebox.showerror("Input Error", f"Hibás paraméter: {e}")

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

    def export_config(self):
        if not HAS_EXPORT: return
        try:
            settings = self.get_settings()
            files = {
                "net": self.net_file,
                "logic": self.logic_file,
                "detector": self.detector_file
            }
            export_to_colab_package(settings, files)
        except Exception as e:
            messagebox.showerror("Export Hiba", str(e))