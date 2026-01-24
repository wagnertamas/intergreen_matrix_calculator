import tkinter as tk
from tkinter import ttk, messagebox
import gymnasium as gym
import numpy as np
import threading
import sys
import io
import queue
import time
import os
import shutil
from typing import Dict, Any, List
import torch

try:
    from stable_baselines3 import DQN
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.buffers import ReplayBuffer
    from stable_baselines3.common.logger import configure
    import wandb
    HAS_RL_LIBS = True
except ImportError as e:
    print(e)
    HAS_RL_LIBS = False

from sumo_rl_environment import SumoRLEnvironment


class IndependentDQNTrainer:
    """
    Manages training of multiple independent DQN agents.
    Performs the training loop manually to allow simultaneous stepping of the SUMO environment.
    """
    def __init__(self, 
                 net_file, 
                 logic_file, 
                 detector_file, 
                 total_timesteps=100000, 
                 wandb_project="sumo-rl",
                 wandb_api_key=None, 
                 log_queue=None,
                 hyperparams=None,
                 **env_kwargs):
        
        self.net_file = net_file
        self.logic_file = logic_file
        self.detector_file = detector_file
        self.total_timesteps = total_timesteps
        self.project_name = wandb_project
        self.log_queue = log_queue
        
        # Load hyperparams with defaults
        self.hyperparams = hyperparams or {}
        
        self.learning_rate = self.hyperparams.get("learning_rate", 1e-4)
        self.batch_size = self.hyperparams.get("batch_size", 32)
        self.buffer_size = self.hyperparams.get("buffer_size", 10000)
        self.gamma = self.hyperparams.get("gamma", 0.99)
        self.exploration_fraction = self.hyperparams.get("exploration_fraction", 0.1)
        self.train_freq = self.hyperparams.get("train_freq", 4)
        self.gradient_steps = self.hyperparams.get("gradient_steps", 1)
        
        self.env_kwargs = env_kwargs
        
        self.stop_requested = False
        
    def log(self, msg):
        if self.log_queue:
            self.log_queue.put(msg)
        else:
            print(msg)

    def run(self):
        self.log("Initializing SUMO Environment...")
        
        # Init WandB
        if wandb.run is None:
            config = {
                "policy": "IndependentDQN", 
                "total_timesteps": self.total_timesteps,
                "learning_rate": self.learning_rate,
                "batch_size": self.batch_size,
                "buffer_size": self.buffer_size,
                "gamma": self.gamma,
                "exploration_fraction": self.exploration_fraction
            }
            wandb.init(
                project=self.project_name, 
                config=config,
                sync_tensorboard=True
            )
            
            # [SWEEP SUPPORT]
            # Override local params with WandB config (if provided by Sweep Controller)
            # This allows the Hyperparameter Tuning to control these values.
            self.learning_rate = wandb.config.get("learning_rate", self.learning_rate)
            self.batch_size = wandb.config.get("batch_size", self.batch_size)
            self.buffer_size = wandb.config.get("buffer_size", self.buffer_size)
            self.gamma = wandb.config.get("gamma", self.gamma)
            self.exploration_fraction = wandb.config.get("exploration_fraction", self.exploration_fraction)
            
            self.log(f"Hyperparameters: LR={self.learning_rate}, Batch={self.batch_size}, Gamma={self.gamma}")
        
        # 1. Initialize Environment
        
        # [ABSOLUTE PATHS FIX]
        # Ensure all generated files (routes, models, logs) are in the SAME folder as the Network File.
        # This prevents files being scattered in the script directory.
        self.work_dir = os.path.dirname(os.path.abspath(self.net_file))
        
        # Define absolute path for the dynamic route file
        route_file_abs = os.path.join(self.work_dir, "random_traffic.rou.xml")
        
        self.env = SumoRLEnvironment(
            net_file=self.net_file,
            logic_json_file=self.logic_file,
            detector_file=self.detector_file,
            route_file=route_file_abs, # Force absolute path
            sumo_gui=False,
            min_green_time=5,
            delta_time=5,
            **self.env_kwargs
        )
        
        # RESET ENV FIRST to initialize SUMO and spaces!
        self.log("Resetting Environment to initialize spaces...")
        obs, infos = self.env.reset()
        
        from collections import deque
        
        # 2. Instantiate Agents (One DQN per Intersection)
        self.agents: Dict[str, DQN] = {}
        self.episode_reward_buffers = {}
        self.best_mean_rewards = {}
        
        # Early Stopping state
        self.es_enabled = self.hyperparams.get("early_stopping", {}).get("enabled", False)
        self.es_patience = self.hyperparams.get("early_stopping", {}).get("patience", 20)
        self.es_min_delta = self.hyperparams.get("early_stopping", {}).get("min_delta", 0.01)
        self.frozen_agents = set() # Set of JIDs that have converged

        # Network Architecture
        # Get defaults from hyperparams (set by GUI)
        num_layers = self.hyperparams.get("num_layers", 2)
        layer_size = self.hyperparams.get("layer_size", 64)
        
        # Override with WandB (if running Sweep)
        if wandb.run:
            num_layers = wandb.config.get("num_layers", num_layers)
            layer_size = wandb.config.get("layer_size", layer_size)
            self.log(f"Architecture Override from WandB: Layers={num_layers}, Size={layer_size}")
            
        net_arch = [layer_size] * num_layers
        policy_kwargs = dict(net_arch=net_arch)

        # Paths for Logs
        runs_dir = os.path.join(self.work_dir, "runs")

        for jid, sumo_agent in self.env.agents.items():
            self.log(f"Creating DQN Agent for {jid} (Net: {net_arch}, Layers: {num_layers}x{layer_size})...")
            
            # WORKAROUND: Create a simple dummy environment that matches the spaces of this specific agent
            # just for initialization.
            dummy_env = self._create_dummy_env_for_agent(sumo_agent)
            
            # Check if spaces are valid now
            if dummy_env.observation_space is None or dummy_env.action_space is None:
                raise ValueError(f"Agent {jid} has None spaces! Env reset failed to setup spaces.")
            
            # Constant LR (as user requested)
            
            # Determine Device (M3 Mac Support)
            if torch.backends.mps.is_available():
                device = "mps"
                self.log(f"Agent {jid}: Using Apple Silicon GPU optimization (MPS).")
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "auto" # CPU or other
            
            tb_log = os.path.join(runs_dir, self.project_name, jid)
            model = DQN(
                "MultiInputPolicy", 
                dummy_env, # Used only for getting spaces
                verbose=0,
                tensorboard_log=tb_log, # Relative to Work Dir
                buffer_size=self.buffer_size,
                learning_rate=self.learning_rate, # Constant LR
                batch_size=self.batch_size,
                gamma=self.gamma, # Initial Gamma
                exploration_fraction=self.exploration_fraction,
                # New: Network Size
                policy_kwargs=policy_kwargs,
                target_update_interval=1000,
                train_freq=self.train_freq,
                gradient_steps=self.gradient_steps,
                device=device
            )
            
            # Manually setup logger since we don't call learn()
            # SB3 requires a logger for train() and save()
            new_logger = configure(tb_log, ["stdout", "tensorboard"])
            model.set_logger(new_logger)
            
            self.agents[jid] = model
            # Window size for smoothing and early stopping
            self.episode_reward_buffers[jid] = deque(maxlen=self.es_patience)
            self.best_mean_rewards[jid] = -float('inf')
            
        # 3. Training Loop
        self.log(f"Starting Independent Training Loop (ES={self.es_enabled})...")
        
        # Define Save Directory based on WandB Run
        run_id = wandb.run.id if wandb.run else "manual"
        run_name = wandb.run.name if wandb.run else "local"
        
        # Models Dir relative to Work Dir
        self.save_base_dir = os.path.join(self.work_dir, "models", self.project_name, f"{run_name}_{run_id}")
        
        self.log(f"Models will be saved to: {self.save_base_dir}")
        self.log(f"Configuring Training in Working Directory: {self.work_dir}")
        
        # obs is already available from the reset above
        
        global_step = 0
        episode_rewards = {jid: 0.0 for jid in self.agents}
        # Component Accumulators for Episode
        episode_metrics = {jid: {"tt": 0.0, "co2": 0.0, "count": 0} for jid in self.agents}
        
        # Smoothed Reward Tracking
        smoothed_rewards = {jid: 0.0 for jid in self.agents}
        smoothing_alpha = 0.05
        
        episode_counts = 0
        
        start_time = time.time()
        
        while global_step < self.total_timesteps and not self.stop_requested:
            global_step += 1
            
            # Calculate Progress (1.0 -> 0.0)
            progress_remaining = 1.0 - (float(global_step) / float(self.total_timesteps))
            
            # [GAMMA DECAY SCHEDULE]
            # LINEAR DECAY to near ZERO.
            # Decays from self.gamma to 0.0 over the first 85% of training.
            # Stays at 0.0 for the last 15%.
            progress_done = 1.0 - progress_remaining
            decay_duration = 0.85 # Reach 0 at 85%
            
            if progress_done >= decay_duration:
                current_gamma = 0.01 # Floor value (near zero)
            else:
                 # Linear interpolation: Start -> 0
                 ratio = progress_done / decay_duration
                 current_gamma = self.gamma * (1.0 - ratio)
            
            current_gamma = max(0.01, current_gamma)
            
            # A. Collect Actions
            actions = {}
            for jid, model in self.agents.items():
                # Manually update tracking (needed for Exploration Decay inside SB3)
                model._current_progress_remaining = max(0.0, progress_remaining)
                
                # Update Gamma (Manual Schedule)
                model.gamma = current_gamma
                
                # model.predict returns (action, state)
                action, _ = model.predict(obs[jid], deterministic=False)
                actions[jid] = int(action)
                
            # B. Step Environment
            next_obs, rewards, terminated, truncated, infos = self.env.step(actions)
            
            sim_done = terminated.get("__all__", False) or truncated.get("__all__", False)
            
            # C. Store Transitions & Train
            for jid, model in self.agents.items():
                # Accumulate component metrics from info
                if jid in infos:
                    info = infos[jid]
                    if "metric_travel_time" in info:
                         episode_metrics[jid]["tt"] += info["metric_travel_time"]
                         episode_metrics[jid]["co2"] += info["metric_co2"]
                         episode_metrics[jid]["count"] += 1

                agent_done = terminated.get(jid, False) or truncated.get(jid, False)
                episode_rewards[jid] += rewards[jid]
                
                # Update Smoothed Reward
                prev_smooth = smoothed_rewards[jid]
                if prev_smooth == 0.0 and global_step < 100: # Initialize
                     smoothed_rewards[jid] = rewards[jid]
                else:
                     smoothed_rewards[jid] = smoothing_alpha * rewards[jid] + (1 - smoothing_alpha) * prev_smooth
                
                # If frozen, SKIP Replay Buffer and Training
                if jid in self.frozen_agents:
                    model.num_timesteps += 1 # Keep stepping tick for logging
                    continue
                
                # Add to replay buffer
                # SB3 ReplayBuffer.add() signature: (obs, next_obs, action, reward, done, infos)
                
                # Convert to batch size 1 and enforce types
                _obs = obs[jid]
                _next_obs = next_obs[jid]
                _action = np.array([actions[jid]])
                _reward = np.array([rewards[jid]], dtype=np.float32)
                _done = np.array([agent_done], dtype=np.float32)
                
                # Explicit Type Casting for Dict Obs
                def cast_obs(o_dict):
                    new_dict = {}
                    for k, v in o_dict.items():
                         if k == "phase":
                             # Phase is int (usually int32 or int64)
                             new_dict[k] = v.reshape(1, *v.shape).astype(np.int64)
                         else:
                             # Occupancy/Flow are float
                             new_dict[k] = v.reshape(1, *v.shape).astype(np.float32)
                    return new_dict
                
                obs_batched = cast_obs(_obs)
                next_obs_batched = cast_obs(_next_obs)
                
                try:
                    model.replay_buffer.add(
                        obs_batched,
                        next_obs_batched,
                        _action,
                        _reward,
                        _done,
                        [infos.get(jid, {})]
                    )
                except Exception as e:
                    print(f"[ERROR] ReplayBuffer Add failed for {jid}: {e}")
                    print(f"Obs: {obs_batched}")
                    print(f"NextObs: {next_obs_batched}")
                    raise e
                
                # Train
                if global_step > 100 and global_step % self.train_freq == 0:
                     model.train(gradient_steps=self.gradient_steps, batch_size=self.batch_size)
                     # SB3 Logger Dump szükségeltetik a Tensorboard íráshoz
                     if global_step % 100 == 0:
                         model.logger.record("time/total_timesteps", global_step, exclude="tensorboard")
                         model.logger.dump(step=global_step)
                     
                # Update Target Network
                if global_step % 1000 == 0:
                    try:
                        pass # Auto handled by SB3 usually if properly configured
                    except:
                        pass
                
                model.num_timesteps += 1

            # Update Obs
            obs = next_obs
            
            # Frequent Logging (e.g. every 100 steps)
            if global_step % 100 == 0:
                log_dict = {
                    "global_step": global_step,
                    "train/gamma": current_gamma  # Log current gamma
                }
                # Log Smoothed Reward
                for jid, r in smoothed_rewards.items():
                     log_dict[f"reward_smooth/{jid}"] = r
                     
                # Optional: Log running reward too if desired
                # for jid, r in episode_rewards.items():
                #      log_dict[f"running_reward/{jid}"] = r
                     
                wandb.log(log_dict)
            
            # Handle Episode End
            if sim_done:
                episode_counts += 1
                avg_rew = sum(episode_rewards.values()) / len(episode_rewards)
                duration = time.time() - start_time
                fps = global_step / duration
                
                # Check Convergence / Early Stopping
                active_agents = len(self.agents) - len(self.frozen_agents)
                msg = f"Ep: {episode_counts} | AvgReward: {avg_rew:.2f} | Active Agents: {active_agents}"
                self.log(msg)
                
                # [ARCHIVE ROUTE FILE]
                # Save the route file used for this specific episode
                if hasattr(self.env, 'route_file') and os.path.exists(self.env.route_file):
                    routes_dir = os.path.join(self.save_base_dir, "routes")
                    os.makedirs(routes_dir, exist_ok=True)
                    route_dst = os.path.join(routes_dir, f"episode_{episode_counts}.rou.xml")
                    try:
                        shutil.copy(self.env.route_file, route_dst)
                    except Exception as e:
                        print(f"Failed to archive route file: {e}")
                
                # WandB Log
                log_dict = {"global_step": global_step, "episode_count": episode_counts, "avg_reward": avg_rew, "fps": fps}
                for jid, r in episode_rewards.items():
                    log_dict[f"reward/{jid}"] = r
                    
                    # Log Component Metrics
                    m = episode_metrics[jid]
                    if m["count"] > 0:
                        log_dict[f"metric/avg_travel_time_sec/{jid}"] = m["tt"] / m["count"]
                        log_dict[f"metric/avg_co2_kg/{jid}"] = m["co2"] / m["count"]
                    
                    # Update buffers
                    self.episode_reward_buffers[jid].append(r)
                    
                    # Check for Best Model
                    if len(self.episode_reward_buffers[jid]) > 0:
                        mean_reward = np.mean(self.episode_reward_buffers[jid])
                        log_dict[f"mean_reward/{jid}"] = mean_reward
                        
                        if mean_reward > self.best_mean_rewards[jid]:
                            self.best_mean_rewards[jid] = mean_reward
                if episode_counts > 0 and episode_counts % 30 == 0:
                    self.log(f"Auto-saving models at Episode {episode_counts}...")
                    save_dir = os.path.join(self.save_base_dir, f"ep_{episode_counts}")
                    os.makedirs(save_dir, exist_ok=True)
                    
                    for jid, model in self.agents.items():
                        # Helyi mentés
                        path = os.path.join(save_dir, f"dqn_agent_{jid}")
                        model.save(path)
                        # WandB Feltöltés
                        wandb.save(path + ".zip", base_path=save_dir)
                    
                    # Save Route File
                    if hasattr(self.env, 'route_file') and os.path.exists(self.env.route_file):
                        route_dst = os.path.join(save_dir, "routes.rou.xml")
                        shutil.copy(self.env.route_file, route_dst)
                        wandb.save(route_dst, base_path=save_dir)
                    
                    self.log(f"Models and Routes uploaded to WandB.")
                
                # Reset
                obs, infos = self.env.reset()
                episode_rewards = {jid: 0.0 for jid in self.agents}
                
        # Save Models (Final)
        self.log("Saving Final Agents...")
        final_dir = os.path.join(self.save_base_dir, "final")
        os.makedirs(final_dir, exist_ok=True)
        for jid, model in self.agents.items():
            path = os.path.join(final_dir, f"dqn_agent_{jid}")
            model.save(path)
            wandb.save(path + ".zip", base_path=final_dir)
        
        # Save Route File (Final)
        if hasattr(self.env, 'route_file') and os.path.exists(self.env.route_file):
            route_dst = os.path.join(final_dir, "routes.rou.xml")
            shutil.copy(self.env.route_file, route_dst)
            wandb.save(route_dst, base_path=final_dir)
        
        self.env.close()
        wandb.finish()
        self.log("Training Finished.")
        
    def _create_dummy_env_for_agent(self, sumo_agent):
        """Creates a dummy Gym environment matching the agent's spaces."""
        
        # Capture spaces in closure
        obs_space = sumo_agent.observation_space
        act_space = sumo_agent.action_space
        
        class DummyEnv(gym.Env):
            def __init__(self):
                self.observation_space = obs_space
                self.action_space = act_space
            def reset(self, seed=None, options=None):
                # Return dummy observation
                return self.observation_space.sample(), {}
            def step(self, action):
                return self.observation_space.sample(), 0.0, False, False, {}
                
        return DummyEnv()


class TrainingDialog(tk.Toplevel):
    def __init__(self, parent, net_file, logic_file, detector_file):
        super().__init__(parent)
        self.title("RL Training - Independent DQN")
        self.geometry("700x600")
        
        self.net_file = net_file
        self.logic_file = logic_file
        self.detector_file = detector_file
        
        self.training_thread = None
        self.trainer = None
        self.log_queue = queue.Queue()
        
        if not HAS_RL_LIBS:
            tk.Label(self, text="HIBA: Stable Baselines 3 vagy WandB hiányzik!", fg="red").pack(pady=20)
            return
            
        self._setup_ui()
        self._start_log_updater()

    def _setup_ui(self):
        # Settings Frame
        frame_sett = tk.LabelFrame(self, text="Settings", padx=10, pady=10)
        frame_sett.pack(fill="x", padx=10, pady=5)
        
        tk.Label(frame_sett, text="Total Timesteps:").grid(row=0, column=0, sticky="w")
        self.entry_steps = tk.Entry(frame_sett)
        self.entry_steps.insert(0, "100000")
        self.entry_steps.grid(row=0, column=1)
        
        tk.Label(frame_sett, text="WandB Project:").grid(row=1, column=0, sticky="w")
        self.entry_project = tk.Entry(frame_sett)
        self.entry_project.insert(0, "sumo-rl-independent")
        self.entry_project.grid(row=1, column=1)
        
        tk.Label(frame_sett, text="WandB API Key (optional):").grid(row=2, column=0, sticky="w")
        self.entry_apikey = tk.Entry(frame_sett, show="*")
        self.entry_apikey.grid(row=2, column=1)

        # Hyperparameters
        tk.Label(frame_sett, text="Learning Rate:").grid(row=3, column=0, sticky="w")
        self.entry_lr = tk.Entry(frame_sett)
        self.entry_lr.insert(0, "0.0001")
        self.entry_lr.grid(row=3, column=1)

        tk.Label(frame_sett, text="Batch Size (WandB: 32-256):").grid(row=4, column=0, sticky="w")
        self.entry_batch = tk.Entry(frame_sett)
        self.entry_batch.insert(0, "32")
        self.entry_batch.grid(row=4, column=1)
        
        tk.Label(frame_sett, text="Buffer Size:").grid(row=5, column=0, sticky="w")
        self.entry_buffer = tk.Entry(frame_sett)
        self.entry_buffer.insert(0, "10000")
        self.entry_buffer.grid(row=5, column=1)

        tk.Label(frame_sett, text="Gamma (WandB: 0.9-0.999):").grid(row=6, column=0, sticky="w")
        self.entry_gamma = tk.Entry(frame_sett)
        self.entry_gamma.insert(0, "0.99")
        self.entry_gamma.grid(row=6, column=1)

        tk.Label(frame_sett, text="Parallel Envs (Exper.):").grid(row=7, column=0, sticky="w")
        self.entry_n_envs = tk.Entry(frame_sett)
        self.entry_n_envs.insert(0, "1")
        self.entry_n_envs.grid(row=7, column=1)

        tk.Label(frame_sett, text="Exploration Fraction (Duration):").grid(row=8, column=0, sticky="w")
        self.entry_expl = tk.Entry(frame_sett)
        self.entry_expl.insert(0, "0.85")
        self.entry_expl.grid(row=8, column=1)

        # Performance Tuning (New)
        tk.Label(frame_sett, text="Train Freq (Higher=Faster Sim):").grid(row=9, column=0, sticky="w")
        frame_perf = tk.Frame(frame_sett)
        frame_perf.grid(row=9, column=1, sticky="w")
        
        self.entry_train_freq = tk.Entry(frame_perf, width=5)
        self.entry_train_freq.insert(0, "4")
        self.entry_train_freq.pack(side="left")
        
        tk.Label(frame_perf, text="Grad Steps:").pack(side="left", padx=5)
        self.entry_grad_steps = tk.Entry(frame_perf, width=5)
        self.entry_grad_steps.insert(0, "1")
        self.entry_grad_steps.pack(side="left")

        # Advanced Settings
        tk.Label(frame_sett, text="NN Architecture:").grid(row=10, column=0, sticky="w")
        frame_nn = tk.Frame(frame_sett)
        frame_nn.grid(row=10, column=1, sticky="w")
        
        tk.Label(frame_nn, text="Layers (1-3):").pack(side="left")
        self.entry_num_layers = tk.Entry(frame_nn, width=5)
        self.entry_num_layers.insert(0, "2")
        self.entry_num_layers.pack(side="left")
        
        tk.Label(frame_nn, text="Size (64-256):").pack(side="left", padx=5)
        self.entry_layer_size = tk.Entry(frame_nn, width=5)
        self.entry_layer_size.insert(0, "64")
        self.entry_layer_size.pack(side="left")

        tk.Label(frame_sett, text="Reward Weights (Time, CO2):").grid(row=11, column=0, sticky="w")
        frame_rw = tk.Frame(frame_sett)
        frame_rw.grid(row=11, column=1, sticky="w")
        self.entry_w_time = tk.Entry(frame_rw, width=5)
        self.entry_w_time.insert(0, "1.0")
        self.entry_w_time.pack(side="left")
        self.entry_w_co2 = tk.Entry(frame_rw, width=5)
        self.entry_w_co2.insert(0, "0.1")
        self.entry_w_co2.pack(side="left", padx=5)

        # Early Stopping
        self.var_early_stop = tk.BooleanVar(value=False)
        self.chk_early_stop = tk.Checkbutton(frame_sett, text="Enable Early Stopping", variable=self.var_early_stop)
        self.chk_early_stop.grid(row=12, column=0, sticky="w")

        frame_es = tk.Frame(frame_sett)
        frame_es.grid(row=12, column=1, sticky="w")
        tk.Label(frame_es, text="Patience:").pack(side="left")
        self.entry_patience = tk.Entry(frame_es, width=5)
        self.entry_patience.insert(0, "20")
        self.entry_patience.pack(side="left")
        tk.Label(frame_es, text="Min Delta:").pack(side="left", padx=5)
        self.entry_min_delta = tk.Entry(frame_es, width=5)
        self.entry_min_delta.insert(0, "0.01")
        self.entry_min_delta.pack(side="left")
        
        # Infinite Training
        self.var_infinite = tk.BooleanVar(value=False)
        self.chk_infinite = tk.Checkbutton(frame_sett, text="Infinite Training (Auto-Restart)", variable=self.var_infinite, fg="blue")
        self.chk_infinite.grid(row=13, column=0, columnspan=2, sticky="w", pady=5)


        # Buttons
        frame_btns = tk.Frame(self)
        frame_btns.pack(pady=10)
        
        self.btn_start = tk.Button(frame_btns, text="Start Training", command=self.start_training, bg="green", fg="black", font=("Arial", 12, "bold"))
        self.btn_start.pack(side="left", padx=10)
        
        self.btn_stop = tk.Button(frame_btns, text="Stop", command=self.stop_training, state="disabled", bg="red", fg="black")
        self.btn_stop.pack(side="left", padx=10)
        
        # Log Area
        self.txt_log = tk.Text(self, height=15)
        self.txt_log.pack(fill="both", expand=True, padx=10, pady=5)
        
    def _start_log_updater(self):
        try:
            while True:
                msg = self.log_queue.get_nowait()
                self.txt_log.insert(tk.END, str(msg) + "\n")
                self.txt_log.see(tk.END)
        except queue.Empty:
            pass
        self.after(100, self._start_log_updater)

    def start_training(self):
        project = self.entry_project.get()
        api_key = self.entry_apikey.get()
        
        # hyperparams
        try:
            total_timesteps = int(self.entry_steps.get())
            lr = float(self.entry_lr.get())
            batch_size = int(self.entry_batch.get())
            buffer_size = int(self.entry_buffer.get())
            gamma = float(self.entry_gamma.get())
            n_envs_requested = int(self.entry_n_envs.get())
            expl = float(self.entry_expl.get())
            
            # New Params
            w_time = float(self.entry_w_time.get())
            w_co2 = float(self.entry_w_co2.get())
            
            enable_es = self.var_early_stop.get()
            es_patience = int(self.entry_patience.get())
            es_min_delta = float(self.entry_min_delta.get())
            
            num_layers = int(self.entry_num_layers.get())
            layer_size = int(self.entry_layer_size.get())
            
            # New Performance Tuning
            train_freq = int(self.entry_train_freq.get())
            gradient_steps = int(self.entry_grad_steps.get())
            
            if n_envs_requested > 1:
                tk.messagebox.showwarning("Figyelem", "A Párhuzamos Tanítás (Parallel Training) backend implementációja még folyamatban van.\nJelenleg 1 környezettel fut.")
                n_envs_requested = 1
            
        except ValueError as e:
            tk.messagebox.showerror("Hiba", f"Érvénytelen paraméter: {e}")
            return
            
        self.btn_start.config(state="disabled")
        self.btn_stop.config(state="normal")
        self.entry_steps.config(state="disabled")
        
        def run_thread():
            if api_key:
                os.environ["WANDB_API_KEY"] = api_key
            
            while True:
                self.trainer = IndependentDQNTrainer(
                    net_file=self.net_file,
                    logic_file=self.logic_file,
                    detector_file=self.detector_file,
                    wandb_project=project,
                    total_timesteps=total_timesteps,
                    log_queue=self.log_queue,
                    hyperparams={
                        "learning_rate": lr,
                        "batch_size": batch_size,
                        "buffer_size": buffer_size,
                        "gamma": gamma,
                        "exploration_fraction": expl,
                        "es_enabled": enable_es,
                        "es_patience": es_patience,
                        "es_min_delta": es_min_delta,
                        "num_layers": num_layers,  
                        "layer_size": layer_size,
                        "train_freq": train_freq,
                        "gradient_steps": gradient_steps
                    },
                    reward_weights={'time': w_time, 'co2': w_co2},
                )
                
                try:
                    self.trainer.run()
                except Exception as e:
                    self.log_queue.put(f"ERROR: {e}")
                    import traceback
                    traceback.print_exc()
                
                # Check for infinite loop
                if not self.var_infinite.get():
                    break
                    
                if self.trainer.stop_requested: # If user manually clicked Stop button
                    break
                    
                self.log_queue.put("--- Infinite Mode: Restarting Training in 5 seconds... ---")
                time.sleep(5)

            # Cleanup
            self.btn_start.config(state="normal")
            self.btn_stop.config(state="disabled")
            self.entry_steps.config(state="normal")
            
        self.training_thread = threading.Thread(target=run_thread, daemon=True)
        self.training_thread.start()

    def stop_training(self):
        if self.trainer:
            self.trainer.stop_requested = True
        self.log_queue.put("Stopping requested...")
        self.btn_stop.config(state="disabled")
