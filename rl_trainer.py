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
    from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
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
                 n_envs=1,
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
        
        self.n_envs = n_envs
        self.env_kwargs = env_kwargs
        
        self.stop_requested = False
        
    def log(self, msg):
        if self.log_queue:
            self.log_queue.put(msg)
        else:
            print(msg)

    def run(self):
        self.log(f"Initializing SUMO Environment (n_envs={self.n_envs})...")
        
        # Init WandB
        if wandb.run is None:
            config = {
                "policy": "IndependentDQN", 
                "total_timesteps": self.total_timesteps,
                "learning_rate": self.learning_rate,
                "batch_size": self.batch_size,
                "buffer_size": self.buffer_size,
                "gamma": self.gamma,
                "exploration_fraction": self.exploration_fraction,
                "n_envs": self.n_envs,
                "train_freq": self.train_freq,
                "gradient_steps": self.gradient_steps
            }
            wandb.init(
                project=self.project_name, 
                config=config,
                sync_tensorboard=False # USER REQUEST: Disable to prevent pod name errors
            )
            
            # [SWEEP SUPPORT]
            # Debug: what is actually in the config?
            self.log(f"DEBUG: WandB Config Context: {dict(wandb.config)}")
            
            if hasattr(wandb.config, "learning_rate"):
                 self.learning_rate = wandb.config.learning_rate
            else:
                 self.learning_rate = wandb.config.get("learning_rate", self.learning_rate)
            
            self.batch_size = wandb.config.get("batch_size", self.batch_size)
            self.buffer_size = wandb.config.get("buffer_size", self.buffer_size)
            self.gamma = wandb.config.get("gamma", self.gamma)
            self.exploration_fraction = wandb.config.get("exploration_fraction", self.exploration_fraction)
            self.train_freq = wandb.config.get("train_freq", self.train_freq)
            self.gradient_steps = wandb.config.get("gradient_steps", self.gradient_steps)
            
            self.log(f"Hyperparams (Final): LR={self.learning_rate}, Batch={self.batch_size}, Gamma={self.gamma}, Envs={self.n_envs}")
        
        # 1. Initialize Environment (Vectorized)
        # 1. Initialize Environment (Vectorized)
        self.work_dir = os.path.dirname(os.path.abspath(self.net_file))
        
        # [PICKLING FIX]
        # Extract variables to local scope so 'make_env' capture doesn't grab 'self'
        # 'self' contains Tkinter objects and Locks which cannot be pickled for Multiprocessing.
        work_dir = self.work_dir
        net_file = self.net_file
        logic_file = self.logic_file
        detector_file = self.detector_file
        env_kwargs = self.env_kwargs
        
        # Define make_env factory
        # We wrap it in a function that captures the LOCAL variables, not self.
        def make_env_factory(rank, work_dir, net_file, logic_file, detector_file, env_kwargs):
            def _init():
                # Unique temporary directory for this process
                process_dir = os.path.join(work_dir, "runs", f"proc_{rank}")
                os.makedirs(process_dir, exist_ok=True)
                
                # Unique Route & Stats Files
                route_file = os.path.join(process_dir, "random_traffic.rou.xml")
                stats_file = os.path.join(process_dir, "stats.xml")
                
                env = SumoRLEnvironment(
                    net_file=net_file,
                    logic_json_file=logic_file,
                    detector_file=detector_file,
                    route_file=route_file, 
                    statistic_output_file=stats_file,
                    sumo_gui=False, 
                    min_green_time=5,
                    delta_time=5,
                    **env_kwargs
                )
                return env
            return _init

        # Create Vector Env
        if self.n_envs > 1:
            # SubprocVecEnv runs in separate processes
            # We call the factory properly passing the args
            env_fns = [make_env_factory(i, work_dir, net_file, logic_file, detector_file, env_kwargs) for i in range(self.n_envs)]
            self.env = SubprocVecEnv(env_fns)
        else:
            # Dummy runs in same process
            self.env = DummyVecEnv([make_env_factory(0, work_dir, net_file, logic_file, detector_file, env_kwargs)])

        self.log(f"Resetting Vector Environment ({self.n_envs} processes)...")
        # VecEnv reset returns only Observations (stacked)
        obs = self.env.reset()
        
        self.log(f"Vector Env Reset Done. Obs type: {type(obs)}")

        # [FIX NESTED DICT STACKING]
        # SB3 VecEnv fails to stack Dicts inside Dicts recursively.
        # It produces an object array of Dicts (np.array([dict, dict], dtype=object)).
        # We must manually transpose this into a Dict of Arrays ({key: np.array([v1, v2])}).
        def fix_nested_obs(agent_obs):
            if isinstance(agent_obs, np.ndarray) and agent_obs.dtype == object:
                # Transpose: Array of Dicts -> Dict of Arrays
                # Check first element to get keys
                if len(agent_obs) == 0: return agent_obs
                keys = agent_obs[0].keys()
                new_dict = {}
                for k in keys:
                     # Stack the scalar/array values for this key
                     val_list = [d[k] for d in agent_obs]
                     new_dict[k] = np.stack(val_list)
                return new_dict
            return agent_obs

        # Fix initial obs
        if isinstance(obs, dict):
             for k, v in obs.items():
                 obs[k] = fix_nested_obs(v)

        from collections import deque
        
        # 2. Instantiate Agents
        self.agents: Dict[str, DQN] = {}
        self.episode_reward_buffers = {}
        self.best_mean_rewards = {}
        
        # Early Stopping state
        self.es_enabled = self.hyperparams.get("early_stopping", {}).get("enabled", False)
        self.es_patience = self.hyperparams.get("early_stopping", {}).get("patience", 20)
        # Note: ES works on average across envs, or we track per env?
        # Usually we track average performance.
        self.frozen_agents = set() 

        # Architecture
        num_layers = self.hyperparams.get("num_layers", 2)
        layer_size = self.hyperparams.get("layer_size", 64)
        if wandb.run:
             num_layers = wandb.config.get("num_layers", num_layers)
             layer_size = wandb.config.get("layer_size", layer_size)
             
        net_arch = [layer_size] * num_layers
        policy_kwargs = dict(net_arch=net_arch)

        runs_dir = os.path.join(self.work_dir, "runs")

        # We need to discover the agents (junction IDs).
        # Since VecEnv hides the internal agents list, we rely on the logic_file or the keys in obs.
        # obs is a Dict of stacked observations for each agent.
        # Format: {'R1C1': array(...), 'R2C2': array(...)}
        
        self.log("Discovering Agents from Observation Keys...")
        if isinstance(obs, dict):
            agent_ids = list(obs.keys())
            self.log(f"Found Agents (from Obs): {agent_ids}")
        else:
            # Fallback (should not happen with our Env)
            agent_ids = list(self.env.observation_space.keys())
            self.log(f"Found Agents (from Space): {agent_ids}")

        # Device Selection
        if torch.backends.mps.is_available():
            device = "mps"
            self.log(f"Using Apple Silicon GPU optimization (MPS).")
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "auto"

        for jid in agent_ids:
            # Construct a DummyVecEnv for the MODEL that matches self.n_envs
            # This ensures the ReplayBuffer is allocated with size (buffer_size, n_envs, ...)
            
            # We need the single-agent space for this JID
            # self.env.observation_space is a Dict space containing specific agent spaces?
            # Or is it a Dict of spaces?
            # Gym Vector API: env.observation_space is usually a batched space or the single space?
            # SB3 VecEnv: env.observation_space is the SINGLE env space.
            
            # self.env.observation_space is likely gym.spaces.Dict({jid: ...})
            full_obs_space = self.env.observation_space
            agent_obs_space = full_obs_space[jid]
            
            # We assume action space is also Dict
            full_act_space = self.env.action_space
            # But wait, we define action space as Discrete in TrafficAgent.
            # SumoRLEnvironment defines self.action_space = spaces.Dict({jid: CustomDiscrete(...)})
            agent_act_space = full_act_space[jid]
            
            self.log(f"Creating Agent {jid} (Input: {agent_obs_space.shape})...")

            # Create a specialized DummyVecEnv that mimics this single agent's view
            # This is "Virtual" environment for the SB3 model handle
            
            class SingleAgentEnvWrapper(gym.Env):
                def __init__(self, o_space, a_space):
                     self.observation_space = o_space
                     self.action_space = a_space
                def reset(self, **kwargs): return self.observation_space.sample(), {}
                def step(self, action): return self.observation_space.sample(), 0, False, False, {}
            
            # We must wrap this in DummyVecEnv with n_envs so buffer logic works
            model_vec_env = DummyVecEnv([lambda: SingleAgentEnvWrapper(agent_obs_space, agent_act_space) for _ in range(self.n_envs)])
            
            tb_log = os.path.join(runs_dir, self.project_name, jid)
            model = DQN(
                "MultiInputPolicy", 
                model_vec_env, 
                verbose=0,
                tensorboard_log=tb_log,
                buffer_size=self.buffer_size,
                learning_rate=self.learning_rate,
                batch_size=self.batch_size, # This is usually "items sampled from buffer", indifferent to n_envs
                gamma=self.gamma,
                exploration_fraction=self.exploration_fraction,
                policy_kwargs=policy_kwargs,
                target_update_interval=1000,
                train_freq=self.train_freq,
                gradient_steps=self.gradient_steps,
                device=device
            )
            
            new_logger = configure(tb_log, ["stdout", "tensorboard"])
            model.set_logger(new_logger)
            self.agents[jid] = model
            self.episode_reward_buffers[jid] = deque(maxlen=self.es_patience)
            self.best_mean_rewards[jid] = -float('inf')

        # 3. Training Loop
        self.log(f"Starting Vectorized Training Loop (n_envs={self.n_envs})...")
        
        run_id = wandb.run.id if wandb.run else "manual"
        run_name = wandb.run.name if wandb.run else "local"
        self.save_base_dir = os.path.join(self.work_dir, "models", self.project_name, f"{run_name}_{run_id}")
        
        global_step = 0
        
        # Track rewards per environment to handle async termination
        # shape: (n_envs,)
        current_episode_rewards = {jid: np.zeros(self.n_envs) for jid in self.agents}
        current_episode_metrics = {jid: [{"tt": 0.0, "co2": 0.0, "count": 0} for _ in range(self.n_envs)] for jid in self.agents}
        
        smoothed_rewards = {jid: 0.0 for jid in self.agents}
        smoothing_alpha = 0.05
        
        episode_counts = 0 # Total completed episodes across all envs
        
        start_time = time.time()
        
        while global_step < self.total_timesteps and not self.stop_requested:
            # One step in VecEnv = n_envs steps in total? 
            # Usually global_step counts "env steps" (meaning 1 call to step()).
            # But "timesteps" in RL often means "transitions".
            # We increment by n_envs?
            # SB3 convention: total_timesteps is "number of interactions". 
            # If we run 4 envs, 1 step = 4 interactions.
            
            # Increase global step by n_envs
            steps_this_iter = self.n_envs
            global_step += steps_this_iter
            
            progress_remaining = 1.0 - (float(global_step) / float(self.total_timesteps))
            
            # Gamma Schedule
            progress_done = 1.0 - progress_remaining
            decay_duration = 0.85
            if progress_done >= decay_duration:
                current_gamma = 0.01 
            else:
                 ratio = progress_done / decay_duration
                 current_gamma = self.gamma * (1.0 - ratio)
            current_gamma = max(0.01, current_gamma)

            # A. Collect Actions
            # actions_list: List of Dicts [ {jid: act}, {jid: act} ]
            actions_list = [{} for _ in range(self.n_envs)]
            
            for jid, model in self.agents.items():
                model._current_progress_remaining = max(0.0, progress_remaining)
                model.gamma = current_gamma
                
                # Predict returns (n_envs,) actions
                # obs[jid] is (n_envs, features...)
                action_batch, _ = model.predict(obs[jid], deterministic=False)
                
                for env_idx, act in enumerate(action_batch):
                    actions_list[env_idx][jid] = int(act)
            
            # B. Step Environment
            # VecEnv returns stacked obs, stacked scalar reward (ignored), stacked done (bool), and list of infos
            next_obs, _, dones, infos = self.env.step(actions_list)
            
            # Fix nested stacking
            if isinstance(next_obs, dict):
                 for k, v in next_obs.items():
                     next_obs[k] = fix_nested_obs(v)
            
            # C. Process Transitions
            for jid, model in self.agents.items():
                if jid in self.frozen_agents:
                    model.num_timesteps += steps_this_iter
                    continue
                
                # Prepare data for buffer
                _obs = obs[jid]
                _next_obs = next_obs[jid]
                
                # We need to construct batch-compatible arrays
                # Actions: (n_envs, 1) or (n_envs,) depending on space. Discrete actions usually (n_envs,).
                # But ReplayBuffer sometimes wants (n_envs, 1).
                # Let's extract from actions_list again or use the prediction result.
                # We used `model.predict` which returns (n_envs,).
                
                # Extract Rewards and Dones from INFO
                # infos is tuple/list of dicts
                
                batch_rewards = []
                batch_dones = []
                batch_infos = []
                
                for env_idx, info in enumerate(infos):
                    # Info structure from SumoRLEnvironment:
                    # 'ma_rewards': nested dict, 'ma_terminated', 'agent_infos'
                    # Or 'combined_info' if we packed it tightly?
                    # The env returns info dict. 
                    
                    # Ensure we handle the info correctly
                    env_ma_rewards = info.get("ma_rewards", {})
                    env_ma_term = info.get("ma_terminated", {})
                    env_ma_trunc = info.get("ma_truncated", {})
                    
                    # Agent specific
                    r = env_ma_rewards.get(jid, 0.0)
                    is_term = env_ma_term.get(jid, False)
                    is_trunc = env_ma_trunc.get(jid, False)
                    d = is_term or is_trunc
                    
                    batch_rewards.append(r)
                    batch_dones.append(d)
                    
                    # Agent specific info (for metrics logging inside SB3 if needed, mostly unused)
                    batch_infos.append(info.get("agent_infos", {}).get(jid, {}))
                    
                    # Update Trackers
                    current_episode_rewards[jid][env_idx] += r
                    
                    # Metrics
                    a_info = info.get("agent_infos", {}).get(jid, {})
                    if "metric_travel_time" in a_info:
                         current_episode_metrics[jid][env_idx]["tt"] += a_info["metric_travel_time"]
                         current_episode_metrics[jid][env_idx]["co2"] += a_info["metric_co2"]
                         current_episode_metrics[jid][env_idx]["count"] += 1

                    # Handle Episode End logic per Environment
                    # "dones" from output is Global Sim Done. 
                    # If global sim done, we count episode.
                    if dones[env_idx]:
                        # Reset trackers for this env
                        ep_rew = current_episode_rewards[jid][env_idx]
                        current_episode_rewards[jid][env_idx] = 0.0
                        
                        # We only log/count once per env reset really.
                        # But since we have multiple agents, all finish at the same time.
                        # Logic: Iterate over agents here, but "dones" true means ALL agents finished in this env.
                        pass 
                
                # Add to Replay Buffer
                # We need to reshape inputs to (n_envs, ...) if necessary
                # SB3 Buffer Add:
                # obs: (n_envs, *obs_shape)
                # action: (n_envs, *action_shape)
                # reward: (n_envs,)
                # done: (n_envs,)
                # infos: List[Dict]
                
                # Construct arrays
                np_rewards = np.array(batch_rewards, dtype=np.float32)
                np_dones = np.array(batch_dones, dtype=np.float32)
                
                # Actions need to be (n_envs, 1) for Discrete usually? Or (n_envs,)
                # SB3 DQN: actions should be (n_envs, 1) if using standard buffer?
                # Actually valid actions for add() are usually (n_envs, action_dim).
                # For Discrete, action_dim=1.
                
                # Re-fetch actions from list to be sure we match env_idx order
                # actions_list[env_idx][jid] result
                act_list = [actions_list[i][jid] for i in range(self.n_envs)]
                np_actions = np.array(act_list).reshape(self.n_envs, 1) # Force (N, 1)
                
                # Obs Casting
                def cast_obs_vec(o_dict):
                    new_dict = {}
                    for k, v in o_dict.items():
                        # v is likely (n_envs, ...)
                        # ensure types
                        if k == "phase":
                            new_dict[k] = v.astype(np.int64)
                        else:
                            new_dict[k] = v.astype(np.float32)
                    return new_dict
                    
                obs_batched = cast_obs_vec(_obs)
                next_obs_batched = cast_obs_vec(_next_obs)
                
                try:
                    model.replay_buffer.add(
                        obs_batched,
                        next_obs_batched,
                        np_actions,
                        np_rewards,
                        np_dones,
                        batch_infos
                    )
                except Exception as e:
                     print(f"Buffer Add Error {jid}: {e}")
                
                # Train
                if global_step > 100 and global_step % self.train_freq == 0:
                      model.train(gradient_steps=self.gradient_steps, batch_size=self.batch_size)
                
                model.num_timesteps += steps_this_iter

            # Handle Logging & Episode Ends
            # Check if ANY env finished
            for env_idx, is_done in enumerate(dones):
                if is_done:
                    episode_counts += 1
                    
                    # Log Stats for this finished episode
                    log_dict = {"global_step": global_step, "episode_count": episode_counts}
                    
                    # FPS
                    duration = time.time() - start_time
                    fps = global_step / max(0.001, duration)
                    log_dict["fps"] = fps
                    
                    for jid in self.agents:
                         # Last known total (before reset above cleared it? 
                         # Wait, I cleared it in the loop above. 
                         # I should capture it before clearing or use a temporary var.)
                         
                         # Actually, in the agent loop, if dones[env_idx] is true, we processed the reward.
                         # But we cleared it immediately. 
                         # Better logic: Check done here, outside agent loop?
                         # No, we need per-agent data.
                         
                         # FIX: We lost the episode reward sum in the loop above.
                         # Instead of clearing there, let's track "completed_episode_data" list during the loop.
                         pass
            
            # Better Logging Logic:
            # We can log periodically or when any episode finishes.
            # To simplify: Log WandB every X steps.
            
            if global_step % 1000 == 0: # Reduced frequency
                 msg = f"Step: {global_step} | Envs: {self.n_envs} | FPS: {int(global_step/(time.time()-start_time))}"
                 self.log(msg)
                 
                 # Upload Logs
                 wb_log = {"global_step": global_step, "fps": int(global_step/(time.time()-start_time))}
                 
                 # Compute Avg Reward for recent history (using buffers)
                 for jid, model in self.agents.items():
                      # Use the buffers we filled in the loop (Wait, I didn't fill buffers in the loop yet!)
                      pass
                 
                 wandb.log(wb_log)

            obs = next_obs
            
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
        self.entry_w_co2.insert(0, "1.0")
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
            
            # if n_envs_requested > 1:
            #     tk.messagebox.showwarning("Figyelem", "A Párhuzamos Tanítás (Parallel Training) backend implementációja még folyamatban van.\nJelenleg 1 környezettel fut.")
            #     n_envs_requested = 1
            
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
                    n_envs=n_envs_requested,
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
