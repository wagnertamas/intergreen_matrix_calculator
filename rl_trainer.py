import gymnasium as gym
import numpy as np
import threading
import queue
import time
import os
import sys
import json

# --- JAVÍTÁS: Feltételes GUI import ---
try:
    import tkinter as tk
    from tkinter import ttk, messagebox
    HAS_GUI = True
except ImportError:
    HAS_GUI = False
    # Headless módban (pl. Docker) ez ne okozzon hibát,
    # amíg nem próbáljuk megnyitni az ablakot.
# ---------------------------------------

# Opcionális importok ellenőrzése — egyenként, hogy lássuk melyik hiányzik
HAS_RL_LIBS = True
_missing_libs = []
try:
    from stable_baselines3 import DQN
    from stable_baselines3.common.logger import configure
    from stable_baselines3.common.vec_env import DummyVecEnv
except ImportError as e:
    HAS_RL_LIBS = False
    _missing_libs.append(f"stable-baselines3: {e}")
try:
    from sb3_contrib import QRDQN
except ImportError as e:
    HAS_RL_LIBS = False
    _missing_libs.append(f"sb3-contrib: {e}")
try:
    import wandb
except ImportError as e:
    HAS_RL_LIBS = False
    _missing_libs.append(f"wandb: {e}")
try:
    import torch
except ImportError as e:
    HAS_RL_LIBS = False
    _missing_libs.append(f"torch: {e}")

if not HAS_RL_LIBS:
    print(f"[ERROR] RL könyvtárak hiányoznak! Python: {sys.executable}")
    for m in _missing_libs:
        print(f"  - {m}")
    print(f"  Tipp: aktiváld a megfelelő conda/venv környezetet, vagy futtasd:")
    print(f"  pip install stable-baselines3 sb3-contrib wandb torch")

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
                 n_envs=1,
                 single_agent_id=None,
                 sumo_gui=False,
                 load_model_path=None,  # [NEW] Path to pre-trained model .zip
                 **env_kwargs):

        
        self.net_file = net_file
        self.logic_file = logic_file
        self.detector_file = detector_file
        self.sumo_gui = sumo_gui
        self.total_timesteps = total_timesteps
        self.project_name = wandb_project
        self.log_queue = log_queue
        self.hyperparams = hyperparams or {}
        self.reward_weights = reward_weights or {'waiting': 1.0, 'co2': 1.0}
        self.single_agent_id = single_agent_id
        self.single_agent_id = single_agent_id
        self.load_model_path = load_model_path
        self.env_kwargs = env_kwargs
        
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

        # Config összefésülése
        current_config = self.hyperparams.copy()
        if HAS_RL_LIBS and wandb.run:
            for key in wandb.config.keys():
                current_config[key] = wandb.config[key]
            self.log("WandB config applied (Sweep compatible).")

        # 2. Környezet létrehozása
        # Forgalmi tartomány: epizódonként randomizálódik (target+spread)
        self.env = SumoRLEnvironment(
            net_file=self.net_file,
            logic_json_file=self.logic_file,
            detector_file=self.detector_file,
            reward_weights=self.reward_weights,
            sumo_gui=self.sumo_gui,
            min_green_time=5,
            delta_time=5,
            random_traffic=True,
            single_agent_id=self.single_agent_id,
            run_id=f"{self.single_agent_id or 'ALL'}_{int(time.time())}_{os.getpid()}",
            **self.env_kwargs
        )

        # 3. KÖRNYEZET INDÍTÁSA
        self.log("Starting SUMO...")
        try:
            obs, infos = self.env.reset()
        except Exception as e:
            self.log(f"CRITICAL ERROR during env.reset(): {e}")
            import traceback
            traceback.print_exc()
            return

        agent_ids = list(self.env.agents.keys())
        self.log(f"Agents discovered: {agent_ids}")

        if not HAS_RL_LIBS:
            self.log("RL libraries missing. Stopping.")
            self.env.close()
            return

        # 4. Modellek létrehozása
        runs_dir = os.path.join(os.path.dirname(self.net_file), "runs")

        # Device detektálás: CUDA > MPS (Apple Silicon) > CPU
        import torch
        if torch.cuda.is_available():
            rl_device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            rl_device = "mps"
        else:
            rl_device = "cpu"
        print(f"[INFO] RL device: {rl_device}")

        # Paraméterek betöltése
        lr = float(current_config.get("learning_rate", 1e-4))
        bs = int(current_config.get("batch_size", 32))
        buf = int(current_config.get("buffer_size", 10000))
        gamma = float(current_config.get("gamma", 0.99))
        expl_fraction = float(current_config.get("exploration_fraction", 0.5))
        num_layers = int(current_config.get("num_layers", 2))
        layer_size = int(current_config.get("layer_size", 64))
        train_freq = int(current_config.get("train_freq", 4))
        grad_steps = int(current_config.get("gradient_steps", 1))
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
            
            # [NEW] Transfer Learning Logic
            if self.load_model_path and os.path.exists(self.load_model_path):
                self.log(f"Loading pre-trained model from {self.load_model_path} for agent {jid}...")
                try:
                    if self.load_model_path.endswith('.onnx') or self.load_model_path.endswith('.onnx.zip'):
                        self.log(f"Detected ONNX file or ONNX.ZIP archive. Performing manual PyTorch weight injection...")
                        
                        actual_onnx_path = self.load_model_path
                        if self.load_model_path.endswith('.zip'):
                            import zipfile
                            import tempfile
                            temp_dir = tempfile.mkdtemp()
                            with zipfile.ZipFile(self.load_model_path, 'r') as z:
                                z.extractall(temp_dir)
                                # Find first .onnx file
                                onnx_files = [f for f in os.listdir(temp_dir) if f.endswith('.onnx')]
                                if onnx_files:
                                    actual_onnx_path = os.path.join(temp_dir, onnx_files[0])
                                else:
                                    raise ValueError("No .onnx file found inside the provided zip archive.")
                                    
                        # 1. Initialize fresh PyTorch QRDQN model
                        self.agents[jid] = QRDQN(
                            "MultiInputPolicy",
                            env=model_env,
                            learning_rate=lr,
                            buffer_size=buf,
                            batch_size=bs,
                            gamma=gamma,
                            exploration_fraction=expl_fraction,
                            verbose=0,
                            tensorboard_log=tb_log,
                            device="cpu", # Force CPU for reliable loading
                            policy_kwargs=dict(net_arch=[layer_size] * num_layers)
                        )
                        
                        # 2. Extract weights from ONNX and inject
                        import onnx
                        from onnx import numpy_helper
                        import torch
                        model_proto = onnx.load(actual_onnx_path, load_external_data=False)
                        try:
                            onnx_weights = []
                            for init in model_proto.graph.initializer:
                                onnx_weights.append(numpy_helper.to_array(init))
                        except Exception as e:
                            self.log(f"CRITICAL ERROR: Could not extract weights from ONNX.")
                            self.log(f"If you exported a 'fused' model, PyTorch may have saved the weights into a separate '.onnx.data' file!")
                            self.log(f"Please use a standard non-fused ONNX model, or the original '.zip' PyTorch model for fine-tuning.")
                            raise ValueError(f"Missing ONNX external data: {e}")
                            
                        target_sd = self.agents[jid].policy.state_dict()
                        target_keys = [k for k in target_sd.keys() if "weight" in k or "bias" in k]
                        used_idx = set()
                        
                        for k in target_keys:
                            target_param = target_sd[k]
                            t_shape = tuple(target_param.shape)
                            for i, w in enumerate(onnx_weights):
                                if i in used_idx: continue
                                w_shape = tuple(w.shape)
                                if w_shape == t_shape:
                                    with torch.no_grad(): target_param.copy_(torch.from_numpy(w))
                                    used_idx.add(i)
                                    break
                                elif len(w_shape) == 2 and len(t_shape) == 2 and w_shape[::-1] == t_shape:
                                    with torch.no_grad(): target_param.copy_(torch.from_numpy(w.T))
                                    used_idx.add(i)
                                    break
                                    
                        # Switch back to optimal device
                        self.agents[jid] = self.agents[jid].to("auto")
                        self.log(f"ONNX Model weights injected successfully into SB3 PyTorch graph for {jid}.")
                        
                    else:
                        # SB3 load method for native .zip files
                        self.agents[jid] = QRDQN.load(
                            self.load_model_path,
                            env=model_env,
                            print_system_info=True,
                            force_reset=False,
                            custom_objects=None,
                            device=rl_device
                        )
                        
                        # Update parameters for fine-tuning
                        self.agents[jid].learning_rate = lr
                        self.agents[jid].buffer_size = buf
                        self.agents[jid].batch_size = bs
                        self.agents[jid].gamma = gamma
                        self.agents[jid].exploration_fraction = expl_fraction
                        self.agents[jid].verbose = 0
                        self.agents[jid].tensorboard_log = tb_log
                        
                        self.log(f"PyTorch Model loaded successfully from {self.load_model_path}.")
                        
                except Exception as e:
                    import traceback
                    self.log(f"FAILED to load model for {jid}: {e}\n{traceback.format_exc()}")
                    self.log("Falling back to clean initialization.")
                    # Fallback to copy-paste of initialization below
                    self.agents[jid] = QRDQN(
                        "MultiInputPolicy",
                        model_env,
                        learning_rate=lr,
                        buffer_size=buf,
                        batch_size=bs,
                        gamma=gamma,
                        exploration_fraction=expl_fraction,
                        policy_kwargs=dict(net_arch=net_arch),
                        verbose=0,
                        tensorboard_log=tb_log,
                        device=rl_device
                    )
            elif self.load_model_path and self.load_model_path.endswith(".onnx"):
                # [NEW] ONNX Loading Strategy
                self.log(f"Detected ONNX model: {self.load_model_path}")
                
                # 1. Auto-detect architecture from ONNX
                detected_arch = self.detect_architecture_from_onnx(self.load_model_path)
                if detected_arch:
                    self.log(f"Auto-detected architecture from ONNX: {detected_arch}")
                    net_arch = detected_arch
                else:
                    self.log(f"Could not detect architecture. using default/GUI settings: {net_arch}")

                # 2. Initialize fresh agent with correct architecture
                self.agents[jid] = QRDQN(
                    "MultiInputPolicy",
                    model_env,
                    learning_rate=lr,
                    buffer_size=buf,
                    batch_size=bs,
                    gamma=gamma,
                    exploration_fraction=expl_fraction,
                    policy_kwargs=dict(net_arch=net_arch),
                    verbose=0,
                    tensorboard_log=tb_log,
                    device=rl_device
                )
                
                # 3. Load weights
                try:
                    self._load_from_onnx(self.agents[jid], self.load_model_path)
                    self.log(f"Successfully loaded ONNX weights for {jid}!")
                except Exception as e:
                    self.log(f"FAILED to load ONNX weights: {e}")
                    # Agent remains initialized with random weights
            else:
                if self.load_model_path:
                    self.log(f"Warning: Model file {self.load_model_path} not found or unknown format. Initializing new model.")

                self.agents[jid] = QRDQN(
                    "MultiInputPolicy",
                    model_env,
                    learning_rate=lr,
                    buffer_size=buf,
                    batch_size=bs,
                    gamma=gamma,
                    exploration_fraction=expl_fraction,
                    policy_kwargs=dict(net_arch=net_arch),
                    verbose=0,
                    tensorboard_log=tb_log,
                    device=rl_device
                )
            
            self.agents[jid].set_logger(configure(tb_log, ["stdout", "tensorboard"]))

        # =========================================================================
        # 5. TANÍTÁSI CIKLUS (TRY-FINALLY BLOKKAL VÉDVE)
        # =========================================================================
        self.log(f"Starting Training Loop (Gamma={gamma}, Expl_Fraction={expl_fraction})...")
        global_step = 0
        episode_count = 0
        episode_step = 0
        start_time = time.time()

        # --- PROFILING ---
        _prof_predict = 0.0
        _prof_env_step = 0.0
        _prof_train = 0.0
        _prof_buffer = 0.0
        _prof_count = 0
        
        try: # <--- KEZDŐDIK A BIZTONSÁGI BLOKK
            while global_step < self.total_timesteps and not self.stop_requested:
                
                progress = global_step / self.total_timesteps
                remaining_progress = 1.0 - progress
                
                for model in self.agents.values():
                    model._current_progress_remaining = remaining_progress
                    # CRITICAL FIX: Manually update exploration_rate because we are not using model.learn()
                    model.exploration_rate = model.exploration_schedule(remaining_progress)

                # --- AKCIÓVÁLASZTÁS ---
                _t0 = time.perf_counter()
                actions = {}
                for jid, model in self.agents.items():
                    agent_obs = {k: v.reshape(1, *v.shape) for k, v in obs[jid].items()}
                    action, _ = model.predict(agent_obs, deterministic=False)
                    actions[jid] = int(action[0])
                _prof_predict += time.perf_counter() - _t0

                # --- LÉPÉS ---
                _t0 = time.perf_counter()
                next_obs, rewards, global_done, _, infos = self.env.step(actions)
                _prof_env_step += time.perf_counter() - _t0
                global_step += 1

                # --- BUFFER & TRAIN ---
                _t_buf = 0.0
                _t_train = 0.0
                for jid, model in self.agents.items():
                    o = {k: v.reshape(1, *v.shape) for k, v in obs[jid].items()}
                    no = {k: v.reshape(1, *v.shape) for k, v in next_obs[jid].items()}
                    a = np.array([[actions[jid]]])
                    r = np.array([rewards[jid]])
                    d = np.array([global_done])

                    _tb = time.perf_counter()
                    model.replay_buffer.add(o, no, a, r, d, [infos[jid]])
                    _t_buf += time.perf_counter() - _tb

                    self.reward_smoothing[jid] = 0.95 * self.reward_smoothing[jid] + 0.05 * rewards[jid]

                    if global_step > 100 and global_step % train_freq == 0:
                        _tt = time.perf_counter()
                        model.train(gradient_steps=grad_steps, batch_size=bs)
                        _t_train += time.perf_counter() - _tt

                    model.num_timesteps += 1
                _prof_buffer += _t_buf
                _prof_train += _t_train
                _prof_count += 1

                obs = next_obs
                episode_step += 1

                # --- LOGGING ---
                if global_step % 100 == 0:
                    elapsed = time.time() - start_time
                    fps = int(global_step / (elapsed + 1e-5))
                    self.log(f"Step: {global_step}/{self.total_timesteps} | FPS: {fps}")

                    # --- PROFILING REPORT (every 100 steps) ---
                    if _prof_count > 0:
                        _total = _prof_predict + _prof_env_step + _prof_train + _prof_buffer
                        if _total > 0:
                            self.log(
                                f"  [PROFILE] predict={_prof_predict/_prof_count*1000:.1f}ms "
                                f"env.step={_prof_env_step/_prof_count*1000:.1f}ms "
                                f"train={_prof_train/_prof_count*1000:.1f}ms "
                                f"buffer={_prof_buffer/_prof_count*1000:.1f}ms "
                                f"| split: env={_prof_env_step/_total*100:.0f}% "
                                f"train={_prof_train/_total*100:.0f}% "
                                f"predict={_prof_predict/_total*100:.0f}%"
                            )
                        _prof_predict = _prof_env_step = _prof_train = _prof_buffer = 0.0
                        _prof_count = 0
                    
                    if wandb.run:
                        log_dict = {
                            "global_step": global_step, 
                            "fps": fps,
                            "train/gamma": gamma,
                        }
                        for jid, model in self.agents.items():
                            curr_lr = model.policy.optimizer.param_groups[0]["lr"]
                            curr_loss = model.logger.name_to_value.get("train/loss", 0.0)
                            curr_epsilon = model.exploration_schedule(remaining_progress)
                            
                            log_dict[f"{jid}/train/learning_rate"] = curr_lr
                            log_dict[f"{jid}/train/loss"] = curr_loss
                            log_dict[f"{jid}/train/epsilon"] = curr_epsilon
                            log_dict[f"reward_smooth/{jid}"] = self.reward_smoothing[jid]

                        # Calculate global average reward for Sweep optimization
                        if self.reward_smoothing:
                            avg_reward = sum(self.reward_smoothing.values()) / len(self.reward_smoothing)
                            log_dict["avg_reward"] = avg_reward

                        wandb.log(log_dict, commit=True)
                        
                # --- CHECKPOINTS ---
                if global_step in [10000, 25000, 50000, 75000, 100000]:
                    try:
                        self.log(f"--- Saving CHECKPOINT at {global_step} steps ---")
                        # Kimentjük a modellt "_10k", "_25k" stb. suffixszel
                        self.save_pytorch_models(runs_dir, step_suffix=f"{global_step//1000}k")
                        self.export_onnx_models(runs_dir, step_suffix=f"{global_step//1000}k")
                    except Exception as e:
                        self.log(f"Failed to save checkpoint at {global_step} steps: {e}")

                if global_done:
                    episode_count += 1
                    elapsed = time.time() - start_time
                    avg_r = sum(self.reward_smoothing.values()) / max(len(self.reward_smoothing), 1)
                    eps_rate = self.agents[agent_ids[0]].exploration_rate if agent_ids else 0
                    self.log(
                        f"Episode {episode_count} done | "
                        f"steps={episode_step} | "
                        f"total={global_step}/{self.total_timesteps} | "
                        f"avg_reward={avg_r:.4f} | "
                        f"epsilon={eps_rate:.3f} | "
                        f"elapsed={elapsed:.0f}s"
                    )
                    if wandb.run:
                        wandb.log({
                            "episode": episode_count,
                            "episode_length": episode_step,
                            "episode/avg_reward": avg_r,
                        }, commit=False)
                    episode_step = 0
                    obs, _ = self.env.reset()

        except KeyboardInterrupt:
            self.log("Training interrupted by user.")
        except Exception as e:
            self.log(f"Training crashed with error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # =================================================================
            # 6. EZ MINDENKÉPPEN LEFUT (HIBA ESETÉN IS)
            # =================================================================
            self.log("Closing environment and EXPORTING models...")
            
            # Először exportálunk
            try:
                self.save_pytorch_models(runs_dir)
                self.export_onnx_models(runs_dir)
            except Exception as e:
                self.log(f"Final export failed: {e}")

            # Aztán zárunk
            if self.env:
                self.env.close()
            
            if wandb.run:
                wandb.finish()
            
            self.log("Cleanup done.")

    def save_pytorch_models(self, runs_dir, step_suffix=""):
        """Mentés a natív SB3 .zip formátumba Transfer Learning-hez!"""
        if not HAS_RL_LIBS: return
        
        export_dir = os.path.join(runs_dir, "pytorch_models")
        os.makedirs(export_dir, exist_ok=True)
        self.log(f"Saving PyTorch ZIP models to {export_dir}...")
        
        suffix_str = f"_{step_suffix}" if step_suffix else ""
        
        for jid, model in self.agents.items():
            try:
                zip_path = os.path.join(export_dir, f"{jid}{suffix_str}.zip")
                model.save(zip_path)
                self.log(f"Saved PyTorch {jid} to {zip_path}")
                
                if wandb.run:
                    wandb.save(zip_path, base_path=runs_dir)
            except Exception as e:
                self.log(f"Failed to save PyTorch zip for {jid}: {e}")

    def export_onnx_models(self, runs_dir, step_suffix=""):
        """Export models to ONNX format."""
        if not HAS_RL_LIBS: return

        import torch
        export_dir = os.path.join(runs_dir, "onnx_models")
        os.makedirs(export_dir, exist_ok=True)
        self.log(f"Exporting ONNX models to {export_dir}...")

        suffix_str = f"_{step_suffix}" if step_suffix else ""

        for jid, model in self.agents.items():
            try:
                # 1. Create dummy observation
                obs_space = self.env.observation_space[jid]
                dummy_obs = {}
                for key, space in obs_space.spaces.items():
                    # Create dummy tensor with batch size 1
                    shape = (1,) + space.shape
                    dummy_obs[key] = torch.zeros(shape).to(model.device)

                # 2. Get the policy
                policy = model.policy
                policy.set_training_mode(False) # Ensure eval mode for export

                # 3. Export
                onnx_path = os.path.join(export_dir, f"{jid}{suffix_str}.onnx")
                
                # QRDQN policy forward pass expects 'obs'
                # We trace the 'predict' method or the policy forward
                # SB3 policies usually take 'obs' as input. 
                # For Dict observation, forward expects a dict of tensors.
                
                # Note: Tracing dictionary inputs with torch.onnx can be tricky.
                # simpler approach: Export the feature extractor + q-net head logic?
                # Or just try standard export on policy.
                
                # Wrapper for dictionary handling
                # Dynamically build wrapper from observation space keys
                obs_keys = sorted(obs_space.spaces.keys())  # deterministic order

                class OnnxWrapper(torch.nn.Module):
                    def __init__(self, policy, keys):
                        super().__init__()
                        self.policy = policy
                        self.keys = keys

                    def forward(self, *args):
                        # Reconstruct dict from positional args
                        obs = {k: v for k, v in zip(self.keys, args)}
                        return self.policy(obs)

                if isinstance(model.policy.observation_space, gym.spaces.Dict):
                    # We need to decompose the dict
                    wrapper = OnnxWrapper(policy, obs_keys)
                    inputs = tuple(dummy_obs[k] for k in obs_keys)
                    
                    dynamic_axes_dict = {k: {0: "batch_size"} for k in obs_keys}
                    dynamic_axes_dict["action_logits"] = {0: "batch_size"}

                    torch.onnx.export(
                        wrapper,
                        inputs,
                        onnx_path,
                        export_params=True,
                        keep_initializers_as_inputs=False,
                        opset_version=18,
                        input_names=obs_keys,
                        output_names=["action_logits"],
                        dynamic_axes=dynamic_axes_dict
                    )
                else:
                    # Flat observation
                    dummy_input = torch.zeros((1,) + obs_space.shape).to(model.device)
                    torch.onnx.export(
                        policy,
                        dummy_input,
                        onnx_path,
                        export_params=True,
                        keep_initializers_as_inputs=False,
                        opset_version=18,
                        input_names=["input"],
                        output_names=["output"]
                    )
                
                self.log(f"Exported {jid} to {onnx_path}")
                
                # Upload to WandB
                if wandb.run:
                    wandb.save(onnx_path, base_path=runs_dir)
                    self.log(f"Uploaded {jid} ONNX to WandB")



                # --- FUSED EXPORT (Optimized for Inference) ---
                if isinstance(model, QRDQN):
                    try:
                        fused_onnx_path = os.path.join(export_dir, f"{jid}{suffix_str}_fused.onnx")
                        n_quantiles = model.policy.n_quantiles
                        self.log(f"Creating FUSED model for {jid} (collapsing {n_quantiles} quantiles)...")

                        # 1. Get the Quantile Network (final layer)
                        
                        class FusedQRDQNPolicy(torch.nn.Module):
                            def __init__(self, original_policy, keys):
                                super().__init__()
                                self.keys = keys
                                # QRDQN stores features_extractor inside quantile_net
                                self.features_extractor = original_policy.quantile_net.features_extractor

                                # Access the internal sequential network
                                q_net_seq = original_policy.quantile_net.quantile_net

                                # The last layer expands to n_actions * n_quantiles
                                last_layer = q_net_seq[-1]

                                # Body = all layers EXCEPT the last one
                                self.body = torch.nn.Sequential(*list(q_net_seq.children())[:-1])

                                n_quantiles = original_policy.n_quantiles
                                n_actions = model.action_space.n

                                W = last_layer.weight.data
                                B = last_layer.bias.data

                                # Reshape and Average quantiles
                                W_fused = W.view(n_actions, n_quantiles, -1).mean(dim=1)
                                B_fused = B.view(n_actions, n_quantiles).mean(dim=1)

                                self.fused_head = torch.nn.Linear(W_fused.shape[1], n_actions)
                                self.fused_head.weight.data = W_fused
                                self.fused_head.bias.data = B_fused

                            def forward(self, *args):
                                obs = {k: v for k, v in zip(self.keys, args)}
                                features = self.features_extractor(obs)
                                latent = self.body(features)
                                return self.fused_head(latent)

                        fused_model = FusedQRDQNPolicy(model.policy, obs_keys)

                        fused_inputs = tuple(dummy_obs[k] for k in obs_keys)

                        fused_dynamic = {k: {0: "batch_size"} for k in obs_keys}
                        fused_dynamic["action_logits"] = {0: "batch_size"}

                        torch.onnx.export(
                            fused_model,
                            fused_inputs,
                            fused_onnx_path,
                            export_params=True,
                            keep_initializers_as_inputs=False,
                            opset_version=18,
                            input_names=obs_keys,
                            output_names=["action_logits"],
                            dynamic_axes=fused_dynamic
                        )
                        
                        self.log(f"Exported FUSED {jid} to {fused_onnx_path}")
                        
                        if wandb.run:
                            wandb.save(fused_onnx_path, base_path=runs_dir)
                            self.log(f"Uploaded {jid} FUSED ONNX to WandB")

                    except Exception as e:
                        self.log(f"Failed to export FUSED model for {jid}: {e}")

            except Exception as e:
                self.log(f"Failed to export {jid}: {e}")

    def _load_from_onnx(self, agent, onnx_path):
        """
        Attempts to load weights from an ONNX file into the SB3 agent.
        Uses a Greedy Shape Matching strategy because names often differ.
        """
        import onnx
        from onnx import numpy_helper
        
        self.log(f"Parsing ONNX model from {onnx_path}...")
        model_proto = onnx.load(onnx_path)
        
        # 1. Extract Initializers (Weights/Biases) from ONNX Graph
        # We store them as a list of numpy arrays
        onnx_weights = []
        for initializer in model_proto.graph.initializer:
            w = numpy_helper.to_array(initializer)
            onnx_weights.append({'name': initializer.name, 'data': w})
            
        self.log(f"Found {len(onnx_weights)} parameters in ONNX file.")
        
        # 2. Get PyTorch State Dict
        target_sd = agent.policy.state_dict()
        loaded_count = 0
        
        # 3. Greedy Match
        onnx_idx = 0
        used_onnx_indices = set()
        
        # Filter target keys to strictly those we expect to change (weights/biases)
        target_keys = [k for k in target_sd.keys() if "weight" in k or "bias" in k]
        
        for k in target_keys:
            target_param = target_sd[k]
            target_shape = tuple(target_param.shape)
            
            # Start search from last used index to preserve order
            match_found = False
            for i in range(len(onnx_weights)):
                if i in used_onnx_indices: continue
                
                onnx_w = onnx_weights[i]['data']
                onnx_shape = tuple(onnx_w.shape)
                
                # Direct Match
                if onnx_shape == target_shape:
                    with torch.no_grad():
                        target_param.copy_(torch.from_numpy(onnx_w).to(agent.device))
                    used_onnx_indices.add(i)
                    loaded_count += 1
                    match_found = True
                    break
                    
                # Transpose Match
                if len(onnx_shape) == 2 and len(target_shape) == 2:
                    if onnx_shape[::-1] == target_shape:
                        with torch.no_grad():
                            target_param.copy_(torch.from_numpy(onnx_w.T).to(agent.device))
                        used_onnx_indices.add(i)
                        loaded_count += 1
                        match_found = True
                        self.log(f"Matched {k} (TRANSPOSED) with ONNX param {i}")
                        break
            
            if not match_found:
                self.log(f"Warning: No matching ONNX weight found for {k} {target_shape}")

        self.log(f"Loaded {loaded_count} / {len(target_keys)} layers from ONNX.")
        
        # Copy to target network as well to start synced
        agent.policy.quantile_net_target.load_state_dict(agent.policy.quantile_net.state_dict())
        self.log("Synced target network.")

    def detect_architecture_from_onnx(self, onnx_path):
        """
        Analyzes the ONNX file to deduce the hidden layer sizes (net_arch).
        Assumes a standard MLP structure where weights are (Out, In).
        Returns a list of integers (e.g., [64, 64]) or None if detection fails.
        """
        import onnx
        from onnx import numpy_helper
        
        try:
            model_proto = onnx.load(onnx_path)
            # Filter for weights (excluding biases)
            weights = []
            for init in model_proto.graph.initializer:
                if init.name.endswith("weight"):
                    w = numpy_helper.to_array(init)
                    if len(w.shape) == 2:
                        weights.append({'name': init.name, 'shape': w.shape, 'index': int(init.name.split('.')[-2]) if init.name.split('.')[-2].isdigit() else 999})
            
            # Sort by index
            weights.sort(key=lambda x: x['index'])
            
            hidden_sizes = []
            
            # We assume the last weight is the HEAD, so we exclude it.
            if len(weights) > 1:
                # Iterate through all but the last one
                for i in range(len(weights) - 1):
                    shape = weights[i]['shape']
                    hidden_sizes.append(shape[0])
                    
                return hidden_sizes
            
        except Exception as e:
            self.log(f"Arch detection failed: {e}")
            
        return None



# =============================================================================
# 2. GUI DIALOG
# =============================================================================

class TrainingDialog:
    def __init__(self, parent, net_file, logic_file, detector_file):
        # Ha nincs GUI könyvtár, dobjon hibát az osztály példányosítása
        if not HAS_GUI:
            print("Hiba: Tkinter nincs telepítve, GUI nem indítható.")
            return

        self.top = tk.Toplevel(parent)
        self.top.title("Reinforcement Learning Trainer")
        self.top.geometry("600x750")
        
        self.net_file = net_file
        self.logic_file = logic_file
        self.detector_file = detector_file
        
        self.log_queue = queue.Queue()
        self.trainer_thread = None
        self.trainer_instance = None
        
        # Load available junctions for the dropdown
        self.available_junctions = []
        if self.logic_file and os.path.exists(self.logic_file):
            try:
                with open(self.logic_file, 'r') as f:
                    data = json.load(f)
                    self.available_junctions = list(data.keys())
            except Exception as e:
                print(f"Error loading logic file for dropdown: {e}")
        
        self.setup_ui()
        self.check_files()
        self.top.after(100, self.process_logs)

    # A további GUI metódusok csak akkor hívódnak meg, ha a HAS_GUI igaz,
    # mert a __init__ visszatér, ha hamis. 
    # De a biztonság kedvéért a definíciók maradhatnak változatlanul,
    # mert a headless mód sosem példányosítja a TrainingDialog-ot.
    
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

        tk.Label(frame_reward, text="Waiting Weight:").grid(row=0, column=0)
        self.entry_w_waiting = tk.Entry(frame_reward, width=10)
        self.entry_w_waiting.insert(0, "1.0")
        self.entry_w_waiting.grid(row=0, column=1)

        tk.Label(frame_reward, text="CO2 Weight:").grid(row=0, column=2)
        self.entry_w_co2 = tk.Entry(frame_reward, width=10)
        self.entry_w_co2.insert(0, "1.0")
        self.entry_w_co2.grid(row=0, column=3)

        # --- SINGLE AGENT SELECTION ---
        frame_single = tk.LabelFrame(self.top, text="Single Intersection Mode", padx=10, pady=5)
        frame_single.pack(fill="x", padx=10, pady=5)
        
        self.var_single_enabled = tk.BooleanVar(value=False)
        chk_single = tk.Checkbutton(frame_single, text="Train ONLY one intersection", variable=self.var_single_enabled, command=self.toggle_single_agent_ui)
        chk_single.grid(row=0, column=0, sticky="w")
        
        tk.Label(frame_single, text="Select Junction ID:").grid(row=0, column=1, padx=10)
        self.combo_agent = ttk.Combobox(frame_single, values=getattr(self, 'available_junctions', []), state="disabled")
        self.combo_agent.grid(row=0, column=2, sticky="ew")

        # --- SUMO GUI CHECKBOX ---
        self.var_gui_enabled = tk.BooleanVar(value=False)
        chk_gui = tk.Checkbutton(frame_single, text="Enable SUMO GUI", variable=self.var_gui_enabled)
        chk_gui.grid(row=1, column=0, sticky="w", pady=5)

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
            "w_waiting": float(self.entry_w_waiting.get()),
            "w_co2": float(self.entry_w_co2.get()),
            "num_layers": int(self.entry_num_layers.get()),
            "layer_size": int(self.entry_layer_size.get()),
            "single_agent_id": self.combo_agent.get() if self.var_single_enabled.get() else None,
            "sumo_gui": self.var_gui_enabled.get()
        }

    def toggle_single_agent_ui(self):
        if self.var_single_enabled.get():
            self.combo_agent.config(state="readonly")
            if self.available_junctions:
                self.combo_agent.current(0)
        else:
            self.combo_agent.config(state="disabled")
            self.combo_agent.set("")

    def start_training(self):
        if not HAS_RL_LIBS:
            messagebox.showerror("Hiba", "RL könyvtárak (SB3, WandB) hiányoznak!")
            return

        try:
            settings = self.get_settings()

            # --- Subprocess-ben indítjuk a tanítást ---
            # libsumo C++ singleton ütközik a tkinter event loop-pal macOS-en (segfault),
            # ezért KÜLÖN processzben fut a SUMO szimuláció.
            cmd = [
                sys.executable, "main_headless.py",
                "--config", "training_config.yaml",
                "--timesteps", str(settings["total_timesteps"]),
                "--project", settings["wandb_project"],
                # GUI-ból beírt hiperparaméterek átadása
                "--learning-rate", str(settings["learning_rate"]),
                "--batch-size", str(settings["batch_size"]),
                "--buffer-size", str(settings["buffer_size"]),
                "--gamma", str(settings["gamma"]),
                "--exploration-fraction", str(settings["exploration_fraction"]),
                "--num-layers", str(settings["num_layers"]),
                "--layer-size", str(settings["layer_size"]),
                "--w-waiting", str(settings["w_waiting"]),
                "--w-co2", str(settings["w_co2"]),
            ]

            if settings["sumo_gui"]:
                cmd.append("--gui")

            if settings["single_agent_id"]:
                cmd.extend(["--single-agent", settings["single_agent_id"]])

            # Környezeti változók
            env = os.environ.copy()
            if settings["wandb_api_key"]:
                env["WANDB_API_KEY"] = settings["wandb_api_key"]

            # libsumo: gyorsabb, de nem támogat GUI-t
            if settings["sumo_gui"]:
                env["USE_LIBSUMO"] = "0"
                sumo_engine = "traci (GUI)"
            else:
                env["USE_LIBSUMO"] = "1"
                sumo_engine = "libsumo"

            env["SWEEP_PROJECT"] = settings["wandb_project"]
            env["SWEEP_TIMESTEPS"] = str(settings["total_timesteps"])

            self.log(f"Tanítás indítása subprocess-ben...")
            self.log(f"  Parancs: {' '.join(cmd)}")
            self.log(f"  SUMO: {sumo_engine}")

            import subprocess as sp
            self.training_process = sp.Popen(
                cmd,
                stdout=sp.PIPE,
                stderr=sp.STDOUT,
                env=env,
                bufsize=1,
                universal_newlines=True
            )

            # Háttérszál olvassa a subprocess stdout-ját és logba írja
            self.trainer_thread = threading.Thread(target=self._read_subprocess_output, daemon=True)
            self.trainer_thread.start()

            self.btn_start.config(state="disabled")
            self.btn_stop.config(state="normal")

        except ValueError as e:
            messagebox.showerror("Input Error", f"Hibás paraméter: {e}")
        except Exception as e:
            messagebox.showerror("Hiba", str(e))
            import traceback
            traceback.print_exc()

    def _read_subprocess_output(self):
        """Olvassa a subprocess stdout-ját és a GUI logba írja."""
        try:
            proc = self.training_process
            for line in proc.stdout:
                line = line.rstrip('\n')
                if line:
                    self.log(line)
            proc.wait()
            exit_code = proc.returncode
            if exit_code == 0:
                self.log("Tanítás sikeresen befejezve.")
            else:
                self.log(f"Tanítás hibával zárult (exit code: {exit_code})")
        except Exception as e:
            self.log(f"Subprocess olvasási hiba: {e}")
        finally:
            self.top.after(0, self.on_training_finished)

    def stop_training(self):
        if hasattr(self, 'training_process') and self.training_process:
            self.log("Leállítás kérve...")
            try:
                self.training_process.terminate()
                # Várunk max 5 mp-et
                try:
                    self.training_process.wait(timeout=5)
                except:
                    self.training_process.kill()
                self.log("Subprocess leállítva.")
            except Exception as e:
                self.log(f"Leállítási hiba: {e}")

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