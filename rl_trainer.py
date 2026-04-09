import gymnasium as gym
import numpy as np
import random
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
    from stable_baselines3 import DQN, PPO, A2C
    from stable_baselines3.common.logger import configure
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.callbacks import BaseCallback
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
    # --- PYTORCH 2.6 COMPATIBILITY FIX ---
    # Stable-Baselines3 commonly saves complex objects (like LR schedules) inside the .zip models. 
    # PyTorch 2.6 changed `weights_only=True` by default, which throws WeightsUnpickler errors and 
    # crashes SB3.load() on older SB3 versions. We monkey-patch it here globally.
    _original_torch_load = torch.load
    def _patched_torch_load(*args, **kwargs):
        kwargs['weights_only'] = False
        return _original_torch_load(*args, **kwargs)
    torch.load = _patched_torch_load
    # -------------------------------------
except ImportError as e:
    HAS_RL_LIBS = False
    _missing_libs.append(f"torch: {e}")

if not HAS_RL_LIBS:
    print(f"[ERROR] RL könyvtárak hiányoznak! Python: {sys.executable}")
    for m in _missing_libs:
        print(f"  - {m}")
    print(f"  Tipp: aktiváld a megfelelő conda/venv környezetet, vagy futtasd:")
    print(f"  pip install stable-baselines3 sb3-contrib wandb torch")

# =============================================================================
# GUIDED EXPLORATION — ALGORITMUS-SPECIFIKUS KONFIGURÁLÁS
# =============================================================================

# DQN / QRDQN: Replay buffer beoltása aktuált lépésekkel
# Ennyi arányban cserélődik le az ágens akciója aktuált vezérlésre.
# Az aktuált tapasztalatok a bufferbe kerülnek, javítva az early-phase stabilitást.
ACTUATED_SEED_RATIO = 0.65          # 65% lépés legyen aktuált

# PPO: Epizód eleji aktuált warmup (az env.reset()-ben fut, PPO nem látja ezeket)
# ~200 SUMO-lépés ≈ 1000 sim-sec ≈ ~16 perc "valós" forgalmi stabilizálás
PPO_ACTUATED_WARMUP_STEPS = 200

# A2C: Rövidebb warmup (az A2C n_steps=5, epizódok rövidek)
# + Entrópia-boost az early explorationhoz, hogy ne ragadjon bele egy rosszba
A2C_ACTUATED_WARMUP_STEPS  = 60
A2C_ENTROPY_BOOST_FACTOR   = 3.0   # ent_coef × 3 a training első 30%-ában
A2C_ENTROPY_BOOST_FRACTION = 0.30  # utána lineárisan visszaáll az alap értékre

# --- Támogatott algoritmusok ---
SUPPORTED_ALGORITHMS = {
    'qrdqn': {'class': 'QRDQN', 'type': 'off_policy', 'source': 'sb3_contrib'},
    'dqn':   {'class': 'DQN',   'type': 'off_policy', 'source': 'sb3'},
    'ppo':   {'class': 'PPO',   'type': 'on_policy',  'source': 'sb3'},
    'a2c':   {'class': 'A2C',   'type': 'on_policy',  'source': 'sb3'},
}

def get_algorithm_class(name):
    """SB3 algoritmus osztály lekérdezése név alapján."""
    info = SUPPORTED_ALGORITHMS.get(name.lower())
    if not info:
        raise ValueError(f"Ismeretlen algoritmus: {name}. Támogatott: {list(SUPPORTED_ALGORITHMS.keys())}")
    cls_name = info['class']
    if cls_name == 'QRDQN': return QRDQN
    elif cls_name == 'DQN': return DQN
    elif cls_name == 'PPO': return PPO
    elif cls_name == 'A2C': return A2C
    raise ValueError(f"Nem elérhető: {cls_name}")

from sumo_rl_environment import SumoRLEnvironment

# Export modul importálása (ha létezik)
try:
    from export_utils import export_to_colab_package
    HAS_EXPORT = True
except ImportError:
    HAS_EXPORT = False

# =============================================================================
# SINGLE-AGENT GYM WRAPPER
# =============================================================================

class SingleAgentGymWrapper(gym.Env):
    """
    A multi-agent SumoRLEnvironment csomagoló, ami egyetlen junction-re
    standard Gym API-t ad. Ezzel bármely SB3 algoritmus (PPO, A2C, DQN, QRDQN)
    használható model.learn()-nel.
    """
    metadata = {"render_modes": []}

    def __init__(self, env, agent_id):
        super().__init__()
        self.env = env
        self.agent_id = agent_id
        self.observation_space = env.observation_space[agent_id]
        self.action_space = env.action_space[agent_id]
        self._last_infos = {}
        self.action_counts = {}

    def reset(self, seed=None, options=None):
        obs_dict, info_dict = self.env.reset(seed=seed, options=options)
        self._last_infos = info_dict.get(self.agent_id, {})
        return obs_dict[self.agent_id], self._last_infos

    def step(self, action):
        action_int = int(action)
        self.action_counts[action_int] = self.action_counts.get(action_int, 0) + 1
        actions = {self.agent_id: action_int}
        obs_dict, rewards_dict, terminated_raw, truncated_raw, info_dict = self.env.step(actions)
        obs = obs_dict[self.agent_id]
        reward = rewards_dict[self.agent_id]
        info = info_dict.get(self.agent_id, {})
        self._last_infos = info
        # terminated/truncated lehet bool (SumoRLEnvironment global_done)
        # vagy dict (multi-agent API) — mindkettőt kezeljük
        if isinstance(terminated_raw, dict):
            terminated = bool(terminated_raw.get("__all__", False) or terminated_raw.get(self.agent_id, False))
        else:
            terminated = bool(terminated_raw)
        if isinstance(truncated_raw, dict):
            truncated = bool(truncated_raw.get("__all__", False) or truncated_raw.get(self.agent_id, False))
        else:
            truncated = bool(truncated_raw)
            
        if terminated or truncated:
            print(f"[ACTION LOG] {self.agent_id} Action distribution this episode: {self.action_counts}")
            self.action_counts = {}
            
        return obs, reward, terminated, truncated, info

    def close(self):
        self.env.close()

    def render(self):
        pass


# =============================================================================
# WANDB CALLBACK (on-policy algoritmusokhoz)
# =============================================================================

if HAS_RL_LIBS:
    class WandBCallback(BaseCallback):
        """
        WandB logging callback SB3 model.learn()-höz (on-policy).
        Ugyanazokat a metrikákat logollja, mint az off-policy manuális ciklus,
        hogy a WandB dashboard-on konzisztensek legyenek a chart-ok.
        """
        def __init__(self, agent_id, log_freq=100, verbose=0):
            super().__init__(verbose)
            self.agent_id = agent_id
            self.log_freq = log_freq
            self.reward_smooth = 0.0
            self.episode_count = 0
            self.episode_rewards = []
            self.action_counts = {}
            self._start_time = None

        def _on_training_start(self):
            self._start_time = time.time()

        def _on_step(self) -> bool:
            if not wandb.run:
                return True

            # Reward smoothing (azonos az off-policy 0.95 decay-jel)
            reward = self.locals.get("rewards", [0.0])
            if len(reward) > 0:
                r = float(reward[0])
                self.reward_smooth = 0.95 * self.reward_smooth + 0.05 * r
                self.episode_rewards.append(r)

            # Action tracking
            actions = self.locals.get("actions", [])
            if len(actions) > 0:
                a = int(actions[0])
                self.action_counts[a] = self.action_counts.get(a, 0) + 1

            # Periodikus logolás (100 step-enként, mint az off-policy)
            if self.num_timesteps % self.log_freq == 0:
                elapsed = time.time() - (self._start_time or time.time())
                fps = int(self.num_timesteps / (elapsed + 1e-5))

                log_dict = {
                    "global_step": self.num_timesteps,
                    "fps": fps,
                    "train/gamma": self.model.gamma,
                    f"reward_smooth/{self.agent_id}": self.reward_smooth,
                    "avg_reward": self.reward_smooth,
                }

                # Learning rate
                try:
                    curr_lr = self.model.policy.optimizer.param_groups[0]["lr"]
                    log_dict[f"{self.agent_id}/train/learning_rate"] = curr_lr
                except Exception:
                    pass

                # Loss (SB3 on-policy loggerből)
                try:
                    logger_vals = self.model.logger.name_to_value
                    for key in ["train/loss", "train/policy_gradient_loss",
                                "train/value_loss", "train/entropy_loss"]:
                        if key in logger_vals:
                            log_dict[f"{self.agent_id}/{key}"] = logger_vals[key]
                except Exception:
                    pass

                # Metrikák az info-ból
                if self.locals.get("infos"):
                    for info in self.locals["infos"]:
                        if isinstance(info, dict):
                            if "metric_avg_speed" in info:
                                log_dict["metric/avg_speed"] = info["metric_avg_speed"]
                            if "metric_throughput" in info:
                                log_dict["metric/throughput"] = info["metric_throughput"]

                wandb.log(log_dict, commit=True)

            # Epizód vége detektálás
            dones = self.locals.get("dones", [False])
            if len(dones) > 0 and dones[0]:
                self.episode_count += 1
                ep_reward = sum(self.episode_rewards) / max(len(self.episode_rewards), 1)

                ep_log = {
                    "episode": self.episode_count,
                    "episode_length": len(self.episode_rewards),
                    "episode/avg_reward": ep_reward,
                }

                # Action distribution logging (minden epizódban)
                if self.action_counts:
                    total_a = sum(self.action_counts.values()) or 1
                    
                    # Log to WandB
                    for a_id, cnt in self.action_counts.items():
                        ep_log[f"{self.agent_id}/action_pct/phase_{a_id}"] = cnt / total_a * 100
                        
                    # Print to console via model logger if possible, or just standard print
                    dist_str = " ".join([f"a{k}={v/total_a*100:.0f}% ({v})" for k, v in sorted(self.action_counts.items())])
                    print(f"  [ACTION PPO] {self.agent_id}: {dist_str}")

                wandb.log(ep_log, commit=False)
                self.episode_rewards = []
                self.action_counts = {}

            return True


# =============================================================================
# A2C ENTROPIA-BOOST CALLBACK
# =============================================================================

if HAS_RL_LIBS:
    class A2CEntropyScheduleCallback(BaseCallback):
        """
        A2C számára: az entrópia-együttható lineárisan csökkentő ütemezése.

        Az early exploration során az ent_coef boost_factor-szorosára nő,
        hogy az ágens ne ragadjon bele a warmup utáni első "jó elég" megoldásba.
        A boost lineárisan csökken 0-ra a boost_fraction × total_timesteps lépésig,
        majd marad az alap értéken.

        Különbség a többi algotól:
          DQN/QRDQN: lépésszintű aktuált injekció (15%)
          PPO:       epizódeleji env-szintű warmup (ágens nem látja)
          A2C:       epizódeleji env-szintű warmup + ent_coef boost (ágens maga explorálja)
        """
        def __init__(self, base_ent_coef: float, boost_factor: float,
                     boost_fraction: float, total_timesteps: int, verbose: int = 0):
            super().__init__(verbose)
            self.base_ent_coef    = base_ent_coef
            self.boosted_ent_coef = base_ent_coef * boost_factor
            self.boost_steps      = int(boost_fraction * total_timesteps)

        def _on_step(self) -> bool:
            if self.num_timesteps < self.boost_steps:
                # Lineáris decay: boosted → base
                t = self.num_timesteps / max(1, self.boost_steps)
                ent = self.boosted_ent_coef + t * (self.base_ent_coef - self.boosted_ent_coef)
            else:
                ent = self.base_ent_coef
            self.model.ent_coef = float(ent)

            if wandb.run and self.num_timesteps % 100 == 0:
                wandb.log({"train/ent_coef_actual": float(ent),
                           "global_step": self.num_timesteps}, commit=False)
            return True

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
                 load_model_path=None,
                 fixed_flow=None,
                 algorithm="qrdqn",
                 reward_mode="speed_throughput",
                 junction_params_path=None,
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
        self.load_model_path = load_model_path
        self.fixed_flow = fixed_flow
        self.algorithm_name = algorithm.lower()
        self.reward_mode = reward_mode
        self.junction_params_path = junction_params_path
        self.env_kwargs = env_kwargs

        self.stop_requested = False
        self.agents = {}
        self.env = None
        self.reward_smoothing = {}
        self.action_counts = {}

        # Algoritmus validáció
        algo_info = SUPPORTED_ALGORITHMS.get(self.algorithm_name)
        if not algo_info:
            raise ValueError(f"Ismeretlen algoritmus: {algorithm}. "
                           f"Támogatott: {list(SUPPORTED_ALGORITHMS.keys())}")
        self.algo_type = algo_info['type']  # 'on_policy' vagy 'off_policy'

    def _build_reset_options(self):
        """Reset options összeállítása fix forgalom beállításokkal."""
        options = {}
        if self.fixed_flow:
            target = self.fixed_flow.get('target', 300)
            spread = self.fixed_flow.get('spread', 50)
            flow_range = (max(100, target - spread), min(650, target + spread))
            options['flow_range'] = flow_range
            # Single agent focused traffic: period-re konvertálás
            # target veh/h/lane → period = 3600 / target sec/veh
            if self.single_agent_id and spread == 0:
                options['traffic_period'] = round(3600.0 / max(target, 1), 4)
        return options if options else None

    def log(self, msg):
        if self.log_queue: self.log_queue.put(msg)
        else: print(msg)

    def _get_actuated_action(self, jid: str, obs: dict) -> int:
        """
        Max-pressure közelítő aktuált akció-becslés (DQN/QRDQN buffer-beoltáshoz).

        A detektorokat egyenlően osztja el a fázisok között, majd a legmagasabb
        átlagos occupancy-jú fázist preferálja. Ha a jelenlegi fázis terhelése
        legalább 50%-a a maximumnak ÉS érdemi (>0.05), kiterjeszti azt.
        Különben a legzsúfoltabb fázisra vált.

        Visszaad: phase_idx (int)
        """
        agent = self.env.agents.get(jid)
        if agent is None:
            return 0
        if not agent.is_ready_for_action():
            return agent.current_logic_idx

        occ = obs.get('occupancy', np.array([]))
        n_phases = agent.num_phases

        if len(occ) == 0 or n_phases <= 1:
            # Nincs detektor-adat → round-robin
            return (agent.current_logic_idx + 1) % n_phases

        # Detektorokat egyenlően osztjuk fel a fázisok közé (közelítés)
        n_dets  = len(occ)
        chunk   = max(1, n_dets // n_phases)
        phase_pressure = []
        for p in range(n_phases):
            start = (p * chunk) % n_dets
            end   = min(start + chunk, n_dets)
            phase_pressure.append(float(np.mean(occ[start:end])))

        best_p    = max(phase_pressure)
        current_p = phase_pressure[agent.current_logic_idx]

        # Kiterjesztés, ha a jelenlegi fázis elég terhelt
        if current_p >= 0.5 * best_p and current_p > 0.05:
            return agent.current_logic_idx
        else:
            return int(np.argmax(phase_pressure))

    def _run_warmup_calibration(self, initial_obs):
        """3 órás szimuláció a nyers metrikák leméréséhez (súlyok frissítése nélkül)."""
        if not self.load_model_path or not self.single_agent_id:
            return initial_obs

        jid = self.single_agent_id
        model = self.agents.get(jid)
        if not model: return initial_obs

        self.log(f"==================================================")
        self.log(f"[WARMUP] Indítás: kalibráció 6 darab félórás epizóddal...")
        
        options = self._build_reset_options() or {}
        options['traffic_duration'] = 1800
        
        raw_metrics = {'speed': [], 'throughput': [], 'halt': [], 'co2': [], 'speedstd': []}
        
        local_obs = initial_obs
        
        for ep in range(6):
            # Enforce 3600 duration manually in the options (so single override)
            curr_obs, _ = self.env.reset(options=options)
            local_obs = curr_obs
            
            global_done = False
            step_idx = 0
            
            while not global_done and not self.stop_requested:
                # Transzfer tanulásnál az előtanított modell egy teljesen ismeretlen 
                # kereszteződésbe kerülhet, ahol könnyen "beragad" egyetlen fázisba (pl. a3=100%).
                # Ezért a bemelegítésnél tiszta, véletlenszerű akciókat (random) használunk 
                # a torzítatlan kalibrációhoz!
                act = int(self.env.action_space[jid].sample())
                if act not in raw_metrics: raw_metrics[act] = 0
                raw_metrics[act] += 1
                
                next_obs, rewards, global_done, _, infos = self.env.step({jid: act})
                
                ag = self.env.agents[jid]
                # Log(x + 1e-5) transformation to match reward scaler logic
                if ag.steps_measured > 0: # Ensure valid measurements
                    raw_metrics['speed'].append(np.log(ag.get_avg_speed_metric() + 1e-5))
                    raw_metrics['throughput'].append(np.log(ag.get_throughput_metric() + 1e-5))
                    raw_metrics['halt'].append(np.log(ag.get_halt_ratio_metric() + 1e-5))
                    raw_metrics['co2'].append(np.log((ag.accumulated_co2 / ag.steps_measured) + 1e-5))
                    raw_metrics['speedstd'].append(np.log(ag.get_speed_std_metric() + 1e-5))
                
                local_obs = next_obs
                step_idx += 1
                
            self.log(f"  [WARMUP] Epizód {ep+1}/6 vége ({step_idx} lépés).")
            # Action distribution dump
            acts_only = {k: v for k, v in raw_metrics.items() if isinstance(k, int)}
            if acts_only:
                ta = sum(acts_only.values())
                dist_str = " ".join([f"a{k}={v/ta*100:.0f}% ({v})" for k, v in sorted(acts_only.items())])
                self.log(f"  [WARMUP AKCIÓK] {dist_str}")
                # Reset actions for next episode
                for k in acts_only.keys(): del raw_metrics[k]
                
            if self.stop_requested:
                break
                
        # --- Parameter Updates ---
        if not self.stop_requested and len(raw_metrics['speed']) > 0:
            new_params = {}
            if "speed" in self.reward_mode or "throughput" in self.reward_mode:
                new_params['MU_SPEED'] = float(np.mean(raw_metrics['speed']))
                new_params['STD_SPEED'] = float(np.std(raw_metrics['speed'])) + 1e-9
                new_params['MU_THROUGHPUT'] = float(np.mean(raw_metrics['throughput']))
                new_params['STD_THROUGHPUT'] = float(np.std(raw_metrics['throughput'])) + 1e-9
                self.log(f"  [KALIBRÁLVA] SPEED: mu={new_params['MU_SPEED']:.3f}, std={new_params['STD_SPEED']:.3f}")
                self.log(f"  [KALIBRÁLVA] THROU: mu={new_params['MU_THROUGHPUT']:.3f}, std={new_params['STD_THROUGHPUT']:.3f}")
                
            if "halt" in self.reward_mode:
                new_params['MU_HALT'] = float(np.mean(raw_metrics['halt']))
                new_params['STD_HALT'] = float(np.std(raw_metrics['halt'])) + 1e-9
                self.log(f"  [KALIBRÁLVA] HALT : mu={new_params['MU_HALT']:.3f}, std={new_params['STD_HALT']:.3f}")
                
            if "co2" in self.reward_mode or "speedstd" in self.reward_mode:
                new_params['MU_CO2'] = float(np.mean(raw_metrics['co2']))
                new_params['STD_CO2'] = float(np.std(raw_metrics['co2'])) + 1e-9
                new_params['MU_SPEEDSTD'] = float(np.mean(raw_metrics['speedstd']))
                new_params['STD_SPEEDSTD'] = float(np.std(raw_metrics['speedstd'])) + 1e-9
                self.log(f"  [KALIBRÁLVA] CO2  : mu={new_params['MU_CO2']:.3f}, std={new_params['STD_CO2']:.3f}")
                self.log(f"  [KALIBRÁLVA] S_STD: mu={new_params['MU_SPEEDSTD']:.3f}, std={new_params['STD_SPEEDSTD']:.3f}")
                
            self.env.update_reward_params(jid, new_params)
            self.log(f"==================================================")
            
            # Reset environment after warmup to start fresh
            final_obs, _ = self.env.reset(options=self._build_reset_options())
            return final_obs
        
        return initial_obs

    def run(self):
        self.log(f"Initializing Environment (algorithm={self.algorithm_name}, reward={self.reward_mode})...")

        AlgoClass = get_algorithm_class(self.algorithm_name)
        self.log(f"[INFO] Algorithm: {self.algorithm_name} ({self.algo_type})")
        self.log(f"[INFO] Reward mode: {self.reward_mode}")

        # 1. WandB init és SWEEP Támogatás
        if HAS_RL_LIBS:
            wandb_config = {
                **self.hyperparams,
                'algorithm': self.algorithm_name,
                'reward_mode': self.reward_mode,
                'junction_id': self.single_agent_id or 'ALL',
            }
            if wandb.run is None:
                try:
                    wandb_kwargs = {
                        'project': self.project_name,
                        'config': wandb_config,
                        'sync_tensorboard': False,
                    }
                    # Per-junction group (ha single agent)
                    if self.single_agent_id:
                        wandb_kwargs['group'] = self.single_agent_id
                        wandb_kwargs['tags'] = [self.single_agent_id, self.algorithm_name, self.reward_mode]
                    # WANDB_RUN_NAME env var támogatás (grid search script)
                    run_name = os.environ.get('WANDB_RUN_NAME')
                    if run_name:
                        wandb_kwargs['name'] = run_name
                    wandb.init(**wandb_kwargs)
                except Exception as e:
                    self.log(f"WandB init failed (skipped): {e}")

        # Config összefésülése
        current_config = self.hyperparams.copy()
        if HAS_RL_LIBS and wandb.run:
            for key in wandb.config.keys():
                current_config[key] = wandb.config[key]
            self.log("WandB config applied (Sweep compatible).")

        # 2. Környezet létrehozása
        # Algoritmus-specifikus aktuált warmup lépések meghatározása
        # (env_kwargs-ban felülírható, ha kell)
        _default_warmup = {
            'ppo':   PPO_ACTUATED_WARMUP_STEPS,
            'a2c':   A2C_ACTUATED_WARMUP_STEPS,
            'dqn':   0,   # DQN: lépésszintű buffer-beoltás, nem env-szintű warmup
            'qrdqn': 0,
        }
        _warmup_steps = self.env_kwargs.pop('actuated_warmup_steps',
                                            _default_warmup.get(self.algorithm_name, 0))
        self.log(f"[INFO] Aktuált warmup lépések (env.reset-ben): {_warmup_steps} "
                 f"({self.algorithm_name})")

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
            reward_mode=self.reward_mode,
            junction_params_path=self.junction_params_path,
            actuated_warmup_steps=_warmup_steps,
            **self.env_kwargs
        )

        # 3. KÖRNYEZET INDÍTÁSA
        self.log("Starting SUMO...")
        # Fix forgalom beállítása (ha van)
        reset_options = self._build_reset_options()
        try:
            obs, infos = self.env.reset(options=reset_options)
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
        ent_coef = float(current_config.get("ent_coef", 0.01))
        net_arch = [layer_size] * num_layers

        # =====================================================================
        # ON-POLICY ÚT (PPO, A2C) — SingleAgentGymWrapper + model.learn()
        # =====================================================================
        if self.algo_type == 'on_policy':
            if len(agent_ids) > 1:
                self.log("[WARNING] On-policy algoritmus (PPO/A2C) csak single-agent módban támogatott!")
                self.log("         Használj --single-agent-t vagy válts off-policy algoritmusra (QRDQN/DQN).")
                self.env.close()
                return

            jid = agent_ids[0]
            self.log(f"[INFO] On-policy training: {self.algorithm_name} → {jid}")

            # Az env-et resetelni kell, hogy a observation/action space-ek inicializálódjanak
            # (_map_network_elements() csak reset() után fut le, ami beállítja a tereket)
            self.log("[INFO] Env reset az observation/action space inicializálásához...")
            self.env.reset()

            # SingleAgentGymWrapper: standard Gym API a SUMO env felett
            gym_env = SingleAgentGymWrapper(self.env, jid)
            tb_log = os.path.join(runs_dir, self.project_name, jid)

            # Modell létrehozása
            model_kwargs = {
                'policy': "MultiInputPolicy",
                'env': gym_env,
                'learning_rate': lr,
                'gamma': gamma,
                'verbose': 1,
                'tensorboard_log': tb_log,
                'device': rl_device,
                'policy_kwargs': dict(net_arch=net_arch),
            }
            # PPO/A2C specifikus paraméterek
            if self.algorithm_name == 'ppo':
                model_kwargs['batch_size'] = bs
                model_kwargs['n_steps'] = int(current_config.get("n_steps", 2048))
                model_kwargs['n_epochs'] = int(current_config.get("n_epochs", 10))
                model_kwargs['ent_coef'] = ent_coef
            elif self.algorithm_name == 'a2c':
                model_kwargs['n_steps'] = int(current_config.get("n_steps", 5))
                model_kwargs['ent_coef'] = ent_coef

            # Transfer learning
            if self.load_model_path and os.path.exists(self.load_model_path):
                self.log(f"Loading pre-trained model: {self.load_model_path}")
                try:
                    model = AlgoClass.load(self.load_model_path, env=gym_env, device=rl_device)
                    model.learning_rate = lr
                    model.gamma = gamma
                    self.log("Pre-trained model loaded successfully.")
                except Exception as e:
                    self.log(f"Failed to load model: {e}. Initializing fresh model.")
                    model = AlgoClass(**model_kwargs)
            else:
                model = AlgoClass(**model_kwargs)

            self.agents[jid] = model

            # --- TRANSFER LEARNING WARMUP ---
            obs = self._run_warmup_calibration(obs)

            # WandB callback + algoritmus-specifikus callback-ok
            callbacks = []
            if wandb.run:
                callbacks.append(WandBCallback(agent_id=jid, log_freq=100))

            # A2C: entrópia-boost callback (lineáris decay boosted → base)
            if self.algorithm_name == 'a2c':
                a2c_ent_boost_cb = A2CEntropyScheduleCallback(
                    base_ent_coef    = ent_coef,
                    boost_factor     = A2C_ENTROPY_BOOST_FACTOR,
                    boost_fraction   = A2C_ENTROPY_BOOST_FRACTION,
                    total_timesteps  = self.total_timesteps,
                )
                callbacks.append(a2c_ent_boost_cb)
                self.log(f"[A2C] Entrópia-boost aktív: {ent_coef:.4f} → "
                         f"{ent_coef * A2C_ENTROPY_BOOST_FACTOR:.4f} (első "
                         f"{A2C_ENTROPY_BOOST_FRACTION*100:.0f}% lépés), "
                         f"majd lineárisan visszaáll.")

            # Tanítás
            try:
                self.log(f"Starting on-policy training: {self.total_timesteps} timesteps...")
                model.learn(
                    total_timesteps=self.total_timesteps,
                    callback=callbacks,
                    log_interval=10,
                )
                self.log("On-policy training completed.")
            except KeyboardInterrupt:
                self.log("Training interrupted by user.")
            except Exception as e:
                self.log(f"Training crashed: {e}")
                import traceback
                traceback.print_exc()
            finally:
                self.log("Closing environment and EXPORTING models...")
                try:
                    self.save_pytorch_models(runs_dir)
                    self.export_onnx_models(runs_dir)
                except Exception as e:
                    self.log(f"Final export failed: {e}")
                if self.env:
                    self.env.close()
                if wandb.run:
                    wandb.finish()
                self.log("Cleanup done.")
            return

        # =====================================================================
        # OFF-POLICY ÚT (DQN, QRDQN) — Eredeti manuális tréning ciklus
        # =====================================================================
        for jid in agent_ids:
            self.reward_smoothing[jid] = 0.0
            agent_obs_space = self.env.observation_space[jid]
            agent_act_space = self.env.action_space[jid]

            def make_dummy_wrapper():
                class _DummyEnv(gym.Env):
                    def __init__(self):
                        self.observation_space = agent_obs_space
                        self.action_space = agent_act_space
                    def reset(self, **kwargs): return self.observation_space.sample(), {}
                    def step(self, a): return self.observation_space.sample(), 0, False, False, {}
                return _DummyEnv()

            model_env = DummyVecEnv([make_dummy_wrapper])
            tb_log = os.path.join(runs_dir, self.project_name, jid)

            # Off-policy modell paraméterek
            model_kwargs = {
                'policy': "MultiInputPolicy",
                'env': model_env,
                'learning_rate': lr,
                'buffer_size': buf,
                'batch_size': bs,
                'gamma': gamma,
                'exploration_fraction': expl_fraction,
                'policy_kwargs': dict(net_arch=net_arch),
                'verbose': 0,
                'tensorboard_log': tb_log,
                'device': rl_device,
            }

            # Transfer Learning
            if self.load_model_path and os.path.exists(self.load_model_path):
                self.log(f"Loading pre-trained model from {self.load_model_path} for agent {jid}...")
                try:
                    if self.load_model_path.endswith('.onnx') or self.load_model_path.endswith('.onnx.zip'):
                        self.log(f"Detected ONNX file. Performing weight injection...")
                        actual_onnx_path = self.load_model_path
                        if self.load_model_path.endswith('.zip'):
                            import zipfile, tempfile
                            temp_dir = tempfile.mkdtemp()
                            with zipfile.ZipFile(self.load_model_path, 'r') as z:
                                z.extractall(temp_dir)
                                onnx_files = [f for f in os.listdir(temp_dir) if f.endswith('.onnx')]
                                if onnx_files:
                                    actual_onnx_path = os.path.join(temp_dir, onnx_files[0])
                                else:
                                    raise ValueError("No .onnx file found in zip.")

                        self.agents[jid] = AlgoClass(**{**model_kwargs, 'device': 'cpu'})

                        import onnx
                        from onnx import numpy_helper
                        model_proto = onnx.load(actual_onnx_path, load_external_data=False)
                        onnx_weights = [numpy_helper.to_array(init) for init in model_proto.graph.initializer]

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
                        self.agents[jid] = self.agents[jid].to("auto")
                        self.log(f"ONNX weights injected for {jid}.")
                    else:
                        # Native .zip load
                        self.agents[jid] = AlgoClass.load(
                            self.load_model_path, env=model_env,
                            print_system_info=True, force_reset=False, device=rl_device
                        )
                        self.agents[jid].learning_rate = lr
                        self.agents[jid].gamma = gamma
                        self.agents[jid].exploration_fraction = expl_fraction
                        self.log(f"Model loaded from {self.load_model_path}.")

                except Exception as e:
                    import traceback
                    self.log(f"FAILED to load model for {jid}: {e}\n{traceback.format_exc()}")
                    self.log("Falling back to clean initialization.")
                    self.agents[jid] = AlgoClass(**model_kwargs)
            else:
                if self.load_model_path:
                    self.log(f"Warning: Model file {self.load_model_path} not found.")
                self.agents[jid] = AlgoClass(**model_kwargs)

            self.agents[jid].set_logger(configure(tb_log, ["stdout", "tensorboard"]))

        # --- TRANSFER LEARNING WARMUP ---
        obs = self._run_warmup_calibration(obs)

        # =========================================================================
        # 5. TANÍTÁSI CIKLUS — OFF-POLICY (TRY-FINALLY BLOKKAL VÉDVE)
        # =========================================================================
        self.log(f"Starting Training Loop ({self.algorithm_name}, gamma={gamma}, expl={expl_fraction})...")
        global_step = 0
        episode_count = 0
        episode_step = 0
        start_time = time.time()

        _prof_predict = 0.0
        _prof_env_step = 0.0
        _prof_train = 0.0
        _prof_buffer = 0.0
        _prof_count = 0

        try:
            while global_step < self.total_timesteps and not self.stop_requested:

                progress = global_step / self.total_timesteps
                remaining_progress = 1.0 - progress

                for model in self.agents.values():
                    model._current_progress_remaining = remaining_progress
                    model.exploration_rate = model.exploration_schedule(remaining_progress)

                # --- AKCIÓVÁLASZTÁS (DQN/QRDQN) ---
                # ACTUATED_SEED_RATIO arányban aktuált akcióval oltjuk be a replay buffert,
                # hogy az early-phase ne ragadjon dugós állapotba random exploráció miatt.
                _t0 = time.perf_counter()
                actions = {}
                _used_actuated = False
                for jid, model in self.agents.items():
                    # A háló beépített deterministic=False epsilon-greedy-jét manuálisan elvégezzük
                    epsilon = model.exploration_rate
                    if random.random() < epsilon:
                        # Felfedezés (Exploration)
                        if random.random() < ACTUATED_SEED_RATIO:
                            # Aktuált vezérlés: max-pressure közelítés
                            actions[jid] = self._get_actuated_action(jid, obs[jid])
                            _used_actuated = True
                        else:
                            # Sima random fázis kiválasztás
                            actions[jid] = int(self.env.action_space[jid].sample())
                    else:
                        # Okos döntés (Exploitation) - predikció determinisztikusan
                        agent_obs = {k: v.reshape(1, *v.shape) for k, v in obs[jid].items()}
                        action, _ = model.predict(agent_obs, deterministic=True)
                        actions[jid] = int(action[0])
                    if jid not in self.action_counts:
                        self.action_counts[jid] = {}
                    a = actions[jid]
                    self.action_counts[jid][a] = self.action_counts[jid].get(a, 0) + 1
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
                            "train/actuated_step": int(_used_actuated),
                        }
                        for jid, model in self.agents.items():
                            curr_lr = model.policy.optimizer.param_groups[0]["lr"]
                            curr_loss = model.logger.name_to_value.get("train/loss", 0.0)
                            curr_epsilon = model.exploration_schedule(remaining_progress)

                            log_dict[f"{jid}/train/learning_rate"] = curr_lr
                            log_dict[f"{jid}/train/loss"] = curr_loss
                            log_dict[f"{jid}/train/epsilon"] = curr_epsilon
                            log_dict[f"reward_smooth/{jid}"] = self.reward_smoothing[jid]

                        if self.reward_smoothing:
                            avg_reward = sum(self.reward_smoothing.values()) / len(self.reward_smoothing)
                            log_dict["avg_reward"] = avg_reward

                        wandb.log(log_dict, commit=True)

                # --- CHECKPOINTS ---
                if global_step in [10000, 25000, 50000, 75000, 100000]:
                    try:
                        self.log(f"--- Saving CHECKPOINT at {global_step} steps ---")
                        self.save_pytorch_models(runs_dir, step_suffix=f"{global_step//1000}k")
                        self.export_onnx_models(runs_dir, step_suffix=f"{global_step//1000}k")
                    except Exception as e:
                        self.log(f"Failed to save checkpoint at {global_step} steps: {e}")

                if global_done:
                    episode_count += 1
                    elapsed = time.time() - start_time
                    avg_r = sum(self.reward_smoothing.values()) / max(len(self.reward_smoothing), 1)
                    eps_rate = self.agents[agent_ids[0]].exploration_rate if agent_ids else 0

                    for jid in agent_ids:
                        counts = self.action_counts.get(jid, {})
                        total_a = sum(counts.values()) or 1
                        dist_str = " ".join([f"a{k}={v/total_a*100:.0f}% ({v})" for k, v in sorted(counts.items())])
                        self.log(f"  [ACTION] {jid}: {dist_str}")
                        if wandb.run:
                            for a_id, cnt in counts.items():
                                wandb.log({f"{jid}/action_pct/phase_{a_id}": cnt / total_a * 100}, commit=False)

                    self.action_counts = {}

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
                    obs, _ = self.env.reset(options=self._build_reset_options())

        except KeyboardInterrupt:
            self.log("Training interrupted by user.")
        except Exception as e:
            self.log(f"Training crashed with error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.log("Closing environment and EXPORTING models...")
            try:
                self.save_pytorch_models(runs_dir)
                self.export_onnx_models(runs_dir)
            except Exception as e:
                self.log(f"Final export failed: {e}")

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
        self.top.geometry("650x820")
        
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

        # --- ALGORITHM & REWARD MODE ---
        frame_algo = tk.LabelFrame(self.top, text="Algorithm & Reward", padx=10, pady=5)
        frame_algo.pack(fill="x", padx=10, pady=5)
        frame_algo.columnconfigure(1, weight=1)
        frame_algo.columnconfigure(3, weight=1)

        tk.Label(frame_algo, text="RL Algorithm:").grid(row=0, column=0, sticky="w")
        self.combo_algorithm = ttk.Combobox(frame_algo,
            values=list(SUPPORTED_ALGORITHMS.keys()),
            state="readonly", width=12)
        self.combo_algorithm.set("qrdqn")
        self.combo_algorithm.grid(row=0, column=1, padx=5, sticky="w")
        self.combo_algorithm.bind("<<ComboboxSelected>>", self._on_algorithm_changed)

        tk.Label(frame_algo, text="Reward Mode:").grid(row=0, column=2, sticky="w", padx=(15, 0))
        self.combo_reward_mode = ttk.Combobox(frame_algo,
            values=["speed_throughput", "halt_ratio", "co2_speedstd"],
            state="readonly", width=18)
        self.combo_reward_mode.set("speed_throughput")
        self.combo_reward_mode.grid(row=0, column=3, padx=5, sticky="w")

        # Algoritmus info label
        self.lbl_algo_info = tk.Label(frame_algo, text="QRDQN — Off-policy, distributional DQN (replay buffer)",
                                       font=("Arial", 8), fg="gray")
        self.lbl_algo_info.grid(row=1, column=0, columnspan=4, sticky="w", pady=(2, 0))

        # Reward info label
        self.lbl_reward_info = tk.Label(frame_algo,
            text="speed_throughput — AvgSpeed + Throughput, log-tanh (η²=0.120, legjobb)",
            font=("Arial", 8), fg="gray")
        self.lbl_reward_info.grid(row=2, column=0, columnspan=4, sticky="w")
        self.combo_reward_mode.bind("<<ComboboxSelected>>", self._on_reward_mode_changed)

        # --- JUNCTION SELECTION ---
        frame_single = tk.LabelFrame(self.top, text="Junction Selection", padx=10, pady=5)
        frame_single.pack(fill="x", padx=10, pady=5)

        self.var_single_enabled = tk.BooleanVar(value=False)
        chk_single = tk.Checkbutton(frame_single, text="Train ONLY one junction",
                                     variable=self.var_single_enabled,
                                     command=self.toggle_single_agent_ui)
        chk_single.grid(row=0, column=0, sticky="w")

        tk.Label(frame_single, text="Junction ID:").grid(row=0, column=1, padx=10)
        self.combo_agent = ttk.Combobox(frame_single,
            values=getattr(self, 'available_junctions', []),
            state="disabled", width=15)
        self.combo_agent.grid(row=0, column=2, sticky="ew")

        # --- SUMO GUI CHECKBOX ---
        self.var_gui_enabled = tk.BooleanVar(value=False)
        chk_gui = tk.Checkbutton(frame_single, text="Enable SUMO GUI", variable=self.var_gui_enabled)
        chk_gui.grid(row=1, column=0, sticky="w", pady=5)

        # --- FIX FORGALOM ---
        frame_flow = tk.LabelFrame(self.top, text="Traffic Generation", padx=10, pady=5)
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

    def _on_algorithm_changed(self, event=None):
        """Algoritmus kiválasztásakor info label frissítése + off/on-policy specifikus elemek."""
        algo = self.combo_algorithm.get()
        info_map = {
            'qrdqn': 'QRDQN — Off-policy, distributional DQN (replay buffer)',
            'dqn':   'DQN — Off-policy, vanilla DQN (replay buffer)',
            'ppo':   'PPO — On-policy, policy gradient (rollout buffer, stabil)',
            'a2c':   'A2C — On-policy, advantage actor-critic (gyors, egyszerű)',
        }
        self.lbl_algo_info.config(text=info_map.get(algo, ''))

        # Off-policy specifikus mezők engedélyezése/tiltása
        algo_info = SUPPORTED_ALGORITHMS.get(algo, {})
        is_off_policy = algo_info.get('type') == 'off_policy'
        state_off = "normal" if is_off_policy else "disabled"
        self.entry_buffer.config(state=state_off)
        self.entry_expl.config(state=state_off)

        # On-policy figyelmeztetés multi-agent módban
        if not is_off_policy and not self.var_single_enabled.get():
            self.lbl_algo_info.config(
                text=info_map.get(algo, '') + '  ⚠ Csak single-junction módban!',
                fg='red')
        else:
            self.lbl_algo_info.config(fg='gray')

    def _on_reward_mode_changed(self, event=None):
        """Reward mód kiválasztásakor info label frissítése."""
        mode = self.combo_reward_mode.get()
        info_map = {
            'speed_throughput': 'AvgSpeed + Throughput, log-tanh (η²=0.120, legjobb)',
            'halt_ratio':       'HaltRatio, log-tanh (η²=0.156, legrobusztusabb)',
            'co2_speedstd':     'TotalCO2 + SpeedStd, log-tanh (η²=0.224)',
        }
        self.lbl_reward_info.config(text=f"{mode} — {info_map.get(mode, '')}")

    def get_settings(self):
        return {
            "total_timesteps": int(self.entry_steps.get()),
            "wandb_project": self.entry_project.get(),
            "wandb_api_key": self.entry_apikey.get(),
            "learning_rate": float(self.entry_lr.get()),
            "batch_size": int(self.entry_batch.get()),
            "buffer_size": int(self.entry_buffer.get()) if self.entry_buffer.cget("state") != "disabled" else 10000,
            "gamma": float(self.entry_gamma.get()),
            "exploration_fraction": float(self.entry_expl.get()) if self.entry_expl.cget("state") != "disabled" else 0.5,
            "num_layers": int(self.entry_num_layers.get()),
            "layer_size": int(self.entry_layer_size.get()),
            "algorithm": self.combo_algorithm.get(),
            "reward_mode": self.combo_reward_mode.get(),
            "single_agent_id": self.combo_agent.get() if self.var_single_enabled.get() else None,
            "sumo_gui": self.var_gui_enabled.get(),
            "fixed_flow": self.var_fixed_flow.get(),
            "flow_target": int(self.entry_flow_target.get()) if self.var_fixed_flow.get() else None,
            "flow_spread": int(self.entry_flow_spread.get()) if self.var_fixed_flow.get() else None,
        }

    def toggle_single_agent_ui(self):
        if self.var_single_enabled.get():
            self.combo_agent.config(state="readonly")
            if self.available_junctions:
                self.combo_agent.current(0)
        else:
            self.combo_agent.config(state="disabled")
            self.combo_agent.set("")

    def toggle_fixed_flow(self):
        state = "normal" if self.var_fixed_flow.get() else "disabled"
        self.entry_flow_target.config(state=state)
        self.entry_flow_spread.config(state=state)

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
                "--algorithm", settings["algorithm"],
                "--reward-mode", settings["reward_mode"],
                # Hiperparaméterek
                "--learning-rate", str(settings["learning_rate"]),
                "--batch-size", str(settings["batch_size"]),
                "--buffer-size", str(settings["buffer_size"]),
                "--gamma", str(settings["gamma"]),
                "--exploration-fraction", str(settings["exploration_fraction"]),
                "--num-layers", str(settings["num_layers"]),
                "--layer-size", str(settings["layer_size"]),
            ]

            if settings["sumo_gui"]:
                cmd.append("--gui")

            if settings["single_agent_id"]:
                cmd.extend(["--junction", settings["single_agent_id"]])

            if settings.get("fixed_flow") and settings.get("flow_target") is not None:
                cmd.extend(["--flow-target", str(settings["flow_target"])])
                cmd.extend(["--flow-spread", str(settings.get("flow_spread", 0))])

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
            self.log(f"  Algoritmus: {settings['algorithm']} | Reward: {settings['reward_mode']}")
            self.log(f"  Junction: {settings.get('single_agent_id') or 'ALL'}")
            self.log(f"  SUMO: {sumo_engine}")
            self.log(f"  Parancs: {' '.join(cmd)}")

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