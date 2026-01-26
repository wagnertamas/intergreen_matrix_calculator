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
        
        # Paraméterek betöltése
        lr = float(current_config.get("learning_rate", 1e-4))
        bs = int(current_config.get("batch_size", 32))
        buf = int(current_config.get("buffer_size", 10000))
        gamma = float(current_config.get("gamma", 0.99))
        expl_fraction = float(current_config.get("exploration_fraction", 0.5))
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
                gamma=gamma,
                exploration_fraction=expl_fraction,
                policy_kwargs=dict(net_arch=net_arch),
                verbose=0,
                tensorboard_log=tb_log,
                device="auto"
            )
            self.agents[jid].set_logger(configure(tb_log, ["stdout", "tensorboard"]))

        # =========================================================================
        # 5. TANÍTÁSI CIKLUS (TRY-FINALLY BLOKKAL VÉDVE)
        # =========================================================================
        self.log(f"Starting Training Loop (Gamma={gamma}, Expl_Fraction={expl_fraction})...")
        global_step = 0
        start_time = time.time()
        
        try: # <--- KEZDŐDIK A BIZTONSÁGI BLOKK
            while global_step < self.total_timesteps and not self.stop_requested:
                
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
                
                # --- LOGGING ---
                if global_step % 100 == 0:
                    elapsed = time.time() - start_time
                    fps = int(global_step / (elapsed + 1e-5))
                    self.log(f"Step: {global_step}/{self.total_timesteps} | FPS: {fps}")
                    
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

                        wandb.log(log_dict, commit=True)

                if global_done:
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
                self.export_onnx_models(runs_dir)
            except Exception as e:
                self.log(f"Final export failed: {e}")

            # Aztán zárunk
            if self.env:
                self.env.close()
            
            if wandb.run:
                wandb.finish()
            
            self.log("Cleanup done.")