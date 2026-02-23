
import torch
import os
import copy

from isaaclab_rl.rsl_rl.exporter import _OnnxPolicyExporter

def export_policy_as_onnx_custom(
    policy: object,  estimator_nn: object, num_scan, priv_states_dim, num_priv_latent, num_prop, history_len, num_actor_backbone_prop_hist,path: str, normalizer: object | None = None, filename="policy.onnx", verbose=False
):
    """Export policy into a Torch ONNX file.

    Args:
        policy: The policy torch module.
        normalizer: The empirical normalizer module. If None, Identity is used.
        path: The path to the saving directory.
        filename: The name of exported ONNX file. Defaults to "policy.onnx".
        verbose: Whether to print the model summary. Defaults to False.
    """
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    policy_exporter = _OnnxPolicyExporterCustom(policy, estimator_nn, num_scan, priv_states_dim, num_priv_latent, num_prop, history_len, num_actor_backbone_prop_hist, normalizer, verbose)
    policy_exporter.export(path, filename)



class _OnnxPolicyExporterCustom(_OnnxPolicyExporter):

    def __init__(self, policy, estimator, num_scan, priv_states_dim, num_priv_latent, num_prop, history_len, num_actor_backbone_prop_hist,normalizer=None, verbose=False):
        # call parent constructor (this builds self.actor, self.normalizer, etc.)
        super().__init__(policy, normalizer=normalizer, verbose=verbose)
        self.estimator = copy.deepcopy(estimator)
        self.actor_backbone = copy.deepcopy(policy.actor.actor_backbone)
        self.history_encoder = copy.deepcopy(policy.actor.history_encoder)
        self.num_scan = num_scan

        self.scan_encoder = None

        if self.num_scan != 0:
            self.scan_encoder = copy.deepcopy(policy.actor.scan_encoder)

        self.priv_states_dim = priv_states_dim
        self.num_priv_latent = num_priv_latent
        self.num_prop = num_prop
        self.history_len = history_len

        self.num_actor_backbone_prop_hist = num_actor_backbone_prop_hist


    def forward(self, x):

    #      scan_dot = ObsTerm(func=mdp.scan_dot, 
    #             scale = 1,
    #             params={
    #                 "sensor_cfg": SceneEntityCfg("scan_dot",),
    #                 #"asset_cfg": SceneEntityCfg("robot", body_names=".*torso_link.*")
    #             },
    #             history_length=0
    #     )



    #     base_lin_vel = ObsTerm(func=mdp.base_lin_vel, history_length=0) #Will be replaced by estimator output during rollouts, and will be used as ground truth during learning phase
        
    #     priv_latent_gains_stiffness = ObsTerm(func=mdp.priv_latent_gains_stiffness, history_length=0,scale=1,params={"scale_val": .2})
    #     priv_latent_gains_damping = ObsTerm(func=mdp.priv_latent_gains_damping, history_length=0,scale=1,params={"scale_val": .2})
    #     priv_latent_mass = ObsTerm(func=mdp.priv_latent_mass,params={"asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
    #                                                                  "scale_val": .2}, history_length=0,scale=1,)
    #     priv_latent_com = ObsTerm(func=mdp.priv_latent_com, history_length=0)
    #     priv_latent_friction= ObsTerm(func=mdp.priv_latent_friction, history_length=0)

    #    # priv_latent = ObsTerm(func=mdp.priv_latent, history_length=0)

    #     joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel,history_length=10, noise=Unoise(n_min=-0.01, n_max=0.01),) #updated in post init 
    #     joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, history_length=10, scale=0.05, noise=Unoise(n_min=-1.5, n_max=1.5),)

    #     ########END EXTREME PARKOUS OBS#################

    #     base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2, history_length = 5,noise=Unoise(n_min=-0.2, n_max=0.2),)
    #     projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05),history_length=5)
    #     velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"},history_length=5)
    #     last_action = ObsTerm(func=mdp.last_action,history_length=5)




     
        #For exported policy this is the order we need to match the history encoder inputs as well as the actor inputs:
        #1. Scan dots (num_scan)
        #2. Joint pos rel
        #3. Joint Vel rel
        #4. Everything else

        
        obs_raw = x.clone()

        non_extreme_parkour_obs = obs_raw[:,self.num_scan + self.history_len*self.num_prop:]

      
        scan_raw = obs_raw[:, :self.num_scan]

        scan_latent = None

        if self.num_scan !=0 and self.scan_encoder is not None:
            scan_latent = self.scan_encoder(scan_raw)

        else:
            scan_latent = scan_raw

        
        hist = obs_raw[:,self.num_scan: self.num_scan + self.history_len*self.num_prop]

        num_envs = hist.shape[0]
        H = self.history_len           # e.g. 10
        J = self.num_prop // 2      # 29 joints
        P = self.num_prop           # 58 = 29 pos + 29 vel

        # split into position and velocity blocks
        pos = hist[:, :H * J]              # (num_envs, 290)
        vel = hist[:, H * J : 2 * H * J]   # (num_envs, 290)

        # reshape into (env, time, joints)
        pos = pos.view(num_envs, H, J)     # (num_envs, 10, 29)
        vel = vel.view(num_envs, H, J)     # (num_envs, 10, 29)

        # interleave into (env, time, 58)
        hist_interleaved = torch.cat([pos, vel], dim=2)  # (num_envs, 10, 58)

        hist_latent =  self.history_encoder(hist_interleaved)
    
        base = self.num_scan
        half_prop = self.num_prop // 2
        hist_offset = (self.history_len - 1) * half_prop
        part1 = obs_raw[:, base + hist_offset: base + hist_offset+ half_prop]
        part2 = obs_raw[:, base + self.history_len*half_prop + hist_offset : base + self.history_len*half_prop + hist_offset + half_prop]
        estimator_input = torch.cat([part1, part2], dim=1)
        priv_states_estimated = self.estimator(estimator_input)
        #left = obs_raw[:, :self.num_scan]
        #right = obs_raw[:, self.num_scan + self.priv_states_dim:]

        hist_offset_reg_prop = (self.history_len - self.num_actor_backbone_prop_hist) * half_prop

        
        actor_backbone_prop_pos = obs_raw[:,base + hist_offset_reg_prop: base + hist_offset_reg_prop + half_prop*self.num_actor_backbone_prop_hist]

        actor_backbone_prop_vel = obs_raw[:,base + half_prop*self.history_len + hist_offset_reg_prop: base + half_prop*self.history_len + hist_offset_reg_prop +half_prop*self.num_actor_backbone_prop_hist]

    


        #obs_raw = torch.cat([left, priv_states_estimated, hist_latent, right], dim=1)
        #backbone_input = torch.cat([obs_scan_actor_backbone_input, obs_priv_explicit, latent,actor_backbone_prop_pos,actor_backbone_prop_vel,non_exr_pkr_obs], dim=1)
           

        obs_backbone = torch.cat([scan_latent, priv_states_estimated, hist_latent, actor_backbone_prop_pos, actor_backbone_prop_vel,non_extreme_parkour_obs],dim=1)
        # print(f"scan_latent: {scan_latent.shape}")
        # print(f"priv_states_estimated: {priv_states_estimated.shape}")
        # print(f"hist_latent: {hist_latent.shape}")
        # print(f"actor_backbone_prop_pos: {actor_backbone_prop_pos.shape}")
        # print(f"actor_backbone_prop_vel: {actor_backbone_prop_vel.shape}")
        # print(f"non_extreme_parkour_obs: {non_extreme_parkour_obs.shape}")
       
        

        # print(f"actor backbone: {obs_backbone.shape}")



    #    hist = obs[:,self.num_scan + self.num_priv_explicit + self.num_priv_latent: self.num_scan + self.num_priv_explicit + self.num_priv_latent + self.num_hist*self.num_prop]
        
    #     #print(f"hist: {hist}")
    #    # print(f"proprio hist: {hist}")

    #     num_envs = hist.shape[0]
    #     H = self.num_hist           # e.g. 10
    #     J = self.num_prop // 2      # 29 joints
    #     P = self.num_prop           # 58 = 29 pos + 29 vel

    #     # split into position and velocity blocks
    #     pos = hist[:, :H * J]              # (num_envs, 290)
    #     vel = hist[:, H * J : 2 * H * J]   # (num_envs, 290)

    #     # reshape into (env, time, joints)
    #     pos = pos.view(num_envs, H, J)     # (num_envs, 10, 29)
    #     vel = vel.view(num_envs, H, J)     # (num_envs, 10, 29)

    #     # interleave into (env, time, 58)
    #     hist_interleaved = torch.cat([pos, vel], dim=2)  # (num_envs, 10, 58)

    #     assert hist_interleaved.shape[1] == self.num_hist
    #     assert hist_interleaved.shape[2] == self.num_prop

    #     # check semantic alignment for env 0, t = 0
    #     #print("t0 pos:", hist_interleaved[0, self.num_hist - 1, :J])
    #    # print("t0 vel:", hist_interleaved[0, self.num_hist - 1, J:])

    #     # check semantic alignment for env 0, t = 0
    #    # print("t0_last pos:", hist_interleaved[0, 0, :J])
    #    # print("t0_last vel:", hist_interleaved[0, 0, J:])

    #    # print(f"after weaving hist: {hist_interleaved}")


    #     #print(f"shape of new: {hist_interleaved.shape}" )

    #     #print(f"shape of orignal: {hist.view(-1, self.num_hist, self.num_prop).shape}")

    #     #return self.history_encoder(hist.view(-1, self.num_hist, self.num_prop))

    #     return self.history_encoder(hist_interleaved)




        # ######################################################3
        # obs_raw = x.clone()
        # #print(f"obs_raw shape before esitmator call: {obs_raw.shape}")


        # base = self.num_scan + self.priv_states_dim + self.num_priv_latent
        # half_prop = self.num_prop // 2

        # hist_offset = (self.history_len - 1) * half_prop

        # # print(f"actor_obs size: {actor_obs.shape}")
        # # print(f"actor_obs: {actor_obs}")

        # part1 = obs_raw[:, base + hist_offset: base + hist_offset+ half_prop]

        # # print(f"size of part1:{part1.shape}")
        # # print(f"part1: {part1}")

        # part2 = obs_raw[:, base + self.history_len*half_prop + hist_offset : base + self.history_len*half_prop + hist_offset + half_prop]

        # #print(f"size of part2:{part2.shape}")
        # #print(f"part2: {part2}")


        # estimator_input = torch.cat([part1, part2], dim=1)
        # #print(f"forward passing estimator obs: {estimator_input}")
        # priv_states_estimated = self.estimator(estimator_input)

        # left = obs_raw[:, :self.num_scan]
        # right = obs_raw[:, self.num_scan + self.priv_states_dim:]

        # obs_raw = torch.cat([left, priv_states_estimated, right], dim=1)

        # print(f"obs_raw shape after esitmator call: {obs_raw.shape}")

        # #obs_raw[:, self.num_scan: self.num_scan + self.priv_states_dim] = priv_states_estimated

        # #print(f"shape of obs in export: {x.shape}")


        #return self.actor(self.normalizer(obs_raw),True)
        return self.actor_backbone(self.normalizer(obs_backbone))
    
    
    def export(self, path, filename):
        self.to("cpu")
        self.eval()
        if self.is_recurrent:
            obs = torch.zeros(1, self.rnn.input_size)
            h_in = torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size)

            if self.rnn_type == "lstm":
                c_in = torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size)
                torch.onnx.export(
                    self,
                    (obs, h_in, c_in),
                    os.path.join(path, filename),
                    export_params=True,
                    opset_version=11,
                    verbose=self.verbose,
                    input_names=["obs", "h_in", "c_in"],
                    output_names=["actions", "h_out", "c_out"],
                    dynamic_axes={},
                )
            elif self.rnn_type == "gru":
                torch.onnx.export(
                    self,
                    (obs, h_in),
                    os.path.join(path, filename),
                    export_params=True,
                    opset_version=11,
                    verbose=self.verbose,
                    input_names=["obs", "h_in"],
                    output_names=["actions", "h_out"],
                    dynamic_axes={},
                )
            else:
                raise NotImplementedError(f"Unsupported RNN type: {self.rnn_type}")
        else:
            #obs = torch.zeros(1, self.actor[0].in_features) #before this was an mlp
            # if self.blind and self.num_scan == 0:
            #     obs = torch.zeros(1, self.actor.blind_deploy_policy_input)
            # elif not self.blind and self.num_scan != 0:
            #     obs = torch.zeros(1, self.actor.scan_dot_deploy_policy_input)
            # else:
            #     raise ValueError("OnnxExporter Custom doesn't have the correct obs test size.")
            obs = torch.zeros(1, self.actor.total_raw_actor_inp_size)

            torch.onnx.export(
                self,
                obs,
                os.path.join(path, filename),
                export_params=True,
                opset_version=11,
                verbose=self.verbose,
                input_names=["obs"],
                output_names=["actions"],
                dynamic_axes={},
            )