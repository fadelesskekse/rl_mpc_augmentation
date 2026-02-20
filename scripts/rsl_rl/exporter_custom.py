
import torch
import os
import copy

from isaaclab_rl.rsl_rl.exporter import _OnnxPolicyExporter

def export_policy_as_onnx_custom(
    policy: object,  estimator_nn: object, num_scan, priv_states_dim, num_priv_latent, num_prop, history_len, path: str, normalizer: object | None = None, filename="policy.onnx", verbose=False
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
    policy_exporter = _OnnxPolicyExporterCustom(policy, estimator_nn, num_scan, priv_states_dim, num_priv_latent, num_prop, history_len, normalizer, verbose)
    policy_exporter.export(path, filename)



class _OnnxPolicyExporterCustom(_OnnxPolicyExporter):

    def __init__(self, policy, estimator, num_scan, priv_states_dim, num_priv_latent, num_prop, history_len, normalizer=None, verbose=False):
        # call parent constructor (this builds self.actor, self.normalizer, etc.)
        super().__init__(policy, normalizer=normalizer, verbose=verbose)
        self.estimator = copy.deepcopy(estimator)
        self.num_scan = num_scan
        self.priv_states_dim = priv_states_dim
        self.num_priv_latent = num_priv_latent
        self.num_prop = num_prop
        self.history_len = history_len


    def forward(self, x):
      #  print("Hi i am running once")

        
        obs_raw = x.clone()

        base = self.num_scan + self.priv_states_dim + self.num_priv_latent
        half_prop = self.num_prop // 2

        hist_offset = (self.history_len - 1) * half_prop

        # print(f"actor_obs size: {actor_obs.shape}")
        # print(f"actor_obs: {actor_obs}")

        part1 = obs_raw[:, base + hist_offset: base + hist_offset+ half_prop]

        # print(f"size of part1:{part1.shape}")
        # print(f"part1: {part1}")

        part2 = obs_raw[:, base + self.history_len*half_prop + hist_offset : base + self.history_len*half_prop + hist_offset + half_prop]

        #print(f"size of part2:{part2.shape}")
        #print(f"part2: {part2}")


        estimator_input = torch.cat([part1, part2], dim=1)
        #print(f"forward passing estimator obs: {estimator_input}")
        priv_states_estimated = self.estimator(estimator_input)

        left = obs_raw[:, :self.num_scan]
        right = obs_raw[:, self.num_scan + self.priv_states_dim:]

        obs_raw = torch.cat([left, priv_states_estimated, right], dim=1)

        #obs_raw[:, self.num_scan: self.num_scan + self.priv_states_dim] = priv_states_estimated

        #print(f"shape of obs in export: {x.shape}")


        return self.actor(self.normalizer(obs_raw),True)
    
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