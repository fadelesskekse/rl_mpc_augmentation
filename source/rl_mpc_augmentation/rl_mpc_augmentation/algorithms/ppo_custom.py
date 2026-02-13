# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim
from itertools import chain

from rsl_rl.modules import ActorCritic
from rsl_rl.modules.rnd import RandomNetworkDistillation
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import string_to_callable
from ..modules.actor_critic_custom import ActorCriticRMA

class PPOCustom:
    """Proximal Policy Optimization algorithm (https://arxiv.org/abs/1707.06347)."""

    policy: ActorCriticRMA
    """The actor critic module."""

    def __init__(
        self,
        policy,
        estimator,
        estimator_paras,
        num_learning_epochs=5,
        num_mini_batches=4,
        clip_param=0.2,
        gamma=0.99,
        lam=0.95,
        value_loss_coef=1.0,
        entropy_coef=0.01,
        learning_rate=0.001,
        max_grad_norm=1.0,
        use_clipped_value_loss=True,
        schedule="adaptive",
        desired_kl=0.01,
        device="cpu",
        dagger_update_freq=20,
        priv_reg_coef_schedual = [0, 0, 0],
        normalize_advantage_per_mini_batch=False,
        # RND parameters
        rnd_cfg: dict | None = None,
        # Symmetry parameters
        symmetry_cfg: dict | None = None,
        # Distributed training parameters
        multi_gpu_cfg: dict | None = None,
        **kwargs,
    ):
       
        # device-related parameters
        self.device = device
        self.is_multi_gpu = multi_gpu_cfg is not None
        # Multi-GPU parameters
        if multi_gpu_cfg is not None:
            self.gpu_global_rank = multi_gpu_cfg["global_rank"]
            self.gpu_world_size = multi_gpu_cfg["world_size"]
        else:
            self.gpu_global_rank = 0
            self.gpu_world_size = 1

        # RND components
        if rnd_cfg is not None:
            # Extract parameters used in ppo
            rnd_lr = rnd_cfg.pop("learning_rate", 1e-3)
            # Create RND module
            self.rnd = RandomNetworkDistillation(device=self.device, **rnd_cfg)
            # Create RND optimizer
            params = self.rnd.predictor.parameters()
            self.rnd_optimizer = optim.Adam(params, lr=rnd_lr)
        else:
            self.rnd = None
            self.rnd_optimizer = None

        # Symmetry components
        if symmetry_cfg is not None:
            # Check if symmetry is enabled
            use_symmetry = symmetry_cfg["use_data_augmentation"] or symmetry_cfg["use_mirror_loss"]
            # Print that we are not using symmetry
            if not use_symmetry:
                print("Symmetry not used for learning. We will use it for logging instead.")
            # If function is a string then resolve it to a function
            if isinstance(symmetry_cfg["data_augmentation_func"], str):
                symmetry_cfg["data_augmentation_func"] = string_to_callable(symmetry_cfg["data_augmentation_func"])
            # Check valid configuration
            if symmetry_cfg["use_data_augmentation"] and not callable(symmetry_cfg["data_augmentation_func"]):
                raise ValueError(
                    "Data augmentation enabled but the function is not callable:"
                    f" {symmetry_cfg['data_augmentation_func']}"
                )
            # Store symmetry configuration
            self.symmetry = symmetry_cfg
        else:
            self.symmetry = None

        # PPO components
        self.policy = policy
        self.policy.to(self.device)
        # Create optimizer
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        # Create rollout storage
        self.storage: RolloutStorage = None  # type: ignore
        self.transition = RolloutStorage.Transition()

        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate
        self.normalize_advantage_per_mini_batch = normalize_advantage_per_mini_batch

        #Adaptation
        self.hist_encoder_optimizer = optim.Adam(self.policy.actor.history_encoder.parameters(), lr=learning_rate)
        self.priv_reg_coef_schedual = priv_reg_coef_schedual
        self.counter = 0
        

        #Estimator
        self.estimator = estimator
        self.priv_states_dim = estimator_paras["priv_states_dim"]
        self.num_prop = estimator_paras["num_prop"] #size of one time steps worth of pos and vel joint data
        self.num_scan = estimator_paras["num_scan"]
        self.num_priv_latent = estimator_paras["num_priv_latent"]
        self.history_len = estimator_paras["history_len"]
        
        self.estimator_optimizer = optim.Adam(self.estimator.parameters(), lr=estimator_paras["learning_rate"])
        self.train_with_estimated_states = estimator_paras["train_with_estimated_states"]
       # self.train_with_estimated_states = False



    def init_storage(self, training_type, num_envs, num_transitions_per_env, obs, actions_shape):
        # create rollout storage
        self.storage = RolloutStorage(
            training_type,
            num_envs,
            num_transitions_per_env,
            obs,
            actions_shape,
            self.device,
        )

    # def act(self, obs):
    #     if self.policy.is_recurrent:
    #         self.transition.hidden_states = self.policy.get_hidden_states()
    #     # compute the actions and values
    #    # print("policy actually selecting actions")
    #     self.transition.actions = self.policy.act(obs).detach()
    #     self.transition.values = self.policy.evaluate(obs).detach()
    #     self.transition.actions_log_prob = self.policy.get_actions_log_prob(self.transition.actions).detach()
    #     self.transition.action_mean = self.policy.action_mean.detach()
    #     self.transition.action_sigma = self.policy.action_std.detach()
    #     # need to record obs before env.step()
    #     self.transition.observations = obs
    #     return self.transition.actions

    def act(self, obs,hist_encoding=False):

        if self.policy.is_recurrent:
            self.transition.hidden_states = self.policy.get_hidden_states()
        # compute the actions and values
       # print("I am taking actions during rollouts")

       

        if self.train_with_estimated_states:
            obs_est = obs.clone()

            actor_obs = obs_est.get("policy")
  
            base = self.num_scan + self.priv_states_dim + self.num_priv_latent
            half_prop = self.num_prop // 2

            hist_offset = (self.history_len - 1) * half_prop

           # print(f"actor_obs size: {actor_obs.shape}")
           # print(f"actor_obs: {actor_obs}")

            part1 = actor_obs[:, base + hist_offset: base + hist_offset+ half_prop]

           # print(f"size of part1:{part1.shape}")
           # print(f"part1: {part1}")

            part2 = actor_obs[:, base + self.history_len*half_prop + hist_offset : base + self.history_len*half_prop + hist_offset + half_prop]

            #print(f"size of part2:{part2.shape}")
            #print(f"part2: {part2}")

           

            estimator_input = torch.cat([part1, part2], dim=1)
            #print(f"forward passing estimator obs: {estimator_input}")
            priv_states_estimated = self.estimator(estimator_input)

           # print(f"estimator_input shape: {estimator_input.shape}")

            #print(f"estimator_output shape: {priv_states_estimated.shape}")

            #print(f"current velocity reading in ppo custom: {actor_obs[:, self.num_scan: self.num_scan + self.priv_states_dim]}")

         
            actor_obs[:, self.num_scan: self.num_scan + self.priv_states_dim] = priv_states_estimated

            #print(f"velocity reading in ppo custom after estimated updated: {actor_obs[:, self.num_scan: self.num_scan + self.priv_states_dim]}")


            obs_est["policy"] = actor_obs

           # print("Rollout acting")
            #print(f"obs size in rollout act: {obs_est.shape}")
            self.transition.actions = self.policy.act(obs_est, hist_encoding).detach()

            #Note: In current RSL_RL, we have 'policy' and 'critic' keys in obs when its passed to PPO.act. When we grab the proprio for the estimated
            #linear velocity calculation, we need to grab the policy version. 


        else:
            self.transition.actions = self.policy.act(obs, hist_encoding).detach()

      
        #self.transition.actions = self.policy.act(obs).detach()
        self.transition.values = self.policy.evaluate(obs).detach()
        self.transition.actions_log_prob = self.policy.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.policy.action_mean.detach()
        self.transition.action_sigma = self.policy.action_std.detach()
   
        self.transition.observations = obs
        return self.transition.actions

    def process_env_step(self, obs, rewards, dones, extras):
        # update the normalizers
        self.policy.update_normalization(obs)
        if self.rnd:
            self.rnd.update_normalization(obs)

        # Record the rewards and dones
        # Note: we clone here because later on we bootstrap the rewards based on timeouts
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones

        # Compute the intrinsic rewards and add to extrinsic rewards
        if self.rnd:
            # Compute the intrinsic rewards
            self.intrinsic_rewards = self.rnd.get_intrinsic_reward(obs)
            # Add intrinsic rewards to extrinsic rewards
            self.transition.rewards += self.intrinsic_rewards

        # Bootstrapping on time outs
        if "time_outs" in extras:
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values * extras["time_outs"].unsqueeze(1).to(self.device), 1
            )

        # record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.policy.reset(dones)

    def compute_returns(self, obs):
        # compute value for the last step
        last_values = self.policy.evaluate(obs).detach()
        self.storage.compute_returns(
            last_values, self.gamma, self.lam, normalize_advantage=not self.normalize_advantage_per_mini_batch
        )

    def update_counter(self):
        self.counter += 1

    def update(self):  # noqa: C901
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_entropy = 0
        mean_estimator_loss = 0
        mean_priv_reg_loss = 0

        mean_value_batch =0
        mean_returns_batch  = 0 
        mean_target_value_batch = 0
        mean_value_obs_batch = 0
        mean_value_clipped_batch = 0



        mean_advantages_batch = 0
        mean_ratio = 0
        mean_actions_log_prob_batch = 0
        mean_old_actions_log_prob_batch = 0
        mean_obs_batch= 0
        mean_actions_batch = 0


        

        


        # -- RND loss
        if self.rnd:
            mean_rnd_loss = 0
        else:
            mean_rnd_loss = None
        # -- Symmetry loss
        if self.symmetry:
            mean_symmetry_loss = 0
        else:
            mean_symmetry_loss = None

        # generator for mini batches
        if self.policy.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        grad_update_idx = 0
        grad_stats = {}

        # iterate over batches
        for (
            obs_batch,
            actions_batch,
            target_values_batch,
            advantages_batch,
            returns_batch,
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
            hid_states_batch,
            masks_batch,
        ) in generator:

            # number of augmentations per sample
            # we start with 1 and increase it if we use symmetry augmentation
            num_aug = 1
            # original batch size
            # we assume policy group is always there and needs augmentation
            original_batch_size = obs_batch.batch_size[0]

            # check if we should normalize advantages per mini batch
            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1e-8)

            # Perform symmetric augmentation
            if self.symmetry and self.symmetry["use_data_augmentation"]:
                # augmentation using symmetry
                data_augmentation_func = self.symmetry["data_augmentation_func"]
                # returned shape: [batch_size * num_aug, ...]
                obs_batch, actions_batch = data_augmentation_func(
                    obs=obs_batch,
                    actions=actions_batch,
                    env=self.symmetry["_env"],
                )
                # compute number of augmentations per sample
                # we assume policy group is always there and needs augmentation
                num_aug = int(obs_batch.batch_size[0] / original_batch_size)
                # repeat the rest of the batch
                # -- actor
                old_actions_log_prob_batch = old_actions_log_prob_batch.repeat(num_aug, 1)
                # -- critic
                target_values_batch = target_values_batch.repeat(num_aug, 1)
                advantages_batch = advantages_batch.repeat(num_aug, 1)
                returns_batch = returns_batch.repeat(num_aug, 1)

            # Recompute actions log prob and entropy for current batch of transitions
            # Note: we need to do this because we updated the policy with the new parameters
            # -- actor
           # print("I am updating")
           
           # print("calling policy act which will not call infer_priv_latent.")
            self.policy.act(obs_batch, masks=masks_batch, hidden_states=hid_states_batch[0])
            actions_log_prob_batch = self.policy.get_actions_log_prob(actions_batch)
            # -- critic
            value_batch = self.policy.evaluate(obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1])

           # print(f"value_batch shape: {value_batch.shape}")
            #for terms in obs_batch[]
            #print(f"obs_batch shape: {obs_batch}")
           # print("update PPO for loop iteration")


           
            # -- entropy
            # we only keep the entropy of the first augmentation (the original one)
            mu_batch = self.policy.action_mean[:original_batch_size]
            sigma_batch = self.policy.action_std[:original_batch_size]
            entropy_batch = self.policy.entropy[:original_batch_size]

            actor_obs = self.policy.get_actor_obs(obs_batch)
            # Adaptation module update
          #  print("directly calling infer priv_latent in ppo update")
           # print(f"size of obs_batch passed to direct call of infer priv latent: {actor_obs.shape}")
            priv_latent_batch = self.policy.actor.infer_priv_latent(actor_obs) #this call isn't working
            with torch.inference_mode():
                hist_latent_batch = self.policy.actor.infer_hist_latent(actor_obs)
            priv_reg_loss = (priv_latent_batch - hist_latent_batch.detach()).norm(p=2, dim=1).mean()
            priv_reg_stage = min(max((self.counter - self.priv_reg_coef_schedual[2]), 0) / self.priv_reg_coef_schedual[3], 1)
            priv_reg_coef = priv_reg_stage * (self.priv_reg_coef_schedual[1] - self.priv_reg_coef_schedual[0]) + self.priv_reg_coef_schedual[0]
            #coefficient goes from 0 to .1

            #priv_reg_coef_schedual = [0, 0.1, 2000, 3000],
            #priv_reg_coef_schedual_resume = [0, 0.1, 0, 1],

            # Estimator
            #priv_states_predicted = self.estimator(obs_batch[:, :self.num_prop])  # obs in batch is with true priv_states
            base = self.num_scan + self.priv_states_dim + self.num_priv_latent
            half_prop = self.num_prop // 2
            hist_offset = (self.history_len - 1) * half_prop
            part1 = actor_obs[:, base + hist_offset: base + hist_offset+ half_prop]
            part2 = actor_obs[:, base + self.history_len*half_prop + hist_offset : base + self.history_len*half_prop + hist_offset + half_prop]
            estimator_input = torch.cat([part1, part2], dim=1)
            priv_states_predicted = self.estimator(estimator_input) 
            estimator_loss = (priv_states_predicted - actor_obs[:, self.num_scan: self.num_scan + self.priv_states_dim]).pow(2).mean()
            #estimator_loss = (priv_states_predicted - obs_batch[:, self.num_prop+self.num_scan:self.num_prop+self.num_scan+self.priv_states_dim]).pow(2).mean()
            self.estimator_optimizer.zero_grad()
            estimator_loss.backward()
            nn.utils.clip_grad_norm_(self.estimator.parameters(), self.max_grad_norm)
            self.estimator_optimizer.step()



            # KL
            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                        + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
                        / (2.0 * torch.square(sigma_batch))
                        - 0.5,
                        axis=-1,
                    )
                    kl_mean = torch.mean(kl)

                    # Reduce the KL divergence across all GPUs
                    if self.is_multi_gpu:
                        torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                        kl_mean /= self.gpu_world_size

                    # Update the learning rate
                    # Perform this adaptation only on the main process
                    # TODO: Is this needed? If KL-divergence is the "same" across all GPUs,
                    #       then the learning rate should be the same across all GPUs.
                    if self.gpu_global_rank == 0:
                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    # Update the learning rate for all GPUs
                    if self.is_multi_gpu:
                        lr_tensor = torch.tensor(self.learning_rate, device=self.device)
                        torch.distributed.broadcast(lr_tensor, src=0)
                        self.learning_rate = lr_tensor.item()

                    # Update the learning rate for all parameter groups
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            # Surrogate loss
            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # Value function loss
            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                    -self.clip_param, self.clip_param
                )
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            

            #loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()
            loss = surrogate_loss + \
                self.value_loss_coef * value_loss - \
                self.entropy_coef * entropy_batch.mean() + \
                priv_reg_coef*priv_reg_loss
            
            #print(f"priv_reg_coef: {priv_reg_coef}")

            # Symmetry loss
            if self.symmetry:
                # obtain the symmetric actions
                # if we did augmentation before then we don't need to augment again
                if not self.symmetry["use_data_augmentation"]:
                    data_augmentation_func = self.symmetry["data_augmentation_func"]
                    obs_batch, _ = data_augmentation_func(obs=obs_batch, actions=None, env=self.symmetry["_env"])
                    # compute number of augmentations per sample
                    num_aug = int(obs_batch.shape[0] / original_batch_size)

                # actions predicted by the actor for symmetrically-augmented observations
                mean_actions_batch = self.policy.act_inference(obs_batch.detach().clone())

                # compute the symmetrically augmented actions
                # note: we are assuming the first augmentation is the original one.
                #   We do not use the action_batch from earlier since that action was sampled from the distribution.
                #   However, the symmetry loss is computed using the mean of the distribution.
                action_mean_orig = mean_actions_batch[:original_batch_size]
                _, actions_mean_symm_batch = data_augmentation_func(
                    obs=None, actions=action_mean_orig, env=self.symmetry["_env"]
                )

                # compute the loss (we skip the first augmentation as it is the original one)
                mse_loss = torch.nn.MSELoss()
                symmetry_loss = mse_loss(
                    mean_actions_batch[original_batch_size:], actions_mean_symm_batch.detach()[original_batch_size:]
                )
                # add the loss to the total loss
                if self.symmetry["use_mirror_loss"]:
                    loss += self.symmetry["mirror_loss_coeff"] * symmetry_loss
                else:
                    symmetry_loss = symmetry_loss.detach()

            # Random Network Distillation loss
            # TODO: Move this processing to inside RND module.
            if self.rnd:
                # extract the rnd_state
                # TODO: Check if we still need torch no grad. It is just an affine transformation.
                with torch.no_grad():
                    rnd_state_batch = self.rnd.get_rnd_state(obs_batch[:original_batch_size])
                    rnd_state_batch = self.rnd.state_normalizer(rnd_state_batch)
                # predict the embedding and the target
                predicted_embedding = self.rnd.predictor(rnd_state_batch)
                target_embedding = self.rnd.target(rnd_state_batch).detach()
                # compute the loss as the mean squared error
                mseloss = torch.nn.MSELoss()
                rnd_loss = mseloss(predicted_embedding, target_embedding)

            # Compute the gradients
            # -- For PPO
            self.optimizer.zero_grad()
            loss.backward()
            # -- For RND
            if self.rnd:
                self.rnd_optimizer.zero_grad()  # type: ignore
                rnd_loss.backward()

            # Collect gradients from all GPUs
            if self.is_multi_gpu:
                self.reduce_parameters()

            
            if self.policy.log_std.grad is not None:
                grad = self.policy.log_std.grad
                #print("log_std.grad shape:", grad.shape)
               # print("log_std.grad:", grad)


            actor_grads = []

            for param in self.policy.actor.parameters():
                if param.grad is not None:
                    actor_grads.append(param.grad.view(-1))

            if actor_grads:
                g_actor = torch.cat(actor_grads)
                actor_grad_avg = g_actor.abs().mean().item()
            else:
                actor_grad_avg = 0.0

            #print(f"actor_grads:{actor_grads}")

            critic_grads = []

            for param in self.policy.critic.parameters():
                if param.grad is not None:
                    critic_grads.append(param.grad.view(-1))

            if critic_grads:
                g_critic = torch.cat(critic_grads)
                critic_grad_avg = g_critic.abs().mean().item()
            else:
                critic_grad_avg = 0.0

           # print(f"critic_grads:{critic_grads}")

            grad_stats[f"grad_actor_avg_{grad_update_idx}"] = actor_grad_avg
            grad_stats[f"grad_critic_avg_{grad_update_idx}"] = critic_grad_avg
            grad_update_idx += 1


            # Apply the gradients
            # -- For PPO
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()
            # -- For RND
            if self.rnd_optimizer:
                self.rnd_optimizer.step()

            # Store the losses
            mean_advantages_batch += advantages_batch.mean().item()
            mean_ratio += ratio.mean().item()
            mean_actions_log_prob_batch += actions_log_prob_batch.mean().item()
            mean_old_actions_log_prob_batch += old_actions_log_prob_batch.mean().item()
            mean_value_batch += value_batch.mean().item()
            mean_value_clipped_batch += value_clipped.mean().item()
            mean_returns_batch += returns_batch.mean().item()
            mean_target_value_batch += target_values_batch.mean().item()

           
            mean_obs_batch += obs_batch["policy"].mean().item()
            mean_value_obs_batch += obs_batch["critic"].mean().item()
            mean_actions_batch += actions_batch.mean().item()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy_batch.mean().item()
            mean_priv_reg_loss += priv_reg_loss.item()
            # -- RND loss
            if mean_rnd_loss is not None:
                mean_rnd_loss += rnd_loss.item()
            # -- Symmetry loss
            if mean_symmetry_loss is not None:
                mean_symmetry_loss += symmetry_loss.item()

       

        # -- For PPO
        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_value_batch /= num_updates
        mean_returns_batch /= num_updates
        mean_target_value_batch /= num_updates
        mean_advantages_batch /= num_updates
        mean_ratio /= num_updates
        mean_actions_log_prob_batch /= num_updates
        mean_old_actions_log_prob_batch /= num_updates
        mean_obs_batch /= num_updates
        mean_actions_batch /= num_updates
        mean_value_obs_batch /= num_updates
        mean_value_clipped_batch /= num_updates



        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates

        mean_estimator_loss /= num_updates
        mean_priv_reg_loss /= num_updates

        
        # -- For RND
        if mean_rnd_loss is not None:
            mean_rnd_loss /= num_updates
        # -- For Symmetry
        if mean_symmetry_loss is not None:
            mean_symmetry_loss /= num_updates
        # -- Clear the storage

        
        self.print_debug()
        if mean_value_loss >= .5:
            raise ValueError(f"Value Loss is above .5, something is happening. Value loss is: {mean_value_loss}")

        self.storage.clear()


        #         mean_value_batch =0
        # mean_returns_batch  = 0 
        # mean_target_value_batch = 0



        # mean_advantages_batch = 0
        # mean_ratio = 0
        # mean_actions_log_prob_batch = 0
        # mean_old_actions_log_prob_batch = 0
        # mean_obs_batch= 0
        # mean_actions_batch = 0

        # construct the loss dictionary
        loss_dict = {
            "value_function": mean_value_loss,

            "value_batch":   mean_value_batch,
            "returns_batch":mean_returns_batch,
            "target_value_batch": mean_target_value_batch,
            "mean_advantages_batch":mean_advantages_batch,
            "mean_ratio":mean_ratio,
            "mean_actions_log_prob_batch":mean_actions_log_prob_batch ,
            "mean_old_actions_log_prob_batch":mean_old_actions_log_prob_batch,
            "mean_obs_batch":mean_obs_batch,
            "mean_actions_batch":mean_actions_batch,
            "mean_value_obs_batch":mean_value_obs_batch,
            "mean_value_clipped_batch": mean_value_clipped_batch,



            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
            "estimator": estimator_loss,
            "priv_reg": mean_priv_reg_loss,
            "kl_mean": kl_mean
        }
        if self.rnd:
            loss_dict["rnd"] = mean_rnd_loss
        if self.symmetry:
            loss_dict["symmetry"] = mean_symmetry_loss

        # ---- Add gradient statistics ----
        loss_dict.update(grad_stats)
        self.update_counter()

        #if self.counter == 7105:
            #mean_value_loss =1
            #print(f"value loss: {mean_value_loss}")
       # print(f"counter: {self.counter}")
      
        return loss_dict
        
    def print_debug(self):

        observations = self.storage.observations.flatten(0, 1)
        actions = self.storage.actions.flatten(0, 1)
        values = self.storage.values.flatten(0, 1)
        returns = self.storage.returns.flatten(0, 1)

        actor_obs = observations["policy"]      # [39600, num_act_obs]
        critic_obs = observations["critic"]     # [39600, num_crit_obs]

        size = actor_obs.shape[0]

        print(f"\nTotal transitions: {size}")
        print(f"Actor obs dim: {actor_obs.shape[1]}")
        print(f"Critic obs dim: {critic_obs.shape[1]}")

        # --------------------------------------------------
        # Define observation structure (name, dimension)
        # --------------------------------------------------

        obs_structure = [
            ("scan_dot", 297),
            ("base_lin_vel", 3),
            ("priv_latent_gains_stiffness", 29),
            ("priv_latent_gains_damping", 29),
            ("priv_latent_mass", 1),
            ("priv_latent_com", 3),
            ("priv_latent_friction", 4),
            ("joint_pos_rel", 290),
            ("joint_vel_rel", 290),
            ("base_ang_vel", 15),
            ("projected_gravity", 15),
            ("velocity_commands", 15),
            ("last_action", 150),
            ("gait_phase", 10),
        ]

        # --------------------------------------------------
        # Helper function to parse and average
        # --------------------------------------------------

        def parse_and_average(obs_tensor, label):
            print(f"\n--- {label} ---")

            offset = 0
            results = {}

            for name, dim in obs_structure:
                slice_ = obs_tensor[:, offset:offset + dim]  # [39600, dim]
                mean_val = slice_.mean(dim=0)                # [dim]

                results[name] = mean_val

                print(f"{name:35s} | dim: {dim:4d} | mean(abs): {mean_val.abs().mean().item():.6f}")

                offset += dim

            return results

        # --------------------------------------------------
        # Run for actor and critic
        # --------------------------------------------------

        actor_means = parse_and_average(actor_obs, "ACTOR OBS")
        critic_means = parse_and_average(critic_obs, "CRITIC OBS")
        
        

    


    def update_dagger(self):
        mean_hist_latent_loss = 0
        if self.policy.is_recurrent:
            generator = self.storage.reccurent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        for obs_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, \
            old_mu_batch, old_sigma_batch, hid_states_batch, masks_batch in generator:
                
                actor_obs = self.policy.get_actor_obs(obs_batch)
   
                with torch.inference_mode():
                    self.policy.act(obs_batch, hist_encoding=True, masks=masks_batch, hidden_states=hid_states_batch[0])

                # Adaptation module update
                with torch.inference_mode():
                    priv_latent_batch = self.policy.actor.infer_priv_latent(actor_obs)
                hist_latent_batch = self.policy.actor.infer_hist_latent(actor_obs)
                hist_latent_loss = (priv_latent_batch.detach() - hist_latent_batch).norm(p=2, dim=1).mean()
                self.hist_encoder_optimizer.zero_grad()
                hist_latent_loss.backward()
                nn.utils.clip_grad_norm_(self.policy.actor.history_encoder.parameters(), self.max_grad_norm)
                self.hist_encoder_optimizer.step()
                
                mean_hist_latent_loss += hist_latent_loss.item()
        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_hist_latent_loss /= num_updates
        self.storage.clear()
        self.update_counter()
        return mean_hist_latent_loss

    """
    Helper functions
    """

    def broadcast_parameters(self):
        """Broadcast model parameters to all GPUs."""
        # obtain the model parameters on current GPU
        model_params = [self.policy.state_dict()]
        if self.rnd:
            model_params.append(self.rnd.predictor.state_dict())
        # broadcast the model parameters
        torch.distributed.broadcast_object_list(model_params, src=0)
        # load the model parameters on all GPUs from source GPU
        self.policy.load_state_dict(model_params[0])
        if self.rnd:
            self.rnd.predictor.load_state_dict(model_params[1])

    def reduce_parameters(self):
        """Collect gradients from all GPUs and average them.

        This function is called after the backward pass to synchronize the gradients across all GPUs.
        """
        # Create a tensor to store the gradients
        grads = [param.grad.view(-1) for param in self.policy.parameters() if param.grad is not None]
        if self.rnd:
            grads += [param.grad.view(-1) for param in self.rnd.parameters() if param.grad is not None]
        all_grads = torch.cat(grads)

        # Average the gradients across all GPUs
        torch.distributed.all_reduce(all_grads, op=torch.distributed.ReduceOp.SUM)
        all_grads /= self.gpu_world_size

        # Get all parameters
        all_params = self.policy.parameters()
        if self.rnd:
            all_params = chain(all_params, self.rnd.parameters())

        # Update the gradients for all parameters with the reduced gradients
        offset = 0
        for param in all_params:
            if param.grad is not None:
                numel = param.numel()
                # copy data back from shared buffer
                param.grad.data.copy_(all_grads[offset : offset + numel].view_as(param.grad.data))
                # update the offset for the next parameter
                offset += numel
