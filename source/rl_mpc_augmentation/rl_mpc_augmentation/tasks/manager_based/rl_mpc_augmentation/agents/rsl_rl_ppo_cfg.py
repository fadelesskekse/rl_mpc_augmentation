# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

from rl_mpc_augmentation.cfgs.cfgs import RslRlPpoAlgorithmCfgCustom,RslRlPpoActorCriticCfgCustom, EstimatorCfg

#from rl_mpc_augmentation.tasks.manager_based.rl_mpc_augmentation.blind_rl_cfg import RlMpcAugmentationEnvCfg

from rl_mpc_augmentation.algorithms.ppo_custom import PPOCustom

@configclass
class PPORunnerCfg(RslRlOnPolicyRunnerCfg):

    num_steps_per_env = 24
    max_iterations = 50000
    save_interval = 100
    experiment_name = "rl_mpc_augmentation"  # same as task name
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        noise_std_type="log",
        
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )

# PPORunnerCfg is what is used as rsl_rl_cfg_entry_point
#Which is what is the default for agent_cfg:parser.add_argument(
#     "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
# )
@configclass
class PPORunnerCfgCustom(RslRlOnPolicyRunnerCfg):

    # class_name = "OnPolicyRunnerCustom"
    # num_steps_per_env = 24
    # max_iterations = 50000
    # save_interval = 100
    # experiment_name = "rl_mpc_augmentation"  # same as task name
    # empirical_normalization = False
    # policy = RslRlPpoActorCriticCfg(
    #     init_noise_std=1,
    #     actor_hidden_dims=[512, 256, 128],
    #     critic_hidden_dims=[512, 256, 128],
    #     activation="elu",
    #     noise_std_type="log",
        
    # )
    # algorithm = RslRlPpoAlgorithmCfg(
    #     value_loss_coef=1.0,
    #     use_clipped_value_loss=True,
    #     clip_param=0.2,
    #     entropy_coef=0.01,
    #     num_learning_epochs=5,
    #     num_mini_batches=4,
    #     learning_rate=1.0e-3,
    #     schedule="adaptive",
    #     gamma=0.99,
    #     lam=0.95,
    #     desired_kl=0.01,
    #     max_grad_norm=1.0,
    # )

    class_name = "OnPolicyRunnerCustom"
    num_steps_per_env = 24
    max_iterations = 50000
    save_interval = 100
    experiment_name = "rl_mpc_augmentation"  # same as task name
    empirical_normalization = False

    policy = RslRlPpoActorCriticCfgCustom(
        class_name="ActorCriticRMA",
        init_noise_std=1,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        noise_std_type="log",

        scan_encoder_dims = [128, 64, 32],
        #priv_encoder_dims = [64, 20],
        priv_encoder_dims = [128,64, 32],
        # only for 'ActorCriticRecurrent':
        rnn_type = 'lstm',
        rnn_hidden_size = 512,
        rnn_num_layers = 1,
        tanh_encoder_output = False,
        
    )
    
    algorithm = RslRlPpoAlgorithmCfgCustom(
        class_name= "PPOCustom",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,#.01
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,#1.0e-3
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        # dagger params
        dagger_update_freq = 20,
        priv_reg_coef_schedual = [0, 0.1, 2000, 3000],#[0, 0.1, 2000, 3000],
        priv_reg_coef_schedual_resume = [0, 0.1, 0, 1],
    )


    estimator = EstimatorCfg(
        train_with_estimated_states = True,
        learning_rate = 1.e-3,
        hidden_dims = [128, 64],
    )

    #depth_encoder = 

 