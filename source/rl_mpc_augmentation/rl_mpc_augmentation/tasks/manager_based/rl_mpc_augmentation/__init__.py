# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

TASK_IDS = [

    "blind_rl",
    "phase_1_scan_rl",
]

ENTRY_POINTS = [

    #ppo rsl_rl with blind_rl config
    {
        "env_cfg_entry_point": f"{__name__}.blind_rl_cfg:RlMpcAugmentationEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
        "play_env_cfg_entry_point": f"{__name__}.blind_rl_cfg:RobotPlayEnvCfg",
    },

    {
        "env_cfg_entry_point": f"{__name__}.blind_rl_cfg:RlMpcAugmentationEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfgCustom",
        "play_env_cfg_entry_point": f"{__name__}.blind_rl_cfg:RobotPlayEnvCfg",
    },

    
]

for i in range(len(TASK_IDS)):
    gym.register(
        id = TASK_IDS[i],
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs=ENTRY_POINTS[i]

    )

# gym.register(
#     id="Template-Rl-Mpc-Augmentation-v0",
#     entry_point="isaaclab.envs:ManagerBasedRLEnv",
#     disable_env_checker=True,
#     kwargs={
#         "env_cfg_entry_point": f"{__name__}.rl_mpc_augmentation_env_cfg:RlMpcAugmentationEnvCfg",
#         "play_env_cfg_entry_point": f"{__name__}.rl_mpc_augmentation_env_cfg:RobotPlayEnvCfg",
#         "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
#     },
# )