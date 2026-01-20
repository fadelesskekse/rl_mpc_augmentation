# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym
import torch
from tensordict import TensorDict

#from rsl_rl.env import VecEnv

from isaaclab_rl.rsl_rl.vecenv_wrapper import RslRlVecEnvWrapper

from isaaclab.envs import DirectRLEnv, ManagerBasedRLEnv


class RslRlVecEnvWrapperCustom(RslRlVecEnvWrapper):

    def __init__(self, env: ManagerBasedRLEnv | DirectRLEnv, clip_actions: float | None = None):

        
        """Initializes the wrapper.

        Note:
            The wrapper calls :meth:`reset` at the start since the RSL-RL runner does not call reset.

        Args:
            env: The environment to wrap around.
            clip_actions: The clipping value for actions. If ``None``, then no clipping is done.

        Raises:
            ValueError: When the environment is not an instance of :class:`ManagerBasedRLEnv` or :class:`DirectRLEnv`.
        """

        # # check that input is valid
        # if not isinstance(env.unwrapped, ManagerBasedRLEnv) and not isinstance(env.unwrapped, DirectRLEnv):
        #     raise ValueError(
        #         "The environment must be inherited from ManagerBasedRLEnv or DirectRLEnv. Environment type:"
        #         f" {type(env)}"
        #     )

        # initialize the wrapper
        self.env = env
        self.clip_actions = clip_actions

        # store information required by wrapper
        self.num_envs = self.unwrapped.num_envs
        self.device = self.unwrapped.device
        self.max_episode_length = self.unwrapped.max_episode_length

        # obtain dimensions of the environment
        if hasattr(self.unwrapped, "action_manager"):
            self.num_actions = self.unwrapped.action_manager.total_action_dim
        else:
            self.num_actions = gym.spaces.flatdim(self.unwrapped.single_action_space)

        # modify the action space to the clip range
        self._modify_action_space()

        # reset at the start since the RSL-RL runner does not call reset
        self.env.reset()