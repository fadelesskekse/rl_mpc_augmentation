# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING


import isaaclab.utils.string as string_utils
from isaaclab.assets.articulation import Articulation
from isaaclab.managers.action_manager import ActionTerm

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from .pass_to_env_action_cfg import PassToEnvironmentCfg


class PassToEnvironment(ActionTerm):

    cfg: PassToEnvironmentCfg

    _clip: torch.Tensor
    #Applied to input action

    def __init__(self,cfg: PassToEnvironmentCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        self._var_names = self.cfg.var_names

        self._num_joints = self.cfg.num_vars
        # create tensors for raw and processed actions
        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._processed_actions = torch.zeros_like(self.raw_actions)
        
        #print(f"joint_names: {self._var_names}")


        # parse clip
        if self.cfg.clip is not None:
            if isinstance(cfg.clip, dict):
                self._clip = torch.tensor([[-float("inf"), float("inf")]], device=self.device).repeat(
                    self.num_envs, self.action_dim, 1
                )
                #If I have a list[str] with mpc_0,mpc_1,mpc_2 = var_names, and then clip also has
                #mpc_0,mpc_1,mpc_2, then resolve_matching_names_values() 

                #
                index_list, _, value_list = string_utils.resolve_matching_names_values(self.cfg.clip, self._var_names)
                self._clip[:, index_list] = torch.tensor(value_list, device=self.device)
            
                # print(f"index_list: {index_list}")
                # print(f"names: {names}")
                # print(f"value_list: {value_list}")
                # print(f"print the clip {self._clip}")
            
            else:
                raise ValueError(f"Unsupported clip type: {type(cfg.clip)}. Supported types are dict.")


    @property
    def action_dim(self) -> int:
        return self._num_joints

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions
    
    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions

        print(f"raw_actions: {self.raw_actions}")

        if self.cfg.clip is not None:

            self._processed_actions = torch.clamp(
                self._raw_actions, min=self._clip[:,:,0],
                max=self._clip[:,:,1] 
            )

        print(f"processed_actions: {self.processed_actions}")


    def apply_actions(self):
        pass

    