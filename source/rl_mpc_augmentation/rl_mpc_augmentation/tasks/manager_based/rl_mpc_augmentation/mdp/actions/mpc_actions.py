# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import numpy as np
import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.assets.articulation import Articulation
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.envs import ManagerBasedEnv


from . import mpc_actions_cfg
from .robot_helper import RobotCore

class BlindMPCAction(ActionTerm):

    cfg: mpc_actions_cfg.BlindMPCActionCfg

    _asset: Articulation


    def __init__(self, cfg: mpc_actions_cfg.BlindMPCActionCfg, env: ManagerBasedEnv):
        # initialize the action term
        super().__init__(cfg, env)

        body_names = self._asset.data.body_names
        foot_idx = [i for i in range(len(body_names)) if body_names[i] in ["L_sole", "R_sole"]]
        self.robot_api = RobotCore(self._asset, torch.tensor(foot_idx, device=self.device, dtype=torch.long), self.num_envs, self.device)

        # resolve the joints over which the action term is applied
        self._joint_ids, self._joint_names = self._asset.find_joints(self.cfg.joint_names, preserve_order=True)
        self._num_joints = len(self._joint_ids)

        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._processed_actions = torch.zeros_like(self.raw_actions)
        self._joint_actions = np.zeros((self.num_envs, self._num_joints), dtype=np.float32)
    


    @property
    def action_dim(self) -> int:
        """
        mpc control parameters:
        - gait stepping frequency 
        - swing foot height 
        - swing trajectory control points
        """
        return 3

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions
    
    def process_actions(self, actions: torch.Tensor):
        #where I will take in policy output nad run mpc framework
        pass

    def apply_actions(self):
        #The actual actions applied to the robot
        pass

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        
        #Will reset things related to mpc when the environment needs to be reset. 
        pass