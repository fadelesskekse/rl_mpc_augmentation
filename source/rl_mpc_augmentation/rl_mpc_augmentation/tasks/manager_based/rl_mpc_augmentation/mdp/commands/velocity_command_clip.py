# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module containing command generators for the velocity-based locomotion task."""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from .velocity_command_cfg_clip import UniformLevelVelocityCommandCfgClip

from isaaclab.envs.mdp.commands.velocity_command import UniformVelocityCommand
#from .velocity_command_cfg_clip import UniformLevelVelocityCommandCfgClip

class UniformVelocityCommandClip(UniformVelocityCommand):
    
    cfg: UniformLevelVelocityCommandCfgClip

    def __init__(self,cfg: UniformLevelVelocityCommandCfgClip, env: ManagerBasedEnv):
        super().__init__(cfg,env)

    def _resample_command(self, env_ids: Sequence[int]):
        # sample velocity commands
        r = torch.empty(len(env_ids), device=self.device)
        # -- linear velocity - x direction
        self.vel_command_b[env_ids, 0] = r.uniform_(*self.cfg.ranges.lin_vel_x)
        # -- linear velocity - y direction
        self.vel_command_b[env_ids, 1] = r.uniform_(*self.cfg.ranges.lin_vel_y)

        vx = self.vel_command_b[env_ids, 0]
        vy = self.vel_command_b[env_ids, 1]

        print(f"vel_cmd_before_clipping: {self.vel_command_b}")

        vx[vx < self.cfg.clip_threshold] = 0.0
        vy[vy < self.cfg.clip_threshold] = 0.0

        self.vel_command_b[env_ids, 0] = vx
        self.vel_command_b[env_ids, 1] = vy


        # print(f"vel_cmd_after_clipping: {self.vel_command_b}")
        # print(f"clip_threshold: {self.cfg.clip_threshold}")
        # print(f"env_ids:{env_ids}")
 
        
        # -- ang vel yaw - rotation around z
        self.vel_command_b[env_ids, 2] = r.uniform_(*self.cfg.ranges.ang_vel_z)
        # heading target
        if self.cfg.heading_command:
            self.heading_target[env_ids] = r.uniform_(*self.cfg.ranges.heading)
            # update heading envs
            self.is_heading_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_heading_envs
        # update standing envs
        self.is_standing_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_standing_envs

    