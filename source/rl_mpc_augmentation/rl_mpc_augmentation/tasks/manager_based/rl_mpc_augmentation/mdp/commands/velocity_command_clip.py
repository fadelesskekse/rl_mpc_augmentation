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

    # def _resample_command(self, env_ids: Sequence[int]):
    #     # sample velocity commands
    #     r = torch.empty(len(env_ids), device=self.device)
    #     # -- linear velocity - x direction
    #     self.vel_command_b[env_ids, 0] = r.uniform_(*self.cfg.ranges.lin_vel_x)
    #     # -- linear velocity - y direction
    #     self.vel_command_b[env_ids, 1] = r.uniform_(*self.cfg.ranges.lin_vel_y)

    #     vx = self.vel_command_b[env_ids, 0]
    #     vy = self.vel_command_b[env_ids, 1]

    #     #print(f"vel_cmd_before_clipping: {self.vel_command_b}")

    
    #     max_lin_vel_x = self.cfg.ranges.lin_vel_x[1]
    #     max_lin_vel_y = self.cfg.ranges.lin_vel_y[1]

    #    # print(f"max x vel: {max_lin_vel_x}")

    #     # Only apply clipping if max range is above clip_start_threshold
    #     if max_lin_vel_x >= self.cfg.clip_start_threshold:
    #         vx[torch.abs(vx) < self.cfg.clip_threshold] = 0.0
    #         self.vel_command_b[env_ids, 0] = vx
    #        # print("Im clipping")

    #     if max_lin_vel_y >= self.cfg.clip_start_threshold:
    #         vy[torch.abs(vy) < self.cfg.clip_threshold] = 0.0
    #         self.vel_command_b[env_ids, 1] = vy



    #     # vx[vx < self.cfg.clip_threshold] = 0.0
    #     # vy[vy < self.cfg.clip_threshold] = 0.0

    #     # self.vel_command_b[env_ids, 0] = vx
    #     # self.vel_command_b[env_ids, 1] = vy


    #     # print(f"vel_cmd_after_clipping: {self.vel_command_b}")
    #     # print(f"clip_threshold: {self.cfg.clip_threshold}")
    #     # print(f"env_ids:{env_ids}")
 
        
    #     # -- ang vel yaw - rotation around z
    #     self.vel_command_b[env_ids, 2] = r.uniform_(*self.cfg.ranges.ang_vel_z)
    #     # heading target
    #     if self.cfg.heading_command:
    #         self.heading_target[env_ids] = r.uniform_(*self.cfg.ranges.heading)
    #         # update heading envs
    #         self.is_heading_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_heading_envs
    #     # update standing envs
    #     self.is_standing_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_standing_envs

    def _resample_command(self, env_ids: Sequence[int]):
        r = torch.empty(len(env_ids), device=self.device)

        # -- linear velocity - x direction
        ext_lower_x, ext_upper_x = self.cfg.ranges.lin_vel_x
        if ext_upper_x >= self.cfg.clip_start_threshold:
            # Sample from [clip_threshold, ext_upper_x]
            self.vel_command_b[env_ids, 0] = r.uniform_(self.cfg.clip_threshold, ext_upper_x)
        else:
            # Sample from full external range
            self.vel_command_b[env_ids, 0] = r.uniform_(ext_lower_x, ext_upper_x)

        # -- linear velocity - y direction
        ext_lower_y, ext_upper_y = self.cfg.ranges.lin_vel_y
        if ext_upper_y >= self.cfg.clip_start_threshold:
            # Sample from [clip_threshold, ext_upper_y]
            self.vel_command_b[env_ids, 1] = r.uniform_(self.cfg.clip_threshold, ext_upper_y)
        else:
            # Sample from full external range
            self.vel_command_b[env_ids, 1] = r.uniform_(ext_lower_y, ext_upper_y)

        # -- ang vel yaw - rotation around z (unchanged)
        self.vel_command_b[env_ids, 2] = r.uniform_(*self.cfg.ranges.ang_vel_z)

        # heading target (unchanged)
        if self.cfg.heading_command:
            self.heading_target[env_ids] = r.uniform_(*self.cfg.ranges.heading)
            self.is_heading_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_heading_envs

        # update standing envs (unchanged)
        self.is_standing_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_standing_envs