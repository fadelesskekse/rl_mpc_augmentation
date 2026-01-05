# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to activate certain terminations.

The functions can be passed to the :class:`isaaclab.managers.TerminationTermCfg` object to enable
the termination introduced by the function.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def sub_terrain_out_of_bounds(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), distance_buffer: float = 0.0
) -> torch.Tensor:
    """Terminate when the actor move too close to the edge of the terrain.

    If the actor moves too close to the edge of the terrain, the termination is activated. The distance
    to the edge of the terrain is calculated based on the size of the terrain and the distance buffer.
    """
    if env.scene.cfg.terrain.terrain_type == "plane":
        # we have infinite terrain because it is a plane
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    elif env.scene.cfg.terrain.terrain_type == "generator":
        # obtain the size of the sub-terrains

        env_origins =env.scene.env_origins[:, :2]

        terrain_gen_cfg = env.scene.terrain.cfg.terrain_generator
        grid_width, grid_length = terrain_gen_cfg.size

        border_width = terrain_gen_cfg.border_width

        if distance_buffer > border_width:
            raise ValueError("Distance Buffer should be smaller than Border_width, it is not.")


        # extract the used quantities (to enable type-hinting)
        asset: RigidObject = env.scene[asset_cfg.name]

        #print(f"grid_width: {grid_width}, grid_length: {grid_length}, distance_buffer: {distance_buffer}")

        #print(f"env_origins: {env_origins}")
        #print(f"root_pos_w: {asset.data.root_pos_w[:,:2]}")


        #print(f"diff_pos: {asset.data.root_pos_w[:,:2] - env_origins}")
       # print(f"diff_buffer: {grid_width/2 + distance_buffer}")


        # check if the agent is out of bounds
        x_out_of_bounds = torch.abs(asset.data.root_pos_w[:, 0]  - env_origins[:,0]) > grid_width/2 + distance_buffer
        y_out_of_bounds = torch.abs(asset.data.root_pos_w[:, 1]  - env_origins[:,1]) > grid_length/2 + distance_buffer

        # if torch.any(x_out_of_bounds):
        #     print("//////////////////////////////////////////////////////////////////")
        #     print(f"x_out_of_bounds: {x_out_of_bounds.nonzero(as_tuple=True)[0]}")

        # if torch.any(y_out_of_bounds):
        #     print("//////////////////////////////////////////////////////////////////")
        #     print(f"y_out_of_bounds: {y_out_of_bounds.nonzero(as_tuple=True)[0]}")
        
        return torch.logical_or(x_out_of_bounds, y_out_of_bounds)
    else:
        raise ValueError("Received unsupported terrain type, must be either 'plane' or 'generator'.")


    terrain: TerrainImporter = env.scene.terrain
    command = env.command_manager.get_command("base_velocity")
    # compute the distance the robot walked
    distance = torch.norm(asset.data.root_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1)