# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to enable different events.

Events include anything related to altering the simulation state. This includes changing the physics
materials, applying external forces, and resetting the state of the asset.

The functions can be passed to the :class:`isaaclab.managers.EventTermCfg` object to enable
the event introduced by the function.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING


import isaaclab.utils.math as math_utils

from isaaclab.assets import Articulation,  RigidObject
from isaaclab.managers import  SceneEntityCfg


if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

def reset_root_state_uniform_grouped_yaws(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    max_jitter: tuple[float, float] = (0.0, 0.0),  # NEW
):
    """Reset the asset root state to a random position and velocity uniformly within the given ranges.

    This function randomizes the root position and velocity of the asset.

    * It samples the root position from the given ranges and adds them to the default root position, before setting
      them into the physics simulation.
    * It samples the root orientation from the given ranges and sets them into the physics simulation.
    * It samples the root velocity from the given ranges and sets them into the physics simulation.

    The function takes a dictionary of pose and velocity ranges for each axis and rotation. The keys of the
    dictionary are ``x``, ``y``, ``z``, ``roll``, ``pitch``, and ``yaw``. The values are tuples of the form
    ``(min, max)``. If the dictionary does not contain a key, the position or velocity is set to zero for that axis.
    """

    #print(f"env_ids LENGTH: {len(env_ids)}")
    #print(f"env_ids in reset: {env_ids}")
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    # get default root state
    root_states = asset.data.default_root_state[env_ids].clone()

    # poses
    range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=asset.device)
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=asset.device)

        # ------------------------------------------------
    # Snap yaw to cardinal + jitter (DEGREES → RADIANS)
    # ------------------------------------------------
    yaw = rand_samples[:, 5]

    pi = torch.pi
    half_pi = 0.5 * pi

    # Clamp for numerical safety
    yaw_clamped = torch.clamp(yaw, -pi, pi - 1e-6)

    # Quadrant index over [-π, π)
    quadrant = torch.floor((yaw_clamped + pi) / half_pi).long()
    quadrant = torch.clamp(quadrant, min=0, max=3)

    # Cardinal angles: [-π, -π/2, 0, π/2]
    cardinal_angles = torch.tensor(
        [-pi, -half_pi, 0.0, half_pi],
        device=asset.device,
    )

    base_yaw = cardinal_angles[quadrant]

    # ---- jitter in DEGREES from params ----
    jitter_deg_min, jitter_deg_max = max_jitter
    jitter_deg_min = float(jitter_deg_min)
    jitter_deg_max = float(jitter_deg_max)

    if jitter_deg_max < jitter_deg_min:
        raise ValueError(f"max_jitter must satisfy min <= max (degrees), got {max_jitter}")

    # Convert degrees → radians
    deg2rad = pi / 180.0
    jitter_rad_min = jitter_deg_min * deg2rad
    jitter_rad_max = jitter_deg_max * deg2rad

    # Sample magnitude ∈ [jitter_rad_min, jitter_rad_max]
    mag = jitter_rad_min + (jitter_rad_max - jitter_rad_min) * torch.rand_like(base_yaw)

    # Random sign
    sign = torch.where(
        torch.rand_like(base_yaw) < 0.5,
        -torch.ones_like(base_yaw),
        torch.ones_like(base_yaw),
    )

    jitter = sign * mag

    yaw_final = base_yaw + jitter

    # Wrap back to [-π, π]
    yaw_final = torch.remainder(yaw_final + pi, 2.0 * pi) - pi

    rand_samples[:, 5] = yaw_final


    positions = root_states[:, 0:3] + env.scene.env_origins[env_ids] + rand_samples[:, 0:3]
    orientations_delta = math_utils.quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
    orientations = math_utils.quat_mul(root_states[:, 3:7], orientations_delta)
    # velocities
    range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=asset.device)
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=asset.device)

    velocities = root_states[:, 7:13] + rand_samples

    # set into the physics simulation
    asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim(velocities, env_ids=env_ids)



def push_by_setting_velocity_delayed(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    velocity_range: dict[str, tuple[float, float]],
    #delayed_iteration: int,
    curr_lim: float = .2,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Push the asset by setting the root velocity to a random value within the given ranges.

    This creates an effect similar to pushing the asset with a random impulse that changes the asset's velocity.
    It samples the root velocity from the given ranges and sets the velocity into the physics simulation.

    The function takes a dictionary of velocity ranges for each axis and rotation. The keys of the dictionary
    are ``x``, ``y``, ``z``, ``roll``, ``pitch``, and ``yaw``. The values are tuples of the form ``(min, max)``.
    If the dictionary does not contain a key, the velocity is set to zero for that axis.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]

    command_term = env.command_manager.get_term("base_velocity")
    max_vel_cmd = command_term.cfg.ranges.lin_vel_x[1]

    #print(f"pushign robots: {env_ids}")


    # velocities
    vel_w = asset.data.root_vel_w[env_ids]

    if max_vel_cmd >= curr_lim:
        # sample random velocities
        range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=asset.device)
        vel_w += math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], vel_w.shape, device=asset.device)
        # set the velocities into the physics simulation
        #print("im in here")
    asset.write_root_velocity_to_sim(vel_w, env_ids=env_ids)