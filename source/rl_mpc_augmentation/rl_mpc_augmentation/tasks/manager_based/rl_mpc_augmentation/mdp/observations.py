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

from isaaclab.sensors import RayCaster

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedEnv

def gait_cycle_var(
    env: ManagerBasedRLEnv,
    offset: list[float],
) -> torch.Tensor:
 
    
    period = env.action_manager.get_term("gait_cycle").processed_actions
    
    eps = .5 # or any small positive value
    period = torch.where(period == 0, eps, period)
    #print(f"period in obs: {period}")

    #global_phase = ((env.episode_length_buf * env.step_dt) % period / period).unsqueeze(1)

    elapsed_time = (env.episode_length_buf * env.step_dt).unsqueeze(1)    # (num_envs, 1)

    global_phase = (elapsed_time % period) / period     
   # print(f"global phase shape in obs: {period.shape}")

  #  print(f"global phase: {global_phase}")

    phases = []
    for offset_ in offset:
        phase = (global_phase + offset_) % 1.0
        phases.append(phase)

    leg_phase = torch.cat(phases, dim=-1)

    #print(f"Glboal phase in observation: {global_phase}")


    return leg_phase

def gait_cycle(
    env: ManagerBasedRLEnv,
    period: float,
    offset: list[float],
) -> torch.Tensor:
 
    


    global_phase = ((env.episode_length_buf * env.step_dt) % period / period).unsqueeze(1)
  
   # print(f"global phase shape in obs: {period.shape}")

  #  print(f"global phase: {global_phase}")

    phases = []
    for offset_ in offset:
        phase = (global_phase + offset_) % 1.0
        phases.append(phase)

    leg_phase = torch.cat(phases, dim=-1)

    #print(f"Glboal phase in observation: {global_phase}")


    return leg_phase


def scan_dot(env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:

 
    # extract the used quantities (to enable type-hinting)
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]

    #print(f"sensor data count: {sensor.num_instances}") #One scan_dot sensor
    data = sensor.data

    

    return sensor.data.pos_w[:, 2].unsqueeze(1) - sensor.data.ray_hits_w[..., 2]