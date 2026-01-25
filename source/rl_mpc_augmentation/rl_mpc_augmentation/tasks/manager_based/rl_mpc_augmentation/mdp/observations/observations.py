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

from isaaclab.assets import RigidObject, Articulation
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

    # (B, N, 3) - (B, 1, 3) -> (B, N, 3)
    delta = sensor.data.ray_hits_w - sensor.data.pos_w[:, None, :]
   # print(f"sensor pos world: {sensor.data.pos_w[:, None, :]}")
   # print(f"sensor pos world shape: {sensor.data.pos_w[:, None, :].shape}")

    #print(f"sensor.data.ray_hits_w : {sensor.data.ray_hits_w }")
   # print(f"sensor.data.ray_hits_w shape : {sensor.data.ray_hits_w.shape}")
    





  #  print(f"delta shape: {delta.shape}")

    # Euclidean norm over x,y,z -> (B, N)
    out = torch.norm(delta, dim=-1)
    
    return out
    #return torch.ones_like(out)

def priv_latent(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:

    asset: Articulation = env.scene[asset_cfg.name]

    stiffness = asset.data.joint_stiffness[:, asset_cfg.joint_ids]
    damping = asset.data.joint_damping[:, asset_cfg.joint_ids]

    friction_left = asset.root_physx_view.get_material_properties()[:, 9:16]
    friction_right = asset.root_physx_view.get_material_properties()[:, 16:23]

    friction_left  = friction_left[:, :, :2]     # (num_envs, 7, 2)
    friction_right = friction_right[:, :, :2]    # (num_envs, 7, 2)

    left_avg_fric  = friction_left.mean(dim=1)     # (num_envs, 2)
    right_avg_fric = friction_right.mean(dim=1)    # (num_envs, 2)

    device = stiffness.device

    left_avg_fric  = left_avg_fric.to(device)
    right_avg_fric = right_avg_fric.to(device)

    mass = asset.root_physx_view.get_masses()[:, 9].unsqueeze(-1).to(device)
    com = asset.root_physx_view.get_coms()[:,9,:3].to(device)


    #print(f"mass shape: {mass.shape}")
    #print(f"com shape: {com.shape}")




    # print(f"left fric {left_avg_fric}")
    # print(f"right fric {right_avg_fric}")
    
    #print(f"stiffness in obs: {stiffness}")
   # print(f"dampingin obs: {damping}")

   # print(f"mass for torso: {mass}")
   # print(f"com for torso {com}")


    priv = torch.cat(
    [stiffness, damping, left_avg_fric, right_avg_fric,mass,com],
    dim=-1
)

   # print(f"shape of priv: {priv.shape}")

    return priv

# riv_latent_gains_stiffness = ObsTerm(func=mdp., history_length=0)
#         priv_latent_gains_damping = ObsTerm(func=mdp., history_length=0)
#         priv_latent_mass = ObsTerm(func=mdp., history_length=0)
#         priv_latent_com = ObsTerm(func=mdp., history_length=0)
#         priv_latent_friction= ObsTerm(func=mdp., history_length=0)


def priv_latent_gains_stiffness(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:

    asset: Articulation = env.scene[asset_cfg.name]

    stiffness = asset.data.joint_stiffness[:, asset_cfg.joint_ids]

    #return torch.ones_like(stiffness)

    return stiffness
    

def priv_latent_gains_damping(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:

    asset: Articulation = env.scene[asset_cfg.name]
    
    damping = asset.data.joint_damping[:, asset_cfg.joint_ids]
    #return torch.ones_like(damping)

    return damping



def priv_latent_mass(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:

    asset: Articulation = env.scene[asset_cfg.name]

    device = env.device

    mass = asset.root_physx_view.get_masses()[:, 9].unsqueeze(-1).to(device)

    #return torch.ones_like(mass)

    return mass

def priv_latent_com(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:

    asset: Articulation = env.scene[asset_cfg.name]

    device = env.device

    com = asset.root_physx_view.get_coms()[:,9,:3].to(device)

   # return torch.ones_like(com)

    return com

def priv_latent_friction(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:

    asset: Articulation = env.scene[asset_cfg.name]

    friction_left = asset.root_physx_view.get_material_properties()[:, 9:16]
    friction_right = asset.root_physx_view.get_material_properties()[:, 16:23]

    friction_left  = friction_left[:, :, :2]     # (num_envs, 7, 2)
    friction_right = friction_right[:, :, :2]    # (num_envs, 7, 2)

    left_avg_fric  = friction_left.mean(dim=1)     # (num_envs, 2)
    right_avg_fric = friction_right.mean(dim=1)    # (num_envs, 2)

    device = env.device

    left_avg_fric  = left_avg_fric.to(device)
    right_avg_fric = right_avg_fric.to(device)

    priv = torch.cat(
        [left_avg_fric,right_avg_fric],
        dim=-1
    )

    #return torch.ones_like(priv)

    return priv



