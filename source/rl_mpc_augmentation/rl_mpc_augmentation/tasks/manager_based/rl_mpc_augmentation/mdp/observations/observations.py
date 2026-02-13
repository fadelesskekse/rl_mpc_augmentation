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
    # avg_leg_phase = leg_phase.abs().mean()
    # if avg_leg_phase > 5:
    #     raise Exception(f"Average leg_phase value too high: {avg_leg_phase.item()}")


    return leg_phase
    #return torch.zeros_like(leg_phase)

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


# def scan_dot(env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg,asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:

#    # asset: Articulation = env.scene[asset_cfg.name]

#     #sensor_pos = asset.data.body_link_pose_w[:,asset_cfg.body_ids, :3]

#    # print(f"sensor pose from asset: {sensor_pos}")


#     sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
#     # Now safe to access
#     ray_hit = sensor.data.ray_hits_w 
#     ray_hit = torch.where(torch.isinf(ray_hit), torch.zeros_like(ray_hit), ray_hit)
    
#     #*sensor.cfg.max_distance
#     sensor_start = sensor.data.pos_w[:, None, :]

#    # print(f"sensor pos world from sensor cfg: {sensor_start}")
#    # print(f"ray hit from sensor cfg: {ray_hit}")
    
#     # Replace NaN with zeros
#     #ray_hit = torch.where(torch.isnan(ray_hit), torch.zeros_like(ray_hit), ray_hit)
#     #sensor_start = torch.where(torch.isnan(sensor_start), torch.zeros_like(sensor_start), sensor_start)
    
#     delta = ray_hit - sensor_start
#     out = torch.norm(delta, dim=-1)
#     out_avg = out.mean(dim=-1, keepdim=True)  # (num_envs, 1)
#    # sensor_start_avg = sensor_start.mean(dim=1)
#    # ray_hit_avg = ray_hit.mean(dim=1)

#     print(f"start: {sensor_start}")
#     print(f"ray_hit: {ray_hit}")
#     print(f"delta {delta}")
#     print(f"out {out}")
#     print(f"average out: {out_avg}")
#     #print(f"average ray_hit: {ray_hit_avg}")
#     #print(f"average sensor_start: {sensor_start_avg}")
    
#     #out = torch.where(torch.isnan(out), torch.zeros_like(out), out)
    
#     return out

def scan_dot(env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:

    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    
    ray_hit = sensor.data.ray_hits_w 
    sensor_start = sensor.data.pos_w[:, None, :]
    
    # Check which rays missed (have inf values)
    is_miss = torch.isinf(ray_hit).any(dim=-1)  # (num_envs, num_rays)
    
    # Replace inf with zeros temporarily for calculation
    ray_hit = torch.where(torch.isinf(ray_hit), torch.zeros_like(ray_hit), ray_hit)
    
    delta = ray_hit - sensor_start

   # print(f"ray_hit: {ray_hit}")
    out = torch.norm(delta, dim=-1)  # (num_envs, num_rays)
    
    # For missed rays, set distance to max_distance
    out = torch.where(is_miss, torch.full_like(out, sensor.cfg.max_distance), out)

    
  #  out_avg = out.mean(dim=-1, keepdim=True)

   # print(f"start: {sensor_start}")
   # print(f"ray_hit: {ray_hit}")
   # print(f"delta: {delta}")
   # print(f"out: {out}")
   # print(f"average out: {out_avg}")

    out_normalized = 1.0 - (out / sensor.cfg.max_distance)
    out_normalized = torch.clamp(out_normalized, 0.0, 1.0)

    # N = out_normalized.shape[1]
    # batch_size = out_normalized.shape[0]
    # group_size = 27

    # pattern = torch.arange(N // group_size).repeat_interleave(group_size)
    # if pattern.shape[0] < N:
    #     pattern = torch.cat([pattern, torch.full((N - pattern.shape[0],), pattern[-1])])
    # test_value = pattern.unsqueeze(0).repeat(batch_size, 1).to(out_normalized.device, dtype=out_normalized.dtype)
    # return test_value
    
    #out_avg = out_normalized.mean(dim=-1, keepdim=True)

   # print(f"out avg: {out_avg}")


    #return torch.zeros_like(out_normalized)
    # avg_out_normalized = out_normalized.abs().mean()
    # if avg_out_normalized > 5:
    #     raise Exception(f"Average scan_dot normalized value too high: {avg_out_normalized.item()}")

    return out_normalized
   

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


def priv_latent_gains_stiffness(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    scale_val: float = .2,
) -> torch.Tensor:

    asset: Articulation = env.scene[asset_cfg.name]

    stiffness = asset.data.joint_stiffness[:, asset_cfg.joint_ids]
    stiffness_default = asset.data.default_joint_stiffness[:, asset_cfg.joint_ids]

    # avoid divide-by-zero just in case
    eps = 1e-6
    scale = scale_val * stiffness_default + eps

    stiffness_norm = (stiffness - stiffness_default) / scale

    #print(f"stiffness: {stiffness_norm}")

   # print(f"stiffness_default: {stiffness_default}")

  # print(f"stiffness random: {stiffness}")
   
    # avg_stiffness_noirm = stiffness_norm.mean()
    # if avg_stiffness_noirm > 5:
    #     raise Exception(f"Average stiffness_norm too high: {avg_stiffness_noirm.item()}")
    
    return stiffness_norm
    #return torch.ones_like(stiffness_norm)*0

def priv_latent_gains_damping(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    scale_val: float = .2,
) -> torch.Tensor:

    asset: Articulation = env.scene[asset_cfg.name]

    damping = asset.data.joint_damping[:, asset_cfg.joint_ids]
    damping_default = asset.data.default_joint_damping[:, asset_cfg.joint_ids]

   # print(f"dampign default: {damping_default}")

    # avoid divide-by-zero just in case
    eps = 1e-6
    scale = scale_val * damping_default + eps

    damping_norm = (damping - damping_default) / scale
    #print(f"damping_norm: {damping_norm}")
    # avg_damping_norm = damping_norm.mean()
    # if avg_damping_norm > 5:
    #     raise Exception(f"Average damping_norm too high: {avg_damping_norm.item()}")
    

    return damping_norm
    #return torch.ones_like(damping_norm)*0
   

def priv_latent_mass(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    scale_val: float = .2,
) -> torch.Tensor:

    asset: Articulation = env.scene[asset_cfg.name]
    device = env.device

    # current randomized mass (per env)
    mass = asset.root_physx_view.get_masses()[:, 9].unsqueeze(-1).to(device)

    # nominal/default mass
    mass_default = asset.data.default_mass[:,asset_cfg.body_ids].to(device)

   # print(f"body names: {asset_cfg.body_names}")

   # print(f"mass default shape {mass_default.shape}")

   # print(f"mass default: {mass_default}")

    #print(f"mass nwe: {mass}")

    # avoid divide-by-zero just in case
    eps = 1e-6
    scale = scale_val * mass_default + eps

    mass_norm = (mass - mass_default) / scale

   # print(f"mass norm {mass_norm}")

    # avg_mass_norm = mass_norm.mean()
    # if avg_mass_norm > 5:
    #     raise Exception(f"Average mass_norm too high: {avg_mass_norm.item()}")
    

    return mass_norm
    #return torch.ones_like(mass_norm)*100

    

def priv_latent_com(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:

    asset: Articulation = env.scene[asset_cfg.name]

    device = env.device

    com = asset.root_physx_view.get_coms()[:,9,:3].to(device)

   # return torch.ones_like(com)
    # avg_com = com.mean()
    # if avg_com > 5:
    #     raise Exception(f"Average com_norm too high: {avg_com.item()}")
    

    return com
    #return torch.ones_like(com)*100

    

def priv_latent_friction(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:

    asset: Articulation = env.scene[asset_cfg.name]

    friction_left = asset.root_physx_view.get_material_properties()[:, 9:16]
    friction_right = asset.root_physx_view.get_material_properties()[:, 16:23]

    #print(f"left friction during observation: {friction_left}")
   # print(f"right friction during observation: {friction_right}")

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

    # avg_fric = priv.mean()
    # if avg_fric > 5:
    #     raise Exception(f"Average friction_norm too high: {avg_fric.item()}")
    


    #return torch.ones_like(priv)*0

    return priv


