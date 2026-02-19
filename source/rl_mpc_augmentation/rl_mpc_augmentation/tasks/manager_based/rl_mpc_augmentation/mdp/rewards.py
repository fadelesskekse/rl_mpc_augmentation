# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor,RayCaster

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def ankle_torque_min(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    
    asset: Articulation = env.scene[asset_cfg.name]

   # print(f"asset.data.soft_joint_pos_limits: {asset.data.soft_joint_pos_limits}")

   # print(f"asset.data.ids: {asset.data.joint_names}")

    asset.data.joint_vel_limits
    
    ankle_torque = asset.data.applied_torque[:, asset_cfg.joint_ids]
    ankle_torque_mean = ankle_torque.mean(dim=1)
    return ankle_torque_mean



def gait(
    env: ManagerBasedRLEnv,
    offset: list[float],
    sensor_cfg: SceneEntityCfg,
    nominal: float = .5,
    threshold: float = 0.5,
    command_name=None,
) -> torch.Tensor:
    
    
    period = env.action_manager.get_term("gait_cycle").processed_actions
    

    eps = nominal # or any small positive value
    period = torch.where(period == 0, eps, period)

   # print(f"period in reward: {period}")

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    is_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0

    elapsed_time = (env.episode_length_buf * env.step_dt).unsqueeze(1) 

    global_phase = (elapsed_time % period) / period    


    #print(f"global_phase in gait reward: {global_phase}")
    phases = []
    for offset_ in offset:
        phase = (global_phase + offset_) % 1.0
        phases.append(phase)
    leg_phase = torch.cat(phases, dim=-1)


    reward = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
    for i in range(len(sensor_cfg.body_ids)):
        is_stance = leg_phase[:, i] < threshold
        reward += ~(is_stance ^ is_contact[:, i])

    if command_name is not None:
        cmd_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
        reward *= cmd_norm > 0.1
    return reward

def gait_no_vel_cmd(
    env: ManagerBasedRLEnv,
    offset: list[float],
    sensor_cfg: SceneEntityCfg,
    nominal: float = .5,
    threshold: float = 0.5,
    command_name=None,
) -> torch.Tensor:
    
    
    period = env.action_manager.get_term("gait_cycle").processed_actions
    

    eps = nominal # or any small positive value
    period = torch.where(period == 0, eps, period)

   # print(f"period in reward: {period}")

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    is_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0

    elapsed_time = (env.episode_length_buf * env.step_dt).unsqueeze(1) 

    global_phase = (elapsed_time % period) / period    


    #print(f"global_phase in gait reward: {global_phase}")
    phases = []
    for offset_ in offset:
        phase = (global_phase + offset_) % 1.0
        phases.append(phase)
    leg_phase = torch.cat(phases, dim=-1)


    reward = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
    for i in range(len(sensor_cfg.body_ids)):
        is_stance = leg_phase[:, i] < threshold
        reward += ~(is_stance ^ is_contact[:, i])

    if command_name is not None:
        walk_bool = env.command_manager.get_command(command_name)[:, 0]
        reward *= walk_bool == 1

       # print(f"walk bool in gait rew: {walk_bool}")
    return reward


def gait_deviation(env: ManagerBasedRLEnv,
                   nominal: float = .5,) -> torch.Tensor:
   
   period_action = env.action_manager.get_term("gait_cycle").raw_actions

  # print(f"shape of period: {period_action}")

   nominal_period = nominal

   lam: float = 4.6

   gait_err = torch.abs(period_action - nominal_period)

  # print(f"shape fo gait err: {gait_err}")

   rew = torch.exp(-lam*gait_err)

   rew = rew.squeeze(-1)

   #print(f"shape fo rew: {rew}")

   

   return rew

def lin_vel_z_negative_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    z_vel = asset.data.root_lin_vel_b[:, 2]

    # clamp to negative part, then square
    z_vel_neg = torch.clamp(z_vel, max=0.0)
    return torch.square(z_vel_neg)


def lin_vel_z_positive_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    z_vel = asset.data.root_lin_vel_b[:, 2]

    # clamp to positive part, then square
    z_vel_pos = torch.clamp(z_vel, min=0.0)
    return torch.square(z_vel_pos)


# def body_lin_acc_l2_z(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
#     """Penalize the linear acceleration of bodies using L2-kernel."""
#     asset: Articulation = env.scene[asset_cfg.name]
#     return torch.sum(torch.norm(asset.data.body_lin_acc_w[:, asset_cfg.body_ids, :], dim=-1), dim=1)

def body_lin_acc_l2_z(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize only the z-axis linear acceleration of bodies."""
    asset: Articulation = env.scene[asset_cfg.name]

    # Extract only z acceleration (index 2)
    z_acc = asset.data.body_lin_acc_w[:, asset_cfg.body_ids, 2]

    # L2 kernel → square it
    return torch.sum(torch.square(z_acc), dim=1)


def lin_vel_x(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    x_vel = asset.data.root_lin_vel_b[:, 0]

    walk_bool = env.command_manager.get_command(command_name)[:, 0]

    reward = torch.where(
        walk_bool == True,
        x_vel,
        -torch.abs(x_vel)
    )

   # print(f"Walk_bool: {walk_bool}")
   # print(f"x_vel{x_vel}")
    #print(f"reward to be scaled: {reward}")
    
    return reward



def ang_vel_z_exp(
    env: ManagerBasedRLEnv, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # compute the error
    ang_vel_error = torch.square(asset.data.root_ang_vel_b[:, 2])
    return torch.exp(-ang_vel_error / std**2)


def scan_dot_avg_reward(
    env: ManagerBasedRLEnv, 
    sensor_cfg: SceneEntityCfg,
    target: float = 0.5,
    std: float = 0.25,
) -> torch.Tensor:
    """Reward based on average normalized scan distance.
    
    Higher reward when obstacles are closer (scan_dot values closer to 1.0).
    This encourages the robot to navigate toward/over terrain features.
    """
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    
    ray_hit = sensor.data.ray_hits_w 
    sensor_start = sensor.data.pos_w[:, None, :]
    
    # Check which rays missed (have inf values)
    is_miss = torch.isinf(ray_hit).any(dim=-1)  # (num_envs, num_rays)
    
    # Replace inf with zeros temporarily for calculation
    ray_hit = torch.where(torch.isinf(ray_hit), torch.zeros_like(ray_hit), ray_hit)
    
    delta = ray_hit - sensor_start
    out = torch.norm(delta, dim=-1)  # (num_envs, num_rays)
    
    # For missed rays, set distance to max_distance
    out = torch.where(is_miss, torch.full_like(out, sensor.cfg.max_distance), out)

    # Normalize: 1.0 = close, 0.0 = far
    out_normalized = 1.0 - (out / sensor.cfg.max_distance)
    out_normalized = torch.clamp(out_normalized, 0.0, 1.0)

    # Return average across all rays (shape: num_envs)
    out_avg = out_normalized.mean(dim=-1)

    error = torch.square(out_avg - target)
    return torch.exp(-error / (std ** 2))
    
   # return out_avg

def scan_foot_placement_rew(
    env: ManagerBasedRLEnv, 
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    obs_term: str = "scan_dot",
    threshold: float = .1,
) -> torch.Tensor:

    scan_obs = None

    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    asset: RigidObject = env.scene[asset_cfg.name]

    if env.observation_manager._obs_buffer is None:
        env.observation_manager.compute()
    critic_obs = env.observation_manager._obs_buffer["critic"]   

    names = env.observation_manager._group_obs_term_names["critic"]
    dims  = env.observation_manager._group_obs_term_dim["critic"] 

    offset = 0
    for name, shape in zip(names, dims):
        width = int(np.prod(shape))

        if name == obs_term:
            # <-- THIS is the important line
            scan_obs =  critic_obs[:, offset:offset+width]
            break

        offset += width

    if scan_obs is None:
        raise ValueError(f"obs_term '{obs_term}' not found in critic group")

    #At this point we have the scan_dot critic obs to use in our reward


   # print([asset.body_names[i] for i in asset_cfg.body_ids])
    #print(f"sensor ray hits: {sensor.data.ray_hits_w }")
    danger_mask = scan_obs < threshold 
    ray_hits = sensor.data.ray_hits_w 
    valid_hits = ~torch.isinf(ray_hits).any(dim=-1)
    candidate_mask = danger_mask & valid_hits

    hit_xy = ray_hits[..., :2].clone()
    mask3 = candidate_mask.unsqueeze(-1)

    hit_xy[~mask3.expand_as(hit_xy)] = float('nan')

    feet_pos = asset.data.body_pos_w[:, asset_cfg.body_ids, :2]  

    feet_xy_exp = feet_pos.unsqueeze(2)   # [E, F, 1, 2]
    hit_xy_exp  = hit_xy.unsqueeze(1)    # [E, 1, R, 2]

    dist = torch.norm(feet_xy_exp - hit_xy_exp, dim=-1) 

        # mask of valid distances
    valid_dist_mask = ~torch.isnan(dist)

    # replace NaNs with 0 for summation
    dist_no_nan = torch.nan_to_num(dist, nan=0.0)

    # count valid rays per foot
    valid_counts = valid_dist_mask.sum(dim=(1,2)) 

    total_dist = dist_no_nan.sum(dim=(1,2))  

    no_candidates = valid_counts == 0

    avg_dist = torch.zeros_like(total_dist)

    avg_dist[~no_candidates] = (
    total_dist[~no_candidates] /
    valid_counts[~no_candidates]
    )

    reward = torch.zeros_like(avg_dist)

    reward[~no_candidates] = torch.exp(-avg_dist[~no_candidates])

    #print(f"reward: {reward}")

    # environments with no candidate rays automatically get 0
    return reward