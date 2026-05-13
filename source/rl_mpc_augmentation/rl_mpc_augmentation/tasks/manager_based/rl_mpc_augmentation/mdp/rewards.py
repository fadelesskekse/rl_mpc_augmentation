# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg, ManagerTermBase
from isaaclab.sensors import ContactSensor,RayCaster
from isaaclab.utils.math import quat_apply_inverse, quat_apply
import numpy as np

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def foot_contact_vel_penalty(
    env: ManagerBasedRLEnv,
    offset: list[float],
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    nominal: float = 0.5,
    threshold: float = 0.5,
    buffer: float = 0.05,
    lamda: float = 1.0,
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


    asset: Articulation = env.scene[asset_cfg.name]

    # full robot linear velocity magnitude in world frame
    robot_vel = asset.data.root_lin_vel_w
    robot_speed = torch.linalg.norm(robot_vel, dim=1)   # shape: [num_envs]

    penalty = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
    stance_cutoff = threshold - buffer
    for i in range(len(sensor_cfg.body_ids)):
        is_stance = leg_phase[:, i] < stance_cutoff

        # leg should be swinging, but is in contact
        caught_contact = (~is_stance) & is_contact[:, i]

        # penalize robot speed when this happens
        penalty += robot_speed * caught_contact.float()

    if command_name is not None:
        cmd_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
        penalty *= (cmd_norm > 0.4).float()

    penalty = 1.0 - torch.exp(-lamda* penalty)

    return penalty

def soft_landing(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, command_name: str, command_threshold: float) -> torch.Tensor:
    """Penalize high impact forces at landing to encourage soft footfalls."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    sensor_data = contact_sensor.data
    forces = sensor_data.net_forces_w[:, sensor_cfg.body_ids, :]  # (num_envs, num_bodies, 3)
    force_magnitude = torch.norm(forces, dim=-1)  # (num_envs, num_bodies)
    first_contact = contact_sensor.compute_first_contact(dt=env.step_dt)[:, sensor_cfg.body_ids]  # (num_envs, num_bodies)
    landing_impact = force_magnitude * first_contact.float()
    cost = torch.sum(landing_impact, dim=1)  # (num_envs,)
    command = env.command_manager.get_command(command_name)
    linear_norm = torch.norm(command[:, :2], dim=1)  # (num_envs,)
    angular_norm = torch.abs(command[:, 2])  # (num_envs,)
    total_command = linear_norm + angular_norm
    actice = (total_command > command_threshold).float()
    cost = cost * actice
    return cost


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

def foot_collision_joint_movement(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),) -> torch.Tensor:
    

    asset: RigidObject = env.scene[asset_cfg.name]

    hip_roll_joint_vel = asset.data.joint_vel[:, asset_cfg.joint_ids] #left hip right hip, left knee right knee
   

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids]

    # print(f"joint_ids: {asset_cfg.joint_ids}")
    # print(f"body_ids: {sensor_cfg.body_ids}")
    # print(f"body_names: {sensor_cfg.body_names}")

   # print(f"joint_angles: {hip_roll_joint_angles}")
  #  print(f"joint_angles.shape: {hip_roll_joint_angles.shape}")

    #print(f"forces: {forces}")
   # print(f"forces shape: {forces.shape}")

    left_hip_vel = hip_roll_joint_vel[:,0]
    left_knee_vel = hip_roll_joint_vel[:,2]


    right_hip_vel = hip_roll_joint_vel[:,1]
    right_knee_vel = hip_roll_joint_vel[:,3]

    


    left_foot_x_force = torch.abs(forces[:, 0, 0])
    left_foot_z_force = forces[:, 0, 2]

    right_foot_x_force = torch.abs(forces[:, 1, 0])
    right_foot_z_force = forces[:, 1, 2]


    left_collision = (left_foot_z_force < 100.0) & (left_foot_x_force > 5.0)
    right_collision = (right_foot_z_force < 100.0) & (right_foot_x_force > 5.0)


    left_reward = left_collision * (
            torch.clamp(left_knee_vel, min=0.0) + torch.clamp(-left_hip_vel, min=0.0)
        )

    right_reward = right_collision * (
            torch.clamp(right_knee_vel, min=0.0) + torch.clamp(-right_hip_vel, min=0.0)
        )

    reward = left_reward + right_reward

    return reward



class foot_collision_joint_movement_latched(ManagerTermBase):
    """Keep the collision-escape reward active for a short number of steps."""

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._left_latch = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
        self._right_latch = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)

    def reset(self, env_ids: torch.Tensor | None = None) -> dict[str, float]:
        if env_ids is None:
            self._left_latch.zero_()
            self._right_latch.zero_()
        else:
            self._left_latch[env_ids] = 0
            self._right_latch[env_ids] = 0
        return {}

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        sensor_cfg: SceneEntityCfg,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        latch_steps: int = 5,
        z_force_threshold: float = 100.0,
        xy_force_threshold: float = 5.0,
        decay: bool = False,
    ) -> torch.Tensor:
        asset: RigidObject = env.scene[asset_cfg.name]
        joint_vel = asset.data.joint_vel[:, asset_cfg.joint_ids]

        contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
        forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids]

        left_hip_vel = joint_vel[:, 0]
        right_hip_vel = joint_vel[:, 1]
        left_knee_vel = joint_vel[:, 2]
        right_knee_vel = joint_vel[:, 3]

        #left_foot_x_force = torch.abs(forces[:, 0, 0])
       # left_foot_y_force = torch.abs(forces[:, 0, 1])
        left_foot_z_force = torch.abs(forces[:, 0, 2])

       # right_foot_x_force = torch.abs(forces[:, 1, 0])
       # right_foot_y_force = torch.abs(forces[:, 1, 1])
        right_foot_z_force = torch.abs(forces[:, 1, 2])

        left_foot_xy_force = torch.linalg.norm(forces[:, 0, :2], dim=1)
        right_foot_xy_force = torch.linalg.norm(forces[:, 1, :2], dim=1)


        left_collision = (left_foot_z_force < z_force_threshold) & (left_foot_xy_force > xy_force_threshold)
        right_collision = (right_foot_z_force < z_force_threshold) & (right_foot_xy_force > xy_force_threshold)

        self._left_latch[left_collision] = latch_steps
        self._right_latch[right_collision] = latch_steps

        left_active = self._left_latch > 0
        right_active = self._right_latch > 0

        if decay and latch_steps > 0:
            left_scale = self._left_latch.float() / float(latch_steps)
            right_scale = self._right_latch.float() / float(latch_steps)
        else:
            left_scale = left_active.float()
            right_scale = right_active.float()

        left_reward = left_scale * (
            torch.clamp(left_knee_vel, min=0.0) + torch.clamp(-left_hip_vel, min=0.0)
        )
        right_reward = right_scale * (
            torch.clamp(right_knee_vel, min=0.0) + torch.clamp(-right_hip_vel, min=0.0)
        )

        self._left_latch[left_active] -= 1
        self._right_latch[right_active] -= 1

        return left_reward + right_reward


def foot_collision_joint_movement_test(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),) -> torch.Tensor:
    

    asset: RigidObject = env.scene[asset_cfg.name]

    hip_roll_joint_vel = asset.data.joint_vel[:, asset_cfg.joint_ids] #left hip right hip, left knee right knee
   

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids]

    # print(f"joint_ids: {asset_cfg.joint_ids}")
    # print(f"body_ids: {sensor_cfg.body_ids}")
    # print(f"body_names: {sensor_cfg.body_names}")

   # print(f"joint_angles: {hip_roll_joint_angles}")
  #  print(f"joint_angles.shape: {hip_roll_joint_angles.shape}")

    #print(f"forces: {forces}")
   # print(f"forces shape: {forces.shape}")

    left_hip_vel = hip_roll_joint_vel[:,0]
    left_knee_vel = hip_roll_joint_vel[:,2]


    right_hip_vel = hip_roll_joint_vel[:,1]
    right_knee_vel = hip_roll_joint_vel[:,3]

    


    left_foot_x_force = torch.abs(forces[:, 0, 0])
    left_foot_z_force = forces[:, 0, 2]

    right_foot_x_force = torch.abs(forces[:, 1, 0])
    right_foot_z_force = forces[:, 1, 2]


    left_collision = (left_foot_z_force < 50.0) & (left_foot_x_force > 20.0)
    right_collision = (right_foot_z_force < 50.0) & (right_foot_x_force > 20.0)


    left_reward = left_collision * (
            torch.clamp(left_knee_vel, min=0.0) + torch.clamp(-left_hip_vel, min=0.0)
        )

    right_reward = right_collision * (
            torch.clamp(right_knee_vel, min=0.0) + torch.clamp(-right_hip_vel, min=0.0)
        )

    reward = left_reward + right_reward

    return left_foot_z_force


 

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

class is_terminated_term_time_out_included(ManagerTermBase):
    """Penalize termination for specific terms that don't correspond to episodic timeouts.

    The parameters are as follows:

    * attr:`term_keys`: The termination terms to penalize. This can be a string, a list of strings
      or regular expressions. Default is ".*" which penalizes all terminations.

    The reward is computed as the sum of the termination terms that are not episodic timeouts.
    This means that the reward is 0 if the episode is terminated due to an episodic timeout. Otherwise,
    if two termination terms are active, the reward is 2.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        # initialize the base class
        super().__init__(cfg, env)
        # find and store the termination terms
        term_keys = cfg.params.get("term_keys", ".*")
        self._term_names = env.termination_manager.find_terms(term_keys)

        self.target_name = "base_velocity"   # or whatever you want


    def __call__(self, env: ManagerBasedRLEnv, term_keys: str | list[str] = ".*") -> torch.Tensor:
        # Return the unweighted reward for the termination terms
        reset_buf = torch.zeros(env.num_envs, device=env.device)
        
        for term in self._term_names:
            # Sums over terminations term values to account for multiple terminations in the same step

            if term == "time_out":
                vel_cmd_term = env.command_manager.get_term(self.target_name)

                term_val = env.termination_manager.get_term(term)
                standing_cmd = vel_cmd_term.is_standing_env
                moving_envs = ~standing_cmd

                term_val *= moving_envs #Don't penalize timing out for environments that are commanded to stand still

                reset_buf += term_val
            
            else:
                reset_buf += env.termination_manager.get_term(term)

        
        return reset_buf.float()

def air_time_vel_penalty(env: ManagerBasedRLEnv, 
                         sensor_cfg: SceneEntityCfg,
                         asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
                         nominal_air_time = .5,
                         lamda: float = 10.0,):
    
   
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    sensor_data = contact_sensor.data
    
    air_time = sensor_data.current_air_time[:, sensor_cfg.body_ids]  # (num_envs, num_bodies, 3)
    #left is 0, right is 1
    #print(f"body names: {sensor_cfg.body_names}")
    #print(f"body names: {sensor_cfg.body_ids}")

    air_time_excess = torch.clamp(air_time - nominal_air_time, min=0.0, max=1)  # (num_envs, num_feet)
    avg_air_time_excess = torch.mean(air_time_excess, dim=1)  # (num_envs,)


    asset: RigidObject = env.scene[asset_cfg.name]
    vel = asset.data.root_lin_vel_b[:, :2]

    vel_mag = torch.linalg.norm(vel, dim=1)

    
    #print(f"penalty {air_time}")
   # print(f"penalty {air_time.shape}")

     # Scale by current planar speed
    scaled_excess = avg_air_time_excess * vel_mag  # (num_envs,)

    # Reward near 1 when behavior is good, decays when excess airtime and speed increase
    reward = torch.exp(-lamda * scaled_excess)

    return reward


    
def forward_distance_reward(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), min_dist: float = 0.15, max_dist: float = 0.30, sharpness: float = 100.0) -> torch.Tensor:
    """Reward lateral distance between left and right parts of the robot."""
    asset: Articulation = env.scene[asset_cfg.name]
    body1_pos = asset.data.body_pos_w[:, asset_cfg.body_ids[0], :]
    body2_pos = asset.data.body_pos_w[:, asset_cfg.body_ids[1], :]
    root_quaternion_w = asset.data.root_link_pose_w[:, 3:7]
    body1_pos_local = quat_apply_inverse(root_quaternion_w, body1_pos - asset.data.root_link_pos_w)
    body2_pos_local = quat_apply_inverse(root_quaternion_w, body2_pos - asset.data.root_link_pos_w)

    # forward separation only
    dist = (torch.abs(body1_pos_local[:, 0]) + torch.abs(body2_pos_local[:, 0])) / 2.0

    #dist = torch.abs(body1_pos_local[:, 0] - body2_pos_local[:, 0])

    #d_min = torch.clamp(dist - min_dist, min=-0.5, max=0.0)
   # d_max = torch.clamp(dist - max_dist, min=0.0, max=0.5)

    #reward = (torch.exp(-sharpness * torch.abs(d_min)) +
             # torch.exp(-sharpness * torch.abs(d_max))) / 2.0
    
    return dist

def com_ahead_of_feet_vel_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    lamda: float = 1.0,
) -> torch.Tensor:
    """
    Reward based on how far the robot CoM is ahead of its feet, scaled by robot speed.

    For each foot:
      contribution = max(x_com - x_foot, 0)

    Then average across feet and multiply by robot velocity magnitude.
    """


    asset: Articulation = env.scene[asset_cfg.name]


    body1_pos = asset.data.body_pos_w[:, asset_cfg.body_ids[0], :]
    body2_pos = asset.data.body_pos_w[:, asset_cfg.body_ids[1], :]

    root_quaternion_w = asset.data.root_link_pose_w[:, 3:7]

    robot_pos = asset.data.root_link_pos_w 

    robot_pos[:, 2] += 2

  


    body1_pos_local = quat_apply_inverse(root_quaternion_w,  body1_pos - robot_pos )
    body2_pos_local = quat_apply_inverse(root_quaternion_w, body2_pos - robot_pos)

    body1_x = body1_pos_local[:, 0]
    body2_x = body2_pos_local[:, 0]

  #  print(f"Left Foot Rel Body Pos: {body1_x}")
   # print(f"Right Foot Rel Body Pos: {body2_x}")

    max_positive_ahead = torch.clamp(torch.max(body1_x, body2_x), min=0.0)

    
    #print(f"size of max_positive_ahead: {max_positive_ahead.shape}")

   # Robot velocity magnitude
    robot_vel = asset.data.root_lin_vel_w                     # [num_envs, 3]
    robot_speed = torch.linalg.norm(robot_vel, dim=1)         # [num_envs]

    reward = max_positive_ahead * robot_speed

   # reward = 1.0 - torch.exp(-lamda * reward)

    return reward

    #return max_positive_ahead

  
