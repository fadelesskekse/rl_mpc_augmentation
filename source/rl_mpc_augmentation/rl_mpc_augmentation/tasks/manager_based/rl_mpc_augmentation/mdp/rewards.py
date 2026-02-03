# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor


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