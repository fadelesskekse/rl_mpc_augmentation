
from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg, TerminationManager
from isaaclab.terrains import TerrainImporter

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv



def terrain_levels_vel_cust(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    vel_cmd_threshold: float = 0.5,
) -> dict:
    # extract the used quantities (to enable type-hinting)
    _asset: Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain
    termination_manager: TerminationManager = env.termination_manager
    command = env.command_manager.get_command("base_velocity")
    term_names = termination_manager.find_terms(".*")

    term_matrix = torch.zeros((len(env_ids), len(term_names)), dtype=torch.bool, device=env.device)
    for idx, term_name in enumerate(term_names):
        term_matrix[:, idx] = termination_manager.get_term(term_name)[env_ids].bool()

    term_counts = term_matrix.sum(dim=1)

    def _only_term_mask(term_name: str) -> torch.Tensor:
        if term_name not in term_names:
            return torch.zeros(len(env_ids), dtype=torch.bool, device=env.device)
        term_idx = term_names.index(term_name)
        return term_matrix[:, term_idx] & (term_counts == 1)

    only_course_complete = _only_term_mask("course_complete")
    only_bad_orientation = _only_term_mask("bad_orientation")
    only_time_out = _only_term_mask("time_out")

    cmd_for_envs = command[env_ids]
    cmd_all_zeros = torch.all(torch.isclose(cmd_for_envs, torch.zeros_like(cmd_for_envs)), dim=1)
    lin_vel_mag = torch.norm(cmd_for_envs[:, :2], dim=1)
    cmd_above_threshold = lin_vel_mag > vel_cmd_threshold

    move_up = torch.zeros(len(env_ids), dtype=torch.bool, device=env.device)
    move_down = torch.zeros(len(env_ids), dtype=torch.bool, device=env.device)

    move_up |= only_course_complete
    move_down |= only_bad_orientation

    # timeout-only nested branch:
    # if all-zero command -> stay (no-op)
    # else if command above threshold -> move up
    timeout_go_up = only_time_out & (~cmd_all_zeros) & cmd_above_threshold
    move_up |= timeout_go_up

    terrain.update_env_origins(env_ids, move_up, move_down)

    terrain_metrics: dict = {}

    # #Uncomment during inference
    #     #Change fraction of total terrain to cover to 1 (instead of .5) during inference
    #     #Set a normal velocity and episode time
    #     #Yields average stair difficulty successfully and entirely traversed at a given speed 

    # num_col = terrain.terrain_origins.shape[1] #num of terrains
    # # Calculate the average terrain level for each terrain type
   
    # for terrain_type_id in range(num_col):
    #     # Find all terrain levels that correspond to the current terrain type
    #     matching_levels = []
    #     for level, t_type in zip(terrain.terrain_levels, terrain.terrain_types):
    #         if t_type.item() == terrain_type_id:
    #             matching_levels.append(level.item())
    #     # Compute the average if there are any matching levels
    #     if matching_levels:
    #         avg_level = sum(matching_levels) / len(matching_levels)

    #     else:
    #         avg_level = 0  # or float('nan') if you want to indicate no data
        
    #     terrain_metrics[f"terrain {terrain_type_id}"] = avg_level
        

    terrain_metrics["total_average_difficulty"] = torch.mean(terrain.terrain_levels.float())


    return terrain_metrics


def terrain_levels_vel_v2(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int], 
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    threshold: float = .5 #.5 for training
) -> dict:
    """Curriculum based on the distance the robot walked.

    This term is used to increase the difficulty of the terrain when the robot walks far enough 
    (beyond threshold * terrain size) and decrease the difficulty otherwise.

    .. note::
        It is only possible to use this term with the terrain type ``generator``. For further information
        on different terrain types, check the :class:`isaaclab.terrains.TerrainImporter` class.

    Returns:
        The mean terrain level for the given environment ids.
    """
    
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain
    
    # compute the distance the robot walked
    distance = torch.norm(asset.data.root_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1)
    
    # robots that walked far enough progress to harder terrains
    move_up = distance >= terrain.cfg.terrain_generator.size[0] * threshold
    
    # robots that didn't walk far enough go to simpler terrains
    move_down = distance < terrain.cfg.terrain_generator.size[0] * threshold
    
    terrain.update_env_origins(env_ids, move_up, move_down)
    
    terrain_metrics: dict = {}
    terrain_metrics["total_average_difficulty"] = torch.mean(terrain.terrain_levels.float())
    
    return terrain_metrics


def lin_vel_cmd_levels_cust(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str = "track_lin_vel_xy",
) -> torch.Tensor:

    command_term = env.command_manager.get_term("base_velocity")
    ranges = command_term.cfg.ranges
    limit_ranges = command_term.cfg.limit_ranges

    reward_term = env.reward_manager.get_term_cfg(reward_term_name)

  #  print(f"Shape of reward_term: {env.reward_manager._episode_sums[reward_term_name][env_ids].shape}")
    reward = torch.mean(env.reward_manager._episode_sums[reward_term_name][env_ids]) / env.max_episode_length_s

    #print(f"Lengh of Ids: {len(env_ids)}. Shape of reward: {reward.shape}")
    #print(f"env_ids in lin_vel_curr: {env_ids}")
   # print(f"reward weight {reward_term.weight}")
    #print(f"reward: {reward}")
 

    if env.common_step_counter % env.max_episode_length == 0:
        if reward > reward_term.weight * 0.8:

            print(f"I made it here for ids: {env_ids}")
            delta_command = torch.tensor([-0.1, 0.1], device=env.device)
            ranges.lin_vel_x = torch.clamp(
                torch.tensor(ranges.lin_vel_x, device=env.device) + delta_command,
                limit_ranges.lin_vel_x[0],
                limit_ranges.lin_vel_x[1],
            ).tolist()
            ranges.lin_vel_y = torch.clamp(
                torch.tensor(ranges.lin_vel_y, device=env.device) + delta_command,
                limit_ranges.lin_vel_y[0],
                limit_ranges.lin_vel_y[1],
            ).tolist()

    return torch.tensor(ranges.lin_vel_x[1], device=env.device)


def ang_vel_cmd_levels_cust(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str = "track_ang_vel_z",
) -> torch.Tensor:
    command_term = env.command_manager.get_term("base_velocity")
    ranges = command_term.cfg.ranges
    limit_ranges = command_term.cfg.limit_ranges

    reward_term = env.reward_manager.get_term_cfg(reward_term_name)
    reward = torch.mean(env.reward_manager._episode_sums[reward_term_name][env_ids]) / env.max_episode_length_s

    if env.common_step_counter % env.max_episode_length == 0:
        if reward > reward_term.weight * 0.8:
            delta_command = torch.tensor([-0.1, 0.1], device=env.device)
            ranges.ang_vel_z = torch.clamp(
                torch.tensor(ranges.ang_vel_z, device=env.device) + delta_command,
                limit_ranges.ang_vel_z[0],
                limit_ranges.ang_vel_z[1],
            ).tolist()

    return torch.tensor(ranges.ang_vel_z[1], device=env.device)
