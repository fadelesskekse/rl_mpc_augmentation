# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module containing command generators for the velocity-based locomotion task."""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.managers import CommandTerm

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from .commands_cfg import boolCommandCfg

class boolCommand(CommandTerm):
    
    cfg: boolCommandCfg

    def __init__(self,cfg: boolCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg,env)

        # Initialize the walk_bool tensor for all environments
        self.walk_bool = torch.zeros(self.num_envs, 1, device=self.device)


    @property
    def command(self) -> torch.Tensor:
        """The desired base velocity command in the base frame. Shape is (num_envs, 3)."""
        return self.walk_bool
    
    @property
    def has_debug_vis_implementation(self) -> bool:
        return False
    
    def _resample_command(self, env_ids: Sequence[int]):
        """Resample the walk boolean from Bernoulli distribution for specified environments."""
        # Sample from Bernoulli distribution with probability p
        #print("I am called")
        prob = self.cfg.probability if hasattr(self.cfg, 'probability') else 0.5

        self.walk_bool[env_ids] = torch.bernoulli(
            torch.full((len(env_ids), 1), prob, dtype=torch.float, device=self.device)
        )

    def _update_command(self):
        """Update the command (no-op for boolean command as it doesn't change between resamples)."""
        pass

    def _update_metrics(self):
        """Update metrics for the command (no metrics for boolean command)."""
        pass
    
    def _set_debug_vis_impl(self, debug_vis: bool):
        """Set debug visualization (not implemented for boolean command)."""
        pass
    
    def _debug_vis_callback(self, event):
        """Callback for debug visualization (not implemented for boolean command)."""
        pass
    


    