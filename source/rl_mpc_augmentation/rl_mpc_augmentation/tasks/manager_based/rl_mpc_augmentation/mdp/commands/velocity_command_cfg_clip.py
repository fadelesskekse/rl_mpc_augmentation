from __future__ import annotations

from dataclasses import MISSING
from isaaclab.utils import configclass
from unitree_rl_lab.tasks.locomotion.mdp.commands.velocity_command import UniformLevelVelocityCommandCfg
#from .velocity_command_clip import UniformVelocityCommandClip
from typing import TYPE_CHECKING
#if TYPE_CHECKING:
from .velocity_command_clip import UniformVelocityCommandClip

@configclass
class UniformLevelVelocityCommandCfgClip(UniformLevelVelocityCommandCfg):
    clip_threshold: float = MISSING
    clip_start_threshold: float = MISSING
    
    class_type: type = UniformVelocityCommandClip