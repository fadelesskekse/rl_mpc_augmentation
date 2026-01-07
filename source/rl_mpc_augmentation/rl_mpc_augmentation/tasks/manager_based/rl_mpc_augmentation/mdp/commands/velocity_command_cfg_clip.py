from __future__ import annotations

from dataclasses import MISSING

from unitree_rl_lab.tasks.locomotion.mdp.commands.velocity_command import UniformLevelVelocityCommandCfg
from isaaclab.utils import configclass


@configclass
class UniformLevelVelocityCommandCfgClip(UniformLevelVelocityCommandCfg):
    clip_threshold = MISSING