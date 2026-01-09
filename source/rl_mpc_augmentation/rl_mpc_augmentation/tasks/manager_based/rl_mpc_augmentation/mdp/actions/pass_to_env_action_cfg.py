# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING


from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

from .pass_to_env_action import PassToEnvironment

@configclass
class PassToEnvironmentCfg(ActionTermCfg):

    class_type: type[ActionTerm] = PassToEnvironment

    num_vars: int = MISSING

    var_names: list[str] = MISSING

    
    #Make sure in the actual action term that we confirm the length
    #var names ='s the num_vars

