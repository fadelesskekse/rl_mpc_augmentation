# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import builtins
import torch
from collections.abc import Sequence
from typing import Any

import isaacsim.core.utils.torch as torch_utils
import omni.log
import omni.physx
from isaacsim.core.simulation_manager import SimulationManager
from isaacsim.core.version import get_version

from isaaclab.managers import ActionManager, EventManager, ObservationManager, RecorderManager
from isaaclab.scene import InteractiveScene
from isaaclab.sim import SimulationContext
from isaaclab.sim.utils import attach_stage_to_usd_context, use_stage
from isaaclab.ui.widgets import ManagerLiveVisualizer
from isaaclab.utils.timer import Timer

#from .common import VecEnvObs

from isaaclab.envs.common import VecEnvObs

from isaaclab.envs.manager_based_env_cfg import ManagerBasedEnvCfg
from isaaclab.envs.ui import ViewportCameraController
from isaaclab.envs.utils.io_descriptors import export_articulations_data, export_scene_data

#from .manager_based_env_cfg import ManagerBasedEnvCfg
#from .ui import ViewportCameraController
#from .utils.io_descriptors import export_articulations_data, export_scene_data

from isaaclab.envs.manager_based_env import ManagerBasedEnv

from ..mdp.observations.observation_manager_custom import ObservationManagerCustom
class ManagerBasedEnvCustom(ManagerBasedEnv):

    def __init__(self, cfg: ManagerBasedEnvCfg):
        super().__init__(cfg)

    def load_managers(self):
        """Load the managers for the environment.

        This function is responsible for creating the various managers (action, observation,
        events, etc.) for the environment. Since the managers require access to physics handles,
        they can only be created after the simulator is reset (i.e. played for the first time).

        .. note::
            In case of standalone application (when running simulator from Python), the function is called
            automatically when the class is initialized.

            However, in case of extension mode, the user must call this function manually after the simulator
            is reset. This is because the simulator is only reset when the user calls
            :meth:`SimulationContext.reset_async` and it isn't possible to call async functions in the constructor.

        """
        # prepare the managers
        # -- event manager (we print it here to make the logging consistent)
        print("[INFO] Event Manager: ", self.event_manager)
        # -- recorder manager
        self.recorder_manager = RecorderManager(self.cfg.recorders, self)
        print("[INFO] Recorder Manager: ", self.recorder_manager)
        # -- action manager
        self.action_manager = ActionManager(self.cfg.actions, self)
        print("[INFO] Action Manager: ", self.action_manager)
        # -- observation manager
        self.observation_manager = ObservationManagerCustom(self.cfg.observations, self)
        print("[INFO] Observation Manager:", self.observation_manager)

        # perform events at the start of the simulation
        # in-case a child implementation creates other managers, the randomization should happen
        # when all the other managers are created
        if self.__class__ == ManagerBasedEnv and "startup" in self.event_manager.available_modes:
            self.event_manager.apply(mode="startup")

    def setup_manager_visualizers(self):
        """Creates live visualizers for manager terms."""

        self.manager_visualizers = {
            "action_manager": ManagerLiveVisualizer(manager=self.action_manager),
            "observation_manager": ManagerLiveVisualizer(manager=self.observation_manager),
        }