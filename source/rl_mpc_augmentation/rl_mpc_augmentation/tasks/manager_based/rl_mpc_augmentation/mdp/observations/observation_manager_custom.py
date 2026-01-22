# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Observation manager for computing observation signals for a given world."""

from __future__ import annotations

import inspect
import numpy as np
import torch
from collections.abc import Sequence
from prettytable import PrettyTable
from typing import TYPE_CHECKING

from isaaclab.utils import class_to_dict, modifiers, noise
from isaaclab.utils.buffers import CircularBuffer

from isaaclab.managers.manager_base import ManagerBase, ManagerTermBase
from isaaclab.managers.manager_term_cfg import ObservationGroupCfg, ObservationTermCfg

#from .manager_base import ManagerBase, ManagerTermBase
#from .manager_term_cfg import ObservationGroupCfg, ObservationTermCfg
from isaaclab.managers import ObservationManager

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    #from isaaclab.managers import ObservationManager


class ObservationManagerCustom(ObservationManager):

    def __init__(self, cfg: object, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        """
    Helper functions.
    """

    def _prepare_terms(self):
        """Prepares a list of observation terms functions."""
        # create buffers to store information for each observation group
        # TODO: Make this more convenient by using data structures.
        self._group_obs_term_names: dict[str, list[str]] = dict()
        self._group_obs_term_dim: dict[str, list[tuple[int, ...]]] = dict()
        self._group_obs_term_cfgs: dict[str, list[ObservationTermCfg]] = dict()
        self._group_obs_class_term_cfgs: dict[str, list[ObservationTermCfg]] = dict()
        self._group_obs_concatenate: dict[str, bool] = dict()
        self._group_obs_concatenate_dim: dict[str, int] = dict()

        self._group_obs_term_history_buffer: dict[str, dict] = dict()
        # create a list to store classes instances, e.g., for modifiers and noise models
        # we store it as a separate list to only call reset on them and prevent unnecessary calls
        self._group_obs_class_instances: list[modifiers.ModifierBase | noise.NoiseModel] = list()

        # make sure the simulation is playing since we compute obs dims which needs asset quantities
        if not self._env.sim.is_playing():
            raise RuntimeError(
                "Simulation is not playing. Observation manager requires the simulation to be playing"
                " to compute observation dimensions. Please start the simulation before using the"
                " observation manager."
            )

        # check if config is dict already
        if isinstance(self.cfg, dict):
            group_cfg_items = self.cfg.items()
        else:
            group_cfg_items = self.cfg.__dict__.items()
        # iterate over all the groups
        for group_name, group_cfg in group_cfg_items:
            # check for non config
            if group_cfg is None:
                continue
            # check if the term is a curriculum term
            if not isinstance(group_cfg, ObservationGroupCfg):
                raise TypeError(
                    f"Observation group '{group_name}' is not of type 'ObservationGroupCfg'."
                    f" Received: '{type(group_cfg)}'."
                )
            # initialize list for the group settings
            self._group_obs_term_names[group_name] = list()
            self._group_obs_term_dim[group_name] = list()
            self._group_obs_term_cfgs[group_name] = list()
            self._group_obs_class_term_cfgs[group_name] = list()
            group_entry_history_buffer: dict[str, CircularBuffer] = dict()
            # read common config for the group
            self._group_obs_concatenate[group_name] = group_cfg.concatenate_terms
            self._group_obs_concatenate_dim[group_name] = (
                group_cfg.concatenate_dim + 1 if group_cfg.concatenate_dim >= 0 else group_cfg.concatenate_dim
            )
            # check if config is dict already
            if isinstance(group_cfg, dict):
                group_cfg_items = group_cfg.items()
            else:
                group_cfg_items = group_cfg.__dict__.items()
            # iterate over all the terms in each group

            obs_term_id = 0
            for term_name, term_cfg in group_cfg_items:

                # skip non-obs settings
                if term_name in [
                    "enable_corruption",
                    "concatenate_terms",
                    "history_length",
                    "flatten_history_dim",
                    "concatenate_dim",
                ]:
                    continue

              #  print(f"term name {term_name}")
              #  print(f"term cfg {term_cfg}")
                # check for non config
                if term_cfg is None:
                    continue
                if not isinstance(term_cfg, ObservationTermCfg):
                    raise TypeError(
                        f"Configuration for the term '{term_name}' is not of type ObservationTermCfg."
                        f" Received: '{type(term_cfg)}'."
                    )
                # resolve common terms in the config
                self._resolve_common_term_cfg(f"{group_name}/{term_name}", term_cfg, min_argc=1)

                # check noise settings
                if not group_cfg.enable_corruption:
                    term_cfg.noise = None
                # check group history params and override terms
                if group_cfg.history_length is not None:
                    term_cfg.history_length = group_cfg.history_length
                    term_cfg.flatten_history_dim = group_cfg.flatten_history_dim
                # add term config to list to list
                self._group_obs_term_names[group_name].append(term_name)
                self._group_obs_term_cfgs[group_name].append(term_cfg)

                # call function the first time to fill up dimensions
                obs_dims = tuple(term_cfg.func(self._env, **term_cfg.params).shape)

                # if scale is set, check if single float or tuple
                if term_cfg.scale is not None:
                    if not isinstance(term_cfg.scale, (float, int, tuple)):
                        raise TypeError(
                            f"Scale for observation term '{term_name}' in group '{group_name}'"
                            f" is not of type float, int or tuple. Received: '{type(term_cfg.scale)}'."
                        )
                    if isinstance(term_cfg.scale, tuple) and len(term_cfg.scale) != obs_dims[1]:
                        raise ValueError(
                            f"Scale for observation term '{term_name}' in group '{group_name}'"
                            f" does not match the dimensions of the observation. Expected: {obs_dims[1]}"
                            f" but received: {len(term_cfg.scale)}."
                        )

                    # cast the scale into torch tensor
                    term_cfg.scale = torch.tensor(term_cfg.scale, dtype=torch.float, device=self._env.device)

                # prepare modifiers for each observation
                if term_cfg.modifiers is not None:
                    # initialize list of modifiers for term
                    for mod_cfg in term_cfg.modifiers:
                        # check if class modifier and initialize with observation size when adding
                        if isinstance(mod_cfg, modifiers.ModifierCfg):
                            # to list of modifiers
                            if inspect.isclass(mod_cfg.func):
                                if not issubclass(mod_cfg.func, modifiers.ModifierBase):
                                    raise TypeError(
                                        f"Modifier function '{mod_cfg.func}' for observation term '{term_name}'"
                                        f" is not a subclass of 'ModifierBase'. Received: '{type(mod_cfg.func)}'."
                                    )
                                mod_cfg.func = mod_cfg.func(cfg=mod_cfg, data_dim=obs_dims, device=self._env.device)

                                # add to list of class modifiers
                                self._group_obs_class_instances.append(mod_cfg.func)
                        else:
                            raise TypeError(
                                f"Modifier configuration '{mod_cfg}' of observation term '{term_name}' is not of"
                                f" required type ModifierCfg, Received: '{type(mod_cfg)}'"
                            )

                        # check if function is callable
                        if not callable(mod_cfg.func):
                            raise AttributeError(
                                f"Modifier '{mod_cfg}' of observation term '{term_name}' is not callable."
                                f" Received: {mod_cfg.func}"
                            )

                        # check if term's arguments are matched by params
                        term_params = list(mod_cfg.params.keys())
                        args = inspect.signature(mod_cfg.func).parameters
                        args_with_defaults = [arg for arg in args if args[arg].default is not inspect.Parameter.empty]
                        args_without_defaults = [arg for arg in args if args[arg].default is inspect.Parameter.empty]
                        args = args_without_defaults + args_with_defaults
                        # ignore first two arguments for env and env_ids
                        # Think: Check for cases when kwargs are set inside the function?
                        if len(args) > 1:
                            if set(args[1:]) != set(term_params + args_with_defaults):
                                raise ValueError(
                                    f"Modifier '{mod_cfg}' of observation term '{term_name}' expects"
                                    f" mandatory parameters: {args_without_defaults[1:]}"
                                    f" and optional parameters: {args_with_defaults}, but received: {term_params}."
                                )

                # prepare noise model classes
                if term_cfg.noise is not None and isinstance(term_cfg.noise, noise.NoiseModelCfg):
                    noise_model_cls = term_cfg.noise.class_type
                    if not issubclass(noise_model_cls, noise.NoiseModel):
                        raise TypeError(
                            f"Class type for observation term '{term_name}' NoiseModelCfg"
                            f" is not a subclass of 'NoiseModel'. Received: '{type(noise_model_cls)}'."
                        )
                    # initialize func to be the noise model class instance
                    term_cfg.noise.func = noise_model_cls(
                        term_cfg.noise, num_envs=self._env.num_envs, device=self._env.device
                    )
                    self._group_obs_class_instances.append(term_cfg.noise.func)

                # create history buffers and calculate history term dimensions
                if term_cfg.history_length > 0:
                    group_entry_history_buffer[term_name] = CircularBuffer(
                        max_len=term_cfg.history_length, batch_size=self._env.num_envs, device=self._env.device
                    )
                    old_dims = list(obs_dims)
                    old_dims.insert(1, term_cfg.history_length)
                    obs_dims = tuple(old_dims)
                    if term_cfg.flatten_history_dim:
                        obs_dims = (obs_dims[0], np.prod(obs_dims[1:]))

             
               #print(f"obs_dims: {obs_dims[1]}")
               # print(f" n_scan before rewrite{self._env.cfg.n_scan}")
                #print(f"obs_term_id {obs_term_id}")
                
                # if obs_term_id == 0 and group_name == "policy": 
                #     if term_name != "scan_dot":
                #         raise TypeError(f"I have the wrong obs term in the 0th obs group slot. I have {term_name} where it should be 'scan_dot' ")
                #     if self._env.cfg.n_scan != obs_dims[1]:
                #         raise ValueError("n_scan in the 0th obs group has changed to something not equal to env.cfg set val. Check obs group order")
                #     self._env.cfg.n_scan = obs_dims[1]
                #     print(f"n_scan {obs_dims[1]}")
                elif obs_term_id == 0 and group_name == "policy":
                    if term_name != "base_lin_vel":
                        raise TypeError(f"I have the wrong obs term in the 1st obs group slot. I have {term_name} where it should be 'base_lin_vel'")
                    if self._env.cfg.n_priv != obs_dims[1]:
                        raise ValueError("n_priv in the 1st obs group has changed to something not equal to env.cfg set val. Check obs group order")
                    self._env.cfg.n_priv = obs_dims[1]
                    print(f"priv {obs_dims[1]}")
                # elif obs_term_id == 2 and group_name == "policy":
                #     if term_name != "priv_latent":
                #         raise TypeError(f"I have the wrong obs term in the 1st obs group slot. I have {term_name} where it should be 'priv_latent'")
                #     if self._env.cfg.n_priv_latent != obs_dims[1]:
                #         raise ValueError("n_priv_latent in the 2nd obs group has changed to something not equal to env.cfg set val. Check obs group order")
                #     self._env.cfg.n_priv_latent = obs_dims[1]
                #     print(f"priv_latent {obs_dims[1]}")
                elif obs_term_id == 1 and group_name == "policy":
                    if term_name != "priv_latent_gains_stiffness":
                        raise TypeError(f"I have the wrong obs term in the 2nd obs group slot. I have {term_name} where it should be 'priv_latent_gains_stiffness'")
                    if self._env.cfg.n_priv_latent_gains_stiffness != obs_dims[1]:
                        raise ValueError("n_priv_latent_gains_stiffness in the 2nd obs group has changed to something not equal to env.cfg set val. Check obs group order")
                    self._env.cfg.n_priv_latent_gains_stiffness = obs_dims[1]
                    print(f"n_priv_latent_gains_stiffness {obs_dims[1]}")

                elif obs_term_id == 2 and group_name == "policy":
                    if term_name != "priv_latent_gains_damping":
                        raise TypeError(f"I have the wrong obs term in the 3rd obs group slot. I have {term_name} where it should be 'priv_latent_gains_damping'")
                    if self._env.cfg.n_priv_latent_gains_stiffness != obs_dims[1]:
                        raise ValueError("n_priv_latent_gains_damping in the 3rd obs group has changed to something not equal to env.cfg set val. Check obs group order")
                    self._env.cfg.n_priv_latent_gains_damping = obs_dims[1]
                    print(f"n_priv_latent_gains_damping {obs_dims[1]}")

                
                elif obs_term_id == 3 and group_name == "policy":
                    if term_name != "priv_latent_mass":
                        raise TypeError(f"I have the wrong obs term in the 4th obs group slot. I have {term_name} where it should be 'priv_latent_mass'")
                    if self._env.cfg.n_priv_latent_mass != obs_dims[1]:
                        raise ValueError("n_priv_latent_mass in the 4th obs group has changed to something not equal to env.cfg set val. Check obs group order")
                    self._env.cfg.n_priv_latent_mass = obs_dims[1]
                    print(f"n_priv_latent_mass {obs_dims[1]}")

                elif obs_term_id == 4 and group_name == "policy":
                    if term_name != "priv_latent_com":
                        raise TypeError(f"I have the wrong obs term in the 5th obs group slot. I have {term_name} where it should be 'priv_latent_com'")
                    if self._env.cfg.n_priv_latent_com != obs_dims[1]:
                        raise ValueError("n_priv_latent_com in the 5th obs group has changed to something not equal to env.cfg set val. Check obs group order")
                    self._env.cfg.n_priv_latent_com = obs_dims[1]
                    print(f"n_priv_latent_com {obs_dims[1]}")

                
                elif obs_term_id == 5 and group_name == "policy":
                    if term_name != "priv_latent_friction":
                        raise TypeError(f"I have the wrong obs term in the 6th obs group slot. I have {term_name} where it should be 'priv_latent_friction'")
                    if self._env.cfg.n_priv_latent_friction != obs_dims[1]:
                        raise ValueError("n_priv_latent_friction in the 6th obs group has changed to something not equal to env.cfg set val. Check obs group order")
                    self._env.cfg.n_priv_latent_friction = obs_dims[1]
                    print(f"n_priv_latent_friction {obs_dims[1]}")

                elif obs_term_id == 6 and group_name == "policy":
                    if term_name != "joint_pos_rel":
                        raise TypeError(f"I have the wrong obs term in the 7th obs group slot. I have {term_name} where it should be 'joint_pos_rel'")
                    if self._env.cfg.n_proprio/2 != obs_dims[1]/self._env.cfg.history_len:
                        raise ValueError("n_proprio in the 7th obs group has changed to something not equal to env.cfg set val. Check obs group order")
                    #self._env.cfg.n_priv_latent = obs_dims[1]
                    print(f"joint_pos_rel {obs_dims[1]}")

                elif obs_term_id == 7 and group_name == "policy":
                    if term_name != "joint_vel_rel":
                        raise TypeError(f"I have the wrong obs term in the 8th obs group slot. I have {term_name} where it should be 'joint_vel_rel'")
                    if self._env.cfg.n_proprio/2 != obs_dims[1]/self._env.cfg.history_len:
                        raise ValueError("n_proprio in the 8th obs group has changed to something not equal to env.cfg set val. Check obs group order")
                    #self._env.cfg.n_priv_latent = obs_dims[1]
                    print(f"joint_vel_rel {obs_dims[1]}")
                
            
                
                """IMPORTANT:
                    
                    First obs should be scan_dot
                    Second obs should be lin vel
                    third obs should be priv_latent_pre_encode
                    4th 
                    """


                self._group_obs_term_dim[group_name].append(obs_dims[1:])

                # add term in a separate list if term is a class
                if isinstance(term_cfg.func, ManagerTermBase):
                    self._group_obs_class_term_cfgs[group_name].append(term_cfg)
                    # call reset (in-case above call to get obs dims changed the state)
                    term_cfg.func.reset()

                obs_term_id += 1
            # add history buffers for each group
            self._group_obs_term_history_buffer[group_name] = group_entry_history_buffer