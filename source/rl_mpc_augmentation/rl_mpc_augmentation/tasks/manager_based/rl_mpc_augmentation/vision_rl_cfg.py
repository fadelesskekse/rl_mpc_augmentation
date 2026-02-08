# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
import torch

from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab.sensors import ContactSensorCfg #CameraCfg, TiledCameraCfg
from isaaclab.sensors.ray_caster import RayCasterCfg, patterns
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_from_euler_xyz

from . import mdp

from assets.g1.g1_bm import G1_BM_CFG # pyright: ignore[reportMissingImports]

PLAYGROUND = terrain_gen.TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=2.0,
    border_height=0,
    num_rows=10,
    num_cols=4,
    horizontal_scale=.025,
    vertical_scale=.025,#.005
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    sub_terrains={
        "stairs_up": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(step_height_range = (.01,.17),
                                                        step_width = .3,
                                                        border_width=1.0,
                                                        platform_width=1.5,
                                                        proportion = .25,),

        "stairs_down": terrain_gen.MeshPyramidStairsTerrainCfg(step_height_range = (.01,.17),
                                                        step_width = .3,
                                                        border_width=1.0,
                                                        platform_width=1.25,
                                                        proportion = .25,),

        "rough_flat": terrain_gen.MeshRandomGridTerrainCfg(proportion = .25,
                                                           grid_height_range = (.01,.125),
                                                           grid_width = .75,
                                                           platform_width = .75,),

        "stepping_stones": terrain_gen.HfSteppingStonesTerrainCfg(proportion=.25,
                                                                  stone_height_max = 0,
                                                                  stone_distance_range= (.025,.3),
                                                                  stone_width_range= (.3,.45),
                                                                  platform_width=.75,
                                                                  border_width=1.0,
                                                                  holes_depth=-1),
                                                        
    },
)

@configclass
class RlMpcAugmentationSceneCfg(InteractiveSceneCfg):
    
    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",  # "plane", "generator"
        terrain_generator=PLAYGROUND,  # None, ROUGH_TERRAINS_CFG
        max_init_terrain_level=PLAYGROUND.num_rows - 1,
        #max_init_terrain_level=0,
        
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
        
    )

    # robot
    robot: ArticulationCfg = G1_BM_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)

    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )

    # tiled_camera: TiledCameraCfg = TiledCameraCfg(
    # #prim_path="/World/envs/env_.*/Camera",
    # prim_path="{ENV_REGEX_NS}/Robot/torso_link/d435_link/depth_camera",
    # offset=TiledCameraCfg.OffsetCfg(pos=(0, 0.0, 0), rot=(1, 0.0, 0, 0.0), convention="world"),
    # data_types=["depth"],
    # #width=848,
    # #height=480,
    # width=424,
    # height=240,

    # spawn=sim_utils.PinholeCameraCfg(
    # focal_length=.193, f_stop = 0.0, focus_distance=1, horizontal_aperture=.384, vertical_aperture=.24,clipping_range=(0.001, 1000000.0)
    # ),
    # )

    # # sensors
    scan_dot = RayCasterCfg(
    prim_path="{ENV_REGEX_NS}/Robot/torso_link",
    offset=RayCasterCfg.OffsetCfg(
       pos=(.9, 0.01753, 0.42987),#0.0576235, 0.01753, 0.42987
       #rot=(0.9238795, 0, 0.3826834, 0)
        rot=tuple(quat_from_euler_xyz(
            torch.tensor([0]), 
            torch.tensor([0]), #-0.8307767239493009
            torch.tensor([0])
        )[0].tolist())
    ),
    ray_alignment="yaw",
    max_distance=4,



    # pattern_cfg=patterns.PinholeCameraPatternCfg(
    # focal_length=.193,
    # horizontal_aperture=.384,
    # vertical_aperture=.24,
    # width=424,
    # height=240,),

    pattern_cfg=patterns.GridPatternCfg(
        resolution=.196, #in meters, length then width #was .196
        size=(1.625,2.6), #in meters,length then width
    ),

    debug_vis=False,
    #update_period=1/50,
    mesh_prim_paths=["/World/ground"],
    )

@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel,params={"threshold": .45})
    lin_vel_cmd_levels = CurrTerm(mdp.lin_vel_cmd_levels)

##
# MDP settings
##

@configclass 
class CommandsCfg:

    # base_velocity = mdp.UniformLevelVelocityCommandCfg(
    #     asset_name="robot",
    #     resampling_time_range=(10, 10),
    #     rel_standing_envs=0,
    #     rel_heading_envs=1.0,
    #     heading_command=False,

    #     debug_vis=False,
        
    #     ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
    #         lin_vel_x=(0, .1), lin_vel_y=(0.0, 0.0), ang_vel_z=(0.0, 0.0)
    #     ),
    #     limit_ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
    #         lin_vel_x=(0, 1), lin_vel_y=(0.0, 0.0), ang_vel_z=(0.0, 0.0)
    #     ),

    #     # limit_ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
    #     #     lin_vel_x=(1, 1), lin_vel_y=(0.0, 0.0), ang_vel_z=(0.0, 0.0)
    #     # ),
    # )

    base_velocity = mdp.UniformLevelVelocityCommandCfgClip(
        asset_name="robot",
        resampling_time_range=(2, 12),
        rel_standing_envs=0.05,
        rel_heading_envs=1.0,
        heading_command=False,
        debug_vis=False,
        clip_threshold=.5,
        clip_start_threshold=1,
        
        ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(0, .1), lin_vel_y=(0.0, 0.0), ang_vel_z=(0.0, 0.0)
        ),
        limit_ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(0, 1), lin_vel_y=(0.0, 0.0), ang_vel_z=(0.0, 0.0)
        ),

        # limit_ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
        #     lin_vel_x=(1, 1), lin_vel_y=(0.0, 0.0), ang_vel_z=(0.0, 0.0)
        # ),
    )

@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    #Justification: MPC Torque output + joint position output.
    # It doesn't really matter if NN outputs positions or torques
    # as it will learn to scale the position outputs as needed.
    JointPositionAction = mdp.JointPositionActionCfg(
        asset_name="robot", 
        joint_names=[".*"], 
        scale=.25, #0.25
        use_default_offset=True,
        #clip={"a":(1,1)},
        
    )

    gait_cycle = mdp.PassToEnvironmentCfg(
        asset_name="robot",
        num_vars = 1,
        var_names = ["gait_cycle",],
        clip = {"gait_cycle": (.3, 2)}
        )

@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""


    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        """IMPORTANT:
        
        First obs should be scan_dot
        Second obs should be lin vel
        third obs should be priv_latent_pre_encode
        4th should be hist proprio
        """

        #######EXTREME PARKOUR OBS####################

        # # # observation terms (order preserved)
        scan_dot = ObsTerm(func=mdp.scan_dot, 
                scale = 1,
                params={
                    "sensor_cfg": SceneEntityCfg("scan_dot",),
                    #"asset_cfg": SceneEntityCfg("robot", body_names=".*torso_link.*")
                },
                history_length=0
        )



        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, history_length=0) #Will be replaced by estimator output during rollouts, and will be used as ground truth during learning phase
        
        priv_latent_gains_stiffness = ObsTerm(func=mdp.priv_latent_gains_stiffness, history_length=0,scale=1)
        priv_latent_gains_damping = ObsTerm(func=mdp.priv_latent_gains_damping, history_length=0,scale=1)
        priv_latent_mass = ObsTerm(func=mdp.priv_latent_mass,params={"asset_cfg": SceneEntityCfg("robot", body_names="torso_link")}, history_length=0,scale=1,)
        priv_latent_com = ObsTerm(func=mdp.priv_latent_com, history_length=0)
        priv_latent_friction= ObsTerm(func=mdp.priv_latent_friction, history_length=0)

       # priv_latent = ObsTerm(func=mdp.priv_latent, history_length=0)

        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel,history_length=10, noise=Unoise(n_min=-0.01, n_max=0.01),) #updated in post init 
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, history_length=10, scale=0.05, noise=Unoise(n_min=-1.5, n_max=1.5),)

        ########END EXTREME PARKOUS OBS#################

        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2, history_length = 5,noise=Unoise(n_min=-0.2, n_max=0.2),)
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05),history_length=5)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"},history_length=5)
        last_action = ObsTerm(func=mdp.last_action,history_length=5)

        gait_phase = ObsTerm(func = mdp.gait_cycle_var, params={
                                                            "offset": [0,.5],
                                                            },history_length=5)


        # gait_phase = ObsTerm(func = mdp.gait_cycle, params={"period": .6,
        #                                                     "offset": [0,.5],
        #                                                     })

        def __post_init__(self):
            #self.history_length = 5
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()

    @configclass
    class CriticCfg(ObsGroup):
        """Observations for critic group."""

        #######EXTREME PARKOUR OBS####################

        # # # # observation terms (order preserved)
        scan_dot = ObsTerm(func=mdp.scan_dot, 
                scale = .1,
                params={
                    "sensor_cfg": SceneEntityCfg("scan_dot",),
                   # "asset_cfg": SceneEntityCfg("robot", body_names=".*torso_link.*"),
                },
                history_length=0
        )

        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, history_length=0) #not sensitive
       # priv_latent = ObsTerm(func=mdp.priv_latent, history_length=0)
        priv_latent_gains_stiffness = ObsTerm(func=mdp.priv_latent_gains_stiffness, history_length=0,scale=1)#not sensitive
        priv_latent_gains_damping = ObsTerm(func=mdp.priv_latent_gains_damping, history_length=0,scale=1)#not sensitive
        priv_latent_mass = ObsTerm(func=mdp.priv_latent_mass,params={"asset_cfg": SceneEntityCfg("robot", body_names="torso_link")}, history_length=0,scale=1)#not sensitive
        priv_latent_com = ObsTerm(func=mdp.priv_latent_com, history_length=0,scale=1)#not sensitive
        priv_latent_friction= ObsTerm(func=mdp.priv_latent_friction, history_length=0,scale=1)#not sensitive

        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel,history_length=10)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, history_length=10,scale=0.05,)

        ########END EXTREME PARKOUS OBS#################

        
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2,history_length=5) #sensitive
        projected_gravity = ObsTerm(func=mdp.projected_gravity,history_length=5) #sensitive
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"},history_length=5) #sensitive
        last_action = ObsTerm(func=mdp.last_action,history_length=5)
        # gait_phase = ObsTerm(func = mdp.gait_cycle, params={"period": .6,
        #                                                     "offset": [0,.5],
        #                                                     })
        gait_phase = ObsTerm(func = mdp.gait_cycle_var, params={ #NOT SENSITIVE
                                                            "offset": [0,.5],
                                                            },history_length=5)


       # def __post_init__(self):
            #self.history_length = 5

    # privileged observations
    critic: CriticCfg = CriticCfg()




@configclass
class EventCfg:
    """Configuration for events."""

    # startup
    #(1) Randomize friction parameters of all rigid bodies on robot
    #Justification: Domain Randomization
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.3, 1), #was .3 to 1
            "dynamic_friction_range": (0.3, 1),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
            "make_consistent":True,
        },
    )

    # (2) Randomize mass of base link
    #Justification: Domain Randomization
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        #interval_range_s=(.48, .48),
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "mass_distribution_params": (.8, 1.2), #was .8 to 1.2
            "operation": "scale",
        },
    )

    # change_base_com = EventTerm(
    #     func=mdp.randomize_rigid_body_com,
    #     mode="startup",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
    #         "com_range": {"x": (-0.075, .075),
    #                       "y": (-0.075, .075),
    #                       "z": (-0.075, .075),}
     
    #     },
    # )

    change_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params":(.8,1.2), #was .8 to 1.2
            "damping_distribution_params": (.8,1.2),
            "operation": "scale"
        },
    )

 
 



    # # reset
    # base_external_force_torque = EventTerm(
    #     func=mdp.apply_external_force_torque,
    #     mode="reset",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
    #         "force_range": (0.0, 0.0),
    #         "torque_range": (-0.0, 0.0),
    #     },
    # )

    # (3) Reset base position and orientation upon reset
    #Justification: Standard practice to randomize initial pose slightly
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform_grouped_yaws,
        mode="reset",
        params={
            "pose_range": {"x": (0, 0), "y": (0, 0), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-.025, 0.025),
                "y": (-0.025, 0.025),
                "z": (-0.025, 0.025),
                "roll": (-0.087, 0.087),
                "pitch": (-0.087, 0.087),
                "yaw": (-0.087, 0.087),
            },
            "max_jitter": (-10,10),
        },
    )

    # (4) Reset robot joint positions and velocities to some scaled value of the default values given in articulation_cfg
    #Justification: Standard practice to randomize initial joint positions slightly
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (1.0, 1.0),
            "velocity_range": (-1.0, 1.0),
        },
    )

    # (5) Robust against pushes at random intervals
    #Justification: Improve robustness by pushing robot at random
    # interval
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity_delayed,
        mode="interval",
        interval_range_s=(.25, 7),
        params={"velocity_range": {"x": (-.5, .5), "y": (-.5, .5)},"curr_lim":.5} #was +-.5
               
    )




@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # (1) Constant running reward
    alive = RewTerm(func=mdp.is_alive, weight=.15)

    # scan_target = RewTerm(
    #     func=mdp.scan_dot_avg_reward,
    #     weight=.25,
    #     params={
    #         "sensor_cfg": SceneEntityCfg("scan_dot"),
    #         "target": 0.4,   # Target normalized distance
    #         "std": 0.158,      # Controls reward sharpness
    #     },
    # )


    
    # (2) Track yaw frame linear velocity commands in XY Plane
    #Justification: Need to track a body frame velocity
    track_lin_vel_xy = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=3,#was 1.5
        params={"command_name": "base_velocity", 
                "std": math.sqrt(0.25)},
    )

    # bad_orientation = RewTerm(func=mdp.is_terminated_term,
    #                           params={"term_keys": "bad_orientation"},
    #                           weight=-3,
    #                           )



    # (3) Track yaw angular velocity command
    #Justification: Need to track 0 angular velocity to 
        #keep robot pointed in direction it started in 
    track_ang_vel_z = RewTerm(
        func=mdp.track_ang_vel_z_exp, 
        weight=2.5, 
        params={"command_name": "base_velocity", 
                "std": math.sqrt(0.25)}
    )

    # (4) Minimize angular velocity in XY plane
    #Justification: Encourage robot to not tip over
    base_angular_velocity_xy = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    #base_linear_velocity = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)

    neg_z_vel = RewTerm(func=mdp.lin_vel_z_negative_l2, weight=-3.0)
    #z_accel = RewTerm(func=mdp.body_lin_acc_l2_z,weight=-1)
    #lin_accel = RewTerm(func=mdp.body_lin_acc_l2,weight = -1/3000)
    #pos_z_vel = RewTerm(func=mdp.lin_vel_z_positive_l2, weight=-2.0)

    # (5) Minimize joint effort, action_rate, energy, and penalize hitting joint limit
    # Justification: Keep energy minimal, concurrent actions similar, minimize fast joints
    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-0.001)
    joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.05)

    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-5.0)
    dof_vel_limits = RewTerm(func=mdp.joint_vel_limits, weight=-5.0, params={"soft_ratio": .9})

    energy = RewTerm(func=mdp.energy, weight=-2e-5)

    gait_deviation = RewTerm(
        func=mdp.gait_deviation,
        weight = 1,
        params={
            "nominal": .5
        }
    )

    #(6) Joint deviation from defaults
    # Justification: Encourage robot to stay near nominal pose
    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.45,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_shoulder_.*_joint",
                    ".*_elbow_joint",
                    ".*_wrist_.*",
                ],
            )
        },
    )
    joint_deviation_waists = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-1,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    "waist.*",
                ],
            )
        },
    )
    joint_deviation_legs = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-1.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_roll_joint", ".*_hip_yaw_joint"])},
    )

    joint_deviation_ankle = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-.05,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_ankle_pitch_joint"])},
    )

    #(7) Flat Orientation
    #Justification: Promote robot to be upright
    # -- robot
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-15.0) #was -9
    #base_height = RewTerm(func=mdp.base_height_l2, weight=-10, params={"target_height": 0.78})

    #() Minimize Ankle Torque
    ankle_torque_min = RewTerm(
        func=mdp.ankle_torque_min,
        weight = -.005,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_ankle_pitch_joint"])}
    )

    #(8) Induce a gait pattern
    #Justification: End-to-end RL needs help to learn a gait
    # -- feet #last_action
    # gait = RewTerm(
    #     func=mdp.feet_gait,
    #     weight=0.5,
    #     params={
    #         "period": 0.6,
    #         "offset": [0.0, 0.5],
    #         "threshold": 0.55,
    #         "command_name": "base_velocity",
    #         "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),
    #     },
    # )

    gait = RewTerm(
        func=mdp.gait,
        weight=1,
        params={
            #"period": 0.6,
            "offset": [0.0, 0.5],
            "threshold": 0.55,
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),
        },
    )

    # (9) Penalize foot slip
    #Justification: Penalizes foot velocity during stance.
        # Encourages stable contact when foot is on ground
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.2,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll.*"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),
        },
    )

    # (10) Foot clearance reward hard-coded for end-to-end RL
    #Justification: Encourages gait with end-to-end blind RL
        #Try removing when not blind.
        #
    # feet_clearance = RewTerm(
    #     func=mdp.foot_clearance_reward,
    #     weight=1.0,
    #     params={
    #         "std": 0.05,
    #         "tanh_mult": 2.0,
    #         "target_height": 0.2,
    #         "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll.*"),
    #     },
    # )

    #(11) Penalizes contact with body parts other than feet into anything. 
    #Justification: Encourage only feet to make contact with ground
    # -- other

    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1,
        params={
            "threshold": 1,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["(?!.*ankle.*).*"]),
        },
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    #base_height = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": 0.2})
    bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 0.8})
    course_complete = DoneTerm(func=mdp.sub_terrain_out_of_bounds, params={
        "distance_buffer": 0,
    })



##
# Environment configuration
##



@configclass
class RlMpcAugmentationEnvCfg(ManagerBasedRLEnvCfg):
    # Scene settings
    scene: RlMpcAugmentationSceneCfg = RlMpcAugmentationSceneCfg(num_envs=4096, env_spacing=4.0)
    # Basic settings
    commands: CommandsCfg = CommandsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    n_scan:int = 126 #used for exeception raising on obsGroup order. #was 126
    n_priv:int = 3 #used for exeception raising on obsGroup order.

    n_priv_latent_gains_stiffness = 29
    n_priv_latent_gains_damping = 29
    n_priv_latent_mass = 1
    n_priv_latent_com = 3
    n_priv_latent_friction = 4

    n_priv_latent:int = n_priv_latent_gains_stiffness +  n_priv_latent_gains_damping + n_priv_latent_mass + n_priv_latent_com + n_priv_latent_friction#66. used for exeception raising on obsGroup order.
    n_proprio:int = 29 + 29
    history_len:int = 10
    history_len_for_regular_proprio_actor:int = 5

    #critic observations take in raw priv info and raw n_scan
    num_critic_obs:int = n_scan + (history_len)*n_proprio + n_priv_latent + n_priv 

    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""
        # general settings

        # self.observations.policy.joint_pos_rel.history_length=self.history_len
        # self.observations.policy.joint_vel_rel.history_length=self.history_len
        # self.observations.critic.joint_pos_rel.history_length=self.history_len
        # self.observations.critic.joint_vel_rel.history_length=self.history_len

        self.decimation = 4
        self.episode_length_s = 12
        # viewer settings
        self.viewer.eye = (8.0, 0.0, 5.0)
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation

        self.sim.physx.gpu_collision_stack_size = 150_000_000
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15

        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        self.scene.contact_forces.update_period = self.sim.dt
        

        # check if terrain levels curriculum is enabled - if so, enable curriculum for terrain generator
        # this generates terrains with increasing difficulty and is useful for training
        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False


@configclass
class RobotPlayEnvCfg(RlMpcAugmentationEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 100
        self.episode_length_s = 12

        self.use_hist_encoder = True
        self.use_estimator = True


        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges

