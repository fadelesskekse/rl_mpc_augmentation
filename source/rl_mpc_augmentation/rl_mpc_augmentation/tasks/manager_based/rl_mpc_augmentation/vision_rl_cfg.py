# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

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
from isaaclab.sensors import ContactSensorCfg#, CameraCfg, TiledCameraCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.utils import configclass
from isaaclab.sensors.ray_caster import RayCasterCfg, patterns
from . import mdp

from assets.g1.g1_bm import G1_BM_CFG # pyright: ignore[reportMissingImports]

PLAYGROUND = terrain_gen.TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=2.0,
    num_rows=10,
    num_cols=3,
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
                                                        proportion = .33,),

        "stairs_down": terrain_gen.MeshPyramidStairsTerrainCfg(step_height_range = (.01,.17),
                                                        step_width = .3,
                                                        border_width=1.0,
                                                        platform_width=1.25,
                                                        proportion = .33,),

        "rough_flat": terrain_gen.MeshRandomGridTerrainCfg(proportion = .33,
                                                           grid_height_range = (.01,.125),
                                                           grid_width = .75,
                                                           platform_width = .75,),

        # "stepping_stones": terrain_gen.HfSteppingStonesTerrainCfg(proportion=.25,
        #                                                           stone_height_max = 0,
        #                                                           stone_distance_range= (.025,.3),
        #                                                           stone_width_range= (.3,.45),
        #                                                           platform_width=.75,
        #                                                           border_width=1.0,
        #                                                           holes_depth=-1),
                                                        
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

    # sensors
    scan_dot = RayCasterCfg(
    prim_path="{ENV_REGEX_NS}/Robot/torso_link/d435_link",
    #offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
    ray_alignment="yaw",

    # pattern_cfg=patterns.PinholeCameraPatternCfg(
    # focal_length=.193,
    # horizontal_aperture=.384,
    # vertical_aperture=.24,
    # width=424,
    # height=240,),

    pattern_cfg=patterns.GridPatternCfg(
        resolution=.196, #in meters, length then width
        size=(1.625,2.6), #in meters,length then width
    ),

    debug_vis=True,
    update_period=1/60,
    mesh_prim_paths=["/World/ground"],
    )

@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel,params={"threshold": .5})
    lin_vel_cmd_levels = CurrTerm(mdp.lin_vel_cmd_levels)

##
# MDP settings
##

@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    # Uniform level velocity command
    # Justification: Provides agent with body frame velocity commands to track 
    base_velocity = mdp.UniformLevelVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10, 10),
        rel_standing_envs=0,
        rel_heading_envs=1.0,
        heading_command=False,
        debug_vis=False,
        
        ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(0, .1), lin_vel_y=(0.0, 0.0), ang_vel_z=(0.0, 0.0)
        ),
        limit_ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(0, 1), lin_vel_y=(0.0, 0.0), ang_vel_z=(0.0, 0.0)
        ),
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
        clip = {"gait_cycle": (.5, 2)}
        )

@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        scan_dot = ObsTerm(func=mdp.scan_dot, 
                scale = .2,
                params={
                    #"asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll.*"),
                    "sensor_cfg": SceneEntityCfg("scan_dot",),
                },
        )


        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2, noise=Unoise(n_min=-0.2, n_max=0.2))

        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))

        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})

        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05, noise=Unoise(n_min=-1.5, n_max=1.5))

        last_action = ObsTerm(func=mdp.last_action)

        #base_lin_vel = ObsTerm(func=mdp.base_lin_vel,noise=Unoise(n_min=-0.01, n_max=0.01))
        #base_z_pos = ObsTerm(func=mdp.base_pos_z, noise=Unoise(n_min=-0.2, n_max=0.2))

        gait_phase = ObsTerm(func = mdp.gait_cycle_var, params={
                                                            "offset": [0,.5],
                                                            })


        # gait_phase = ObsTerm(func = mdp.gait_cycle, params={"period": .6,
        #                                                     "offset": [0,.5],
        #                                                     })

        def __post_init__(self):
            self.history_length = 5
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()

    @configclass
    class CriticCfg(ObsGroup):
        """Observations for critic group."""


        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_z_pos = ObsTerm(func=mdp.base_pos_z)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05)
        last_action = ObsTerm(func=mdp.last_action)
        # gait_phase = ObsTerm(func = mdp.gait_cycle, params={"period": .6,
        #                                                     "offset": [0,.5],
        #                                                     })
        gait_phase = ObsTerm(func = mdp.gait_cycle_var, params={
                                                            "offset": [0,.5],
                                                            })


        def __post_init__(self):
            self.history_length = 5

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
            "static_friction_range": (0.3, 1.0),
            "dynamic_friction_range": (0.3, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    # (2) Randomize mass of base link
    #Justification: Domain Randomization
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "mass_distribution_params": (-1.0, 3.0),
            "operation": "add",
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
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (0, 0), "y": (0, 0), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
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
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(5.0, 5.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    )




@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # (1) Constant running reward
    alive = RewTerm(func=mdp.is_alive, weight=.15)
    
    # (2) Track yaw frame linear velocity commands in XY Plane
    #Justification: Need to track a body frame velocity
    track_lin_vel_xy = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=1.0,
        params={"command_name": "base_velocity", 
                "std": math.sqrt(0.25)},
    )



    # (3) Track yaw angular velocity command
    #Justification: Need to track 0 angular velocity to 
        #keep robot pointed in direction it started in 
    track_ang_vel_z = RewTerm(
        func=mdp.track_ang_vel_z_exp, 
        weight=0.5, 
        params={"command_name": "base_velocity", 
                "std": math.sqrt(0.25)}
    )

    # (4) Minimize angular velocity in XY plane
    #Justification: Encourage robot to not tip over
    base_angular_velocity_xy = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    base_linear_velocity = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    # (5) Minimize joint effort, action_rate, energy, and penalize hitting joint limit
    # Justification: Keep energy minimal, concurrent actions similar, minimize fast joints
    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-0.001)
    joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.05)

    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-5.0)
   # dof_vel_limits = RewTerm(func=mdp.joint_vel_limits, weight=-5.0, params={"soft_ratio": .9})

    energy = RewTerm(func=mdp.energy, weight=-2e-5)

    gait_deviation = RewTerm(
        func=mdp.gait_deviation,
        weight = 0.25,
        params={
            "nominal": .5
        }
    )

    #(6) Joint deviation from defaults
    # Justification: Encourage robot to stay near nominal pose
    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.13,
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
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_roll_joint", ".*_hip_yaw_joint"])},
    )

    joint_deviation_ankle = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_ankle_pitch_joint"])},
    )

    #(7) Flat Orientation
    #Justification: Promote robot to be upright
    # -- robot
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-5.0)
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
        weight=0.5,
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

    n_scan:int = 132 #not an observation dimension. Don't specify in obs group.
    n_priv:int = 3+3 +3 #is an obs dimension
    n_priv_latent = 4 + 1 + 12 +12 #not an obs dimension
    n_proprio = 3 + 2 + 3 + 4 + 36 + 5 #is an obs dimension
    history_len = 10


    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 20.0
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
        self.scene.num_envs = 32
        self.episode_length_s = 20


        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges

