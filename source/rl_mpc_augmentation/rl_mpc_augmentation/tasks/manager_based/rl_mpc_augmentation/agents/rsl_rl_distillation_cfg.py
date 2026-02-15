# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Distillation agent config for blind_rl: student (policy obs) learns to match teacher (privileged obs)."""

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlDistillationAlgorithmCfg,
    RslRlDistillationRunnerCfg,
    RslRlDistillationStudentTeacherCfg,
)


@configclass
class DistillationRunnerCfg(RslRlDistillationRunnerCfg):
    """Distillation config for blind_rl: teacher = privileged (critic) obs, student = policy obs."""

    num_steps_per_env = 24
    max_iterations = 5000
    save_interval = 100
    experiment_name = "rl_mpc_augmentation"
    # "policy" = student obs (PolicyCfg), "teacher" = teacher obs (CriticCfg / privileged)
    obs_groups = {"policy": ["policy"], "teacher": ["critic"]}
    # Build 2 separate MLPs for student and teacher
    # 1. Teacher forward pass with privileged obs
    # 2. Student forward pass with policy observations
    # 3. Calculate Loss between student and teacher outputs
    # 4. Backprop only student's weights
    policy = RslRlDistillationStudentTeacherCfg(
        # Add noise to student obs for robustness (not memorization)
        init_noise_std=0.1,
        noise_std_type="scalar",
        student_obs_normalization=False,
        teacher_obs_normalization=False,
        student_hidden_dims=[512, 256, 128],
        teacher_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlDistillationAlgorithmCfg(
        num_learning_epochs=5,
        learning_rate=1.0e-3,
        # Match num_steps_per_env to calculate gradient over the entire short traj segment
        gradient_length=24,
        loss_type="mse",
    )
