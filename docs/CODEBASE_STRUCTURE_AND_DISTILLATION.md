# Codebase Structure & Policy Distillation

## 1. Codebase structure

### Top level

```
rl_mpc_augmentation/
├── scripts/                    # Entry points (train, play, list envs)
├── source/rl_mpc_augmentation/  # Python package
│   ├── config/                  # extension.toml (Isaac Lab extension metadata)
│   └── rl_mpc_augmentation/     # Main package
│       ├── tasks/              # Envs + MDP configs (scene, obs, rewards, actions)
│       ├── algorithms/         # PPO / custom algorithm (e.g. PPOCustom)
│       ├── modules/            # Networks (actor-critic, encoders, estimator)
│       ├── runners/            # Training loops (OnPolicyRunnerCustom)
│       └── cfgs/                # Algorithm/policy config dataclasses
```

### Task registration and config flow

- **Tasks** are registered in  
  `source/rl_mpc_augmentation/rl_mpc_augmentation/tasks/manager_based/rl_mpc_augmentation/__init__.py`.
- Each task binds:
  - **env_cfg_entry_point** → env config (scene, terrain, robot, MDP)
  - **rsl_rl_cfg_entry_point** → agent config (policy, algorithm, runner) for PPO
  - **play_env_cfg_entry_point** → env config used for play/eval

Current tasks:

| Task ID           | Env config      | Agent config (PPO)   |
|-------------------|-----------------|----------------------|
| `blind_rl`        | blind_rl_cfg    | PPORunnerCfg         |
| `phase_1_scan_rl` | vision_rl_cfg   | PPORunnerCfgCustom   |

### Where things live

| What you care about        | Where it is |
|----------------------------|-------------|
| Scene (terrain, robot, sensors) | `tasks/.../blind_rl_cfg.py` → `RlMpcAugmentationSceneCfg`, `ObservationsCfg`, etc. |
| Observation groups (policy vs critic) | `blind_rl_cfg.py` → `ObservationsCfg.PolicyCfg`, `CriticCfg` |
| Rewards / terminations    | `tasks/.../mdp/rewards.py`, `terminations.py` |
| Actions (joints, gait, etc.) | `tasks/.../mdp/actions/`, `blind_rl_cfg.ActionsCfg` |
| PPO / algorithm config     | `tasks/.../agents/rsl_rl_ppo_cfg.py` + `cfgs/cfgs.py` |
| Custom algorithm (e.g. PPOCustom) | `algorithms/ppo_custom.py` |
| Custom actor-critic        | `modules/actor_critic_custom.py` |
| Training loop (steps, logging) | `runners/on_policy_runner_custom.py` |
| Train script               | `scripts/rsl_rl/train.py` |
| Play script                | `scripts/rsl_rl/play.py` |

### How a training run is chosen

1. **Task** (e.g. `blind_rl`) → selects env config and **default** agent config via entry points.
2. **Agent** (e.g. `rsl_rl_cfg_entry_point`) → overrides which agent config is loaded from the same task’s kwargs (e.g. PPO vs distillation).
3. **Hydra** in `train.py` calls `load_cfg_from_registry(task_name, agent_cfg_entry_point)` to get `env_cfg` and `agent_cfg`.
4. **Runner** is created from `agent_cfg.class_name` (`OnPolicyRunner`, `OnPolicyRunnerCustom`, or `DistillationRunner`).

So: **task** = which env + which default agent; **agent** = which of the task’s registered configs to use (PPO vs distillation).

---

## 2. Policy distillation (teacher → student)

### Idea

- **Teacher**: policy trained with **privileged** (e.g. sim-only) observations (e.g. true base_lin_vel, base_z_pos).
- **Student**: policy that only sees **sensor-like** observations (e.g. no base_lin_vel/base_z_pos).
- **Distillation**: student is trained to match teacher **actions** (behavior cloning on teacher’s outputs), so the student learns to behave like the teacher from fewer observations.

Your `blind_rl` setup already fits this:

- **PolicyCfg** ≈ student (no base_lin_vel, no base_z_pos).
- **CriticCfg** ≈ teacher (includes base_lin_vel, base_z_pos).

So the natural distillation setup is: teacher uses critic-like (privileged) obs, student uses policy-like (blind) obs.

### What this codebase already has

- **DistillationRunner** from `rsl_rl` is used in `scripts/rsl_rl/train.py` and `play.py` when `agent_cfg.class_name == "DistillationRunner"`.
- **No distillation config is registered yet**: there is no `rsl_rl_distillation_cfg_entry_point` for `blind_rl` or `phase_1_scan_rl`, and no agent config that sets `class_name = "DistillationRunner"` and the right policy/algorithm.

So to “do policy distillation” you need to:

1. Add a **distillation agent config** (student/teacher nets + distillation algorithm).
2. Register it under a **distillation entry point** (e.g. `rsl_rl_distillation_cfg_entry_point`) for your task.
3. Define **observation groups** so that the **student** gets policy-like obs and the **teacher** gets critic-like (privileged) obs.
4. Train **teacher** with PPO, then run **distillation** by loading that teacher and using the distillation agent config.

---

## 3. How to add policy distillation

### Step 1: Observation groups for student and teacher

DistillationRunner expects **obs_groups** that map:

- **"policy"** → observations the **student** sees (e.g. same as current `policy` group).
- **"teacher"** → observations the **teacher** sees (e.g. same as current `critic` group).

So you need an observation group that contains the privileged terms (e.g. base_lin_vel, base_z_pos). You already have that as `CriticCfg`. You can either:

- Reuse the same group names as in Isaac Lab’s example: e.g. `policy` = student, `teacher` = teacher, and make the env expose a group that matches your critic (e.g. name it `"teacher"` or map `"teacher"` → critic terms), or
- Add a dedicated `TeacherCfg` that’s identical to `CriticCfg` and register it as the `"teacher"` group.

In the registry, the important part is that the env’s observation manager produces two groups whose names you will use in `obs_groups` in the distillation config (see below).

### Step 2: Distillation agent config

Create a config that uses Isaac Lab’s distillation runner and policy/algorithm types, for example in  
`source/rl_mpc_augmentation/rl_mpc_augmentation/tasks/manager_based/rl_mpc_augmentation/agents/rsl_rl_distillation_cfg.py`:

- **Runner**: `RslRlDistillationRunnerCfg` (from `isaaclab_rl.rsl_rl`), with `class_name = "DistillationRunner"`.
- **Policy**: `RslRlDistillationStudentTeacherCfg` (or recurrent variant): student/teacher hidden dims, activation, obs normalization for student and teacher.
- **Algorithm**: `RslRlDistillationAlgorithmCfg`: learning rate, epochs, gradient length, loss type (e.g. MSE).
- **obs_groups**: e.g. `{"policy": ["policy"], "teacher": ["teacher"]}` — `"policy"` = student obs group name, `"teacher"` = teacher (privileged) obs group name. Use the same group names as in your env’s observation config.

You can copy the structure from Isaac Lab’s example:

- `IsaacLab/source/isaaclab_tasks/.../anymal_d/agents/rsl_rl_distillation_cfg.py`
- `IsaacLab/source/isaaclab_rl/isaaclab_rl/rsl_rl/distillation_cfg.py`

and then adapt:

- `experiment_name`, `num_steps_per_env`, `max_iterations`, `save_interval`
- student/teacher network sizes to match your policy/critic obs dimensions
- `obs_groups` to the group names you defined (e.g. `"policy"` and `"teacher"`).

### Step 3: Register the distillation entry point

In `tasks/manager_based/rl_mpc_augmentation/__init__.py`, for each task that should support distillation, add to the task’s kwargs:

- `rsl_rl_distillation_cfg_entry_point`: string pointing to your new config class, e.g.  
  `f"{agents.__name__}.rsl_rl_distillation_cfg:BlindRLDistillationRunnerCfg"`.

So when you call train with `--agent rsl_rl_distillation_cfg_entry_point`, the loader will use this config and create a `DistillationRunner`.

### Step 4: Train teacher, then distill

1. **Train teacher (PPO)** with privileged info (e.g. critic obs):
   - Use the existing task and PPO config, e.g.  
     `~/IsaacLab/isaaclab.sh -p scripts/rsl_rl/train.py --task blind_rl`
   - Let it save checkpoints (e.g. under `logs/rsl_rl/<experiment_name>/<run_dir>/`).

2. **Run distillation** (student learns to match teacher actions):
   - Use the **same** task name so the env (and its obs groups) are the same.
   - Select the **distillation** agent config via `--agent rsl_rl_distillation_cfg_entry_point`.
   - Load the teacher checkpoint: `--resume --load_run <run_dir> --checkpoint model_XXXX.pt` (or whatever your checkpoint naming is).
   - Example:
     ```bash
     ~/IsaacLab/isaaclab.sh -p scripts/rsl_rl/train.py --task blind_rl --agent rsl_rl_distillation_cfg_entry_point --resume --load_run 2026-01-26_12-00-00 --checkpoint model_5000.pt
     ```

The train script already:

- Checks `agent_cfg.algorithm.class_name == "Distillation"` and `agent_cfg.class_name == "DistillationRunner"` to create `DistillationRunner` and to load a checkpoint before `runner.learn()`.
- So once the distillation config is registered and the checkpoint path is correct, distillation runs like any other training run, with the student trained to imitate the teacher’s actions.

### Step 5: Play / deploy the student

After distillation, the saved policy in the distillation run is the **student**. Use the same play script as for PPO, pointing to the distillation run’s checkpoint and the same task (and agent config if play uses it for obs/wrapper):

- e.g. `--task blind_rl --checkpoint logs/rsl_rl/<experiment_name>/<distillation_run_dir>/model_XXX.pt`

If your play script uses `agent_cfg` to build the env or wrapper, make sure the distillation agent config is used when loading a distilled checkpoint (e.g. by passing `--agent rsl_rl_distillation_cfg_entry_point` when playing a distilled policy).

---

## 4. Summary

- **Structure**: Tasks in `tasks/.../__init__.py`; env MDP in `blind_rl_cfg.py` (and vision_rl_cfg); agent configs in `agents/`; algorithms in `algorithms/`; networks in `modules/`; training loop in `runners/`.
- **Distillation**: Add a distillation agent config and `rsl_rl_distillation_cfg_entry_point` for your task; set **obs_groups** so "policy" = student (e.g. current policy obs), "teacher" = privileged (e.g. critic obs); train teacher with PPO, then run train with `--agent rsl_rl_distillation_cfg_entry_point` and `--resume --load_run/--checkpoint` to distill into the student.

### Ready-to-use distillation (blind_rl)

A distillation config is already added:

- **Config**: `agents/rsl_rl_distillation_cfg.py` → `BlindRLDistillationRunnerCfg`
- **Entry point**: `rsl_rl_distillation_cfg_entry_point` registered for task `blind_rl``

**1. Train teacher (PPO):**
```bash
cd ~/rl_mpc_augmentation
~/IsaacLab/isaaclab.sh -p scripts/rsl_rl/train.py --task blind_rl --max_iterations 5000
```
Checkpoints go to `logs/rsl_rl/rl_mpc_augmentation/<timestamp>_*/`.

**2. Distill student from teacher:**
```bash
~/IsaacLab/isaaclab.sh -p scripts/rsl_rl/train.py --task blind_rl --agent rsl_rl_distillation_cfg_entry_point \
  --resume --load_run <teacher_run_dir> --checkpoint model_5000.pt
```
Use the actual run folder name (e.g. `2026-01-26_12-00-00`) and checkpoint (e.g. `model_5000.pt`). The student is trained to match teacher actions; the saved policy after distillation is the student.

**3. Play student policy:**
```bash
~/IsaacLab/isaaclab.sh -p scripts/rsl_rl/play.py --task blind_rl --agent rsl_rl_distillation_cfg_entry_point \
  --checkpoint logs/rsl_rl/rl_mpc_augmentation/<distillation_run_dir>/model_1000.pt
```
If play does not use the agent config for building the env, you can omit `--agent` and only pass `--checkpoint`.
