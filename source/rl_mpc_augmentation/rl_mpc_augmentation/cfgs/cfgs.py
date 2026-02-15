from __future__ import annotations

from dataclasses import MISSING
from typing import Literal
from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl.rl_cfg import RslRlPpoAlgorithmCfg,RslRlPpoActorCriticCfg

#Takes place of the "policy" component of train_cfg from extreme_parkour
@configclass
class RslRlPpoActorCriticCfgCustom(RslRlPpoActorCriticCfg):
            
    scan_encoder_dims:list = MISSING # scandots
    priv_encoder_dims:list = MISSING 

    depth_encoder_dims: list = MISSING
    depth_latent_dim: int = MISSING
    depth_image_shape: list = MISSING # match camera config
    use_recurrent_backbone: bool = MISSING # Use RNN backbone
    teacher_resume_path: str = MISSING  # Path to the exported .pt or .pth teacher model
    num_learning_epochs: int = MISSING   # Usually higher for student training
    num_mini_batches: int = MISSING

    # Tell student and teacher observations apart
    student_obs_names: list[str] = ["proprioception", "depth_image"]
    teacher_obs_names: list[str] = ["proprioception", "scandots"]

    # Dagger warmup
    warmup_steps: int = MISSING # Steps to run before starting distillation

    # only for 'ActorCriticRecurrent':
    rnn_type: str = MISSING
    rnn_hidden_size: int = MISSING
    rnn_num_layers: int = MISSING
    tanh_encoder_output: bool = MISSING

#Takes place of the "algorithm" component of train_cfg from extreme_parkour
@configclass
class RslRlPpoAlgorithmCfgCustom(RslRlPpoAlgorithmCfg):
    
    
    # dagger params
    dagger_update_freq:int = MISSING
    # Control how closely student mimics teacher
    priv_reg_coef_schedual: list = MISSING
    priv_reg_coef_schedual_resume:list = MISSING

@configclass
class EstimatorCfg:
        train_with_estimated_states:bool = MISSING
        learning_rate:float = MISSING
        hidden_dims:list = MISSING

