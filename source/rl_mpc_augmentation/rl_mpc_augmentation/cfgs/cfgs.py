from __future__ import annotations

from dataclasses import MISSING
from typing import Literal
from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl.rl_cfg import RslRlPpoAlgorithmCfg,RslRlPpoActorCriticCfg

#Takes place of the "policy" component of train_cfg from extreme_parkour
@configclass
class RslRlPpoActorCriticCfgCustom(RslRlPpoActorCriticCfg):
            
    scan_encoder_dims:list = MISSING
    scan_cnn: bool = MISSING
    scan_cnn_output_dim: int = MISSING
    priv_encoder_dims:list = MISSING
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
    priv_reg_coef_schedual: list = MISSING
    priv_reg_coef_schedual_resume:list = MISSING

@configclass
class EstimatorCfg:
        train_with_estimated_states:bool = MISSING
        learning_rate:float = MISSING
        hidden_dims:list = MISSING

