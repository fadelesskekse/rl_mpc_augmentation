from .pass_to_env_action_cfg import PassToEnvironmentCfg
from isaaclab.utils import configclass
from isaaclab.managers.action_manager import ActionTerm 
from .pass_to_env_action import PassToEnvironment

@configclass
class GaitCycleActionCfg(PassToEnvironmentCfg):

    class_type: type[ActionTerm] = PassToEnvironment

    #clip = {"gait_bounds": (.5,1.2)}

    

   


