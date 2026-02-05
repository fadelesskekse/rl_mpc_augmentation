

import math
from dataclasses import MISSING

from isaaclab.managers import CommandTermCfg

from isaaclab.utils import configclass

from .commands import boolCommand

@configclass
class boolCommandCfg(CommandTermCfg):

    class_type: type = boolCommand

    probability: float = 0.5

    resampling_time_range: tuple[float, float] = MISSING

