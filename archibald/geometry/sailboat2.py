# -*- coding: utf-8 -*-
"""
SAILBOAT DESCRIPTION

Created 19/04/2023
Last update: 22/10/2024

@author: Jules Richeux
@contributors: -

"""

import copy

from typing import List, Dict, Union, Optional, Tuple
import archibald.geometry.mesh_utilities as mesh_utils

from archibald.geometry.hull import Hull
from archibald.geometry.lifting_set import Rig, Appendage
from archibald.geometry.propeller import Propeller, BSeriesPropeller

from archibald.performance.operating_point import OperatingPoint

from archibald.dynamics.aero_3D import HydroVortexLatticeMethod, AeroVortexLatticeMethod

import archibald.numpy as np
import archibald.toolbox.units as u

from archibald.common import ArchibaldObject


class Sailboat(ArchibaldObject):
    
    def __init__(
            self,
            name: Optional[str] = None,
            xyz_ref: Union[np.ndarray, List] = None,
            displacement: Union[np.ndarray, List] = None,
            cog: Union[np.ndarray, List] = None,
            rig: Rig = None,
            app: Appendage = None,
            hull: List[Hull] = [],
            propeller: List[Propeller] = [],
        ):
        
        pass
        
