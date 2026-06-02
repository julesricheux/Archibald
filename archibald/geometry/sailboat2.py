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

from archibald.geometry.hull import Hull
from archibald.geometry.lifting_set import Rig, Appendage
from archibald.geometry.propeller import Propeller, BSeriesPropeller

from archibald.performance.operating_point import OperatingPoint

from archibald.dynamics.aero_3D import HydroVortexLatticeMethod, AeroVortexLatticeMethod

import archibald.numpy as np
import archibald.toolbox.units as u

from archibald.common import ArchibaldObject

#%%

def tall(array):
    return np.reshape(array, (-1, 1))


def wide(array):
    return np.reshape(array, (1, -1))

#%%

class Sailboat2(ArchibaldObject):
    
    def __init__(
            self,
            name: Optional[str] = None,
            xyz_ref: Union[np.ndarray, List] = None,
            displacement: Union[np.ndarray, float] = None,
            cog: Union[np.ndarray, List] = None,
            rig: Rig = None,
            app: Appendage = None,
            hulls: List[Hull] = [],
            propellers: List[Propeller] = [],
        ):
        
        self.hulls = hulls
        self.displacement = displacement
        self.cog = wide(np.array(cog))
        
        self.forces = {}
        self.moments = {}
        
    def compute_weight(
            self,
            op_point: OperatingPoint = OperatingPoint(),
        ):
        
        # TODO: bring back cog in underway axes AND transform it with the rest
        
        weight = self.displacement
        g = op_point.environment.gravity
        
        center = op_point.apply_transformations(self.cog)
        
        Fw = np.array([
            0.,
            0.,
            -weight*g
        ])
        
        Mw = np.cross(center, Fw)
        
        self.forces["Fw"] = Fw
        self.moments["Mw"] = Mw
        
        return Fw, Mw
    
    
if __name__=="__main__":
    
    sailboat = Sailboat2(
        displacement=100.,
        cog=[5., 0., 1.],
    )
    
    sailboat.compute_weight()
    
    print(sailboat.forces["Fw"])
    print(sailboat.moments["Mw"])
