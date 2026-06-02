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
        
    def compute_aerodynamics(
            self,
            op_point: OperatingPoint = OperatingPoint(),
        ):
        pass
    
    def compute_hydrodynamics(
            self,
            op_point: OperatingPoint = OperatingPoint(),
        ):
        pass
        
    def compute_weight(
            self,
            op_point: OperatingPoint = OperatingPoint(),
        ):
        
        # TODO: bring back cog in underway axes AND transform it with the rest
        
        weight = self.displacement
        g = op_point.environment.gravity
        
        center = op_point.apply_transformations(wide(self.cog))
        
        Fw = wide(np.array([
            0.,
            0.,
            -weight*g
        ]))
        
        Mw = np.cross(center, Fw)
        
        self.forces["Fw"] = Fw
        self.moments["Mw"] = Mw
        
        return Fw, Mw
    
    def compute_buoyancy(
            self,
            op_point: OperatingPoint,
            recompute_statics: bool = True,
        ):
        self.forces["Fb"] = wide(np.zeros(3))
        self.moments["Mb"] = wide(np.zeros(3))
        
        for i, hull in enumerate(self.hulls):
            Fb_i, Mb_i = hull.compute_buoyancy(op_point, recompute_statics)
            self.forces[f"Fb_{i}"] = Fb_i
            self.moments[f"Mb_{i}"] = Mb_i
            
            self.forces["Fb"] += Fb_i
            self.moments["Mb"] += Mb_i
            
        
    def compute_torsor(
            self,
            op_point: OperatingPoint = OperatingPoint(),
        ):
        self.compute_aerodynamics(op_point)
        self.compute_hydrodynamics(op_point)
        self.compute_weight(op_point)
        self.compute_buoyancy(op_point)
        
        self.forces["Ftot"] = self.forces["Fb"] + self.forces["Fw"]
        self.moments["Mtot"] = self.moments["Mb"] + self.moments["Mw"]
        
        return self.forces["Ftot"], self.moments["Mtot"]
    
if __name__=="__main__":
    
    import os
    from archibald.optimization import Opti
    from archibald.geometry.hull2 import Hull2
    
    T0 = 1.3
    heel0 = 0.
    trim0 = 0.
    leeway0 = 0.
    
    opti = Opti()
    
    T = opti.variable(init_guess=T0)
    heel = opti.variable(init_guess=T0)
    trim = opti.variable(init_guess=T0)
    leeway = opti.parameter(leeway0)
    
    op_point = OperatingPoint(
        dz=-T,
        heel=heel,
        trim=trim,
        leeway=leeway,
    )
    

    stl = os.path.abspath(r"..\..\examples\02 - Geometry\data\molenez2_data\hull.stl")
    hull = Hull2(mesh=stl)
    
    sailboat = Sailboat2(
        displacement=116e3,
        cog=[12., 0., 1.],
        hulls=[hull],
    )
    
    Ftot, Mtot = sailboat.compute_torsor(
        op_point
    )
    
    opti.subject_to(Ftot[2] == 0)
    opti.subject_to(Mtot[0] == 0)
    opti.subject_to(Mtot[1] == 0)
    
    opti.minimize(heel)
    
    sol = opti.solve()
    
    forces = sol(sailboat.forces)
    moments = sol(sailboat.moments)
    
    print(sol((T, heel, trim)))
    
    print(forces["Fw"])
    print(forces["Fb"])
    print(moments["Mw"]/1e3)
    print(moments["Mb"]/1e3)
    print(sol(Ftot), sol(Mtot))
