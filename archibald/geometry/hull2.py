# -*- coding: utf-8 -*-
"""
Created on Wed May 27 16:23:04 2026

@author: jrich
"""

import archibald.numpy as np

from typing import Union, List
from archibald.common import ArchibaldObject
from archibald.environment import Environment
from archibald.performance import OperatingPoint
from archibald.geometry.mesh import ArchibaldMesh


def tall(array):
    return np.reshape(array, (-1, 1))


def wide(array):
    return np.reshape(array, (1, -1))


class Hull2(ArchibaldObject):
    def __init__(
            self,
            name: str = 'hull',
            displacement: float = 0.0,
            cog: Union[np.ndarray, List] = np.zeros(3),
            mesh: Union[ArchibaldMesh, str] = None,
            inv_x: bool = True,
            env: Environment = Environment()
        ):
        if type(mesh) == str:
            from archibald.toolbox.mesh_utils import load_stl
            mesh_path = mesh
            mesh = ArchibaldMesh(*load_stl(mesh_path))
        
        self.name = name
        self.displacement = displacement
        self.mesh = mesh
        
        self.hydrostatics_data = {}
        
    def __repr__(self):
        return f"Hull object '{self.name}'"
    
    def draw(self):
        if self.mesh:
            self.mesh.draw(backend="matplotlib")
            
    def compute_hydrostatics_properties(
            self,
            op_point: OperatingPoint,
        ):
        point = -op_point.xyz
        normal = wide(np.array([0., 0., 1.])) @ op_point.mat
    
        volume, cob = self.mesh.hydrostatics(point, normal)
        
        self.hydrostatics_data["volume"] = volume
        self.hydrostatics_data["cob"] = wide(cob)
        
    def compute_buoyancy(
            self,
            op_point: OperatingPoint = OperatingPoint(),
            recompute_statics: bool = True,
        ):
        
        if recompute_statics:
            self.compute_hydrostatics_properties(op_point)
        
        center = op_point.apply_transformations(self.hydrostatics_data['cob'])
        volume = self.hydrostatics_data['volume']
        rho = op_point.environment.water.density
        g = op_point.environment.gravity
        
        Fb = wide(np.array([
            0.,
            0.,
            volume*rho*g
        ]))
        
        Mb = np.cross(center, Fb)
        
        return Fb, Mb
        
        
if __name__=="__main__":
    import os
    from archibald.optimization import Opti
    
    stl = os.path.abspath(r"..\..\examples\02 - Geometry\data\molenez2_data\hull.stl")
    hull = Hull2(mesh=stl)
    
    op_point = OperatingPoint()
    
    hull.draw()
    
    opti = Opti()
    
    T0, ref = 1.3, 116.487
    
    T = opti.variable(init_guess = T0)
    # T = T0
    
    op_point = OperatingPoint(
        dz=-T,
        leeway=45.,
        # trim=0.1,
        heel=45.
    )
    
    hull.compute_buoyancy(op_point)
    
    volume = hull.hydrostatics_data['volume']
    
    opti.minimize((volume - ref)**2.)
    
    sol = opti.solve()
    
    print(sol(T))
    
    