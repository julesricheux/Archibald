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
        
    def __repr__(self):
        return f"Hull object '{self.name}'"
    
    def draw(self):
        if self.mesh:
            self.mesh.draw(backend="matplotlib")
            
    def compute_hydrostatics_properties(
            self,
            op_point: OperatingPoint,
        ):
        pass
        
        
if __name__=="__main__":
    import os
    stl = os.path.abspath(r"..\..\examples\02 - Geometry\data\molenez2_data\hull.stl")
    hull = Hull2(mesh=stl)
    
    op_point = OperatingPoint()
    
    hull.draw()