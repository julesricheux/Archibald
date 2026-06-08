# -*- coding: utf-8 -*-
"""
Created on Wed May 27 16:23:04 2026

@author: jrich
"""

import archibald.numpy as np
import archibald.toolbox.units as u

import archibald.dynamics.hydro.dsyhs as dsyhs
import archibald.dynamics.hydro.holtrop as holtrop

from typing import Union, List
from archibald.common import ArchibaldObject
from archibald.environment import Environment
from archibald.performance import OperatingPoint
from archibald.geometry.mesh import ArchibaldMesh




def tall(array):
    return np.reshape(array, (-1, 1))


def wide(array):
    return np.reshape(array, (1, -1))


class Hull(ArchibaldObject):
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
    
    def draw(
            self,
            op_point: OperatingPoint = None,
            set_axis_visibility: bool = None
        ):
        if self.mesh:
            if op_point is None:
                self.mesh.draw(backend="matplotlib", set_axis_visibility=set_axis_visibility)
            else:
                # Bring the global water plane into the boat's local frame
                point = op_point.apply_transformations(
                    geometry=np.array([[0., 0., 0.]]), 
                    inverse=True,
                )
                normal = op_point.apply_transformations(
                    geometry=np.array([[0., 0., 1.]]), 
                    inverse=True,
                    is_vector=True,
                )
                self.mesh.draw(
                    point=point,
                    normal=normal,
                    draw_plane=True,
                    backend="pyvista",
                    set_axis_visibility=set_axis_visibility,
                )
            
    def compute_hydrostatics_properties(
            self,
            op_point: OperatingPoint,
        ):
    
        # Bring the global water plane into the boat's local frame
        point = op_point.apply_transformations(
            geometry=np.array([[0., 0., 0.]]), 
            inverse=True,
        )
        normal = op_point.apply_transformations(
            geometry=np.array([[0., 0., 1.]]), 
            inverse=True,
            is_vector=True,
        )
        
        # _temp = self.mesh.copy()
        # _temp.transform(op_point)
    
        # volume, cob = _temp.hydrostatics()
        
        self.hydrostatics_data = self.mesh.hydrostatics(point, normal)
        
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
            1.,
        ])) * volume*rho*g
        
        Mb = np.cross(center, Fb)
        
        return Fb, Mb
    
    
    def _compute_resistance_dsyhs(
            self,
            stw,
            rho: float = OperatingPoint().environment.water.density,
            nu: float = OperatingPoint().environment.water.kinematic_viscosity,
            g : float = OperatingPoint().environment.gravity,
            recompute_statics: bool = False,
        ):
        
        if recompute_statics:
            self.compute_hydrostatics_properties(op_point)
            
        V = stw * u.kt
            
        Re = V / (self.hydrostatics_data["Lwl"] * nu + 1e-12)
        Fr = V / np.sqrt(self.hydrostatics_data["Lwl"] * g + 1e-12)
        
        # Additionnal parameters needed to compute DSYHS resistance
        env_params = {
            'g': g,         # m/s^2
            'nu': nu,       # m^2/s
            'rho': rho,     # kg/m^3
        }
        
        state_params = {
            'stw': stw,
            'Re': np.softplus(Re, beta=1e3),
            'Fr': np.softplus(Fr, beta=1e3),
        }
        
        Rf = dsyhs.compute_Rf_dsyhs(**self.hydrostatics_data, **env_params, **state_params)
        Rw = dsyhs.compute_Rw_dsyhs(**self.hydrostatics_data, **env_params, **state_params)
        Rtr = dsyhs.compute_Rtr_dsyhs(**self.hydrostatics_data, **env_params, **state_params)
        
        print(Fr)
        print(Rf)
        print(Rw)
        print(Rtr)
        
        return Rf + Rw + Rtr
    
    
    def compute_resistance(
            self,
            op_point: OperatingPoint = OperatingPoint(),
            method: str = str(),
            recompute_statics: bool = True,
        ):
        
        if recompute_statics:
            self.compute_hydrostatics_properties(op_point)
            
        center = op_point.apply_transformations(self.hydrostatics_data['cow']) # TODO refine position
        rho = op_point.environment.water.density
        g = op_point.environment.gravity
        
        R = self._compute_resistance_dsyhs(
            op_point.stw,
            rho,
            g,
            recompute_statics=False,
        )
        
        Fh = wide(np.array([
            -1.,
            0.,
            0.,
        ])) * R
        
        Mh = np.cross(center, Fh)
        
        return Fh, Mh
        
        
if __name__=="__main__":
    import os
    from archibald.optimization import Opti
    
    stl = os.path.abspath(r"..\..\examples\02 - Geometry\data\molenez2_data\hull.stl")
    hull = Hull(mesh=stl)
    
    T0, ref = 1.3, 116.487
    
    # hull.draw(op_point, set_axis_visibility=True)
    
    opti = Opti()
    
    # T = opti.variable(init_guess = T0)
    T = T0
    
    op_point = OperatingPoint(
        stw=2.,
        dz=-T,
        # leeway=45.,
        # trim=0.1,
        # heel=45.
    )
    
    hull.compute_buoyancy(op_point)
    
    volume = hull.hydrostatics_data['volume']
    
    opti.minimize((volume - ref)**2.)
    
    sol = opti.solve()
    
    print(sol(T))
    
    hull.compute_resistance(op_point)
    
    