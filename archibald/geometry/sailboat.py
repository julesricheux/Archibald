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
from archibald.geometry.planform import Rig, Appendage
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

class Sailboat(ArchibaldObject):
    
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
            Fb_i, Mb_i = hull.compute_buoyancy(
                op_point=op_point,
                recompute_statics=recompute_statics,
            )
            self.forces[f"Fb_{i}"] = Fb_i
            self.moments[f"Mb_{i}"] = Mb_i
            
            self.forces["Fb"] += Fb_i
            self.moments["Mb"] += Mb_i
            
            
    def compute_resistance(
            self,
            op_point: OperatingPoint,
            method: str,
            recompute_statics: bool = True,
            **kwargs,
        ):
        self.forces["Fh"] = wide(np.zeros(3))
        self.moments["Mh"] = wide(np.zeros(3))
        
        for i, hull in enumerate(self.hulls):
            Fb_i, Mb_i = hull.compute_resistance(
                op_point=op_point,
                method=method,
                recompute_statics=recompute_statics,
                **kwargs,
            )
            self.forces[f"Fh_{i}"] = Fb_i
            self.moments[f"Mh_{i}"] = Mb_i
            
            self.forces["Fh"] += Fb_i
            self.moments["Mh"] += Mb_i
            
        
    def compute_torsor(
            self,
            op_point: OperatingPoint,
            method: str,
            recompute_statics: bool = True,
            **kwargs,
        ):
        self.compute_aerodynamics(op_point)
        self.compute_hydrodynamics(op_point)
        self.compute_weight(op_point)
        self.compute_buoyancy(op_point)
        self.compute_resistance(
            op_point=op_point,
            method=method,
            recompute_statics=recompute_statics,
            **kwargs,
        )
        
        self.forces["Ftot"] = self.forces["Fb"] + self.forces["Fw"] + self.forces["Fh"]
        self.moments["Mtot"] = self.moments["Mb"] + self.moments["Mw"] + self.moments["Mh"]
        
        return self.forces["Ftot"], self.moments["Mtot"]
    
    
    def _get_target_residual(
            self,
            target_name: str,
            Ftot: np.ndarray,
            Mtot: np.ndarray
        ):
        """Helper to map string targets to actual torsor arrays."""
        target_map = {
            'Fx': Ftot[0], 'Fy': Ftot[1], 'Fz': Ftot[2],
            'Mx': Mtot[0], 'My': Mtot[1], 'Mz': Mtot[2]
        }
        if target_name not in target_map:
            raise ValueError(f"Target '{target_name}' not recognized. Use one of: {list(target_map.keys())}")
        return target_map[target_name]


    def find_equilibrium(
            self,
            initial_op: OperatingPoint,
            free_variables: List[str] = ["dz", "heel", "trim"],
            targets: List[str] = ["Fz", "Mx", "My"],
            additional_constraints: Optional[List[callable]] = None,
            objective: Optional[callable] = None,
            verbose: bool = True
        ) -> Dict[str, any]:
        """
        Solves the static equilibrium of the sailboat.
        
        Parameters:
        -----------
        initial_op : OperatingPoint
            The starting operating point containing initial guesses and fixed parameters.
        free_variables : List[str]
            List of attributes on the OperatingPoint that the solver is allowed to vary.
        targets : List[str]
            List of forces/moments that must equal zero (e.g., ['Fz', 'Mx', 'My']).
        additional_constraints : List[callable], optional
            A list of functions: f(opti, Ftot, Mtot, vars_dict) -> constraint_expression.
        objective : callable, optional
            A function to minimize: f(opti, Ftot, Mtot, vars_dict) -> objective_expression.
            
        Returns:
        --------
        Dict containing the converged OperatingPoint, and the resulting forces/moments.
        """
        from archibald.optimization import Opti
        opti = Opti()
        
        # 1. Dynamically build the solver's OperatingPoint
        solver_kwargs = {}
        vars_dict = {}
        
        # Assuming OperatingPoint exposes its attributes (or defaults to 0)
        standard_attributes = ["dx", "dy", "dz", "heel", "trim", "leeway"]
        
        for attr in standard_attributes:
            init_val = getattr(initial_op, attr, 0.0)
            
            if attr in free_variables:
                var = opti.variable(init_guess=init_val)
                solver_kwargs[attr] = var
                vars_dict[attr] = var
            else:
                param = opti.parameter(init_val)
                solver_kwargs[attr] = param
                vars_dict[attr] = param
                
        solver_op = OperatingPoint(**solver_kwargs)
        
        # 2. Compute Torsor with symbolic/optimization variables
        Ftot, Mtot = self.compute_torsor(solver_op)
        
        # 3. Apply equilibrium constraints (Residuals == 0)
        for target in targets:
            residual = self._get_target_residual(target, Ftot, Mtot)
            opti.subject_to(residual == 0)
            
        # 4. Apply custom constraints
        if additional_constraints is not None:
            for constraint_func in additional_constraints:
                opti.subject_to(constraint_func(opti, Ftot, Mtot, vars_dict))
                
        # 5. Apply objective function
        if objective is not None:
            opti.minimize(objective(opti, Ftot, Mtot, vars_dict))
            
        # 6. Solve the system
        try:
            sol = opti.solve()
        except RuntimeError as e:
            if verbose: print(f"Solver failed to converge: {e}")
            raise
            
        # 7. Reconstruct the numerical OperatingPoint and Results
        resolved_kwargs = {}
        for attr in standard_attributes:
            resolved_kwargs[attr] = float(sol(solver_kwargs[attr]))
            
        converged_op = OperatingPoint(**resolved_kwargs)
        
        return {
            "converged_op": converged_op,
            "Ftot": np.array(sol(Ftot)).flatten(),
            "Mtot": np.array(sol(Mtot)).flatten(),
            "sol_object": sol, # Keep underlying sol object if user wants to query inner forces
            "success": True
        }

