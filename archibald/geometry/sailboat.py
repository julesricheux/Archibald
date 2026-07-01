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
from archibald.geometry.planform import Rig, Appendage, Planform
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
            hulls: List[Hull] = [],
            rigs: List[Rig] = [],
            appendages: List[Appendage] = [],
            propellers: List[Propeller] = [],
        ):
        
        self.hulls = hulls
        self.rigs = rigs
        self.appendages = appendages
        self.propellers = propellers
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
    
    
    # def draw(self,
    #          backend: str = "pyvista",
    #          thin_wings: bool = False,
    #          ax=None,
    #          mesh_color: str = 'lightgrey',
    #          show: bool = True,
    #          set_axis_visibility: bool = True,
    #          show_kwargs: Dict = None,
    #          **kwargs):
    #     """
    #     Visualizes the sailboat by aggregating all wings, rigs, and hull meshes.
    #     """
    #     if show_kwargs is None:
    #         show_kwargs = {}

    #     # 1. Collect all meshes from the sailboat's components
    #     all_points = []
    #     all_faces = []
        
    #     # Add Rigs (assuming rigs are objects with a mesh_body method)
    #     for rig in self.rigs:
    #         p, f = rig.mesh_body(method="quad", thin_wings=thin_wings)
    #         all_points.append(p)
    #         all_faces.append(f)
            
    #     # Add Hulls
    #     for h in self.hulls:
    #         if hasattr(h, 'mesh') and h.mesh:
    #             # Assuming h.mesh has .vertices and .faces
    #             all_points.append(h.mesh.vertices)
    #             all_faces.append(h.mesh.faces)

    #     # 2. Visualization Logic
    #     if backend == "pyvista":
    #         import pyvista as pv
    #         import archibald.toolbox.mesh_utils as mesh_utils
            
    #         meshes = []
    #         for p, f in zip(all_points, all_faces):
    #             meshes.append(pv.PolyData(*mesh_utils.convert_mesh_to_polydata_format(p, f)))
            
    #         fig = meshes[0]
    #         for m in meshes[1:]:
    #             fig = fig.merge(m)
            
    #         if show:
    #             fig.plot(show_edges=True, show_grid=True, **show_kwargs)
    #         return fig

    #     elif backend == "plotly":
    #         import plotly.graph_objects as go
            
    #         data = []
    #         for p, f in zip(all_points, all_faces):
    #             # Convert quads to triangles if needed for Mesh3d
    #             i, j, k = f[:, [0, 1, 2]].T
    #             data.append(go.Mesh3d(x=p[:,0], y=p[:,1], z=p[:,2], i=i, j=j, k=k, 
    #                                   color=mesh_color, opacity=1.0))
            
    #         fig = go.Figure(data=data)
    #         # Setup scene aspect and axis visibility
    #         scene_layout = dict(aspectmode='data')
            
    #         if set_axis_visibility is False:
    #             scene_layout.update(
    #                 xaxis=dict(visible=False),
    #                 yaxis=dict(visible=False),
    #                 zaxis=dict(visible=False)
    #             )
    #         elif set_axis_visibility is True:
    #             scene_layout.update(
    #                 xaxis=dict(visible=True),
    #                 yaxis=dict(visible=True),
    #                 zaxis=dict(visible=True)
    #             )
                
    #         fig.update_layout(scene=scene_layout)

    #         if show:
    #             from plotly.offline import plot
    #             plot(fig)
    #         return fig

    #     elif backend == "matplotlib":
    #         import matplotlib.pyplot as plt
    #         from mpl_toolkits.mplot3d.art3d import Poly3DCollection
            
    #         fig = plt.figure()
    #         ax = fig.add_subplot(111, projection='3d')
    #         for p, f in zip(all_points, all_faces):
    #             ax.add_collection(Poly3DCollection(p[f], facecolors=mesh_color, alpha=0.8))
            
    #         if show:
    #             plt.show()
    #         return ax
    
    def draw(self,
             backend: str = "pyvista",
             thin_wings: bool = False,
             mesh_color: str = 'lightgrey',
             show: bool = True,
             show_kwargs: Dict = None,
             **kwargs):
        """
        Visualizes the sailboat by aggregating all wings, rigs, and hull meshes
        into a single unified scene.
        """
        if show_kwargs is None:
            show_kwargs = {}

        # --- PYVISTA BACKEND ---
        if backend == "pyvista":
            import pyvista as pv
            import archibald.toolbox.mesh_utils as mesh_utils
            
            # Create the master plotter for the whole sailboat
            plotter = pv.Plotter()

            # 1. Add Rigs
            for rig in self.rigs:
                p, f = rig.mesh_body(method="quad", thin_wings=thin_wings)
                mesh = pv.PolyData(*mesh_utils.convert_mesh_to_polydata_format(p, f))
                plotter.add_mesh(mesh, color=mesh_color, show_edges=True)
                
            for app in self.appendages:
                p, f = app.mesh_body(method="quad", thin_wings=thin_wings)
                mesh = pv.PolyData(*mesh_utils.convert_mesh_to_polydata_format(p, f))
                plotter.add_mesh(mesh, color=mesh_color, show_edges=True)

            # 2. Add Hulls
            for h in self.hulls:
                # We pass the shared plotter so the hull adds itself to the sailboat's scene
                # We also pass **kwargs to ensure draw_plane, point, normal, etc., are passed down
                h.draw(backend="pyvista", show=False, plotter=plotter, **kwargs)

            if show:
                plotter.show(**show_kwargs)
            return plotter

        # --- PLOTLY BACKEND ---
        elif backend == "plotly":
            import plotly.graph_objects as go
            fig = go.Figure()

            # 1. Add Rigs
            for rig in self.rigs:
                p, f = rig.mesh_body(method="quad", thin_wings=thin_wings)
                i, j, k = f[:, [0, 1, 2]].T
                fig.add_trace(go.Mesh3d(x=p[:,0], y=p[:,1], z=p[:,2], i=i, j=j, k=k, 
                                        color=mesh_color, opacity=1.0))
                
            for app in self.appendages:
                p, f = app.mesh_body(method="quad", thin_wings=thin_wings)
                i, j, k = f[:, [0, 1, 2]].T
                fig.add_trace(go.Mesh3d(x=p[:,0], y=p[:,1], z=p[:,2], i=i, j=j, k=k, 
                                        color=mesh_color, opacity=1.0))

            # 2. Add Hulls
            for h in self.hulls:
                h.draw(backend="plotly", show=False, fig=fig, **kwargs)

            fig.update_layout(scene=dict(aspectmode='data'))
            if show:
                fig.show()
            return fig

        # --- MATPLOTLIB BACKEND ---
        elif backend == "matplotlib":
            import matplotlib.pyplot as plt
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            
            for rig in self.rigs:
                p, f = rig.mesh_body(method="quad", thin_wings=thin_wings)
                # ... (add rig poly collection to ax)
                
            for app in self.appendages:
                p, f = app.mesh_body(method="quad", thin_wings=thin_wings)
                # ... (add rig poly collection to ax)
                
            for h in self.hulls:
                h.draw(backend="matplotlib", show=False, ax=ax, **kwargs)
            
            if show: plt.show()
            return ax
    
    
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

