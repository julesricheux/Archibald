# -*- coding: utf-8 -*-
"""
VORTEX LATTICE METHOD DESCRIPTION

Created 21/12/2023
Last update: 18/10/2024

@author: Jules Richeux
@contributors: -

Citation:
    Adapted from:         aerodynamics.vortex_lattice_method in AeroSandbox
    Author:               Peter D Sharpe
    Date of Retrieval:    18/10/2024

"""
import os
import sys

__root_dir = os.path.dirname(os.path.abspath(__file__))
if __root_dir not in sys.path:
    sys.path.append(os.path.dirname(__root_dir))

from archibald2 import ExplicitAnalysis
from archibald2.geometry import *
from archibald2.geometry.airfoil.thin_section import leading_edge_camber
from archibald2.performance.operating_point import OperatingPoint
from archibald2.environment.environment import Fluid
from archibald2.dynamics.aero_3D.singularities.uniform_strength_horseshoe_singularities import \
    calculate_induced_velocity_horseshoe
from typing import Dict, Any, List, Callable
import copy
from abc import abstractmethod

from archibald2.tools.geom_utils import *
from archibald2.tools.math_utils import rotation_matrix, ramp, ReLU
from archibald2.tools.dyn_utils import *
from archibald2.geometry.lifting_set import LiftingSet, Rig, Appendage

from typing import Tuple, Union, Dict, List

import archibald2.numpy as np

import neuralfoil as nf

### Define some helper functions that take a vector and make it a Nx1 or 1xN, respectively.
# Useful for broadcasting with matrices later.
def tall(array):
    return np.reshape(array, (-1, 1))


def wide(array):
    return np.reshape(array, (1, -1))


def normalize(array):
    magnitude = tall(np.linalg.norm(array, axis=1))
    
    return array / magnitude
    # return array / np.tile(np.linalg.norm(array, axis=1), (array.shape[1],1)).T
    
    
def change_basis(x: Union[float, np.ndarray],
                 y: Union[float, np.ndarray],
                 z: Union[float, np.ndarray],
                 from_axes: np.ndarray = np.diag(np.ones(3)),
                 to_axes: np.ndarray = np.diag(np.ones(3)),
                 ) -> Tuple[float, float, float]:
    """
    Converts a vector [x_from, y_from, z_from], as given to an equivalent vector [x_to,
    y_to, z_to], as given in the `to_axes` frame.

    """
    m0 = from_axes
    m1 = to_axes

    x0 = m0[0,0] * x + m0[1,0] * y + m0[2,0] * z
    y0 = m0[0,1] * x + m0[1,1] * y + m0[2,1] * z
    z0 = m0[0,2] * x + m0[1,2] * y + m0[2,2] * z
    
    x_to = m1[0,0] * x0 + m1[0,1] * y0 + m1[0,2] * z0
    y_to = m1[1,0] * x0 + m1[1,1] * y0 + m1[1,2] * z0
    z_to = m1[2,0] * x0 + m1[2,1] * y0 + m1[2,2] * z0

    return x_to, y_to, z_to


def stall_factor(alpha: Union[float, np.ndarray],
                 alpha_i: Union[float, np.ndarray],
                 alpha_stall: Union[float, np.ndarray],
                 ):
    
    alpha_corr = np.abs(alpha)
    alpha_i_corr = ReLU(alpha_i * np.sign(alpha)) + 1e-3
    
    raw_fac = 1. + (alpha_stall - alpha_corr)/alpha_i_corr
    
    return np.fmin(1.,
                   np.fmax(raw_fac,
                           0.)
                   )


class VortexLatticeMethod(ExplicitAnalysis):
    """
    An asbtract explicit (linear) vortex-lattice-method aerodynamics analysis.
    
    Citation:
        Adapted from:         aerodynamics.VortexLatticeMethod in AeroSandbox
        Author:               Peter D Sharpe
        Date of Retrieval:    18/10/2024

    Usage example:
        >>> analysis = asb.VortexLatticeMethod(
        >>>     airplane=my_airplane,
        >>>     op_point=asb.OperatingPoint(
        >>>         velocity=100, # m/s
        >>>         alpha=5, # deg
        >>>         beta=4, # deg
        >>>         p=0.01, # rad/sec
        >>>         q=0.02, # rad/sec
        >>>         r=0.03, # rad/sec
        >>>     )
        >>> )
        >>> aero_data = analysis.run()
        >>> analysis.draw()
    """
    
    # @abstractmethod
    def __init__(self,
                 airplane: Rig,
                 op_point: OperatingPoint,
                 xyz_ref: List[float] = None,
                 IZsym: bool = False,
                 to_sym: List[bool] = None,
                 Zsym: float = 0.0,
                 verbose: bool = False,
                 spanwise_resolution: int = 10,
                 spanwise_spacing_function: Callable[[float, float, float], np.ndarray] = np.cosspace,
                 chordwise_resolution: int = 10,
                 chordwise_spacing_function: Callable[[float, float, float], np.ndarray] = np.cosspace,
                 vortex_core_radius: float = 1e-8,
                 align_trailing_vortices_with_wind: bool = False,
                 ):
        super().__init__()
        
        ### Set defaults
        if xyz_ref is None:
            xyz_ref = airplane.xyz_ref

        ### Initialize
        self.airplane = airplane
        self.op_point = op_point
        self.xyz_ref = xyz_ref
        self.verbose = verbose
        self.spanwise_resolution = spanwise_resolution
        self.spanwise_spacing_function = spanwise_spacing_function
        self.chordwise_resolution = chordwise_resolution
        self.chordwise_spacing_function = chordwise_spacing_function
        self.vortex_core_radius = vortex_core_radius
        self.align_trailing_vortices_with_wind = align_trailing_vortices_with_wind
        
        self.Zsym = Zsym
        self.IZsym = IZsym
        
        if to_sym is None:
            to_sym = [IZsym] * len(self.airplane.wings)
        self.to_sym = to_sym
        
        self.fluid = Fluid()
        
        self.visc_corr = 0.
        
        self.wing_spanwise_resolution = [(len(w.xsecs)-1)*self.spanwise_resolution*(1+w.symmetric) for w in self.airplane.wings]
        # Total number of spanwise cells for all wings
        self.total_spanwise_resolution = np.sum(self.wing_spanwise_resolution)
        
        
    def __repr__(self):
        return self.__class__.__name__ + "(\n\t" + "\n\t".join([
            f"airplane={self.airplane}",
            f"op_point={self.op_point}",
            f"xyz_ref={self.xyz_ref}",
        ]) + "\n)"
    
    def fluid_dynamic_pressure(self):
        return 1.
    
    def reference_speed(self,
                        z: float = None):
        return 1.
    
    def draw_horseshoes(self):
        ##### Check single vortex
        u, v, w = calculate_induced_velocity_horseshoe(
            x_field=0,
            y_field=0,
            z_field=0,
            x_left=-1,
            y_left=-1,
            z_left=0,
            x_right=-1,
            y_right=1,
            z_right=0,
            gamma=1,
        )
        print(u, v, w)

        ##### Plot grid of single vortex
        args = (-2, 2, 30)
        x = np.linspace(*args)
        y = np.linspace(*args)
        z = np.linspace(*args)
        X, Y, Z = np.meshgrid(x, y, z)

        Xf = X.flatten()
        Yf = Y.flatten()
        Zf = Z.flatten()

        left = [0, -1, 0]
        right = [0, 1, 0]

        Uf, Vf, Wf = calculate_induced_velocity_horseshoe(
            x_field=Xf,
            y_field=Yf,
            z_field=Zf,
            x_left=left[0],
            y_left=left[1],
            z_left=left[2],
            x_right=right[0],
            y_right=right[1],
            z_right=right[2],
            gamma=1,
        )

        pos = np.stack((Xf, Yf, Zf)).T
        dir = np.stack((Uf, Vf, Wf)).T

        dir_norm = np.reshape(np.linalg.norm(dir, axis=1), (-1, 1))

        dir = dir / dir_norm * dir_norm**0.2

        import pyvista as pv

        pv.set_plot_theme("dark")
        plotter = pv.Plotter()
        plotter.add_arrows(cent=pos, direction=dir, mag=0.15)
        plotter.add_lines(
            lines=np.array(
                [[Xf.max(), left[1], left[2]], left, right, [Xf.max(), right[1], right[2]]]
            )
        )
        plotter.show_grid()
        plotter.show()

        ##### Check multiple vortices
        args = (-2, 2, 10)
        x = np.linspace(*args)
        y = np.linspace(*args)
        z = np.linspace(*args)
        X, Y, Z = np.meshgrid(x, y, z)

        Xf = X.flatten()
        Yf = Y.flatten()
        Zf = Z.flatten()

        left = self.left_vortex_vertices
        center = self.collocation_points
        right = self.right_vortex_vertices

        lefts = np.array([left, center])
        rights = np.array([center, right])
        # strengths = np.array([2, 1])

        Uf_each, Vf_each, Wf_each = calculate_induced_velocity_horseshoe(
            x_field=wide(Xf),
            y_field=wide(Yf),
            z_field=wide(Zf),
            x_left=tall(lefts[:, 0]),
            y_left=tall(lefts[:, 1]),
            z_left=tall(lefts[:, 2]),
            x_right=tall(rights[:, 0]),
            y_right=tall(rights[:, 1]),
            z_right=tall(rights[:, 2]),
            gamma=1.,
            # gamma=tall(strengths),
        )

        Uf = np.sum(Uf_each, axis=0)
        Vf = np.sum(Vf_each, axis=0)
        Wf = np.sum(Wf_each, axis=0)

        pos = np.stack((Xf, Yf, Zf)).T
        dir = np.stack((Uf, Vf, Wf)).T

        dir_norm = np.reshape(np.linalg.norm(dir, axis=1), (-1, 1))

        dir = dir / dir_norm * dir_norm**0.2

        import pyvista as pv

        pv.set_plot_theme("dark")
        plotter = pv.Plotter()
        plotter.add_arrows(cent=pos, direction=dir, mag=0.15)
        plotter.add_lines(
            lines=np.array(
                [
                    [Xf.max(), left[1], left[2]],
                    left,
                    center,
                    # [Xf.max(), center[1], center[2]],
                    center,
                    right,
                    [Xf.max(), right[1], right[2]],
                ]
            )
        )
        plotter.show_grid()
        plotter.show()
    
    def mesh_geometry(self):
        ##### Make Panels
        front_left_vertices = []
        back_left_vertices = []
        back_right_vertices = []
        front_right_vertices = []
        is_trailing_edge = []
        is_leading_edge = []
        # is_trailing_edge_idx = []
        # is_leading_edge_idx = []
        # strips = []
        is_symmetric = []
        
        strips_vertices = []
        
        is_soft = []

        for i, wing in enumerate(self.airplane.wings):
            if self.spanwise_resolution > 1:
                wing = wing.subdivide_sections(
                    ratio=self.spanwise_resolution,
                    spacing_function=self.spanwise_spacing_function
                    )
                    
            points, faces = wing.mesh_thin_surface(
                method="quad",
                chordwise_resolution=self.chordwise_resolution,
                chordwise_spacing_function=self.chordwise_spacing_function,
                add_camber=True
            )
            
            front_left_vertices.append(points[faces[:, 0], :])
            back_left_vertices.append(points[faces[:, 1], :])
            back_right_vertices.append(points[faces[:, 2], :])
            front_right_vertices.append(points[faces[:, 3], :])
            is_trailing_edge.append(
                (np.arange(len(faces)) + 1) % self.chordwise_resolution == 0
                )
            # is_trailing_edge_idx.append(
            #     np.arange((self.chordwise_resolution-1)+len(faces)*i,
            #               len(faces)*(i+1),
            #               self.chordwise_resolution)
            #     )
            is_leading_edge.append(
                (np.arange(len(faces)) + 1) % self.chordwise_resolution == 1
                )
            # is_leading_edge_idx.append(
            #     np.arange((self.chordwise_resolution-1)+len(faces)*i,
            #               len(faces)*(i+1),
            #               self.chordwise_resolution)
            #     )
            
            wing_spanwise_resolution = self.wing_spanwise_resolution[i]
            
            for k in range(wing_spanwise_resolution):
                strip_k_vertices = [
                    points[k::wing_spanwise_resolution+1, :], # left vertices
                    points[k+1::wing_spanwise_resolution+1, :], # right vertices
                ]
                
                strips_vertices.append(strip_k_vertices)
            
            # split = [np.sum(array[self.chordwise_resolution*i: self.chordwise_resolution*(i+1), :], axis=0) \
            #          for i in range(self.total_spanwise_resolution)]
            # strips_vertices.append()
            
            is_symmetric.append([float(self.to_sym[i])] * points[faces[:, 0], :].shape[0])
            
            is_soft.append([wing.is_soft for k in range(len(faces))])

        front_left_vertices = np.concatenate(front_left_vertices)
        back_left_vertices = np.concatenate(back_left_vertices)
        back_right_vertices = np.concatenate(back_right_vertices)
        front_right_vertices = np.concatenate(front_right_vertices)
        is_trailing_edge = np.concatenate(is_trailing_edge)
        is_leading_edge = np.concatenate(is_leading_edge)
        # is_trailing_edge_idx = np.concatenate(is_trailing_edge_idx)
        # is_leading_edge_idx = np.concatenate(is_leading_edge_idx)
        is_symmetric = np.concatenate(is_symmetric)
        is_soft = np.concatenate(is_soft)

        ### Compute panel statistics
        diag1 = front_right_vertices - back_left_vertices
        diag2 = front_left_vertices - back_right_vertices
        cross = np.cross(diag1, diag2)
        cross_norm = np.linalg.norm(cross, axis=1)
        normal_directions = cross / tall(cross_norm)
        areas = cross_norm / 2

        # Compute the location of points of interest on each panel
        left_vortex_vertices = 0.75 * front_left_vertices + 0.25 * back_left_vertices
        right_vortex_vertices = 0.75 * front_right_vertices + 0.25 * back_right_vertices
        vortex_centers = (left_vortex_vertices + right_vortex_vertices) / 2
        vortex_bound_leg = right_vortex_vertices - left_vortex_vertices
        collocation_points = (
                0.5 * (0.25 * front_left_vertices + 0.75 * back_left_vertices) +
                0.5 * (0.25 * front_right_vertices + 0.75 * back_right_vertices)
        )
        
        # Compute local strips axes
        
        ### Compute spanwise directions by strips (i.e. local vectors along leading and trailing edges)
        le_idx = np.arange(is_leading_edge.shape[0])[is_leading_edge]
        te_idx = np.arange(is_trailing_edge.shape[0])[is_trailing_edge]
        
        # le_minus = front_left_vertices[is_leading_edge]
        # le_plus = front_right_vertices[is_leading_edge]
        # te_minus = back_left_vertices[is_trailing_edge]
        # te_plus = back_right_vertices[is_trailing_edge]
        le_minus = front_left_vertices[le_idx, :]
        le_plus = front_right_vertices[le_idx, :]
        te_minus = back_left_vertices[te_idx, :]
        te_plus = back_right_vertices[te_idx, :]
        
        le_vec = le_plus-le_minus # Leading edge-wise local vectors by strips
        te_vec = te_plus-te_minus # Trailing edge-wise local vectors by strips
        
        strips_span_vec = (le_vec + te_vec) / 2 # Mean spanwise local vectors by strips
        
        ### Compute chordwise directions by strips
        strips_chord_vec_minus = te_minus - le_minus
        strips_chord_vec_plus = te_plus - le_plus
        strips_chord_vec = (strips_chord_vec_minus + strips_chord_vec_plus) / 2 # Mean chordwise local vectors by strips
        
        # le_mid = (le_minus + le_plus) / 2
        # te_mid = (te_minus + te_plus) / 2
        
        ### Compute strips X and Y by normalizing spanwise and chordwise local directions
        # strips_x = strips_chord_vec / np.tile(np.linalg.norm(strips_chord_vec, axis=1), (3,1)).T # strips X from chordwise local vectors
        # strips_y = strips_span_vec / np.tile(np.linalg.norm(strips_span_vec, axis=1), (3,1)).T # strips Y from spanwise local vectors
            
        strips_x = normalize(strips_chord_vec)
        strips_y = normalize(strips_span_vec)
        
        ### Compute strips chords
        strips_chords = np.linalg.norm(strips_chord_vec, axis=1)

        ### Save things to the instance for later access
        self.front_left_vertices = front_left_vertices
        self.back_left_vertices = back_left_vertices
        self.back_right_vertices = back_right_vertices
        self.front_right_vertices = front_right_vertices
        self.is_trailing_edge = is_trailing_edge
        self.is_leading_edge = is_leading_edge
        # self.is_trailing_edge_idx = is_trailing_edge_idx
        # self.is_leading_edge_idx = is_leading_edge_idx
        self.is_symmetric = is_symmetric
        self.normal_directions = normal_directions
        self.areas = areas
        self.left_vortex_vertices = left_vortex_vertices
        self.right_vortex_vertices = right_vortex_vertices
        self.vortex_centers = vortex_centers
        self.vortex_bound_leg = vortex_bound_leg
        self.collocation_points = collocation_points
        
        self.strips_vertices = strips_vertices
        
        self.is_soft = is_soft
        
        self.strips_x = strips_x
        self.strips_y = strips_y
        self.strips_chords = strips_chords
        self.strips_areas = self.sum_by_strips(tall(self.areas))
        self.strips_centers = self.sum_by_strips(tall(self.areas) * self.vortex_centers) / tall(self.strips_areas)
        self.strips_quarters = self.strips_centers - self.strips_x*tall(self.strips_chords)/4
        
    def get_freestream_velocity_at_points(self,
                                          points: np.ndarray,
                                          ) -> np.ndarray:
        """
        Computes the freestream velocity at a set of points in the flowfield.

        Args:
            points: A Nx3 array of points that you would like to know the induced velocities at. Given in geometry axes.

        Returns: A Nx3 of the induced velocity at those points. Given in geometry axes.

        """ 
        
        freestream_velocities = np.ones(points.shape) * np.array([1.,0.,0.])
        
        return freestream_velocities
        
    
    def calculate_freestream_influences(self):
        
        freestream_velocities = self.get_freestream_velocity_at_points(self.collocation_points)
        freestream_direction = normalize(freestream_velocities)
        
        # Nx3, represents the normal freestream velocity at each panel collocation point
        freestream_influences = np.sum(freestream_velocities * self.normal_directions, axis=1)
        
        self.freestream_direction = freestream_direction
        self.freestream_velocities = freestream_velocities
        
        return freestream_influences
    
    
    def calculate_vortices_influences(self):
        u_collocations_unit, v_collocations_unit, w_collocations_unit = calculate_induced_velocity_horseshoe(
            x_field=tall(self.collocation_points[:, 0]),
            y_field=tall(self.collocation_points[:, 1]),
            z_field=tall(self.collocation_points[:, 2]),
            x_left=wide(self.left_vortex_vertices[:, 0]),
            y_left=wide(self.left_vortex_vertices[:, 1]),
            z_left=wide(self.left_vortex_vertices[:, 2]),
            x_right=wide(self.right_vortex_vertices[:, 0]),
            y_right=wide(self.right_vortex_vertices[:, 1]),
            z_right=wide(self.right_vortex_vertices[:, 2]),
            trailing_vortex_direction=(
                self.freestream_direction if self.align_trailing_vortices_with_wind
                else None
            ),
            gamma=1.,
            vortex_core_radius=self.vortex_core_radius
        )
        
        if self.IZsym: # If the run is symmetric
            # Mirror the vortex vertices across the Z symmetry plane
            left_vortex_vertices_mirrored = copy.copy(self.left_vortex_vertices)
            left_vortex_vertices_mirrored[:, 2] = 2 * self.Zsym - self.left_vortex_vertices[:, 2]
            
            right_vortex_vertices_mirrored = copy.copy(self.right_vortex_vertices)
            right_vortex_vertices_mirrored[:, 2] = 2 * self.Zsym - self.right_vortex_vertices[:, 2]
            
            # Compute the influence of mirrored panel (with opposite strength)
            u_collocations_unit_mirrored, v_collocations_unit_mirrored, w_collocations_unit_mirrored = calculate_induced_velocity_horseshoe(
                x_field=tall(self.collocation_points[:, 0]),
                y_field=tall(self.collocation_points[:, 1]),
                z_field=tall(self.collocation_points[:, 2]),
                x_left=wide(left_vortex_vertices_mirrored[:, 0]),
                y_left=wide(left_vortex_vertices_mirrored[:, 1]),
                z_left=wide(left_vortex_vertices_mirrored[:, 2]),
                x_right=wide(right_vortex_vertices_mirrored[:, 0]),
                y_right=wide(right_vortex_vertices_mirrored[:, 1]),
                z_right=wide(right_vortex_vertices_mirrored[:, 2]),
                trailing_vortex_direction=(
                    self.freestream_direction if self.align_trailing_vortices_with_wind
                    else None
                ),
                gamma=wide(-1. * self.is_symmetric),
                vortex_core_radius=self.vortex_core_radius
            )
            
            # Add mirror vortices influences
            u_collocations_unit = u_collocations_unit + u_collocations_unit_mirrored
            v_collocations_unit = v_collocations_unit + v_collocations_unit_mirrored
            w_collocations_unit = w_collocations_unit + w_collocations_unit_mirrored
            
        AIC = (
                u_collocations_unit * tall(self.normal_directions[:, 0]) +
                v_collocations_unit * tall(self.normal_directions[:, 1]) +
                w_collocations_unit * tall(self.normal_directions[:, 2])
              )
        
        return AIC
    
    def sum_by_strips(self, array):
        split = [np.sum(array[self.chordwise_resolution*i: self.chordwise_resolution*(i+1), :], axis=0) \
                 for i in range(self.total_spanwise_resolution)]
            
        return np.stack(split)
    
   
    def viscous_computation(self, alpha_stall, model_size):
        
        # Aerodynamic forces by strips in geometry axes

        self.strips_forces = self.sum_by_strips(self.panels_forces)
        
        self.strips_dynamic_pressure = self.dynamic_pressure.reshape(
            (self.total_spanwise_resolution, self.chordwise_resolution)
            )
        
        self.strips_pressure_coef = self.pressure_coef.reshape(
            (self.total_spanwise_resolution, self.chordwise_resolution)
            )
        
        # Aerodynamic moments by strips in geometry axes

        self.strips_moments = self.sum_by_strips(self.panels_moments)
            
        # Mean freestream by strips in geometry axes

        self.strips_freestream = self.sum_by_strips(tall(self.areas) * self.freestream_velocities) / tall(self.strips_areas)
        
        # Compute viscous forces by strip using Neuralfoil
        
        ### Project strips forces in strips local planes
        strips_forces_transverse = tall(np.sum(self.strips_forces * self.strips_y, axis=1)) * self.strips_y
        strips_freestream_transverse = tall(np.sum(self.strips_freestream * self.strips_y, axis=1)) * self.strips_y
        
        self.strips_forces_in_strips_planes = self.strips_forces - strips_forces_transverse # strips forces in strips planes
        self.strips_freestream_in_strips_planes = self.strips_freestream - strips_freestream_transverse # strips freestream in strips planes
        
        self.strips_freestream_x = normalize(self.strips_freestream_in_strips_planes) # strips Y from spanwise local vectors
        
        ### Compute strips Z
        self.strips_z = normalize(np.cross(self.strips_freestream_x, self.strips_y))  # (n, 3)

        ### Join strips X, Y and Z 
        
        self.strips_freestream_axes = [
                np.array([[1.,0.,0.],]*3)*tall(self.strips_freestream_x[i,:]) +\
                np.array([[0.,1.,0.],]*3)*tall(self.strips_y[i,:]) +\
                np.array([[0.,0.,1.],]*3)*tall(self.strips_z[i,:]) \
            for i in range(self.total_spanwise_resolution)
        ]
        
        strips_freestream_axes_inv = [np.linalg.inv(ax) for ax in self.strips_freestream_axes]
        self.strips_freestream_axes_inv = strips_freestream_axes_inv
        
        ### Compute aerodynamic forces by freestream axes by strips

        strips_forces_strips_freestream = np.array([
            change_basis(x=self.strips_forces_in_strips_planes[i,0],
                         y=self.strips_forces_in_strips_planes[i,1],
                         z=self.strips_forces_in_strips_planes[i,2],
                         from_axes = self.strips_freestream_axes[i],
                         # to_axes = self.strips_freestream_axes[i],
                         # to_axes = self.strips_freestream_axes_inv[i],
                         )
            for i in range(self.total_spanwise_resolution)
        ])
        
        # strips_forces_strips_freestream = self.strips_forces_in_strips_planes * self.strips_freestream_axes[0][2,2]
            
        self.strips_forces_strips_freestream = strips_forces_strips_freestream
        
        ### Compute induced angle of attack by strips (= arctan(L/Di))
        # self.alphai = np.rad2deg(np.arctan(strips_forces_strips_freestream[:,0]/strips_forces_strips_freestream[:,2]))
        self.alphai = np.arctan(strips_forces_strips_freestream[:,0]/strips_forces_strips_freestream[:,2]) * 180/np.pi
        
        ### Freestream angle of attack by strips
        stream_angle = np.arctan2(-self.strips_freestream_x[:,1], self.strips_freestream_x[:,0])
        deflection_angle = np.arctan2(-self.strips_x[:,1], self.strips_x[:,0])
        # self.alpha = np.rad2deg(stream_angle-deflection_angle)
        alpha = (stream_angle-deflection_angle) * 180/np.pi
        
        self.alpha = np.mod(alpha + 180., 360.) - 180.
        
        ### Separate strips freestream velocities
        strips_U = np.linalg.norm(self.strips_freestream_in_strips_planes, axis=1) # Z,X plane
        strips_V = np.linalg.norm(strips_freestream_transverse, axis=1) # Z,X plane
        # NOW NEED TO HANDLE TRANSVERSE FRICTION
        # FLAT PLATE COEFFICIENT WITH PROJECTION IN STRIP SPANWISE ORIENTATION?
        
        self.strips_U = strips_U
        self.strips_V = strips_V
        self.strips_reynolds = self.strips_chords * strips_U / self.fluid.kinematic_viscosity
        
        ### Get airfoils by strips
        airfoils = []
        for i, w in enumerate(self.airplane.wings):
            for i in range(len(w.xsecs)-1):
                # Add mean strip airfoil for each strip
                airfoils += [w.xsecs[i].airfoil.blend_with_another_airfoil(w.xsecs[i+1].airfoil)]*\
                    self.spanwise_resolution*(1+w.symmetric)
            
        self.airfoils = np.array(airfoils)
        
        ### Call Neuralfoil for each strip
        self.stall_fac = stall_factor(
            self.alpha,
            self.alphai,
            alpha_stall,
            ) # 0 when stalled, 1 when laminar
        
        # when using soft wings, a negative vortex strength at the l.e. means stall
        
        mean_z = np.mean(self.vortex_centers[:,2])
        
        le_idx = np.arange(self.is_leading_edge.shape[0])[self.is_leading_edge]
        te_idx = np.arange(self.is_trailing_edge.shape[0])[self.is_trailing_edge]
     
        # TODO: find a better low-aoa / soft-wing stall criterion
        # for the moment, funcitonnal because z_mean > 0 for sails and < 0 for appendages
        
        # def kC(alpha_eff, xc, mc):
        #     k = 2.
        #     # D = leading_edge_camber(xc, mc)
        #     D = mc/xc
        #     C = np.sign(D) * (alpha_eff - 0.5*180/np.pi * np.arctan(D))
            
        #     return k*C
        
        # self.alphaeff0 = self.alpha - self.alphai * self.stall_fac
        
        # xc = 0.4
        
        # self.cambers = np.array([af.max_camber() for i, af in enumerate(self.airfoils)])
        
        # self.insight0 = [af.max_camber() for i, af in enumerate(self.airfoils)]
        # self.insight1 = [leading_edge_camber(xc, af.max_camber()) for i, af in enumerate(self.airfoils)]
        # self.insight2 = [kC(self.alphaeff0[i], xc, af.max_camber()) for i, af in enumerate(self.airfoils)]
        # self.insight3 = smooth_ramp(self.vortex_strengths[le_idx] * self.vortex_strengths[te_idx])
        # self.insight4 = smooth_ramp(self.alphaeff0 * self.cambers)
        
        # self.soft_stall_fac = np.concatenate([
        #     (np.tanh(kC(self.alphaeff0[i], xc, af.max_camber()) + 1.) + 1.)/2.
        #     for i, af in enumerate(self.airfoils)
        # ])
        # self.soft_stall_fac = np.fmax(1 - self.is_soft, (
        #     smooth_ramp(self.vortex_strengths[le_idx] * self.vortex_strengths[te_idx]) *\
        #     smooth_ramp(self.alphaeff0 * 100*self.cambers)  # < 0 when stalled, >= 0 when laminar
        # )) # 0 when stalled, 1 when laminar
        
        self.soft_stall_fac = ramp(
        # self.soft_stall_fac = smooth_ramp(
            self.vortex_strengths[le_idx] *\
            mean_z *\
            self.is_soft[le_idx], # < 0 when stalled, >= 0 when laminar
        )# 0 when stalled, 1 when laminar
        
        # self.vertices_soft_stall_fac = tall(np.zeros(self.total_spanwise_resolution + len(self.airplane.wings)))
        # self.vertices_stall_fac = tall(np.zeros(self.total_spanwise_resolution + len(self.airplane.wings)))
        self.vertices_soft_stall_fac = []
        self.vertices_stall_fac = []
        
        # extend the stall factors to the borders of strips (useful for 3D animated display)
        for i, wing in enumerate(self.airplane.wings):
            wing_spanwise_resolution = self.wing_spanwise_resolution[i]
            
            start = i*(wing_spanwise_resolution)
            end = (i+1)*(wing_spanwise_resolution)
            
            self.vertices_soft_stall_fac.append(self.soft_stall_fac[start])
            self.vertices_stall_fac.append(self.stall_fac[start])
            
            for k in range(start, end-1):
                self.vertices_soft_stall_fac.append((self.soft_stall_fac[k] + self.soft_stall_fac[k+1]) / 2)
                self.vertices_stall_fac.append((self.stall_fac[k] + self.stall_fac[k+1]) / 2)
                # self.vertices_soft_stall_fac.append((self.soft_stall_fac[k, 0] + self.soft_stall_fac[k+1, 0]) / 2)
                # self.vertices_stall_fac.append((self.stall_fac[k, 0] + self.stall_fac[k+1, 0]) / 2)
                
            self.vertices_soft_stall_fac.append(self.soft_stall_fac[end-1])
            self.vertices_stall_fac.append(self.stall_fac[end-1])
            # self.vertices_soft_stall_fac.append(self.soft_stall_fac[end-1, 0])
            # self.vertices_stall_fac.append(self.stall_fac[end-1, 0])
        
        
        self.alphaeff = self.alpha - self.alphai * self.soft_stall_fac * self.stall_fac
        
        self.nf_results = [
                af.get_aero_from_neuralfoil(
                    alpha=self.alphaeff[i],
                    Re=self.strips_reynolds[i],
                    # model_size="xxlarge",
                    # model_size="xlarge",
                    # model_size="large",
                    # model_size="medium",
                    # model_size="small",
                    # model_size="xsmall",
                    # model_size="xxsmall",
                    model_size=model_size,
                    n_crit = 9.0,
                    xtr_upper = 0.1,
                    xtr_lower = 0.1,
                ) for i, af in enumerate(self.airfoils)
            ]
        
        self.strips_CL = np.array([res['CL'][0] for res in self.nf_results])
        self.strips_CD = np.array([res['CD'][0] for res in self.nf_results])
        self.strips_CF = np.sqrt(self.strips_CL**2 + self.strips_CD**2)
        self.strips_CM = np.array([res['CM'][0] for res in self.nf_results])
        
        self.strips_L = tall(self.strips_CL) * 1/2 * self.fluid.density * self.strips_areas * tall(self.strips_U)**2 *\
            tall(self.soft_stall_fac) # lift force crumbles when the angle of attack of the soft wing becomes too small
            
        self.strips_D = tall(self.strips_CD) * 1/2 * self.fluid.density * self.strips_areas * tall(self.strips_U)**2
        
        self.strips_Li = tall(self.strips_forces_strips_freestream[:, 2])
        
        self.strips_Di = tall(self.strips_forces_strips_freestream[:, 0]) *\
            tall(self.soft_stall_fac) *\
            tall(ReLU(self.strips_L / self.strips_Li))

        self.strips_F = np.multiply(tall(self.strips_L), wide(np.array([0., 0., 1.]))) +\
                       np.multiply(tall(self.strips_D), wide(np.array([1., 0., 0.])))
        
        self.strips_M = tall(self.strips_CM) * 1/2 * self.fluid.density * self.strips_areas * tall(self.strips_U)**2 * tall(self.strips_chords)


        self.strips_coe = self.strips_quarters - self.strips_x*tall(self.strips_CM/self.strips_CF)
        
        
        ### Reproject strips viscous drag forces from strips freestream axes to geometry axes
        
        self.L_vec = np.multiply(tall(self.strips_L), wide(np.array([0., 0., 1.])))
                       
        self.D_vec = np.multiply(tall(self.strips_D), wide(np.array([1., 0., 0.])))
        
        self.Di_vec = np.multiply(tall(self.strips_Di), wide(np.array([1., 0., 0.])))
        
        self.M_vec = np.multiply(tall(self.strips_M), wide(np.array([0., 0., 1.])))
                    
        visc_forces_geometry = np.array([
            change_basis(x=self.D_vec[i,0],
                          y=self.D_vec[i,1],
                          z=self.D_vec[i,2],
                          to_axes = self.strips_freestream_axes[i],
                          )
            for i in range(self.total_spanwise_resolution)
        ])
        
        induced_forces_geometry = np.array([
            change_basis(x=self.Di_vec[i,0],
                          y=self.Di_vec[i,1],
                          z=self.Di_vec[i,2],
                          to_axes = self.strips_freestream_axes[i],
                          )
            for i in range(self.total_spanwise_resolution)
        ])
        
        lift_forces_geometry = np.array([
            change_basis(x=self.L_vec[i,0],
                          y=self.L_vec[i,1],
                          z=self.L_vec[i,2],
                          to_axes = self.strips_freestream_axes[i],
                          )
            for i in range(self.total_spanwise_resolution)
        ])
        
        visc_moments_geometry = np.cross(
            np.add(self.strips_coe, -wide(np.array(self.xyz_ref))),
            visc_forces_geometry
        )
        
        induced_moments_geometry = 0. # TODO: need to separate VLM lift moment from VLM induced drag moment
        
        lift_moments_geometry = np.cross(
            np.add(self.strips_coe, -wide(np.array(self.xyz_ref))),
            lift_forces_geometry
        )
        
        visc_force_geometry = np.sum(visc_forces_geometry, axis=0) # NF viscous drag total force in the global geometry axes
        visc_moment_geometry = np.sum(visc_moments_geometry, axis=0) # NF viscous drag total force in the global geometry axes
        
        induced_force_geometry = np.sum(induced_forces_geometry, axis=0) # VLM induced drag total force in the global geometry axes
        induced_moment_geometry = np.sum(induced_moments_geometry, axis=0) # VLM induced drag total force in the global geometry axes
        
        lift_force_geometry = np.sum(lift_forces_geometry, axis=0) # NF lift total force in the global geometry axes
        lift_moment_geometry = np.sum(lift_moments_geometry, axis=0) # NF lift total force in the global geometry axes
        
        self.visc_forces_geometry = visc_forces_geometry
        self.visc_force_geometry = visc_force_geometry
        
        self.visc_moments_geometry = visc_moments_geometry
        self.visc_moment_geometry = visc_moment_geometry
        
        self.induced_forces_geometry = induced_forces_geometry
        self.induced_force_geometry = induced_force_geometry
        
        self.induced_moments_geometry = induced_moments_geometry
        self.induced_moment_geometry = induced_moment_geometry
        
        self.lift_forces_geometry = lift_forces_geometry
        self.lift_force_geometry = lift_force_geometry
        
        self.lift_moments_geometry = lift_moments_geometry
        self.lift_moment_geometry = lift_moment_geometry
        

    def run(
        self,
        alpha_stall: float = 15.,
        model_size: str = "medium",
    ) -> Dict[str, Any]:
        """
        Computes the aerodynamic forces.

        Returns a dictionary with keys:

            - 'F_g' : an [x, y, z] list of forces in geometry axes [N]
            - 'F_b' : an [x, y, z] list of forces in body axes [N]
            - 'F_w' : an [x, y, z] list of forces in wind axes [N]
            - 'M_g' : an [x, y, z] list of moments about geometry axes [Nm]
            - 'M_b' : an [x, y, z] list of moments about body axes [Nm]
            - 'M_w' : an [x, y, z] list of moments about wind axes [Nm]
            - 'L' : the lift force [N]. Definitionally, this is in wind axes.
            - 'Y' : the side force [N]. This is in wind axes.
            - 'D' : the drag force [N]. Definitionally, this is in wind axes.
            - 'l_b', the rolling moment, in body axes [Nm]. Positive is roll-right.
            - 'm_b', the pitching moment, in body axes [Nm]. Positive is pitch-up.
            - 'n_b', the yawing moment, in body axes [Nm]. Positive is nose-right.
            - 'CL', the lift coefficient [-]. Definitionally, this is in wind axes.
            - 'CY', the sideforce coefficient [-]. This is in wind axes.
            - 'CD', the drag coefficient [-]. Definitionally, this is in wind axes.
            - 'Cl', the rolling coefficient [-], in body axes
            - 'Cm', the pitching coefficient [-], in body axes
            - 'Cn', the yawing coefficient [-], in body axes
            
        # TODO: find a more specific way to compute alpha stall

        Nondimensional values are nondimensionalized using reference values in the VortexLatticeMethod.airplane object.
        """

        if self.verbose:
            print("Meshing...")

        self.mesh_geometry()

        ##### Setup Operating Point
        if self.verbose:
            print("Calculating the freestream influence...")
            
        freestream_influences = self.calculate_freestream_influences()

        ##### Setup Geometry
        ### Calculate AIC matrix
        if self.verbose:
            print("Calculating the collocation influence matrix...")

        AIC = self.calculate_vortices_influences()

        ##### Calculate Vortex Strengths
        if self.verbose:
            print("Calculating vortex strengths...")

        self.vortex_strengths = np.linalg.solve(AIC, -freestream_influences)
        
        ##### Calculate forces
        ### Calculate Near-Field Forces and Moments
        # Governing Equation: The force on a straight, small vortex filament is F = rho * cross(V, l) * gamma,
        # where rho is density, V is the velocity vector, cross() is the cross product operator,
        # l is the vector of the filament itself, and gamma is the circulation.

        if self.verbose:
            print("Calculating forces on each panel...")
            
        # Calculate the induced velocity at the center of each bound leg
        V_centers = self.get_velocity_at_points(self.vortex_centers)

        # Calculate forces_inviscid_geometry, the force on the ith panel. Note that this is in GEOMETRY AXES,
        # not WIND AXES or BODY AXES.
        Vi_cross_li = np.cross(V_centers, self.vortex_bound_leg, axis=1)

        forces_geometry = self.fluid.density * Vi_cross_li * tall(self.vortex_strengths)
        moments_geometry = np.cross(
            np.add(self.vortex_centers, -wide(np.array(self.xyz_ref))),
            forces_geometry
        )
        
        self.panels_forces = forces_geometry
        self.panels_moments = moments_geometry
        
        self.panels_normal_forces = tall(np.sum(self.panels_forces * self.normal_directions, axis=1)) * self.normal_directions
        
        normal_forces_norm = np.linalg.norm(self.panels_normal_forces, axis=1)
        normal_forces_dir = np.sum(self.panels_normal_forces * self.normal_directions, axis=1) /normal_forces_norm
        
        self.dynamic_pressure = normal_forces_dir * normal_forces_norm / self.areas
        self.pressure_coef = self.dynamic_pressure / self.fluid_dynamic_pressure()
        
        if self.verbose:
            print("Calculating viscous forces on each strip...")
            
        self.viscous_computation(alpha_stall, model_size)
            
        # Calculate total forces and moments
        force_geometry = np.sum(forces_geometry, axis=0)
        moment_geometry = np.sum(moments_geometry, axis=0)
        
        # centroid_geometry = np.sum(forces_geometry*vortex_centers, axis=0) / force_geometry
        # centroid_geometry = np.dot(np.sum(forces_geometry, axis=1), vortex_centers) / np.sum(forces_geometry)
        # centroid_geometry = np.dot(np.linalg.norm(forces_geometry, axis=1), self.vortex_centers) / np.sum(np.linalg.norm(forces_geometry, axis=1))

        self.force_wind = self.op_point.convert_axes(
            force_geometry[0], force_geometry[1], force_geometry[2],
            from_axes="geometry",
            to_axes="wind"
        )
        self.visc_force_wind = self.op_point.convert_axes(
            self.visc_force_geometry[0], self.visc_force_geometry[1], self.visc_force_geometry[2],
            from_axes="geometry",
            to_axes="wind"
        )
        self.induced_force_wind = self.op_point.convert_axes(
            self.induced_force_geometry[0], self.induced_force_geometry[1], self.induced_force_geometry[2],
            from_axes="geometry",
            to_axes="wind"
        )
        self.lift_force_wind = self.op_point.convert_axes(
            self.lift_force_geometry[0], self.lift_force_geometry[1], self.lift_force_geometry[2],
            from_axes="geometry",
            to_axes="wind"
        )
        self.moment_wind = self.op_point.convert_axes(
            moment_geometry[0], moment_geometry[1], moment_geometry[2],
            from_axes="geometry",
            to_axes="wind"
        )
        self.visc_moment_wind = self.op_point.convert_axes(
            self.visc_moment_geometry[0], self.visc_moment_geometry[1], self.visc_moment_geometry[2],
            from_axes="geometry",
            to_axes="wind"
        )
        
        self.induced_moment_wind = 0.
        
        self.lift_moment_wind = self.op_point.convert_axes(
            self.lift_moment_geometry[0], self.lift_moment_geometry[1], self.lift_moment_geometry[2],
            from_axes="geometry",
            to_axes="wind"
        )

        self.force_underway = self.op_point.convert_axes(
            force_geometry[0], force_geometry[1], force_geometry[2],
            from_axes="geometry",
            to_axes="underway"
        )
        self.visc_force_underway = self.op_point.convert_axes(
            self.visc_force_geometry[0], self.visc_force_geometry[1], self.visc_force_geometry[2],
            from_axes="geometry",
            to_axes="underway"
        )
        self.induced_force_underway = self.op_point.convert_axes(
            self.induced_force_geometry[0], self.induced_force_geometry[1], self.induced_force_geometry[2],
            from_axes="geometry",
            to_axes="underway"
        )
        self.lift_force_underway = self.op_point.convert_axes(
            self.lift_force_geometry[0], self.lift_force_geometry[1], self.lift_force_geometry[2],
            from_axes="geometry",
            to_axes="underway"
        )
        self.moment_underway = self.op_point.convert_axes(
            moment_geometry[0], moment_geometry[1], moment_geometry[2],
            from_axes="geometry",
            to_axes="underway"
        )
        self.visc_moment_underway = self.op_point.convert_axes(
            self.visc_moment_geometry[0], self.visc_moment_geometry[1], self.visc_moment_geometry[2],
            from_axes="geometry",
            to_axes="underway"
        )
        
        self.induced_moment_underway = 0.
        
        self.lift_moment_underway = self.op_point.convert_axes(
            self.lift_moment_geometry[0], self.lift_moment_geometry[1], self.lift_moment_geometry[2],
            from_axes="geometry",
            to_axes="underway"
        )
        
        ### Save things to the instance for later access
        self.forces_geometry = self.panels_forces
        # self.visc_force_wind = visc_force_wind
        self.moments_geometry = moments_geometry
        # self.centroid = centroid_geometry
        self.force_geometry = force_geometry
        self.force_geometry = force_geometry
        # self.force_wind = force_wind
        self.moment_geometry = moment_geometry
        # self.moment_wind = moment_wind
        # self.visc_moment_wind = visc_moment_wind

        # Calculate dimensional forces
        # L = self.force_wind[1]
        L = (self.force_wind[1] + self.visc_force_wind[1])
        Di = self.force_wind[0]
        Dv = self.visc_force_wind[0]
        D = Di + Dv
        Z = self.force_wind[2] + self.visc_force_wind[2]
        # l_b = moment_geometry[0]
        # m_b = moment_geometry[1]
        # n_b = moment_geometry[2]
        Fx = (self.force_underway[0] + self.visc_force_underway[0])
        Fy = (self.force_underway[1] + self.visc_force_underway[1])
        
        # Calculate nondimensional forces
        q = self.fluid_dynamic_pressure()
        s_ref = self.airplane.s_ref
        b_ref = self.airplane.b_ref
        c_ref = self.airplane.c_ref
        CL = L / q / s_ref
        CDi = Di / q / s_ref
        CDv = Dv / q / s_ref
        CD = D / q / s_ref
        CZ = Z / q / s_ref
        Cx = Fx / q / s_ref
        Cy = Fy / q / s_ref
        # Cl = l_b / q / s_ref / b_ref
        # Cm = m_b / q / s_ref / c_ref
        # Cn = n_b / q / s_ref / b_ref

        return {
            # "centroid": centroid_geometry,
            
            "F_ab": self.lift_force_underway + self.visc_force_underway + self.induced_force_underway,
            "Fvlm_ab": self.force_underway,
            "Fnf_ab": self.lift_force_underway,
            "Fv_ab": self.visc_force_underway,
            "Find_ab": self.induced_force_underway,
            
            # "F_g": self.force_geometry + self.visc_force_geometry,
            # "Fi_g": self.force_geometry,
            # "Fv_g": self.visc_force_geometry,
            
            "F_w": self.lift_force_wind + self.visc_force_wind + self.induced_force_wind,
            "Fvlm_w": self.force_wind,
            "Fnf_w": self.lift_force_wind,
            "Fv_w": self.visc_force_wind,
            
            "M_ab": self.lift_moment_underway + self.visc_moment_underway + self.induced_moment_underway,
            "Mvlm_ab": self.moment_underway,
            "Mnf_ab": self.lift_moment_underway,
            "Mv_ab": self.visc_moment_underway,
            "Mind_ab": self.induced_moment_underway,
            
            # "M_g": self.moment_geometry + self.visc_force_geometry,
            # "Mi_g": self.moment_geometry,
            # "Mv_g": self.visc_moment_geometry,
            
            "M_w": self.lift_moment_wind + self.visc_moment_wind + self.induced_moment_wind,
            "Mvlm_w": self.moment_wind,
            "Mnf_w": self.lift_moment_wind,
            "Mv_w": self.visc_moment_wind,
            "Mind_w": self.induced_force_wind,
            
            "L"  : L,
            "D"  : D,
            "Di" : Di,
            "Dv" : Dv,
            "Z"  : Z,
            
            # "l_b": l_b,
            # "m_b": m_b,
            # "n_b": n_b,
            
            "CL" : CL,
            "CD" : CD,
            "CDi": CDi,
            "CDv": CDv,
            "CZ" : CZ,
            "Cx" : Cx,
            "Cy" : Cy,
            
            # "Cl" : Cl,
            # "Cm" : Cm,
            # "Cn" : Cn,
        }

    def run_with_stability_derivatives(self,
                                       alpha=True,
                                       beta=True,
                                       p=True,
                                       q=True,
                                       r=True,
                                       ):
        """
                Computes the aerodynamic forces and moments on the airplane, and the stability derivatives.

                Arguments essentially determine which stability derivatives are computed. If a stability derivative is not
                needed, leaving it False will speed up the computation.

                Args:

                    - alpha (bool): If True, compute the stability derivatives with respect to the angle of attack (alpha).
                    - beta (bool): If True, compute the stability derivatives with respect to the sideslip angle (beta).
                    - p (bool): If True, compute the stability derivatives with respect to the body-axis roll rate (p).
                    - q (bool): If True, compute the stability derivatives with respect to the body-axis pitch rate (q).
                    - r (bool): If True, compute the stability derivatives with respect to the body-axis yaw rate (r).

                Returns: a dictionary with keys:

                    - 'F_g' : an [x, y, z] list of forces in geometry axes [N]
                    - 'F_b' : an [x, y, z] list of forces in body axes [N]
                    - 'F_w' : an [x, y, z] list of forces in wind axes [N]
                    - 'M_g' : an [x, y, z] list of moments about geometry axes [Nm]
                    - 'M_b' : an [x, y, z] list of moments about body axes [Nm]
                    - 'M_w' : an [x, y, z] list of moments about wind axes [Nm]
                    - 'L' : the lift force [N]. Definitionally, this is in wind axes.
                    - 'Y' : the side force [N]. This is in wind axes.
                    - 'D' : the drag force [N]. Definitionally, this is in wind axes.
                    - 'l_b', the rolling moment, in body axes [Nm]. Positive is roll-right.
                    - 'm_b', the pitching moment, in body axes [Nm]. Positive is pitch-up.
                    - 'n_b', the yawing moment, in body axes [Nm]. Positive is nose-right.
                    - 'CL', the lift coefficient [-]. Definitionally, this is in wind axes.
                    - 'CY', the sideforce coefficient [-]. This is in wind axes.
                    - 'CD', the drag coefficient [-]. Definitionally, this is in wind axes.
                    - 'Cl', the rolling coefficient [-], in body axes
                    - 'Cm', the pitching coefficient [-], in body axes
                    - 'Cn', the yawing coefficient [-], in body axes

                    Along with additional keys, depending on the value of the `alpha`, `beta`, `p`, `q`, and `r` arguments. For
                    example, if `alpha=True`, then the following additional keys will be present:

                        - 'CLa', the lift coefficient derivative with respect to alpha [1/rad]
                        - 'CDa', the drag coefficient derivative with respect to alpha [1/rad]
                        - 'CYa', the sideforce coefficient derivative with respect to alpha [1/rad]
                        - 'Cla', the rolling moment coefficient derivative with respect to alpha [1/rad]
                        - 'Cma', the pitching moment coefficient derivative with respect to alpha [1/rad]
                        - 'Cna', the yawing moment coefficient derivative with respect to alpha [1/rad]
                        - 'x_np', the neutral point location in the x direction [m]

                    Nondimensional values are nondimensionalized using reference values in the
                    VortexLatticeMethod.airplane object.

                    Data types:
                        - The "L", "Y", "D", "l_b", "m_b", "n_b", "CL", "CY", "CD", "Cl", "Cm", and "Cn" keys are:

                            - floats if the OperatingPoint object is not vectorized (i.e., if all attributes of OperatingPoint
                            are floats, not arrays).

                            - arrays if the OperatingPoint object is vectorized (i.e., if any attribute of OperatingPoint is an
                            array).

                        - The "F_g", "F_b", "F_w", "M_g", "M_b", and "M_w" keys are always lists, which will contain either
                        floats or arrays, again depending on whether the OperatingPoint object is vectorized or not.

                """
        abbreviations = {
            "alpha": "a",
            "beta" : "b",
            "p"    : "p",
            "q"    : "q",
            "r"    : "r",
        }
        finite_difference_amounts = {
            "alpha": 0.001,
            "beta" : 0.001,
            "p"    : 0.001 * (2 * self.op_point.velocity) / self.airplane.b_ref,
            "q"    : 0.001 * (2 * self.op_point.velocity) / self.airplane.c_ref,
            "r"    : 0.001 * (2 * self.op_point.velocity) / self.airplane.b_ref,
        }
        scaling_factors = {
            "alpha": np.degrees(1),
            "beta" : np.degrees(1),
            "p"    : (2 * self.op_point.velocity) / self.airplane.b_ref,
            "q"    : (2 * self.op_point.velocity) / self.airplane.c_ref,
            "r"    : (2 * self.op_point.velocity) / self.airplane.b_ref,
        }

        original_op_point = self.op_point

        # Compute the point analysis, which returns a dictionary that we will later add key:value pairs to.
        run_base = self.run()

        # Note for the loops below: here, "derivative numerator" and "... denominator" refer to the quantity being
        # differentiated and the variable of differentiation, respectively. In other words, in the expression df/dx,
        # the "numerator" is f, and the "denominator" is x. I realize that this would make a mathematician cry (as a
        # partial derivative is not a fraction), but the reality is that there seems to be no commonly-accepted name
        # for these terms. (Curiously, this contrasts with integration, where there is an "integrand" and a "variable
        # of integration".)

        for derivative_denominator in abbreviations.keys():
            if not locals()[derivative_denominator]:  # Basically, if the parameter from the function input is not True,
                continue  # Skip this run.
                # This way, you can (optionally) speed up this routine if you only need static derivatives,
                # or longitudinal derivatives, etc.

            # These lines make a copy of the original operating point, incremented by the finite difference amount
            # along the variable defined by derivative_denominator.
            incremented_op_point = copy.copy(original_op_point)
            incremented_op_point.__setattr__(
                derivative_denominator,
                original_op_point.__getattribute__(derivative_denominator) + finite_difference_amounts[
                    derivative_denominator]
            )

            vlm_incremented = copy.copy(self)
            vlm_incremented.op_point = incremented_op_point
            run_incremented = vlm_incremented.run()

            for derivative_numerator in [
                "CL",
                "CD",
                "CY",
                "Cl",
                "Cm",
                "Cn",
            ]:
                derivative_name = derivative_numerator + abbreviations[derivative_denominator]  # Gives "CLa"
                run_base[derivative_name] = (
                        (  # Finite-difference out the derivatives
                                run_incremented[derivative_numerator] - run_base[
                            derivative_numerator]
                        ) / finite_difference_amounts[derivative_denominator]
                        * scaling_factors[derivative_denominator]
                )

            ### Try to compute and append neutral point, if possible
            if derivative_denominator == "alpha":
                run_base["x_np"] = self.xyz_ref[0] - (
                        run_base["Cma"] * (self.airplane.c_ref / run_base["CLa"])
                )
            if derivative_denominator == "beta":
                run_base["x_np_lateral"] = self.xyz_ref[0] - (
                        run_base["Cnb"] * (self.airplane.b_ref / run_base["CYb"])
                )

        return run_base

    def get_induced_velocity_at_points(self,
                                       points: np.ndarray,
                                       ) -> np.ndarray:
        """
        Computes the induced velocity at a set of points in the flowfield.

        Args:
            points: A Nx3 array of points that you would like to know the induced velocities at. Given in geometry axes.

        Returns: A Nx3 of the induced velocity at those points. Given in geometry axes.

        """ 
        u_induced, v_induced, w_induced = calculate_induced_velocity_horseshoe(
            x_field=tall(points[:, 0]),
            y_field=tall(points[:, 1]),
            z_field=tall(points[:, 2]),
            x_left=wide(self.left_vortex_vertices[:, 0]),
            y_left=wide(self.left_vortex_vertices[:, 1]),
            z_left=wide(self.left_vortex_vertices[:, 2]),
            x_right=wide(self.right_vortex_vertices[:, 0]),
            y_right=wide(self.right_vortex_vertices[:, 1]),
            z_right=wide(self.right_vortex_vertices[:, 2]),
            trailing_vortex_direction=self.freestream_direction if self.align_trailing_vortices_with_wind else None,
            gamma=wide(self.vortex_strengths),
            vortex_core_radius=self.vortex_core_radius
        )
            
        if self.IZsym: # If the run is symmetric
            # Mirror the vortex vertices across the Z symmetry plane
            left_vortex_vertices_mirrored = copy.copy(self.left_vortex_vertices)
            left_vortex_vertices_mirrored[:, 2] = 2 * self.Zsym - self.left_vortex_vertices[:, 2]
            
            right_vortex_vertices_mirrored = copy.copy(self.right_vortex_vertices)
            right_vortex_vertices_mirrored[:, 2] = 2 * self.Zsym - self.right_vortex_vertices[:, 2]
            
            # Compute the influence of mirrored panel (with opposite strength)
            u_induced_mirrored, v_induced_mirrored, w_induced_mirrored = calculate_induced_velocity_horseshoe(
                x_field=tall(points[:, 0]),
                y_field=tall(points[:, 1]),
                z_field=tall(points[:, 2]),
                x_left=wide(left_vortex_vertices_mirrored[:, 0]),
                y_left=wide(left_vortex_vertices_mirrored[:, 1]),
                z_left=wide(left_vortex_vertices_mirrored[:, 2]),
                x_right=wide(right_vortex_vertices_mirrored[:, 0]),
                y_right=wide(right_vortex_vertices_mirrored[:, 1]),
                z_right=wide(right_vortex_vertices_mirrored[:, 2]),
                trailing_vortex_direction=self.freestream_direction if self.align_trailing_vortices_with_wind else None,
                gamma=wide(-self.vortex_strengths * self.is_symmetric,),
                vortex_core_radius=self.vortex_core_radius
            )
                    
            u_induced += u_induced_mirrored
            v_induced += v_induced_mirrored
            w_induced += w_induced_mirrored
            
        u_induced = np.sum(u_induced, axis=1)
        v_induced = np.sum(v_induced, axis=1)
        w_induced = np.sum(w_induced, axis=1)

        V_induced = np.stack([
            u_induced, v_induced, w_induced
        ], axis=1)

        return V_induced

    def get_velocity_at_points(self,
                               points: np.ndarray
                               ) -> np.ndarray:
        """
        Computes the velocity at a set of points in the flowfield.

        Args:
            points: A Nx3 array of points that you would like to know the velocities at. Given in geometry axes.

        Returns: A Nx3 of the velocity at those points. Given in geometry axes.

        """
        V_induced = self.get_induced_velocity_at_points(points)

        V_freestream = self.get_freestream_velocity_at_points(points)

        V = V_induced + V_freestream
        return V

    def calculate_streamlines(self,
                              seed_points: np.ndarray = None,
                              n_steps: int = 300,
                              length: float = None,
                              ) -> np.ndarray:
        """
        Computes streamlines, starting at specific seed points.

        After running this function, a new instance variable `VortexLatticeFilaments.streamlines` is computed

        Uses simple forward-Euler integration with a fixed spatial stepsize (i.e., velocity vectors are normalized
        before ODE integration). After investigation, it's not worth doing fancier ODE integration methods (adaptive
        schemes, RK substepping, etc.), due to the near-singular conditions near vortex filaments.

        Args:

            seed_points: A Nx3 ndarray that contains a list of points where streamlines are started. Will be
            auto-calculated if not specified.

            n_steps: The number of individual streamline steps to trace. Minimum of 2.

            length: The approximate total length of the streamlines desired, in meters. Will be auto-calculated if
            not specified.

        Returns:
            streamlines: a 3D array with dimensions: (n_seed_points) x (3) x (n_steps).
            Consists of streamlines data.

            Result is also saved as an instance variable, VortexLatticeMethod.streamlines.

        """
        if self.verbose:
            print("Calculating streamlines...")
        if length is None:
            length = self.airplane.c_ref * 5
        if seed_points is None:
            left_TE_vertices = self.back_left_vertices[self.is_trailing_edge.astype(bool)]
            right_TE_vertices = self.back_right_vertices[self.is_trailing_edge.astype(bool)]
            N_streamlines_target = 200
            seed_points_per_panel = np.maximum(1, N_streamlines_target // len(left_TE_vertices))

            nondim_node_locations = np.linspace(0, 1, seed_points_per_panel + 1)
            nondim_seed_locations = (nondim_node_locations[1:] + nondim_node_locations[:-1]) / 2

            seed_points = np.concatenate([
                x * left_TE_vertices + (1 - x) * right_TE_vertices
                for x in nondim_seed_locations
            ])

        streamlines = np.empty((len(seed_points), 3, n_steps))
        streamlines[:, :, 0] = seed_points
        for i in range(1, n_steps):
            V = self.get_velocity_at_points(streamlines[:, :, i - 1])
            streamlines[:, :, i] = (
                    streamlines[:, :, i - 1] +
                    length / n_steps * V / tall(np.linalg.norm(V, axis=1))
            )

        self.streamlines = streamlines

        if self.verbose:
            print("Streamlines calculated.")

        return streamlines

    def draw(self,
             c: np.ndarray = None,
             cmap: str = None,
             colorbar_label: str = None,
             show: bool = True,
             show_kwargs: Dict = None,
             draw_streamlines=True,
             recalculate_streamlines=False,
             plot_axes=True,
             backend: str = "pyvista"
             ):
        """
        Draws the solution. Note: Must be called on a SOLVED AeroProblem object.
        To solve an AeroProblem, use opti.solve(). To substitute a solved solution, use ap = sol(ap).
        :return:
        """
        if show_kwargs is None:
            show_kwargs = {}

        if c is None:
            c = self.vortex_strengths
            colorbar_label = "Vortex Strengths"

        if draw_streamlines:
            if (not hasattr(self, 'streamlines')) or recalculate_streamlines:
                self.calculate_streamlines()

        if backend == "plotly":
            from archibald2.visualization.plotly_Figure3D import Figure3D
            fig = Figure3D()

            for i in range(len(self.front_left_vertices)):
                fig.add_quad(
                    points=[
                        self.front_left_vertices[i, :],
                        self.back_left_vertices[i, :],
                        self.back_right_vertices[i, :],
                        self.front_right_vertices[i, :],
                    ],
                    intensity=c[i],
                    outline=True,
                )

            if draw_streamlines:
                for i in range(self.streamlines.shape[0]):
                    fig.add_streamline(self.streamlines[i, :, :].T)

            return fig.draw(
                show=show,
                colorbar_title=colorbar_label,
                **show_kwargs,
            )

        elif backend == "pyvista":
            import pyvista as pv
            plotter = pv.Plotter()
            plotter.title = "ASB VortexLatticeMethod"
            if plot_axes:
                plotter.add_axes()
                plotter.show_grid(color='gray')

            ### Draw the airplane mesh
            points = np.concatenate([
                self.front_left_vertices,
                self.back_left_vertices,
                self.back_right_vertices,
                self.front_right_vertices
            ])
            N = len(self.front_left_vertices)
            range_N = np.arange(N)
            faces = tall(range_N) + wide(np.array([0, 1, 2, 3]) * N)

            mesh = pv.PolyData(
                *mesh_utils.convert_mesh_to_polydata_format(points, faces)
            )
            scalar_bar_args = {}
            if colorbar_label is not None:
                scalar_bar_args["title"] = colorbar_label
            plotter.add_mesh(
                mesh=mesh,
                scalars=c,
                show_edges=True,
                show_scalar_bar=c is not None,
                scalar_bar_args=scalar_bar_args,
                cmap=cmap,
            )

            ### Draw the streamlines
            if draw_streamlines:
                import archibald2.tools.pretty_plots as p
                for i in range(self.streamlines.shape[0]):
                    plotter.add_mesh(
                        pv.Spline(self.streamlines[i, :, :].T),
                        color=p.adjust_lightness("#7700FF", 1.5),
                        # color=p.adjust_lightness("red", 1.5),
                        # color=p.adjust_lightness("#00d0c5", 1.5),
                        opacity=0.7,
                        line_width=1
                    )

            if show:
                plotter.show(**show_kwargs)
            return plotter

        else:
            raise ValueError("Bad value of `backend`!")
            
    def draw_flow(self,
                  z: float = None,
                  x_bounds: np.ndarray = (None, None),
                  y_bounds: np.ndarray = (None, None),
                  margin: float = 1.,
                  n_cells: int = 50,
                  draw_streamlines: bool = True,
                  color: str = 'black',
                  draw_contour: bool = True,
                  cmap: str = 'viridis',
                  speed_bounds: np.ndarray = (None, None),
                  ):
        
        import matplotlib.pyplot as plt
        
        geometry_center = np.mean(self.collocation_points, axis=0)
        geometry_dims = np.array([
            np.max(self.collocation_points[:,i]) - np.min(self.collocation_points[:,i]) for i in range(3)
            ])
        
        geometry_dims *= (1 + margin)
        
        geometry_box = np.hstack(
            (tall(geometry_center - geometry_dims/2),
             tall(geometry_center + geometry_dims/2)))
        
        geometry_cube = np.hstack(
            (tall(geometry_center - np.max(geometry_dims[:-1])/2),
             tall(geometry_center + np.max(geometry_dims[:-1])/2))
            )
        
        xmin, xmax = x_bounds
        ymin, ymax = y_bounds
        
        if xmin is None:
            xmin = geometry_cube[0, 0]
        if xmax is None:
            xmax = geometry_cube[0, 1]
        if ymin is None:
            ymin = geometry_cube[1, 0]
        if ymax is None:
            ymax = geometry_cube[1, 1]
            
        if z is None:
            z = geometry_center[2]
        
        nx = n_cells
        ny = int(nx * (ymax-ymin)/(xmax-xmin))
        
        x = np.linspace(xmin, xmax, nx)
        y = np.linspace(ymin, ymax, ny)
        
        X, Y = np.meshgrid(x, y)
        
        # Convert meshgrid to (n*m, 2) array
        gridXYZ = np.column_stack([X.ravel(), Y.ravel(), z*np.ones(nx*ny)])
        
        field = self.get_velocity_at_points(gridXYZ)
        
        # Define vector field
        U = field[:,0]
        V = field[:,1]
        
        U = U.reshape(X.shape)
        V = V.reshape(X.shape)
        
        W = np.linalg.norm(field, axis=1)
        
        W = W.reshape(X.shape)
        
        wmin, wmax = speed_bounds
        
        if wmin is None:
            wmin = self.reference_speed(z) * 0.75
        if wmax is None:
            wmax = self.reference_speed(z) * 1.25
            
        # Thresholding the speed magnitude
        W[W > wmax] = wmax
        W[W < wmin] = wmin
        
        lw = W**2/4e2
        
        # Create quiver plot
        plt.figure(figsize=(10, 10*(ymax-ymin)/(xmax-xmin)))
        if draw_streamlines:
            plt.streamplot(X, Y, U, V, color=color, linewidth=lw)
        if draw_contour:
            plt.contourf(X, Y, W, cmap=cmap, levels=10)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Flow at z='+str(round(z,3))+' m')
        # plt.grid()
        plt.colorbar(label='Speed (m/s)')
        plt.axis('equal')
        plt.show()


class AeroVortexLatticeMethod(VortexLatticeMethod):
    """
    An explicit (linear) vortex-lattice-method aerodynamics analysis.

    Usage example:
        >>> analysis = asb.VortexLatticeMethod(
        >>>     airplane=my_airplane,
        >>>     op_point=asb.OperatingPoint(
        >>>         velocity=100, # m/s
        >>>         alpha=5, # deg
        >>>         beta=4, # deg
        >>>         p=0.01, # rad/sec
        >>>         q=0.02, # rad/sec
        >>>         r=0.03, # rad/sec
        >>>     )
        >>> )
        >>> aero_data = analysis.run()
        >>> analysis.draw()
    """

    def __init__(self,
                 airplane: Rig,
                 op_point: OperatingPoint,
                 xyz_ref: List[float] = None,
                 IZsym: int = 0,
                 to_sym: List[bool] = None,
                 Zsym: float = 0.0,
                 verbose: bool = False,
                 spanwise_resolution: int = 10,
                 spanwise_spacing_function: Callable[[float, float, float], np.ndarray] = np.cosspace,
                 chordwise_resolution: int = 10,
                 chordwise_spacing_function: Callable[[float, float, float], np.ndarray] = np.cosspace,
                 vortex_core_radius: float = 1e-8,
                 align_trailing_vortices_with_wind: bool = True,
                 ):
        
        super().__init__(airplane,
                         op_point,
                         xyz_ref,
                         IZsym,
                         to_sym,
                         Zsym,
                         verbose,
                         spanwise_resolution,
                         spanwise_spacing_function,
                         chordwise_resolution,
                         chordwise_spacing_function,
                         vortex_core_radius,
                         align_trailing_vortices_with_wind,
                         )
        
        self.fluid = self.op_point.environment.air
        
        self.visc_corr = 0.
        
    def get_freestream_velocity_at_points(self,
                                          points: np.ndarray,
                                          ) -> np.ndarray:
        """
        Computes the freestream velocity at a set of points in the flowfield.

        Args:
            points: A Nx3 array of points that you would like to know the induced velocities at. Given in geometry axes.

        Returns: A Nx3 of the induced velocity at those points. Given in geometry axes.

        """
        
        # TRUE WIND SPEED
        twa = self.op_point.twa
        rot_mat = np.rotation_matrix_3D(twa*np.pi/180., np.array([0.,0.,1.]))
        
        true_wind_velocities = np.multiply(tall(self.op_point._tws(points[:,2])), wide(np.array([1., 0., 0.])))
        # true_wind_velocities = np.dot(true_wind_velocities, rot_mat)
        
        # true_wind_velocities = np.array([[self.op_point._tws(points[i,2]) * np.cosd(twa),
        #                                   -self.op_point._tws(points[i,2]) * np.sind(twa),
        #                                   0.] for i in range(points.shape[0])
        #     ])
        true_wind_velocities = true_wind_velocities @ rot_mat
        
        self.true_wind = true_wind_velocities
        
        # FAIR WIND SPEED
        ship_speed_velocities = np.ones(points.shape) * np.array([1., 0., 0.]) * self.op_point._stw
        
        # SUM
        freestream_velocities = true_wind_velocities + ship_speed_velocities
        
        return freestream_velocities
    
    def fluid_dynamic_pressure(self):
        return self.op_point.air_dynamic_pressure()
    
    def reference_speed(self,
                        z:float = None):
        return self.op_point._aws(z)
    

#IGNORE    
# class RotorVortexLatticeMethod(AeroVortexLatticeMethod):
#     """
#     An explicit (linear) vortex-lattice-method aerodynamics analysis to handle rotor sails

#     """
#     def __init__(self,
#                  airplane: Appendage,
#                  op_point: OperatingPoint,
#                  xyz_ref: List[float] = None,
#                  IZsym: int = 0,
#                  to_sym: List[bool] = None,
#                  Zsym: float = 0.0,
#                  verbose: bool = False,
#                  spanwise_resolution: int = 10,
#                  spanwise_spacing_function: Callable[[float, float, float], np.ndarray] = np.cosspace,
#                  chordwise_resolution: int = 10,
#                  chordwise_spacing_function: Callable[[float, float, float], np.ndarray] = np.cosspace,
#                  vortex_core_radius: float = 1e-8,
#                  align_trailing_vortices_with_wind: bool = True,
#                  R: float = 1.,
#                  SR: float = 1.
#                  ):
        
#         super().__init__(airplane,
#                          op_point,
#                          xyz_ref,
#                          IZsym,
#                          to_sym,
#                          Zsym,
#                          verbose,
#                          spanwise_resolution,
#                          spanwise_spacing_function,
#                          chordwise_resolution,
#                          chordwise_spacing_function,
#                          vortex_core_radius,
#                          align_trailing_vortices_with_wind,
#                          )


class HydroVortexLatticeMethod(VortexLatticeMethod):
    """
    An explicit (linear) vortex-lattice-method aerodynamics analysis.

    Usage example:
        >>> analysis = asb.VortexLatticeMethod(
        >>>     airplane=my_airplane,
        >>>     op_point=asb.OperatingPoint(
        >>>         velocity=100, # m/s
        >>>         alpha=5, # deg
        >>>         beta=4, # deg
        >>>         p=0.01, # rad/sec
        >>>         q=0.02, # rad/sec
        >>>         r=0.03, # rad/sec
        >>>     )
        >>> )
        >>> aero_data = analysis.run()
        >>> analysis.draw()
    """

    def __init__(self,
                 airplane: Appendage,
                 op_point: OperatingPoint,
                 xyz_ref: List[float] = None,
                 IZsym: int = 0,
                 to_sym: List[bool] = None,
                 Zsym: float = 0.0,
                 verbose: bool = False,
                 spanwise_resolution: int = 10,
                 spanwise_spacing_function: Callable[[float, float, float], np.ndarray] = np.cosspace,
                 chordwise_resolution: int = 10,
                 chordwise_spacing_function: Callable[[float, float, float], np.ndarray] = np.cosspace,
                 vortex_core_radius: float = 1e-8,
                 align_trailing_vortices_with_wind: bool = True,
                 ):
        
        super().__init__(airplane,
                         op_point,
                         xyz_ref,
                         IZsym,
                         to_sym,
                         Zsym,
                         verbose,
                         spanwise_resolution,
                         spanwise_spacing_function,
                         chordwise_resolution,
                         chordwise_spacing_function,
                         vortex_core_radius,
                         align_trailing_vortices_with_wind,
                         )
        
        self.fluid = self.op_point.environment.water
    
    def get_freestream_velocity_at_points(self,
                                          points: np.ndarray,
                                          ) -> np.ndarray:
        """
        Computes the freestream velocity at a set of points in the flowfield.

        Args:
            points: A Nx3 array of points that you would like to know the induced velocities at. Given in geometry axes.

        Returns: A Nx3 of the induced velocity at those points. Given in geometry axes.

        """
        
        freestream_velocities = np.ones(points.shape) * np.array([1., 0., 0.]) * self.op_point._stw
        
        return freestream_velocities
    
    def fluid_dynamic_pressure(self):
        return self.op_point.water_dynamic_pressure()
    
    def reference_speed(self,
                        z: float = None):
        return self.op_point._stw
    

if __name__ == '__main__':
    ### Import Vanilla Geometry
    
    pass

#     from archibald2.dynamics.aero_3D.test_aero_3D.geometries.vanilla import airplane as vanilla

#     ### Do the AVL run
#     vlm = HydroVortexLatticeMethod(
#         airplane=vanilla,
#         op_point=OperatingPoint(
#             stw=10.,
#             tws0=10.,
#             twa=45.,
#         ),
#         spanwise_resolution=10,
#         chordwise_resolution=12,
#     )

#     res = vlm.run()

# #     for k, v in res.items():
# #         print(f"{str(k).rjust(10)} : {v}")