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
import archibald2.geometry.mesh_utilities as mesh_utils

from archibald2.geometry.hull import Hull
from archibald2.geometry.lifting_set import Rig, Appendage
from archibald2.geometry.differentiable_mesh import DifferentiableMesh
from archibald2.geometry.propeller import Propeller, BSeriesPropeller

from archibald2.performance.operating_point import OperatingPoint

from archibald2.dynamics.aero_3D import HydroVortexLatticeMethod, AeroVortexLatticeMethod

import archibald2.numpy as np
import archibald2.tools.units as u

import trimesh

### Define some helper functions that take a vector and make it a Nx1 or 1xN, respectively.
# Useful for broadcasting with matrices later.
def tall(array):
    return np.reshape(array, (-1, 1))


def wide(array):
    return np.reshape(array, (1, -1))


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


def rotate_points(points, matrix, center):
    """
    Rotates a set of 3D points around a given center using a rotation matrix.

    Parameters:
    - points: (n, 3) array-like, 3D points to rotate.
    - matrix: (3, 3) array-like, the rotation matrix.
    - center: (3,) array-like, the center point of rotation.

    Returns:
    - Rotated points in the same format as the input (NumPy or CasADi).

    See: https://numpy.org/doc/stable/reference/generated/numpy.dot.html
    """
    return np.add(np.add(points, -wide(center)) @ matrix, wide(center))


class Sailboat():
    
    def __init__(self,
                 name: Optional[str] = None,
                 xyz_ref: Union[np.ndarray, List] = None,
                 displacement: Union[np.ndarray, List] = None,
                 cog: Union[np.ndarray, List] = None,
                 rig: Rig = None,
                 app: Appendage = None,
                 hull: Hull = None,
                 propeller: BSeriesPropeller = None,
                 mesh: Optional[List[trimesh.Trimesh]] = [],
                 ):
        
        if xyz_ref is None:
            xyz_ref = np.zeros(3)
        
        # TODO: create a MassProperties class and handle a list of mass properties
        
        if displacement is None:
            displacement = 0.
            
        if cog is None:
            cog = xyz_ref.copy()
            
        self.xyz_ref = np.array(xyz_ref)
        
        self.rig = rig
        self.rig_0 = copy.copy(rig)
        
        self.app = app
        self.app_0 = copy.copy(app)
        
        if rig is None and app is None:
            self.wings = []
            self.fuselages = []
        elif rig is None:
            self.wings = app.wings
            self.fuselages = app.fuselages
        elif app is None :
            self.wings = rig.wings
            self.fuselages = rig.fuselages
        else:
            self.wings = rig.wings + app.wings
            self.fuselages = rig.fuselages + app.fuselages
            
        self.propeller = propeller
        
        # TODO: better define what is hull and what is mesh (should mesh even exist? only for non watertight meshes?)
        
        self.displacement = displacement
        self.mesh = [DifferentiableMesh(m.vertices*np.array([[-1., 1., 1.]]), m.faces) for m in mesh]
        self.cog = cog
        
        self.mesh_0 = copy.copy(self.mesh)
        self.cog_0 = copy.copy(cog)
        
        self.hull = hull
        
        self._build_rotation_matrices()
        
        self.aeroVLMdata = None
        self.hydroVLMdata = None
        
        
    def mesh_body(self,
                  method="quad",
                  thin_wings=False,
                  stack_meshes=True,
                  ):
        """
        Returns a surface mesh of the Airplane, in (points, faces) format. For reference on this format,
        see the documentation in `aerosandbox.geometry.mesh_utilities`.

        Args:

            method:

            thin_wings: Controls whether wings should be meshed as thin surfaces, rather than full 3D bodies.

            stack_meshes: Controls whether the meshes should be merged into a single mesh or not.

                * If True, returns a (points, faces) tuple in standard mesh format.

                * If False, returns a list of (points, faces) tuples in standard mesh format.

        Returns:

        """
        if thin_wings:
            wing_meshes = [
                wing.mesh_thin_surface(
                    method=method,
                )
                for wing in self.wings
            ]
        else:
            wing_meshes = [
                wing.mesh_body(
                    method=method,
                )
                for wing in self.wings
            ]

        fuse_meshes = [
            fuse.mesh_body(
                method=method
            )
            for fuse in self.fuselages
        ]

        meshes = wing_meshes + fuse_meshes

        if stack_meshes:
            points, faces = mesh_utils.stack_meshes(*meshes)
            return points, faces
        else:
            return meshes
        
    def reset(self,
              update_geometry: bool = True):
        """
        Cancel all transformations on sailboat geometry

        """
        for sail in self.rig.wings:
            sail.reset()
            
        for fin in self.app.wings:
            fin.reset()
            
        if self.propeller:
            self.propeller.reset()
            
        self.mesh = self.mesh_0.copy()
        
        if update_geometry:
            for sail in self.rig.wings:
                sail.build_xsecs()
                
            for fin in self.app.wings:
                fin.build_xsecs()
    
    def _build_rotation_matrices(self,
                                heel: float = 0.0,
                                trim: float = 0.0,
                                leeway: float = 0.0,
                                ):
        
        self.heel = heel
        self.trim = trim
        self.leeway = leeway
        
        self._heelRot = np.rotation_matrix_3D(np.deg2rad(heel),
                                              np.array([1.,0.,0.]))
        
        self._trimRot = np.rotation_matrix_3D(np.deg2rad(trim),
                                              np.array([0., np.cosd(heel), np.sind(heel)]))
        
        self._leewayRot = np.rotation_matrix_3D(np.deg2rad(leeway),
                                                np.array([0.,0.,1.]))
        
    def translate_geometry(self,
                           dx: float = 0.0,
                           dy: float = 0.0,
                           dz: float = 0.0,
                           ):
        
        tester = wide(dx) + wide(dy) + wide(dz) + wide(self.cog)
        
        if np.is_casadi_type(tester):
            self.cog = tall(np.add(tall(self.cog), tall(np.array([-dx, dy, dz]))))
        else:
            self.cog = np.add(self.cog, np.array([-dx, dy, dz]))
        
        for sail in self.rig.wings:
            sail.translate(np.array([dx, dy, dz]))
            
        for fin in self.app.wings:
            fin.translate(np.array([dx, dy, dz]))
            
        for m in self.mesh:
            m.vertices = np.add(m.vertices, wide(np.array([-dx, dy, dz])))
            
        if self.propeller:
            self.propeller.translate(np.array([dx, dy, dz]))
            
        self.hull.diff_mesh.vertices = np.add(self.hull.diff_mesh.vertices, wide(np.array([-dx, dy, dz])))
            
        
    def rotate_geometry(self,
                        angle: float,
                        axis: Union[np.ndarray, List[float], str],
                        center: Union[np.ndarray, List[float]],
                        matrix: np.ndarray = None,
                        offset_canting: bool = False,
                        offset_raking: bool = False,
                        offset_deflection: bool = False,
                        update_geometry: bool = False):
        """
        Rotate the leading edge from the given angle, axis and center point.

        """
        if matrix is None:
            matrix = np.rotation_matrix_3D((angle)*np.pi/180., axis)
            
        self.cog = np.sum(wide(self.cog) @ matrix, axis=0)
        
        for sail in self.rig.wings:
            sail.global_rotation(angle,
                                 axis,
                                 center,
                                 matrix,
                                 offset_canting,
                                 offset_raking,
                                 offset_deflection,
                                 update_geometry)
        
        for fin in self.app.wings:
            fin.global_rotation(angle,
                                axis,
                                center,
                                matrix,
                                offset_canting,
                                offset_raking,
                                offset_deflection,
                                update_geometry)
        
        for m in self.mesh:
            m.vertices = rotate_points(m.vertices, matrix, center)
            
        if self.propeller:
            self.propeller.global_rotation(angle,
                                           axis,
                                           center,
                                           matrix)
            
        self.hull.diff_mesh.vertices = rotate_points(self.hull.diff_mesh.vertices, matrix, center)
            
    def transform(self,
                  heel: float = 0.0,
                  trim: float = 0.0,
                  leeway: float = 0.0,
                  dx: float = 0.0,
                  dy: float = 0.0,
                  dz: float = 0.0,
                  reset_geometry: bool = True
                  ):
        
        if reset_geometry:
            self.reset(update_geometry=False)

        # HEEL TRANSFORMATION
        self.rotate_geometry(heel, 'x', self.xyz_ref,
                             update_geometry=False)
        
        # TRIM TRANSFORMATION
        self.rotate_geometry(trim, 'y', self.xyz_ref,
                             update_geometry=False)
        
        # LEEWAY TRANSFORMATION
        self.rotate_geometry(leeway, 'z', self.xyz_ref,
                             offset_deflection=True,
                             update_geometry=False)
        
        self.translate_geometry(dx=dx,
                                dy=dy,
                                dz=dz)
        
        for sail in self.rig.wings:
            sail.build_xsecs()
            
        for fin in self.app.wings:
            fin.build_xsecs()
            
    def compute_weight(self,
                       op_point: OperatingPoint = OperatingPoint()
                       ):
        
        # TODO: bring back cog in underway axes AND transform it with the rest
        
        weight = self.displacement
        g = op_point.environment.gravity
        
        center = self.cog
        center[1] *= -1 # TODO: find why lateral inversion is necessary. Heel rotation seems not to be consistent
        
        Fw = np.array([0.,
                       0.,
                       - weight*g])
        
        Mw = np.cross(center, Fw)
        
        return Fw, Mw
    
    def compute_appendages(self,
                           op_point: OperatingPoint = OperatingPoint(),
                           nSpanwise: int = 1,
                           nChordwise: int = 3,
                           run_symmetric: bool = False,
                           to_symmetrize: Union[np.ndarray, List[bool], List[int]] = None,
                           Zsym: float = 0.0,
                           full_output: bool = False,
                           ):
        
        if to_symmetrize is None:
            if run_symmetric:
                to_symmetrize=np.ones(len(self.app.wings))
            else:
                to_symmetrize=np.zeros(len(self.app.wings))
        
        vlmApp = HydroVortexLatticeMethod(
            airplane=self.app,
            op_point=op_point,
            spanwise_resolution=nSpanwise,
            chordwise_resolution=nChordwise,
            chordwise_spacing_function=np.cosspace,
            align_trailing_vortices_with_wind=True,
            IZsym=run_symmetric,
            to_sym=to_symmetrize,
            Zsym=Zsym,
        )

        hydro = vlmApp.run()
        self.hydroVLMdata = vlmApp

        Fapp = hydro['F_ab']
        Mapp = hydro['M_ab']
        
        if full_output:
            return Fapp, Mapp, hydro
        return Fapp, Mapp
    
    
    def compute_sails(self,
                      op_point: OperatingPoint = OperatingPoint(),
                      nSpanwise: int = 1,
                      nChordwise: int = 3,
                      run_symmetric: bool = False,
                      to_symmetrize: Union[np.ndarray, List[bool], List[int]] = None,
                      Zsym: float = 0.0,
                      full_output: bool = False,
                      ):
        
        if to_symmetrize is None:
            if run_symmetric:
                to_symmetrize=np.ones(len(self.rig.wings))
            else:
                to_symmetrize=np.zeros(len(self.rig.wings))
        
        vlmRig = AeroVortexLatticeMethod(
            airplane=self.rig,
            op_point=op_point,
            spanwise_resolution=nSpanwise,
            chordwise_resolution=nChordwise,
            chordwise_spacing_function=np.cosspace,
            align_trailing_vortices_with_wind=True,
            IZsym=run_symmetric,
            to_sym=to_symmetrize,
            Zsym=Zsym,
        )

        aero = vlmRig.run()
        self.aeroVLMdata = vlmRig

        Fsail = aero['F_ab']
        Msail = aero['M_ab']
        
        if full_output:
            return Fsail, Msail, aero
        return Fsail, Msail
    
    def compute_propeller(self,
                          rpm: float,
                          op_point: OperatingPoint = OperatingPoint(),
                          recompute_statics: bool = True,
                          full_output: bool = False,
                          shaft_efficiency: float = 0.98,
                          ):

        if self.propeller:
            if self.hull:
                if recompute_statics:
                    self.hull.compute_statics(op_point)
                w, t, etaR = self.hull.compute_propulsion_coefficients(op_point.stw, self.propeller.blade_area_ratio)
            else:
                w, t, etaR = 0., 0., 1.
            
            etaS = shaft_efficiency
            etaH = (1-t)/(1-w)
            
            leeway = op_point.leeway
            
            n = rpm / 60. # rps
            V = op_point._stw # m/s
            D = self.propeller.diameter # m
            J = (1 - w) * V / (n*D)
            
            Kt, Kq, eta0 = self.propeller.compute_performance(J)
            
            T, Q, P, n = self.propeller.compute_forces(
                J,
                op_point.environment.water.density,
                rpm=rpm
            )
            
            F = np.sum(T*(1-t)*etaR)
            
            center = self.propeller.center
            axis = tall(self.propeller.axis)
            
            prop = {
                'J': J,
                'Kt': Kt,
                'Kq': Kq,
                'eta0': eta0,
                'etaH': etaH,
                'etaS': etaS,
                'etaR': etaR,
                'T': T,
                'Q': Q,
                'P': P,
                'n': n,
                'w': w,
                't': t,
                'F_b': F * np.array([1.,0.,0.]),
                'M_b': np.zeros(3),
                'F_ab': F * axis,
                'M_ab': np.cross(center, F * axis)
            }
            
            Fprop = prop['F_ab']
            Mprop = prop['M_ab']
            
        else:
            Fprop, Mprop, prop = np.zeros(3), np.zeros(3), {}
        
        if full_output:
            return Fprop, Mprop, prop
        return Fprop, Mprop
        
    
    def draw(self,
             backend: str = "pyvista",
             thin_wings: bool = False,
             ax=None,
             use_preset_view_angle: str = None,
             set_background_pane_color: Union[str, Tuple[float, float, float]] = None,
             set_background_pane_alpha: float = None,
             set_lims: bool = True,
             set_equal: bool = True,
             set_axis_visibility: bool = None,
             show: bool = True,
             show_kwargs: Dict = None,
             ):
        """
        Produces an interactive 3D visualization of the airplane.

        Args:

            backend: The visualization backend to use. Options are:

                * "matplotlib" for a Matplotlib backend
                * "pyvista" for a PyVista backend
                * "plotly" for a Plot.ly backend
                * "trimesh" for a trimesh backend

            thin_wings: A boolean that determines whether to draw the full airplane (i.e. thickened, 3D bodies), or to use a
            thin-surface representation for any Wing objects.

            show: A boolean that determines whether to display the object after plotting it. If False, the object is
            returned but not displayed. If True, the object is displayed and returned.

        Returns: The plotted object, in its associated backend format. Also displays the object if `show` is True.

        """
        
        import pyvista as pv

        plotter = pv.Plotter()
        
        if show_kwargs is None:
            show_kwargs = {}

        points, faces = self.mesh_body(method="quad", thin_wings=thin_wings)

        if backend == "matplotlib":
            import matplotlib.pyplot as plt
            import aerosandbox.tools.pretty_plots as p
            from mpl_toolkits.mplot3d.art3d import Poly3DCollection

            if ax is None:
                _, ax = p.figure3d(figsize=(8, 8), computed_zorder=False)

            else:
                if not p.ax_is_3d(ax):
                    raise ValueError("`ax` must be a 3D axis.")

            plt.sca(ax)

            ### Set the view angle
            if use_preset_view_angle is not None:
                p.set_preset_3d_view_angle(use_preset_view_angle)

            ### Set the background pane color
            if set_background_pane_color is not None:
                ax.xaxis.pane.set_facecolor(set_background_pane_color)
                ax.yaxis.pane.set_facecolor(set_background_pane_color)
                ax.zaxis.pane.set_facecolor(set_background_pane_color)

            ### Set the background pane alpha
            if set_background_pane_alpha is not None:
                ax.xaxis.pane.set_alpha(set_background_pane_alpha)
                ax.yaxis.pane.set_alpha(set_background_pane_alpha)
                ax.zaxis.pane.set_alpha(set_background_pane_alpha)

            ax.add_collection(
                Poly3DCollection(
                    points[faces], facecolors='lightgray', edgecolors=(0, 0, 0, 0.1),
                    linewidths=0.5, alpha=0.8, shade=True,
                ),
            )
            
            for h in self.mesh:
                ax.plot_trisurf(h.vertices[:,0],
                                h.vertices[:,1],
                                h.vertices[:,2],
                                triangles=h.faces,
                                shade=True
                                )
                
                ax.add_collection(
                    Poly3DCollection(
                        h.vertices[h.faces], facecolors='lightgray', edgecolors=(0, 0, 0, 0.1),
                        linewidths=0.5, alpha=0.8, shade=True,
                    ),
                )
                

            for prop in self.propulsors:

                ### Disk
                if prop.length == 0:
                    ax.add_collection(
                        Poly3DCollection(
                            np.stack([np.stack(
                                prop.get_disk_3D_coordinates(),
                                axis=1
                            )], axis=0),
                            facecolors='darkgray', edgecolors=(0, 0, 0, 0.2),
                            linewidths=0.5, alpha=0.35, shade=True, zorder=4,
                        )
                    )

            if set_lims:
                ax.set_xlim(points[:, 0].min(), points[:, 0].max())
                ax.set_ylim(points[:, 1].min(), points[:, 1].max())
                ax.set_zlim(points[:, 2].min(), points[:, 2].max())

            if set_equal:
                p.equal()

            if set_axis_visibility is not None:
                if set_axis_visibility:
                    ax.set_axis_on()
                else:
                    ax.set_axis_off()

            if show:
                p.show_plot()

        elif backend == "plotly":

            from aerosandbox.visualization.plotly_Figure3D import Figure3D
            fig = Figure3D()
            for f in faces:
                fig.add_quad((
                    points[f[0]],
                    points[f[1]],
                    points[f[2]],
                    points[f[3]],
                ), outline=True)
                show_kwargs = {
                    "show": show,
                    **show_kwargs
                }
            return fig.draw(**show_kwargs)

        elif backend == "pyvista":

            import pyvista as pv
            
            figW = pv.PolyData(
                *mesh_utils.convert_mesh_to_polydata_format(points, faces)
            )
            
            figsH = []
            
            for h in self.mesh:
                
                figsH.append(pv.PolyData(
                            *mesh_utils.convert_mesh_to_polydata_format(h.vertices,
                                                                        h.faces)
                            ))
                
            figsH.append(pv.PolyData(
                        *mesh_utils.convert_mesh_to_polydata_format(self.hull.diff_mesh.vertices,
                                                                    self.hull.diff_mesh.faces)))
                
                # figsH[-1].decimate(0.1)
                # figsH[-1] = figsH[-1].clean()
            
            fig = figW.merge(figsH)
            # fig = figsH[0]
            # fig = figW
            
            show_kwargs = {
                "show_edges": True,
                "show_grid" : True,
                **show_kwargs,
            }
            if show:
                fig.plot(**show_kwargs)
            return fig

        elif backend == "trimesh":

            import trimesh as tri
            wings = tri.Trimesh(points, faces)
            fig = tri.Trimesh(vertices=np.vstack([wings.vertices]+[h.vertices for h in self.mesh]),
                      faces=np.vstack([wings.faces]+[h.faces for h in self.mesh]))
            if show:
                fig.show(**show_kwargs)
            return fig
        else:
            raise ValueError("Bad value of `backend`!")
            
            
    def draw_three_view(self,
                        style: str = "shaded",
                        show: bool = True,
                        ):
        """
        Draws a standard 4-panel three-view diagram of the airplane using Matplotlib backend. Creates a new figure.

        Args:

            style: Determines what drawing style to use for the three-view. A string, one of:

                * "shaded"
                * "wireframe"

            show: A boolean of whether to show the figure after creating it, or to hold it so   that the user can modify the figure further before showing.

        Returns:

        """
        import matplotlib.pyplot as plt
        import aerosandbox.tools.pretty_plots as p

        preset_view_angles = np.array([
            ["-XY", "-YZ"],
            ["XZ", "left_isometric"]
        ], dtype="O")

        fig, axs = p.figure3d(
            nrows=preset_view_angles.shape[0],
            ncols=preset_view_angles.shape[1],
            figsize=(8, 8),
            computed_zorder=False,
        )

        for i in range(axs.shape[0]):
            for j in range(axs.shape[1]):
                ax = axs[i, j]
                preset_view = preset_view_angles[i, j]

                if style == "shaded":
                    self.draw(
                        backend="matplotlib",
                        ax=ax,
                        set_axis_visibility=False if 'isometric' in preset_view else None,
                        show=False
                    )
                elif style == "wireframe":
                    if preset_view == "XZ":
                        fuselage_longeron_theta = [np.pi / 2, 3 * np.pi / 2]
                    elif preset_view == "XY":
                        fuselage_longeron_theta = [0, np.pi]
                    else:
                        fuselage_longeron_theta = None

                    self.draw_wireframe(
                        ax=ax,
                        set_axis_visibility=False if 'isometric' in preset_view else None,
                        fuselage_longeron_theta=fuselage_longeron_theta,
                        show=False
                    )

                p.set_preset_3d_view_angle(preset_view)

                xres = np.diff(ax.get_xticks())[0]
                yres = np.diff(ax.get_yticks())[0]
                zres = np.diff(ax.get_zticks())[0]

                p.set_ticks(
                    xres, xres / 4,
                    yres, yres / 4,
                    zres, zres / 4,
                )

                ax.xaxis.set_tick_params(color='white', which='minor')
                ax.yaxis.set_tick_params(color='white', which='minor')
                ax.zaxis.set_tick_params(color='white', which='minor')

                if preset_view == 'XY' or preset_view == '-XY':
                    ax.set_zticks([])
                if preset_view == 'XZ' or preset_view == '-XZ':
                    ax.set_yticks([])
                if preset_view == 'YZ' or preset_view == '-YZ':
                    ax.set_xticks([])

        axs[1, 0].set_xlabel("$x_g$ [m]")
        axs[1, 0].set_ylabel("$y_g$ [m]")
        axs[0, 0].set_zlabel("$z_g$ [m]")
        axs[0, 0].set_xticklabels([])
        axs[0, 1].set_yticklabels([])
        axs[0, 1].set_zticklabels([])

        plt.subplots_adjust(
            left=-0.08,
            right=1.08,
            bottom=-0.08,
            top=1.08,
            wspace=-0.38,
            hspace=-0.38,
        )

        if show:
            p.show_plot(
                tight_layout=False,
            )
            
    def draw_vlm(self,
                 z_water: float = 0.,
                 c: np.ndarray = None,
                 streamlines_c: str = "slateblue",
                 mesh_color: str = 'lightgrey',
                 cmap: str = None,
                 colorbar_label: str = None,
                 show: bool = True,
                 show_edges: bool = True,
                 show_kwargs: Dict = None,
                 draw_streamlines=True,
                 draw_colorbar=True,
                 recalculate_streamlines=False,
                 draw_water=True,
                 plot_axes=True,
                 backend: str = "pyvista"
                 ):
        """
        Draws the solution. Note: Must be called on a SOLVED AeroProblem object.
        To solve an AeroProblem, use opti.solve(). To substitute a solved solution, use ap = sol(ap).
        :return:
        """
        import pyvista as pv
        
        rigVlm = self.aeroVLMdata
        appVlm = self.hydroVLMdata
        
        plotter = pv.Plotter()
        
        if show_kwargs is None:
            show_kwargs = {}
            
        if c is None:
            c = (rigVlm.vortex_strengths,
                 appVlm.vortex_strengths)
            
            # colorbar_label = "Vortex Strengths (m/s)"
            colorbar_label = "pressure coeffcient"
            # colorbar_label = "Pressure (Pa)"
            
            # P = (rigVlm.dynamic_pressure,
            #      appVlm.dynamic_pressure)
            
            Cp = (rigVlm.pressure_coef.copy(),
                  appVlm.pressure_coef.copy(),)
            
            c = Cp
            
            
            for elt in c:
                th = np.quantile(rigVlm.pressure_coef, .95)
                elt[elt > th] = th
            
            
        length = np.max([rigVlm.airplane.c_ref, appVlm.airplane.c_ref]) * 5.
        n = 10

        if draw_streamlines:
            if (not hasattr(rigVlm, 'streamlines')) or recalculate_streamlines:
                left_TE_vertices = rigVlm.back_left_vertices[rigVlm.is_trailing_edge]
                right_TE_vertices = rigVlm.back_right_vertices[rigVlm.is_trailing_edge]
                
                seed_points = (left_TE_vertices + right_TE_vertices) / 2.
                
                rigVlm.calculate_streamlines(seed_points=seed_points, length=length)
                
            if (not hasattr(appVlm, 'streamlines')) or recalculate_streamlines:

                left_TE_vertices = appVlm.back_left_vertices[appVlm.is_trailing_edge]
                right_TE_vertices = appVlm.back_right_vertices[appVlm.is_trailing_edge]
                
                seed_points = (left_TE_vertices + right_TE_vertices) / 2.
                
                seed_points = seed_points[seed_points[:,2] < z_water] # delete points above water surface
                
                seed_points = seed_points[::3,:]
                
                appVlm.calculate_streamlines(seed_points=seed_points, length=length)


        if backend == "plotly":
            from aerosandbox.visualization.plotly_Figure3D import Figure3D
            fig = Figure3D()

            for j, vlm in enumerate([rigVlm, appVlm]):
                for i in range(len(vlm.front_left_vertices)):
                    fig.add_quad(
                        points=[
                            vlm.front_left_vertices[i, :],
                            vlm.back_left_vertices[i, :],
                            vlm.back_right_vertices[i, :],
                            vlm.front_right_vertices[i, :],
                        ],
                        intensity=c[j],
                        outline=True,
                    )
    
                if draw_streamlines:
                    for k in range(vlm.streamlines.shape[0]):
                        fig.add_streamline(vlm.streamlines[k, :, :].T)

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
            
            for j, vlm in enumerate([rigVlm, appVlm]):
    
                ### Draw the airplane mesh
                points = np.concatenate([
                    vlm.front_left_vertices,
                    vlm.back_left_vertices,
                    vlm.back_right_vertices,
                    vlm.front_right_vertices
                ])
                N = len(vlm.front_left_vertices)
                range_N = np.arange(N)
                faces = tall(range_N) + wide(np.array([0, 1, 2, 3]) * N)
    
                mesh = pv.PolyData(
                    *mesh_utils.convert_mesh_to_polydata_format(points, faces)
                )
                
                scalar_bar_args = {}
                if colorbar_label is not None:
                    scalar_bar_args["title"] = f'{j*"Hydro"}{(1-j)*"Aero"} {colorbar_label}'
                plotter.add_mesh(
                    mesh=mesh,
                    scalars=c[j],
                    show_edges=show_edges,
                    show_scalar_bar=c[j] is not None,
                    scalar_bar_args=scalar_bar_args,
                    cmap=cmap,
                )
    
                ### Draw the streamlines
                if draw_streamlines:
                    import aerosandbox.tools.pretty_plots as p
                    for i in range(vlm.streamlines.shape[0]):
                        plotter.add_mesh(
                            pv.Spline(vlm.streamlines[i, :, :].T),
                            # color=p.adjust_lightness("#7700FF", 1.5),
                            color=p.adjust_lightness(streamlines_c, 1.0),
                            # color=p.adjust_lightness("red", 1.5),
                            # color=p.adjust_lightness("#00d0c5", 1.5),
                            opacity=0.7,
                            line_width=1
                        )
            
            figsH = []
            
            for h in self.mesh:
                
                figsH.append(pv.PolyData(
                            *mesh_utils.convert_mesh_to_polydata_format(h.vertices,
                                                                        h.faces)
                            ))
                
            figsH.append(pv.PolyData(
                        *mesh_utils.convert_mesh_to_polydata_format(self.hull.diff_mesh.vertices,
                                                                    self.hull.diff_mesh.faces)))
            
            for h in self.mesh:
                plotter.add_mesh(pv.PolyData(
                    *mesh_utils.convert_mesh_to_polydata_format(h.vertices,
                                                                h.faces)),
                    color=mesh_color)
            
            plotter.add_mesh(pv.PolyData(
                *mesh_utils.convert_mesh_to_polydata_format(self.hull.diff_mesh.vertices,
                                                            self.hull.diff_mesh.faces)),
                color=mesh_color)
                
            # for f in self.fuselages:
            #     plotter.add_mesh(f)
                
                
            # WATER SURFACE
            if draw_water:
                size = np.diff(self.hull.diff_mesh.bounds, axis=0)[0]
                center = np.mean(self.hull.diff_mesh.bounds, axis=0)
                offset = 1/2
                
                plotter.add_mesh(pv.Cube(center=(center[0], center[1], z_water), 
                                         x_length=size[0] + size[0]*offset, y_length=size[1] + size[0]*offset, z_length=1e-3),
                                 color='slateblue',
                                 opacity=0.2)

            if show:
                plotter.show(**show_kwargs)
            return plotter

        else:
            raise ValueError("Bad value of `backend`!")