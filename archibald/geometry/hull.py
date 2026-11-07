# -*- coding: utf-8 -*-
"""
Created on Wed May 27 16:23:04 2026

@author: jrich
"""

import archibald.numpy as np
import archibald.toolbox.units as u

import archibald.dynamics.hydro.dsyhs as dsyhs
import archibald.dynamics.hydro.holtrop as holtrop
import archibald.dynamics.hydro.michell as michell
import archibald.dynamics.hydro.jones as jones
import archibald.dynamics.hydro.inoue as inoue
import archibald.dynamics.hydro.kijima as kijima
import archibald.dynamics.hydro.savitsky as savitsky
import archibald.dynamics.hydro.mittendorf as mittendorf

import archibald.dynamics.aero.orc as orc
import archibald.dynamics.aero.day as day
import archibald.dynamics.aero.blendermann as blendermann

from typing import Union, List
from archibald.common import ArchibaldObject
from archibald.environment import Environment
from archibald.performance import OperatingPoint
from archibald.geometry.mesh import ArchibaldMesh


def amidships(
        fpp,
        app,
        cow,
        **kwargs,
    ):
    """
    Point amidships, between the ends of the waterline ``fpp`` and ``app``, at the
    side and height of the wetted surface centroid ``cow``, in the hull frame [m].
    """
    return np.reshape(np.array([
        (fpp[0, 0] + app[0, 0]) / 2.,
        cow[0, 1],
        cow[0, 2],
    ]), (1, -1))


# Hydrodynamic methods of the hull, selected by ``Hull.compute_resistance``.
#
# Each method is a dict of components, and the name of a component sets its
# direction in the boat motion frame (x along the velocity, z up):
#   "R..." : resistance [N], along -x, applied at the wetted surface centroid,
#            except "Ri", the resistance induced by the lift, applied with it.
#   "L"    : lift normal to the flow [N], along +y, applied amidships at the
#            height of the wetted surface centroid. +leeway gives +L, like a fin.
#   "Fz"   : vertical force [N], along +z, applied at the wetted surface centroid
#            unless the method gives its own point.
#   "N"    : yaw moment [N.m], about z, added as a couple.
# The points are in the hull frame, then moved with the operating point, so under
# heel, trim or leeway their x position changes the moment of a force along x.
# A component can be given as ``(function, point_function)``: the point function
# takes the same arguments and returns the application point in the hull frame,
# of shape (1, 3), e.g. ``amidships``. Methods can be combined by passing a list of them, as long as
# no component is given twice. Sources are in each module.
DEFAULT_RESISTANCE_METHODS = {
    "dsyhs": {
        "Rf": dsyhs.compute_Rf_dsyhs,
        "Rw": dsyhs.compute_Rw_dsyhs,
        "Rtr": dsyhs.compute_Rtr_dsyhs,
    },
    "holtrop": {
        "Rf": holtrop.compute_Rf_holtrop,
        "Rw": holtrop.compute_Rw_holtrop,
        "Rb": holtrop.compute_Rb_holtrop,
        "Rtr": holtrop.compute_Rtr_holtrop,
        "Ra": holtrop.compute_Ra_holtrop,
    },
    # Slender hulls: Michell thin-ship wave resistance on an equivalent hull
    # built from the hydrostatics, with the Holtrop friction (needs the same
    # extra kwargs as "holtrop", e.g. Csternchoice).
    "michell": {
        "Rf": holtrop.compute_Rf_holtrop,
        "Rw": michell.compute_Rw_michell,
    },
    # Planing hulls: Savitsky (1964) and Savitsky & Brown (1976), dynamic part
    # only, the hydrostatics carry the rest. Needs trim (keel, bow up [deg]) and
    # deadrise [deg]. The lift acts at the dynamic centre of pressure. Savitsky's
    # pitching moment is not used: the lift point already gives it.
    "savitsky": {
        "Rf": savitsky.compute_Rf_savitsky,
        "Rw": savitsky.compute_Rp_savitsky,
        "Rsp": savitsky.compute_Rsp_savitsky,
        "Rtr": savitsky.compute_Rtr_savitsky,
        "Fz": (savitsky.compute_Fz_savitsky, savitsky.point_Fz_savitsky),
    },
    # Added resistance in waves, to combine with a calm water method.
    # Planing hulls in head seas: Savitsky & Brown (1976), needs swh.
    "savitsky_brown": {
        "Raw": savitsky.compute_Raw_savitsky,
    },
    # Ships in waves of any heading: Mittendorf et al. (2022) on a wave
    # spectrum, needs swh, tp, Le and Lr.
    "mittendorf": {
        "Raw": mittendorf.compute_Raw_mittendorf,
    },
    # Lateral force at leeway, to combine with a calm water method. "Ri" is
    # the resistance added by the leeway.
    # Low aspect ratio wing theory: Jones (1946).
    "jones": {
        "L": jones.compute_L_jones,
        "Ri": jones.compute_Ri_jones,
    },
    # Merchant ships: Inoue, Hirano and Kijima (1981), linear.
    "inoue": {
        "L": inoue.compute_L_inoue,
        "Ri": inoue.compute_Ri_inoue,
    },
    # Merchant ships: Kijima and Nakiri (2003), nonlinear, with the yaw moment.
    "kijima": {
        "L": kijima.compute_L_kijima,
        "Ri": kijima.compute_Ri_kijima,
        "N": kijima.compute_N_kijima,
    },
}


# Windage methods. Each of them returns (Fx, Fy, zce) or (Fx, Fy, zce, dxce),
# see archibald.dynamics.aero.common. The element named "Crew" is not related to
# the hull: ``Sailboat.compute_windage`` evaluates it once for the whole boat.
DEFAULT_WINDAGE_METHODS = {
    "orc": {
        "Hull": orc.compute_Fhull_orc,
        "Crew": orc.compute_Fcrew_orc,
    },
    # Dinghies: Day (2017), anthropometric crew drag.
    "day": {
        "Hull": day.compute_Fhull_day,
        "Crew": day.compute_Fcrew_day,
    },
    # Ships and large powerboats: Blendermann wind load functions (needs a ship_type or its parameters).
    "blendermann": {
        "Hull": blendermann.compute_Fhull_blendermann,
    },
}


def tall(array):
    return np.reshape(array, (-1, 1))


def wide(array):
    return np.reshape(array, (1, -1))


def get_air_density(op_point, default=1.225):
    """
    Air density [kg/m^3] of the operating point environment, if defined, else default.
    """
    air = getattr(op_point.environment, 'air', None)
    return getattr(air, 'density', default)


class Hull(ArchibaldObject):
    def __init__(
            self,
            name: str = 'hull',
            mesh: Union[ArchibaldMesh, str] = None,
        ):
        if type(mesh) == str:
            from archibald.toolbox.mesh_utils import load_stl
            mesh_path = mesh
            mesh = ArchibaldMesh(*load_stl(mesh_path))
        
        self.name = name
        self.mesh = mesh
        
        self.hydrostatics_data = {}
        self.aerostatics_data = {}
        self.resistance_components = {}
        self.resistance_points = {}
        self.resistance_torsors = {}
        self.windage_components = {}

    def __repr__(self):
        return f"Hull object '{self.name}'"
    
    def draw(
            self, 
            backend: str = "pyvista",
            **kwargs
        ):
        """
        Standardized Hull draw method to behave like a component of the Sailboat.
        """
        if self.mesh:
            # Pass all kwargs (like draw_plane, opacity, etc.) directly to the mesh
            return self.mesh.draw(backend=backend, **kwargs)
            
    def get_waterplane(
            self,
            op_point: OperatingPoint,
        ):
        """
        Point and normal of the water plane, expressed in the hull local frame.
        """
        
        xh = np.mean(self.mesh.vertices[:, 0])
        yh = np.mean(self.mesh.vertices[:, 1])
    
        # Bring the global water plane into the boat's local frame
        point = op_point.apply_transformations(
            # geometry=np.array([[0., 0., 0.]]),
            geometry=np.array([[xh, yh, 0.]]),
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
        
        return point, normal
    
    def compute_hydrostatics_properties(
            self,
            op_point: OperatingPoint,
        ):
        
        point, normal = self.get_waterplane(op_point)
        
        self.hydrostatics_data = self.mesh.hydrostatics(point, normal)
        
    def compute_aerostatics_properties(
            self,
            op_point: OperatingPoint,
        ):
        """
        Geometric properties of the hull above the water plane, see ``ArchibaldMesh.aerostatics``.
        """
        
        point, normal = self.get_waterplane(op_point)
        
        self.aerostatics_data = self.mesh.aerostatics(point, normal)
        
    def get_position(
            self,
            op_point: OperatingPoint,
        ):
        """
        Position (x, y) [m] of the hull centre of buoyancy in the water axes,
        i.e. with the heel, trim and leeway of the operating point applied.
        Used to compute the interaction between the hulls of a multihull.
        Hydrostatics must have been computed beforehand.
        """
        cob = op_point.apply_transformations(self.hydrostatics_data['cob'])
        cob = np.reshape(cob, (-1,))
        return cob[0], cob[1]
        
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
    
    
    @staticmethod
    def resolve_resistance_methods(
            method: Union[str, dict, List[Union[str, dict]]],
        ):
        """
        Merge the methods asked for into one dict of components.

        Parameters
        ----------
        method : str, a key of ``DEFAULT_RESISTANCE_METHODS``, dict of components,
            or list of them, e.g. ``["dsyhs", "kijima"]``.

        Returns
        -------
        dict, name of the component: (function, point_function or None).
        """
        methods = method if isinstance(method, (list, tuple)) else [method]
        if len(methods) == 0:
            raise ValueError("No resistance method given.")

        process = {}
        for m in methods:
            if isinstance(m, str):
                if m not in DEFAULT_RESISTANCE_METHODS:
                    raise NotImplementedError(
                        f"Unknown method '{m}'. Should be in {list(DEFAULT_RESISTANCE_METHODS.keys())} or a custom dictionary of methods.")
                m = DEFAULT_RESISTANCE_METHODS[m]
            elif not isinstance(m, dict):
                raise TypeError("method argument should be str, dict, or a list of them")

            for name, entry in m.items():
                if name in process:
                    raise ValueError(f"Component '{name}' is given by more than one method.")
                if not (name.startswith("R") or name in ("L", "Fz", "N")):
                    raise ValueError(
                        f"Unknown component '{name}'. Names set the direction of the force: "
                        "'R...' (resistance, -x), 'L' (lift, +y), 'Fz' (vertical, +z) or 'N' (yaw moment). "
                        "A pitching moment is given by the point of 'Fz'.")
                if isinstance(entry, tuple):
                    func, point = entry
                else:
                    func, point = entry, None
                process[name] = (func, point)

        return process


    def get_resistance_parameters(
            self,
            op_point: OperatingPoint,
        ):
        """
        Arguments given to every hydrodynamic method: hydrostatics, environment
        and state. Hydrostatics must have been computed beforehand.
        """
        V = op_point.stw * u.kt
        rho = op_point.environment.water.density
        nu = op_point.environment.water.kinematic_viscosity
        g = op_point.environment.gravity

        Re = V * self.hydrostatics_data["Lwl"] / nu
        Fr = V / np.sqrt(self.hydrostatics_data["Lwl"] * g + 1e-12)

        env_params = {
            'g': g,         # m/s^2
            'nu': nu,       # m^2/s
            'rho': rho,     # kg/m^3
        }

        state_params = {
            'stw': op_point.stw,                    # kt
            'V': V,                                 # m/s
            'beta': np.radians(op_point.leeway),    # rad, leeway
            'Re': np.softplus(Re, beta=1e3),
            'Fr': np.softplus(Fr, beta=1e3),
        }

        return {**self.hydrostatics_data, **env_params, **state_params}


    def compute_resistance_components(
            self,
            op_point: OperatingPoint,
            method: Union[str, dict, List[Union[str, dict]]] = "dsyhs",
            recompute_statics: bool = True,
            **kwargs,
        ):
        """
        Values of the components of the hydrodynamic methods.

        Parameters
        ----------
        Same as ``compute_resistance``.

        Returns
        -------
        dict, value of each component, plus "_total", the sum of the
        resistances "R..." [N].
        """
        if recompute_statics:
            self.compute_hydrostatics_properties(op_point)

        process = self.resolve_resistance_methods(method)
        params = self.get_resistance_parameters(op_point)

        r = {}
        t = 0.
        for name, (func, _) in process.items():
            r[name] = func(**params, **kwargs)
            if name.startswith("R"):
                t = t + r[name]

        r["_total"] = t

        return r


    def compute_resistance(
            self,
            op_point: OperatingPoint,
            method: Union[str, dict, List[Union[str, dict]]] = "dsyhs",
            recompute_statics: bool = True,
            **kwargs,
        ):
        """
        Hydrodynamic force and moment of the hull, in the boat motion frame.

        Resistance, lateral force, dynamic lift and added resistance in waves
        are all chosen here: see ``DEFAULT_RESISTANCE_METHODS`` for the
        directions and application points of the components. The forces stay
        in the motion frame: the methods are for an upright hull, so heel is
        only seen through the hydrostatics.

        The values are stored in ``resistance_components``, the application
        points (motion frame) in ``resistance_points``, and the force and
        moment of each component in ``resistance_torsors``.

        Parameters
        ----------
        op_point : operating point. Positive leeway is to starboard.
        method : str, a key of ``DEFAULT_RESISTANCE_METHODS``, dict of
            components, or list of them, e.g. ``["dsyhs", "kijima", "mittendorf"]``.
        recompute_statics : bool, recompute the hydrostatics of the hull.
        **kwargs : parameters of the methods, given to all of them.

        Returns
        -------
        Fh, Mh : force [N] and moment [N.m] about the frame origin, of shape (1, 3).
        """
        if recompute_statics:
            self.compute_hydrostatics_properties(op_point)

        process = self.resolve_resistance_methods(method)
        params = self.get_resistance_parameters(op_point)

        self.resistance_components = self.compute_resistance_components(
            op_point=op_point,
            method=method,
            recompute_statics=False,
            **kwargs
        )

        self.resistance_points = {}
        self.resistance_torsors = {}
        Fh = wide(np.zeros(3))
        Mh = wide(np.zeros(3))
        for name, (_, point_func) in process.items():
            value = self.resistance_components[name]

            if name == "N":
                F = wide(np.zeros(3))
                M = wide(np.array([0., 0., 1.])) * value
            else:
                if point_func is not None:
                    point = wide(point_func(**params, **kwargs))
                elif name in ("L", "Ri"):
                    point = amidships(**params)
                else:
                    point = params['cow']
                center = op_point.apply_transformations(point)
                self.resistance_points[name] = center

                if name.startswith("R"):
                    direction = [-1., 0., 0.]
                elif name == "L":
                    direction = [0., 1., 0.]
                else:
                    direction = [0., 0., 1.]
                F = wide(np.array(direction)) * value
                M = np.cross(center, F)

            self.resistance_torsors[name] = (F, M)
            Fh = Fh + F
            Mh = Mh + M

        return Fh, Mh


    def compute_windage_components(
            self,
            op_point: OperatingPoint,
            tws: float,
            twa: float,
            method: Union[str, dict] = "orc",
            recompute_statics: bool = True,
            crew_position: Union[np.ndarray, List] = None,
            **kwargs,
        ):
        """
        Aerodynamic force of the wind on the above-water part of the hull and on the crew.
        
        Parameters
        ----------
        op_point : OperatingPoint, boat attitude and speed through water.
        tws : float, True wind speed [kt], at the reference height of the wind profile.
        twa : float, True wind angle [deg], positive when the wind comes from starboard.
        method : str or dict, key of ``DEFAULT_WINDAGE_METHODS`` or custom dict of methods.
        recompute_statics : bool, whether to recompute the aerostatics of the hull.
        rho_air : float or None, air density [kg/m^3]. Default from the environment, else 1.225.
        crew_position : (1, 3) or (3,) array or None, position (x, y, z) of the crew in the
            boat frame [m], y being non zero for a crew hiking or on a trapeze. It is brought
            in the water axes, and given to the windage methods as ``crew_position``, where the
            "Crew" element uses its z coordinate as the height of its centre of effort.
        **kwargs : parameters of the windage methods (crew_mass, n_crew, Cd_hull, ...).
        
        Returns
        -------
        dict, for each element of the method: ``Fx`` and ``Fy`` [N] along the boat
        x (forward) and y (port) axes, ``zce`` [m], height of its centre of effort
        above the waterplane, and ``dxce`` [m], longitudinal offset of its application
        point from the lateral centre of effort of the hull. ``_total`` gives the sum
        of ``Fx`` and ``Fy``.
        """
        
        if recompute_statics:
            self.compute_aerostatics_properties(op_point)
            
        g = op_point.environment.gravity
        rho_air = op_point.environment.air.density
            
        env_params = {
            'g': g,                 # m/s^2
            'rho_air': rho_air,     # kg/m^3
        }
        
        state_params = {
            'stw': op_point.stw,
            'tws': tws,
            'twa': twa,
            'heel': getattr(op_point, 'heel', 0.),
        }
        
        # Hydrostatic properties used by some windage methods
        hydro_params = {
            key: self.hydrostatics_data[key]
            for key in ("Cwp",)
            if key in self.hydrostatics_data
        }
        
        # Centroids are brought in the water axes, as done for the buoyancy: 
        # their z coordinate is then the height above the water plane.
        aero_data = dict(self.aerostatics_data)
        for key in ("cov", "cod", "cex", "cey"):
            aero_data[key] = op_point.apply_transformations(self.aerostatics_data[key])
        
        # The crew position is given only to the methods when defined, so that they can
        # fall back on their default height otherwise.
        if crew_position is not None:
            aero_data["crew_position"] = op_point.apply_transformations(wide(np.array(crew_position)))
        
        if type(method)==str and method in DEFAULT_WINDAGE_METHODS.keys():
            process = DEFAULT_WINDAGE_METHODS[method]
        elif type(method)==dict:
            process = method
        elif type(method)==str:
            raise NotImplementedError(
                f"Unknown method '{method}'. Should be in {list(DEFAULT_WINDAGE_METHODS.keys())} or a custom dictionary of methods.")
        else:
            raise TypeError("method argument should be str or dict")
            
        r = {}
        Fx_tot = 0.
        Fy_tot = 0.
        for name, func in process.items():
            out = func(**aero_data, **hydro_params, **env_params, **state_params, **kwargs)
            Fx, Fy, zce = out[:3]
            # optional longitudinal offset of the application point from the lateral centroid [m]
            dxce = out[3] if len(out) > 3 else 0.
            r[name] = {"Fx": Fx, "Fy": Fy, "zce": zce, "dxce": dxce}
            Fx_tot = Fx_tot + Fx
            Fy_tot = Fy_tot + Fy
        
        r["_total"] = {"Fx": Fx_tot, "Fy": Fy_tot}
        
        return r
    
    
    def compute_windage(
            self,
            op_point: OperatingPoint,
            tws: float,
            twa: float,
            method: Union[str, dict] = "orc",
            recompute_statics: bool = True,
            crew_position: Union[np.ndarray, List] = None,
            **kwargs,
        ):
        """
        Aerodynamic force and moment of the wind on the above-water part of the hull
        and on the crew, in the water axes. See ``compute_windage_components``.
        
        The hull elements are applied at the horizontal position of the lateral centre
        of effort of the hull (plus the offset ``dxce`` of the method, if any), and at
        their own height. The "Crew" element is applied at ``crew_position`` if given,
        otherwise like the hull elements.
        
        Parameters
        ----------
        crew_position : (1, 3) or (3,) array or None, position (x, y, z) of the crew in the
            boat frame [m]. The force is applied at this point, off the centreline if y
            is not zero, which gives the yaw moment term -y * Fx.
        
        Returns
        -------
        Fa, Ma : (1, 3) arrays, force [N] and moment [N.m] about the origin.
        """
        
        self.windage_components = self.compute_windage_components(
            op_point=op_point,
            tws=tws,
            twa=twa,
            method=method,
            recompute_statics=recompute_statics,
            crew_position=crew_position,
            **kwargs
        )
        
        cey = op_point.apply_transformations(self.aerostatics_data['cey'])
        
        if crew_position is not None:
            crew_center = op_point.apply_transformations(wide(np.array(crew_position)))
        
        Fa = wide(np.zeros(3))
        Ma = wide(np.zeros(3))
        
        for name, c in self.windage_components.items():
            if name == "_total":
                continue
            
            F = (
                wide(np.array([1., 0., 0.])) * c["Fx"]
                + wide(np.array([0., 1., 0.])) * c["Fy"]
            )
            # horizontal position of the application point
            if name == "Crew" and crew_position is not None:
                xy = crew_center
            else:
                xy = cey
            
            center = (
                xy * wide(np.array([1., 1., 0.]))
                + wide(np.array([1., 0., 0.])) * c["dxce"]
                + wide(np.array([0., 0., 1.])) * c["zce"]
            )
            
            Fa = Fa + F
            Ma = Ma + np.cross(center, F)
        
        return Fa, Ma
        
        
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
        stw=6.,
        dz=-T,
        # leeway=45.,
        # trim=0.1,
        # heel=45.
    )
    
    hull.compute_buoyancy(op_point)
    
    volume = hull.hydrostatics_data['volume']
    
    opti.minimize((volume - ref)**2.)
    
    sol = opti.solve()
    
    # print(sol(T))
    
    hull.compute_resistance(
        op_point,
        method="dsyhs",
    )
    print(hull.resistance_components)
    
    hull.compute_resistance(
        op_point,
        method="holtrop",
        **{'Csternchoice': 1, 'Bulbchoice': 0}
    )
    print(hull.resistance_components)
    
    custom_process = {
        "Rf": holtrop.compute_Rf_holtrop,
        "Rw": dsyhs.compute_Rw_dsyhs,
    }
    
    hull.compute_resistance(
        op_point,
        method=custom_process,
        **{'Csternchoice': 1, 'Bulbchoice': 0}
    )
    print(hull.resistance_components)
    
    hull.compute_resistance(
        op_point,
        method="michell",
        **{'Csternchoice': 1, 'Bulbchoice': 0}
    )
    print(hull.resistance_components)
    
    print(hull.get_position(op_point))
    
    hull.compute_aerostatics_properties(op_point)
    print(hull.aerostatics_data)
    
    print(hull.compute_windage(op_point, tws=12., twa=45., method="orc", n_crew=1.))
    print(hull.windage_components)
    
    print(hull.compute_windage(op_point, tws=12., twa=45., method="day", crew_mass=80., crew_height=1.83))
    print(hull.windage_components)