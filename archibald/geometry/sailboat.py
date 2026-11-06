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

import numpy as onp  # plain numpy, used for visualization only

from archibald.geometry.hull import Hull, DEFAULT_WINDAGE_METHODS
from archibald.geometry.planform import Rig, Appendage, Planform
from archibald.geometry.propeller import Propeller, BSeriesPropeller
from archibald.geometry.mesh import ArchibaldMesh

from archibald.performance.operating_point import OperatingPoint
hod, AeroVortexLatticeMethod
from archibald.dynamics.hydro import multihull

from archibald.weights.mass_properties import MassProperties

import archibald.numpy as np
from archibald.dynamics.aero_3D import HydroVortexLatticeMet
import archibald.toolbox.units as u

from archibald.common import ArchibaldObject

#%%

def tall(array):
    return np.reshape(array, (-1, 1))


def wide(array):
    return np.reshape(array, (1, -1))


# Groups of torsor elements, named after the keys of `Sailboat.forces` and `Sailboat.moments`
TORSOR_GROUPS = ("w", "b", "h", "i", "lat", "aa", "sail", "app", "prop")

EQUILIBRIUM_DOFS = ("Fx", "Fy", "Fz", "Mx", "My", "Mz")

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
            fittings: List[ArchibaldMesh] = [],
            masses: Optional[List[MassProperties]] = None,
        ):

        if masses and displacement is not None:
            raise ValueError(
                "Give either `masses` or `displacement` and `cog`, not both: "
                "the weight would otherwise be ambiguous."
            )

        self.hulls = hulls
        self.rigs = rigs
        self.appendages = appendages
        self.propellers = propellers
        self.fittings = fittings
        self.masses = masses if masses is not None else []
        self.displacement = displacement
        self.cog = wide(np.array(cog)) if cog is not None else None

        self.forces = {}
        self.moments = {}
        self.torsor_elements = []
        self._torsor_op_points = {}  # operating point of each computed group
        self.hull_interaction = None
        self.windage_components = {}
        
        self.aero_vlms = []
        self.hydro_vlms = []
        
    # =========================================================================
    # TORSORS
    # =========================================================================

    def _clear_torsor(self, group: str):
        """Forgets the results of a group of torsor elements, e.g. "w" or "sail"."""
        for store, prefix in ((self.forces, f"F{group}"), (self.moments, f"M{group}")):
            for key in [k for k in store if k == prefix or k.startswith(f"{prefix}_")]:
                del store[key]
        self._torsor_op_points.pop(group, None)

    def _start_torsor(self, group: str, op_point: OperatingPoint):
        """Resets a group of torsor elements, and records the operating point of its results."""
        self._clear_torsor(group)
        self.forces[f"F{group}"] = wide(np.zeros(3))
        self.moments[f"M{group}"] = wide(np.zeros(3))
        self._torsor_op_points[group] = op_point

    def _add_torsor(self, group: str, element: Union[int, str], F: np.ndarray, M: np.ndarray):
        """Stores the torsor of an element, and adds it to the total of its group."""
        F, M = wide(F), wide(M)
        self.forces[f"F{group}_{element}"] = F
        self.moments[f"M{group}_{element}"] = M
        self.forces[f"F{group}"] = self.forces[f"F{group}"] + F
        self.moments[f"M{group}"] = self.moments[f"M{group}"] + M

    def compute_aerodynamics(
            self,
            op_point: OperatingPoint = OperatingPoint(),
            chordwise_resolution=10,
            spanwise_resolution=1,
            alpha_stall=20,
        ):
        """
        Aerodynamic force and moment of each rig, from a vortex lattice method.

        The wings follow the boat through the operating point. Results are stored
        in the boat motion reference frame, with moments about its origin:
        ``forces["Fsail_<i>"]``, ``moments["Msail_<i>"]`` for the i-th rig, and
        ``forces["Fsail"]``, ``moments["Msail"]`` for their sum.

        Returns
        -------
        Fsail, Msail : total force [N] and moment [N.m], of shape (1, 3).
        """
        self._start_torsor("sail", op_point)
        self.aero_vlms = []

        for i, rig in enumerate(self.rigs):
            self.aero_vlm = AeroVortexLatticeMethod(
                rig,
                op_point,
                xyz_ref=[0., 0., 0.],
                chordwise_resolution=chordwise_resolution,
                spanwise_resolution=spanwise_resolution,
            )

            res = self.aero_vlm.run(alpha_stall=alpha_stall)
            self.aero_vlms.append(self.aero_vlm)

            self._add_torsor("sail", i, res["F_ab"], res["M_ab"])

        return self.forces["Fsail"], self.moments["Msail"]

    def compute_hydrodynamics(
            self,
            op_point: OperatingPoint = OperatingPoint(),
            chordwise_resolution=10,
            spanwise_resolution=1,
            alpha_stall=20,
        ):
        """
        Hydrodynamic force and moment of each appendage, from a vortex lattice method.

        The wings follow the boat through the operating point. Results are stored
        in the boat motion reference frame, with moments about its origin:
        ``forces["Fapp_<i>"]``, ``moments["Mapp_<i>"]`` for the i-th appendage, and
        ``forces["Fapp"]``, ``moments["Mapp"]`` for their sum.

        Returns
        -------
        Fapp, Mapp : total force [N] and moment [N.m], of shape (1, 3).
        """
        self._start_torsor("app", op_point)
        self.hydro_vlms = []

        for i, appendage in enumerate(self.appendages):
            self.hydro_vlm = HydroVortexLatticeMethod(
                appendage,
                op_point,
                xyz_ref=[0., 0., 0.],
                chordwise_resolution=chordwise_resolution,
                spanwise_resolution=spanwise_resolution,
            )

            res = self.hydro_vlm.run(alpha_stall=alpha_stall)
            self.hydro_vlms.append(self.hydro_vlm)

            self._add_torsor("app", i, res["F_ab"], res["M_ab"])

        return self.forces["Fapp"], self.moments["Mapp"]

    def compute_weight(
            self,
            op_point: OperatingPoint = OperatingPoint(),
        ):
        """
        Weight force and moment of each mass and of the whole boat.

        Each center of gravity is given in boat-fixed coordinates and follows
        the boat through the operating point. Results are stored in the boat
        motion reference frame: ``forces["Fw_<i>"]`` and ``moments["Mw_<i>"]``
        for the i-th mass of ``masses``, ``forces["Fw"]`` and ``moments["Mw"]``
        for their sum. Without ``masses``, ``displacement`` and ``cog`` act as
        a single mass.

        Returns
        -------
        Fw, Mw : total weight force [N] and moment [N.m], of shape (1, 3).
        """
        masses = self.masses

        if not masses and self.displacement is not None:
            if self.cog is None:
                raise ValueError("`displacement` requires `cog` to compute the weight.")
            cog = wide(self.cog)
            masses = [MassProperties(
                mass=self.displacement,
                x_cg=cog[0, 0],
                y_cg=cog[0, 1],
                z_cg=cog[0, 2],
            )]

        self._start_torsor("w", op_point)

        for i, mass in enumerate(masses):
            Fw_i, Mw_i = mass.compute_weight(op_point)
            self._add_torsor("w", i, Fw_i, Mw_i)

        return self.forces["Fw"], self.moments["Mw"]


    def compute_buoyancy(
            self,
            op_point: OperatingPoint,
            recompute_statics: bool = True,
        ):
        """
        Buoyancy force and moment of each hull, stored as ``forces["Fb_<i>"]`` and
        ``moments["Mb_<i>"]``, and ``forces["Fb"]`` and ``moments["Mb"]`` for their sum.

        Returns
        -------
        Fb, Mb : total buoyancy force [N] and moment [N.m], of shape (1, 3).
        """
        self._start_torsor("b", op_point)

        for i, hull in enumerate(self.hulls):
            Fb_i, Mb_i = hull.compute_buoyancy(
                op_point=op_point,
                recompute_statics=recompute_statics,
            )
            self._add_torsor("b", i, Fb_i, Mb_i)

        return self.forces["Fb"], self.moments["Mb"]


    def compute_hull_interaction(
            self,
            op_point: OperatingPoint,
            method: Union[str, Dict] = "michell",
            positions: Optional[List] = None,
            **kwargs,
        ):
        """
        Interaction resistance between the hulls of the boat.

        The hydrostatics of every hull must have been computed beforehand
        (``compute_buoyancy`` or ``hull.compute_hydrostatics_properties``) and,
        for the calibrated / tabulated methods, so must the individual
        resistances (``hull.compute_resistance``).

        The interference of each hull is applied as a drag force along -x at the
        hull centre of water, as done in ``Hull.compute_resistance``.

        Parameters
        ----------
        op_point : OperatingPoint
        method : str or dict, key of ``multihull.INTERACTION_METHODS`` or custom dict.
        positions : list of (x, y) or None, hull positions, default from ``hull.get_position``.
        **kwargs : method parameters (``calibrate``, ``fn_grid``, ``tau_table``...).

        Returns
        -------
        dict, ``dRw`` (n x n), ``dRf``, ``dR_hull`` and ``dR`` [N].
        """
        self.hull_interaction = multihull.compute_hull_interaction(
            self.hulls,
            op_point,
            method=method,
            positions=positions,
            **kwargs,
        )

        self._start_torsor("i", op_point)

        for i, hull in enumerate(self.hulls):
            center = op_point.apply_transformations(hull.hydrostatics_data['cow'])

            Fi_i = wide(np.array([
                -1.,
                0.,
                0.,
            ])) * self.hull_interaction["dR_hull"][i]
            Mi_i = np.cross(center, Fi_i)

            self._add_torsor("i", i, Fi_i, Mi_i)

        return self.hull_interaction


    def compute_lateral_force(
            self,
            op_point: OperatingPoint,
            method: Union[str, Dict] = "kijima",
            recompute_statics: bool = True,
            **kwargs,
        ):
        """
        Lateral force and added resistance of the hulls, computed individually.

        Stored as ``forces["Flat_<i>"]`` and ``moments["Mlat_<i>"]``, with the
        totals in ``forces["Flat"]`` and ``moments["Mlat"]``. No interaction
        between the hulls of a multihull is accounted for.

        Parameters
        ----------
        method : str or dict, see ``Hull.compute_lateral_force``.
        **kwargs : parameters of the method.

        Returns
        -------
        Flat, Mlat : total force [N] and moment [N.m], of shape (1, 3).
        """
        self._start_torsor("lat", op_point)

        for i, hull in enumerate(self.hulls):
            Flat_i, Mlat_i = hull.compute_lateral_force(
                op_point=op_point,
                method=method,
                recompute_statics=recompute_statics,
                **kwargs,
            )
            self._add_torsor("lat", i, Flat_i, Mlat_i)

        return self.forces["Flat"], self.moments["Mlat"]

    def compute_resistance(
            self,
            op_point: OperatingPoint,
            method: str,
            recompute_statics: bool = True,
            interaction: Optional[Union[str, Dict]] = None,
            interaction_kwargs: Optional[Dict] = None,
            **kwargs,
        ):
        """
        Resistance of the hulls, computed individually, plus their interaction.

        The individual resistances are stored as ``forces["Fh_<i>"]`` and
        ``moments["Mh_<i>"]``, the interaction as ``forces["Fi_<i>"]`` and
        ``moments["Mi_<i>"]``. The totals ``forces["Fh"]`` and ``moments["Mh"]``
        include the interaction.

        Parameters
        ----------
        method : str or dict, individual resistance method (see ``Hull.compute_resistance``).
        interaction : str, dict or None, interaction method between hulls (see
            ``compute_hull_interaction``). None: hulls are independent.
        interaction_kwargs : dict, parameters of the interaction method.
        **kwargs : parameters of the individual resistance method.

        Returns
        -------
        Fh, Mh : total resistance force [N] and moment [N.m], of shape (1, 3).
        """
        self._start_torsor("h", op_point)
        self._clear_torsor("i")
        self.hull_interaction = None

        for i, hull in enumerate(self.hulls):
            Fh_i, Mh_i = hull.compute_resistance(
                op_point=op_point,
                method=method,
                recompute_statics=recompute_statics,
                **kwargs,
            )
            self._add_torsor("h", i, Fh_i, Mh_i)

        if interaction is not None and len(self.hulls) > 1:
            self.compute_hull_interaction(
                op_point=op_point,
                method=interaction,
                **(interaction_kwargs or {}),
            )
            self.forces["Fh"] = self.forces["Fh"] + self.forces["Fi"]
            self.moments["Mh"] = self.moments["Mh"] + self.moments["Mi"]

        return self.forces["Fh"], self.moments["Mh"]


    def compute_windage(
            self,
            op_point: OperatingPoint,
            tws: Optional[float] = None,
            twa: Optional[float] = None,
            method: Union[str, Dict] = "orc",
            recompute_statics: bool = True,
            crew_hull: int = 0,
            crew_position: Optional[Union[np.ndarray, List]] = None,
            **kwargs,
        ):
        """
        Aerodynamic drag of the above-water part of the hulls, and of the crew.

        The hull elements of the method are evaluated for every hull. The element
        named "Crew" is not related to a hull, so it is evaluated only once for the
        whole boat, with the aerostatics of the hull ``crew_hull``.

        The aerostatics of every hull are (re)computed unless ``recompute_statics``
        is False. The hydrostatics must have been computed beforehand
        (``compute_buoyancy``) for the methods requiring ``Cwp``.

        The rig is not included: its aerodynamics is the purpose of ``compute_aerodynamics``.

        Parameters
        ----------
        op_point : OperatingPoint
        tws : float or None, True wind speed [kt]. Default: ``op_point.tws0``.
        twa : float or None, True wind angle [deg], positive when the wind comes
            from starboard. Default: ``op_point.twa``.
        method : str or dict, key of ``hull.DEFAULT_WINDAGE_METHODS`` or custom dict of methods.
        recompute_statics : bool, whether to recompute the aerostatics of the hulls.
        crew_hull : int, index of the hull whose geometry (freeboard, beam) gives the
            default height of the crew. Use ``zce_crew`` to set it directly.
        crew_position : (1, 3) or (3,) array or None, position (x, y, z) of the crew in
            the boat frame [m]. The y coordinate is not zero for a crew hiking or on a
            trapeze. The wind force of the crew is applied at this point, and its z
            coordinate, brought in the water axes (heel included), gives the height of
            its centre of effort. Default: horizontal position of the lateral centre of
            effort of ``crew_hull``, and ORC height of the crew.
        **kwargs : parameters of the windage methods (crew_mass, n_crew, ship_type, ...).

        Returns
        -------
        Fa, Ma : (1, 3) arrays, total force [N] and moment [N.m] in the water axes.
        """
        if len(self.hulls) == 0:
            raise ValueError("The boat has no hull.")

        tws = op_point.tws0 if tws is None else tws
        twa = op_point.twa if twa is None else twa

        if type(method) == str and method in DEFAULT_WINDAGE_METHODS.keys():
            process = DEFAULT_WINDAGE_METHODS[method]
        elif type(method) == dict:
            process = method
        elif type(method) == str:
            raise NotImplementedError(
                f"Unknown method '{method}'. Should be in {list(DEFAULT_WINDAGE_METHODS.keys())} or a custom dictionary of methods.")
        else:
            raise TypeError("method argument should be str or dict")

        hull_process = {k: f for k, f in process.items() if k != "Crew"}
        crew_process = {k: f for k, f in process.items() if k == "Crew"}

        self._start_torsor("aa", op_point)
        self.windage_components = {}

        for i, hull in enumerate(self.hulls):
            Fa_i, Ma_i = hull.compute_windage(
                op_point=op_point,
                tws=tws,
                twa=twa,
                method=hull_process,
                recompute_statics=recompute_statics,
                **kwargs,
            )
            self._add_torsor("aa", i, Fa_i, Ma_i)
            self.windage_components[f"hull_{i}"] = hull.windage_components

        if len(crew_process) > 0:
            # aerostatics of the hulls have just been computed above
            Fa_c, Ma_c = self.hulls[crew_hull].compute_windage(
                op_point=op_point,
                tws=tws,
                twa=twa,
                method=crew_process,
                recompute_statics=False,
                crew_position=crew_position,
                **kwargs,
            )
            self._add_torsor("aa", "crew", Fa_c, Ma_c)
            self.windage_components["crew"] = self.hulls[crew_hull].windage_components

        return self.forces["Faa"], self.moments["Maa"]

    def compute_propellers(
            self,
            op_point: OperatingPoint,
            wake_fraction: float = 0.,
            thrust_deduction: float = 0.,
            relative_rotative_efficiency: float = 1.,
            shaft_efficiency: float = 0.98,
        ):
        """
        Thrust of each propeller at its own ``rpm`` (see ``Propeller.compute_propulsion``).

        Results are stored in the boat motion reference frame:
        ``forces["Fprop_<i>"]`` and ``moments["Mprop_<i>"]`` for the i-th propeller,
        ``forces["Fprop"]`` and ``moments["Mprop"]`` for their sum. The details of each
        propeller (J, Kt, torque, power...) are in ``propeller.performance``.

        Parameters
        ----------
        op_point : OperatingPoint
        wake_fraction : float, wake fraction w [-].
        thrust_deduction : float, thrust deduction t [-].
        relative_rotative_efficiency : float, etaR [-].
        shaft_efficiency : float, etaS [-].

        Returns
        -------
        Fprop, Mprop : total force [N] and moment [N.m], of shape (1, 3).
        """
        self._start_torsor("prop", op_point)

        for i, propeller in enumerate(self.propellers):
            Fprop_i, Mprop_i = propeller.compute_propulsion(
                op_point=op_point,
                wake_fraction=wake_fraction,
                thrust_deduction=thrust_deduction,
                relative_rotative_efficiency=relative_rotative_efficiency,
                shaft_efficiency=shaft_efficiency,
            )
            self._add_torsor("prop", i, Fprop_i, Mprop_i)

        return self.forces["Fprop"], self.moments["Mprop"]

    def compute_torsor(
            self,
            op_point: OperatingPoint,
            components: Optional[Union[str, List[str]]] = None,
            exclude: Optional[Union[str, List[str]]] = None,
        ):
        """
        Sums the torsors already computed, without running any physics.

        Each component is computed by its own method, called explicitly with the
        models and settings wanted: ``compute_weight``, ``compute_buoyancy``,
        ``compute_resistance``, ``compute_hull_interaction``, ``compute_windage``,
        ``compute_aerodynamics``, ``compute_hydrodynamics``, ``compute_propellers``.
        A component which was not computed is not in the sum.

        Components are named after the keys of ``forces`` and ``moments``, either a
        group (``"w"``, ``"b"``, ``"h"``, ``"i"``, ``"aa"``, ``"sail"``, ``"app"``,
        ``"prop"``) or one element of a group (``"w_1"``, ``"aa_crew"``, ``"prop_0"``).

        The sum is stored in ``forces["Ftot"]`` and ``moments["Mtot"]``, and the
        elements it contains in ``torsor_elements``.

        >>> boat.compute_weight(op_point)
        >>> boat.compute_buoyancy(op_point)
        >>> boat.compute_propellers(op_point)
        >>> Ftot, Mtot = boat.compute_torsor(op_point, exclude=["prop", "w_2"])

        Parameters
        ----------
        op_point : OperatingPoint, the operating point of the sum. Summing a group
            computed with another one raises, unless that group is excluded.
        components : str, list of str or None, what to sum. Default: all that was computed.
        exclude : str, list of str or None, what to leave out of ``components``.

        Returns
        -------
        Ftot, Mtot : total force [N] and moment [N.m], of shape (1, 3).
        """
        components = [components] if isinstance(components, str) else components
        exclude = [exclude] if isinstance(exclude, str) else (exclude or [])

        for name in (components or []) + exclude:
            if name.split("_")[0] not in TORSOR_GROUPS:
                raise ValueError(f"Unknown component '{name}'. Groups are {list(TORSOR_GROUPS)}.")

        # Groups to sum, in a fixed order
        if components is None:
            groups = [g for g in TORSOR_GROUPS if g in self._torsor_op_points]
        else:
            groups = [g for g in TORSOR_GROUPS if any(n.split("_")[0] == g for n in components)]

        for group in groups:
            if group not in self._torsor_op_points:
                raise ValueError(f"Component '{group}' was asked for but has not been computed.")

        # Every element named must exist, so that a typo never silently changes the sum
        computed = [k[1:] for k in self.forces if k.split("_")[0][1:] in self._torsor_op_points and "_" in k]
        for name in (components or []) + exclude:
            if "_" in name and name.split("_")[0] in self._torsor_op_points and name not in computed:
                raise ValueError(f"Unknown element '{name}'. Computed elements are {computed}.")

        def is_in(element, names):
            return any(element == n or element.startswith(f"{n}_") for n in names)

        elements = []
        for element in computed:
            if element.split("_")[0] in groups and not is_in(element, exclude):
                if components is None or is_in(element, components):
                    elements.append(element)

        # Excluded groups may be stale, but what is summed may not
        for group in {e.split("_")[0] for e in elements}:
            if self._torsor_op_points[group] is not op_point:
                raise ValueError(
                    f"Component '{group}' was computed with another OperatingPoint: "
                    f"compute it again with this one, or exclude it."
                )

        Ftot = wide(np.zeros(3))
        Mtot = wide(np.zeros(3))
        for element in elements:
            Ftot = Ftot + self.forces[f"F{element}"]
            Mtot = Mtot + self.moments[f"M{element}"]

        self.forces["Ftot"] = Ftot
        self.moments["Mtot"] = Mtot
        self.torsor_elements = elements

        return Ftot, Mtot

    @staticmethod
    def equilibrium_residuals(
            Ftot: np.ndarray,
            Mtot: np.ndarray,
            dofs: Union[str, List[str]] = ("Fz", "Mx", "My"),
            force_scale: float = 1.,
            moment_scale: float = 1.,
        ) -> Dict[str, np.ndarray]:
        """
        Residuals of a set of degrees of freedom of a torsor.

        Parameters
        ----------
        Ftot, Mtot : force [N] and moment [N.m] of shape (1, 3).
        dofs : str or list of str, among "Fx", "Fy", "Fz", "Mx", "My", "Mz".
        force_scale : float, reference force dividing the force residuals [N].
        moment_scale : float, reference moment dividing the moment residuals [N.m].

        Returns
        -------
        dict, residual of each degree of freedom, scaled.
        """
        dofs = [dofs] if isinstance(dofs, str) else dofs

        residuals = {}
        for dof in dofs:
            if dof not in EQUILIBRIUM_DOFS:
                raise ValueError(f"Unknown degree of freedom '{dof}'. Use some of {list(EQUILIBRIUM_DOFS)}.")
            torsor, scale = (Ftot, force_scale) if dof[0] == "F" else (Mtot, moment_scale)
            residuals[dof] = wide(torsor)[0, "xyz".index(dof[1])] / scale

        return residuals

    @staticmethod
    def find_equilibrium(
            opti,
            Ftot: np.ndarray,
            Mtot: np.ndarray,
            dofs: Union[str, List[str]] = ("Fz", "Mx", "My"),
            force_scale: float = 1.,
            moment_scale: float = 1.,
            **solve_kwargs,
        ):
        """
        Constrains degrees of freedom of a torsor to zero and solves the problem.

        What varies is what the user made an ``opti.variable`` when building the
        problem: any number of the operating point, of the masses, of the wings or of
        the propellers. Any other constraint or objective goes through
        ``opti.subject_to`` and ``opti.minimize`` before this call, and terms external
        to the boat through the torsor, as in ``Ftot + Ffoil``.

        Call it once per ``Opti``: each call adds its constraints to the problem.

        >>> Ftot, Mtot = boat.compute_torsor(op_point)
        >>> opti.subject_to(boat.propellers[0].performance["P"] <= 500e3)
        >>> sol = boat.find_equilibrium(opti, Ftot, Mtot, dofs=["Fx", "Fz", "Mx", "My"])

        Parameters
        ----------
        opti : Opti, the problem the variables belong to.
        Ftot, Mtot : force [N] and moment [N.m] of shape (1, 3).
        dofs : str or list of str, among "Fx", "Fy", "Fz", "Mx", "My", "Mz".
        force_scale : float, reference force dividing the force residuals [N].
        moment_scale : float, reference moment dividing the moment residuals [N.m].
        **solve_kwargs : arguments of ``opti.solve``.

        Returns
        -------
        OptiSol, the solution.
        """
        residuals = Sailboat.equilibrium_residuals(
            Ftot,
            Mtot,
            dofs=dofs,
            force_scale=force_scale,
            moment_scale=moment_scale,
        )
        opti.subject_to([residual == 0. for residual in residuals.values()])

        return opti.solve(**solve_kwargs)


    # =========================================================================
    # VISUALIZATION
    # =========================================================================

    _THEMES = {
        "dark": dict(
            background="#050818",
            foreground="white",
            water="#4a62c9",
            streamlines="#9db4ff",
            particles="white",
        ),
        "light": dict(
            background="white",
            foreground="black",
            water="slateblue",
            streamlines="slateblue",
            particles="#1b2a6b",
        ),
    }

    # ----------------------------- geometry helpers --------------------------

    @staticmethod
    def _quads_to_triangles(faces):
        """(N, 4) quad faces -> (2N, 3) triangles. Triangle k and N + k come from quad k."""
        faces = onp.asarray(faces, dtype=int)
        if faces.shape[1] == 3:
            return faces
        return onp.vstack([faces[:, [0, 1, 2]], faces[:, [0, 2, 3]]])

    def _body_meshes(self, thin_wings: bool = False):
        """(points, faces) of every rig and appendage."""
        out = []
        for part in list(self.rigs) + list(self.appendages):
            p, f = part.mesh_body(method="quad", thin_wings=thin_wings)
            out.append((onp.asarray(p, dtype=float), onp.asarray(f, dtype=int)))
        return out

    def _hull_vertices(self):
        """Vertices of all the hulls (``hull.mesh.vertices``), stacked as (N, 3).
        Falls back on the rigs / appendages if no hull mesh is available."""
        chunks = []
        for hull in self.hulls:
            try:
                chunks.append(onp.asarray(hull.mesh.vertices, dtype=float).reshape(-1, 3))
            except AttributeError:
                continue
        if not chunks:
            chunks = [p for p, _ in self._body_meshes()]
        return onp.vstack(chunks) if chunks else None

    def _water_plane(
            self,
            origin=(0., 0., 0.),
            normal=(0., 0., 1.),
            margin: float = 0.4,
            vertices=None,
        ):
        """
        Corners (4, 3) of a rectangle lying in the plane (origin, normal), sized
        and centred on the hull vertices projected onto that plane.

        margin : relative enlargement of the hull footprint (0.4 -> 40 % larger).
        Returns None if there is no geometry to fit the plane on.
        """
        if vertices is None:
            vertices = self._hull_vertices()
        if vertices is None or len(vertices) == 0:
            return None

        origin = onp.asarray(origin, dtype=float).reshape(3)
        n = onp.asarray(normal, dtype=float).reshape(3)
        n = n / onp.linalg.norm(n)

        # In-plane orthonormal basis (u, v); for n = +z this gives u = +x, v = +y
        ref = onp.array([1., 0., 0.]) if abs(n[0]) < 0.9 else onp.array([0., 1., 0.])
        u = ref - (ref @ n) * n
        u /= onp.linalg.norm(u)
        v = onp.cross(n, u)

        rel = onp.asarray(vertices, dtype=float) - origin
        s, t = rel @ u, rel @ v

        centre_s, centre_t = 0.5 * (s.min() + s.max()), 0.5 * (t.min() + t.max())
        half_s = max(0.5 * (s.max() - s.min()) * (1. + margin), 1e-6)
        half_t = max(0.5 * (t.max() - t.min()) * (1. + margin), 1e-6)

        ds = centre_s + onp.array([-1., 1., 1., -1.]) * half_s
        dt = centre_t + onp.array([-1., -1., 1., 1.]) * half_t
        return origin + ds[:, None] * u + dt[:, None] * v

    @staticmethod
    def _add_water_pyvista(plotter, corners, color, opacity, outline_color=None):
        import pyvista as pv

        plotter.add_mesh(
            pv.PolyData(corners, faces=onp.array([4, 0, 1, 2, 3])),
            color=color,
            opacity=opacity,
            lighting=False,
            show_scalar_bar=False,
            name="water",
        )
        plotter.add_mesh(
            pv.lines_from_points(onp.vstack([corners, corners[:1]])),
            color=outline_color or color,
            line_width=1,
            opacity=min(1., 2.5 * opacity),
            lighting=False,
            name="water_outline",
        )

    @staticmethod
    def _add_water_plotly(fig, corners, color, opacity):
        import plotly.graph_objects as go

        fig.add_trace(
            go.Mesh3d(
                x=corners[:, 0], y=corners[:, 1], z=corners[:, 2],
                i=[0, 0], j=[1, 2], k=[2, 3],
                color=color,
                opacity=opacity,
                hoverinfo="skip",
                name="Water",
            )
        )

    @staticmethod
    def _add_water_matplotlib(ax, corners, color, opacity):
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

        ax.add_collection3d(
            Poly3DCollection([corners], facecolor=color, alpha=opacity, edgecolor=color)
        )

    # --------------------------------- draw ----------------------------------

    def draw(
        self,
        backend: str = "pyvista",
        thin_wings: bool = False,
        mesh_color: str = "lightgrey",
        show: bool = True,
        show_kwargs: Dict = None,
        draw_water: bool = True,
        water_origin: Union[np.ndarray, List, Tuple] = (0., 0., 0.),
        water_normal: Union[np.ndarray, List, Tuple] = (0., 0., 1.),
        water_margin: float = 0.4,
        water_color: str = "slateblue",
        water_opacity: float = 0.25,
        **kwargs,
    ):
        """
        Visualizes the sailboat by aggregating all wings, appendages and hulls
        into a single scene.

        Parameters
        ----------
        draw_water : bool
            Draw the waterplane.
        water_origin, water_normal :
            Point and normal of the waterplane. Default: (0, 0, 0) and (0, 0, 1).
        water_margin : float
            The waterplane is centred on the hull (``hull.mesh.vertices``) and is
            ``1 + water_margin`` times larger than its footprint.
        water_color, water_opacity :
            Appearance of the waterplane.
        **kwargs :
            Forwarded to ``hull.draw`` and ``fitting.draw``.
        """
        if show_kwargs is None:
            show_kwargs = {}

        water = None
        if draw_water:
            water = self._water_plane(water_origin, water_normal, water_margin)

        # =====================================================================
        # PYVISTA
        # =====================================================================
        if backend == "pyvista":
            import pyvista as pv

            plotter = pv.Plotter()

            for p, f in self._body_meshes(thin_wings):
                faces = onp.hstack([onp.full((len(f), 1), f.shape[1]), f]).ravel()
                plotter.add_mesh(
                    pv.PolyData(p, faces),
                    color=mesh_color,
                    show_edges=True,
                )

            for hull in self.hulls:
                hull.draw(backend="pyvista", show=False, plotter=plotter, **kwargs)

            for fitting in self.fittings:
                fitting.draw(backend="pyvista", show=False, plotter=plotter, **kwargs)

            if water is not None:
                self._add_water_pyvista(plotter, water, water_color, water_opacity)

            if show:
                plotter.show(**show_kwargs)

            return plotter

        # =====================================================================
        # PLOTLY
        # =====================================================================
        elif backend == "plotly":
            import plotly.graph_objects as go
            import plotly.offline as pyo

            fig = go.Figure()

            for p, f in self._body_meshes(thin_wings):
                i, j, k = self._quads_to_triangles(f).T
                fig.add_trace(
                    go.Mesh3d(
                        x=p[:, 0], y=p[:, 1], z=p[:, 2],
                        i=i, j=j, k=k,
                        color=mesh_color,
                        opacity=1.0,
                        flatshading=True,
                    )
                )

            for hull in self.hulls:
                hull.draw(backend="plotly", show=False, fig=fig, **kwargs)

            for fitting in self.fittings:
                fitting.draw(backend="plotly", show=False, fig=fig, **kwargs)

            if water is not None:
                self._add_water_plotly(fig, water, water_color, water_opacity)

            fig.update_layout(scene=dict(aspectmode="data"))

            if show:
                pyo.plot(fig, auto_open=True)

            return fig

        # =====================================================================
        # MATPLOTLIB
        # =====================================================================
        elif backend == "matplotlib":
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d.art3d import Poly3DCollection

            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")

            vertices = []

            for p, f in self._body_meshes(thin_wings):
                ax.add_collection3d(
                    Poly3DCollection(
                        [p[face] for face in f],
                        facecolor=mesh_color,
                        edgecolor="k",
                        linewidth=0.3,
                    )
                )
                vertices.append(p)

            for hull in self.hulls:
                hull.draw(backend="matplotlib", show=False, ax=ax, **kwargs)

            for fitting in self.fittings:
                fitting.draw(backend="matplotlib", show=False, ax=ax, **kwargs)

            if water is not None:
                self._add_water_matplotlib(ax, water, water_color, water_opacity)
                vertices.append(water)

            hull_vertices = self._hull_vertices()
            if hull_vertices is not None:
                vertices.append(hull_vertices)

            # Equal scaling
            if len(vertices):
                xyz = onp.vstack(vertices)
                mins, maxs = xyz.min(axis=0), xyz.max(axis=0)
                centre = 0.5 * (mins + maxs)
                radius = 0.5 * onp.max(maxs - mins)

                ax.set_xlim(centre[0] - radius, centre[0] + radius)
                ax.set_ylim(centre[1] - radius, centre[1] + radius)
                ax.set_zlim(centre[2] - radius, centre[2] + radius)

            ax.set_box_aspect((1, 1, 1))

            if show:
                plt.show()

            return ax

        else:
            raise ValueError(
                f"Unknown backend '{backend}'. "
                "Available backends are 'pyvista', 'plotly' and 'matplotlib'."
            )

    # ------------------------------- draw_vlm --------------------------------

    @staticmethod
    def _vlm_panels(vlm):
        """Quad panel mesh of a VLM: points (4N, 3) and faces (N, 4)."""
        points = onp.concatenate([
            onp.asarray(vlm.front_left_vertices, dtype=float),
            onp.asarray(vlm.back_left_vertices, dtype=float),
            onp.asarray(vlm.back_right_vertices, dtype=float),
            onp.asarray(vlm.front_right_vertices, dtype=float),
        ])
        N = len(vlm.front_left_vertices)
        faces = onp.arange(N)[:, None] + (onp.arange(4) * N)[None, :]
        return points, faces

    @staticmethod
    def _default_seed_points(vlm):
        """Middle of the trailing edge of every trailing-edge panel."""
        mask = vlm.is_trailing_edge
        return 0.5 * (
            onp.asarray(vlm.back_left_vertices, dtype=float)[mask]
            + onp.asarray(vlm.back_right_vertices, dtype=float)[mask]
        )

    def _get_streamlines(
            self,
            vlm,
            below_water: bool,
            length: float,
            seed_stride: int,
            seed_fn: Optional[callable],
            recalculate: bool,
            water_origin,
            water_normal,
        ):
        """
        Streamlines of a VLM as an array (n_lines, n_steps, 3), computed (and cached
        on the VLM object) if necessary. None if there is no streamline to draw.
        """
        if recalculate or not hasattr(vlm, "streamlines"):
            seeds = (seed_fn or self._default_seed_points)(vlm)
            seeds = onp.asarray(seeds, dtype=float).reshape(-1, 3)

            if below_water:  # hydrodynamic seeds above the free surface are meaningless
                n = onp.asarray(water_normal, dtype=float)
                n = n / onp.linalg.norm(n)
                seeds = seeds[(seeds - onp.asarray(water_origin, dtype=float)) @ n < 0.]

            seeds = seeds[::max(int(seed_stride), 1)]
            if len(seeds) == 0:
                return None

            vlm.calculate_streamlines(seed_points=seeds, length=length)

        paths = onp.transpose(onp.asarray(vlm.streamlines, dtype=float), (0, 2, 1))
        paths = paths[onp.all(onp.isfinite(paths), axis=(1, 2))]  # drop diverged lines
        return paths if len(paths) else None

    @staticmethod
    def _streamlines_travel_time(vlm, paths):
        """
        Time [s] taken by a fluid particle to reach each point of the streamlines
        (n_lines, n_steps, 3), from the local velocity of the VLM. Far from the lifting
        surfaces, this velocity tends to the freestream: ``op_point._stw`` in water and
        ``op_point._aws(z)`` in air. As the streamlines run close to the wake filaments,
        where the velocity is singular, the local speed is limited to [0.1, 3] x freestream.
        """
        n, m = paths.shape[:2]
        points = paths.reshape(-1, 3)
        V = onp.asarray(vlm.get_velocity_at_points(points), dtype=float)
        Vinf = onp.asarray(vlm.get_freestream_velocity_at_points(points), dtype=float)
        speed = onp.linalg.norm(V, axis=1).reshape(n, m)
        speed_inf = onp.linalg.norm(Vinf, axis=1).reshape(n, m)
        speed = onp.clip(speed, 0.1 * speed_inf, 3. * speed_inf)

        ds = onp.linalg.norm(onp.diff(paths, axis=1), axis=2)
        dt = ds / (0.5 * (speed[:, 1:] + speed[:, :-1]))
        return onp.hstack([onp.zeros((n, 1)), onp.cumsum(dt, axis=1)])

    def draw_vlm(
        self,
        backend: str = "pyvista",
        *,
        # --- colouring ---
        cmap: str = "viridis",
        clim: Optional[Tuple[float, float]] = None,
        clip_quantiles: Optional[Tuple[float, float]] = (0.02, 0.98),
        colorbar_label: str = "Cp",
        draw_colorbar: bool = True,
        show_edges: bool = False,
        # --- streamlines ---
        draw_streamlines: bool = True,
        streamlines_length: Optional[float] = None,
        streamlines_radius: Optional[float] = None,
        seed_stride: int = 2,
        seed_fn: Optional[callable] = None,
        recalculate_streamlines: bool = False,
        # --- animation (pyvista only) ---
        animate: Optional[bool] = None,
        particles_per_line: int = 4,
        particle_speed: float = 0.1,
        particle_size: float = 7.,
        gif_path: Optional[str] = None,
        n_frames: int = 150,
        fps: int = 30,
        # --- scene ---
        theme: str = "dark",
        draw_hulls: bool = True,
        hull_kwargs: Optional[Dict] = None,
        draw_water: bool = True,
        water_origin: Union[np.ndarray, List, Tuple] = (0., 0., 0.),
        water_normal: Union[np.ndarray, List, Tuple] = (0., 0., 1.),
        water_margin: float = 0.4,
        plot_axes: bool = True,
        window_size: Optional[Tuple[int, int]] = None,
        show: bool = True,
        show_kwargs: Optional[Dict] = None,
    ):
        """
        Draws the VLM solution of the rigs and appendages: the panels are coloured
        by pressure coefficient (``vlm.pressure_coef``) and the flow is shown with
        streamlines, optionally animated by particles advected along them.

        ``compute_aerodynamics`` and/or ``compute_hydrodynamics`` must have been called
        with numerical (not symbolic) values beforehand.

        Parameters
        ----------
        backend : "pyvista" (default, supports animation) or "plotly" (static).
        cmap : colormap name. One colour scale is shared by the sails and the appendages.
        clim : (min, max) of the colour scale. Default: ``clip_quantiles`` of all the Cp.
        clip_quantiles : quantiles of Cp setting the colour limits when ``clim`` is None,
            so that stagnation / tip peaks do not wash out the rest. None: min / max.
        draw_streamlines : draw the streamlines (computed on demand, cached on the VLM).
        streamlines_length : length of the streamlines [m]. Default: 5 x the largest reference chord.
        streamlines_radius : tube radius of the streamlines [m]. Default: 0.15 % of the size of the boat.
        seed_stride : keep one seed out of ``seed_stride`` (appendage seeds above the water are removed).
        seed_fn : optional ``f(vlm) -> (n, 3)`` array of custom seed points.
        recalculate_streamlines : ignore the streamlines already stored on the VLMs.
        animate : animate particles along the streamlines (pyvista only). The window stays
            open until closed or 'q' is pressed. Default: True with pyvista.
        particles_per_line, particle_size : Number of particles per streamline, their size [px].
        particle_speed : playback rate of the particles [s of flow / s of animation at ``fps``], 1 is
            real time. Particles move at the local flow speed, with the same clock in air and water.
        gif_path : if set, renders ``n_frames`` frames off-screen to this GIF instead of opening a window
            (implies ``animate``).
        theme : "dark" or "light".
        draw_hulls : draw the hulls and fittings at the operating point of the VLM, with ``hull_kwargs``.
        draw_water, water_origin, water_normal, water_margin : waterplane, see ``draw``.
        plot_axes : show the orientation axes (pyvista).
        window_size : (width, height) in pixels (pyvista).

        Returns
        -------
        pyvista.Plotter or plotly.graph_objects.Figure
        """
        import warnings

        if theme not in self._THEMES:
            raise ValueError(f"Unknown theme '{theme}'. Available: {list(self._THEMES)}")
        th = self._THEMES[theme]

        show_kwargs = show_kwargs or {}
        hull_kwargs = hull_kwargs or {}

        aero_vlms = list(getattr(self, "aero_vlms", []))
        hydro_vlms = list(getattr(self, "hydro_vlms", []))
        entries = [(v, False) for v in aero_vlms] + [(v, True) for v in hydro_vlms]
        if not entries:
            raise RuntimeError(
                "No VLM solution to draw: call compute_aerodynamics() and/or compute_hydrodynamics() first."
            )

        if animate is None:
            animate = backend == "pyvista"
        animate = animate or gif_path is not None
        if animate and backend != "pyvista":
            warnings.warn("Animation is only available with the 'pyvista' backend: drawing a static figure.")
            animate = False

        # --- panel meshes and pressure coefficient -------------------------------
        panels = [self._vlm_panels(v) for v, _ in entries]
        cps = [onp.asarray(v.pressure_coef, dtype=float).ravel() for v, _ in entries]

        if clim is None:
            allcp = onp.concatenate(cps)
            allcp = allcp[onp.isfinite(allcp)]
            if clip_quantiles is None:
                clim = (float(allcp.min()), float(allcp.max()))
            else:
                clim = tuple(float(q) for q in onp.quantile(allcp, clip_quantiles))
        cps = [onp.clip(c, *clim) for c in cps]

        # --- streamlines ---------------------------------------------------------
        all_points = onp.vstack([p for p, _ in panels])
        diag = float(onp.linalg.norm(all_points.max(axis=0) - all_points.min(axis=0)))

        lines = []
        line_vlms = []
        if draw_streamlines:
            if streamlines_length is None:
                chords = [getattr(getattr(v, "airplane", None), "c_ref", None) for v, _ in entries]
                chords = [float(c) for c in chords if c is not None]
                streamlines_length = 5. * max(chords) if chords else 0.5 * diag
            for vlm, is_hydro in entries:
                paths = self._get_streamlines(
                    vlm, is_hydro, streamlines_length, seed_stride, seed_fn,
                    recalculate_streamlines, water_origin, water_normal,
                )
                if paths is not None:
                    lines.append(paths)
                    line_vlms.append(vlm)

        water = None
        if draw_water:
            water = self._water_plane(water_origin, water_normal, water_margin)

        # The panels and the flow are in the boat motion frame: move copies of the hulls there
        moved_meshes = []
        if draw_hulls:
            for mesh in [hull.mesh for hull in self.hulls if hull.mesh is not None] + list(self.fittings):
                moved_meshes.append(copy.copy(mesh))
                moved_meshes[-1].transform(entries[0][0].op_point)

        # =====================================================================
        # PLOTLY (static)
        # =====================================================================
        if backend == "plotly":
            import plotly.graph_objects as go

            fig = go.Figure()

            for idx, ((points, faces), cp) in enumerate(zip(panels, cps)):
                tri = self._quads_to_triangles(faces)
                fig.add_trace(go.Mesh3d(
                    x=points[:, 0], y=points[:, 1], z=points[:, 2],
                    i=tri[:, 0], j=tri[:, 1], k=tri[:, 2],
                    intensity=onp.concatenate([cp, cp]),  # one value per triangle
                    intensitymode="cell",
                    colorscale=cmap.title(),
                    cmin=clim[0], cmax=clim[1],
                    showscale=draw_colorbar and idx == 0,
                    colorbar=dict(title=dict(text="Cp", font=dict(color=th["foreground"])),
                                  tickfont=dict(color=th["foreground"])),
                    flatshading=True,
                    name=f"{'Hydro' if entries[idx][1] else 'Aero'} VLM",
                ))

            for paths in lines:  # one trace per VLM, lines separated by NaN
                sep = onp.full((len(paths), 1, 3), onp.nan)
                xyz = onp.concatenate([paths, sep], axis=1).reshape(-1, 3)
                fig.add_trace(go.Scatter3d(
                    x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2],
                    mode="lines",
                    line=dict(color=th["streamlines"], width=2),
                    opacity=0.7,
                    hoverinfo="skip",
                    name="Streamlines",
                ))

            if draw_hulls:
                for mesh in moved_meshes:
                    mesh.draw(backend="plotly", show=False, fig=fig, **hull_kwargs)

            if water is not None:
                self._add_water_plotly(fig, water, th["water"], 0.25)

            axis = dict(visible=False)
            fig.update_layout(
                scene=dict(xaxis=axis, yaxis=axis, zaxis=axis, aspectmode="data",
                           bgcolor=th["background"]),
                paper_bgcolor=th["background"],
                showlegend=False,
                margin=dict(l=0, r=0, t=0, b=0),
            )

            if show:
                from plotly.offline import plot
                plot(fig)
            return fig

        # =====================================================================
        # PYVISTA (static or animated)
        # =====================================================================
        elif backend == "pyvista":
            import pyvista as pv

            off_screen = gif_path is not None
            pl = pv.Plotter(off_screen=off_screen, window_size=window_size)
            pl.title = "Sailboat VLM"
            pl.set_background(th["background"])  # flat colour: gradients break shaders on some GPUs

            # Panels, coloured by Cp
            for idx, ((points, faces), cp) in enumerate(zip(panels, cps)):
                cells = onp.hstack([onp.full((len(faces), 1), 4), faces]).ravel()
                mesh = pv.PolyData(points, cells)
                mesh.cell_data["Cp"] = cp
                pl.add_mesh(
                    mesh,
                    scalars="Cp",
                    cmap=cmap,
                    clim=clim,
                    show_edges=show_edges,
                    edge_color=th["background"],
                    line_width=0.5,
                    show_scalar_bar=draw_colorbar and idx == 0,
                    scalar_bar_args=dict(
                        title=colorbar_label,
                        color=th["foreground"],
                        n_labels=5,
                        fmt="%.2f",
                        vertical=True,
                    ),
                    smooth_shading=False,
                    specular=0.3,
                )

            # Static streamlines (one polyline per line)
            radius = streamlines_radius if streamlines_radius is not None else 1.5e-3 * diag
            for paths in lines:
                n, m = paths.shape[:2]
                cells = onp.hstack([onp.full((n, 1), m), onp.arange(n * m).reshape(n, m)]).ravel()
                polylines = pv.PolyData(paths.reshape(-1, 3), lines=cells)
                # real tube geometry rather than the line-as-tube shader, which
                # misbehaves on some GPUs / software renderers
                tubes = polylines.tube(radius=radius, n_sides=6, capping=False)
                pl.add_mesh(
                    tubes,
                    color=th["streamlines"],
                    opacity=0.45 if animate else 0.7,
                    lighting=False,
                    show_scalar_bar=False,
                )

            if draw_hulls:
                for mesh in moved_meshes:
                    mesh.draw(backend="pyvista", show=False, plotter=pl, **hull_kwargs)

            if water is not None:
                self._add_water_pyvista(pl, water, th["water"], 0.25)

            if plot_axes:
                pl.add_axes(color=th["foreground"])

            # --- Particles advected along the streamlines ------------------------
            def make_stepper():
                rng = onp.random.default_rng(0)
                k = max(int(particles_per_line), 1)
                groups = []
                for paths, vlm in zip(lines, line_vlms):
                    n = paths.shape[0]
                    times = self._streamlines_travel_time(vlm, paths)
                    # evenly spaced in time along each line, with a random phase per line
                    phase = (rng.random((n, 1)) + onp.arange(k)[None, :] / k) % 1.
                    groups.append((paths, times, phase * times[:, -1:]))

                def positions(frame):
                    clock = frame * particle_speed / max(int(fps), 1)  # flow time [s]
                    out = []
                    for paths, times, t0 in groups:
                        for line, t_line, t0_line in zip(paths, times, t0):
                            t = (t0_line + clock) % t_line[-1]
                            i0 = onp.clip(onp.searchsorted(t_line, t, side="right") - 1, 0, len(t_line) - 2)
                            w = ((t - t_line[i0]) / (t_line[i0 + 1] - t_line[i0]))[:, None]
                            out.append((1 - w) * line[i0] + w * line[i0 + 1])
                    return onp.vstack(out)

                return positions

            particles = None
            if animate and lines:
                positions = make_stepper()
                particles = pv.PolyData(positions(0))
                pl.add_mesh(
                    particles,
                    color=th["particles"],
                    point_size=particle_size,
                    render_points_as_spheres=False,
                    lighting=False,
                    show_scalar_bar=False,
                )

            pl.view_isometric()

            # --- Render ----------------------------------------------------------
            if gif_path is not None:
                try:
                    pl.open_gif(gif_path, fps=fps)
                except TypeError:  # older pyvista without fps argument
                    pl.open_gif(gif_path)
                pl.show(auto_close=False)
                for frame in range(n_frames):
                    if particles is not None:
                        particles.points[:] = positions(frame)
                        particles.points = particles.points
                    pl.write_frame()
                pl.close()

            elif particles is not None and show:
                pl.show(interactive_update=True, auto_close=False, **show_kwargs)

                state = {"running": True}

                def stop(*_):
                    state["running"] = False

                # fires when the window is closed (X button) or 'q' is pressed
                pl.iren.interactor.AddObserver("ExitEvent", stop)

                frame = 0
                while state["running"] and not getattr(pl, "_closed", False):
                    frame += 1
                    particles.points[:] = positions(frame)  # in-place update
                    particles.points = particles.points      # flag points as modified
                    pl.update(stime=1000 // max(int(fps), 1), force_redraw=True)

                if not getattr(pl, "_closed", False):
                    pl.close()  # release the GL context cleanly

            elif show:
                pl.show(**show_kwargs)

            return pl

        else:
            raise ValueError("Unknown backend. Available backends for draw_vlm are 'pyvista' and 'plotly'.")
