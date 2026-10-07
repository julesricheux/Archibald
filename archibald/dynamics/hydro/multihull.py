# -*- coding: utf-8 -*-
"""
MULTIHULL INTERACTION: COORDINATION OF THE METHODS

The resistance of each hull is computed alone first (``hull.compute_resistance``).
The interaction between the n hulls is then evaluated at the Sailboat level
by :func:`compute_hull_interaction`, which returns an n x n matrix of wave
interference ``dRw`` and a vector of viscous interference ``dRf``:

    R_total = sum_i R_i + sum_ij dRw[i][j] + sum_i dRf[i]

``dRw`` is symmetric for Michell methods; its row i is the share of the
interference attributed to hull i (half of each pair).
"""

#%% DEPENDENCIES

from archibald.dynamics.hydro import michell, insel_molland

#%% METHODS

# method name -> {component: function}. As for ``hull.compute_resistance``, a
# custom dict {"Rw": callable, "Rf": callable} can be given instead of a name.
# "Rw" callables return an n x n matrix, "Rf" callables a list of length n.
INTERACTION_METHODS = {
    "none": {},
    "michell": {
        "Rw": michell.compute_dRw_michell,
    },
    "michell_raw": {
        "Rw": lambda **kw: michell.compute_dRw_michell(**{**kw, "calibrate": False}),
    },
    "insel_molland": {
        "Rw": insel_molland.compute_dRw_insel_molland,
        "Rf": insel_molland.compute_dRf_insel_molland,
    },
}

#%% SAILBOAT-LEVEL ENTRY POINT

def get_hull_position(hull):
    """Position (x, y) [m] of the centre of buoyancy of a hull in the boat frame.

    Single place to adapt to the actual ``Hull`` API: first ``hull.position``,
    then ``hull.hydrostatics_data['cb']`` (only the first two components).
    """
    pos = getattr(hull, "position", None)
    if pos is None:
        pos = hull.hydrostatics_data["cb"]
    return pos[0], pos[1]


def compute_hull_interaction(
        hulls,  # TODO Sailboat method: replace by ``self`` and start with ``hulls = self.hulls``
        op_point,
        method="michell",
        positions=None,
        rho=1025.,
        g=9.81,
        nu=1.189e-6,
        **kwargs,
    ):
    """Interaction resistance between the hulls of a multihull.

    Requires, for every hull, ``hull.hydrostatics_data`` (hydrostatics already
    computed) and, for the calibrated / tabulated methods,
    ``hull.resistance_components`` (individual resistance already computed).

    Parameters
    ----------
    hulls : list of Hull, Hulls with computed hydrostatics.
    op_point : OperatingPoint, Operating point (only ``stw`` is used).
    method : str or dict, A key of ``INTERACTION_METHODS`` or a custom dict
        ``{"Rw": callable, "Rf": callable}``.
    positions : list of (x, y) or None, Hull positions [m]; if None, read with
        :func:`get_hull_position`.
    rho : float, Water density [kg/m³].
    g : float, Gravitational acceleration [m/s²].
    nu : float, Kinematic viscosity of water [m²/s].
    **kwargs : Method parameters (``calibrate``, ``fn_grid``, ``sl_grid``, ``tau_table``...).

    Returns
    -------
    dict with
        ``dRw`` : n x n wave interference matrix [N],
        ``dRf`` : viscous interference per hull [N],
        ``dR_hull`` : total interference attributed to each hull [N],
        ``dR`` : total interference [N].
    """
    n = len(hulls)
    process = INTERACTION_METHODS[method] if isinstance(method, str) else method

    # Flat per-hull lists of every hydrostatic value shared by all hulls
    common = set.intersection(*[set(h.hydrostatics_data) for h in hulls])
    data = {k: [h.hydrostatics_data[k] for h in hulls] for k in common}

    if positions is None:
        positions = [get_hull_position(h) for h in hulls]
    data["x"] = [p[0] for p in positions]
    data["y"] = [p[1] for p in positions]

    # Individual resistances, if already computed
    comps = [getattr(h, "resistance_components", None) for h in hulls]
    for key in ("Rw", "Rf"):
        if all(c is not None and key in c for c in comps):
            data[key] = [c[key] for c in comps]

    args = dict(stw=op_point.stw, rho=rho, g=g, nu=nu, **data)
    args.update(kwargs)

    dRw = [[0. for _ in range(n)] for _ in range(n)]
    dRf = [0. for _ in range(n)]
    if "Rw" in process:
        dRw = process["Rw"](**args)
    if "Rf" in process:
        dRf = process["Rf"](**args)

    dR_hull = [sum(dRw[i]) + dRf[i] for i in range(n)]
    return {"dRw": dRw, "dRf": dRf, "dR_hull": dR_hull, "dR": sum(dR_hull)}
