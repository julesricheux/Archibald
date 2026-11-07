# -*- coding: utf-8 -*-
"""
COMMON FUNCTIONS OF THE MULTIHULL INTERACTION METHODS
"""

#%% DEPENDENCIES

import numpy as onp

import archibald.numpy as np
import archibald.toolbox.units as u
from archibald.modeling.interpolation import InterpolatedModel

EPS = 1e-12

#%% QUADRATURE

def nodes_unit(n):
    """Equispaced nodes on [0, 1] (``n`` is made odd for Simpson's rule)."""
    n = n + 1 if n % 2 == 0 else n
    return np.linspace(0., 1., n)


def simpson_weights(n):
    """Composite Simpson weights on [0, 1] matching :func:`nodes_unit`."""
    n = n + 1 if n % 2 == 0 else n
    h = 1. / (n - 1)
    return np.array([
        h / 3. * (1. if i in (0, n - 1) else (4. if i % 2 else 2.))
        for i in range(n)
    ])

#%% FLOW AND GEOMETRY PARAMETERS

def k_0(
        stw,
        g,
        **kwargs,
    ):
    r"""Fundamental wave number of the Kelvin wave system.

    .. math::
        k_0 = \frac{g}{U^2}

    Parameters
    ----------
    stw : float, Speed through water [kt].
    g : float, Gravitational acceleration [m/s²].

    Returns
    -------
    float, Wave number k0 [1/m].
    """
    Vms = np.softplus(stw * u.kt, beta=1e3) + 1e-6
    return g / Vms ** 2


def froude_number(
        stw,
        L,
        g,
        **kwargs,
    ):
    r"""Length Froude number of a hull.

    .. math::
        F_n = \frac{U}{\sqrt{g\,L}}

    Parameters
    ----------
    stw : float, Speed through water [kt].
    L : float, Reference length [m].
    g : float, Gravitational acceleration [m/s²].

    Returns
    -------
    float, Froude number [-].
    """
    return np.softplus(stw * u.kt, beta=1e3) / np.sqrt(g * L + 1e-6)


def separation_to_length(
        y_i,
        y_j,
        Lwl_i,
        Lwl_j,
        **kwargs,
    ):
    r"""Separation to length ratio S/L of a pair of hulls.

    .. math::
        \frac{S}{L} = \frac{|y_i - y_j|}{(L_{WL,i} + L_{WL,j})/2}

    S is the centre-to-centre transverse distance (smooth near 0 for gradients).

    Parameters
    ----------
    y_i, y_j : float, Transverse positions of the hulls [m].
    Lwl_i, Lwl_j : float, Waterline lengths of the hulls [m].

    Returns
    -------
    float, S/L [-].
    """
    S = np.sqrt((y_i - y_j) ** 2 + EPS)
    return S / (0.5 * (Lwl_i + Lwl_j))

#%% FACTOR TABLES

def table_model(
        fn_grid,
        sl_grid,
        table,
        outside=1.,
        method="bspline",
        **kwargs,
    ):
    r"""Interpolated model of an interference factor table, function of Fn and S/L.

    The table is interpolated by an ``InterpolatedModel``, with one more column
    at S/L = 2 max(``sl_grid``) holding ``outside``: far from the table the factor
    tends to "no interference" (assumption, the measurements stop at the last S/L).
    Outside the grid, Fn and S/L are held at the edges of the grid.

    The CasADi B-spline needs at least 4 nodes on each axis: a shorter axis falls
    back to ``method="linear"``, which is not smooth at the nodes.

    Parameters
    ----------
    fn_grid : list of float, Increasing Froude numbers of the table [-].
    sl_grid : list of float, Increasing S/L values of the table [-].
    table : list of list, ``table[i][j]`` is the factor at ``fn_grid[i]``, ``sl_grid[j]`` [-].
    outside : float, Value far from the table in S/L [-].
    method : str, ``"bspline"`` or ``"linear"``.

    Returns
    -------
    function, ``factor(Fn, SL)`` [-].
    """
    fn = onp.asarray(fn_grid, dtype=float)
    sl = onp.append(onp.asarray(sl_grid, dtype=float), 2. * sl_grid[-1])
    values = onp.column_stack([onp.asarray(table, dtype=float), onp.full(len(fn), float(outside))])
    if min(len(fn), len(sl)) < 4:
        method = "linear"
    model = InterpolatedModel(
        x_data_coordinates={"Fn": fn, "SL": sl},
        y_data_structured=values,
        method=method,
        fill_value=None,
    )

    def factor(Fn, SL):
        # Held explicitly: with numeric inputs, the linear method would extrapolate.
        return model({
            "Fn": np.minimum(np.maximum(Fn, fn[0]), fn[-1]),
            "SL": np.minimum(np.maximum(SL, sl[0]), sl[-1]),
        })

    return factor

#%% ROBUSTNESS HELPERS (HULLS CLOSE TO OR OUT OF THE WATER)

def smooth_clip(
        x,
        lo,
        hi,
        eps=1e-4,
        **kwargs,
    ):
    r"""Smooth, differentiable and overflow free clipping of x in [lo, hi].

    .. math::
        \text{clip}(x) = \frac{1}{2}\left(lo + hi
            + \sqrt{(x - lo)^2 + \epsilon} - \sqrt{(x - hi)^2 + \epsilon}\right)

    Equal to x inside the interval, to lo / hi outside (up to a smoothing of
    order :math:`\sqrt{\epsilon}`), whatever the magnitude of x.

    Parameters
    ----------
    x : float, Value to clip.
    lo, hi : float, Bounds, lo < hi.
    eps : float, Smoothing parameter.

    Returns
    -------
    float, Clipped value.
    """
    return 0.5 * (lo + hi + np.sqrt((x - lo) ** 2 + eps) - np.sqrt((x - hi) ** 2 + eps))


def immersion_weight(
        T,
        T_kiss=0.01,
        **kwargs,
    ):
    r"""Smooth weight of a hull going out of the water.

    .. math::
        w = \frac{T^2}{T^2 + T_{kiss}^2}

    It is 1 for a normally immersed hull (T >> T_kiss), and tends to 0 when
    the hull lifts out of the water, so that the interference of a hull that is
    not (or hardly) wet vanishes smoothly. A negative or zero draught (hull
    out of the water) gives 0.

    Parameters
    ----------
    T : float, Draught of the hull [m].
    T_kiss : float, Draught at which the weight is 1/2 [m].

    Returns
    -------
    float, Immersion weight in [0, 1].
    """
    Tp = 0.5 * (T + np.sqrt(T ** 2 + 1e-8))   # smooth ReLU, ~5e-5 m at T = 0
    return Tp ** 2 / (Tp ** 2 + T_kiss ** 2)
