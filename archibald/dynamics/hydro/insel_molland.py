# -*- coding: utf-8 -*-
"""
INSEL-MOLLAND CATAMARAN INTERFERENCE FACTORS (TABULATED)

Experimental wave and viscous interference factors of demihulls, function of
the Froude number and of the separation to length ratio S/L. The tables are not
shipped: they must be digitised from the literature or from tank tests.
"""

#%% DEPENDENCIES

import archibald.numpy as np

from archibald.dynamics.hydro.multihull_common import (
    froude_number, separation_to_length, interp_table,
)

#%% RESISTANCE COMPONENTS

def compute_dRw_insel_molland(
        stw,
        Lwl,
        y,
        g,
        Rw,
        fn_grid,
        sl_grid,
        tau_table,
        **kwargs,
    ):
    r"""Wave interference from tabulated experimental factors.

    According to :cite:`insel1992investigation`, the catamaran resistance is
    expressed with an interference factor :math:`\tau` applied to the wave
    resistance of the demihull, function of the Froude number and of the
    separation to length ratio S/L:

    .. math::
        C_T = (1 + \beta k)\,C_F + \tau\,C_W

    No table is shipped: they must be digitised from :cite:`insel1992investigation`
    and :cite:`molland1994resistance` or provided from tank tests. For n > 2 hulls
    the factors are superposed pairwise (assumption, the data only covers two hulls):

    .. math::
        \Delta R_{w,ij} = R_{w,i}\,\big(\tau(Fn_i, S_{ij}/L_{ij}) - 1\big)

    Parameters
    ----------
    stw : float, Speed through water [kt].
    Lwl : list of float, Waterline length of each hull [m].
    y : list of float, Transverse position of each hull [m].
    g : float, Gravitational acceleration [m/s²].
    Rw : list of float, Individual wave resistance of each hull [N].
    fn_grid, sl_grid : list of float, Increasing grids of Fn and S/L [-].
    tau_table : list of list, ``tau_table[i][j]`` at (fn_grid[i], sl_grid[j]) [-].

    Returns
    -------
    list of list, Interference matrix dRw[i][j] [N], zero diagonal (not
    necessarily symmetric if the hulls differ).
    """
    n = len(Lwl)
    dRw = [[0. for _ in range(n)] for _ in range(n)]
    for i in range(n):
        Fn = froude_number(stw=stw, L=Lwl[i], g=g)
        for j in range(n):
            if i != j:
                SL = separation_to_length(y[i], y[j], Lwl[i], Lwl[j])
                tau = interp_table(Fn, SL, fn_grid, sl_grid, tau_table)
                dRw[i][j] = Rw[i] * (tau - 1.)
    return dRw


def compute_dRf_insel_molland(
        stw,
        Lwl,
        y,
        g,
        Rf,
        fn_grid,
        sl_grid,
        visc_table,
        **kwargs,
    ):
    r"""Viscous interference from tabulated experimental factors.

    The viscous interference of :cite:`insel1992investigation` changes the form
    factor from :math:`1+k` to :math:`1+\beta k`, :math:`\beta` accounting for the
    pressure field (:math:`\phi`) and the velocity augmentation between hulls
    (:math:`\sigma`). As the individual ``Rf`` already holds :math:`1+k`, the table
    gives the multiplier :math:`m = (1+\beta k)/(1+k)`:

    .. math::
        \Delta R_{f,i} = R_{f,i}\sum_{j\neq i}\big(m(Fn_i, S_{ij}/L_{ij}) - 1\big)

    Parameters
    ----------
    stw : float, Speed through water [kt].
    Lwl : list of float, Waterline length of each hull [m].
    y : list of float, Transverse position of each hull [m].
    g : float, Gravitational acceleration [m/s²].
    Rf : list of float, Individual frictional resistance of each hull [N].
    fn_grid, sl_grid : list of float, Increasing grids of Fn and S/L [-].
    visc_table : list of list, Multiplier m at (fn_grid[i], sl_grid[j]) [-].

    Returns
    -------
    list of float, Viscous interference of each hull dRf[i] [N].
    """
    n = len(Lwl)
    dRf = [0. for _ in range(n)]
    for i in range(n):
        Fn = froude_number(stw=stw, L=Lwl[i], g=g)
        for j in range(n):
            if i != j:
                SL = separation_to_length(y[i], y[j], Lwl[i], Lwl[j])
                m = interp_table(Fn, SL, fn_grid, sl_grid, visc_table)
                dRf[i] = dRf[i] + Rf[i] * (m - 1.)
    return dRf
