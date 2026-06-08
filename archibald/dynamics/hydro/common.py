# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 21:51:50 2026

@author: jules
"""

import archibald.numpy as np
import archibald.toolbox.units as u

def Cf_hull(
        Re: float,
        **kwargs,
    ):
    """
    Computes the friction-drag coefficient of a bare hull following ITTC78
    https://www.ittc.info/media/8017/75-02-03-014.pdf
    
    Modified for continuity at Re=0.

    Parameters
    ----------
    Re (float): Hull Reynolds number

    Returns
    -------
    float: Bare hull friction-drag coefficient

    """
    
    Re_corr = np.softplus(Re - 100) + 100 + 1. # threshold Re above 100 to avoid dividing by 0
    
    return 0.075/(np.log10(Re_corr) - 2)**2

def transom_resistance(
        Vms,
        FrT,
        Atr,
        rho,
        **kwargs,
    ):
    # TODO find reference.
    """
    Calculate transom resistance.
    """
    ctr = 0.2 * (1 - (0.2 * FrT)) # transom drag coefficient
    ctr_ReLu = np.softplus(ctr, beta=1e3)
    Rtr = 0.5 * rho * (Vms ** 2) * Atr * ctr_ReLu
    
    return Rtr