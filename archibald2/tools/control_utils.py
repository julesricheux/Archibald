# -*- coding: utf-8 -*-
"""
Created 10/01/2025
Last update: 10/01/2025

Useful control tools for archibald

@author: Jules Richeux
@university: ENSA Nantes, FRANCE
@contributors: -

"""

#%% DEPENDENCIES

import archibald2.numpy as np
import archibald2.tools.units as u

#%% FUNCTIONS

def twist_law(x, max_twist, power=0.912):
    return max_twist * x**power


def camber_corr_from_aoa(aoa,
                         aoa_base=25.,
                         epsilon=0.03):
    
    tau = - aoa_base*u.rad/np.log(epsilon)
    
    return (1 - np.exp(-aoa*u.rad/tau))**4


def twist_corr_from_awa(awa,
                        awa_base=180.,
                        epsilon=0.01):
    
    tau = - awa_base*u.rad/np.log(epsilon)
    
    return 1 - np.exp(-awa*u.rad/tau)


def aoa_corr_from_awa(awa,
                      aoa_base=20.,
                      awa_base=25.,
                      epsilon=0.33):
    
    tau = - awa_base*u.rad/np.log(epsilon)
     
    awa_rad = awa * u.rad
    
    laminar = (1 - np.exp(-awa_rad/tau)) * aoa_base * u.rad
    turbulent = (1 - 2*aoa_base*u.rad/np.pi) * awa_rad - np.pi/2 + 2*aoa_base*u.rad
    
    return np.fmax(laminar, turbulent) / u.rad


def boom_correction_from_awa(awa,
                             a=15.,
                             b=2.,
                             fac=1./10.):
    
    return np.sin(np.pi * a**(-(awa*u.rad)**b))*fac
    # return np.sin(a**(-(awa0*u.rad)**2 + np.log(np.pi))/np.log(a))*2./20.
    
    
def upwind_sine(x, k=4.):
    
    offset = np.pi**(1/k)
    
    return np.sin((offset - x*u.rad/2.)**k)


def velocity_invert(stw, a=10.):
    
    return a / stw
    

def velocity_init(tws, twa):
    
    return tws*(np.sind(twa)**2 + twa/180.) / 3. + 1.


def leeway_init(stw, tws, twa):
    
    return np.fmin(upwind_sine(twa, k=3.) * velocity_invert(stw) * tws**(2/3), 
                   twa-1e-15)


def fin_init(stw, tws, twa):
    
    return upwind_sine(twa, k=2.) * 15.