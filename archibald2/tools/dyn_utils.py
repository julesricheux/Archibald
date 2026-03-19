# -*- coding: utf-8 -*-
"""
Created 04/07/2023
Last update: 24/12/2023

Useful fluid dynamics tools for archibald

@author: Jules Richeux
@university: ENSA Nantes, FRANCE
@contributors: -

"""

#%% DEPENDENCIES

import archibald2.numpy as np
# import scipy.interpolate as itrp # legacy import, deprecated


#%% FUNCTIONS


def Cf_hull(Re):
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
    
    Re_val = np.abs(Re) + 1.
    Re_sg = np.sign(Re)
    
    return Re_sg * 0.075/((np.log10(Re_val) - 2)**2)


def holtrop_correction_factor(Fr):
    """
    Correction of Holtrop resistance from medium to high Froude numbers.
    
    Data fitting on MAURIC's extrapolation of Force Technologies' towing tank tests
    on N136 at displacement 11 556 t vs Holtrop84
    """
    a=21.935539054532047
    b=-5.094109422609588
    c=1.2054582838282384
    
    return (c + b * Fr + a * Fr**2.)


#%% LEGACY

# def Cd_cambered_plate(alpha, camber):
#     """
#     Interpolates the drag coefficient of a cambered plate
#     following the data given by Glenn Research Center, NASA
#     https://www1.grc.nasa.gov/beginners-guide-to-aeronautics/foilsimstudent/#

#     Parameters
#     ----------
#     alpha (float): Angle of attack of the plate in degrees. Should be between 0. and 20.
#     camber (float): Relative camber of the plate. Should be between 0. and .15

#     Returns
#     -------
#     float: Interpolated drag coefficient of the given plate in the given conditions.

#     """
    
#     aoas = np.array([0,5,10,15,20])
#     cambers = np.array([0., .05, .10, .15])
    
#     if alpha <= 0.0:
#         alpha = 1e-8
#     elif alpha >= 20.0:
#         alpha = 20.0 - 1e-8
        
#     if camber <= 0.0:
#         camber = 1e-8
#     elif alpha >= 0.15:
#         camber = 0.15 - 1e-8
        
#     values = np.array([[.0188, .0543, .0978, .2168],
#                       [.0260, .0609, .1217, .3440],
#                       [.0565, .0957, .1739, .5202],
#                       [.2630, .1608, .2609, .6192],
#                       [.2760, .2400, .3500, .4275]])

#     values.reshape((20))

#     f = itrp.RegularGridInterpolator((aoas, cambers), values, method='cubic')

#     return f((alpha, camber))


# def Cf_airfoil(Re):
#     return 0.074/Re**0.2 - 1742/Re