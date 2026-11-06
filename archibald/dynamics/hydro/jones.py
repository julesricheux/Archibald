# -*- coding: utf-8 -*-
"""
Hull lateral force from low aspect ratio wing theory.

The underwater hull is taken as a wing of span T, mirrored in the free surface,
so of aspect ratio k = 2T/Lwl:

    CL  = pi/2 * k * beta        on the area Lwl * T
    CDi = CL^2 / (pi * k)        i.e. half of the leading edge suction is lost: CDi = CL * beta / 2

Sources
-------
R. T. Jones, "Properties of low-aspect-ratio pointed wings at speeds below and
above the speed of sound", NACA Report 835, 1946.
The use of k = 2T/L as the aspect ratio of a ship hull is the basis of Inoue et
al. (1981), see ``archibald.dynamics.hydro.inoue``.

The theory is linear and has no thickness, viscous or free surface effect, and
it gives no yaw moment here: the force acts amidships in ``Hull``.
"""
import archibald.numpy as np


def compute_L_jones(Lwl, T, rho, V, beta, **kwargs):
    """
    Lift [N], normal to the flow, from Jones (1946).

    Parameters
    ----------
    Lwl : waterline length [m]
    T : draft [m]
    rho : water density [kg/m^3]
    V : speed through water [m/s]
    beta : leeway [rad]
    """
    k = 2. * T / (Lwl + 1e-12)

    return 0.5 * rho * V ** 2 * Lwl * T * np.pi / 2. * k * beta


def compute_Ri_jones(Lwl, T, rho, V, beta, **kwargs):
    """
    Induced resistance [N], along the flow, from Jones (1946).

    Parameters
    ----------
    Same as ``compute_L_jones``.
    """
    k = 2. * T / (Lwl + 1e-12)
    CL = np.pi / 2. * k * beta

    return 0.5 * rho * V ** 2 * Lwl * T * CL ** 2 / (np.pi * k + 1e-12)
