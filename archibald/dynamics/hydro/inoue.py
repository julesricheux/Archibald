# -*- coding: utf-8 -*-
"""
Hull lateral force from the linear sway derivative of Inoue, Hirano and Kijima.

    Y' = Y / (1/2 rho L T U^2) = Y'beta * beta
    Y'beta = pi/2 * k + 1.4 * Cb * B/L        with k = 2T/L

Sources
-------
S. Inoue, M. Hirano and K. Kijima, "Hydrodynamic derivatives on ship manoeuvring",
International Shipbuilding Progress 28(321), 1981.
The formula was read from its figure and text as reproduced in a Korean review of
empirical derivatives (J. Soc. Naval Architects of Korea 57(6), 2020), not from the
original paper. Its N'beta and nonlinear terms could not be checked, so they are
not implemented: the force acts amidships in ``Hull``, with no yaw moment.

Fitted on merchant ship models. Use outside of that range is an extrapolation.
"""
import archibald.numpy as np


def _Y_inoue(Lwl, Bwl, T, Cb, rho, V, beta):
    """Force [N] normal to the hull centreline."""
    k = 2. * T / (Lwl + 1e-12)
    Y_beta = np.pi / 2. * k + 1.4 * Cb * Bwl / (Lwl + 1e-12)

    return 0.5 * rho * V ** 2 * Lwl * T * Y_beta * beta


def compute_L_inoue(Lwl, Bwl, T, Cb, rho, V, beta, **kwargs):
    """
    Lift [N], normal to the flow, from Inoue et al. (1981).

    Parameters
    ----------
    Lwl, Bwl, T : waterline length, waterline beam and draft [m]
    Cb : block coefficient [-]
    rho : water density [kg/m^3]
    V : speed through water [m/s]
    beta : leeway [rad]
    """
    return _Y_inoue(Lwl, Bwl, T, Cb, rho, V, beta) * np.cos(beta)


def compute_Ri_inoue(Lwl, Bwl, T, Cb, rho, V, beta, **kwargs):
    """
    Added resistance [N], along the flow: the part of the force normal to the
    hull centreline that lies along the flow. Inoue's longitudinal derivative is
    not included, as it could not be checked.

    Parameters
    ----------
    Same as ``compute_L_inoue``.
    """
    return _Y_inoue(Lwl, Bwl, T, Cb, rho, V, beta) * np.sin(beta)
