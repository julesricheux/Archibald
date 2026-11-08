# -*- coding: utf-8 -*-
"""
Hull lateral force and yaw moment from the approximate formulae of Kijima and Nakiri.

With no rate of turn, their equations (4) and (5) reduce to:

    X' = X'uu * cos(beta)^2
    Y' = Y'beta * beta + Y'bb * beta * |beta|
    N' = N'beta * beta + N'bb * beta * |beta|

with X', Y' = (X, Y) / (1/2 rho L d U^2), N' = N / (1/2 rho L^2 d U^2), and the
coefficients of their equations (6) and (7) below.

Sources:
    K. Kijima and Y. Nakiri, "On the practical prediction method for ship manoeuvring
    characteristics", MARSIM 2003, Conference Proceedings. Equations were read from the
    paper. Database of 15 ships (container ship, bulk carrier, tanker, cargo ship...)
    in 48 loading conditions: use on other hull forms is an extrapolation.

Assumptions of this implementation:
    - kappa is not defined in the available text of the paper. It is taken as 2d/L, as
      in Inoue et al. (1981), from which the formulae descend.
    - Cwa and Cpa are the waterplane and prismatic coefficients of the aft hull,
      between A.P. and s.s.5. ``Hull`` does not compute them: by default, the
      coefficients of the whole hull are used. Pass Cwa and Cpa when they are known.
    - The paper places the origin of the moment at the centre of gravity. It is taken
      amidships here, which ``Hull`` uses as the reference point.
    - The cos(beta)^2 of X' is not applied to the resistance of the hull.
"""
import archibald.numpy as np


def _abs(x):
    """Smooth absolute value, for beta * |beta| to stay differentiable at zero."""
    return np.sqrt(x ** 2 + 1e-15)


def _aft_hull_parameters(Lwl, Bwl, T, Cp, Cwp, Cpa=None, Cwa=None):
    """Equations (6): e'a, sigma_a and K. Cpa and Cwa default to the whole hull values."""
    Cpa = Cp if Cpa is None else Cpa
    Cwa = Cwp if Cwa is None else Cwa

    ea = Lwl / (Bwl + 1e-12) * (1. - Cpa)
    ea_prime = ea / np.sqrt(1. / 4. + 1. / (Bwl / (T + 1e-12)) ** 2)
    sigma_a = (1. - Cwa) / (1. - Cpa + 1e-12)
    K = (1. / (ea_prime + 1e-12) + 1.5 / (Lwl / (Bwl + 1e-12)) - 0.33) * (0.95 * sigma_a + 0.40)

    return ea_prime, sigma_a, K


def _Y_kijima(Lwl, Bwl, T, Cb, Cp, Cwp, rho, V, beta, Cpa=None, Cwa=None):
    """Force [N] normal to the hull centreline, equations (5) and (7)."""
    _, sigma_a, _ = _aft_hull_parameters(Lwl, Bwl, T, Cp, Cwp, Cpa, Cwa)
    kappa = 2. * T / (Lwl + 1e-12)

    Y_beta = np.pi / 2. * kappa + 1.9257 * (Cb * Bwl / (Lwl + 1e-12)) * sigma_a
    Y_bb = -1.199 * Cb * sigma_a + 1.05

    return 0.5 * rho * V ** 2 * Lwl * T * (Y_beta * beta + Y_bb * beta * _abs(beta))


def compute_L_kijima(Lwl, Bwl, T, Cb, Cp, Cwp, rho, V, beta, Cpa=None, Cwa=None, **kwargs):
    """
    Lift [N], normal to the flow, from Kijima and Nakiri (2003).

    Args:
        Lwl, Bwl, T: waterline length, waterline beam and draft [m]
        Cb, Cp, Cwp: block, prismatic and waterplane coefficients [-]
        rho: water density [kg/m^3]
        V: speed through water [m/s]
        beta: leeway [rad]
        Cpa, Cwa: prismatic and waterplane coefficients of the aft hull [-], whole hull by default.
    """
    return _Y_kijima(Lwl, Bwl, T, Cb, Cp, Cwp, rho, V, beta, Cpa, Cwa) * np.cos(beta)


def compute_Ri_kijima(Lwl, Bwl, T, Cb, Cp, Cwp, rho, V, beta, Cpa=None, Cwa=None, **kwargs):
    """
    Added resistance [N], along the flow: the part of Y that lies along the flow.

    Their X' has no other drift term than cos(beta)^2.

    Args:
        Same as ``compute_L_kijima``.
    """
    return _Y_kijima(Lwl, Bwl, T, Cb, Cp, Cwp, rho, V, beta, Cpa, Cwa) * np.sin(beta)


def compute_N_kijima(Lwl, Bwl, T, Cb, Cp, Cwp, rho, V, beta, Cpa=None, Cwa=None, **kwargs):
    """
    Yaw moment [N.m] about the reference point, equations (5) and (7).

    Same sign as the force times a lever arm forward: it tends to increase the
    leeway.

    Args:
        Same as ``compute_L_kijima``.
    """
    ea_prime, _, K = _aft_hull_parameters(Lwl, Bwl, T, Cp, Cwp, Cpa, Cwa)
    kappa = 2. * T / (Lwl + 1e-12)
    x = T * (1. - Cb) / (Bwl + 1e-12) * ea_prime * K

    N_beta = kappa * (15.0668 * x ** 2 - 23.819 * x + 1.802)
    N_bb = 43.857 * x ** 2 - 3.671 * x + 0.086

    return 0.5 * rho * V ** 2 * Lwl ** 2 * T * (N_beta * beta + N_bb * beta * _abs(beta))
