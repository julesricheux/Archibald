# -*- coding: utf-8 -*-
"""
HOLTROP METHOD FUNCTIONS

Created: -/-/2023
Last update: 11/02/2025

@author: Jules Richeux
@university: ENSA Nantes, FRANCE
@contributors: -
"""

#%% DEPENDENCIES

import archibald.numpy as np
import archibald.toolbox.units as u

from archibald.dynamics.hydro.common import Cf_hull
from archibald.toolbox.math_utils import ReLU

#%% FUNCTIONS

# RESISTANCE COMPONENTS

def compute_Rf_holtrop(
        stw,
        Lbp,
        Lwl,
        volume,
        Bwl,
        T,
        Aws,
        Cp,
        lcb,
        Csternchoice,
        nu,
        rho,
        uInterval=False,
        **kwargs,
    ):
    """
    Calculate the frictional resistance of a ship using the ITTC '57 method
    and Holtrop form factor (1+k).
    
    Parameters:
    -----------
        stw (float): Speed through water in knots
        Lbp (float): Length between perpendiculars (m)
        Lwl (float): Length at waterline (m)
        volume (float): Volume displacement (m^3)
        Bwl (float): Beam at waterline (m)
        T (float): Draft (m)
        Aws (float): Wetted surface area (m^2)
        Cp (float): Prismatic coefficient
        lcb (float): LCB position as a % of L forward of midships L/2
        Csternchoice (int): Stern form (1=transom, 2=V-shaped, 3=U-shaped, 4=Spoon)
        nu (float): Kinematic viscosity of water (m^2/s)
        rho (float): Density of water (kg/m^3)
        uInterval (bool): Whether to return min/max uncertainty intervals
        
    Returns:
    --------
        float or tuple: Frictional resistance of the ship (N)
    """
    Vms = stw * u.kt

    if Csternchoice == 1:
        Cstern = -25.
    elif Csternchoice == 2:
        Cstern = -10.
    elif Csternchoice == 3:
        Cstern = 0.
    elif Csternchoice == 4:
        Cstern = 10.
    else:
        # Default to standard U-shape if invalid
        Cstern = 0.
    
    c14 = 1 + 0.011 * Cstern
    Lr = Lwl * (1 - Cp + 0.06 * Cp * lcb / (4 * Cp - 1))

    k = 0.93 + 0.487118 * c14 * (Bwl/Lwl)**1.06806 * (T/Lwl)**0.46106 * \
        (Lwl/Lr)**0.121563 * (Lwl**3/volume)**0.36486 * (1-Cp)**(-0.604247) - 1
           
    sigmaK = k * 0.046 # std deviation 4.6%
    
    Re_corr = (np.softplus(Vms, beta=1e6) * Lwl) / nu
    
    if uInterval:
        RfMin = (1 + k - 2*sigmaK) * Cf_hull(Re_corr+10.) * (0.5 * rho * Aws * (Vms ** 2))
        RfMax = (1 + k + 2*sigmaK) * Cf_hull(Re_corr+10.) * (0.5 * rho * Aws * (Vms ** 2))
        return RfMin, RfMax
    
    Rf = (1+k) * Cf_hull(Re_corr+10.) * (0.5 * rho * Aws * (Vms ** 2))
    
    return Rf


def compute_Rw_holtrop(
        stw,
        Lwl,
        Lbp,
        Bwl,
        T,
        volume,
        Abt,
        Cp,
        Cwp,
        Atr,
        lcb,
        hB,
        Cx,
        ie,
        rho,
        g,
        **kwargs,
    ):
    """
    Calculates the wave-making resistance of a ship in calm water.
    """
    Vms = stw * u.kt
    Fr = np.softplus(Vms, beta=1e6) / (np.sqrt(g * Lbp))
    
    ### RW FOR FROUDE < 0.40
    Lr = Lwl * ((1 - Cp) - ((0.06 * Cp * lcb) / (4 * Cp - 1)))
    
    # Replaced casadi if_else with numpy where for differentiable branching
    c7 = np.where(
        Bwl / Lbp < 0.11,
        0.229577 * ((Bwl / Lbp) ** 0.3333),
        np.where(
            Bwl / Lbp <= 0.25,
            Bwl / Lbp,
            0.5 - (0.0625 * (Lbp / Bwl))
        )
    )
    
    c1 = 2223105 * (c7 ** 3.78613) * ((T / Bwl) ** 1.07961) * ((90 - ie) ** (-1.37565))
    c3 = ((0.56 * Abt) ** 1.5) / ((Bwl * T) * ((0.31 * (np.sqrt(Abt))) + (T - hB)))
    c2 = np.exp(-1.89 * (np.sqrt(c3)))
    c5 = 1 - (0.8 * (Atr / (Bwl * T * Cx)))
    
    c16 = np.where(
        Cp < 0.8,
        (8.07981 * Cp) - (13.8673 * (Cp ** 2)) + (6.984388 * (Cp ** 3)),
        1.73014 - (0.7067 * Cp)
    )
    
    m1 = (0.014047 * (Lbp / T)) - ((1.75254 * (volume ** (1 / 3))) / Lbp) - (4.79323 * (Bwl / Lbp)) - c16
    
    l = np.where(
        Lbp / Bwl < 12,
        (1.446 * Cp) - (0.03 * (Lbp / Bwl)),
        (1.446 * Cp) - 0.36
    )
    
    L3_V = (Lbp ** 3) / volume
    c15 = np.where(
        L3_V < 512,
        -1.69385,
        np.where(
            L3_V < 1726.91,
            -1.69385 + (((Lbp / (volume ** (1 / 3))) - 8) / 2.36),
            0.
        )
    )
    
    m4 = c15 * 0.4 * (np.exp(-0.034 * (Fr ** -3.29)))
    d_ = -0.9
    
    Rw_to_040 = c1 * c2 * c5 * volume * rho * g * (np.exp((m1 * (Fr ** d_)) + m4 * (np.cos(l * (Fr ** (-2))))))

    # RW FOR 0.40 < FROUDE < 0.55
    rwo_ = c1 * c2 * c5 * volume * rho * g * (np.exp((m1 * (0.40 ** d_)) + m4 * (np.cos(l * (Fr ** (-2))))))
    rwo__ = c1 * c2 * c5 * volume * rho * g * (np.exp((m1 * (0.55 ** d_)) + m4 * (np.cos(l * (Fr ** (-2))))))
    
    Rw_from_040_to_055 = rwo_ + (((10 * Fr) - 4) * ((rwo__ - rwo_) / 1.224))
        
    # RW FOR FROUDE > 0.55
    c17 = (6919.3 * (Cx ** (-1.3346))) * ((volume / (Lbp ** 3)) ** 2.00977) * (((Lbp / Bwl) - 2) ** 1.40692)
    m3 = (-7.2035 * ((Bwl / Lbp) ** 0.326869)) * ((T / Bwl) ** 0.605375)
    
    Rw_from_055 = c17 * c2 * c5 * volume * rho * g * (np.exp((m3 * (Fr ** d_) + (m4 * (np.cos(l * (Fr ** -2)))))))

    is_below_040 = np.where(Fr <= 0.40, 1., 0.)
    is_below_055 = np.where(Fr <= 0.55, 1., 0.)
    
    Rw = Rw_to_040 * is_below_040 + \
         Rw_from_040_to_055 * (1. - is_below_040) * is_below_055 + \
         Rw_from_055 * (1. - is_below_040) * (1. - is_below_055)
         
    return Rw


def compute_Rb_holtrop(
        stw,
        T,
        hB,
        Abt,
        Bulbchoice,
        rho,
        g,
        **kwargs,
    ):
    """
    Calculates the resistance due to bulbous bow using Holtrop's method.
    """
    Vms = stw * u.kt
    
    if Bulbchoice == 1:
        Fri = Vms / (np.sqrt((g * (T - hB - (0.25 * (np.sqrt(Abt))))) + (0.15 * (Vms ** 2))))
        pb = (0.56 * (np.sqrt(Abt))) / (T - (1.5 * hB))
        Rb = 0.11 * (np.exp(((-3) * (pb ** (-2)))) * (Fri ** 3) * (Abt ** 1.5) * rho * g) / (1 + (Fri ** 2))
        return Rb
    else:
        return 0.


def compute_Rtr_holtrop(
        stw,
        Ttr,
        Atr,
        Bwl,
        Cwp,
        rho,
        g,
        **kwargs,
    ):
    """
    Calculate transom resistance using the Holtrop-Mennen method.
    """
    Vms = stw * u.kt
    
    Fr_T = Vms / np.sqrt(g * Ttr)
    
    ctr = 0.2 * (1 - (0.2 * Fr_T))
    ctr_ReLu = ReLU(ctr)
    Rtr = 0.5 * rho * (Vms ** 2) * Atr * ctr_ReLu
    
    return Rtr


def compute_Ra_holtrop(
        stw,
        Lwl,
        Bwl,
        T,
        Cb,
        hB,
        rho,
        Aws,
        Abt,
        uInterval=False,
        **kwargs,
    ):
    """
    Calculates the model-ship correlation resistance RA.
    """
    Vms = stw * u.kt
    
    CA = 0.00675 * (Lwl + 100)**(-1/3) - 0.00064
    sigmaCA = 0.00021
        
    if uInterval:
        RaMin = 0.5 * rho * Aws * (Vms ** 2) * (CA - 2*sigmaCA)
        RaMax = 0.5 * rho * Aws * (Vms ** 2) * (CA + 2*sigmaCA)
        return RaMin, RaMax
    
    Ra = 0.5 * rho * Aws * (Vms ** 2) * CA
    return Ra

# PROPELLER COEFFICIENTS
def c_8(
        Lbp,
        Bwl,
        T,
        D,
        Aws,
        **kwargs,
    ):
    r"""Coefficient c8 used in wake fraction prediction.

    According to :cite:`holtropStatisticalReAnalysisResistance1984`, p. 273:

    .. math::
        c_8 = \begin{cases}
            \frac{B \cdot S}{L \cdot D \cdot T} & \text{if } \frac{B}{T} < 5 \\
            \frac{S \left(7 \frac{B}{T} - 25\right)}{L \cdot D \left(\frac{B}{T} - 3\right)}
            & \text{otherwise}
        \end{cases}

    Parameters
    ----------
    Lbp : float, Length between perpendiculars [m].
    Bwl : float, Waterline beam [m].
    T : float, Mean moulded draught [m].
    D : float, Propeller diameter [m].
    Aws : float, Wetted surface area [m²].

    Returns
    -------
    float, Coefficient c8 [-].
    """
    return np.where(
        Bwl / T < 5.0,
        Bwl * Aws / (Lbp * D * T),
        Aws * (7.0 * (Bwl / T) - 25.0) / (Lbp * D * ((Bwl / T) - 3.0))
    )


def c_9(
        Lbp,
        Bwl,
        T,
        D,
        Aws,
        **kwargs,
    ):
    r"""Coefficient c9 used in wake fraction prediction.

    According to :cite:`holtropStatisticalReAnalysisResistance1984`, p. 273:

    .. math::
        c_9 = \begin{cases}
            c_8 & \text{if } c_8 < 28 \\
            32 - \frac{16}{c_8 - 24} & \text{otherwise}
        \end{cases}

    Parameters
    ----------
    Lbp : float, Length between perpendiculars [m].
    Bwl : float, Waterline beam [m].
    T : float, Mean moulded draught [m].
    D : float, Propeller diameter [m].
    Aws : float, Wetted surface area [m²].

    Returns
    -------
    float, Coefficient c9 [-].
    """
    _c8 = c_8(Lbp=Lbp, Bwl=Bwl, T=T, D=D, Aws=Aws)
    return np.where(
        _c8 < 28.0,
        _c8,
        32.0 - 16.0 / (_c8 - 24.0)
    )


def c_11(
        T,
        D,
        **kwargs,
    ):
    r"""Coefficient c11 used in wake fraction prediction.

    According to :cite:`holtropStatisticalReAnalysisResistance1984`, p. 273:

    .. math::
        c_{11} = \begin{cases}
            \frac{T}{D} & \text{if } \frac{T}{D} < 2 \\
            \frac{1}{12} \left(\frac{T}{D}\right)^3 + \frac{4}{3} & \text{otherwise}
        \end{cases}

    Parameters
    ----------
    T : float, Mean moulded draught [m].
    D : float, Propeller diameter [m].

    Returns
    -------
    float, Coefficient c11 [-].
    """
    return np.where(
        T / D < 2.0,
        T / D,
        0.0833333 * (T / D) ** 3.0 + 1.33333
    )


def c_19(
        Cp,
        Cb,
        Cx,
        **kwargs,
    ):
    r"""Coefficient c19 used in wake fraction prediction.

    According to :cite:`holtropStatisticalReAnalysisResistance1984`, p. 273:

    .. math::
        c_{19} = \begin{cases}
            \frac{0.12997}{0.95 - C_B} - \frac{0.11056}{0.95 - C_P}
            & \text{if } C_P < 0.7 \\
            \frac{0.18567}{1.3571 - C_X} - 0.71276 + 0.38648\,C_P
            & \text{otherwise}
        \end{cases}

    Parameters
    ----------
    Cp : float, Prismatic coefficient [-].
    Cb : float, Block coefficient [-].
    Cx : float, Midship section coefficient [-].

    Returns
    -------
    float, Coefficient c19 [-].
    """
    return np.where(
        Cp < 0.7,
        0.12997 / (0.95 - Cb) - 0.11056 / (0.95 - Cp),
        0.18567 / (1.3571 - Cx) - 0.71276 + 0.38648 * Cp
    )


def c_20(
        Cstern,
        **kwargs,
    ):
    r"""Coefficient c20 used in wake and thrust deduction prediction.

    According to :cite:`holtropStatisticalReAnalysisResistance1984`, p. 273:

    .. math::
        c_{20} = 1 + 0.015\,C_{\text{stern}}

    Parameters
    ----------
    Cstern : float, Stern shape parameter [-].

    Returns
    -------
    float, Coefficient c20 [-].
    """
    return 1.0 + 0.015 * Cstern


def C_P1(
        Cp,
        lcb,
        **kwargs,
    ):
    r"""Modified prismatic coefficient Cp1 used in wake fraction prediction.

    According to :cite:`holtropStatisticalReAnalysisResistance1984`, p. 273:

    .. math::
        C_{P1} = 1.45\,C_P - 0.315 - 0.0225\,\text{lcb}

    Parameters
    ----------
    Cp : float, Prismatic coefficient [-].
    lcb : float, Longitudinal centre of buoyancy, as % of Lbp from midship [-].

    Returns
    -------
    float, Modified prismatic coefficient Cp1 [-].
    """
    return 1.45 * Cp - 0.315 - 0.0225 * lcb


def C_V(
        stw,
        Lbp,
        Bwl,
        T,
        volume,
        Cp,
        Cstern,
        lcb,
        rho,
        g,
        **kwargs,
    ):
    r"""Viscous resistance coefficient CV.

    According to :cite:`holtropStatisticalReAnalysisResistance1984`, p. 272:

    .. math::
        C_V = \left(1 + k\right) C_F + C_A

    where the form factor :math:`(1+k)` is given by
    :cite:`holtrop1982approximate`, p. 166:

    .. math::
        1 + k = 0.93 + 0.487118\,c_{14}
            \left(\frac{B}{L}\right)^{1.06806}
            \left(\frac{T}{L}\right)^{0.46106}
            \left(\frac{L}{L_R}\right)^{0.121563}
            \left(\frac{L^3}{\nabla}\right)^{0.36486}
            \left(1 - C_P\right)^{-0.604247}

    and the correlation allowance :math:`C_A` by
    :cite:`holtropStatisticalReAnalysisResistance1984`, p. 272:

    .. math::
        C_A = 0.00675\,(L + 100)^{-1/3} - 0.00064

    Parameters
    ----------
    stw : float, Speed through water [kt].
    Lbp : float, Length between perpendiculars [m].
    Bwl : float, Waterline beam [m].
    T : float, Mean moulded draught [m].
    volume : float, Displaced volume [m³].
    Cp : float, Prismatic coefficient [-].
    Cstern : float, Stern shape parameter [-].
    lcb : float, Longitudinal centre of buoyancy, as % of Lbp from midship [-].
    rho : float, Water density [kg/m³].
    g : float, Gravitational acceleration [m/s²].

    Returns
    -------
    float, Viscous resistance coefficient CV [-].
    """
    c14 = 1.0 + 0.011 * Cstern
    Lr = Lbp * (1 - Cp + 0.06 * Cp * lcb / (4 * Cp - 1))
    one_plus_k = (
        0.93
        + 0.487118 * c14
        * (Bwl / Lbp) ** 1.06806
        * (T / Lbp) ** 0.46106
        * (Lbp / Lr) ** 0.121563
        * (Lbp ** 3 / volume) ** 0.36486
        * (1 - Cp) ** (-0.604247)
    )
    CA = 0.00675 * (Lbp + 100) ** (-1/3) - 0.00064
    Re = (np.softplus(stw * u.kt, beta=1e6) * Lbp) / 1.2e-6
    return one_plus_k * Cf_hull(Re + 10.0) + CA


def w_single(
        stw,
        Lbp,
        Bwl,
        T,
        T_A,
        Aws,
        Cp,
        Cb,
        Cx,
        D,
        Cstern,
        volume,
        lcb,
        rho,
        g,
        **kwargs,
    ):
    r"""Wake fraction prediction for single screw ships.

    According to :cite:`holtropStatisticalReAnalysisResistance1984`, p. 273:

    .. math::
        w = c_9 c_{20} C_V \frac{L}{T_A} \left( 0.050776 + 0.93405 c_{11}
            \frac{C_V}{1 - C_{P1}} \right) + 0.27915 c_{20}
            \sqrt{\frac{B}{L \left(1 - C_{P1}\right)}} + c_{19} c_{20}

    Parameters
    ----------
    stw : float, Speed through water [kt].
    Lbp : float, Length between perpendiculars [m].
    Bwl : float, Waterline beam [m].
    T : float, Mean moulded draught [m].
    T_A : float, Draught at aft perpendicular [m].
    Cp : float, Prismatic coefficient [-].
    Cb : float, Block coefficient [-].
    Cx : float, Midship section coefficient [-].
    D : float, Propeller diameter [m].
    Cstern : float, Stern shape parameter [-].
    volume : float, Displaced volume [m³].
    lcb : float, Longitudinal centre of buoyancy, as % of Lbp from midship [-].
    rho : float, Water density [kg/m³].
    g : float, Gravitational acceleration [m/s²].

    Returns
    -------
    float, Wake fraction w [-].
    """
    _CV  = C_V(stw=stw, Lbp=Lbp, Bwl=Bwl, T=T, volume=volume, Cp=Cp, Cstern=Cstern, lcb=lcb, rho=rho, g=g)
    _c9  = c_9(Lbp=Lbp, Bwl=Bwl, T=T, D=D, Aws=Aws)
    _c11 = c_11(T=T, D=D)
    _c19 = c_19(Cp=Cp, Cb=Cb, Cx=Cx)
    _c20 = c_20(Cstern=Cstern)
    _CP1 = C_P1(Cp=Cp, lcb=lcb)

    return (
        _c9
        * _c20
        * _CV
        * (Lbp / T_A)
        * (0.050776 + 0.93405 * _c11 * (_CV / (1 - _CP1)))
        + 0.27915 * _c20 * np.sqrt(Bwl / (Lbp * (1 - _CP1)))
        + _c19 * _c20
    )


def eta_R_single(
        Cp,
        lcb,
        Ae_Ao,
        **kwargs,
    ):
    r"""Relative rotative efficiency prediction for single screw ships.

    According to :cite:`holtrop1982approximate`, p. 168:

    .. math::
        \eta_R = 0.9922 - 0.05908 \frac{A_E}{A_O}
            + 0.07424 \left(C_P - 0.0225\,\text{lcb}\right)

    Parameters
    ----------
    Cp : float, Prismatic coefficient [-].
    lcb : float, Longitudinal centre of buoyancy, as % of Lbp from midship [-].
    Ae_Ao : float, Expanded blade area ratio [-].

    Returns
    -------
    float, Relative rotative efficiency eta_R [-].
    """
    return (
        0.9922
        - 0.05908 * (Ae_Ao)
        + 0.07424 * (Cp - 0.0225 * lcb)
    )


def w_single_open_stern(
        stw,
        Lbp,
        Bwl,
        T,
        D,
        Cb,
        rho,
        g,
        **kwargs,
    ):
    r"""Wake fraction prediction for single screw ships with open stern.

    Applied to slender, fast ships. According to :cite:`holtrop1982approximate`, p. 169:

    .. math::
        w = 0.3 C_B + 10 C_V C_B - 0.23 \frac{D}{\sqrt{BT}}

    Parameters
    ----------
    stw : float, Speed through water [kt].
    Lbp : float, Length between perpendiculars [m].
    Bwl : float, Waterline beam [m].
    T : float, Mean moulded draught [m].
    D : float, Propeller diameter [m].
    Cb : float, Block coefficient [-].
    rho : float, Water density [kg/m³].
    g : float, Gravitational acceleration [m/s²].

    Returns
    -------
    float, Wake fraction w [-].
    """
    return (
        0.3 * Cb
        + 10.0 * C_V(stw, Lbp, Bwl, T, rho, g) * Cb
        - 0.23 * D / np.sqrt(Bwl * T)
    )


def t_single_open_stern(**kwargs):
    r"""Thrust deduction fraction for single screw ships with open stern.

    Applied to slender, fast ships. According to :cite:`holtrop1982approximate`, p. 169,
    a constant value is used:

    .. math::
        t = 0.10

    Returns
    -------
    float, Thrust deduction fraction t = 0.1 [-].
    """
    return 0.1


def eta_R_single_open_stern(**kwargs):
    r"""Relative rotative efficiency for single screw ships with open stern.

    Applied to slender, fast ships. According to :cite:`holtrop1982approximate`, p. 168,
    a constant value is used:

    .. math::
        \eta_R = 0.98

    Returns
    -------
    float, Relative rotative efficiency eta_R = 0.98 [-].
    """
    return 0.98


def w_twin(
        stw,
        Lbp,
        Bwl,
        T,
        D,
        Cb,
        rho,
        g,
        **kwargs,
    ):
    r"""Wake fraction prediction for twin screw ships.

    According to :cite:`holtrop1982approximate`, p. 169:

    .. math::
        w = 0.3095 C_B + 10 C_V C_B - 0.23 \frac{D}{\sqrt{BT}}

    Parameters
    ----------
    stw : float, Speed through water [kt].
    Lbp : float, Length between perpendiculars [m].
    Bwl : float, Waterline beam [m].
    T : float, Mean moulded draught [m].
    D : float, Propeller diameter [m].
    Cb : float, Block coefficient [-].
    rho : float, Water density [kg/m³].
    g : float, Gravitational acceleration [m/s²].

    Returns
    -------
    float, Wake fraction w [-].
    """
    return (
        0.3095 * Cb
        + 10.0 * C_V(stw, Lbp, Bwl, T, rho, g) * Cb
        - 0.23 * D / np.sqrt(Bwl * T)
    )


def t_twin(
        Bwl,
        T,
        D,
        Cb,
        **kwargs,
    ):
    r"""Thrust deduction fraction prediction for twin screw ships.

    According to :cite:`holtrop1982approximate`, p. 169:

    .. math::
        t = 0.325 C_B - 0.1885 \frac{D}{\sqrt{BT}}

    Parameters
    ----------
    Bwl : float, Waterline beam [m].
    T : float, Mean moulded draught [m].
    D : float ,Propeller diameter [m].
    Cb : float ,Block coefficient [-].

    Returns
    -------
    float, Thrust deduction fraction t [-].
    """
    return (
        0.325 * Cb
        - 0.1885 * D / np.sqrt(Bwl * T)
    )


def eta_R_twin(
        Cp,
        lcb,
        P,
        D,
        **kwargs,
    ):
    r"""Relative rotative efficiency prediction for twin screw ships.

    According to :cite:`holtrop1982approximate`, p. 168:

    .. math::
        \eta_R = 0.9737 + 0.111 \left(C_P - 0.0225\,\text{lcb}\right)
            + 0.06325 \frac{P}{D}

    Parameters
    ----------
    Cp : float, Prismatic coefficient [-].
    lcb : float, Longitudinal centre of buoyancy, as % of Lbp from midship [-].
    P : float, Propeller pitch [m].
    D : float, Propeller diameter [m].

    Returns
    -------
    float, Relative rotative efficiency eta_R [-].
    """
    return (
        0.9737
        + 0.111 * (Cp - 0.0225 * lcb)
        + 0.06325 * P / D
    )


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    # ---------------------------------------------------------
    # VERIFICATION & PLOTTING (MOCK COMMERCIAL HULL)
    # ---------------------------------------------------------
    
    env_params = {
        'g': 9.81,         # m/s^2
        'rho': 1025.0,     # kg/m^3
        'nu': 1.189e-6,    # m^2/s (15 deg C seawater)
    }
    
    # Mock dimensions for a general cargo vessel / tanker
    hull_params = {
        'Lbp': 150.0,
        'Lwl': 155.0,
        'Bwl': 22.0,
        'T': 8.5,
        'volume': 19000.0,
        'Aws': 4500.0,
        'Cp': 0.65,
        'Cwp': 0.75,
        'Cx': 0.98,
        'Cb': 0.637,
        'lcb': -1.5,       # % of L forward of midships (negative means aft)
        'ie': 15.0,        # Half angle of entrance (degrees)
        'Csternchoice': 1, # Transom
        'Atr': 6.0,        # Transom immersed area [m^2]
        'Ttr': 0.5,        # Transom draft [m]
        'Bulbchoice': 1,
        'Abt': 12.0,       # Bulb cross-sectional area [m^2]
        'hB': 3.0,         # Height of centroid of bulb area [m]
        'uInterval': False
    }

    # Define Froude sweep (Holtrop generally valid for Fn < 0.40)
    fr_array = np.linspace(0.0, 1.0, 1000)
    
    Rf_list, Rw_list, Rb_list, Rtr_list, Ra_list = [], [], [], [], []
    
    for fr in fr_array:
        # Update speed dependent variables
        v_ms = fr * np.sqrt(env_params['g'] * hull_params['Lbp'])
        hull_params['stw'] = v_ms / u.kt
        
        # Compute individual components
        Rf  = compute_Rf_holtrop(**hull_params, **env_params)
        Rw  = compute_Rw_holtrop(**hull_params, **env_params)
        Rb  = compute_Rb_holtrop(**hull_params, **env_params)
        Rtr = compute_Rtr_holtrop(**hull_params, **env_params)
        Ra  = compute_Ra_holtrop(**hull_params, **env_params)
        
        Rf_list.append(Rf)
        Rw_list.append(Rw)
        Rb_list.append(Rb)
        Rtr_list.append(Rtr)
        Ra_list.append(Ra)
        
    Rf_arr  = np.array(Rf_list)
    Rw_arr  = np.array(Rw_list)
    Rb_arr  = np.array(Rb_list)
    Rtr_arr = np.array(Rtr_list)
    Ra_arr  = np.array(Ra_list)

    # ---------------------------------------------------------
    # STACKED AREA PLOT
    # ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6), dpi=120)
    
    layer1 = Rf_arr
    layer2 = layer1 + Rw_arr
    layer3 = layer2 + Rtr_arr
    layer4 = layer3 + Rb_arr
    layer5 = layer4 + Ra_arr
    
    ax.fill_between(fr_array, 0, layer1, color='#1f77b4', alpha=0.6, label='Frictional ($R_f + 1+k$)')
    ax.plot(fr_array, layer1, color='#1f77b4', linewidth=1)
    
    ax.fill_between(fr_array, layer1, layer2, color='#ff7f0e', alpha=0.6, label='Wave ($R_w$)')
    ax.plot(fr_array, layer2, color='#ff7f0e', linewidth=1)

    ax.fill_between(fr_array, layer2, layer3, color='#d62728', alpha=0.6, label='Transom ($R_{tr}$)')
    ax.plot(fr_array, layer3, color='#d62728', linewidth=1)
    
    ax.fill_between(fr_array, layer3, layer4, color='#9467bd', alpha=0.6, label='Bulbous Bow ($R_b$)')
    ax.plot(fr_array, layer4, color='#9467bd', linewidth=1)
    
    ax.fill_between(fr_array, layer4, layer5, color='#8c564b', alpha=0.6, label='Model Correlation ($R_a$)')
    ax.plot(fr_array, layer5, color='#8c564b', linewidth=1)
    
    ax.set_title("Holtrop & Mennen\nComponents of Resistance vs. Froude Number", fontsize=14, pad=15)
    ax.set_xlabel('Froude Number ($F_n$)', fontsize=12)
    ax.set_ylabel('Total Resistance (N)', fontsize=12)
    
    ax.grid(True, linestyle='--', alpha=0.6, zorder=0)
    ax.set_axisbelow(True)
    
    ax.legend(loc='upper left', fontsize=10, framealpha=0.9)
    plt.xlim([fr_array.min(), fr_array.max()])
    plt.ylim(bottom=0)
    plt.tight_layout()
    plt.show()