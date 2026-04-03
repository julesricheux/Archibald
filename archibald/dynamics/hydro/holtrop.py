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

import casadi as ca

from archibald.toolbox.dyn_utils import Cf_hull
from archibald.toolbox.math_utils import ReLU

#%% FUNCTIONS

def compute_Rf_holtrop(Vms,
                       Lbp,
                       Loa,
                       Lwl,
                       volume,
                       Bwl,
                       T,
                       Wsa,
                       Cp,
                       lcb,
                       Csternchoice,
                       nu,
                       rho,
                       origin,
                       uInterval,
                       ):
    """
    Calculate the frictional resistance of a ship using the ITTC '57 method.
    
    Params:
        Vms (float): speed (m/s)
        Lbp (float): Length between perpendiculars (m)
        Loa (float): Length overall (m)
        Lwl (float): Length at waterline (m)
        T (float): Draft (m)
        S (float): Wetted surface area (m^2)
        V (float): Volume (m^3)
        Csternchoice (int): Choice of stern form (1=transom, 2=V-shaped, 3=U-shaped, 4=Spoon-shaped)
        M (float): Mass displacement (kg)
        rho (float): Density of water (kg/m^3)
        
    Returns:
        float: Frictional resistance of the ship (N)
    """

    if Csternchoice == 1:
        Cstern = -25.
    elif Csternchoice == 2:
        Cstern = -10.
    elif Csternchoice == 3:
        Cstern = 0.
    elif Csternchoice == 4:
        Cstern = 10.
    else:
        print("Invalid choice for the stern shape")
    

    c14 = 1 + 0.011*Cstern
    Lr = Lwl*(1 - Cp + 0.06*Cp*lcb/(4*Cp-1))

    k = .93 + .487118*c14*(Bwl/Lwl)**1.06806 * (T/Lwl)**.46106 * (Lwl/Lr)**.121563 *\
           (Lwl**3/volume)**.36486 * (1-Cp)**(-.604247) - 1
           
    sigmaK = k * 0.046 # std deviation 4.6%
    
    Re_ReLU = (ReLU(Vms) * Lwl) / nu
    # ACF = 5.1e-4
    
    if uInterval:
        RfMin = (1 + k - 2*sigmaK) * Cf_hull(Re_ReLU+10.) * (0.5 * rho * Wsa * (Vms ** 2))
        RfMax = (1 + k + 2*sigmaK) * Cf_hull(Re_ReLU+10.) * (0.5 * rho * Wsa * (Vms ** 2))
        
        return RfMin, RfMax
    
    Rf = (1+k) * Cf_hull(Re_ReLU+10.) * (0.5 * rho * Wsa * (Vms ** 2))
    
    return Rf


def compute_Rw_holtrop(Vms,
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
                       ):
    """
    Calculates the wave-making resistance of a ship in calm water.
    
    Params:
        Vms (float): speed (m/s)
        Lwl (float): The length of the waterline in meters.
        Lbp (float): The length between perpendiculars in meters.
        B (float): The beam of the ship in meters.
        T (float): The draft of the ship in meters.
        Abt (float): The area of the bulbous bow in square meters.
        Cp (float): The prismatic coefficient.
        Cwp (float): The coefficient of the waterplane area.
        Atr (float): The transom area in square meters.
        lcb (float): lcb in %
        hB (float): Bulb height (m)
        Cx (float): The midship coefficient.
        rho (float): The density of water in kg/m^3.
        g (float): The acceleration due to gravity in m/s^2.
    
    Returns:
        float: The added resistance of the ship in calm water in (N)
    """
    
    Fr = ReLU(Vms) / (np.sqrt(g * Lbp)) + 1e-1
    
    ### RW FOR FROUDE < 0.40
    
    Lr = Lwl * ((1 - Cp) - ((0.06 * Cp * lcb) / (4 * Cp - 1)))
    
    c7 = ca.if_else(
        Bwl / Lbp < 0.11,
        0.229577 * ((Bwl / Lbp) ** 0.3333),  # if B/L < 0.11
        ca.if_else(
            Bwl / Lbp <= 0.25,
            Bwl / Lbp, # if B/L between 0.11 and 0.25
            0.5 - (0.0625 * (Lbp / Bwl)) # if B/L > 0.25
        )
    )
    
    c1 = 2223105 * (c7 ** 3.78613) * ((T / Bwl) ** 1.07961) * ((90 - ie) ** (-1.37565))
    c3 = ((0.56 * Abt) ** 1.5) / ((Bwl * T) * ((0.31 * (np.sqrt(Abt))) + (T - hB)))
    c2 = np.exp(-1.89 * (np.sqrt(c3)))
    c5 = 1 - (0.8 * (Atr / (Bwl * T * Cx)))
    
    c16 = ca.if_else(
        Cp < 0.8,
        (8.07981 * Cp) - (13.8673 * (Cp ** 2)) + (6.984388 * (Cp ** 3)),
        1.73014 - (0.7067 * Cp)
    )
    
    m1 = (0.014047 * (Lbp / T)) - ((1.75254 * (volume ** (1 / 3))) / Lbp) - (4.79323 * (Bwl / Lbp)) - c16
    
    l = ca.if_else(
        Lbp / Bwl < 12,
        (1.446 * Cp) - (0.03 * (Lbp / Bwl)),
        (1.446 * Cp) - 0.36
    )
    
    c15 = ca.if_else(
        (Lbp ** 3) / volume < 512,
        -1.69385,
        ca.if_else(
            (Lbp ** 3) / volume < 1726.91,
            -1.69385 + (((Lbp / (volume ** (1 / 3))) - 8) / 2.36),
            0.
        )
    )
    
    m4 = c15 * 0.4 * (np.exp(-0.034 * (Fr ** -3.29)))
    d_ = -0.9
    
    Rw_to_040 = c1 * c2 * c5 * volume * rho * g * (np.exp((m1 * (Fr ** d_)) + m4 * (np.cos(l * (Fr ** (-2))))))

    # RW FOR 0.40 < FROUDE < 0.55
    
    d_ = -0.9
    rwo_ = c1 * c2 * c5 * volume * rho * g * (np.exp((m1 * (0.44 ** d_)) + m4 * (np.cos(l * (Fr ** (-2))))))
    rwo__ = c1 * c2 * c5 * volume * rho * g * (np.exp((m1 * (0.55 ** d_)) + m4 * (np.cos(l * (Fr ** (-2))))))
    
    Rw_from_040_to_055 = rwo_ + (((10 * Fr) - 4) * ((rwo__ - rwo_) / 1.5))
        
    
    # RW FOR FROUDE > 0.55
    
    c17 = (6919.3 * (Cx ** (-1.3346))) * ((volume / (Lbp ** 3)) ** 2.00977) * (((Lbp / Bwl) - 2) ** 1.40692)
    m3 = (-7.2035 * ((Bwl / Lbp) ** 0.326869)) * ((T / Bwl) ** 0.605375)
    
    Rw_from_055 = c17 * c2 * c5 * volume * rho * g * (np.exp((m3 * (Fr ** d_) + (m4 * (np.cos(l * (Fr ** -2)))))))

    is_below_040 = ca.if_else(Fr <= 0.40, 1., 0.)
    is_below_055 = ca.if_else(Fr <= 0.55, 1., 0.)
    
    return Rw_to_040 * is_below_040 +\
           Rw_from_040_to_055 * (1.-is_below_040) * is_below_055 +\
           Rw_from_055 * (1.-is_below_040) * (1.-is_below_055)


def compute_Rb_holtrop(Vms,
                       T,
                       hB,
                       Abt,
                       Bulbchoice,
                       rho,
                       g,
                       ):
    """
    Calculates the resistance due to bulbous bow using Holtrop's method.

    Parameters:
    hB (float): height of bulbous bow [m]
    Bulbchoice (int): 1 for bulbous bow, 0 for no bulbous bow
    """
    
    if Bulbchoice == 1:
        Fri = Vms / (np.sqrt((g * (T - hB - (0.25 * (np.sqrt(Abt))))) + (0.15 * (Vms ** 2))))
        pb = (0.56 * (np.sqrt(Abt))) / (T - (1.5 * hB))
        Rb = 0.11 * (np.exp(((-3) * (pb ** (-2)))) * (Fri ** 3) * (Abt ** 1.5) * rho * g) / (1 + (Fri ** 2))
        
        return Rb
    
    elif Bulbchoice == 0:
        return 0
    
    else:
        print("Invalid choice for the bulbous bow")


def compute_Rtr(Vms,
                Ttr,
                Atr,
                Bwl,
                Cwp,
                rho,
                g,
                ):
    """
    Calculate transom resistance using the Holtrop-Mennen method.
    """
    
    # Transom additionnal resistance (Holtrop&Mennen, 1978)
    Fr_T = Vms / np.sqrt(g * Ttr)
    
    # ctr = np.max((0.2 * (1 - (0.2 * Fr_T)), 0.0))
    
    # if Fr_T < 5:
    ctr = 0.2 * (1 - (0.2 * Fr_T))
    ctr_ReLu = ReLU(ctr)
    Rtr = 0.5 * rho * (Vms ** 2) * Atr * ctr_ReLu
    
    return Rtr


def compute_Ra_holtrop(Vms,
                       Lwl,
                       Bwl,
                       T,
                       Cb,
                       hB,
                       rho,
                       Wsa,
                       Abt,
                       uInterval,
                       ):
    """
    Calculates the model-ship correlation resistance RA.
    """
    CA = 0.00675 * (Lwl +100)**(-1/3) - 0.00064
    sigmaCA = 0.00021  # std deviation 0.00021
        
    if uInterval:
        RaMin = 0.5 * rho * Wsa * (Vms ** 2) * (CA - 2*sigmaCA)
        RaMax = 0.5 * rho * Wsa * (Vms ** 2) * (CA + 2*sigmaCA)
        
        return RaMin, RaMax
    
    Ra = 0.5 * rho * Wsa * (Vms ** 2) * CA
    
    return Ra


