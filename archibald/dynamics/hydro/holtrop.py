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

def compute_Rf_holtrop(
        stw,
        Lbp,
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
        Wsa (float): Wetted surface area (m^2)
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
        RfMin = (1 + k - 2*sigmaK) * Cf_hull(Re_corr+10.) * (0.5 * rho * Wsa * (Vms ** 2))
        RfMax = (1 + k + 2*sigmaK) * Cf_hull(Re_corr+10.) * (0.5 * rho * Wsa * (Vms ** 2))
        return RfMin, RfMax
    
    Rf = (1+k) * Cf_hull(Re_corr+10.) * (0.5 * rho * Wsa * (Vms ** 2))
    
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
        Wsa,
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
        RaMin = 0.5 * rho * Wsa * (Vms ** 2) * (CA - 2*sigmaCA)
        RaMax = 0.5 * rho * Wsa * (Vms ** 2) * (CA + 2*sigmaCA)
        return RaMin, RaMax
    
    Ra = 0.5 * rho * Wsa * (Vms ** 2) * CA
    return Ra


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
        'Wsa': 4500.0,
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