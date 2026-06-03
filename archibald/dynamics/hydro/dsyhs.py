# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 20:50:45 2026

@author: jules
"""

import archibald.numpy as np
from archibald.modeling import InterpolatedModel
from archibald.dynamics.hydro.common import Cf_hull

# Delft table data 
# Coefficients a0 to a7
_keunig_coefs = {
    'Fr': np.array([   0.15,    0.20,    0.25,    0.30,    0.35,    0.40,    0.45,    0.50,    0.55,    0.60,    0.65,    0.70,    0.75]),
    'a0': np.array([-0.0005, -0.0003, -0.0002, -0.0009, -0.0026, -0.0064, -0.0218, -0.0388, -0.0347, -0.0361, +0.0008, +0.0108, +0.1023]),
    'a1': np.array([+0.0023, +0.0059, -0.0156, +0.0016, -0.0567, -0.4034, -0.5261, -0.5986, -0.4764, +0.0037, +0.3728, -0.1238, +0.7726]),
    'a2': np.array([-0.0086, -0.0064, +0.0031, +0.0337, +0.0446, -0.1250, -0.2945, -0.3038, -0.2361, -0.2960, -0.3667, -0.2026, +0.5040]),
    'a3': np.array([-0.0015, +0.0070, -0.0021, -0.0285, -0.1091, +0.0273, +0.2485, +0.6033, +0.8726, +0.9661, +1.3957, +1.1282, +1.7867]),
    'a4': np.array([+0.0061, +0.0014, -0.0070, -0.0367, -0.0707, -0.1341, -0.2428, -0.0430, +0.4219, +0.6123, +1.0343, +1.1836, +2.1934]),
    'a5': np.array([+0.0010, +0.0013, +0.0148, +0.0218, +0.0914, +0.3578, +0.6293, +0.8332, +0.8990, +0.7534, +0.3230, +0.4973, -1.5479]),
    'a6': np.array([+0.0001, +0.0005, +0.0010, +0.0015, +0.0021, +0.0045, +0.0081, +0.0106, +0.0096, +0.0100, +0.0072, +0.0038, -0.0115]),
    'a7': np.array([+0.0052, -0.0020, -0.0043, -0.0172, -0.0078, +0.1115, +0.2086, +0.1336, -0.2272, -0.3352, -0.4632, -0.4477, -0.0977])
}

_keunig_interpolators = {}

for i in range(8):
    _keunig_interpolators[f"a{i}"] = InterpolatedModel(
        x_data_coordinates=_keunig_coefs["Fr"],
        y_data_structured=_keunig_coefs[f"a{i}"],
        method="bspline"
    )
    

def compute_Rf_dsyhs(
        Re,
        rho,
        Aws,
        stw,
    ):
    """
    Calculates the frctionnal resistance (Rf) based on the 
    Delft Systematic Yacht Hull series polynomial equation.
    
    Parameters:
    -----------
    ...
    
    Returns:
    --------
    ...
    """
    V = stw * u.kt # speed through water in m/s
    
    Rf = 0.5 * Cf_hull(Re) * rho * Aws * V**2
    return Rf


def compute_Rw_dsyhs(
        Fr,
        rho,
        g,
        volume,
        Lwl,
        Bwl,
        T,
        Awp,
        Cp,
        Cx,
        LCB_fpp,
        LCF_fpp,
    ):
    """
    Calculates the residuary resistance (Rw) based on the 
    Delft Systematic Yacht Hull series polynomial equation.
    
    Parameters:
    -----------
    Fr      : float, Froude number (valid range roughly 0.15 - 0.75)
    volume  : float, Canoe body volume displacement (∇) [m^3]
    rho     : float, Density of water [kg/m^3]
    g       : float, Acceleration due to gravity [m/s^2]
    LCB_fpp : float, Longitudinal Center of Buoyancy from fpp [m]
    Lwl     : float, Waterline length [m]
    Cp      : float, Prismatic coefficient
    Awp     : float, Waterplane area [m^2]
    Bwl     : float, Waterline beam [m]
    LCF_fpp : float, Longitudinal Center of Flotation from fpp [m]
    T       : float, Canoe body draft [m]
    Cx      : float, Midship section coefficient
    
    Returns:
    --------
    Rw     : float, Residuary resistance [N]
    """
    
    a = []

    for i in range(8):
        a.append(
            _keunig_interpolators[f"a{i}"](Fr)
        )
    
    # Calculate dimensional ratios
    vol_ratio = (volume**(1/3)) / Lwl
    
    # Calculate terms inside the parentheses
    term1 = (a[1] * LCB_fpp / Lwl) + (a[2] * Cp) + (a[3] * (volume**(2/3)) / Awp) + (a[4] * Bwl / Lwl)
    term2 = (a[5] * LCB_fpp / LCF_fpp) + (a[6] * Bwl / T) + (a[7] * Cx)
    
    # Calculate the non-dimensional resistance coefficient
    resistance_coeff = a[0] + (term1 * vol_ratio) + (term2 * vol_ratio)
    
    # Convert back to dimensional resistance in Newtons
    Rw = resistance_coeff * volume * rho * g
    
    return Rw


if __name__=="__main__":
    import archibald.toolbox.units as u
    # VERIFICATION
    # --- Example Usage based on the image's check values ---
    # The image notes: R_Rc / (∇ * p * g) = 0.00649 at Fr = 0.35
    # Assuming arbitrary hull dimensions that yield exactly this coefficient to prove the math:
    
    # Mock dimensions # TODO find correct mock dimensions for verification
    hull_params = {
        'Fr': 0.35,
        'rho': 1025.0,     # kg/m^3 (seawater)
        'g': 9.81,         # m/s^2
        'volume': 6.0,      # m^3 (approx 6 tons)
        'LCB_fpp': 5.2,    # m
        'Lwl': 10.0,      # m
        'Cp': 0.55,
        'Awp': 18.0,       # m^2
        'Bwl': 3.0,       # m
        'LCF_fpp': 5.4,    # m
        'T': 0.8,        # m
        'Cx': 0.70,
    }
    
    res = compute_Rw_dsyhs(**hull_params)
    print(f"Boat speed: {hull_params['Fr']*np.sqrt(hull_params['g']*hull_params['Lwl'])/u.kt:.2f} kts")
    print(f"Calculated Residuary Resistance: {res:.2f} N")
    
    # PLOT
    import matplotlib.pyplot as plt
    x_coords = _keunig_coefs["Fr"]
    x_fine = np.linspace(
        _keunig_coefs["Fr"].min(), 
        _keunig_coefs["Fr"].max(), 
        100
    )
    
    for i in range(8):
        # 3. Evaluate the interpolator
        
        y_data = _keunig_coefs[f"a{i}"]
        y_interp = _keunig_interpolators[f"a{i}"](x_fine)
    
        # 4. Plot results
        plt.figure(figsize=(8, 5))
        plt.plot(x_coords, y_data, 'ok', label="Original Data Points")
        plt.plot(x_fine, y_interp, '-', label="B-Spline Interpolation")
        plt.legend()
        plt.grid()
        plt.title(f"1D Interpolation for coef a{i}")
        plt.xlabel("Froude")
        # plt.ylabel("y")
        plt.show()