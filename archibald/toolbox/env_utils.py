# -*- coding: utf-8 -*-
"""
Created 2024-10-18

Useful environmental tools for Archibald

@author: Jules Richeux
@university: ENSA Nantes
@company: NEOLINE

"""

#%% DEPENDENCIES

import archibald.numpy as np


#%% FUNCTIONS

def grad_wind(tws0, z, z0=10.0, a=0.12):
    """
    Computes true wind speed at a given height following : Heier, Siegfried (2005).
    Grid Integration of Wind Energy Conversion Systems. Chichester: John Wiley &
    Sons. p. 45. ISBN 978-0-470-86899-7.

    Parameters
    ----------
    tws0 (float): Reference true wind speed at z=z0
    z (float): Unit should be the same as z.
    z0 (float): Reference height. The default is 10.0 (in meters)
    a (float): Hellman exponent, governing the gradient intensity. The default is 0.12 (according to ITTC)
    
    Other references (according to "Renewable energy: technology, economics, and
                      environment" by Martin Kaltschmitt, Wolfgang Streicher,
                      Andreas Wiese, (Springer, 2007, ISBN 3-540-70947-9,
                      ISBN 978-3-540-70947-3), page 55) :
    Unstable air above open water surface : 0.06
    Neutral air above open water surface : 0.10
    Unstable air above flat open coast : 0.11
    Neutral air above flat open coast : 0.16
    Stable air above open water surface : 0.27
    Unstable air above human inhabited areas : 0.27
    Neutral air above human inhabited areas : 0.34
    Stable air above flat open coast : 0.40
    Stable air above human inhabited areas : 0.60

    Returns
    -------
    float: True wind speed at z. Unit is the same as tws0.
    
    """
    # return tws0 * (z/z0)**a
    return tws0 * (np.abs(z)/z0)**a


