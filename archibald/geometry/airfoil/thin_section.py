# -*- coding: utf-8 -*-
"""
Created on Mon Sep 22 10:39:31 2025

@author: jrich
"""

from archibald.geometry.airfoil.airfoil_families import get_kulfan_parameters
from archibald.geometry import Airfoil, KulfanAirfoil
import archibald.numpy as np

from typing import Union

_valid_results = ['coordinates', 'parameters', 'kulfan_parameters', 'airfoil', 'kulfan_airfoil']

def tall(array):
    return np.reshape(array, (-1, 1))

def wide(array):
    return np.reshape(array, (1, -1))

# def wide(vector):
#     return np.tile(np.reshape(vector, (1, dims[1])), (dims[0], 1))

# def tall(vector):
#     return np.tile(np.reshape(vector, (dims[0], 1)), (1, dims[1]))

def leading_edge_camber(xc, mc):
    D = tall(-((xc**3 - 4*xc + 2) * mc) / ((xc - 1)**3 * xc))
    
    return D


def camberline(x, xc, mc):
    # if (xc <= 0).any() or (xc >= 1).any():
    #     raise ValueError("xc must be strictly between 0 and 1.")

    # denom = (xc - 1)**2 * xc**2
    
    # A = (mc - 2 * xc * mc) / denom
    # B = ((3 * xc**2 - 1) * mc) / denom
    # C = ((2 - 3 * xc) * mc) / ((xc - 1)**2 * xc)
    # A = tall((mc - 2 * xc * mc) / denom)
    # B = tall(((3 * xc**2 - 1) * mc) / denom)
    # C = tall(((2 - 3 * xc) * mc) / ((xc - 1)**2 * xc))
    
    # x = wide(x)
    
    # return A @ x**3 + B @ x**2 + C @ x
    
    denom = (xc - 1)**3 * xc**2

    A = tall(((xc**2 - 3*xc + 1) * mc) / denom)
    B = tall(((-2*xc**3 + 3*xc**2 + 4*xc - 2) * mc) / denom)
    C = tall(((xc**4 + 2*xc**3 - 8*xc**2 + xc + 1) * mc) / denom)
    D = leading_edge_camber(xc, mc)
    
    x = wide(x)
    
    return A @ x**4 + B @ x**3 + C @ x**2 + D @ x


def generate_base_airfoil(
    baseAirfoil: str = "e376",
    n: int = 100,
    ):
    
    return Airfoil(baseAirfoil)
    

def thin_airfoil(
    xc: Union[float, np.ndarray],
    mc: Union[float, np.ndarray],
    baseAirfoil: str = "e376",
    n_points_per_side: int = 100,
    result: str = 'kulfan_airfoil',
    ):
    
    data = []
    
    x = np.concatenate(
        (
            np.cosspace(1., 0., n_points_per_side),
            np.cosspace(0., 1., n_points_per_side)[1:],
        )
    )
    Y = camberline(x, xc, mc)
    
    afModel = Airfoil(baseAirfoil).repanel(n_points_per_side)
    
    th = afModel.local_thickness(x)
    base = np.copy(wide(th))
    base[:, :n_points_per_side-1] *= -1.
    base /= 2.
    
    x = wide(x)
    
    for i in range(Y.shape[0]):
        
        y = Y[i, :]
        
        coordinates = np.concatenate((x, base+y)).T
        
        if 'kulfan' in result:
            kp = get_kulfan_parameters(
                coordinates,
                n_weights_per_side = 8,
                N1 = 0.5,
                N2 = 1.0,
                n_points_per_side = n_points_per_side,
                # normalize_coordinates: bool = True,
                use_leading_edge_modification = True,
                method = "least_squares",
            )
        
        if result == 'airfoil':
            data.append(Airfoil(coordinates=coordinates))
            
        if result == 'kulfan_airfoil':
            data.append(KulfanAirfoil(**kp))
            
        elif result == 'parameters':
            data.append({"x":x, "thickness": th, "camber": y})
            
        elif result == 'kulfan_parameters':
            data.append(kp)
            
        elif result == 'coordinates':
            data.append(coordinates)
            
        else:
            raise ValueError(f"Invalid argument '{result}'. Must be in {_valid_results}.")
        
    return data


if __name__=="__main__":
    from archibald.optimization import Opti
    
    opti = Opti()
    
    # xc, mc = 0.318, 0.08
    xc, mc = 0.40, 0.1862
    # xc, mc = 0.55, 0.10
    # xc = opti.variable(init_guess=0.3)
    # mc = opti.variable(init_guess=0.1)
    # xc = opti.parameter(0.3)
    n = 50
    
    # xc = np.array([0.5, 0.4, 0.3])
    # mc = np.array([0.05, 0.06, 0.07])
    
    AF = thin_airfoil(
        xc,
        mc,
        baseAirfoil="e376",
        # baseAirfoil="e379",
        n_points_per_side=n,
        result="kulfan_airfoil",
    )
    
    alpha = 0.
    Re = 1e6

    opti.minimize(np.sum((mc - 0.00)**2))
        
    sol = opti.solve(verbose=False)
    
    for af in AF:
    
        # aeroModel = afModel.get_aero_from_neuralfoil(alpha, Re)
        # aeroSym = afSym.get_aero_from_neuralfoil(alpha, Re)
        aero = sol(af.get_aero_from_neuralfoil(alpha, Re))
        
        # afModel.to_kulfan_airfoil().draw()
        # afSym.draw()
        sol(af).draw()
        
        print(aero["analysis_confidence"], aero["CL"], aero["CD"])
        print(180/np.pi * np.arctan(leading_edge_camber(xc, mc)))
        
    





