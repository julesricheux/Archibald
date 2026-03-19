# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 15:20:21 2025

@author: jrich
"""

import jax.numpy as np


J_max = 10/33 # Overall normalized max damage length
J_kn = 5/33 # Knuckle point in the distribution
p_k = 11/12 # Cumulative probability at J_kn
l_max = 60 # Maximum absolute damage length (meters)
L_star = 260 # Length where normalized distribution ends (meters)
b0 = 2*(p_k/J_kn - (1-p_k)/(J_max - J_kn)) # Probability density at J = 0:
    

def R(ship_type: str,
      L_s: float,
      N: int = None,
      )-> float:
    """
    Calculate the required subdivision index R based on ship type, length, and number of persons.
    
    Parameters:
        ship_type (str): The type of the ship ('cargo', 'passenger').
        L_s (float): The length of the ship in meters.
        N (int): Total number of persons on board (required for passenger ships).
        
    Returns:
        float: The required subdivision index R.
    """
    if ship_type == 'cargo':
        if L_s > 100:
            R = 1 - 128 / (L_s + 152)
        elif 80 <= L_s <= 100:
            R_base = 1 - 128 / (L_s + 152)
            R = R_base * (1 + 80 / L_s)
        else:
            raise ValueError("Length must be at least 80 meters for cargo ships.")
    
    elif ship_type == 'passenger':
        if N is None:
            raise ValueError("Number of persons on board must be provided for passenger ships.")
        
        if N < 400:
            R = 0.722
        elif 400 <= N < 1350:
            R = N/7580 + 0.66923
        elif 1350 <= N < 6000:
            R = 0.0369 * np.log(N + 89.048) + 0.579
        elif N >= 6000:
            R = 1 - (852.5 + 0.03875 * N) / (N + 5000)
        else:
            raise ValueError("Invalid number of persons for passenger ships.")
    else:
        raise ValueError("Invalid ship type. Use 'cargo' or 'passenger'.")
    
    return R


def A(A_s: float,
      A_p: float,
      A_l: float,
      ) -> float:
    """
    Calculate the subdivision index A from the partial indices A_s, A_p, and A_l.
    
    Parameters:
        A_s (float): The partial index for the ship's subdivision.
        A_p (float): The partial index for the passenger ship's subdivision.
        A_l (float): The partial index for the cargo ship's subdivision.
        
    Returns:
        float: The total subdivision index A.
    """
    return 0.4 * A_s + 0.4 * A_p + 0.2 * A_l


def Mpassenger(Np: int,
               B: float,
               ) -> float:
    """
    Calculate the heeling moment Mpassenger due to passenger movement.
    
    Parameters:
        Np (int): Number of passengers onboard.
        B (float): Breadth of the ship (meters).
        
    Returns:
        float: The heeling moment due to passenger movement (t.m).
    """
    return 0.075 * Np * 0.45 * B


def Mwind(A: float,
          Z: float,
          ) -> float:
    """
    Calculate the heeling moment Mwind due to wind.
    
    Parameters:
        A (float): Projected lateral area above the waterline (m²).
        Z (float): Distance from the center of lateral area to T/2 (meters).
        
    Returns:
        float: The heeling moment due to wind (t.m).
    """
    P = 120. # Wind pressure (N/m²)
    g = 9.806 # Gravitionnal acceleration (m/s²)
    
    return P * A * Z / g / 1000  # Convert to t.m (from N.m)


def Msurvivalcraft(N_lifeboats: int,
                   Weight_lifeboat: float,
                   Arm_lifeboat: float,
                   ) -> float:
    """
    Calculate the heeling moment Msurvivalcraft due to launching of lifeboats.
    
    Parameters:
        N_lifeboats (int): Number of lifeboats on the heeled side.
        Weight_lifeboat (float): Weight of a single lifeboat (tons).
        Arm_lifeboat (float): Distance from the centerline to the arm of the lifeboat (meters).
        
    Returns:
        float: The heeling moment due to lifeboat launching (t.m).
    """
    return N_lifeboats * Weight_lifeboat * Arm_lifeboat


def Mheel(Mpassenger: float,
          Mwind: float,
          Msurvivalcraft: float,
          ) -> float:
    """
    Calculate the maximum heeling moment Mheel.
    
    Parameters:
        Mpassenger (float): Heeling moment due to passenger movement (t-m).
        Mwind (float): Heeling moment due to wind (t-m).
        Msurvivalcraft (float): Heeling moment due to lifeboat launching (t-m).
        
    Returns:
        float: The maximum heeling moment among the given values (t-m).
    """
    return np.max((Mpassenger, Mwind, Msurvivalcraft))

def permeability(space_type: str,
                 draught: str = None,
                 ) -> float:
    """
    Compute the permeability of a space based on its type and the considered draught.
    
    Parameters:
        space_type (str): The type of the space ('store', 'accommodation', 'machinery', 
                          'void', 'cargo_dry', 'cargo_container', 'cargo_roro', 'cargo_liquid').
        draught (str, optional): The considered draught ('d_s', 'd_p', or 'd_l') 
                                 for cargo spaces. Not required for other spaces.
        
    Returns:
        float: The permeability of the space.
    """
    permeability_data = {
        'store': 0.60,
        'accommodation': 0.95,
        'machinery': 0.85,
        'void': 0.95,
        'cargo_dry': {'d_s': 0.70, 'd_p': 0.80, 'd_l': 0.95},
        'cargo_container': {'d_s': 0.70, 'd_p': 0.80, 'd_l': 0.95},
        'cargo_roro': {'d_s': 0.90, 'd_p': 0.90, 'd_l': 0.95},
        'cargo_liquid': {'d_s': 0.70, 'd_p': 0.80, 'd_l': 0.95},
    }
    
    if space_type in permeability_data:
        
        data = permeability_data[space_type]
    
        if type(data) == float:
            return data
        else:
            if draught not in ['d_s', 'd_p', 'd_l']:
                raise ValueError(f"Draught must be 'd_s', 'd_p', or 'd_l' for {space_type}.")
            return data[draught]
    
    raise ValueError(f"Unknown space type: {space_type}")


# def p_i(j: int,
#         n: int,
#         k: int,
#         x1: float,
#         x2: float,
#         b: float,
#         ) -> float:
#     """
#     Compute the factor p_i based on Regulation 7-1, Section 1.
    
#     Parameters:
#         j (int): the aftmost damage zone number involved in the damage starting with No. 1 at the stern;
        
#         n (int): the number of adjacent damage zones involved in the damage;
        
#         k (int): the number of a particular longitudinal bulkhead as barrier for transverse penetration in a damage
#                  zone counted from shell towards the centreline. The shell has k = 0;
                 
#         x1 (float): the distance from the aft terminal of L, to the aft end of the zone in question;
        
#         x2 (float): the distance from the aft terminal of L, to the forward end of the zone in question;
        
#         b (float): the mean transverse distance in metres measured at right angles to the centreline at the deepest
#                    subdivision draught between the shell and an assumed vertical plane extended between the
#                    longitudinal timits used in calculating the factor p; and which is a tangent to, or common with, all
#                    or part of the outermost portion of the longitudinal bulkhead under consideration. This vertical
#                    plane shall be so orientated that the mean transverse distance to the shell is a maximum, but
#                    not more than twice the least distance between the plane and the shell. If the upper part of
#                    a longitudinal bulkhead is below the deepest subdivision draught the vertical plane used for
#                    determination of b is assumed to extend upwards
        
#     Returns:
#         float: The factor p_i.
#     """
#     # Validate inputs
#     if n < 1:
#         raise ValueError("The number of adjacent zones (n) must be at least 1.")
#     if k < 0:
#         raise ValueError("The bulkhead number (k) cannot be negative.")
#     if b < 0:
#         raise ValueError("The mean transverse distance (b) cannot be negative.")
#     if x1 >= x2:
#         raise ValueError("x1 must be less than x2.")

#     # Base formula for p_i
#     if n == 1:
#         p_i = p(x1[j], x2[j]) * (p(x2[j] - x1[j])
#     elif n == 2:
#         p_i = (x2 - x1) / (b * n)
#     else:
#         p_i = ((x2 - x1) / b) / n

#     # Include the effect of k (bulkhead barrier factor)
#     p_i *= (1 - (k / (k + 1)))

#     return p_i

def compute_J_b(b: float,
                B: float,
                ) -> float:
    """
    Compute the factor J_b based on b and B.
    
    Parameters:
        b (float): the mean transverse distance in metres measured at right angles to the centreline at the deepest
                   subdivision draught between the shell and an assumed vertical plane extended between the
                   longitudinal limits used in calculating the factor p; and which is a tangent to, or common with, all
                   or part of the outermost portion of the longitudinal bulkhead under consideration. This vertical
                   plane shall be so orientated that the mean transverse distance to the shell is a maximum, but
                   not more than twice the least distance between the plane and the shell. If the upper part of
                   a longitudinal bulkhead is below the deepest subdivision draught the vertical plane used for
                   determination of b is assumed to extend upwards
        B (float): Ship breadth (meters).
        
    Returns:
        float: The factor J_b.
    """

    return b / (15 * B)


def compute_J(x1: float,
              x2: float,
              L_s: float,
              ) -> float:
    """
    Compute the non-dimensional damage length J based on x1 and x2.
    
    Parameters:
        x1 (float): the distance from the aft terminal of L, to the aft end of the zone in question.
        x2 (float): the distance from the aft terminal of L, to the forward end of the zone in question.
        
    Returns:
        float: The factor J.
    """

    return (x2 - x1)/L_s


def compute_G(j: float,
              x1: float,
              x2: float,
              condition: str,
              J_b: float,
              J: float,
              ) -> float:
    """
    Compute the factor G based on the given condition.
    
    Parameters:
        j (float): Non-dimensional damage length.
        x1 (float): Distance from the aft terminal of L to the aft end of the zone (meters).
        x2 (float): Distance from the aft terminal of L to the forward end of the zone (meters).
        condition (str): Condition type ('entire', 'coincides', or 'neither').
        J_b (float): Factor computed from b and B.
        J (float): The non-dimensional damage length J based on x1 and x2.
        
    Returns:
        float: The factor G.
    """
    
    #TODO : implement b_11 and b_12 computation
    b_11 = 1.
    b_12 = 1.
    
    # Regulation 7-1, 1.2.1
    G1 = 1/2 * b_11 * J_b**2 + b_12 * J_b
    
    # Regulation 7-1, 1.2.1
    J_0 = np.min((J, J_b))
    G2 = -1/3 * b_11 * J_0**3 + 1/2 * (b_11*J - b_12) * J_0**2 + b_12 * J * J_0
    
    if condition == 'entire': # the compartment extends over the full lengh
        return G1
    elif condition == 'neither': # neither of the compartment limits are at the ship ends
        return G2
    elif condition == 'coincides': # only one of the compartment limits are at one only of the ship ends
        return 1/2 * (G2 + G1*J)
    else:
        raise ValueError("Invalid condition. Must be 'entire', 'coincides', or 'neither'.")
        
        
def compute_C(J_b: float,
              ) -> float:
    """
    Compute the factor C from J_b.
    
    Parameters:
        J_b (float): Factor computed from b and B.
        
    Returns:
        float: The factor C.
    """
    return 12 * J_b * (-45 * J_b + 4)


def compute_r(x1: float,
              x2: float,
              b: float,
              ) -> float:
    """
    Compute the factor r based on x1, x2, and b.
    
    Parameters:
        x1 (float): Distance from the aft terminal of L to the aft end of the zone (meters).
        x2 (float): Distance from the aft terminal of L to the forward end of the zone (meters).
        b (float): Mean transverse distance to the shell (meters).
        
    Returns:
        float: The factor r.
    """
    #TODO : implement p(x1, x2) computation
    
    return 1 - (1 - C) * (1 - G/p(x1, x2))


def compute_J_k(L_s: float,
                ) -> float:
    """
    Compute the factor J_k: Cumulative probability knuckle point.
    
    Parameters:
        L_s (float): The length of the ship in meters.
        
    Returns:
        float: The factor J_m.
    """
    J_m = compute_J_m(L_s)
    
    if L_s <= L_star:
        J_k = J_m / 2 + (1 - np.sqrt(1 + (1 - 2*p_k) * b0 * J_m + 0.25 * b0**2 * J_m**2)) / b0
    else:
        J_m_star = np.min((J_max, l_max/L_star))
        J_k_star = J_m / 2 + (1 - np.sqrt(1 + (1 - 2*p_k) * b0 * J_m_star + 0.25 * b0**2 * J_m_star**2)) / b0
        J_k = J_k_star * L_star / L_s
        
    return J_k

def compute_J_n(J: float,
                J_m: float,
                ) -> float:
    """
    Compute the factor J_n: The normalized length of a compartment or group of compartments.
    
    Parameters:
        J (float): The length of the ship in meters.
        J_m (float): ?
        
    Returns:
        float: The factor J_m.
    """
    
    return np.min((J, J_m))


def compute_J_m(L_s: float,
                ) -> float:
    """
    Compute the factor J_m.
    
    Parameters:
        J (float): The non-dimensional damage length J based on x1 and x2.
        J_m (float): The non-dimensional damage length J based on x1 and x2.
        
    Returns:
        float: The factor J_m.
    """
    if L_s <= L_star:
        J_m = np.min((J_max, l_max/L_s))
    else:
        J_m_star = np.min((J_max, l_max/L_star))
        J_m = J_m_star * L_star / L_s
        
    return J_m



if __name__=='__main__':
    
    
    print(R('cargo', 169, N=24))  # Output: R value
    
    print(R('passenger', 169, N=400))  # Output: R value
    
    # Example inputs
    print(permeability('store'))                        # Output: 0.60
    print(permeability('accommodation'))               # Output: 0.95
    print(permeability('cargo_dry', draught='d_s'))    # Output: 0.70
    print(permeability('cargo_roro', draught='d_l'))   # Output: 0.95


