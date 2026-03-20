# -*- coding: utf-8 -*-
"""
This module relies on Peter D. Sharpe's awesome work on AeroSandbox.

AeroSandbox
Author: Peter D. Sharpe
Repository: https://github.com/peterdsharpe/AeroSandbox
Version used: 4.2.9
Date retrieved: 2026-03-20

AeroSandbox is distributed under its original MIT license.
"""

#%% DEPENDENCIES

import os
import sys

__root_dir = os.path.dirname(os.path.abspath(__file__))
if __root_dir not in sys.path:
    sys.path.append(os.path.dirname(__root_dir))

from archibald2.tools.math_utils import read_coefs, build_interpolation
import archibald2.tools.units as u

### Define constants
gas_constant_universal = 8.31432  # J/(mol*K); universal gas constant
molecular_mass_air = 28.9644e-3  # kg/mol; molecular mass of air
gas_constant_air = gas_constant_universal / molecular_mass_air  # J/(kg*K); gas constant of air
effective_collision_diameter = 0.365e-9  # m, effective collision diameter of an air molecule

WATER_DATA = '../data/seawater_ittc_2011.csv'


#%% CLASSES


class Fluid():
    """
    Citation:
        Adapted from:         atmosphere.Atmosphere in AeroSandbox
        Author:               Peter D Sharpe
        Date of Retrieval:    17/10/2024
        
    """
    def __init__(self,
                 temperature: float = 15.,  # Celsius
                 valid_temperature_range: tuple = (None, None), # Celsius
                 method: str = "differentiable",
                 ):
        """
        Abstract class for common fluid features.

        """
        self.temperature = temperature
        self.temperature_K = temperature + u.kelvin
        self._valid_temperature_range = valid_temperature_range
        self.method = method
        
    def set_temperature(self, temperature: float = 15.):
        self.temperature = temperature
        self.temperature_K = temperature + u.kelvin
    
    @property
    def density(self):
        """
        Returns the density, in kg/m^3.
        
        """
        return 1.
    
    @property
    def dynamic_viscosity(self):
        """
        Returns the dynamic viscosity (mu)
        
        """
        return 1.
    
    @property
    def kinematic_viscosity(self):
        """
        Returns the kinematic viscosity (nu)
        
        """
        return self.dynamic_viscosity / self.density
    
    
class Water(Fluid):
    def __init__(self,
                 temperature: float = 15.,  # Celsius
                 salinity: float = 35., # g/kg
                 valid_temperature_range: tuple = (None, None), # Celsius
                 valid_salinity_range: tuple = (None, None), # g/kg
                 ):
        
        super().__init__(temperature=temperature,
                         valid_temperature_range=valid_temperature_range)
        
        self.salinity=salinity
        self.valid_salinity_range=valid_salinity_range
        
        current = os.path.dirname(os.path.realpath(__file__))
        self._measurements = read_coefs(os.path.join(current, WATER_DATA),
                                        skipRows=1,
                                        delim='\t')
        self._interpolators = build_interpolation(self._measurements)
    
    def __repr__(self) -> str:
        string = f"temperature: {self.temperature:.1f} °C / salinity {self.salinity:.1f} g/kg"

        return f"Water ({string}, method: '{self.method}')"
        
    def set_salinity(self, salinity: float = 35.):
        self.salinity = salinity        
        
    @property
    def density(self):
        """
        Returns the density, in kg/m^3.
        
        """
        rho = self._interpolators[0](self.temperature) * 1.

        return rho
    
    @property
    def dynamic_viscosity(self):
        """
        Returns the dynamic viscosity (mu), in kg/(m*s).

        Based on ITTC 2011 recommandations.
        
        """
        mu = self._interpolators[1](self.temperature) * 1.

        return mu
    
    
class Air(Fluid):
    """
    Citation:
        Adapted from:         atmosphere.Atmosphere in AeroSandbox
        Author:               Peter D Sharpe
        Date of Retrieval:    17/10/2024
        
    """
    def __init__(self,
                 temperature: float = 15.,  # Celsius
                 pressure: float = 101325., # Pa
                 valid_temperature_range: tuple = (None, None), # Celsius
                 valid_pressure_range: tuple = (None, None), # Pa
                 ):
        
        super().__init__(temperature=temperature,
                         valid_temperature_range=valid_temperature_range)
        
        self.pressure=pressure
        self.valid_pressure_range=valid_pressure_range
        
    def __repr__(self) -> str:
        string = f"temperature: {self.temperature:.1f} °C / pressure {self.pressure/100.:.1f} hPa"

        return f"Air ({string}, method: '{self.method}')"
        
    def set_pressure(self, pressure: float = 101325.):
        self.pressure = pressure     
            
    @property
    def density(self):
        """
        Returns the density, in kg/m^3.
        """
        rho = self.pressure / (self.temperature_K * gas_constant_air)

        return rho

    @property
    def dynamic_viscosity(self):
        """
        Returns the dynamic viscosity (mu), in kg/(m*s).

        Based on Sutherland's Law, citing `https://www.cfd-online.com/Wiki/Sutherland's_law`.

        According to Rathakrishnan, E. (2013). Theoretical aerodynamics. John Wiley & Sons.:
        This relationship is valid from 0.01 to 100 atm, and between 0 and 3000K.

        According to White, F. M., & Corfield, I. (2006). Viscous fluid flow (Vol. 3, pp. 433-434). New York: McGraw-Hill.:
        The error is no more than approximately 2% for air between 170K and 1900K.
        """

        # Sutherland constants
        C1 = 1.458e-6  # kg/(m*s*sqrt(K))
        S = 110.4  # K

        # Sutherland equation
        temperature = self.temperature_K
        mu = C1 * temperature ** 1.5 / (temperature + S)

        return mu
        
             
class Environment():
    def __init__(self,
                 water_temperature: float = 15., # Celsius
                 water_salinity: float = 35., # g/kg
                 air_temperature: float = 15., # Celsius
                 air_pressure: float = 101325., # Pa
                 ):
        
        self.water = Water(temperature=water_temperature,
                           salinity=water_salinity)
        
        self.air = Air(temperature=air_temperature,
                       pressure=air_pressure)
        
        self.gravity = 9.80665 # m/s-2
        
    
    def __repr__(self):
        string = str(self.water) + '\n' + str(self.air)
        
        return 'Environment :\n' + string


if __name__ == '__main__':
    env = Environment(water_temperature = 15., # Celsius
                      water_salinity = 35., # g/kg
                      air_temperature = 15., # Celsius
                      air_pressure = 101325., # Pa
                      )
    
    print(env)