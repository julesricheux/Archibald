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

import warnings
import archibald.toolbox.units as u

### Define constants
gravitationnal_acceleration = 9.80665 # m/s-2
gas_constant_universal = 8.31432  # J/(mol*K); universal gas constant
molecular_mass_air = 28.9644e-3  # kg/mol; molecular mass of air
gas_constant_air = gas_constant_universal / molecular_mass_air  # J/(kg*K); gas constant of air
effective_collision_diameter = 0.365e-9  # m, effective collision diameter of an air molecule

# LEGACY
# WATER_DATA = '../data/seawater_ittc_2011.csv' # TODO remove

warnings.simplefilter("always", RuntimeWarning)

#%% CLASSES

def _check_validity_range(_value, _range):
    vmin, vmax = _range

    if vmin is not None and _value < vmin:
        return True, False
    elif vmax is not None and _value > vmax:
        return False, True
    else:
        return False, False


class Fluid():
    """
    Citation:
        Adapted from:         atmosphere.Atmosphere in AeroSandbox
        Author:               Peter D Sharpe
        Date of Retrieval:    17/10/2024
        
    """
    def __init__(
            self,
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
        
        self._check_temperature_range()
        
    def _check_temperature_range(self):
        too_low, too_high = _check_validity_range(self.temperature, self._valid_temperature_range)
        vmin, vmax = self._valid_temperature_range
        
        if too_low:
            warnings.warn(
                f"⚠️ Temperature {self.temperature:.2f} °C below valid range ({vmin}, {vmax})",
                RuntimeWarning
            )
        elif too_high:
            warnings.warn(
                f"⚠️ Temperature {self.temperature:.2f} °C above valid range ({vmin}, {vmax})",
                RuntimeWarning
            )
        
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
    
# LEGACY
# TODO remove
# class Water(Fluid):
#     def __init__(self,
#                  temperature: float = 15.,  # Celsius
#                  salinity: float = 35., # g/kg
#                  valid_temperature_range: tuple = (None, None), # Celsius
#                  valid_salinity_range: tuple = (None, None), # g/kg
#                  ):
        
#         super().__init__(temperature=temperature,
#                          valid_temperature_range=valid_temperature_range)
        
#         self.salinity=salinity
#         self.valid_salinity_range=valid_salinity_range
        
#         current = os.path.dirname(os.path.realpath(__file__))
#         self._measurements = read_coefs(os.path.join(current, WATER_DATA),
#                                         skipRows=1,
#                                         delim='\t')
#         self._interpolators = build_interpolation(self._measurements)
    
#     def __repr__(self) -> str:
#         string = f"temperature: {self.temperature:.1f} °C / salinity {self.salinity:.1f} g/kg"

#         return f"Water ({string}, method: '{self.method}')"
        
#     def set_salinity(self, salinity: float = 35.):
#         self.salinity = salinity        
        
#     @property
#     def density(self):
#         """
#         Returns the density, in kg/m^3.
        
#         """
#         rho = self._interpolators[0](self.temperature) * 1.

#         return rho
    
#     @property
#     def dynamic_viscosity(self):
#         """
#         Returns the dynamic viscosity (mu), in kg/(m*s).

#         Based on ITTC 2011 recommandations.
        
#         """
#         mu = self._interpolators[1](self.temperature) * 1.

#         return mu


class Water(Fluid):
    """
    Describes water properties to be used in an environment.

    Density:
        Sharqawy et al. (2010), based on UNESCO EOS-80, recommended by ITTC 2011.

    Notes:
        - Temperature in °C
        - Salinity in g/kg
        - Internally converted to mass fraction (kg/kg)
    """

    def __init__(
        self,
        temperature: float = 15.,        # Celsius
        salinity: float = 35.,           # g/kg
        valid_temperature_range: tuple = (0., 40.),
        valid_salinity_range: tuple = (0., 42.),
    ):
        super().__init__(
            temperature=temperature,
            valid_temperature_range=valid_temperature_range
        )

        self.salinity = salinity
        self._valid_salinity_range = valid_salinity_range
        
        self._check_salinity_range()
        
    def _check_salinity_range(self):
        too_low, too_high = _check_validity_range(self.salinity, self._valid_salinity_range)
        vmin, vmax = self._valid_salinity_range
        
        if too_low:
            warnings.warn(
                f"⚠️ Salinity {self.salinity:.2f} g/kg below valid range ({vmin}, {vmax})",
                RuntimeWarning
            )
        elif too_high:
            warnings.warn(
                f"⚠️ Salinity {self.salinity:.2f} g/kg above valid range ({vmin}, {vmax})",
                RuntimeWarning
            )

    def __repr__(self) -> str:
        string = (
            f"temperature: {self.temperature:.1f} °C / "
            f"salinity {self.salinity:.1f} g/kg"
        )
        return f"Water ({string}, method: '{self.method}')"

    def set_salinity(
        self,
        salinity: float = 35.,
    ):
        self.salinity = salinity

    @property
    def density(self):
        """
        Returns density [kg/m^3].

        Based on Sharqawy et al. (2010). Thermophysical properties of seawater:
        a review of existing correlations and data.

        Validity:
        - -2 < T < 40 °C
        - 0 < S < 42 g/kg
        Accuracy: ±0.01 %
        """

        T = self.temperature
        S = self.salinity * 1e-3  # g/kg -> kg/kg

        # Pure water density
        rho_w = (
            999.842594
            + 6.793952e-2 * T
            - 9.095290e-3 * T**2
            + 1.001685e-4 * T**3
            - 1.120083e-6 * T**4
            + 6.536332e-9 * T**5
        )

        # Salinity coefficients
        A = (
            0.824493
            - 4.0899e-3 * T
            + 7.6438e-5 * T**2
            - 8.2467e-7 * T**3
            + 5.3875e-9 * T**4
        )

        B = (
            -5.72466e-3
            + 1.0227e-4 * T
            - 1.6546e-6 * T**2
        )

        C = 4.8314e-4

        rho = (
            rho_w
            + A * S
            + B * S**1.5
            + C * S**2
        )

        return rho

    @property
    def dynamic_viscosity(self):
        """
        Returns dynamic viscosity μ [Pa·s].

        Based on Sharqawy et al. (2010). Thermophysical properties of seawater:
        a review of existing correlations and data.
        
        Used in ITTC 2011 Recommended Procedures (https://ittc.info/media/4048/75-02-01-03.pdf)
        
        Validity:
        - 0 < T < 180 °C
        - 0 < S < 150 g/kg
        Accuracy: ±1.5 %
        """

        T = self.temperature
        S = self.salinity * 1e-3  # g/kg -> kg/kg

        # Pure water viscosity
        mu_w = (
            4.2844e-5
            + 1.0 / (
                0.157 * (T + 64.993)**2
                - 91.296
            )
        )

        # Salinity correction
        A = (
            1.541
            + 1.998e-2 * T
            - 9.52e-5 * T**2
        )

        B = (
            7.974
            - 7.561e-2 * T
            + 4.724e-4 * T**2
        )

        mu = mu_w * (1 + A * S + B * S**2)

        return mu
    
    
class Air(Fluid):
    
    """
    Describes air properties to be used in an environment.
    
    Notes:
        - Temperature in °C
    """
    
    def __init__(self,
                 temperature: float = 15.,  # Celsius
                 pressure: float = u.atm, # Pa
                 valid_temperature_range: tuple = (0-u.kelvin, 3000+u.kelvin), # Celsius
                 valid_pressure_range: tuple = (0.01*u.atm, 100*u.atm), # Pa
                 ):
        
        super().__init__(
            temperature=temperature,
            valid_temperature_range=valid_temperature_range
        )
        
        self.pressure=pressure
        self.valid_pressure_range=valid_pressure_range
        
    def _check_pressure_range(self):
        too_low, too_high = _check_validity_range(self.pressure, self._valid_pressure_range)
        vmin, vmax = self._valid_pressure_range
        
        if too_low:
            warnings.warn(
                f"⚠️ Pressure {self.salinity:.2f} Pa below valid range ({vmin}, {vmax})",
                RuntimeWarning
            )
        elif too_high:
            warnings.warn(
                f"⚠️ Pressure {self.salinity:.2f} Pa above valid range ({vmin}, {vmax})",
                RuntimeWarning
            )
        
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
                 air_pressure: float = u.atm, # Pa
                 ):
        
        self.water = Water(temperature=water_temperature,
                           salinity=water_salinity)
        
        self.air = Air(temperature=air_temperature,
                       pressure=air_pressure)
        
        self.gravity = gravitationnal_acceleration
        
    
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