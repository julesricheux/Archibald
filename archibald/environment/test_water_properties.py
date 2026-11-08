# -*- coding: utf-8 -*-
"""
Water properties against ITTC Recommended Procedure 7.5-02-01-03.

Fresh Water and Seawater Properties, revision 02 (2011), at atmospheric pressure:
    Table 1: fresh water, by temperature.
    Table 3: standard seawater, absolute salinity 35.16504 g/kg, by temperature.
    Table 4: seawater at 15 °C, by absolute salinity.
"""
import pytest

from archibald.environment.environment import Environment, Water

# (temperature [°C], absolute salinity [g/kg], density [kg/m^3], dynamic viscosity [Pa.s])
ITTC_2011 = [
    (15., 0., 999.1026, 0.001138),          # Table 1
    (20., 0., 998.2072, 0.001002),          # Table 1
    (15., 10., 1006.7950, 0.001159),        # Table 4
    (15., 20., 1014.4192, 0.001182),        # Table 4
    (15., 35., 1025.8944, 0.001220),        # Table 4
    (15., 35.16504, 1026.0210, 0.001220),   # Table 3
    (20., 35.16504, 1024.8103, 0.001077),   # Table 3
]


@pytest.mark.parametrize("temperature, salinity, density, viscosity", ITTC_2011)
def test_density_against_ittc(temperature, salinity, density, viscosity):
    water = Water(temperature=temperature, salinity=salinity)
    assert water.density == pytest.approx(density, abs=0.06)


@pytest.mark.parametrize("temperature, salinity, density, viscosity", ITTC_2011)
def test_viscosity_against_ittc(temperature, salinity, density, viscosity):
    water = Water(temperature=temperature, salinity=salinity)
    assert water.dynamic_viscosity == pytest.approx(viscosity, rel=0.01)


def test_default_environment_is_sea_water():
    """The default salinity of 35 g/kg used to change the density by only 0.03 kg/m^3."""
    assert Environment().water.density == pytest.approx(1025.8944, abs=0.06)
    assert Environment(water_salinity=0.).water.density == pytest.approx(999.1026, abs=0.06)
