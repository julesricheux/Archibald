# -*- coding: utf-8 -*-
"""Windage formulas against their sources, see ``examples/03 - Validation/windage.py``."""
import math

import numpy as onp
import pytest

from archibald.dynamics.aero import blendermann, common, day


@pytest.fixture(autouse=True)
def default_floating_point_errors():
    """``test_all_operations_run.py`` calls np.seterr(all="raise") when collected, for the whole session."""
    with onp.errstate(all="warn", under="ignore"):
        yield


KNOT = 0.5144444
WIND = dict(tws=10., stw=0., rho_air=1.225, z_ref=10., wind_alpha=0.)
Q = 0.5 * 1.225 * (10. * KNOT) ** 2


def test_apparent_wind():
    aws, awa = common.apparent_wind(tws=10., twa=90., stw=10., z=10.)
    assert aws == pytest.approx(10. * math.sqrt(2.) * KNOT, rel=1e-3)
    assert awa == pytest.approx(45., rel=1e-3)


def test_day_crew_uses_the_du_bois_area():
    A_front, A_side = day.crew_areas_day(80., 1.83, 9.81)
    # Du Bois & Du Bois (1916) in kg and cm, Day (2017) writes it in N and m
    assert A_front / 0.326 == pytest.approx(0.007184 * 80. ** 0.425 * 183. ** 0.725, rel=0.005)
    assert A_side / A_front == pytest.approx(0.219 / 0.326)


def test_day_hull_areas():
    Af, As, zce = day.hull_areas_day(Fb=0.5, Boa=2., Loa=5., Cwp=0.7, heel=20.)
    heeled = 0.5 + 0.5 * 2. * 0.7 * math.sin(math.radians(20.))
    assert Af == pytest.approx(1.)
    assert As == pytest.approx(5. * heeled, rel=1e-4)
    assert zce == pytest.approx(0.66 * heeled, rel=1e-4)


def test_blendermann_parameters_of_his_table():
    # Blendermann (2004), Table 3, laden container ship. CDlAF and delta are not in that table.
    parameters = blendermann.BLENDERMANN_SHIP_TYPES["container_ship_loaded"]
    assert parameters["CDt"] == pytest.approx(0.90)
    assert parameters["kappa"] == pytest.approx(1.4)


@pytest.mark.parametrize("twa, index, expected", [
    (0., 0, -Q * 1000. * 0.55),
    (90., 1, Q * 5000. * 0.90),
    (180., 0, Q * 1000. * 0.55),
])
def test_blendermann_limits(twa, index, expected):
    forces = blendermann.compute_Fhull_blendermann(
        twa=twa, Ax=1000., Ay=5000., Loa=300., cey=onp.array([[0., 0., 20.]]), ship_type="container_ship_loaded", **WIND,
    )
    assert float(forces[index]) == pytest.approx(expected, rel=1e-3)
