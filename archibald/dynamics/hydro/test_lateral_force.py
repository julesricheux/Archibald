# -*- coding: utf-8 -*-
"""Lateral force formulas against values worked out by hand from their sources."""
import math

import pytest

from archibald.dynamics.hydro import inoue, jones, kijima


HULL = dict(Lwl=20., Bwl=6., T=2., Cb=0.5, Cp=0.6, Cwp=0.75, rho=1000., V=4., beta=math.radians(5.))
Q = 0.5 * 1000. * 4. ** 2 * 20. * 2.
BETA = HULL["beta"]
KAPPA = 2. * 2. / 20.


def test_jones_lift_and_induced_drag():
    lift = Q * math.pi / 2. * KAPPA * BETA
    assert jones.compute_L_jones(**HULL) == pytest.approx(lift)
    # Half of the leading edge suction is lost
    assert jones.compute_Ri_jones(**HULL) == pytest.approx(lift * BETA / 2.)


def test_inoue_normal_force():
    Y = Q * (math.pi / 2. * KAPPA + 1.4 * 0.5 * 6. / 20.) * BETA
    assert inoue.compute_L_inoue(**HULL) == pytest.approx(Y * math.cos(BETA))
    assert inoue.compute_Ri_inoue(**HULL) == pytest.approx(Y * math.sin(BETA))


def test_kijima_force_and_moment():
    ea_prime = 20. / 6. * 0.4 / math.sqrt(1. / 4. + 1. / 3. ** 2)
    sigma_a = 0.25 / 0.4
    K = (1. / ea_prime + 1.5 / (20. / 6.) - 0.33) * (0.95 * sigma_a + 0.40)
    x = 2. * 0.5 / 6. * ea_prime * K

    Y_beta = math.pi / 2. * KAPPA + 1.9257 * (0.5 * 6. / 20.) * sigma_a
    Y_bb = -1.199 * 0.5 * sigma_a + 1.05
    Y = Q * (Y_beta * BETA + Y_bb * BETA ** 2)
    assert kijima.compute_L_kijima(**HULL) == pytest.approx(Y * math.cos(BETA))
    assert kijima.compute_Ri_kijima(**HULL) == pytest.approx(Y * math.sin(BETA))

    N_beta = KAPPA * (15.0668 * x ** 2 - 23.819 * x + 1.802)
    N_bb = 43.857 * x ** 2 - 3.671 * x + 0.086
    assert kijima.compute_N_kijima(**HULL) == pytest.approx(Q * 20. * (N_beta * BETA + N_bb * BETA ** 2))


@pytest.mark.parametrize("compute", [jones.compute_L_jones, inoue.compute_L_inoue, kijima.compute_L_kijima])
def test_lift_is_odd(compute):
    other = dict(HULL, beta=-BETA)
    assert compute(**other) == pytest.approx(-compute(**HULL))


@pytest.mark.parametrize("compute", [jones.compute_Ri_jones, inoue.compute_Ri_inoue, kijima.compute_Ri_kijima])
def test_added_resistance_is_even_and_positive(compute):
    other = dict(HULL, beta=-BETA)
    assert compute(**other) == pytest.approx(compute(**HULL))
    assert compute(**HULL) > 0.
