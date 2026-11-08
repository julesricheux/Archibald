# -*- coding: utf-8 -*-
"""Force arrows of draw_vlm: application points on the line of action, split and summed."""
import numpy as onp
import pytest

from archibald.geometry.sailboat import Sailboat
from archibald.performance import OperatingPoint
from archibald.weights import MassProperties


@pytest.fixture
def boat():
    boat = Sailboat(masses=[
        MassProperties(mass=100., x_cg=1., y_cg=0.5, z_cg=2.),
        MassProperties(mass=300., x_cg=3., y_cg=-0.5, z_cg=0.),
    ])
    op_point = OperatingPoint(stw=5.)
    boat.compute_weight(op_point)
    boat.compute_torsor(op_point)
    return boat


def test_split_weight_is_at_each_centre_of_gravity(boat):
    arrows = boat.get_force_arrows(split=True)
    assert [a["name"] for a in arrows] == ["w_0", "w_1"]
    assert arrows[0]["point"] == pytest.approx([1., 0.5, 2.])
    assert arrows[1]["point"] == pytest.approx([3., -0.5, 0.])
    assert arrows[1]["force"][2] < 0.


def test_summed_weight_keeps_the_moment(boat):
    (arrow,) = boat.get_force_arrows(split=False)
    F = onp.asarray(boat.forces["Fw"], dtype=float).ravel()
    M = onp.asarray(boat.moments["Mw"], dtype=float).ravel()
    assert arrow["force"] == pytest.approx(F)
    assert onp.cross(arrow["point"], arrow["force"]) + arrow["couple"] == pytest.approx(M)
    # Total centre of gravity in x and y, at the force-weighted mean height
    assert arrow["point"] == pytest.approx([2.5, -0.25, 0.5])


def test_couple_along_the_force_is_reported():
    F = onp.array([0., 0., -10.])
    M = onp.cross([1., 2., 3.], F) + onp.array([0., 0., 7.])
    point = Sailboat._line_of_action_point(F, M, onp.zeros(3))
    assert point == pytest.approx([1., 2., 0.])


def test_unknown_unit_raises(boat):
    with pytest.raises(ValueError):
        boat._force_arrow_geometry(boat.get_force_arrows(), 1., None, "lbf")
