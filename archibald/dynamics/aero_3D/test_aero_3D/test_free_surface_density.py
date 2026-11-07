# -*- coding: utf-8 -*-
"""Density of VLM panels across the free surface at z = 0."""
import casadi as cas
import numpy as onp
import pytest

from archibald.dynamics.aero_3D.vortex_lattice_method import (
    FREE_SURFACE_TRANSITION,
    AeroVortexLatticeMethod,
    HydroVortexLatticeMethod,
)
from archibald.performance import OperatingPoint


@pytest.fixture(params=[AeroVortexLatticeMethod, HydroVortexLatticeMethod])
def vlm(request):
    v = request.param.__new__(request.param)  # only the operating point is needed here
    v.op_point = OperatingPoint(stw=5.)
    return v


def test_water_fraction_reaches_99_percent_at_the_transition(vlm):
    d = FREE_SURFACE_TRANSITION
    w = onp.asarray(vlm.water_fraction(onp.array([-d, 0., d])), dtype=float)
    assert w == pytest.approx([0.99, 0.5, 0.01], abs=1e-9)


def test_density_is_water_below_and_air_above_for_any_analysis(vlm):
    env = vlm.op_point.environment
    rho = onp.asarray(vlm.local_density(onp.array([[0., 0., -1.], [0., 0., 1.]])), dtype=float)
    assert rho == pytest.approx([env.water.density, env.air.density], rel=1e-9)


def test_density_is_differentiable(vlm):
    dz = cas.MX.sym("dz")
    points = cas.horzcat(cas.MX.zeros(1, 2), dz)
    rho = vlm.local_density(points)
    f = cas.Function("f", [dz], [rho, cas.jacobian(rho, dz)])
    value, slope = f(0.)
    env = vlm.op_point.environment
    assert float(value) == pytest.approx(0.5 * (env.water.density + env.air.density))
    assert float(slope) < 0.  # denser going down
