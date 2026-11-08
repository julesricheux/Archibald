# -*- coding: utf-8 -*-
"""Hull methods selected through ``Hull.compute_resistance``: directions, points and torsors."""
import os
import warnings

import casadi as cas
import numpy as onp
import pytest

from archibald.dynamics.hydro import kijima, mittendorf, savitsky
from archibald.geometry.hull import Hull, amidships
from archibald.performance import OperatingPoint

STL = os.path.join(
    os.path.dirname(__file__), "..", "..", "..",
    "examples", "03 - Geometry", "data", "molenez2_data", "hull.stl",
)

SAVITSKY_KW = dict(trim=4., deadrise=15.)
WAVES_KW = dict(swh=2., tp=7., Le=8., Lr=8.)


@pytest.fixture(autouse=True)
def default_floating_point_errors():
    """``test_all_operations_run.py`` calls np.seterr(all="raise") when collected, for the whole session."""
    with onp.errstate(all="warn", under="ignore"):
        yield


@pytest.fixture(scope="module")
def hull():
    with warnings.catch_warnings(), onp.errstate(all="warn", under="ignore"):
        warnings.simplefilter("ignore")
        return Hull(name="molenez", mesh=STL)


@pytest.fixture(scope="module")
def op_point():
    return OperatingPoint(stw=8., dz=-1.3, leeway=4., heel=10., trim=0.5)


def as_float(x):
    return onp.array(x, dtype=float).ravel()


def test_lateral_force_matches_its_formula(hull, op_point):
    """Kijima with DSYHS: L along +y and Ri along -x amidships, N as a couple."""
    F0, M0 = hull.compute_resistance(op_point, method="dsyhs")
    F, M = hull.compute_resistance(op_point, method=["dsyhs", "kijima"])

    params = hull.get_resistance_parameters(op_point)
    L = kijima.compute_L_kijima(**params)
    Ri = kijima.compute_Ri_kijima(**params)
    N = kijima.compute_N_kijima(**params)
    center = op_point.apply_transformations(amidships(**params))
    Flat = onp.array([[-Ri, L, 0.]])
    Mlat = onp.cross(as_float(center), Flat[0]) + onp.array([0., 0., N])

    assert as_float(F) == pytest.approx(as_float(F0) + Flat[0], rel=1e-12)
    assert as_float(M) == pytest.approx(as_float(M0) + Mlat, rel=1e-12)
    assert L > 0.  # +leeway gives +y, like a fin
    # Only the resistances are in the total
    c = hull.resistance_components
    assert c["_total"] == pytest.approx(c["Rf"] + c["Rw"] + c["Rtr"] + c["Ri"])


def test_component_torsors_add_up(hull, op_point):
    F, M = hull.compute_resistance(op_point, method=["dsyhs", "kijima"])
    torsors = hull.resistance_torsors.values()
    assert as_float(F) == pytest.approx(sum(as_float(t[0]) for t in torsors))
    assert as_float(M) == pytest.approx(sum(as_float(t[1]) for t in torsors))
    assert set(hull.resistance_points) == {"Rf", "Rw", "Rtr", "L", "Ri"}  # N is a couple


def test_savitsky_lift_is_vertical_at_its_centre_of_pressure(hull):
    op = OperatingPoint(stw=12., dz=-1.2)
    F, M = hull.compute_resistance(op, method="savitsky", **SAVITSKY_KW)
    c = hull.resistance_components
    Fz = c["Fz"]
    assert Fz > 0.
    assert as_float(F)[2] == pytest.approx(Fz)
    assert -as_float(F)[0] == pytest.approx(c["_total"])
    assert c["_total"] == pytest.approx(c["Rf"] + c["Rw"] + c["Rsp"] + c["Rtr"])

    h = hull.hydrostatics_data
    lam = savitsky.lambda_(Lwl=h["Lwl"], Bwl=h["Bwl"], **SAVITSKY_KW)
    x_cp = h["app"][0, 0] + 0.75 * lam * h["Bwl"]
    point = as_float(hull.resistance_points["Fz"])
    assert point[0] == pytest.approx(x_cp)  # upright, no trim: hull and motion frames share x
    # The lift ahead of the origin pitches the bow up (-My)
    Mz_lift = as_float(hull.resistance_torsors["Fz"][1])
    assert Mz_lift[1] == pytest.approx(-x_cp * Fz)


def test_added_resistance_in_waves_through_resistance(hull, op_point):
    F0, _ = hull.compute_resistance(op_point, method="dsyhs")
    F, _ = hull.compute_resistance(op_point, method=["dsyhs", "mittendorf"], **WAVES_KW)
    h = hull.hydrostatics_data
    Raw = mittendorf.compute_Raw_mittendorf(
        stw=op_point.stw, rho=op_point.environment.water.density, g=op_point.environment.gravity,
        Lbp=h["Lbp"], Bwl=h["Bwl"], T=h["T"], Cb=h["Cb"], **WAVES_KW,
    )
    assert Raw > 0.
    assert hull.resistance_components["Raw"] == pytest.approx(Raw)
    assert as_float(F)[0] == pytest.approx(as_float(F0)[0] - Raw)


def test_added_resistance_in_waves_symbolic_speed(hull):
    stw = cas.MX.sym("stw")
    op = OperatingPoint(stw=stw, dz=-1.3)
    hull.compute_hydrostatics_properties(op)
    expr = hull.compute_resistance_components(
        op, method="mittendorf", recompute_statics=False, **WAVES_KW)["Raw"]
    value = float(cas.Function("f", [stw], [expr])(6.))
    numeric = hull.compute_resistance_components(OperatingPoint(stw=6., dz=-1.3), method="mittendorf", **WAVES_KW)["Raw"]
    assert value == pytest.approx(float(numeric), rel=1e-9)


@pytest.mark.parametrize("method, error", [
    (["savitsky_brown", "mittendorf"], ValueError),           # Raw given twice
    ({"My": savitsky.compute_My_savitsky}, ValueError),       # no direction for My
    ("unknown", NotImplementedError),
    ([], ValueError),
    (3, TypeError),
])
def test_bad_methods_raise(method, error):
    with pytest.raises(error):
        Hull.resolve_resistance_methods(method)


@pytest.mark.parametrize("gamma", [1., 3.3])
def test_wave_spectrum_energy_and_peak(gamma):
    """m0 = Hs^2 / 16 (exactly for Pierson-Moskowitz, within 0.5 % for JONSWAP), peak at 2 pi / Tp."""
    w = onp.linspace(0.01, 20., 200001)
    S = mittendorf.wave_spectrum(w, swh=2., tp=8., gamma=gamma)
    assert onp.trapezoid(S, w) == pytest.approx(2. ** 2 / 16., rel=5e-3 if gamma != 1. else 1e-4)
    assert w[onp.argmax(S)] == pytest.approx(2. * onp.pi / 8., abs=1e-3)


def test_dry_hull_has_zero_force_and_finite_derivatives(hull):
    """Out of the water, the forces are zero and so are their first and second derivatives (no NaN for IPOPT)."""
    F, M = hull.compute_resistance(OperatingPoint(stw=8., dz=1., leeway=4.), method=["dsyhs", "jones"])
    assert as_float(F) == pytest.approx(0., abs=1e-12)
    assert as_float(M) == pytest.approx(0., abs=1e-12)

    x = cas.MX.sym("x", 2)
    F, M = hull.compute_resistance(OperatingPoint(stw=x[0], dz=1., leeway=x[1]), method=["dsyhs", "jones"])
    e = cas.sum1(cas.vec(F)) + cas.sum1(cas.vec(M))
    f = cas.Function("f", [x], [e, cas.jacobian(e, x), cas.hessian(e, x)[0]])
    for value in f(cas.DM([8., 4.])):
        assert onp.all(onp.asarray(value) == 0.)
