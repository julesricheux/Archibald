# -*- coding: utf-8 -*-
"""The strip lift of the VLM is perpendicular to the flow: the force along the flow is only drag."""
import contextlib
import io
import warnings

import numpy as onp
import pytest

from archibald.dynamics.aero_3D.vortex_lattice_method import HydroVortexLatticeMethod
from archibald.geometry import Airfoil, Appendage, Fin, WingXSec
from archibald.performance import OperatingPoint


@pytest.mark.parametrize("leeway", [-6., 3.])
def test_force_along_flow_is_profile_plus_induced_drag(leeway):
    sections = [(onp.array([0.02 * z, 0., -1. + 0.3 * z]), 0.3 - 0.02 * z) for z in range(4)]
    fin = Fin(name="fin", xsecs=[WingXSec(xyz_le=x, chord=c, airfoil=Airfoil("naca0012"), twist=0.) for x, c in sections])
    vlm = HydroVortexLatticeMethod(Appendage(name="a", fins=[fin]), OperatingPoint(stw=8., leeway=leeway),
                                   xyz_ref=[0., 0., 0.], chordwise_resolution=6, spanwise_resolution=1)
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()), onp.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        result = vlm.run(alpha_stall=20)

    as_float = lambda x: onp.asarray(x, dtype=float).ravel()
    flow = as_float(onp.asarray(vlm.strips_freestream_x, dtype=float).mean(axis=0))
    F = as_float(result["F_ab"])
    drag = as_float(vlm.strips_D).sum() + as_float(vlm.strips_Di).sum()
    lift = as_float(vlm.strips_L).sum()

    # with the former oblique axes, the lift leaned back by alpha: F . flow = drag + |L| sin(alpha)
    assert F @ flow == pytest.approx(drag, rel=1e-6)
    assert onp.linalg.norm(F - (F @ flow) * flow) == pytest.approx(abs(lift), rel=1e-6)
    assert drag < 0.1 * abs(lift)


def test_horizontal_foil_lift_follows_pitch():
    """alpha used to be read in the boat x-y plane, which gave ~0 for a horizontal strip whatever the pitch."""
    lifts = {}
    for order in (1, -1):
        ys = onp.linspace(-0.5, 0.5, 5)[::order]
        foil = Fin(name="foil", xsecs=[WingXSec(xyz_le=[0., y, -1.], chord=0.15, airfoil=Airfoil("naca0012"), twist=0.)
                                       for y in ys])
        for trim in (-3., 3.):  # bow up, then bow down
            vlm = HydroVortexLatticeMethod(Appendage(name="a", fins=[foil]), OperatingPoint(stw=10., trim=trim),
                                           xyz_ref=[0., 0., 0.], chordwise_resolution=6, spanwise_resolution=1)
            with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()), onp.errstate(all="ignore"):
                warnings.simplefilter("ignore")
                result = vlm.run(alpha_stall=15)
            lifts[order, trim] = float(onp.asarray(result["F_ab"], dtype=float).ravel()[2])
            assert onp.abs(onp.asarray(vlm.alpha, dtype=float)) == pytest.approx(3., abs=0.05)

    # 523 N at 10 kt and 3 deg, i.e. CL 0.26 for an aspect ratio of 6.7
    assert lifts[1, -3.] == pytest.approx(523., rel=0.05)
    assert lifts[1, 3.] == pytest.approx(-lifts[1, -3.], rel=1e-6)
    assert lifts[-1, -3.] == pytest.approx(lifts[1, -3.], rel=1e-6)
