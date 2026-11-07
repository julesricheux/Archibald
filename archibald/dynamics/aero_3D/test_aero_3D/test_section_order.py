# -*- coding: utf-8 -*-
"""The force of a VLM wing must not depend on the order of its sections."""
import contextlib
import io
import warnings

import numpy as onp
import pytest

from archibald.dynamics.aero_3D.vortex_lattice_method import HydroVortexLatticeMethod
from archibald.geometry import Airfoil, Appendage, Fin, WingXSec
from archibald.performance import OperatingPoint


def fin_force(airfoil, reverse, leeway):
    sections = [(onp.array([0.02 * z, 0., -1. + 0.3 * z]), 0.3 - 0.02 * z) for z in range(4)]
    if reverse:
        sections = sections[::-1]
    fin = Fin(name="fin", xsecs=[WingXSec(xyz_le=x, chord=c, airfoil=airfoil, twist=0.) for x, c in sections])
    op_point = OperatingPoint(stw=8., leeway=leeway)
    vlm = HydroVortexLatticeMethod(Appendage(name="a", fins=[fin]), op_point, xyz_ref=[0., 0., 0.],
                                   chordwise_resolution=6, spanwise_resolution=1)
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()), onp.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        result = vlm.run(alpha_stall=20)
    return onp.asarray(result["F_g"], dtype=float).ravel(), onp.asarray(result["F_vlm_g"], dtype=float).ravel()


@pytest.mark.parametrize("leeway", [-3., 3.])
def test_symmetric_fin_is_independent_of_section_order(leeway):
    F_up, Fvlm = fin_force(Airfoil("naca0012"), False, leeway)
    F_down, _ = fin_force(Airfoil("naca0012"), True, leeway)
    assert F_down == pytest.approx(F_up, rel=1e-6, abs=1e-6)
    # +leeway gives +y, like the inviscid VLM
    assert onp.sign(F_up[1]) == onp.sign(leeway) == onp.sign(Fvlm[1])


def test_cambered_fin_follows_its_mesh():
    """Reversing the sections and flipping the airfoil describe the same fin."""
    F_up, Fvlm_up = fin_force(Airfoil("e854"), False, 0.)
    F_flipped, Fvlm_flipped = fin_force(Airfoil("e854").scale(scale_y=-1.), True, 0.)
    assert F_flipped == pytest.approx(F_up, rel=1e-6, abs=1e-6)
    assert Fvlm_flipped == pytest.approx(Fvlm_up, rel=1e-6, abs=1e-6)
    assert onp.sign(F_up[1]) == onp.sign(Fvlm_up[1])  # NeuralFoil and the panels agree on the camber side
