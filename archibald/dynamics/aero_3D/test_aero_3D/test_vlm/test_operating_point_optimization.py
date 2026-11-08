"""Finds the leeway of best lift to drag of a fin through the VLM, and compares with a sweep."""
import contextlib
import io
import warnings

import numpy as onp
import pytest

import archibald as arb

fin = arb.Fin(
    name="fin",
    xsecs=[
        arb.WingXSec(xyz_le=[0., 0., z], chord=0.3, airfoil=arb.Airfoil("naca0012"))
        for z in onp.linspace(-2., -0.5, 6)
    ],
)


def lift_to_drag(leeway):
    """
    Computes the side force of the fin over its drag.

    Args:
        leeway: Leeway [deg].

    Returns: Side force over drag.
    """
    vlm = arb.HydroVortexLatticeMethod(
        arb.Appendage(name="appendage", fins=[fin]),
        arb.OperatingPoint(stw=8., leeway=leeway),
        xyz_ref=[0., 0., 0.],
        chordwise_resolution=4,
        spanwise_resolution=1,
    )
    force = vlm.run(alpha_stall=20)["F_ab"]
    return force[1] / -force[0]


def test_vlm_optimization_operating_point():
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()), onp.errstate(all="ignore"):
        warnings.simplefilter("ignore")

        opti = arb.Opti()
        leeway = opti.variable(init_guess=3., lower_bound=1., upper_bound=12.)
        opti.minimize(-lift_to_drag(leeway))
        sol = opti.solve(verbose=False, max_iter=40)

        sweep = onp.linspace(3., 9., 13)
        ratios = [float(lift_to_drag(float(angle))) for angle in sweep]

    assert float(sol(leeway)) == pytest.approx(sweep[int(onp.argmax(ratios))], abs=0.5)


if __name__ == "__main__":
    pytest.main([__file__])
