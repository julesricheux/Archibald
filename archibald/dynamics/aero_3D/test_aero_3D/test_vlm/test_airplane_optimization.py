"""Optimizes the taper and angle of a mirrored wing through the VLM, and compares with a search over the same variables."""
import contextlib
import io
import warnings

import numpy as onp
import pytest

import archibald as arb


def lift_to_drag(trim, taper_ratio):
    """
    Computes the lift to drag ratio of a mirrored tapered wing, deep under water.

    Args:
        trim: Trim [deg]. A negative trim is a positive angle of attack.
        taper_ratio: Tip chord over root chord.

    Returns: Lift over drag.
    """
    planform = arb.Planform(
        wings=[
            arb.Wing(
                symmetric=True,
                xsecs=[
                    arb.WingXSec(
                        xyz_le=[-0.25 * (1 - (1 - taper_ratio) * eta), 2 * eta, 0],
                        chord=1 - (1 - taper_ratio) * eta,
                        airfoil=arb.Airfoil("naca0012"),
                    )
                    for eta in onp.linspace(0, 1, 5)
                ],
            )
        ]
    )
    vlm = arb.HydroVortexLatticeMethod(
        planform,
        arb.OperatingPoint(stw=10., trim=trim, dz=-5.),
        chordwise_resolution=3,
        spanwise_resolution=1,
    )
    force = vlm.run(alpha_stall=20)["F_ab"]
    return force[2] / -force[0]


def test_airplane_optimization():
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()), onp.errstate(all="ignore"):
        warnings.simplefilter("ignore")

        opti = arb.Opti()
        trim = opti.variable(init_guess=-4., lower_bound=-15., upper_bound=-1.)
        taper_ratio = opti.variable(init_guess=0.7, lower_bound=0.2, upper_bound=1.)
        opti.minimize(-lift_to_drag(trim, taper_ratio))
        sol = opti.solve(verbose=False, max_iter=60)

        best = float(lift_to_drag(float(sol(trim)), float(sol(taper_ratio))))
        search = [
            float(lift_to_drag(t, r))
            for t in onp.linspace(-12., -2., 6)
            for r in onp.linspace(0.2, 1., 5)
        ]

    assert best >= max(search) - 0.01 * abs(max(search))


if __name__ == "__main__":
    pytest.main([__file__])
