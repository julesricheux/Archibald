"""Runs the VLM on complete wing sets, submerged far from the free surface, and checks the sign and size of the forces."""
import contextlib
import io
import warnings

import numpy as onp
import pytest

import archibald as arb

STW = 10.  # [kt]
DEPTH = 5.  # [m]
RHO = 1025.85  # [kg/m^3], default sea water
SPEED = STW * 1852. / 3600.  # [m/s]


def compute_force(planform, trim, subdivide=1):
    """
    Runs a hydro VLM on a planform, deep enough to ignore the free surface.

    Args:
        planform: The wings to analyze.
        trim: Trim [deg]. Positive is bow down, so a negative trim is a positive angle of attack.
        subdivide: Strips per section. Used instead of `spanwise_resolution`, which has to stay at 1.

    Returns: The force (Fx, Fy, Fz) in the boat motion frame [N].
    """
    wings = [wing.subdivide_sections(ratio=subdivide) for wing in planform.wings] if subdivide > 1 else planform.wings
    vlm = arb.HydroVortexLatticeMethod(
        arb.Planform(name=planform.name, xyz_ref=planform.xyz_ref, wings=wings),
        arb.OperatingPoint(stw=STW, trim=trim, dz=-DEPTH),
        chordwise_resolution=4,
        spanwise_resolution=1,
    )
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()), onp.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        result = vlm.run(alpha_stall=20)
    return onp.asarray(result["F_ab"], dtype=float).ravel()


def test_conventional():
    from archibald.dynamics.aero_3D.test_aero_3D.geometries.conventional import (
        airplane,
    )

    Fx, Fy, Fz = compute_force(airplane, trim=-5., subdivide=4)
    assert Fz > 0
    assert Fx < 0
    assert abs(Fy) < 0.01 * Fz  # the wing and stabilizer are mirrored, the fin is symmetric and at no leeway


def test_vanilla():
    from archibald.dynamics.aero_3D.test_aero_3D.geometries.vanilla import (
        airplane,
    )

    Fx, Fy, Fz = compute_force(airplane, trim=-5., subdivide=4)
    assert Fz > 0
    assert Fx < 0
    assert abs(Fy) < 0.01 * Fz


def test_flat_plate():
    from archibald.dynamics.aero_3D.test_aero_3D.geometries.flat_plate import (
        airplane,
    )

    up = compute_force(airplane, trim=-5.)
    down = compute_force(airplane, trim=5.)
    assert down[2] == pytest.approx(-up[2], rel=1e-6)

    # Lift slope of a finite wing from lifting line theory, CL = 2 pi alpha / (1 + 2 / AR) with an elliptical
    # loading (Anderson, Fundamentals of Aerodynamics, ch. 5), here for an aspect ratio of 10 and a rectangular plate.
    CL = up[2] / (0.5 * RHO * SPEED**2 * 10.)
    assert CL == pytest.approx(2 * onp.pi * onp.radians(5.) / (1 + 2 / 10.), rel=0.1)


def test_flat_plate_mirrored():
    from archibald.dynamics.aero_3D.test_aero_3D.geometries.flat_plate_mirrored import (
        airplane,
    )

    mirrored = compute_force(airplane, trim=-5.)

    # The same wing, with its sections listed one by one across the span
    listed = arb.Planform(
        name="Listed",
        xyz_ref=[0, 0, 0],
        wings=[
            arb.Wing(
                symmetric=False,
                xsecs=[
                    arb.WingXSec(xyz_le=[0, y, 0], chord=1, airfoil=arb.Airfoil("naca0008"))
                    for y in onp.linspace(-5, 5, 11)
                ],
            )
        ],
    )
    reference = compute_force(listed, trim=-5.)

    assert mirrored[2] > 0
    assert abs(mirrored[1]) < 1e-6 * mirrored[2]
    assert mirrored[0] == pytest.approx(reference[0], rel=1e-3)
    assert mirrored[2] == pytest.approx(reference[2], rel=1e-3)


def test_mirrored_wing_keeps_the_airfoil_of_each_section():
    """Strips are one side and then the other, so sections with different airfoils must not get swapped."""
    sections = [(0., "naca0008"), (1., "naca0008"), (2., "naca0024"), (3., "naca0024")]
    mirrored = arb.Planform(
        name="Mirrored",
        xyz_ref=[0, 0, 0],
        wings=[
            arb.Wing(
                symmetric=True,
                xsecs=[arb.WingXSec(xyz_le=[0, y, 0], chord=1, airfoil=arb.Airfoil(name)) for y, name in sections],
            )
        ],
    )
    vlm = arb.HydroVortexLatticeMethod(
        mirrored,
        arb.OperatingPoint(stw=STW, trim=-5., dz=-DEPTH),
        chordwise_resolution=4,
        spanwise_resolution=1,
    )
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()), onp.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        vlm.run(alpha_stall=20)

    thickness = [float(airfoil.max_thickness()) for airfoil in vlm.airfoils]
    assert thickness[:3] == pytest.approx(thickness[3:], rel=1e-6)
    assert thickness[0] < thickness[1] < thickness[2]


if __name__ == "__main__":
    pytest.main([__file__])
