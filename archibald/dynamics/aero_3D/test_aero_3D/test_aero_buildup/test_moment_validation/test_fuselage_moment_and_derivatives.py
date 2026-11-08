import archibald as arb
import archibald.numpy as np
from typing import Type
import pytest

rtol = 0.10


def make_fuselage(alpha_geometric=0.0, beta_geometric=0.0) -> arb.Fuselage:
    x = [0, 0.2, 0.8, 1]

    return arb.Fuselage(
        xsecs=[
            arb.FuselageXSec(
                xyz_c=[
                    xi * np.cosd(alpha_geometric) * np.cosd(beta_geometric),
                    xi * np.sind(beta_geometric),
                    xi * np.sind(-alpha_geometric),
                ],
                radius=np.where(
                    xi < 0.2,
                    0.25 * xi / 0.2,
                    np.where(xi < 0.8, 0.25, 0.25 * (1 - xi) / 0.2),
                ),
            )
            for xi in x
        ]
    ).subdivide_sections(10)


def test_derivatives_at_zero_geometric_angles(
    AeroAnalysis: Type = arb.AeroBuildup,
):

    fuselage = make_fuselage(alpha_geometric=0.0, beta_geometric=0.0)

    airplane = arb.Airplane(
        fuselages=[fuselage],
        s_ref=np.pi,
        c_ref=1,
        b_ref=0.5,
    )

    analysis = AeroAnalysis(
        airplane=airplane,
        op_point=arb.OperatingPoint(
            velocity=100,
            alpha=0,
            beta=0,
        ),
        xyz_ref=np.array([0, 0, 0]),
    )

    try:
        aero = analysis.run_with_stability_derivatives()
    except AttributeError:
        aero = analysis.run()

    print(f"Aerodynamic coefficients with {AeroAnalysis.__name__}:")
    for key in ["Cm", "Cma", "Cn", "Cnb"]:
        print(f"{key.rjust(10)}: {aero[key]:20.4f}")

    assert aero["Cm"] == pytest.approx(0, abs=1e-3)
    assert aero["Cn"] == pytest.approx(0, abs=1e-3)

    assert aero["Cma"] == pytest.approx(
        2 * fuselage.volume() / airplane.s_ref / airplane.c_ref, rel=rtol
    )
    assert aero["Cnb"] == pytest.approx(
        -2 * fuselage.volume() / airplane.s_ref / airplane.b_ref, rel=rtol
    )


def test_derivatives_at_constant_offset(
    AeroAnalysis: Type = arb.AeroBuildup,
):

    fuselage = make_fuselage(alpha_geometric=5.0, beta_geometric=10.0)

    airplane = arb.Airplane(
        fuselages=[fuselage],
        s_ref=np.pi,
        c_ref=1.0,
        b_ref=0.5,
    )

    analysis = AeroAnalysis(
        airplane=airplane,
        op_point=arb.OperatingPoint(
            velocity=100.0,
            alpha=-5.0,
            beta=-10.0,
        ),
        xyz_ref=np.array([0, 0, 0]),
    )

    try:
        aero = analysis.run_with_stability_derivatives()
    except AttributeError:
        aero = analysis.run()

    print(f"Aerodynamic coefficients with {AeroAnalysis.__name__}:")
    for key in ["Cm", "Cma", "Cn", "Cnb"]:
        print(f"{key.rjust(10)}: {aero[key]:20.4f}")

    # assert aero["Cm"] == pytest.approx(0, abs=1e-3)
    # assert aero["Cn"] == pytest.approx(0, abs=1e-3)

    assert aero["Cma"] == pytest.approx(
        2 * fuselage.volume() / airplane.s_ref / airplane.c_ref, rel=rtol
    )
    assert aero["Cnb"] == pytest.approx(
        -2 * fuselage.volume() / airplane.s_ref / airplane.b_ref, rel=rtol
    )


def test_derivatives_at_incidence(
    AeroAnalysis: Type = arb.AeroBuildup,
):

    fuselage = make_fuselage(alpha_geometric=0.0, beta_geometric=0.0)

    airplane = arb.Airplane(
        fuselages=[fuselage],
        s_ref=np.pi,
        c_ref=1.0,
        b_ref=0.5,
    )

    analysis = AeroAnalysis(
        airplane=airplane,
        op_point=arb.OperatingPoint(
            velocity=100.0,
            alpha=5.0,
            beta=5.0,
        ),
        xyz_ref=np.array([0, 0, 0]),
    )

    try:
        aero = analysis.run_with_stability_derivatives()
    except AttributeError:
        aero = analysis.run()

    print(f"Aerodynamic coefficients with {AeroAnalysis.__name__}:")
    for key in ["Cm", "Cma", "Cn", "Cnb", "CL", "CY", "CD"]:
        print(f"{key.rjust(10)}: {aero[key]:20.4f}")

    assert aero["Cm"] == pytest.approx(
        2 * fuselage.volume() / airplane.s_ref / airplane.c_ref * np.radians(5),
        rel=rtol,
    )
    assert aero["Cn"] == pytest.approx(
        -2 * fuselage.volume() / airplane.s_ref / airplane.b_ref * np.radians(5),
        rel=rtol,
    )

    assert aero["Cma"] == pytest.approx(
        2 * fuselage.volume() / airplane.s_ref / airplane.c_ref, rel=rtol
    )
    assert aero["Cnb"] == pytest.approx(
        -2 * fuselage.volume() / airplane.s_ref / airplane.b_ref, rel=rtol
    )


def test_cambered_fuselage(
    AeroAnalysis: Type = arb.AeroBuildup,
):
    fuselage = arb.Fuselage(
        xsecs=[
            arb.FuselageXSec(
                xyz_c=[xi, 0, xi * (1 - xi) * 4 * 0.15], radius=xi**0.5 * (1 - xi) * 0.2
            )
            for xi in np.sinspace(0, 1, 20)
        ]
    )
    # fuselage.draw_three_view()

    airplane = arb.Airplane(
        fuselages=[fuselage],
        s_ref=1,
        c_ref=1,
        b_ref=1,
    )

    analysis = AeroAnalysis(
        airplane=airplane,
        op_point=arb.OperatingPoint(
            velocity=100,
            alpha=5,
            beta=5,
        ),
        xyz_ref=np.array([0.5, 0, 0]),
    )

    try:
        aero = analysis.run_with_stability_derivatives()
    except AttributeError:
        aero = analysis.run()

    print(f"Aerodynamic coefficients with {AeroAnalysis.__name__}:")
    for key in ["Cm", "Cma", "Cn", "Cnb"]:
        print(f"{key.rjust(10)}: {aero[key]:20.4f}")

    # assert aero["Cm"] == pytest.approx(0, abs=1e-3)
    # assert aero["Cn"] == pytest.approx(0, abs=1e-3)

    assert aero["Cma"] == pytest.approx(
        2 * fuselage.volume() / airplane.s_ref / airplane.c_ref,
        rel=0.5,  # this is not-strictly-speaking exact, hence the large tolerance
    )
    assert aero["Cnb"] == pytest.approx(
        -2 * fuselage.volume() / airplane.s_ref / airplane.b_ref,
        rel=0.5,  # this is not-strictly-speaking exact, hence the large tolerance
    )


def test_fuselage_with_base_drag(
    AeroAnalysis: Type = arb.AeroBuildup,
):
    def r(x):  # radius as a function of distance along the fuselage, x
        return 0.1 * x**0.5 * (2 - x)

    def z(
        x,
    ):  # vertical displacement (in geometry axes) of the centerline, as a function of distance along fuselage, x
        return 0.15 * x * (2 - x)

    def dzdx(x):  # derivative of z(x) with respect to x
        return 0.15 * ((2 - x) - x)

    def A(x):  # cross-sectional area as a function of distance along fuselage, x
        return np.pi * r(x) ** 2

    fuselage = arb.Fuselage(
        xsecs=[
            arb.FuselageXSec(xyz_c=[xi, 0, z(xi)], radius=r(xi))
            for xi in np.sinspace(0, 1, 100)
        ]
    )
    fuselage.draw_three_view()

    airplane = arb.Airplane(
        fuselages=[fuselage],
        s_ref=1,
        c_ref=1,
        b_ref=1,
    )

    analysis = AeroAnalysis(
        airplane=airplane,
        op_point=arb.OperatingPoint(
            velocity=100,
            alpha=0,
            beta=0,
        ),
        xyz_ref=np.array([0.0, 0, 0]),
    )

    try:
        aero = analysis.run_with_stability_derivatives()
    except AttributeError:
        aero = analysis.run()

    print(f"Aerodynamic coefficients with {AeroAnalysis.__name__}:")
    for key in ["CL", "Cm", "Cma", "Cn", "Cnb"]:
        print(f"{key.rjust(10)}: {aero[key]:20.4f}")

    # TODO add assertions

    ##### Slender body theory
    l = 1  # length of fuselage
    alpha_rad = np.radians(analysis.op_point.alpha)

    from scipy import integrate

    def alpha_tilde(x):
        return alpha_rad - dzdx(x)

    CL = 2 / airplane.s_ref * A(l) * alpha_tilde(l)
    print(f"CL_slb: {CL}")
    assert aero["CL"] == pytest.approx(CL, abs=0.01)

    Cm = (
        2
        / airplane.s_ref
        / airplane.c_ref
        * (
            (fuselage.volume() - l * A(l)) * alpha_rad
            + l * A(l) * dzdx(l)
            - integrate.quad(lambda xi: A(xi) * dzdx(xi), 0, l)[0]
        )
    )

    print(f"Cm_slb: {Cm}")
    assert aero["Cm"] == pytest.approx(
        Cm, rel=0.5
    )  # this is not-strictly-speaking exact, hence the large tolerance
    assert aero["Cn"] == pytest.approx(0, abs=1e-3)

    Cma = 2 / airplane.s_ref / airplane.c_ref * (fuselage.volume() - l * A(l))

    # print(f"Cma_slb: {Cma}")
    # assert aero["Cma"] == pytest.approx(
    #     Cma,
    #     rel=0.5
    # )
    # assert aero["Cnb"] == pytest.approx(
    #     -2 * fuselage.volume() / airplane.s_ref / airplane.b_ref,
    #     rel=0.5 # this is not-strictly-speaking exact, hence the large tolerance
    # )


if __name__ == "__main__":
    pass
    # arb.Airplane(fuselages=[make_fuselage()]).draw_three_view()
    # # test_derivatives_at_zero_geometric_angles(arb.AVL)
    # test_derivatives_at_zero_geometric_angles(arb.AeroBuildup)
    # # test_derivatives_at_constant_offset(arb.AVL)
    # test_derivatives_at_constant_offset(arb.AeroBuildup)
    # # test_derivatives_at_incidence(arb.AVL)
    # test_derivatives_at_incidence(arb.AeroBuildup)
    # # test_cambered_fuselage(arb.AVL)
    # test_cambered_fuselage(arb.AeroBuildup)
    # test_fuselage_with_base_drag(arb.AVL)
    test_fuselage_with_base_drag(arb.AeroBuildup)



