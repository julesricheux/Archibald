import archibald as arb
import archibald.numpy as np
import pandas as pd
import pytest

af = arb.Airfoil(
    "naca0012",
    CL_function=lambda alpha, Re, mach: 2 * np.pi * np.radians(alpha),
    CD_function=lambda alpha, Re, mach: np.zeros_like(alpha),
    CM_function=lambda alpha, Re, mach: np.zeros_like(alpha),
)

solvers = [
    # arb.AVL,
    arb.VortexLatticeMethod,
    arb.AeroBuildup,
]


def test_horizontal():
    airplane_horizontal = arb.Airplane(
        wings=[
            arb.Wing(
                symmetric=True,
                color="blue",
                xsecs=[
                    arb.WingXSec(xyz_le=[0, 0, 0], chord=1, airfoil=af),
                    arb.WingXSec(xyz_le=[0, 1, 0], chord=1, airfoil=af),
                ],
            )
        ]
    )
    # airplane_horizontal.draw_three_view()

    op_point_horiz = arb.OperatingPoint(velocity=100, alpha=5)

    horiz = (
        pd.DataFrame(
            {
                solver.__name__: solver(airplane_horizontal, op_point_horiz).run()
                for solver in solvers
            }
        )
        .dropna()
        .loc[["L", "CL", "CD"], :]
    )

    print("\nHorizontal\n" + "-" * 80)
    print(horiz)

    assert horiz["VortexLatticeMethod"]["L"] == pytest.approx(
        horiz["AeroBuildup"]["L"], rel=0.1
    )


def test_v_tail():
    airplane_v_tail = arb.Airplane(
        wings=[
            arb.Wing(
                symmetric=True,
                color="purple",
                xsecs=[
                    arb.WingXSec(xyz_le=[0, 0, 0], chord=1, airfoil=af),
                    arb.WingXSec(xyz_le=[0, 1, 1], chord=1, airfoil=af),
                ],
            )
        ]
    )

    # airplane_v_tail.draw_three_view()

    op_point_v_tail = arb.OperatingPoint(velocity=100, alpha=5, beta=5)

    v_tail = (
        pd.DataFrame(
            {
                solver.__name__: solver(airplane_v_tail, op_point_v_tail).run()
                for solver in solvers
            }
        )
        .dropna()
        .loc[["L", "CL", "Y", "CY", "CD"], :]
    )

    print("\nV-Tail\n" + "-" * 80)
    print(v_tail)

    # TODO add fix for force direction, and add test


def test_vertical():
    airplane_vertical = arb.Airplane(
        wings=[
            arb.Wing(
                symmetric=True,
                color="red",
                xsecs=[
                    arb.WingXSec(xyz_le=[0, 1, 0], chord=1, airfoil=af),
                    arb.WingXSec(xyz_le=[0, 1, 1], chord=1, airfoil=af),
                ],
            )
        ]
    )

    # airplane_vertical.draw_three_view()

    op_point_vert = arb.OperatingPoint(velocity=100, beta=5)

    vert = (
        pd.DataFrame(
            {
                solver.__name__: solver(airplane_vertical, op_point_vert).run()
                for solver in solvers
            }
        )
        .dropna()
        .loc[["Y", "CY", "CD"], :]
    )

    print("\nVertical\n" + "-" * 80)
    print(vert)

    assert vert["VortexLatticeMethod"]["Y"] == pytest.approx(
        vert["AeroBuildup"]["Y"], rel=0.1
    )


if __name__ == "__main__":
    test_horizontal()
    test_v_tail()
    test_vertical()

    pytest.main()



