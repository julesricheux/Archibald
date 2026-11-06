import archibald as arb
import archibald.numpy as np

sd7037 = arb.Airfoil("sd7037")

airplane = arb.LiftingSet(
    name="Vanilla",
    xyz_ref=[0.5, 0, 0],
    s_ref=9,
    c_ref=0.9,
    b_ref=10,
    wings=[
        arb.Wing(
            name="Wing",
            symmetric=True,
            xsecs=[
                arb.WingXSec(
                    xyz_le=[0, 0, 0],
                    chord=1,
                    twist=2,
                    airfoil=sd7037,
                ),
                arb.WingXSec(
                    xyz_le=[0.2, 5, 1],
                    chord=0.6,
                    twist=2,
                    airfoil=sd7037,
                ),
            ],
        ),
        arb.Wing(
            name="H-stab",
            symmetric=True,
            xsecs=[
                arb.WingXSec(
                    xyz_le=[0, 0, 0], chord=0.7, airfoil=arb.Airfoil("naca0012")
                ),
                arb.WingXSec(
                    xyz_le=[0.14, 1.25, 0], chord=0.42, airfoil=arb.Airfoil("naca0012")
                ),
            ],
        ).translate([4, 0, 0]),
        arb.Wing(
            name="V-stab",
            xsecs=[
                arb.WingXSec(
                    xyz_le=[0, 0, 0], chord=0.7, airfoil=arb.Airfoil("naca0012")
                ),
                arb.WingXSec(
                    xyz_le=[0.14, 0, 1], chord=0.42, airfoil=arb.Airfoil("naca0012")
                ),
            ],
        ).translate([4, 0, 0]),
    ],
    fuselages=[
        arb.Fuselage(
            name="Fuselage",
            xsecs=[
                arb.FuselageXSec(
                    xyz_c=[xi * 5 - 0.5, 0, 0],
                    radius=arb.Airfoil("naca0024").local_thickness(x_over_c=xi),
                )
                for xi in np.cosspace(0, 1, 30)
            ],
        )
    ],
)

if __name__ == "__main__":
    airplane.draw()



