import archibald as arb

airfoil = arb.Airfoil("naca0008")

airplane = arb.Airplane(
    name="Flat Plate",
    xyz_ref=[0, 0, 0],
    wings=[
        arb.Wing(
            name="Wing",
            symmetric=True,
            xsecs=[
                arb.WingXSec(
                    xyz_le=[0, 0, 0],
                    chord=1,
                    twist=0,
                    airfoil=airfoil,
                ),
                arb.WingXSec(
                    xyz_le=[0, 5, 0],
                    chord=1,
                    twist=0,
                    airfoil=airfoil,
                ),
            ],
        )
    ],
)

if __name__ == "__main__":
    airplane.draw()



