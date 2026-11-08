import archibald as arb
import archibald.numpy as np

airfoil = arb.Airfoil("naca0008")

airplane = arb.Planform(
    name="Flat Plate",
    xyz_ref=[0, 0, 0],
    wings=[
        arb.Wing(
            name="Wing",
            symmetric=True,
            xsecs=[
                arb.WingXSec(
                    xyz_le=[0, y, 0],
                    chord=1,
                    twist=0,
                    airfoil=airfoil,
                )
                for y in np.linspace(0, 5, 6)
            ],
        )
    ],
)

if __name__ == "__main__":
    airplane.draw()
