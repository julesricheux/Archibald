import archibald as arb
from archibald.dynamics.aero_3D.test_aero_3D.geometries.conventional import (
    airplane,
)


def test_aero_buildup():
    analysis = arb.AeroBuildup(
        airplane=airplane,
        op_point=arb.OperatingPoint(),
    )
    return analysis.run()


if __name__ == "__main__":
    aero = test_aero_buildup()
    # pytest.main()



