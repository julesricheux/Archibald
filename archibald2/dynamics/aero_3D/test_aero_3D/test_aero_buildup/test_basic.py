import archibald2 as asb
from archibald2.dynamics.aero_3D.test_aero_3D.geometries.conventional import (
    airplane,
)


def test_aero_buildup():
    analysis = asb.AeroBuildup(
        airplane=airplane,
        op_point=asb.OperatingPoint(),
    )
    return analysis.run()


if __name__ == "__main__":
    aero = test_aero_buildup()
    # pytest.main()
