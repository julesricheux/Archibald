import archibald as arb


def test_conventional():
    from archibald.dynamics.aero_3D.test_aero_3D.geometries.conventional import (
        airplane,
    )

    analysis = arb.VortexLatticeMethod(
        airplane=airplane,
        op_point=arb.OperatingPoint(alpha=10),
    )
    return analysis.run()


def test_vanilla():
    from archibald.dynamics.aero_3D.test_aero_3D.geometries.vanilla import (
        airplane,
    )

    analysis = arb.VortexLatticeMethod(
        airplane=airplane,
        op_point=arb.OperatingPoint(alpha=10),
    )
    return analysis.run()


def test_flat_plate():
    from archibald.dynamics.aero_3D.test_aero_3D.geometries.flat_plate import (
        airplane,
    )

    analysis = arb.VortexLatticeMethod(
        airplane=airplane,
        op_point=arb.OperatingPoint(alpha=10),
    )
    return analysis.run()


def test_flat_plate_mirrored():
    from archibald.dynamics.aero_3D.test_aero_3D.geometries.flat_plate_mirrored import (
        airplane,
    )

    analysis = arb.VortexLatticeMethod(
        airplane=airplane,
        op_point=arb.OperatingPoint(alpha=10),
        spanwise_resolution=1,
        chordwise_resolution=3,
    )
    return analysis.run()


if __name__ == "__main__":
    # test_conventional()
    # test_vanilla()
    # test_flat_plate()['CL']
    # test_flat_plate_mirrored()
    # pytest.main()
    from archibald.dynamics.aero_3D.test_aero_3D.geometries.conventional import (
        airplane,
    )

    analysis = arb.VortexLatticeMethod(
        airplane=airplane,
        op_point=arb.OperatingPoint(alpha=10),
    )
    aero = analysis.run()
    analysis.draw()



