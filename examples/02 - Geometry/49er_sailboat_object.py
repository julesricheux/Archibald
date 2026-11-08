# -*- coding: utf-8 -*-
"""
Sailing equilibrium of the 49er in the horizontal plane, 2 dofs.

Boat speed and leeway are solved so that the drive of the rig balances the hull
resistance (Fx) and the side force of the rig is carried by the daggerboard,
the rudder and the hull (Fy). Same structure as ``nacra15_equilibrium.py``,
without its first solve of the immersion.

What is held fixed:
- heel (``HEEL``), meant to come from the crew position later.
- the immersion dz, an ``opti.parameter``. At dz = 0 and zero heel the hull
  displaces 192 kg, close to the 200.5 kg of the masses. The weight and the
  buoyancy are vertical in the motion frame, so they do not enter Fx and Fy.
- the sail trims and the rudder angle (``RUDDER_ANGLE``).
- yaw: Mz is not balanced, nor is the heeling moment of the rig (about
  -11.5 kN.m), so the result is the speed of a boat held upright.

At TWA 50 and TWS 20 the balance is near 18.6 kt and -2.2 deg of leeway. IPOPT
needs an initial guess close to it: from (12 kt, -5 deg) it does not converge in
50 iterations.

The daggerboard and the rudder use the symmetric ``e836`` section, so the order
of their sections (bottom to top in the DXF) does not change the side of their
lift.
"""
import archibald.numpy as np

from archibald.dynamics.hydro import holtrop, dsyhs, jones
from archibald.geometry import Hull, Sailboat, Rig, Sail, WingXSec, ThinAirfoil, Fin, Appendage, Airfoil
from archibald.geometry.mesh import ArchibaldMesh
from archibald.optimization import Opti
from archibald.performance import OperatingPoint
from archibald.toolbox.geom_utils import dxf_to_le_chords
from archibald.toolbox.mesh_utils import load_stl
from archibald.weights import MassProperties


#%% SETTINGS

TWA = 50.               # true wind angle [deg]
TWS = 20.               # true wind speed at the reference height [kt]
HELLMANN = 0.035        # wind gradient exponent
HEEL = 0.               # fixed heel, positive to starboard [deg]
DZ = 0.                 # fixed immersion, positive lifts the boat [m]
MAINSAIL_TRIM = -10.1   # [deg]
JIB_TRIM = -10.5        # [deg]
RUDDER_ANGLE = 0.       # about its leading edge, bottom to top; positive pushes to port [deg]


#%% HULL AND FITTINGS

hull = Hull(name="hull", mesh="data/49er_data/hull.stl")

wings = ArchibaldMesh(*load_stl("data/49er_data/wings.stl"))  # racks, drawn only


#%% RIG

main_le, main_chords = dxf_to_le_chords("data/49er_data/gv.dxf", 10)
jib_le, jib_chords = dxf_to_le_chords("data/49er_data/jibsail.dxf", 7)

rig = Rig(
    name="49er_rig",
    sails=[
        Sail(
            name="mainsail",
            xsecs=[
                WingXSec(
                    xyz_le=xyz,
                    chord=c,
                    airfoil=ThinAirfoil(xc=0.45, mc=0.12),
                    twist=-i / 10 * 10,
                )
                for i, (xyz, c) in enumerate(zip(main_le, main_chords))
            ],
        ),
        Sail(
            name="jibsail",
            xsecs=[
                WingXSec(
                    xyz_le=xyz,
                    chord=c,
                    airfoil=ThinAirfoil(xc=0.45, mc=0.18),
                    twist=-i / 7 * 60,
                )
                for i, (xyz, c) in enumerate(zip(jib_le, jib_chords))
            ],
        ),
    ],
)

rig["mainsail"] = rig["mainsail"].rotate_local(
    angle_deg=MAINSAIL_TRIM,
    axis=jib_le[-1] - main_le[0],
    origin=main_le[0],
)
rig["jibsail"] = rig["jibsail"].rotate_local(
    angle_deg=JIB_TRIM,
    axis=jib_le[-1] - jib_le[0],
    origin=jib_le[0],
)


#%% APPENDAGES

fins = []
for name in ("dagger", "rudder"):
    le, chords = dxf_to_le_chords(f"data/49er_data/{name}.dxf", 10)
    fin = Fin(
        name=name,
        xsecs=[
            WingXSec(
                xyz_le=xyz,
                chord=c,
                airfoil=Airfoil("e836"),
                twist=0.,
            )
            for xyz, c in zip(le, chords)
        ],
    )
    if name == "rudder":
        fin = fin.rotate_local(
            angle_deg=RUDDER_ANGLE,
            axis=le[-1] - le[0],
            origin=le[0],
        )
    fins.append(fin)

appendage = Appendage(name="49er_appendages", fins=fins)


#%% MASSES

# PLACEHOLDER, NOT SOURCED: the displacement and centre of gravity of the former
# version of this example, to be replaced by the class rules or weighings.
# Boat axes: x from the transom (hull 0 to 5 m), y to port, z up.
masses = [
    MassProperties(mass=200.5, x_cg=2., y_cg=0., z_cg=1.),  # boat and crew
]
mass_names = ["boat and crew"]


#%% SAILBOAT

boat = Sailboat(
    name="49er",
    hulls=[hull],
    rigs=[rig],
    appendages=[appendage],
    fittings=[wings],
    masses=masses,
)

resistance_method = {
    "Rf": dsyhs.compute_Rf_dsyhs,
    "Rw": dsyhs.compute_Rw_dsyhs,
    "Rtr": holtrop.compute_Rtr_holtrop,
    "L": jones.compute_L_jones,
    "Ri": jones.compute_Ri_jones,
}


#%% EQUILIBRIUM

opti = Opti()

stw = opti.variable(init_guess=18., lower_bound=1.)  # boat speed [kt]
leeway = opti.variable(init_guess=-2.5)              # positive to starboard [deg]
dz = opti.parameter(DZ)                             # immersion, positive lifts the boat [m]

op_point = OperatingPoint(
    stw=stw,
    twa=TWA,
    tws0=TWS,
    a=HELLMANN,
    heel=HEEL,
    leeway=leeway,
    dz=dz,
)

hull.compute_hydrostatics_properties(op_point)

boat.compute_weight(op_point)
boat.compute_buoyancy(op_point)
boat.compute_resistance(
    op_point,
    method=resistance_method,
    **{"Csternchoice": 1, "Bulbchoice": 0},
)
boat.compute_aerodynamics(op_point)
boat.compute_hydrodynamics(op_point)

# Weight and buoyancy are vertical: they are summed but do not enter Fx and Fy.
Ftot, Mtot = boat.compute_torsor(op_point, components=["h", "sail", "app", "b", "w"])

sol = boat.find_equilibrium(
    opti,
    Ftot,
    Mtot,
    dofs=["Fx", "Fy"],
    force_scale=100.,
    verbose=True,
    max_iter=50,
)


#%% RESULTS

print(f"\nTWA {TWA:.0f} deg, TWS {TWS:.0f} kt, heel {HEEL:+.1f} deg, dz {DZ:+.3f} m, rudder {RUDDER_ANGLE:+.1f} deg")
print(f"Boat speed : {sol(stw):.2f} kt")
print(f"Leeway     : {sol(leeway):+.2f} deg (positive to starboard)")

print("\nResiduals")
print(f"Fx : {sol(Ftot[0, 0]):+.2e} N")
print(f"Fy : {sol(Ftot[0, 1]):+.2e} N")
print(f"Fz : {sol(Ftot[0, 2]) / 9.8066:+.1f} kg (not balanced, dz fixed)")
print(f"Mz : {sol(Mtot[0, 2]):+.1f} N.m (not balanced)")

print("\nTorsor at equilibrium, motion frame")
print(f"{'':>10} {'Fx [N]':>9} {'Fy [N]':>9} {'Fz [N]':>9} {'Mx [N.m]':>10} {'Mz [N.m]':>10}")
for key, name in (("h", "hull"), ("sail", "rig"), ("app", "appendages"), ("w", "weight"), ("b", "buoyancy")):
    F = sol(boat.forces[f"F{key}"]).ravel()
    M = sol(boat.moments[f"M{key}"]).ravel()
    print(f"{name:>10} {F[0]:9.1f} {F[1]:9.1f} {F[2]:9.1f} {M[0]:10.1f} {M[2]:10.1f}")

print("\nMasses (placeholder)")
for i, name in enumerate(mass_names):
    M_i = sol(boat.moments[f"Mw_{i}"]).ravel()
    print(f"{name:>14} {masses[i].mass:6.1f} kg, Mx {M_i[0]:+8.1f} N.m")


#%% DRAWING

boat_sol = sol(boat)
boat_sol.draw_vlm(cmap="fire", draw_forces=True)
