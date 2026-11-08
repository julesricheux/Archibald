# -*- coding: utf-8 -*-
"""
TEMPLATE - Cruising multihull sailboat (trimaran, or catamaran with two hulls)

What it computes: in a given true wind, the boat speed, the leeway, the heel
and the draft for which the boat sails steadily:
    - the drive of the sails balances the resistance of all hulls (Fx = 0),
    - the daggerboard and rudder carry the side force of the sails (Fy = 0),
    - the buoyancy of all hulls carries the weight                 (Fz = 0),
    - the righting moment balances the heeling moment              (Mx = 0).

Each hull is a separate mesh, already placed in boat axes: the floats at their
own y. Each hull gets its own resistance, then the interaction between hulls is
added (thin ship theory, Michell). When the windward float leaves the water it
simply carries no force. On a multihull the heel stays small and the righting
moment comes from the leeward float, so the first guesses matter: start from a
small heel to leeward.

How to use it: replace every line marked TODO with your own data, then run the
file from this folder. Same structure as monohull_sailboat.py.

Complete working example: ``examples/02 - Geometry/nacra15_equilibrium.py`` (a
racing catamaran).
"""
import archibald.numpy as np

from archibald.geometry import Hull, Sailboat, Rig, Sail, Appendage, Fin, WingXSec, Airfoil, ThinAirfoil
from archibald.optimization import Opti
from archibald.performance import OperatingPoint
from archibald.weights import MassProperties


#%% 1. SETTINGS

TWS = 14.               # TODO: true wind speed at 10 m [kt]
TWA = 70.               # TODO: true wind angle [deg], > 0 for wind from starboard
MAINSAIL_TRIM = -6.     # TODO: sheeting angle [deg]
JIB_TRIM = -10.         # TODO

# TODO: one closed mesh per hull, each already placed in boat axes (metres)
HULL_FILES = {
    "main_hull": "data/my_trimaran_main_hull.stl",
    "port_float": "data/my_trimaran_port_float.stl",
    "starboard_float": "data/my_trimaran_starboard_float.stl",
}


#%% 2. HULLS

hulls = [Hull(name=name, mesh=path) for name, path in HULL_FILES.items()]

RESISTANCE_METHOD = "dsyhs"     # for each hull. Slender floats: "michell" may suit them better
INTERACTION_METHOD = "michell"  # between hulls, see archibald/dynamics/hydro/multihull.py


def wing_sections(sections, airfoil, twists=None):
    """Sections from a list of (x_le, y_le, z_le, chord), listed bottom to top [m]."""
    twists = twists if twists is not None else [0.] * len(sections)
    return [
        WingXSec(xyz_le=[x, y, z], chord=chord, airfoil=airfoil, twist=twist)
        for (x, y, z, chord), twist in zip(sections, twists)
    ]


#%% 3. APPENDAGES

DAGGERBOARD_SECTIONS = [    # TODO: tip to root, boat axes [m]
    (6.2, 0., -1.8, 0.45),
    (6.1, 0., -0.2, 0.55),
]
RUDDER_SECTIONS = [         # TODO
    (0.3, 0., -1.2, 0.30),
    (0.2, 0., -0.2, 0.40),
]

appendages = Appendage(
    name="appendages",
    fins=[
        Fin(name="daggerboard", xsecs=wing_sections(DAGGERBOARD_SECTIONS, Airfoil("naca0012"))),
        Fin(name="rudder", xsecs=wing_sections(RUDDER_SECTIONS, Airfoil("naca0012"))),
    ],
)


#%% 4. RIG

MAINSAIL_SECTIONS = [       # TODO: foot to head, boat axes [m]
    (5.0, 0., 1.5, 5.0),
    (5.4, 0., 8.0, 3.8),
    (5.8, 0., 14.0, 2.0),
    (6.0, 0., 17.0, 0.8),
]
JIB_SECTIONS = [            # TODO
    (10.5, 0., 1.4, 4.5),
    (8.8, 0., 7.0, 2.6),
    (7.2, 0., 12.5, 0.6),
]

mainsail = Sail(name="mainsail", xsecs=wing_sections(MAINSAIL_SECTIONS, ThinAirfoil(xc=0.45, mc=0.10),
                                                     twists=[0., -3., -6., -9.]))
jib = Sail(name="jib", xsecs=wing_sections(JIB_SECTIONS, ThinAirfoil(xc=0.40, mc=0.12),
                                           twists=[0., -5., -10.]))

mainsail = mainsail.rotate_local(angle_deg=MAINSAIL_TRIM, axis=[0., 0., 1.], origin=MAINSAIL_SECTIONS[0][:3])
jib = jib.rotate_local(
    angle_deg=JIB_TRIM,
    axis=np.array(JIB_SECTIONS[-1][:3]) - np.array(JIB_SECTIONS[0][:3]),
    origin=JIB_SECTIONS[0][:3],
)

rig = Rig(name="rig", sails=[mainsail, jib])


#%% 5. MASSES

masses = [
    MassProperties(mass=3500., x_cg=5.5, y_cg=0., z_cg=0.8),     # TODO: boat [kg], CoG [m]
    MassProperties(mass=600., x_cg=5.0, y_cg=0., z_cg=1.0),      # TODO: crew, stores, water
]


#%% 6. THE BOAT

boat = Sailboat(
    name="my_trimaran",
    hulls=hulls,
    rigs=[rig],
    appendages=[appendages],
    masses=masses,
)


#%% 7. UNKNOWNS

opti = Opti()

stw = opti.variable(init_guess=9., lower_bound=0.5)                     # boat speed [kt]
leeway = opti.variable(init_guess=-2.)                                  # [deg]
heel = opti.variable(init_guess=-4., lower_bound=-20., upper_bound=20.)  # [deg], small to leeward (port here)
draft = opti.variable(init_guess=0.6, lower_bound=-0.5)                 # TODO: draft of the main hull [m]

op_point = OperatingPoint(
    stw=stw,
    twa=TWA,
    tws0=TWS,
    heel=heel,
    leeway=leeway,
    dz=-draft,
)


#%% 8. FORCES AND EQUILIBRIUM

boat.compute_weight(op_point)
boat.compute_buoyancy(op_point)
boat.compute_resistance(op_point, method=RESISTANCE_METHOD, interaction=INTERACTION_METHOD)
boat.compute_aerodynamics(op_point)
boat.compute_hydrodynamics(op_point)

Ftot, Mtot = boat.compute_torsor(op_point)

sol = boat.find_equilibrium(
    opti,
    Ftot,
    Mtot,
    dofs=["Fx", "Fy", "Fz", "Mx"],
    force_scale=1e3,
    moment_scale=1e4,
)


#%% 9. RESULTS AND DRAWING

print(f"True wind  : {TWS:.0f} kt at {TWA:.0f} deg")
print(f"Boat speed : {sol(stw):.2f} kt")
print(f"Leeway     : {sol(leeway):+.2f} deg")
print(f"Heel       : {sol(heel):+.1f} deg")
for i, hull in enumerate(hulls):
    print(f"{hull.name:>16} carries {sol(boat.forces[f'Fb_{i}'][0, 2]) / 9.81:7.0f} kg")

boat_solved = sol(boat)
boat_solved.draw_vlm(draw_forces=True)
