# -*- coding: utf-8 -*-
"""
TEMPLATE - Cruising monohull sailboat

What it computes: in a given true wind, the boat speed, the leeway, the heel
and the draft for which the boat sails steadily:
    - the drive of the sails balances the resistance              (Fx = 0),
    - the keel and rudder carry the side force of the sails       (Fy = 0),
    - the buoyancy carries the weight                             (Fz = 0),
    - the righting moment balances the heeling moment             (Mx = 0).
Trim and yaw (rudder balance) are not solved here.

The sails are rigid wings computed with a vortex lattice method, and their
section is a thin cambered airfoil. Sail trim is a setting: try a few values to
find the fastest one.

How to use it: replace every line marked TODO with your own data, then run the
file from this folder. Start with heel and draft fixed (see step 7), then free
them once the speed and leeway converge.

Complete working examples: ``examples/02 - Geometry/49er_sailboat_object.py``
(a skiff, 2 unknowns) and ``molenez_hybrid.py`` (a cargo sailboat, 5 unknowns).
"""
import archibald.numpy as np

from archibald.geometry import Hull, Sailboat, Rig, Sail, Appendage, Fin, WingXSec, Airfoil, ThinAirfoil
from archibald.optimization import Opti
from archibald.performance import OperatingPoint
from archibald.weights import MassProperties


#%% 1. SETTINGS

TWS = 12.               # TODO: true wind speed at 10 m [kt]
TWA = 60.               # TODO: true wind angle [deg], > 0 for wind from starboard
MAINSAIL_TRIM = -8.     # TODO: sheeting angle of the mainsail [deg], < 0 turns its leech to port
JIB_TRIM = -12.         # TODO: sheeting angle of the jib [deg]

HULL_FILE = "data/my_cruiser_hull.stl"      # TODO: closed hull mesh, metres, boat axes


#%% 2. HULL

hull = Hull(name="hull", mesh=HULL_FILE)

RESISTANCE_METHOD = "dsyhs"     # Delft Systematic Yacht Hull Series, see archibald/dynamics/hydro/dsyhs.py
RESISTANCE_OPTIONS = {}


def wing_sections(sections, airfoil, twists=None):
    """Sections from a list of (x_le, y_le, z_le, chord), listed bottom to top [m]."""
    twists = twists if twists is not None else [0.] * len(sections)
    return [
        WingXSec(xyz_le=[x, y, z], chord=chord, airfoil=airfoil, twist=twist)
        for (x, y, z, chord), twist in zip(sections, twists)
    ]


#%% 3. APPENDAGES

# TODO: leading edge point and chord of a few sections of each appendage, from
# the tip (bottom) to the root (top), boat axes [m].
# Or read a planform drawing:
#     from archibald.toolbox.geom_utils import dxf_to_le_chords
#     le, chords = dxf_to_le_chords("data/keel.dxf", 10)
KEEL_SECTIONS = [
    (4.6, 0., -1.9, 0.9),
    (4.3, 0., -1.0, 1.2),
    (4.1, 0., -0.3, 1.5),
]
RUDDER_SECTIONS = [
    (0.9, 0., -1.5, 0.35),
    (0.8, 0., -0.8, 0.45),
    (0.7, 0., -0.1, 0.55),
]

keel = Fin(name="keel", xsecs=wing_sections(KEEL_SECTIONS, Airfoil("naca0012")))       # TODO: sections
rudder = Fin(name="rudder", xsecs=wing_sections(RUDDER_SECTIONS, Airfoil("naca0012")))

appendages = Appendage(name="appendages", fins=[keel, rudder])


#%% 4. RIG

# TODO: sections of each sail, from the foot to the head, boat axes [m]
MAINSAIL_SECTIONS = [
    (3.4, 0., 1.8, 4.0),
    (3.6, 0., 6.0, 3.0),
    (3.8, 0., 10.0, 1.8),
    (4.0, 0., 13.5, 0.5),
]
JIB_SECTIONS = [
    (8.6, 0., 1.5, 4.2),
    (7.4, 0., 6.0, 2.8),
    (6.2, 0., 10.0, 1.4),
    (5.4, 0., 12.5, 0.4),
]

# Thin cambered section: xc = position of the maximum camber, mc = camber, as fractions of the chord
mainsail = Sail(name="mainsail", xsecs=wing_sections(MAINSAIL_SECTIONS, ThinAirfoil(xc=0.45, mc=0.10),
                                                     twists=[0., -3., -6., -9.]))    # TODO: twist [deg]
jib = Sail(name="jib", xsecs=wing_sections(JIB_SECTIONS, ThinAirfoil(xc=0.40, mc=0.12),
                                           twists=[0., -4., -8., -12.]))

# Sheeting: each sail turns about its luff
mainsail = mainsail.rotate_local(angle_deg=MAINSAIL_TRIM, axis=[0., 0., 1.], origin=MAINSAIL_SECTIONS[0][:3])
jib = jib.rotate_local(
    angle_deg=JIB_TRIM,
    axis=np.array(JIB_SECTIONS[-1][:3]) - np.array(JIB_SECTIONS[0][:3]),    # the forestay
    origin=JIB_SECTIONS[0][:3],
)

rig = Rig(name="rig", sails=[mainsail, jib])


#%% 5. MASSES

masses = [
    MassProperties(mass=4500., x_cg=4.3, y_cg=0., z_cg=0.3),     # TODO: boat [kg], CoG [m]
    MassProperties(mass=2000., x_cg=4.4, y_cg=0., z_cg=-1.6),    # TODO: keel ballast
    MassProperties(mass=320., x_cg=2.0, y_cg=0.0, z_cg=1.2),     # TODO: crew (move y_cg to windward to hike)
]


#%% 6. THE BOAT

boat = Sailboat(
    name="my_cruiser",
    hulls=[hull],
    rigs=[rig],
    appendages=[appendages],
    masses=masses,
)


#%% 7. UNKNOWNS

opti = Opti()

stw = opti.variable(init_guess=6., lower_bound=0.5)                     # boat speed [kt]
leeway = opti.variable(init_guess=-3.)                                  # [deg], < 0 with wind from starboard
heel = opti.variable(init_guess=-15., lower_bound=-45., upper_bound=45.)  # [deg], < 0 with wind from starboard
draft = opti.variable(init_guess=1.4, lower_bound=0.1)                  # TODO: draft of the canoe body [m]
# To start with fewer unknowns, replace a variable by a fixed number, for instance
# heel = -15., and remove its line from the dofs below (here "Mx").

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
boat.compute_resistance(op_point, method=RESISTANCE_METHOD, **RESISTANCE_OPTIONS)
boat.compute_aerodynamics(op_point)     # sails
boat.compute_hydrodynamics(op_point)    # keel and rudder
# boat.compute_windage(op_point, method="orc")   # optional: windage of the hull and crew

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
print(f"Draft      : {sol(draft):.3f} m")
print(f"Yaw moment : {sol(Mtot[0, 2]):+.0f} N.m (not balanced: > 0 turns the bow to port)")

boat_solved = sol(boat)
boat_solved.draw_vlm(draw_forces=True)  # sails and appendages coloured by pressure, with the forces
