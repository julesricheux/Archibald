# -*- coding: utf-8 -*-
"""
TEMPLATE - Racing monohull with a canting keel and foils (IMOCA, Class40, Mini 6.50...)

What it computes: in a given true wind, for a chosen keel cant and foil rake,
the boat speed, the leeway, the heel and the ride height:
    - the drive of the sails balances the drag                                  (Fx = 0),
    - the leeward foil, the keel fin and the rudders carry the side force       (Fy = 0),
    - buoyancy and foil lift carry the weight                                   (Fz = 0),
    - the righting moment of the canted bulb, crew, ballast and foil balances the sails (Mx = 0).
Trim and yaw are not solved. Try several cant angles and foil rakes to find the
fastest set-up.

The keel fin turns about its pivot by ``CANT_ANGLE``, and the bulb follows the
tip of the fin. Only the leeward foil is deployed. The sails, the fin, the
rudder and the foil are lifting surfaces computed with a vortex lattice
method. Where a surface crosses the water, its strips blend from water to air
density over about 1 cm.

Scale: the numbers below are those of a boat of about 18 m (IMOCA). For a Mini
6.50, divide lengths by about 2.8 and masses by about 8.

How to use it: replace every line marked TODO with your own data, then run the
file from this folder. Start with heel and ride height fixed, solving only speed
and leeway, then free them one at a time.
"""
import archibald.numpy as np

from archibald.geometry import Hull, Sailboat, Rig, Sail, Appendage, Fin, WingXSec, Airfoil, ThinAirfoil
from archibald.optimization import Opti
from archibald.performance import OperatingPoint
from archibald.weights import MassProperties


#%% 1. SETTINGS

TWS = 16.               # TODO: true wind speed at 10 m [kt]
TWA = 100.              # TODO: true wind angle [deg], > 0 for wind from starboard (leeward is port)
MAINSAIL_TRIM = -10.    # TODO: sheeting angles [deg]
HEADSAIL_TRIM = -16.

CANT_ANGLE = 30.        # TODO: keel cant to windward [deg]
KEEL_PIVOT = [8.0, 0., 0.2]                 # TODO: pivot of the canting keel, boat axes [m]
FOIL_RAKE = 2.          # TODO: foil rake, leading edge up [deg]

HULL_FILE = "data/my_imoca_hull.stl"        # TODO: closed hull mesh, metres, boat axes


#%% 2. HULL

hull = Hull(name="hull", mesh=HULL_FILE)

RESISTANCE_METHOD = "dsyhs"     # TODO: check the range of validity for your hull (dsyhs.py)


def wing_sections(sections, airfoil, twists=None):
    """Sections from a list of (x_le, y_le, z_le, chord) [m], in order along the surface."""
    twists = twists if twists is not None else [0.] * len(sections)
    return [
        WingXSec(xyz_le=[x, y, z], chord=chord, airfoil=airfoil, twist=twist)
        for (x, y, z, chord), twist in zip(sections, twists)
    ]


#%% 3. APPENDAGES

# Keel fin, upright: from the bulb (bottom) to the pivot (top), boat axes [m]
KEEL_SECTIONS = [               # TODO
    (8.0, 0., -4.3, 0.55),
    (8.0, 0., -2.0, 0.65),
    (8.0, 0., 0.0, 0.75),
]
keel = Fin(name="keel_fin", xsecs=wing_sections(KEEL_SECTIONS, Airfoil("naca0012")))
# Canting to windward. Wind from starboard: windward is -y. Check on the drawing
# that the fin swings to windward, and change the sign if it does not.
keel = keel.rotate_local(angle_deg=-CANT_ANGLE, axis=[1., 0., 0.], origin=KEEL_PIVOT)
bulb_position = keel.xsecs[0].xyz_le      # the bulb follows the tip of the fin

RUDDER_SECTIONS = [             # TODO: leeward rudder (twin rudders: add the windward one)
    (0.6, 1.2, -2.0, 0.35),
    (0.5, 1.2, -0.2, 0.45),
]
rudder = Fin(name="rudder", xsecs=wing_sections(RUDDER_SECTIONS, Airfoil("naca0012")))

# Leeward foil (port here): from where it leaves the hull, outwards and down to
# the tip, boat axes [m]. TODO: your foil shape.
FOIL_SECTIONS = [
    (9.5, 2.4, 0.3, 0.60),
    (9.5, 3.4, -0.3, 0.55),
    (9.5, 4.2, -1.2, 0.45),
    (9.5, 4.4, -2.0, 0.35),
]
foil = Fin(name="foil", xsecs=wing_sections(FOIL_SECTIONS, Airfoil("naca0012")))
foil = foil.rotate_local(
    angle_deg=-FOIL_RAKE,       # a negative rotation about +y raises the leading edge
    axis=[0., 1., 0.],
    origin=FOIL_SECTIONS[0][:3],
)

appendages = Appendage(name="appendages", fins=[keel, rudder, foil])


#%% 4. RIG

MAINSAIL_SECTIONS = [           # TODO: foot to head, boat axes [m]
    (5.8, 0., 2.5, 7.5),
    (6.2, 0., 12.0, 6.0),
    (6.6, 0., 21.0, 3.5),
    (7.0, 0., 28.0, 1.5),
]
HEADSAIL_SECTIONS = [           # TODO
    (16.5, 0., 2.0, 8.0),
    (13.5, 0., 12.0, 4.5),
    (10.5, 0., 22.0, 1.2),
]

mainsail = Sail(name="mainsail", xsecs=wing_sections(MAINSAIL_SECTIONS, ThinAirfoil(xc=0.45, mc=0.10),
                                                     twists=[0., -3., -7., -11.]))
headsail = Sail(name="headsail", xsecs=wing_sections(HEADSAIL_SECTIONS, ThinAirfoil(xc=0.40, mc=0.14),
                                                     twists=[0., -6., -12.]))

mainsail = mainsail.rotate_local(angle_deg=MAINSAIL_TRIM, axis=[0., 0., 1.], origin=MAINSAIL_SECTIONS[0][:3])
headsail = headsail.rotate_local(
    angle_deg=HEADSAIL_TRIM,
    axis=np.array(HEADSAIL_SECTIONS[-1][:3]) - np.array(HEADSAIL_SECTIONS[0][:3]),
    origin=HEADSAIL_SECTIONS[0][:3],
)

rig = Rig(name="rig", sails=[mainsail, headsail])


#%% 5. MASSES

masses = [
    MassProperties(mass=5000., x_cg=8.5, y_cg=0., z_cg=1.0),     # TODO: hull, deck, rig [kg], CoG [m]
    MassProperties(mass=3000.,                                   # TODO: bulb, at the tip of the canted fin
                   x_cg=bulb_position[0] + 0.3,
                   y_cg=bulb_position[1],
                   z_cg=bulb_position[2] - 0.2),
    MassProperties(mass=500., x_cg=6.0, y_cg=-1.5, z_cg=0.8),    # TODO: skipper, food, sails stacked to windward
    # MassProperties(mass=1000., x_cg=7.0, y_cg=-2.0, z_cg=0.5),  # water ballast, windward tank
]


#%% 6. THE BOAT

boat = Sailboat(
    name="my_imoca",
    hulls=[hull],
    rigs=[rig],
    appendages=[appendages],
    masses=masses,
)


#%% 7. UNKNOWNS

opti = Opti()

stw = opti.variable(init_guess=18., lower_bound=1.)                     # boat speed [kt]
leeway = opti.variable(init_guess=-2.)                                  # [deg]
heel = opti.variable(init_guess=-12., lower_bound=-35., upper_bound=10.)  # [deg], to port (leeward) here
dz = opti.variable(init_guess=-0.6, lower_bound=-2., upper_bound=1.)    # TODO: ride height, -draft of the canoe body [m]
# To start: heel = -12. and dz = -0.6 as fixed numbers, and dofs=["Fx", "Fy"]

op_point = OperatingPoint(
    stw=stw,
    twa=TWA,
    tws0=TWS,
    heel=heel,
    leeway=leeway,
    dz=dz,
)


#%% 8. FORCES AND EQUILIBRIUM

boat.compute_weight(op_point)
boat.compute_buoyancy(op_point)
boat.compute_resistance(op_point, method=RESISTANCE_METHOD)
boat.compute_aerodynamics(op_point)     # sails
boat.compute_hydrodynamics(op_point)    # keel fin, rudder and foil

Ftot, Mtot = boat.compute_torsor(op_point)

sol = boat.find_equilibrium(
    opti,
    Ftot,
    Mtot,
    dofs=["Fx", "Fy", "Fz", "Mx"],
    force_scale=1e4,
    moment_scale=1e5,
)


#%% 9. RESULTS AND DRAWING

weight = sum(m.mass for m in masses) * 9.81
print(f"True wind       : {TWS:.0f} kt at {TWA:.0f} deg, cant {CANT_ANGLE:.0f} deg, foil rake {FOIL_RAKE:+.1f} deg")
print(f"Boat speed      : {sol(stw):.2f} kt")
print(f"Leeway          : {sol(leeway):+.2f} deg")
print(f"Heel            : {sol(heel):+.1f} deg")
print(f"Ride height dz  : {sol(dz):+.3f} m")
print(f"Weight on foils : {100 * sol(boat.forces['Fapp'][0, 2]) / weight:.0f} %")

boat_solved = sol(boat)
boat_solved.draw_vlm(draw_forces=True)
