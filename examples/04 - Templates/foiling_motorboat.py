# -*- coding: utf-8 -*-
"""
TEMPLATE - Motorboat on foils (RIB with a U-foil forward and a T-foil on the leg)

What it computes: at a given speed, the thrust, the ride height and the trim
for which the boat is in equilibrium:
    - the thrust balances the drag of the hull and the foils     (Fx = 0),
    - buoyancy, hull lift and foil lift carry the weight          (Fz = 0),
    - the trim settles where the pitching moment is zero          (My = 0).
Run it at several speeds to follow the take-off, from a boat floating on its
hull to a hull fully out of the water.

Ride height: a deep foil at a fixed angle gives the same lift at any height, so
once the hull is dry nothing sets the ride height and the problem has no
unique answer. Real foilers control the foil angle (a wand, a sensor and a
flap). Here the angle of the front foil decreases linearly as the boat rises:
choose ``RIDE_GAIN`` from your control system. See
``examples/02 - Geometry/49er_foiling.py`` for a working take-off.

The foils are lifting surfaces computed with a vortex lattice method. Where a
foil crosses the water surface, its strips blend from water to air density over
about 1 cm. No ventilation, cavitation or free surface effect on the foil
is modelled.

How to use it: replace every line marked TODO with your own data, then run the
file from this folder.
"""
import archibald.numpy as np

from archibald.geometry import Hull, Sailboat, Appendage, Fin, WingXSec, Airfoil
from archibald.optimization import Opti
from archibald.performance import OperatingPoint
from archibald.weights import MassProperties


#%% 1. SETTINGS

SPEED = 22.                                 # TODO: boat speed [kt]
HULL_FILE = "data/my_rib_hull.stl"          # TODO: closed hull mesh, metres, boat axes
DEADRISE = 22.                              # TODO: deadrise at the transom [deg]
THRUST_POINT = [-0.3, 0., -0.6]             # TODO: propeller position, boat axes [m]

FRONT_FOIL_INCIDENCE = 3.   # TODO: angle of the front foil at dz = 0, leading edge up [deg]
RIDE_GAIN = 4.              # TODO: front foil angle lost per metre of rise [deg/m]
REAR_FOIL_INCIDENCE = 1.    # TODO: angle of the rear foil, leading edge up [deg]


#%% 2. HULL

hull = Hull(name="hull", mesh=HULL_FILE)


#%% 3. UNKNOWNS
# (declared before the foils here, because the front foil angle depends on the ride height)

opti = Opti()

dz = opti.variable(init_guess=0., lower_bound=-0.6, upper_bound=1.5)    # rise of the boat [m]: > 0 lifts it
trim = opti.variable(init_guess=-1., lower_bound=-8., upper_bound=5.)   # positive bow down [deg]
thrust = opti.variable(init_guess=2000., lower_bound=0.)                # [N]

op_point = OperatingPoint(
    stw=SPEED,
    dz=dz,
    trim=trim,
)


#%% 4. FOILS

def wing_sections(sections, airfoil):
    """Sections from a list of (x_le, y_le, z_le, chord) [m], in order along the foil."""
    return [WingXSec(xyz_le=[x, y, z], chord=chord, airfoil=airfoil, twist=0.) for x, y, z, chord in sections]


# TODO: the U-foil, as one surface from the top of the port leg, down, across,
# and up the starboard leg, boat axes [m]. With a cambered section, list the
# horizontal part from port (y > 0) to starboard so that the camber lifts up.
U_FOIL_SECTIONS = [
    (3.6, 0.70, 0.00, 0.20),
    (3.6, 0.70, -0.50, 0.20),
    (3.6, 0.55, -0.70, 0.20),
    (3.6, 0.00, -0.72, 0.20),
    (3.6, -0.55, -0.70, 0.20),
    (3.6, -0.70, -0.50, 0.20),
    (3.6, -0.70, 0.00, 0.20),
]
REAR_FOIL_SECTIONS = [      # TODO: horizontal foil at the bottom of the outboard leg
    (-0.35, 0.40, -0.80, 0.15),
    (-0.35, 0.00, -0.80, 0.15),
    (-0.35, -0.40, -0.80, 0.15),
]

front_incidence = FRONT_FOIL_INCIDENCE - RIDE_GAIN * dz
u_foil = Fin(name="u_foil", xsecs=wing_sections(U_FOIL_SECTIONS, Airfoil("naca0012")))
u_foil = u_foil.rotate_local(
    angle_deg=-front_incidence,     # a negative rotation about +y raises the leading edge
    axis=[0., 1., 0.],
    origin=U_FOIL_SECTIONS[3][:3],
)

rear_foil = Fin(name="rear_foil", xsecs=wing_sections(REAR_FOIL_SECTIONS, Airfoil("naca0012")))
rear_foil = rear_foil.rotate_local(angle_deg=-REAR_FOIL_INCIDENCE, axis=[0., 1., 0.], origin=REAR_FOIL_SECTIONS[1][:3])

foils = Appendage(name="foils", fins=[u_foil, rear_foil])


#%% 5. MASSES

masses = [
    MassProperties(mass=1100., x_cg=2.3, y_cg=0., z_cg=0.5),     # TODO: boat, engine, foils [kg], CoG [m]
    MassProperties(mass=300., x_cg=2.8, y_cg=0., z_cg=0.9),      # TODO: crew and fuel
]


#%% 6. THE BOAT

boat = Sailboat(
    name="my_foiling_rib",
    hulls=[hull],
    appendages=[foils],
    masses=masses,
)


#%% 7. FORCES AND EQUILIBRIUM

boat.compute_weight(op_point)
boat.compute_buoyancy(op_point)
boat.compute_resistance(
    op_point,
    method="savitsky",          # the hull while it still touches the water; zero once it is dry
    trim=-trim,                 # Savitsky's keel trim, bow up [deg] (keel along x assumed)
    deadrise=DEADRISE,
)
boat.compute_hydrodynamics(op_point)    # the foils

Ftot, Mtot = boat.compute_torsor(op_point)

F_thrust = np.reshape(np.array([thrust, 0., 0.]), (1, 3))
thrust_point = op_point.apply_transformations(np.reshape(np.array(THRUST_POINT), (1, 3)))
Ftot = Ftot + F_thrust
Mtot = Mtot + np.cross(thrust_point, F_thrust)

sol = boat.find_equilibrium(
    opti,
    Ftot,
    Mtot,
    dofs=["Fx", "Fz", "My"],
    force_scale=1e3,
    moment_scale=1e3,
)


#%% 8. RESULTS AND DRAWING

weight = sum(m.mass for m in masses) * 9.81
hull_share = sol(boat.forces["Fb"][0, 2] + boat.forces["Fh"][0, 2]) / weight
print(f"Speed            : {SPEED:.1f} kt")
print(f"Rise             : {sol(dz):+.3f} m")
print(f"Trim             : {sol(trim):+.2f} deg (positive bow down)")
print(f"Front foil angle : {sol(front_incidence):+.2f} deg")
print(f"Thrust           : {sol(thrust):.0f} N, effective power {sol(thrust) * SPEED * 0.5144 / 1e3:.1f} kW")
print(f"Weight on hull   : {100 * hull_share:.0f} % (0 % = flying)")

# To follow the take-off, make the speed an opti.parameter and solve at
# increasing speeds, each solve starting from the previous one: see
# examples/02 - Geometry/49er_foiling.py.

boat_solved = sol(boat)
boat_solved.draw_vlm(draw_forces=True)
