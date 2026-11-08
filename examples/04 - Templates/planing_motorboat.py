# -*- coding: utf-8 -*-
"""
TEMPLATE - Planing motorboat (RIB, runabout, patrol boat...)

What it computes: at a given speed, the thrust, the draft and the running trim
for which the boat is in equilibrium:
    - the thrust balances the resistance                         (Fx = 0),
    - buoyancy and dynamic lift carry the weight                 (Fz = 0),
    - the trim settles where the pitching moment is zero         (My = 0).

The hull forces come from Savitsky (1964) and Savitsky & Brown (1976): dynamic
lift, pressure and spray drag, friction. Their range of validity (prismatic
hull, trim, deadrise, speed coefficient) is given in
archibald/dynamics/hydro/savitsky.py. Below planing speeds, use a displacement
method instead (see motor_ship.py).

The engine is a simple thrust at the propeller. To model the propeller itself,
see motor_ship.py.

How to use it: replace every line marked TODO with your own data, then run the
file from this folder.
"""
import archibald.numpy as np

from archibald.geometry import Hull, Sailboat
from archibald.optimization import Opti
from archibald.performance import OperatingPoint
from archibald.weights import MassProperties


#%% 1. SETTINGS

SPEED = 25.                                 # TODO: boat speed [kt]
HULL_FILE = "data/my_rib_hull.stl"          # TODO: closed hull mesh (with the tubes if they touch the water), metres
DEADRISE = 22.                              # TODO: deadrise at the transom [deg]
# or measured on the mesh: hull.mesh.deadrise_angles(locations=(0.66,))["deadrise"]

THRUST_POINT = [-0.3, 0., -0.4]             # TODO: where the thrust acts (propeller), boat axes [m]
KEEL_ANGLE = 0.                             # TODO: keel angle to the x axis of the mesh, bow up [deg]


#%% 2. HULL

hull = Hull(name="hull", mesh=HULL_FILE)


#%% 3. APPENDAGES

# None: on a RIB the outboard leg and the fins are small. See foiling_motorboat.py to add foils.


#%% 4. PROPULSION

# A thrust force, found by the equilibrium. Its power is printed at the end.


#%% 5. MASSES

masses = [
    MassProperties(mass=900., x_cg=2.2, y_cg=0., z_cg=0.5),      # TODO: boat and engine [kg], CoG [m]
    MassProperties(mass=300., x_cg=2.8, y_cg=0., z_cg=0.9),      # TODO: crew and fuel
]


#%% 6. THE BOAT

boat = Sailboat(
    name="my_rib",
    hulls=[hull],
    masses=masses,
)


#%% 7. UNKNOWNS

opti = Opti()

draft = opti.variable(init_guess=0.25, lower_bound=-0.5)                # TODO: first guess [m]
trim = opti.variable(init_guess=-3., lower_bound=-10., upper_bound=2.)  # positive bow down: planing trim is negative [deg]
thrust = opti.variable(init_guess=3000., lower_bound=0.)                # [N]

op_point = OperatingPoint(
    stw=SPEED,
    dz=-draft,
    trim=trim,
)


#%% 8. FORCES AND EQUILIBRIUM

boat.compute_weight(op_point)
boat.compute_buoyancy(op_point)
boat.compute_resistance(
    op_point,
    method="savitsky",
    trim=KEEL_ANGLE - trim,     # Savitsky's trim is the keel trim, bow up [deg]
    deadrise=DEADRISE,
)

Ftot, Mtot = boat.compute_torsor(op_point)

# The thrust, added by hand: along x, applied at the propeller
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


#%% 9. RESULTS AND DRAWING

power = sol(thrust) * SPEED * 0.5144    # effective power, thrust x speed [W]
print(f"Speed           : {SPEED:.1f} kt")
print(f"Draft           : {sol(draft):.3f} m")
print(f"Running trim    : {-sol(trim):+.2f} deg (bow up)")
print(f"Thrust          : {sol(thrust):.0f} N")
print(f"Effective power : {power / 1e3:.1f} kW (divide by the propulsive efficiency for the engine power)")

boat_solved = sol(boat)
boat_solved.hulls[0].mesh.transform(sol(op_point))
boat_solved.draw()
