# -*- coding: utf-8 -*-
"""
TEMPLATE - Displacement motor ship (cargo vessel, trawler, ferry...)

What it computes: at a given speed, the propeller rpm, the draft and the trim
for which the ship is in equilibrium:
    - the propeller thrust balances the resistance  (Fx = 0),
    - the buoyancy carries the weight                (Fz = 0),
    - the trim settles where the pitching moment is zero (My = 0).

How to use it: replace every line marked TODO with your own data, then run the
file from this folder. Everything else can stay as it is to start with.

A complete working version of this problem, with a real hull, is
``examples/03 - Geometry/molenez_propulsion.py``.
"""
import archibald.numpy as np

from archibald.geometry import Hull, Sailboat
from archibald.geometry.propeller import BSeriesPropeller
from archibald.optimization import Opti
from archibald.performance import OperatingPoint
from archibald.weights import MassProperties


#%% 1. SETTINGS

SPEED = 10.                                 # TODO: ship speed [kt]
HULL_FILE = "data/my_ship_hull.stl"         # TODO: closed hull mesh, in metres, boat axes

# Resistance method: "holtrop" suits most displacement ships (Holtrop & Mennen 1982,
# 1984). Check its range of validity in archibald/dynamics/hydro/holtrop.py.
RESISTANCE_METHOD = "holtrop"
HOLTROP_OPTIONS = {
    "Csternchoice": 2,      # TODO: stern shape coefficient of the method (see holtrop.py)
    "Bulbchoice": 0,        # TODO: 0 without bulbous bow, 1 with
}


#%% 2. HULL

# Boat axes: x forward, y to port, z up, in metres. With the keel at z = 0, the
# draft T is found as dz = -T.
hull = Hull(name="hull", mesh=HULL_FILE)


#%% 3. APPENDAGES

# None here: a rudder and skeg change the resistance little, and the lateral
# balance is not solved. See monohull_sailboat.py to add appendages.


#%% 4. PROPULSION

# Wageningen B-series propeller. TODO: your propeller.
propeller = BSeriesPropeller(
    name="propeller",
    blades_number=4,
    blade_area_ratio=0.55,
    pitch_ratio=1.0,
    diameter=3.0,                   # [m]
    xyz_center=[2.0, 0., 1.8],      # propeller centre, boat axes [m]
)

# Hull-propeller interaction, usually from model tests or from the Holtrop method
WAKE_FRACTION = 0.25                # TODO
THRUST_DEDUCTION = 0.18             # TODO


#%% 5. MASSES

# One line per mass: lightship, cargo, fuel, ballast... Centres of gravity in boat axes.
masses = [
    MassProperties(mass=2000e3, x_cg=45., y_cg=0., z_cg=6.),    # TODO: lightship [kg], [m]
    MassProperties(mass=3000e3, x_cg=50., y_cg=0., z_cg=5.),    # TODO: cargo
]


#%% 6. THE BOAT

ship = Sailboat(            # "Sailboat" is the name of the boat object, for every boat type
    name="my_ship",
    hulls=[hull],
    propellers=[propeller],
    masses=masses,
)


#%% 7. UNKNOWNS

opti = Opti()

draft = opti.variable(init_guess=5., lower_bound=0.5)                  # TODO: first guess of the draft [m]
trim = opti.variable(init_guess=0., lower_bound=-5., upper_bound=5.)   # positive bow down [deg]
propeller.rpm = opti.variable(init_guess=100., lower_bound=10.)        # [rpm]

op_point = OperatingPoint(
    stw=SPEED,
    dz=-draft,
    trim=trim,
)


#%% 8. FORCES AND EQUILIBRIUM

ship.compute_weight(op_point)
ship.compute_buoyancy(op_point)
ship.compute_resistance(op_point, method=RESISTANCE_METHOD, **HOLTROP_OPTIONS)
ship.compute_propellers(
    op_point,
    wake_fraction=WAKE_FRACTION,
    thrust_deduction=THRUST_DEDUCTION,
)

Ftot, Mtot = ship.compute_torsor(op_point)     # sum of everything computed above

# Optional: limit the engine power, or anything else, with a constraint
# opti.subject_to(propeller.performance["BHP"] <= 3000e3)  # [W]

sol = ship.find_equilibrium(
    opti,
    Ftot,
    Mtot,
    dofs=["Fx", "Fz", "My"],
    force_scale=1e5,        # typical size of the forces [N], helps the solver
    moment_scale=1e6,       # typical size of the moments [N.m]
)


#%% 9. RESULTS AND DRAWING

performance = sol(propeller.performance)
print(f"Speed       : {SPEED:.1f} kt")
print(f"Draft       : {sol(draft):.3f} m")
print(f"Trim        : {sol(trim):+.2f} deg (positive bow down)")
print(f"Propeller   : {sol(propeller.rpm):.0f} rpm, thrust {performance['T'] / 1e3:.1f} kN")
print(f"Brake power : {performance['BHP'] / 1e3:.0f} kW")
print(f"Resistance  : {-sol(ship.forces['Fh'][0, 0]) / 1e3:.1f} kN")

ship_solved = sol(ship)
ship_solved.hulls[0].mesh.transform(sol(op_point))   # move the hull to its solved position
ship_solved.draw()
