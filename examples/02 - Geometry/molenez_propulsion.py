# -*- coding: utf-8 -*-
"""
Equilibrium of the Molenez hull under power, at a given speed.

Everything is explicit, so that any part can be changed in place:
    - What varies: any number made an `opti.variable`, here on the operating
      point, on a mass and on the propeller.
    - What is computed: each `boat.compute_*` called, with its own settings.
    - What is summed: the `components` and `exclude` of `boat.compute_torsor`.
    - What is imposed: the `dofs` of `boat.find_equilibrium`, plus anything given
      to `opti.subject_to` and `opti.minimize` before it.
"""
from archibald.dynamics.hydro import dsyhs
from archibald.geometry.hull import Hull
from archibald.geometry.propeller import BSeriesPropeller
from archibald.geometry.sailboat import Sailboat
from archibald.optimization import Opti
from archibald.performance import OperatingPoint
from archibald.weights import MassProperties


#%% DATA

propeller = BSeriesPropeller(
    blades_number=4,
    blade_area_ratio=0.65,
    pitch_ratio=1.0,
    diameter=2.5,
    xyz_center=[1.0, 0.0, 0.5],
)

boat = Sailboat(
    hulls=[Hull(name='hull', mesh=r'data//molenez2_data//hull.stl')],
    propellers=[propeller],
    masses=[
        MassProperties(mass=100e3, x_cg=12.3, y_cg=0., z_cg=2.2),  # lightship
        MassProperties(mass=14e3, x_cg=12.0, y_cg=0., z_cg=0.5),   # ballast
        MassProperties(mass=10e3, x_cg=15.0, y_cg=3.0, z_cg=3.0),  # deck cargo, moved below
    ],
)


#%% WHAT VARIES

opti = Opti()

# Operating point
T = opti.variable(init_guess=1.3, lower_bound=0.2, upper_bound=4.)      # immersion [m]
heel = opti.variable(init_guess=0., lower_bound=-30., upper_bound=30.)  # positive to starboard [deg]
trim = opti.variable(init_guess=0., lower_bound=-10., upper_bound=10.)  # positive bow down [deg]

op_point = OperatingPoint(
    stw=8.,
    dz=-T,
    heel=heel,
    trim=trim,
)

# A mass position
boat.masses[2].y_cg = opti.variable(init_guess=3., lower_bound=-6., upper_bound=6.)  # [m]

# The propeller
propeller.rpm = opti.variable(init_guess=150., lower_bound=30., upper_bound=800.)     # [rpm]
propeller.pitch_ratio = opti.variable(init_guess=1., lower_bound=0.6, upper_bound=1.4)  # [-]


#%% WHAT IS COMPUTED

boat.compute_weight(op_point)
boat.compute_buoyancy(op_point)
boat.compute_resistance(
    op_point,
    method={
        "Rf": dsyhs.compute_Rf_dsyhs,
        "Rtr": dsyhs.compute_Rtr_dsyhs,
    },
    Csternchoice=1,
    Bulbchoice=0,
)
boat.compute_propellers(
    op_point,
    wake_fraction=0.2,
    thrust_deduction=0.15,
)


#%% WHAT IS SUMMED AND IMPOSED

# To set an element aside, e.g. the cargo and the propeller: exclude=["w_2", "prop"]
Ftot, Mtot = boat.compute_torsor(op_point)

# Other constraints and objectives go directly to the Opti object
brake_power = propeller.performance["BHP"]

opti.subject_to(heel == 0.)  # the cargo is moved to keep the boat upright
opti.subject_to(brake_power <= 500e3)
opti.minimize(brake_power / 1e5)  # rpm and pitch for the least power

sol = boat.find_equilibrium(
    opti,
    Ftot,
    Mtot,
    dofs=["Fx", "Fz", "Mx", "My"],
    force_scale=1e4,
    moment_scale=1e5,
    verbose=False,
)


#%% RESULTS

print(f"Immersion   : {sol(T):.3f} m")
print(f"Heel        : {sol(heel):+.2f} deg")
print(f"Trim        : {sol(trim):+.2f} deg (positive bow down)")
print(f"Cargo y     : {sol(boat.masses[2].y_cg):+.3f} m")
print(f"Rotation    : {sol(propeller.rpm):.1f} rpm")
print(f"Pitch ratio : {sol(propeller.pitch_ratio):.3f}")

performance = sol(propeller.performance)
print(f"Thrust      : {performance['T'] / 1e3:.2f} kN")
print(f"Brake power : {performance['BHP'] / 1e3:.1f} kW")
print(f"Efficiency  : {performance['eta']:.3f}")

print(f"\nSummed : {boat.torsor_elements}")
print(f"Ftot : {sol(Ftot).ravel()} N")
print(f"Mtot : {sol(Mtot).ravel()} N.m")
