# -*- coding: utf-8 -*-
"""
Take-off of a foiling 49er: from displacement to flying, under a fixed thrust.

The 49er hull, daggerboard and rudder carry a horizontal foil at the tip of the
daggerboard. A thrust along +x stands for the drive of the rig, so that the
problem has only 2 dofs:
    - Fx: the thrust balances the drag of the hull and the appendages (boat speed),
    - Fz: the weight is carried by the buoyancy, the hull dynamic lift and the foil (immersion dz).
Heel, trim and leeway are held at zero, and the moments are not balanced.

Ride height control: with a fixed incidence, a deep foil gives the same lift at
any height, so once the hull is dry nothing in Fz depends on dz and the problem
is singular (the first version of this example stopped converging there). As
the wand of a foiling Moth does with a flap, the incidence of the whole foil here
decreases linearly as the boat rises, which sets the flying height.

Two sweeps, each solve starting from the previous one:
1. Fixed thrusts from 50 to 450 N, solving stw and dz (Fx, Fz). Every solve
   converges, in 3 to 15 iterations, but the speed jumps between 275 and 300 N,
   from 12.6 to 21.7 kt: the drag falls once the hull leaves the water (the
   take-off hump, 297 N at 15 kt, then 255 N at 17 kt), so for some thrusts there
   are three equilibria and the thrust sweep jumps from the displacement branch
   to the flying one.
2. Fixed speeds from 3 to 32 kt, solving dz (Fz) and reading the thrust needed.
   This follows the whole curve continuously through the lift-off, in 2 to 9
   iterations per solve.

When flying, the daggerboard and the rudder cross the free surface, where the
density of each strip blends from water to air over about 1 cm
(``FREE_SURFACE_TRANSITION``). Each strip that leaves the water is a small step
in the drag. With 10 sections the steps made the drag curve non-monotonic, and 2
of the fixed thrusts did not converge. With 25 (``N_SECTIONS``) they all do. No
free surface effect on the foil, ventilation or cavitation is modelled.

PLACEHOLDERS, NOT SOURCED: the foil (span, chord, section, incidence, position),
the ride height control, and the mass, which is the 200.5 kg of
``49er_sailboat_object.py``.
"""
import matplotlib.pyplot as plt
import numpy as onp

import archibald.numpy as np
from archibald.dynamics.hydro import holtrop, dsyhs
from archibald.geometry import Hull, Sailboat, WingXSec, Fin, Appendage, Airfoil
from archibald.optimization import Opti
from archibald.performance import OperatingPoint
from archibald.toolbox.geom_utils import dxf_to_le_chords
from archibald.weights import MassProperties


#%% SETTINGS

THRUSTS = onp.arange(50., 451., 25.)    # fixed thrusts of the first sweep [N]
SPEEDS = onp.arange(3., 32.1, 1.)       # fixed boat speeds of the second sweep [kt]

# PLACEHOLDER foil, at the tip of the daggerboard (leading edge at x 2.11 m, z -1.163 m)
FOIL_SPAN = 1.2         # [m]
FOIL_CHORD = 0.16       # [m]
FOIL_LE = [2.05, 0., -1.163]

# PLACEHOLDER ride height control: the incidence decreases linearly as the boat rises
FOIL_INCIDENCE = 4.     # nose up, to the hull baseline, at dz = 0 [deg]
WAND_GAIN = 5.          # incidence lost per metre of rise [deg/m]

N_SECTIONS = 25         # sections of the daggerboard and the rudder, which cross the free surface when flying

MASS = 200.5            # PLACEHOLDER, as in 49er_sailboat_object.py [kg]

resistance_method = {
    "Rf": dsyhs.compute_Rf_dsyhs,
    "Rw": dsyhs.compute_Rw_dsyhs,
    "Rtr": holtrop.compute_Rtr_holtrop,
}


#%% MODEL

def make_boat(incidence):
    """The 49er hull, daggerboard, rudder and foil, with the foil at an incidence [deg] that may be symbolic."""
    fins = []
    for name in ("dagger", "rudder"):
        le, chords = dxf_to_le_chords(f"data/49er_data/{name}.dxf", N_SECTIONS)
        fins.append(
            Fin(
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
        )

    # Symmetric section, so the order of the sections does not set the side of the lift
    foil = Fin(
        name="foil",
        xsecs=[
            WingXSec(
                xyz_le=[FOIL_LE[0], y, FOIL_LE[2]],
                chord=FOIL_CHORD,
                airfoil=Airfoil("naca0012"),
                twist=0.,
            )
            for y in onp.linspace(FOIL_SPAN / 2, -FOIL_SPAN / 2, 9)
        ],
    )
    fins.append(
        foil.rotate_local(
            angle_deg=-incidence,  # a negative rotation about +y raises the leading edge
            axis=[0., 1., 0.],
            origin=FOIL_LE,
        )
    )

    return Sailboat(
        name="49er",
        hulls=[Hull(name="hull", mesh="data/49er_data/hull.stl")],
        appendages=[Appendage(name="49er_appendages", fins=fins)],
        masses=[MassProperties(mass=MASS, x_cg=2., y_cg=0., z_cg=1.)],
    )


def compute_forces(boat, op_point):
    """Weight, buoyancy, hull and appendage forces; returns the total torsor."""
    boat.compute_weight(op_point)
    boat.compute_buoyancy(op_point)
    boat.compute_resistance(
        op_point,
        method=resistance_method,
        Csternchoice=1,
        Bulbchoice=0,
    )
    boat.compute_hydrodynamics(op_point)
    return boat.compute_torsor(op_point)


def hull_share(boat):
    """Fraction of the weight carried by the hull (buoyancy and dynamic lift)."""
    return (boat.forces["Fb"][0, 2] + boat.forces["Fh"][0, 2]) / (MASS * 9.81)


#%% SWEEP 1: FIXED THRUST, SOLVE SPEED AND IMMERSION

opti = Opti()

thrust = opti.parameter(THRUSTS[0])                                     # [N]
stw = opti.variable(init_guess=4., lower_bound=0.5, upper_bound=45.)    # boat speed [kt]
dz = opti.variable(init_guess=0., lower_bound=-0.3, upper_bound=1.5)    # immersion, positive lifts the boat [m]
incidence = FOIL_INCIDENCE - WAND_GAIN * dz

boat = make_boat(incidence)
Ftot, Mtot = compute_forces(boat, OperatingPoint(stw=stw, dz=dz))
Ftot = Ftot + np.reshape(np.array([thrust, 0., 0.]), (1, 3))

residuals = Sailboat.equilibrium_residuals(Ftot, Mtot, dofs=["Fx", "Fz"], force_scale=100.)
opti.subject_to([residual == 0. for residual in residuals.values()])

print("Sweep 1: fixed thrust")
print(f"{'thrust [N]':>10} {'stw [kt]':>9} {'dz [m]':>8} {'inc [deg]':>9} {'hull [%]':>9} {'iter':>5}")
fixed_thrust = []
for value in THRUSTS:
    sol = opti.solve(parameter_mapping={thrust: value}, verbose=False, max_iter=100, behavior_on_failure="return_last")
    stats = sol.stats()
    fixed_thrust.append((value, sol(stw), sol(dz), sol(hull_share(boat)), stats["success"]))
    print(f"{value:10.0f} {sol(stw):9.2f} {sol(dz):8.3f} {sol(incidence):9.2f} {100 * sol(hull_share(boat)):9.1f} "
          f"{stats['iter_count']:5d}" + ("" if stats["success"] else "  NOT CONVERGED"))
    opti.set_initial_from_sol(sol, initialize_duals=False)
fixed_thrust = onp.array(fixed_thrust, dtype=float)


#%% SWEEP 2: FIXED SPEED, SOLVE IMMERSION, READ THE THRUST NEEDED

opti = Opti()

stw = opti.parameter(SPEEDS[0])                                         # boat speed [kt]
dz = opti.variable(init_guess=0., lower_bound=-0.3, upper_bound=1.5)    # immersion, positive lifts the boat [m]
incidence = FOIL_INCIDENCE - WAND_GAIN * dz

boat = make_boat(incidence)
Ftot, Mtot = compute_forces(boat, OperatingPoint(stw=stw, dz=dz))
thrust_needed = -Ftot[0, 0]

residuals = Sailboat.equilibrium_residuals(Ftot, Mtot, dofs=["Fz"], force_scale=100.)
opti.subject_to([residual == 0. for residual in residuals.values()])

print("\nSweep 2: fixed speed")
print(f"{'stw [kt]':>9} {'thrust [N]':>10} {'dz [m]':>8} {'inc [deg]':>9} {'hull [%]':>9} {'iter':>5}")
fixed_speed = []
for value in SPEEDS:
    sol = opti.solve(parameter_mapping={stw: value}, verbose=False, max_iter=100, behavior_on_failure="return_last")
    stats = sol.stats()
    fixed_speed.append((sol(thrust_needed), value, sol(dz), sol(hull_share(boat)), stats["success"]))
    print(f"{value:9.1f} {sol(thrust_needed):10.1f} {sol(dz):8.3f} {sol(incidence):9.2f} "
          f"{100 * sol(hull_share(boat)):9.1f} {stats['iter_count']:5d}" + ("" if stats["success"] else "  NOT CONVERGED"))
    opti.set_initial_from_sol(sol, initialize_duals=False)
fixed_speed = onp.array(fixed_speed, dtype=float)


#%% DRAWING

fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
for ax, column, label in zip(axes, (0, 2, 3), ("Thrust [N]", "dz [m]", "Weight carried by the hull [-]")):
    for data, style, name in ((fixed_speed, "-", "fixed speed"), (fixed_thrust, "o", "fixed thrust")):
        ok = data[:, 4] > 0.5
        ax.plot(data[ok, 1], data[ok, column], style, markersize=4, label=name)
        if (~ok).any():
            ax.plot(data[~ok, 1], data[~ok, column], "x", color="red", label=f"{name}, not converged")
    ax.set_xlabel("Boat speed [kt]")
    ax.set_ylabel(label)
    ax.grid(True, alpha=0.3)
axes[0].legend()
fig.suptitle("49er with a foil: take-off under a fixed thrust")
fig.tight_layout()
fig.savefig("49er_foiling.png", dpi=150)
plt.show()
