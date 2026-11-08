# -*- coding: utf-8 -*-
"""
Hybrid cargo sailboat: the Molenez under sails and propeller.

At a given boat speed and true wind, the equilibrium is solved on 5 dofs:
    - propeller rpm, so that the thrust tops up the drive of the sails (Fx),
    - leeway, so that the fins and the rudder carry the side force (Fy),
    - immersion, heel and trim, so that the hull floats the load case (Fz, Mx, My).
Yaw (Mz) is not balanced: the rudder angle is a setting.

The wind is light on purpose: at 8 kt of boat speed and 15 kt of true wind the
wings alone drive 12 kN against about 6 kN of drag, so the propeller would have
to brake and the problem has no solution with rpm > 0. At 8 kt of true wind the
wings give about 5 kN and the propeller tops up the rest.

Geometry from ``data/molenez2_data``: the hull mesh, two rigid wing sails
(``sail_fore``, ``sail_aft``, SD7037 sections), two fins (``fin_prt``,
``fin_sbd``) and a rudder (NACA 0015). The fins and the rudder are drawn as
planform outlines, read with ``outline_to_le_chords``.
"""
import archibald.numpy as np

from archibald.dynamics.hydro import dsyhs
from archibald.geometry import Hull, Sailboat, Rig, Sail, WingXSec, Fin, Appendage, Airfoil
from archibald.geometry.propeller import BSeriesPropeller
from archibald.optimization import Opti
from archibald.performance import OperatingPoint
from archibald.toolbox.geom_utils import dxf_to_le_chords, outline_to_le_chords
from archibald.weights import MassProperties


#%% SETTINGS

STW = 8.                # boat speed [kt]
TWA = 60.               # true wind angle [deg], wind from starboard
TWS = 8.                # true wind speed at the reference height [kt]
SAIL_TRIM = -15.        # both wings, about their leading edge; negative turns the trailing edge to port [deg]
RUDDER_ANGLE = 0.       # about its leading edge, bottom to top; positive pushes to port [deg]
LOAD_CASE = "containers"

# Load cases of the screenshot provided with the request. The columns are read
# as displacement [t], LCG [m], TCG [m] and VCG [m], in the axes of the hull
# mesh (x from the transom, y to port, z from the keel): to be confirmed.
LOAD_CASES = {
    "lightship": (61.75, 11.13, -0.07, 1.78),      # navire lege arme
    "minimal": (66.65, 11.79, -0.012, 1.74),       # conditions minimales
    "containers": (120.43, 12.10, -0.001, 2.03),   # pleine charge conteneurs
    "mixed": (149.96, 12.12, -0.005, 1.93),        # pleine charge mixte
}

# Initial guesses of the immersion at the transom [m] and trim [deg], from the
# free-floating equilibrium of each load case without sails
FLOATING = {
    "lightship": (1.17, -1.6),
    "minimal": (1.07, -0.9),
    "containers": (1.38, -0.2),
    "mixed": (1.49, 0.3),
}


#%% HULL

hull = Hull(name="hull", mesh="data/molenez2_data/hull.stl")


#%% RIG

sails = []
for name in ("sail_fore", "sail_aft"):
    le, chords = dxf_to_le_chords(f"data/molenez2_data/{name}.dxf", 10)
    sail = Sail(
        name=name,
        xsecs=[
            WingXSec(
                xyz_le=xyz,
                chord=c,
                airfoil=Airfoil("sd7037"),  # sections bottom to top: camber lift to port, leeward here
                twist=0.,
            )
            for xyz, c in zip(le, chords)
        ],
    )
    sails.append(
        sail.rotate_local(
            angle_deg=SAIL_TRIM,
            axis=le[-1] - le[0],
            origin=le[0],
        )
    )

rig = Rig(name="molenez_rig", sails=sails)


#%% APPENDAGES

fins = []
for name in ("fin_prt", "fin_sbd", "rudder"):
    le, chords = outline_to_le_chords(f"data/molenez2_data/{name}.dxf", 8)
    fin = Fin(
        name=name,
        xsecs=[
            WingXSec(
                xyz_le=xyz,
                chord=c,
                airfoil=Airfoil("naca0015"),
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

appendage = Appendage(name="molenez_appendages", fins=fins)


#%% PROPELLER

# PLACEHOLDER, NOT SOURCED: a small propeller behind the rudder (rudder trailing
# edge at x = 0.13 m), on the centreline, below the waterline.
propeller = BSeriesPropeller(
    name="propeller",
    blades_number=4,
    blade_area_ratio=0.55,
    pitch_ratio=1.0,
    diameter=1.2,
    xyz_center=[-0.6, 0., 0.3],
)


#%% MASSES

mass, lcg, tcg, vcg = LOAD_CASES[LOAD_CASE]
masses = [
    MassProperties(mass=mass * 1e3, x_cg=lcg, y_cg=tcg, z_cg=vcg),
]


#%% SAILBOAT

boat = Sailboat(
    name="molenez",
    hulls=[hull],
    rigs=[rig],
    appendages=[appendage],
    propellers=[propeller],
    masses=masses,
)

resistance_method = {
    "Rf": dsyhs.compute_Rf_dsyhs,
    "Rw": dsyhs.compute_Rw_dsyhs,
    "Rtr": dsyhs.compute_Rtr_dsyhs,
}


#%% EQUILIBRIUM

opti = Opti()

T0, trim0 = FLOATING[LOAD_CASE]
T = opti.variable(init_guess=T0, lower_bound=0.2, upper_bound=4.)          # immersion at the transom [m]
heel = opti.variable(init_guess=-5., lower_bound=-30., upper_bound=30.)    # positive to starboard [deg]
trim = opti.variable(init_guess=trim0, lower_bound=-10., upper_bound=10.)  # positive bow down [deg]
leeway = opti.variable(init_guess=-1.5)                                    # positive to starboard [deg]
propeller.rpm = opti.variable(init_guess=120., lower_bound=10.)            # [rpm]

op_point = OperatingPoint(
    stw=STW,
    twa=TWA,
    tws0=TWS,
    dz=-T,
    heel=heel,
    trim=trim,
    leeway=leeway,
)

boat.compute_weight(op_point)
boat.compute_buoyancy(op_point)
boat.compute_resistance(
    op_point,
    method=resistance_method,
    Csternchoice=1,
    Bulbchoice=0,
)
boat.compute_aerodynamics(op_point)
boat.compute_hydrodynamics(op_point)
boat.compute_propellers(
    op_point,
    wake_fraction=0.2,      # PLACEHOLDERS, NOT SOURCED
    thrust_deduction=0.15,
)

Ftot, Mtot = boat.compute_torsor(op_point)

sol = boat.find_equilibrium(
    opti,
    Ftot,
    Mtot,
    dofs=["Fx", "Fy", "Fz", "Mx", "My"],
    force_scale=1e4,
    moment_scale=1e5,
    verbose=True,
    max_iter=100,
)


#%% RESULTS

print(f"\nLoad case {LOAD_CASE}: {mass:.2f} t, CoG ({lcg}, {tcg}, {vcg}) m")
print(f"STW {STW:.1f} kt, TWA {TWA:.0f} deg, TWS {TWS:.0f} kt, sail trim {SAIL_TRIM:+.0f} deg, rudder {RUDDER_ANGLE:+.0f} deg")
print(f"Immersion  : {sol(T):.3f} m at the transom")
print(f"Heel       : {sol(heel):+.2f} deg (positive to starboard)")
print(f"Trim       : {sol(trim):+.2f} deg (positive bow down)")
print(f"Leeway     : {sol(leeway):+.2f} deg (positive to starboard)")

performance = sol(propeller.performance)
print(f"Propeller  : {sol(propeller.rpm):.0f} rpm, thrust {performance['T'] / 1e3:.2f} kN, "
      f"brake power {performance['BHP'] / 1e3:.1f} kW")

print("\nResiduals")
for i, name in enumerate(("Fx", "Fy", "Fz")):
    print(f"{name} : {sol(Ftot[0, i]):+.2e} N")
for i, name in enumerate(("Mx", "My", "Mz")):
    print(f"{name} : {sol(Mtot[0, i]):+.2e} N.m" + (" (not balanced)" if name == "Mz" else ""))

print("\nTorsor at equilibrium, motion frame")
print(f"{'':>11} {'Fx [kN]':>9} {'Fy [kN]':>9} {'Fz [kN]':>9} {'Mx [kN.m]':>10} {'Mz [kN.m]':>10}")
for key, name in (("h", "hull"), ("sail", "sails"), ("app", "appendages"), ("prop", "propeller"),
                  ("w", "weight"), ("b", "buoyancy")):
    F = sol(boat.forces[f"F{key}"]).ravel() / 1e3
    M = sol(boat.moments[f"M{key}"]).ravel() / 1e3
    print(f"{name:>11} {F[0]:9.2f} {F[1]:9.2f} {F[2]:9.1f} {M[0]:10.1f} {M[2]:10.1f}")


#%% DRAWING

boat_sol = sol(boat)
boat_sol.draw_vlm(cmap="fire", draw_forces=True, force_unit="kN")
