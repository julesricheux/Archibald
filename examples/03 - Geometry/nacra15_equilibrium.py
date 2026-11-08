# -*- coding: utf-8 -*-
"""
Sailing equilibrium of the Nacra 15 in the horizontal plane, 2 dofs.

Boat speed and leeway are solved so that the drive of the rig balances the hull
resistance (Fx) and the side force of the rig is carried by the daggerboard,
the rudders and the hulls (Fy). The geometry and the trims are those of
``nacra15_resistance.py``.

What is held fixed:
- heel at -8 deg (to port, leeward at TWA 55 with the wind from starboard). It is
  meant to come from the crew position later.
- the immersion dz, solved first so that the hulls float the masses at that heel
  (Fz = 0, section DRAFT). The weight does not enter Fx and Fy, as it is vertical
  in the motion frame. At -8 deg the windward hull is barely wet: a hull out of
  the water has zero hydrodynamic force (``hull.DRY_VOLUME``).
- the rudder angle, a parameter here (``RUDDER_ANGLE``).
- yaw: Mz is not balanced. The rig force acts ahead of the daggerboard and the
  rudders sit at x about 0, so they have no lever arm in yaw (see CLAUDE.md).

The daggerboard section is ``Airfoil("e854").scale(scale_y=-1)``, with the
sections listed bottom to top as in the DXF: its camber lift goes to -y, the
windward side here.
"""
import os

import archibald.numpy as np

from archibald.dynamics.hydro import holtrop, dsyhs, kijima, jones
from archibald.geometry import Hull, Sailboat, Rig, Sail, WingXSec, ThinAirfoil, Fin, Appendage, Airfoil
from archibald.optimization import Opti
from archibald.performance import OperatingPoint
from archibald.toolbox.geom_utils import dxf_to_le_chords
from archibald.weights import MassProperties


#%% SETTINGS

TWA = 55.            # true wind angle [deg]
TWS = 12.            # true wind speed at the reference height [kt]
HEEL = -4.           # fixed heel, positive to starboard [deg]
MAINSAIL_TRIM = -10. # [deg]
JIB_TRIM = -20.      # [deg]
# RUDDER_ANGLE = 0.    # both rudders, about their leading edge, bottom to top; positive pushes to port [deg]


#%% HULLS

hulls = [
    Hull(name=name, mesh=f"data/nacra15_data/{name}.stl")
    for name in ("hull_sbd", "hull_prt")
]


#%% RIG

sails = []
for name in ("mainsail", "jib"):
    le, chords = dxf_to_le_chords(f"data/nacra15_data/{name}.dxf", 10)
    sails.append(
        Sail(
            name=name,
            xsecs=[
                WingXSec(
                    xyz_le=xyz,
                    chord=c,
                    airfoil=ThinAirfoil(xc=0.4, mc=0.12),
                    twist=-i / 10 * 10,
                )
                for i, (xyz, c) in enumerate(zip(le, chords))
            ],
        )
    )

rig = Rig(name="nacra_rig", sails=sails)

main_le, _ = dxf_to_le_chords("data/nacra15_data/mainsail.dxf", 10)
jib_le, _ = dxf_to_le_chords("data/nacra15_data/jib.dxf", 7)

rig["mainsail"] = rig["mainsail"].rotate_local(
    angle_deg=MAINSAIL_TRIM,
    axis=jib_le[-1] - main_le[0],
    origin=main_le[0],
)
rig["jib"] = rig["jib"].rotate_local(
    angle_deg=JIB_TRIM,
    axis=jib_le[-1] - jib_le[0],
    origin=jib_le[0],
)


#%% APPENDAGES

fins = []
for name, af in zip(("fin_prt", "rudder_sbd", "rudder_prt"), ("e854", "e836", "e836")):
    le, chords = dxf_to_le_chords(f"data/nacra15_data/{name}.dxf", 10)
    fin = Fin(
        name=name,
        xsecs=[
            WingXSec(
                xyz_le=xyz,
                chord=c,
                airfoil=Airfoil(af).scale(scale_y=-1.0),
                twist=0.,
            )
            for xyz, c in zip(le, chords)
        ],
    )

    fins.append(fin)

appendage = Appendage(name="nacra_appendage", fins=fins)

for wing in appendage.wings:
    if wing.name.startswith("fin"):
        fin_le, _ = dxf_to_le_chords(f"data/nacra15_data/{wing.name}.dxf", 10)
        appendage[wing.name] = appendage[wing.name].rotate_local(
            angle_deg=-5,
            axis=fin_le[-1] - fin_le[0],
            origin=fin_le[0],
        )

#%% MASSES

# PLACEHOLDERS, NOT SOURCED: to be replaced by the class rules or weighings.
# Boat axes: x from the transom (hulls 0 to 4.57 m), y to port (hulls at +-1.01 m), z up.
# The crew sits and trapezes to windward, at -y for this tack.
masses = [
    MassProperties(mass=130., x_cg=2.0, y_cg=0., z_cg=0.3),    # platform and rig
    MassProperties(mass=60., x_cg=1.6, y_cg=-1.8, z_cg=0.5),   # helm
    MassProperties(mass=60., x_cg=2.4, y_cg=-1.8, z_cg=0.8),   # crew
]
mass_names = ["platform", "helm", "crew"]


#%% SAILBOAT

nacra = Sailboat(
    name="nacra",
    hulls=hulls,
    rigs=[rig],
    appendages=[appendage],
    masses=masses,
)

resistance_method = {
    "Rf": dsyhs.compute_Rf_dsyhs,
    "Rw": dsyhs.compute_Rw_dsyhs,
    "Rtr": holtrop.compute_Rtr_holtrop,
    # "L": kijima.compute_L_kijima,
    # "Ri": kijima.compute_Ri_kijima,
    "L": jones.compute_L_jones,
    "Ri": jones.compute_Ri_jones,
}


#%% DRAFT

# # Immersion from the weight: the hulls float the masses at the fixed heel (Fz = 0).
# # Positive dz lifts the boat. The hull meshes have their keel at z = -0.128 m.
# opti_dz = Opti()
# dz_var = opti_dz.variable(init_guess=0.)
# op_point_dz = OperatingPoint(heel=HEEL, dz=dz_var)
# nacra.compute_weight(op_point_dz)
# nacra.compute_buoyancy(op_point_dz)
# F_dz, M_dz = nacra.compute_torsor(op_point_dz, components=["w", "b"])
# sol_dz = nacra.find_equilibrium(opti_dz, F_dz, M_dz, dofs=["Fz"], force_scale=1e3, verbose=False)
# dz0 = float(sol_dz(dz_var))
# print(f"Immersion  : dz {dz0:+.4f} m, for {sum(m.mass for m in masses):.0f} kg at heel {HEEL:+.1f} deg")


#%% EQUILIBRIUM

opti = Opti()

stw = opti.variable(init_guess=10., lower_bound=1.)  # boat speed [kt]
# stw = 9.
leeway = opti.variable(init_guess=-1.)             # positive to starboard [deg]
# dz = opti.variable(init_guess=0.)             # positive to starboard [deg]
dz = opti.parameter(0.1)             # positive to starboard [deg]

for name in ("rudder_sbd", "rudder_prt"):
    rud_le, _ = dxf_to_le_chords(f"data/nacra15_data/{name}.dxf", 10)
    appendage[name] = appendage[name].rotate_local(
        # angle_deg=10,
        angle_deg=-leeway,
        axis=rud_le[-1] - rud_le[0],
        origin=rud_le[0],
    )

op_point = OperatingPoint(
    stw=stw,
    twa=TWA,
    tws0=TWS,
    heel=HEEL,
    leeway=leeway,
    dz=dz,
    
)

for hull in hulls:
    hull.compute_hydrostatics_properties(op_point)

nacra.compute_weight(op_point)
nacra.compute_buoyancy(op_point)
nacra.compute_resistance(
    op_point,
    method=resistance_method,
    interaction="michell_jamaluddin",
    **{"Csternchoice": 1, "Bulbchoice": 0},
)
nacra.compute_aerodynamics(op_point)
nacra.compute_hydrodynamics(op_point)

# Weight and buoyancy are vertical and balanced by the choice of dz,
# so only the horizontal forces are summed.
Ftot, Mtot = nacra.compute_torsor(op_point, components=["h", "sail", "app"])

sol = nacra.find_equilibrium(
    opti,
    Ftot,
    Mtot,
    dofs=["Fx", "Fy"],
    # dofs=["Fy", "Fz"],
    # dofs=["Fy"],
    force_scale=100.,
    verbose=True,
    max_iter=50,
)


#%% RESULTS

print(f"\nTWA {TWA:.0f} deg, TWS {TWS:.0f} kt, heel {HEEL:+.1f} deg")
print(f"Boat speed : {sol(stw):.2f} kt")
print(f"Leeway     : {sol(leeway):+.2f} deg (positive to starboard)")

print("\nResiduals")
print(f"Fx : {sol(Ftot[0, 0]):+.2e} N")
print(f"Fy : {sol(Ftot[0, 1]):+.2e} N")
print(f"FZ : {sol(Ftot[0, 2]/9.8066):+.2e} kg")
print(f"Mz : {sol(Mtot[0, 2]):+.1f} N.m (not balanced)")

print("\nTorsor at equilibrium, motion frame")
print(f"{'':>10} {'Fx [N]':>9} {'Fy [N]':>9} {'Fz [N]':>9} {'Mx [N.m]':>10} {'Mz [N.m]':>10}")
for key, name in (("h", "hulls"), ("sail", "rig"), ("app", "appendages"), ("w", "weight"), ("b", "buoyancy")):
    F = sol(nacra.forces[f"F{key}"]).ravel()
    M = sol(nacra.moments[f"M{key}"]).ravel()
    print(f"{name:>10} {F[0]:9.1f} {F[1]:9.1f} {F[2]:9.1f} {M[0]:10.1f} {M[2]:10.1f}")

print("\nMasses (placeholders)")
for i, name in enumerate(mass_names):
    M_i = sol(nacra.moments[f"Mw_{i}"]).ravel()
    print(f"{name:>10} {masses[i].mass:6.0f} kg, Mx {M_i[0]:+8.1f} N.m")


#%% DRAWING

nacra_sol = sol(nacra)
nacra_sol.draw_vlm(cmap="fire", draw_forces=True)
