# -*- coding: utf-8 -*-
"""
Windage: Day (2017), ORC VPP and Blendermann (1994).

No published force was found for these models, so each is checked on what can be checked:

- Day, "Performance prediction for sailing dinghies", 2017. Areas and crew from the equations of the paper,
  and the Du Bois surface area with its usual units (kg, cm) as an independent formula.
- Blendermann, "Wind loading of ships: collected data from wind tunnel tests in uniform flow"
  (TUHH, Proc. IMechE M 218, 2004), Table 3: confirms CDt and kappa of the shipped ship type.
  The two limits of the loading function give back its parameters by construction. CDlAF and delta
  of ``container_ship_loaded`` are not in that table: data/blendermann_1994_parameters.csv is a blank.
- ORC VPP documentation, section 5.3: not found online, data/orc_vpp_windage.csv is a blank for its areas.
  The ORC and Day crews are compared, because the difference is large and should be looked at.
"""
import matplotlib.pyplot as plt
import numpy as onp

from archibald.dynamics.aero import blendermann, common, day, orc

from validation_common import BLUE, GREY, ORANGE, AQUA, comparison_table, finish, read_table, style


rho_air, g = 1.225, 9.81
uniform = dict(z_ref=10., wind_alpha=0.)
wind = dict(tws=10., stw=0., rho_air=rho_air, **uniform)
q = 0.5 * rho_air * (10. * 0.5144444) ** 2


#%% APPARENT WIND

aws, awa = common.apparent_wind(tws=10., twa=90., stw=10., z=10.)
aws_low, _ = common.apparent_wind(tws=10., twa=0., stw=0., z=1.)
comparison_table("Apparent wind, by hand", [
    ("aws, tws = stw, beam [m/s]", 10. * onp.sqrt(2.) * 0.5144444, aws, ""),
    ("awa, tws = stw, beam [deg]", 45., awa, ""),
    ("speed at 1 m, power 1/7 [m/s]", 10. * 0.5144444 * 0.1 ** (1. / 7.), aws_low, ""),
])


#%% DAY (2017)

mass, height = 80., 1.83
du_bois = 0.007184 * mass ** 0.425 * (height * 100.) ** 0.725  # Du Bois & Du Bois (1916), kg and cm
A_front, A_side = day.crew_areas_day(mass, height, g)
Af, As, zce = day.hull_areas_day(Fb=0.5, Boa=2., Loa=5., Cwp=0.7, heel=20.)
heeled = 0.5 + 0.5 * 2. * 0.7 * onp.sin(onp.radians(20.))
comparison_table("Day (2017)", [
    ("Du Bois area from A_front", du_bois, A_front / 0.326, "m2, formula in N and m vs kg and cm"),
    ("A_side / A_front", 0.219 / 0.326, A_side / A_front, ""),
    ("hull frontal area, Eq. 7", 2. * 0.5, Af, "m2"),
    ("hull side area, Eq. 7", 5. * heeled, As, "m2"),
    ("hull zce, Eq. 7", 0.66 * heeled, zce, "m"),
])

head = day.compute_Fcrew_day(twa=0., g=g, Fb=0.5, Boa=2., heel=0., zce_crew=10., **wind)
beam = day.compute_Fcrew_day(twa=90., g=g, Fb=0.5, Boa=2., heel=0., zce_crew=10., **wind)
comparison_table("Crew of Day, with its reductions of 10 % and 20 %", [
    ("head wind Fx", -q * 0.9 * 0.8 * 1.075 * A_front, head[0], "N"),
    ("beam wind Fy", q * 0.9 * 0.8 * 0.954 * A_side, beam[1], "N"),
])


#%% BLENDERMANN

paper = {"container_ship_loaded": dict(CDt=0.90, kappa=1.4)}
parameters = blendermann.BLENDERMANN_SHIP_TYPES["container_ship_loaded"]
comparison_table("Shipped parameters against Blendermann (2004), Table 3, laden container ship", [
    ("CDt", paper["container_ship_loaded"]["CDt"], parameters["CDt"], ""),
    ("kappa", paper["container_ship_loaded"]["kappa"], parameters["kappa"], ""),
])

Ax, Ay, Loa = 1000., 5000., 300.
cey = onp.array([[0., 0., 20.]])


def blender(twa):
    return blendermann.compute_Fhull_blendermann(
        twa=twa, Ax=Ax, Ay=Ay, Loa=Loa, cey=cey, ship_type="container_ship_loaded", **wind,
    )


comparison_table("Blendermann, limits of the loading function", [
    ("X in a head wind", -q * Ax * parameters["CDlAF_head"], blender(0.)[0], "N"),
    ("Y in a beam wind", q * Ay * parameters["CDt"], blender(90.)[1], "N"),
    ("X in a following wind", q * Ax * parameters["CDlAF_stern"], blender(180.)[0], "N"),
    ("yaw arm in a head wind", 0.18 * Loa * onp.pi / 2., blender(0.)[3], "m"),
    ("height", parameters["kappa"] * 20., blender(0.)[2], "m"),
], tolerance=1.0)

e = 45.
CDl = parameters["CDlAF_head"] * Ax / Ay
den = 1. - parameters["delta"] / 2. * (1. - CDl / parameters["CDt"]) * onp.sin(onp.radians(2. * e)) ** 2
comparison_table("Blendermann at 45 deg, by hand", [
    ("X", -q * Ax * parameters["CDlAF_head"] * onp.cos(onp.radians(e)) / den, blender(e)[0], "N"),
    ("Y", q * Ay * parameters["CDt"] * onp.sin(onp.radians(e)) / den, blender(e)[1], "N"),
])


#%% ORC AND DAY CREW

orc_table = read_table("orc_vpp_windage.csv")
if orc_table is not None:
    shipped_orc = dict(Cd_hull=0.816, Cd_crew=1.08, A_front_crew=0.25, A_side_per_crew=0.5, crew_zce_above_deck=0.5)
    comparison_table("ORC VPP documentation against the defaults of orc.py", [
        (name, value, shipped_orc[name], f"p. {page}")
        for name, value, page in zip(orc_table["name"], orc_table["value"], orc_table["page"])
        if name in shipped_orc
    ])

blendermann_table = read_table("blendermann_1994_parameters.csv")
if blendermann_table is not None:
    rows = []
    for i, ship_type in enumerate(blendermann_table["ship_type"]):
        if ship_type in blendermann.BLENDERMANN_SHIP_TYPES:
            for name in ("CDt", "CDlAF_head", "CDlAF_stern", "delta", "kappa"):
                rows.append((f"{ship_type}, {name}", blendermann_table[name][i], blendermann.BLENDERMANN_SHIP_TYPES[ship_type][name], ""))
    comparison_table("Blendermann (1994) against the shipped ship types", rows)

head_orc = orc.compute_Fcrew_orc(twa=0., Fb=0.5, Boa=2., heel=0., zce_crew=10., **wind)
beam_orc = orc.compute_Fcrew_orc(twa=90., Fb=0.5, Boa=2., heel=0., zce_crew=10., **wind)
print("\nDrag area of one crew member [m2]")
print(f"  {'':<14}{'head wind':>11}{'beam wind':>11}")
print(f"  {'ORC, shipped':<14}{float(-head_orc[0] / q):>11.3f}{float(beam_orc[1] / q):>11.3f}")
print(f"  {'Day':<14}{float(-head[0] / q):>11.3f}{float(beam[1] / q):>11.3f}")


#%% PLOTS

angles = onp.linspace(0., 180., 181)
fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))

forces = onp.array([[float(blender(a)[i]) for i in (0, 1)] for a in angles])
axes[0].plot(angles, forces[:, 0] / (q * Ax), color=BLUE, linewidth=2, label="X / (q AF)")
axes[0].plot(angles, forces[:, 1] / (q * Ay), color=ORANGE, linewidth=2, label="Y / (q AL)")
axes[0].plot([0., 90., 180.], [-parameters["CDlAF_head"], parameters["CDt"], parameters["CDlAF_stern"]], "o", color="black", markersize=5, label="Parameters of the ship type")
axes[0].axhline(0., color=GREY, linewidth=1)
style(axes[0], title="Blendermann, loaded container ship", xlabel="Apparent wind angle [deg]")
axes[0].legend(frameon=False, fontsize=8)

force_orc = onp.array([[float(x) for x in orc.compute_Fcrew_orc(twa=a, Fb=0.5, Boa=2., heel=0., zce_crew=10., **wind)[:2]] for a in angles])
force_day = onp.array([[float(x) for x in day.compute_Fcrew_day(twa=a, g=g, Fb=0.5, Boa=2., heel=0., zce_crew=10., **wind)[:2]] for a in angles])
axes[1].plot(angles, onp.hypot(*force_orc.T) / q, color=BLUE, linewidth=2, label="ORC, shipped areas")
axes[1].plot(angles, onp.hypot(*force_day.T) / q, color=ORANGE, linewidth=2, label="Day (2017)")
style(axes[1], title="Drag area of one crew member [m2]", xlabel="Apparent wind angle [deg]")
axes[1].set_ylim(bottom=0.)
axes[1].legend(frameon=False, fontsize=8)

fig.tight_layout()
finish(fig, "windage")
