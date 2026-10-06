# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 14:37:22 2026

@author: jrich
"""

import archibald.numpy as np
from archibald.geometry.airfoil import ThinAirfoil
from archibald.geometry import Hull, Sailboat, Rig, Sail, Airfoil, WingXSec, Appendage, Fin
from archibald.geometry.mesh import ArchibaldMesh
from archibald.toolbox.mesh_utils import load_stl
from archibald.toolbox.geom_utils import dxf_to_le_chords


#%% HULL

hullStl = r'data/49er_data/hull.stl'

hull = Hull(
    mesh=hullStl,
)

# hull.draw(
#     backend="pyvista",
#     draw_plane=True,
#     point=np.array([0, 0, 0]),
# )

#%% MESH

wingsStl = r'data/49er_data/wings.stl'

wings = ArchibaldMesh(*load_stl(wingsStl))

#%% RIG

main_le, main_chords = dxf_to_le_chords(r'data/49er_data/gv.dxf', 10)
jib_le, jib_chords = dxf_to_le_chords(r'data/49er_data/jibsail.dxf', 7)

# main_le[:, 0] *= -1.
# jib_le[:, 0] *= -1.

rig = Rig(
    name="49er_rig",
    sails=[
        Sail(
            name="mainsail",
            xsecs=[
                WingXSec(
                    xyz_le=xyz,
                    chord=c,
                    airfoil=ThinAirfoil(xc=0.45, mc=0.12),
                    twist=-i/10 * 10,
                )
            for i, (xyz, c) in enumerate(zip(main_le, main_chords))]
        ),
        Sail(
            name="jibsail",
            xsecs=[
                WingXSec(
                    xyz_le=xyz,                    
                    chord=c,
                    airfoil=ThinAirfoil(xc=0.45, mc=0.18),
                    twist=-i/7*60,
                )
            for i, (xyz, c) in enumerate(zip(jib_le, jib_chords))]
        ),
    ]
)

area0 = np.sum([wing.area() for wing in rig.wings]) + 1e-2
# area0 = 21.2 # sqm

# SETTINGS

# rig["mainsail"] = rig["mainsail"].rotate_local(
#     angle_deg=-35.,
#     axis=jib_le[-1] - main_le[0],
#     origin=main_le[0],
# )

# rig["jibsail"] = rig["jibsail"].rotate_local(
#     angle_deg=-15,
#     axis=jib_le[-1] - jib_le[0],
#     origin=jib_le[0],
# )

# rig.draw()

#%% APPENDAGE

dag_le, dag_chords = dxf_to_le_chords(r'data/49er_data/dagger.dxf', 10)
rud_le, rud_chords = dxf_to_le_chords(r'data/49er_data/rudder.dxf', 10)

app = Appendage(
    name="49er_appendages",
    fins=[
        Fin(
            name="dagger",
            xsecs=[
                WingXSec(
                    xyz_le=xyz,
                    chord=c,
                    airfoil=Airfoil("e836"),
                    twist=0.,
                )
            for xyz, c in zip(dag_le, dag_chords)]
        ),
        Fin(
            name="rudder",
            xsecs=[
                WingXSec(
                    xyz_le=xyz,
                    chord=c,
                    airfoil=Airfoil("e836"),
                    twist=0.,
                )
            for xyz, c in zip(rud_le, rud_chords)]
        ),
    ]
)

#%% SAILBOAT

# rig.draw(thin_wings=True)
# rig.draw_three_view()
# hull.draw_three_view()

sailboat = Sailboat(
    displacement=200.5,
    cog=[2., 0., 1.],
    hulls=[hull],
    rigs=[rig],
    appendages=[app],
    fittings=[wings],
)

# sailboat.draw(
#     backend="pyvista",
#     draw_plane=True,
#     point=np.array([0, 0, 0]),
#     set_axis_visibility=True,
# )

#%%
from archibald.dynamics.aero_3D.vortex_lattice_method import AeroVortexLatticeMethod, HydroVortexLatticeMethod
from archibald.performance import OperatingPoint
from archibald.optimization import Opti

opti = Opti()

# t = opti.variable(init_guess=0., upper_bound=90, lower_bound=0)

twa = 50.

op_point = OperatingPoint(
    stw=12.,
    tws0=20., 
    # tws0=opti.variable(init_guess=15.), 
    # twa = opti.variable(init_guess=40.),
    twa=twa,
    # z0=10.,
    a=0.035,
    dz=0,
    heel=0.,
    trim=0.,
    # leeway=opti.variable(init_guess=-1.),
    
    # mainsail_trim=opti.variable(init_guess=-twa/3., upper_bound=0., lower_bound=-twa),
    # jib_trim=opti.variable(init_guess=-twa/3., upper_bound=0., lower_bound=-twa),
    # mainsail_twist = opti.variable(init_guess=np.zeros(8), upper_bound=0, lower_bound=-twa),
    # jibsail_twist = opti.variable(init_guess=np.zeros(6), upper_bound=0, lower_bound=-twa),
    
    mainsail_trim=-10.107250619055744,
    jib_trim=-10.4517189920483182,
    mainsail_twist = [ 1.00e-08, -3.21e+00, -7.28e+00, -1.17e+01, -1.63e+01, -2.04e+01, -2.66e+01, -7.76e+01],
    jibsail_twist = [ -6.9,  -16.07, -28.81, -40.48, -46.39, -77.82],
    
)

# for i, xsec in enumerate(rig["mainsail"].xsecs[2:]):
#     xsec.twist = op_point.mainsail_twist[i]

# for i, xsec in enumerate(rig["jibsail"].xsecs[1:]):
#     xsec.twist = op_point.jibsail_twist[i]

rig["mainsail"] = rig["mainsail"].rotate_local(
    angle_deg=op_point.mainsail_trim,
    axis=jib_le[-1] - main_le[0],
    origin=main_le[0],
)

rig["jibsail"] = rig["jibsail"].rotate_local(
    angle_deg=op_point.jib_trim,
    axis=jib_le[-1] - jib_le[0],
    origin=jib_le[0],
)

for w in app.wings:
    app[w.name] = w.rotate_local(
        angle_deg=op_point.leeway,
        axis=[0,0,1],
        origin=[0,0,0],
    )

aeroVLM = AeroVortexLatticeMethod(rig, op_point, chordwise_resolution=10, spanwise_resolution=1)

hydroVLM = HydroVortexLatticeMethod(app, op_point, chordwise_resolution=10, spanwise_resolution=1)

resA = aeroVLM.run(alpha_stall=15)
resH = hydroVLM.run(alpha_stall=15)

opti.maximize((resA["F_ab"][0] + resH["F_ab"][0]))
# opti.maximize(res["CL"])
# opti.maximize(res["F_ab"][0] / res["F_ab"][1])
# opti.maximize(res["F_w"][1] / res["F_w"][0])

area = 0
for wing in rig.wings:
    area += wing.area()

opti.subject_to([
    # (area <= area0),
    # (resA["F_ab"][1] + resH["F_ab"][1]) == 0.,
])

# {'F_ab': np.array([-370.36, -674.33,   27.  ])}


sol = opti.solve()

print(sol((resA["F_ab"][0] + resH["F_ab"][0])))

print(sol(op_point.mainsail_trim))
print(sol(op_point.jib_trim))
# print(sol(op_point.mainsail_twist))
# print(sol(op_point.jibsail_twist))

print(sol(resA["F_ab"][0]))

print("CL", sol(resA["CL"]))
print("CD", sol(resA["CD"]))

# print(sol(aeroVLM.alpha))
# print(aeroVLM.alphaeff)
# print(sol(aeroVLM.soft_stall_fac))

# import matplotlib.pyplot as plt
# # plt.plot(aeroVLM.insight)
# plt.plot(aeroVLM.soft_stall_fac)
# # plt.plot(aeroVLM.stall_fac)
# plt.ylim((-1e-2, 1+1e-2))

# # print(res)
# # aeroVLM.draw_flow()
aeroVLM.draw()

sailboat.draw_flow()


