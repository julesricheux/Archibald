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
                    airfoil=ThinAirfoil(xc=0.4, mc=0.05),
                    twist=-i*2,
                )
            for i, (xyz, c) in enumerate(zip(main_le, main_chords))]
        ),
        Sail(
            name="jibsail",
            xsecs=[
                WingXSec(
                    xyz_le=xyz,
                    chord=c,
                    airfoil=ThinAirfoil(xc=0.4, mc=0.1),
                    twist=-i*4,
                )
            for i, (xyz, c) in enumerate(zip(jib_le, jib_chords))]
        ),
    ]
)

# SETTINGS

rig["mainsail"] = rig["mainsail"].rotate_local(
    angle_deg=-1.,
    axis=jib_le[-1] - main_le[0],
    origin=main_le[0]
)

rig["jibsail"] = rig["jibsail"].rotate_local(
    angle_deg=-10.,
    axis=jib_le[-1] - jib_le[0],
    origin=jib_le[0]
)

rig.draw()

#%% APPENDAGE

dag_le, dag_chords = dxf_to_le_chords(r'data/49er_data/dagger.dxf', 10)
rud_le, rud_chords = dxf_to_le_chords(r'data/49er_data/rudder.dxf', 7)

app = Appendage(
    name="49er_appendages",
    fins=[
        Fin(
            name="dagger",
            xsecs=[
                WingXSec(
                    xyz_le=xyz,
                    chord=c,
                    airfoil=Airfoil("naca0012"),
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
                    airfoil=Airfoil("naca0012"),
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

sailboat.draw(
    backend="pyvista",
    draw_plane=True,
    point=np.array([0, 0, 0]),
    set_axis_visibility=True,
)

#%%
from archibald.dynamics.aero_3D.vortex_lattice_method import AeroVortexLatticeMethod
from archibald.performance import OperatingPoint

op_point = OperatingPoint(
    stw=1e-3,
    tws=10., 
    twa = 90.,
    dz=0,
    heel=0.,
    trim=0.,
    leeway=0.,
)

aeroVLM = AeroVortexLatticeMethod(rig, op_point, chordwise_resolution=10, spanwise_resolution=1)

res = aeroVLM.run()

aeroVLM.draw()