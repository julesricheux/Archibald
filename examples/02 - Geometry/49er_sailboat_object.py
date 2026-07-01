# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 14:37:22 2026

@author: jrich
"""

import archibald.numpy as np
from archibald.geometry.airfoil.thin_section import thin_airfoil
from archibald.geometry import Hull, Sailboat, Rig, Sail, Airfoil, WingXSec, Appendage, Fin
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

meshStl = r'data/49er_data/wings.stl'

#%% RIG

main_le, main_chords = dxf_to_le_chords(r'data/49er_data/gv.dxf', 10)
jib_le, jib_chords = dxf_to_le_chords(r'data/49er_data/jibsail.dxf', 7)

rig = Rig(
    wings=[
        Sail(
            name="mainsail",
            xsecs=[
                WingXSec(
                    xyz_le=xyz,
                    chord=c,
                    airfoil=thin_airfoil(xc=0.4, mc=0.1)[0],
                    twist=180.,
                )
            for xyz, c in zip(main_le, main_chords)]
        ),
        Sail(
            name="jibsail",
            xsecs=[
                WingXSec(
                    xyz_le=xyz,
                    chord=c,
                    airfoil=thin_airfoil(xc=0.4, mc=0.2)[0],
                    twist=180.,
                )
            for xyz, c in zip(jib_le, jib_chords)]
        ),
    ]
)

#%% APPENDAGE

dag_le, dag_chords = dxf_to_le_chords(r'data/49er_data/dagger.dxf', 10)
rud_le, rud_chords = dxf_to_le_chords(r'data/49er_data/rudder.dxf', 7)

app = Appendage(
    wings=[
        Fin(
            name="dagger",
            xsecs=[
                WingXSec(
                    xyz_le=xyz,
                    chord=c,
                    airfoil=Airfoil("naca0012"),
                    twist=180.,
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
                    twist=180.,
                )
            for xyz, c in zip(rud_le, rud_chords)]
        ),
    ]
)

#%% SAILBOAT

# rig.draw(thin_wings=True)
# rig.draw_three_view()
# hull.draw_three_view()

rig.wings[0] = rig.wings[0].rotate_local(
    angle_deg=-45.,
    axis=jib_le[-1] - main_le[0],
    origin=main_le[0]
)

rig.wings[1] = rig.wings[1].rotate_local(
    angle_deg=-45.,
    axis=jib_le[-1] - jib_le[0],
    origin=jib_le[0]
)

sailboat = Sailboat(
    displacement=200.5,
    cog=[2., 0., 1.],
    hulls=[hull],
    rigs=[rig],
    appendages=[app],
)

sailboat.draw(
    backend="pyvista",
    # draw_plane=True,
    # point=np.array([0, 0, 0]),
    set_axis_visibility=True,
)
