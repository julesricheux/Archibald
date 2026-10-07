# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 00:07:57 2026

@author: jules
"""

import os

import archibald.numpy as np

from archibald.geometry import Hull, Sailboat
from archibald.performance import OperatingPoint

hulls = []

stl_files = [
    r'data/nacra15_data/hull_sbd.stl',
    r'data/nacra15_data/hull_prt.stl',
]

for file in stl_files:
    hulls.append(
        Hull(
            name=os.path.basename(file[:-4]),
            mesh=file,
        )
    )
    
nacra = Sailboat(
    name="nacra",
    hulls=hulls,
)

# nacra.draw()

op_point = OperatingPoint(
    stw=10.,
)

for hull in hulls:
    hull.compute_hydrostatics_properties(op_point)
    
hull.mesh.draw(
    draw_plane=True,
    point=np.array([0, 0, 0]),
    backend="matplotlib",
    # backend="plotly",
)


# TODO compute the resistance for the two hulls via the Sailboat class
# TODO compute the interactions the two hulls via the Sailboat class
