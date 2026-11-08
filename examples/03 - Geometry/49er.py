# -*- coding: utf-8 -*-
import archibald.numpy as np

from archibald.optimization import Opti
from archibald.performance import OperatingPoint
from archibald.geometry.mesh import ArchibaldMesh
from archibald.toolbox.mesh_utils import load_stl

from archibald.toolbox.math_utils import waterplane_normal

#%% DATA

rho = 1025.
displacement = 122. + 2*90. # kg

cog = np.array([
    2.40,
    .1,
    0.14
]) # m

hullStl = r'data/49er_data/hull.stl'

mesh = ArchibaldMesh(
    *load_stl(hullStl)
)

T0, ref = 0., 0.200513
# T0, ref = 0.01, 0.234148

mesh.draw(
    draw_plane=True,
    point=np.array([0, 0, T0]),
    # backend="matplotlib",
    # backend="plotly",
)

#%%

opti = Opti()

T = opti.variable(init_guess = T0)
# T = T0

op_point = OperatingPoint(dz=-T)

mesh.transform(op_point)

volume = mesh.hydrostatics()["volume"]

opti.minimize((volume - ref)**2.)

sol = opti.solve()

#%% DRAWING

print(f"Volume : {sol(volume)*1000:.1f} L")
# print(f"Draft error : {sol(T-T0)/T0*100.:.1f} %")
print(f"Volume error : {sol(volume-ref)/ref*100.:.1f} %")
