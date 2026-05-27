# -*- coding: utf-8 -*-
import archibald.numpy as np

from archibald.optimization import Opti
from archibald.geometry.mesh import ArchibaldMesh
from archibald.toolbox.mesh_utils import load_stl

#%% FUNCTIONS

def waterplane_normal(
        heel_deg: float,
        trim_deg: float,
    ) -> np.ndarray:
    """
    Compute the unit normal vector of the waterplane given heel and trim angles.

    Parameters
    ----------
    heel_deg : float
        Heel angle in degrees (rotation around ship longitudinal axis, roll).
        Positive heel = starboard down.
    trim_deg : float
        Trim angle in degrees (rotation around ship transverse axis, pitch).
        Positive trim = bow down.

    Returns
    -------
    np.ndarray
        Normal vector of the waterplane (unit vector, shape (3,)).
    """

    # Convert to radians
    heel = np.deg2rad(heel_deg)
    trim = np.deg2rad(trim_deg)

    # Start with upright ship: waterplane normal is along +z
    n = np.array([0.0, 0.0, 1.0])

    # Rotation matrices
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(heel), -np.sin(heel)],
        [0, np.sin(heel), np.cos(heel)]
    ])

    Ry = np.array([
        [np.cos(trim), 0, np.sin(trim)],
        [0, 1, 0],
        [-np.sin(trim), 0, np.cos(trim)]
    ])

    # Apply rotations: first trim (pitch), then heel (roll)
    n_rot = Ry @ (Rx @ n)

    # Normalize
    return n_rot / np.linalg.norm(n_rot)

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

point = np.array([0., 0., 1.]) * T

volume, _ = mesh.hydrostatics(point)

opti.minimize((volume - ref)**2.)

sol = opti.solve()

#%% DRAWING

# mesh.draw(
#     draw_plane=True,
#     point=np.array([0, 0, sol(T)])
# )

volume, _ = mesh.hydrostatics(point)
print(f"Volume : {sol(volume)*1000:.1f} L")
# print(f"Draft error : {sol(T-T0)/T0*100.:.1f} %")
print(f"Volume error : {sol(volume-ref)/ref*100.:.1f} %")
