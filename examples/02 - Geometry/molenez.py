# -*- coding: utf-8 -*-
import archibald.numpy as np

from archibald.optimization import Opti
from archibald.performance import OperatingPoint
from archibald.geometry.mesh import ArchibaldMesh
from archibald.toolbox.mesh_utils import load_stl


#%% FUNCTIONS

def waterplane_normal(heel_deg: float, trim_deg: float) -> np.ndarray:
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

hullStl = r'data//molenez2_data//hull.stl'

mesh = ArchibaldMesh(
    *load_stl(hullStl)
)

# T0, ref = 1.0, 78.6457
T0, ref = 1.3, 116.487
# T0, ref = 1.4, 130.186
# T0, ref = 1.5, 144.240
# T0, ref = 10., 362.907

heel = 0.
trim = 0.
leeway = 0.

mesh.draw(
    draw_plane=True,
    point=np.array([0, 0, T0]),
    # normal=np.array([0., 1., 0.]),
    # backend="matplotlib",
    # backend="plotly",
    set_axis_visibility=True,
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

print(f"Volume : {sol(volume):.1f} m3")
print(f"Draft error : {sol(T-T0)/T0*100.:.1f} %")
print(f"Volume error : {sol(volume-ref)/ref*100.:.1f} %")
