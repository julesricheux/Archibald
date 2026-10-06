# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 22:00:10 2026

@author: jules
"""

# -*- coding: utf-8 -*-
import archibald.numpy as np

from archibald.optimization import Opti
from archibald.performance import OperatingPoint
from archibald.geometry.mesh import ArchibaldMesh
from archibald.geometry import Hull
from archibald.toolbox.mesh_utils import load_stl

from archibald.toolbox.math_utils import waterplane_normal

from archibald.dynamics.hydro import savitsky, holtrop

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()


DEFAULT_RESISTANCE_METHODS = {
    # "dsyhs": {
    #     "Rf": dsyhs.compute_Rf_dsyhs,
    #     "Rw": dsyhs.compute_Rw_dsyhs,
    #     "Rtr": dsyhs.compute_Rtr_dsyhs,
    # },
    # "holtrop": {
    #     "Rf": holtrop.compute_Rf_holtrop,
    #     "Rw": holtrop.compute_Rw_holtrop,
    #     "Rb": holtrop.compute_Rb_holtrop,
    #     "Rtr": holtrop.compute_Rtr_holtrop,
    #     "Ra": holtrop.compute_Ra_holtrop,
    # },
    "savitsky": {
        "Rf": savitsky.compute_Rf_savitsky,
        "Rp": savitsky.compute_Rp_savitsky,
        "Rsp": savitsky.compute_Rsp_savitsky,
        "Rtr": savitsky.compute_Rtr_savitsky,
        "Fz": savitsky.compute_Fz_savitsky,     # portance verticale (torseur)
        # "My": savitsky.compute_My_savitsky,     # moment de tangage (torseur)
    },
}

#%% DATA


rho = 1025.
displacement = 122. + 2*90. # kg

cog = np.array([
    2.40,
    .1,
    0.14
]) # m

# hullStl = r'data/49er_data/hull.stl'

# mesh = ArchibaldMesh(
#     *load_stl(hullStl)
# )

T0, ref = 0., 0.200513
# T0, ref = 0.01, 0.234148

#%%

T = 0.06
trim = 2.

stw = np.linspace(9., 20., 50)

Rf = []
Rp = []
Rtr = []
Rsp = []
R = []

for s in stw:
    
    hullStl = r'data/49er_data/hull.stl'

    hull = Hull(
        mesh=hullStl,
    )

    op_point = OperatingPoint(
        stw=s,
        dz=-T,
        trim=-trim,
    )

    hull.mesh.transform(op_point)

    hull.compute_hydrostatics_properties(op_point)

    hull.compute_resistance(
        op_point,
        method=DEFAULT_RESISTANCE_METHODS["savitsky"],
        **{
            "trim": trim,
            "deadrise": 10.,
            "use_Aws": True,
        },
    )

    hrc = hull.resistance_components

    Rf.append(hrc["Rf"])
    Rp.append(hrc["Rp"])
    Rtr.append(hrc["Rtr"])
    Rsp.append(hrc["Rsp"])

    R.append(
        hrc["Rf"] +
        hrc["Rp"] +
        hrc["Rtr"] +
        hrc["Rsp"]
    )


Rf = np.asarray(Rf)
Rp = np.asarray(Rp)
Rtr = np.asarray(Rtr)
Rsp = np.asarray(Rsp)
R = np.asarray(R)


#%% PLOT

fig, ax = plt.subplots(figsize=(8, 5))

ax.fill_between(
    stw,
    0.,
    Rf,
    alpha=0.5,
    label="Frictional resistance",
)

ax.fill_between(
    stw,
    Rf,
    Rf + Rp,
    alpha=0.5,
    label="Pressure resistance",
)

ax.fill_between(
    stw,
    Rf + Rp,
    Rf + Rp + Rtr,
    alpha=0.5,
    label="Transom resistance",
)

ax.fill_between(
    stw,
    Rf + Rp + Rtr,
    Rf + Rp + Rtr + Rsp,
    alpha=0.5,
    label="Spray resistance",
)

ax.plot(
    stw,
    R,
    linewidth=2.,
    label="Total resistance",
)

ax.set_xlabel("STW [kn]")
ax.set_ylabel("Resistance [N]")
ax.grid(True, alpha=0.2)
ax.legend()

fig.tight_layout()
plt.show()
