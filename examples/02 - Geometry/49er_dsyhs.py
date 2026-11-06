# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 00:20:18 2026

@author: jules
"""

import archibald.numpy as np

from archibald.optimization import Opti
from archibald.performance import OperatingPoint
from archibald.geometry.mesh import ArchibaldMesh
from archibald.geometry import Hull
from archibald.toolbox.mesh_utils import load_stl

from archibald.toolbox.math_utils import waterplane_normal

from archibald.dynamics.hydro import savitsky, holtrop, dsyhs, michell

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()

METHODS = {
    "composite": {
        
        "Rf": dsyhs.compute_Rf_dsyhs,
        # "Rw": dsyhs.compute_Rw_dsyhs,
        "Rtr": dsyhs.compute_Rtr_dsyhs,
        
        # "Rw": michell.compute_Rw_michell,
        "Rw": holtrop.compute_Rw_holtrop,
        # "Rw": lambda **kwargs: 0.,
        
        # "Rf": savitsky.compute_Rf_savitsky,
        # "Rtr": savitsky.compute_Rtr_savitsky,
        
        # "Rp": savitsky.compute_Rp_savitsky,
        # "Rsp": savitsky.compute_Rsp_savitsky,
        # "Raw": savitsky.compute_Raw_savitsky,
        # "Fz": savitsky.compute_Fz_savitsky,     # portance verticale (torseur)
        # "My": savitsky.compute_My_savitsky,     # moment de tangage (torseur)
    },
}

#%% DATA

hullStl = r'data/49er_data/hull.stl'

hull = Hull(
    mesh=hullStl,
)

rho = 1025.
displacement = 122. + 2*90. # kg

cog = np.array([
    2.40,
    .1,
    0.14
]) # m

T0, ref = 0., 0.290513

#%% OPTIMIZATION & HYDROSTATICS

overhang = 3.775 # deg for 49er

opti = Opti()

T = opti.variable(init_guess=0.)
trim = 3.

op_point = OperatingPoint(
    dz=-T,
    trim=-trim,
)

hull.mesh.transform(op_point)
hull.compute_hydrostatics_properties(op_point)

Fb, Mb = hull.compute_buoyancy(op_point, recompute_statics=False)

opti.subject_to([(Fb[2] - ref * 1025 * 9.8066) == 0.])

sol = opti.solve()

hull = Hull(
    mesh=hullStl,
)

op_point = OperatingPoint(
    dz=-sol(T),
    trim=-trim,
)

hull.mesh.transform(op_point)
hull.compute_hydrostatics_properties(op_point)

stw = np.linspace(0., 18., 50)

Rf = []
Rw = []
Rtr = []
R = []
TRIM = []

for s in stw:
    op_point = OperatingPoint(
        stw=s,
        dz=-sol(T),
        trim=-trim,
    )
    
    hull.compute_resistance(
        op_point,
        # method="dsyhs",
        # method="holtrop",
        # **{'Csternchoice': 1, 'Bulbchoice': 0},
        method=METHODS["composite"],
    )
    
    hrc = hull.resistance_components
    
    Rf.append(hrc["Rf"])
    Rw.append(hrc["Rw"])
    Rtr.append(hrc["Rtr"])
    R.append(hrc["_total"])
    TRIM.append(trim)

g = 9.8066

Rf = np.asarray(Rf) / g
Rw = np.asarray(Rw) / g
Rtr = np.asarray(Rtr) / g
R = np.asarray(R) / g
TRIM = np.asarray(TRIM)

#%% EXPERIMENTAL DATA (Bethwaite & Watin 2007)

speed_exp = np.array([
    4.3, 4.3, 4.5, 5.5, 
    5.9, 5.9, 6.8, 6.9, 8.0, 8.1, 
    9.8, 9.9, 11.5, 11.8, 11.9, 14.0, 14.6, 14.7, 17.8, 18.0, 18.1
])

drag_exp = np.array([
    3.0, 3.5, 3.0, 6.2, 
    12.0, 12.5, 16.8, 17.2, 19.8, 20.0, 
    25.5, 25.8, 34.0, 33.2, 33.0, 42.0, 41.5, 42.2, 51.5, 53.0, 53.5
])

#%% PLOTTING WITH DUAL AXIS

fig, ax1 = plt.subplots(figsize=(8, 5))

b = 1.1228946447372437
L = 4.8577

# Resistance component fill layers
ax1.fill_between(
    stw,
    0,
    Rf,
    alpha=0.5,
    label="Frictional resistance",
)

ax1.fill_between(
    stw,
    Rf,
    Rf + Rw,
    alpha=0.5,
    label="Wave resistance",
)

ax1.fill_between(
    stw,
    Rf + Rw,
    Rf + Rw + Rtr,
    alpha=0.5,
    label="Transom resistance",
)

# Total resistance line
ax1.plot(
    stw,
    R,
    linewidth=2,
    color="darkblue",
    label="Total resistance",
)

# Experimental data scatter points
ax1.scatter(speed_exp, drag_exp, label="Bethwaite & Watin (2007)", c="red", s=10., zorder=5)

# --- Secondary Y-axis for Trim ---
ax2 = ax1.twinx()
ax2.plot(
    stw, 
    TRIM, 
    color="black", 
    linestyle="--", 
    linewidth=2, 
    label="Trim"
)
ax2.set_ylabel("Trim [deg]")
ax2.grid(False)  # Disable grid on secondary axis to prevent clutter
ax2.set_ylim(-5, 15)

# --- Axis Labels and Limits ---
ax1.set_xlabel("STW [kn]")
ax1.set_ylabel("Resistance [kg]")
ax1.set_xlim(stw[0], stw[-1])
ax1.set_ylim(0, 70)

# --- Combined Legend ---
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper left")

fig.tight_layout()
plt.show()