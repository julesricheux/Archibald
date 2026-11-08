# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 18:18:45 2026

@author: jrich
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
    # "savitsky": {
    #     "Rf": savitsky.compute_Rf_savitsky,
    #     "Rp": savitsky.compute_Rp_savitsky,
    #     "Rsp": savitsky.compute_Rsp_savitsky,
    #     "Rtr": savitsky.compute_Rtr_savitsky,
    #     "Raw": savitsky.compute_Raw_savitsky,
    #     "Fz": savitsky.compute_Fz_savitsky,     # portance verticale (torseur)
    #     # "My": savitsky.compute_My_savitsky,     # moment de tangage (torseur)
    # },
    "composite": {
        
        "Rf": dsyhs.compute_Rf_dsyhs,
        # "Rw": dsyhs.compute_Rw_dsyhs,
        "Rtr": dsyhs.compute_Rtr_dsyhs,
        
        # "Rw": michell.compute_Rw_michell,
        "Rw": holtrop.compute_Rw_holtrop,
        # "Rw": lambda **kwargs: 0.,
        
        # "Rf": savitsky.compute_Rf_savitsky,
        # "Rtr": savitsky.compute_Rtr_savitsky,
        
        "Rp": savitsky.compute_Rp_savitsky,
        "Rsp": savitsky.compute_Rsp_savitsky,
        "Raw": savitsky.compute_Raw_savitsky,
        "Fz": savitsky.compute_Fz_savitsky,     # portance verticale (torseur)
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

# hullStl = r'data/49er_data/hull.stl'

# mesh = ArchibaldMesh(
#     *load_stl(hullStl)
# )

T0, ref = 0., 0.290513
# T0, ref = 0.01, 0.234148

#%%

overhang = 3.775 # deg for 49er
hullStl = r'data/49er_data/hull.stl'

stw = np.arange(4, 18+1, 2)

Rf = []
Rw = []
Rtr = []
Rp = []
Rsp = []
R = []

TRIM = []

for s in stw:

    opti = Opti()
    
    hull = Hull(
        mesh=hullStl,
    )
    
    # T = 3e-2
    # trim = overhang
    T = 0.
    trim = 0.
    # T = 0.13
    # trim = 4.
    # # trim = 2.
    # T = s / 100
    # trim = s / 5.

    
    T = opti.variable(init_guess=T)
    # trim = opti.variable(init_guess=trim, lower_bound=overhang + 2.)
    trim = opti.variable(init_guess=trim, lower_bound=-5., upper_bound=10.)
    
    op_point = OperatingPoint(
        stw=s,
        dz=-T,
        trim=-trim,
    )
    
    hull.mesh.transform(op_point)
    
    # hull.mesh.draw(
    #     draw_plane=True,
    #     point=np.array([0, 0, 0]),
    #     # backend="matplotlib",
    #     # backend="plotly",
    # )
        
    hull.compute_hydrostatics_properties(op_point)
    
    volume = hull.hydrostatics_data["volume"]
    
    tau = hull.mesh.keel_line_angle(window=(0.0, 0.5))["keel_angle"]
    deadrise = hull.mesh.deadrise_angles(locations=(0.66,))['deadrise']
    
    hull.compute_resistance(
        op_point,
        method=DEFAULT_RESISTANCE_METHODS["composite"],
        # **{'Csternchoice': 1, 'Bulbchoice': 0}
        **{'trim': tau, 'deadrise': deadrise, "use_Aws": True,}
    )
        
    froude =  op_point._stw / np.sqrt(hull.hydrostatics_data["Lwl"] * 9.8066)
    
    # opti.minimize(
    #     np.sum([hull.resistance_components[c] for c in ["Rf", "Rp", "Rtr", "Rsp"]])
    # )
    hrc = hull.resistance_components
    
    val = savitsky.validity_factor(
        **{
            "trim": trim - overhang,
            "stw": s,
            "Bwl": hull.hydrostatics_data["Bwl"],
            "g": op_point.environment.gravity
        }
    )
    
    # val = 1.
    
    tau = np.softplus(tau, beta=1e2)
    
    lam = hull.hydrostatics_data["Lwl"]
    CL0 = 0.0120 * tau ** 1.1 * np.sqrt(lam)
    CL_b = CL0 - 0.0065 * deadrise * CL0 ** 0.6
    L = 0.5 * 1025 * CL_b * op_point.environment.water.density * (s * 0.5144) ** 2. * hull.hydrostatics_data["Bwl"] ** 2.
    Dp = L * np.tand(tau)
    
    
    # Fz = hrc["Fz"]
    # Fx = hrc["Rp"]
    # Fz = L
    # Fx = Dp
    Fz = 0.
    Fx = 0.
    

    opti.minimize(
        hrc["Rf"] +
        hrc["Rw"] +
        hrc["Rtr"] + 
        Fx +
        # hrc["Rsp"] +
        0.
    )
    
    opti.subject_to([
        ((volume - ref) * rho * 9.8066 + Fz) == 0.,
    ])
    
    sol = opti.solve(
        max_iter=300,
        behavior_on_failure="return_last",
        # verbose=False
    )
    
    hrc = sol(hull.resistance_components)
    
    TRIM.append(sol(trim))
    
    print()
    print(f"SPEED {s} kts")
    print()
    print(f"Trim {TRIM[-1]:.3} deg")
    print(f"Fac {sol(val) * 100:.1f} %")
    print(f"Dyn lift% =\t{sol(Fz / (ref * rho * 9.8066)) * 100:.3f} %")
    print()
    
    Rf.append(hrc["Rf"])
    Rw.append(hrc["Rw"])
    # Rw.append(0.)
    Rtr.append(hrc["Rtr"])
    Rsp.append(hrc["Rsp"])
    Rp.append(sol(Fx))
    R.append(hrc["Rf"]+hrc["Rw"]+hrc["Rtr"]+sol(Fx))

g = 9.8066

Rf = np.asarray(Rf) / g
Rw = np.asarray(Rw) / g
Rtr = np.asarray(Rtr) / g
Rp = np.asarray(Rp) / g
Rsp = np.asarray(Rsp) / g
R = np.asarray(R) / g

#%%

# Bethwaite / Watin results

# Red experimental points
speed_red = [4.3, 4.3, 4.5, 5.5]
drag_red  = [3.0, 3.5, 3.0, 6.2]

# Blue experimental points
speed_blue = [5.9, 5.9, 6.8, 6.9, 8.0, 8.1]
drag_blue  = [12.0, 12.5, 16.8, 17.2, 19.8, 20.0]

# Green / Dark experimental points
speed_green = [9.8, 9.9, 11.5, 11.8, 11.9, 14.0, 14.6, 14.7, 17.8, 18.0, 18.1]
drag_green  = [25.5, 25.8, 34.0, 33.2, 33.0, 42.0, 41.5, 42.2, 51.5, 53.0, 53.5]

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

fig, ax1 = plt.subplots(figsize=(8, 5))

b = 1.1228946447372437
L = 4.8577

# stw = stw * 0.5144 / np.sqrt(L * 9.8066)
# stw = stw * 0.5144 / np.sqrt(b * 9.8066)

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

ax1.fill_between(
    stw,
    Rf + Rw + Rtr,
    Rf + Rw + Rtr + Rp,
    alpha=0.5,
    label="Pressure resistance",
)

ax1.fill_between(
    stw,
    Rf + Rw + Rtr + Rp,
    Rf + Rw + Rtr + Rp + Rsp,
    alpha=0.5,
    label="Spray resistance",
)

ax1.scatter(speed_exp, drag_exp, label="Bethwaite & Watin (2007)", c="red", s=10.)

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
ax2.set_ylim(-5, 15)  # Adjust limits as needed for your trim range

# --- Axis Labels and Limits ---
ax1.set_xlabel("STW [kn]")
ax1.set_ylabel("Resistance [kg]")
ax1.set_xlim(stw[0] - 1, stw[-1] + 1)
ax1.set_ylim(0, 10 * ((np.max(Rf + Rp + Rsp) + 10) // 10))

# --- Combined Legend ---
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper left")

fig.tight_layout()
plt.show()


