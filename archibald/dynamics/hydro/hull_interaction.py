# -*- coding: utf-8 -*-
"""
Multihull resistance: individual hulls first, then the interaction between them.
"""

import os

from archibald.dynamics.hydro import holtrop, dsyhs, multihull
from archibald.geometry.hull import Hull
from archibald.performance import OperatingPoint

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()

stl = os.path.abspath(r"..\..\examples\02 - Geometry\data\molenez2_data\hull.stl")

T0 = 1.3
env = {'rho': 1025., 'g': 9.81, 'nu': 1.189e-6}

# Two identical demihulls, 6 m apart (centre to centre), no stagger.
hulls = [Hull(mesh=stl), Hull(mesh=stl)]
positions = [(0., -3.), (0., 3.)]            # (x, y) of each LCB [m]

op_point = OperatingPoint(stw=6., dz=-T0)

# 1. Individual hulls: hydrostatics, then resistance
for hull in hulls:
    hull.compute_buoyancy(op_point)
    hull.compute_resistance(
        op_point,
        method="holtrop",
        **{'Csternchoice': 1, 'Bulbchoice': 0}
    )
    print(hull.resistance_components)

# 2. Interaction between hulls (to be called as a Sailboat method with self.hulls)
for method in ("michell", "michell_raw"):
    interaction = multihull.compute_hull_interaction(
        hulls, op_point, method=method, positions=positions, **env,
    )
    print(method, interaction['dR'], interaction['dR_hull'])

# 3. Custom process, as for hull.compute_resistance
from archibald.dynamics.hydro import michell

custom_process = {"Rw": michell.compute_dRw_michell}
interaction = multihull.compute_hull_interaction(
    hulls, op_point, method=custom_process, positions=positions, **env,
)
print(interaction['dRw'])

# 4. Wave interference vs. spacing
SL_list = [0.2, 0.3, 0.4, 0.6, 1.0]
Lwl = hulls[0].hydrostatics_data['Lwl']
dR = []
for SL in SL_list:
    pos = [(0., -SL * Lwl / 2), (0., SL * Lwl / 2)]
    dR.append(multihull.compute_hull_interaction(
        hulls, op_point, method="michell", positions=pos, **env)['dR'])

plt.plot(SL_list, dR, marker='o')
plt.xlabel('S/L [-]')
plt.ylabel('Wave interference [N]')
plt.show()
