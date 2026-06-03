# -*- coding: utf-8 -*-
"""
Created on Tue May 26 18:27:52 2026

@author: jrich
"""
import archibald.numpy as np
import matplotlib.pyplot as plt

from archibald.geometry.mesh import ArchibaldMesh
from archibald.toolbox.mesh_utils import load_stl

hullStl = r'data//molenez2_data//hull.stl'

mesh = ArchibaldMesh(
    *load_stl(hullStl)
)

# 1. Define the reference data extracted from your comments
reference_data = {
    1.0:  78.646,
    1.1:  90.707,
    1.2: 103.320,
    1.3: 116.487,
    1.4: 130.186,
    1.5: 144.240,
    1.6: 158.537,
    1.7: 173.011,
    1.8: 187.623,
    1.9: 202.347,
    4.0: 362.907
}

ref_drafts = np.array(list(reference_data.keys()))
ref_volumes = np.array(list(reference_data.values()))

# 2. Calculate the measured volumes at the exact reference drafts
measured_volumes = []
for T in ref_drafts:
    point = np.array([0., 0., 1.]) * T
    # Assuming mesh.hydrostatics can take standard floats/numpy arrays here
    vol, _ = mesh.hydrostatics(point)
    
    # If your mesh function returns a CasADi variable, you might need to extract 
    # the numerical value using float(vol) or sol.value(vol). Assuming float here:
    measured_volumes.append(float(vol))

# 3. (Optional) Generate a smooth curve for the measured mesh across the whole range
smooth_drafts = np.linspace(min(ref_drafts), max(ref_drafts), 50)
smooth_measured = []
for T in smooth_drafts:
    point = np.array([0., 0., 1.]) * T
    vol, _ = mesh.hydrostatics(point)
    smooth_measured.append(float(vol))

# 4. Create the plot
plt.figure(figsize=(8, 5))

# Plot the continuous measured volume from the mesh
plt.plot(smooth_drafts, smooth_measured, label='Archibald measured volume', 
         color='#1f77b4', linestyle='-', linewidth=2, zorder=1)

# # Plot the specific measured points for direct comparison
# plt.scatter(ref_drafts, measured_volumes, label='Measured Points', 
#             color='#1f77b4', marker='s', s=50, zorder=2)

# Plot the reference points
plt.scatter(ref_drafts, ref_volumes, label='IGS file reference volume', 
            color='#d62728', marker='o', s=50, zorder=3)

# Formatting the plot
plt.title('Hull Volume vs. Draft (molenez2_data)', fontsize=14, pad=15)
plt.xlabel('Draft (m)', fontsize=12)
plt.ylabel('Volume (m³)', fontsize=12)
plt.legend(fontsize=10, loc='upper left')
plt.grid(True, linestyle='--', alpha=0.6)

# Improve layout and display
plt.tight_layout()
plt.show()
