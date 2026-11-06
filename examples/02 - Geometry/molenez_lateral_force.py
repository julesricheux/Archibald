# -*- coding: utf-8 -*-
"""
Side force and added resistance of the Molenez hull against leeway.

The hull is alone, at a fixed speed and immersion, and the three methods of
``archibald.geometry.hull.DEFAULT_LATERAL_METHODS`` are compared. Their sources
are in ``archibald/dynamics/hydro/``. Inoue and Kijima were fitted on merchant
ships: the Molenez is outside of their range, so read them as estimates.

Methods can also be mixed, with a dictionary of functions:
    boat.compute_lateral_force(op_point, method={"L": kijima.compute_L_kijima, "Ri": jones.compute_Ri_jones})
"""
import matplotlib.pyplot as plt
import numpy as onp

from archibald.geometry.hull import Hull
from archibald.geometry.sailboat import Sailboat
from archibald.performance import OperatingPoint


#%% DATA

boat = Sailboat(
    hulls=[Hull(name='hull', mesh=r'data//molenez2_data//hull.stl')],
)

stw = 8.         # speed through water [kt]
immersion = 1.3  # [m]
leeways = onp.linspace(0., 12., 13)  # positive to starboard [deg]
methods = ["jones", "inoue", "kijima"]


#%% COMPUTATION

side_force = {method: [] for method in methods}
added_resistance = {method: [] for method in methods}

for leeway in leeways:
    op_point = OperatingPoint(stw=stw, dz=-immersion, leeway=leeway)
    boat.hulls[0].compute_hydrostatics_properties(op_point)

    for method in methods:
        Flat, Mlat = boat.compute_lateral_force(
            op_point,
            method=method,
            recompute_statics=False,
        )
        side_force[method].append(Flat[0, 1])
        added_resistance[method].append(-Flat[0, 0])


#%% RESULTS

print(f"At {leeways[5]:.0f} deg of leeway and {stw:.0f} kt:")
for method in methods:
    print(
        f"{method:>7} : side force {side_force[method][5] / 1e3:6.2f} kN, "
        f"added resistance {added_resistance[method][5] / 1e3:5.2f} kN"
    )

colors = {"jones": "#2a78d6", "inoue": "#eb6834", "kijima": "#1baf7a"}
labels = {
    "jones": "Jones (1946)",
    "inoue": "Inoue et al. (1981)",
    "kijima": "Kijima & Nakiri (2003)",
}

fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharex=True)

panels = (
    (axes[0], side_force, "Side force (lift) [kN]"),
    (axes[1], added_resistance, "Added resistance (induced drag) [kN]"),
)
for ax, results, title in panels:
    for method in methods:
        ax.plot(leeways, onp.array(results[method]) / 1e3, color=colors[method], linewidth=2, label=labels[method])
    ax.set_title(title, loc="left", fontsize=11)
    ax.set_xlabel("Leeway [deg]")
    ax.set_xlim(leeways[0], leeways[-1])
    ax.set_ylim(bottom=0.)
    ax.grid(color="#e5e5e5", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)

axes[0].legend(frameon=False)
fig.suptitle(f"Molenez hull at {stw:.0f} kt and {immersion} m immersion", x=0.01, ha="left", fontsize=12)
fig.tight_layout()
plt.show()
