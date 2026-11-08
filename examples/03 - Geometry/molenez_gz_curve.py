# -*- coding: utf-8 -*-
"""
GZ curve of the Molenez: righting arm against heel, for a load case.

Heel is an ``opti.parameter`` swept from 0 to ``HEEL_MAX``. At each heel the
immersion and the trim are solved so that the hull floats the masses (Fz = 0,
My = 0), and the righting arm is read from the heeling moment of weight and
buoyancy, a couple once Fz = 0:

    GZ = -Mx / (m g)    positive when the moment brings the boat back upright.

Each solve starts from the previous one. The load cases are those of
``molenez_hybrid.py``.

Hull only: the deck edge, deckhouse and downflooding points are not modelled,
and the mesh is taken as watertight at every heel. With the "containers" case
the sweep converges at every step: maximum GZ 0.89 m near 32 deg, vanishing
stability near 82 deg. GZ is -0.005 m upright (TCG -0.001 m, and the centre of
buoyancy of the 908-face mesh is about 3 mm off centre), and the GM printed is
the slope of the first step, sensitive to the mesh resolution.
"""
import matplotlib.pyplot as plt
import numpy as onp

from archibald.geometry import Hull, Sailboat
from archibald.optimization import Opti
from archibald.performance import OperatingPoint
from archibald.weights import MassProperties


#%% SETTINGS

LOAD_CASE = "containers"
HEEL_MAX = 90.          # [deg]
HEEL_STEP = 2.5         # [deg]

# Displacement [t], LCG [m], TCG [m], VCG [m], as in molenez_hybrid.py (columns to be confirmed)
LOAD_CASES = {
    "lightship": (61.75, 11.13, -0.07, 1.78),
    "minimal": (66.65, 11.79, -0.012, 1.74),
    "containers": (120.43, 12.10, -0.001, 2.03),
    "mixed": (149.96, 12.12, -0.005, 1.93),
}


#%% BOAT

mass, lcg, tcg, vcg = LOAD_CASES[LOAD_CASE]

boat = Sailboat(
    name="molenez",
    hulls=[Hull(name="hull", mesh="data/molenez2_data/hull.stl")],
    masses=[MassProperties(mass=mass * 1e3, x_cg=lcg, y_cg=tcg, z_cg=vcg)],
)


#%% PROBLEM

opti = Opti()

heel = opti.parameter(0.)                                                 # positive to starboard [deg]
T = opti.variable(init_guess=1.4, lower_bound=-2., upper_bound=6.)       # immersion at the transom [m]
trim = opti.variable(init_guess=0., lower_bound=-15., upper_bound=15.)   # positive bow down [deg]

op_point = OperatingPoint(
    dz=-T,
    heel=heel,
    trim=trim,
)

boat.compute_weight(op_point)
boat.compute_buoyancy(op_point)
Ftot, Mtot = boat.compute_torsor(op_point)

residuals = Sailboat.equilibrium_residuals(
    Ftot,
    Mtot,
    dofs=["Fz", "My"],
    force_scale=1e4,
    moment_scale=1e5,
)
opti.subject_to([residual == 0. for residual in residuals.values()])

g = op_point.environment.gravity
gz = -Mtot[0, 0] / (mass * 1e3 * g)


#%% SWEEP

heels = onp.arange(0., HEEL_MAX + 1e-9, HEEL_STEP)
results = []

for value in heels:
    sol = opti.solve(
        parameter_mapping={heel: value},
        verbose=False,
        max_iter=100,
        behavior_on_failure="return_last",
    )
    converged = sol.stats()["success"]
    results.append((value, sol(T), sol(trim), sol(gz), converged))
    print(f"heel {value:5.1f} deg | T {sol(T):6.3f} m | trim {sol(trim):+6.2f} deg | GZ {sol(gz):+7.3f} m"
          + ("" if converged else "  NOT CONVERGED"))
    opti.set_initial_from_sol(sol, initialize_duals=False)

results = onp.array(results, dtype=float)


#%% RESULTS

ok = results[:, 4] > 0.5
gz_values = results[:, 3]
i_max = onp.argmax(onp.where(ok, gz_values, -onp.inf))
print(f"\nLoad case {LOAD_CASE}: {mass:.2f} t, VCG {vcg} m")
print(f"Maximum GZ : {gz_values[i_max]:.3f} m at {heels[i_max]:.1f} deg")

# Initial slope: GM = dGZ/dheel at 0, per radian
gm = (gz_values[1] - gz_values[0]) / onp.radians(heels[1] - heels[0])
print(f"GM (slope at 0 deg) : {gm:.3f} m")

vanishing = onp.nonzero(ok[1:] & (gz_values[1:] <= 0.) & (gz_values[:-1] > 0.))[0]
if len(vanishing):
    k = vanishing[0]
    angle = heels[k] + (heels[k + 1] - heels[k]) * gz_values[k] / (gz_values[k] - gz_values[k + 1])
    print(f"Angle of vanishing stability : {angle:.1f} deg")
else:
    print(f"GZ stays positive up to {heels[ok].max():.1f} deg")


#%% DRAWING

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(heels[ok], gz_values[ok], "-o", markersize=3, label=f"{LOAD_CASE}, {mass:.1f} t")
if (~ok).any():
    ax.plot(heels[~ok], gz_values[~ok], "x", color="red", label="not converged")
ax.plot([0., 180. / onp.pi], [0., gm], "--", color="grey", linewidth=1, label=f"GM = {gm:.2f} m")
ax.axhline(0., color="black", linewidth=0.8)
ax.set_xlim(0., HEEL_MAX)
ax.set_ylim(min(0., onp.nanmin(gz_values[ok])) - 0.05, 1.1 * onp.nanmax(gz_values[ok]))
ax.set_xlabel("Heel [deg]")
ax.set_ylabel("GZ [m]")
ax.set_title("Molenez righting arm, free trim")
ax.grid(True, alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig(f"molenez_gz_{LOAD_CASE}.png", dpi=150)
plt.show()
