# -*- coding: utf-8 -*-
"""
DSYHS residuary resistance against the towing tank data of the series itself.

The polynomial of Keuning & Sonnenberg (1998) was fitted on these hulls, so the comparison
shows both the quality of the regression and whether its coefficients were typed correctly.
Data: den Ouden (2022), 4TU.ResearchData, doi 10.4121/21501375 and 10.4121/21501402 (CC0).

The measured residuary resistance is obtained as in the DSYHS papers:
    Rr = Rt - 0.5 rho V^2 S Cf,  Cf = 0.075 / (log10(Re) - 2)^2,  Re = V * 0.7 Lwl / nu
at model scale, from the raw force and water temperature. nu(T) is the ITTC fresh water
formula. rho cancels out in the non-dimensional result to within 0.1 %.

Two predictions are compared with the measurements:
- the polynomial computed here from the table, at the exact Froude number;
- ``compute_Rw_dsyhs`` as shipped, which goes through a smooth clip and a b-spline.

Sysser 1 is the parent hull of the series.
"""
import matplotlib.pyplot as plt
import numpy as onp

from archibald.dynamics.hydro import dsyhs

from validation_common import BLUE, GREY, ORANGE, finish, read_table, style


#%% DATA

g = 9.81
rho = 1000.
hydro = read_table("dsyhs_hydrostatics.csv")
tests = read_table("dsyhs_bare_hull_upright.csv")

hulls = {}
for i, sysser in enumerate(hydro["Sysser"]):
    if hydro["Overhang"][i] == 1. and sysser not in hulls:
        hulls[sysser] = {key: hydro[key][i] for key in hydro}


def hull_parameters(hull):
    """Arguments of ``compute_Rw_dsyhs`` from a line of the hydrostatics table."""
    return dict(
        volume=hull["volc0"],
        Lwl=hull["lwl0"],
        Bwl=hull["bwl0"],
        T=hull["tc0"],
        Awp=hull["aw0"],
        Cp=hull["cp0"],
        Cx=hull["cm0"],
        LCB_fpp=hull["lwl0"] / 2. - hull["lcb0"],
        LCF_fpp=hull["lwl0"] / 2. - hull["lcf0"],
    )


def polynomial(Fr, p):
    """Rr / (rho g volume) from the coefficient table, linearly interpolated in Fr."""
    table = dsyhs._keunig_coefs
    a = [onp.interp(Fr, onp.asarray(table["Fr"]), onp.asarray(table[f"a{i}"])) for i in range(8)]
    L = p["Lwl"]
    return a[0] + (
        a[1] * p["LCB_fpp"] / L
        + a[2] * p["Cp"]
        + a[3] * p["volume"] ** (2. / 3.) / p["Awp"]
        + a[4] * p["Bwl"] / L
        + a[5] * p["LCB_fpp"] / p["LCF_fpp"]
        + a[6] * p["Bwl"] / p["T"]
        + a[7] * p["Cx"]
    ) * p["volume"] ** (1. / 3.) / L


def nu_fresh_water(T_C):
    """Kinematic viscosity of fresh water [m2/s], ITTC formula."""
    return ((0.585e-3 * (T_C - 12.) - 0.03361) * (T_C - 12.) + 1.235) * 1e-6


#%% COMPUTATION

points = []
for i, sysser in enumerate(tests["Sysser"]):
    hull = hulls.get(sysser)
    if hull is None:
        continue
    p = hull_parameters(hull)
    V, Rt = tests["V"][i], tests["Rt"][i]
    Fr = V / onp.sqrt(g * p["Lwl"])
    if not 0.125 <= Fr <= 0.60:
        continue
    Re = V * 0.7 * p["Lwl"] / nu_fresh_water(tests["T_C"][i])
    Cf = 0.075 / (onp.log10(Re) - 2.) ** 2
    Rr = (Rt - 0.5 * rho * V ** 2 * hull["sc0"] * Cf) / (rho * g * p["volume"])
    shipped = float(dsyhs.compute_Rw_dsyhs(Fr=Fr, rho=rho, g=g, **p)) / (rho * g * p["volume"])
    points.append((sysser, Fr, Rr, polynomial(Fr, p), shipped, hull["Series"]))

points = onp.array(points, dtype=float)
sysser, Fr, measured, poly, shipped, series = points.T


#%% RESULTS

print(f"{len(onp.unique(sysser)):.0f} hulls and {len(Fr)} points with 0.125 <= Fn <= 0.60")
print("\nResiduary resistance Rr / (rho g volume) x 1000, mean absolute error, against the tank data:")
print(f"  {'Fn':<14}{'points':>7}{'measured':>10}{'polynomial':>12}{'shipped':>10}")
edges = [0.125, 0.2, 0.3, 0.4, 0.5, 0.6001]
for low, high in zip(edges[:-1], edges[1:]):
    mask = (Fr >= low) & (Fr < high)
    print(
        f"  {low:.3f}-{high:.3f}  {mask.sum():>7}{1e3 * measured[mask].mean():>10.2f}"
        f"{1e3 * onp.abs(poly - measured)[mask].mean():>12.2f}{1e3 * onp.abs(shipped - measured)[mask].mean():>10.2f}"
    )

parent = hulls[1.]
p = hull_parameters(parent)
Fr_fine = onp.linspace(0.13, 0.6, 95)
print("\nEffect of the smooth clip of compute_Rw_dsyhs, parent hull, x 1000:")
print(f"  {'Fn':<8}{'table':>9}{'shipped':>10}{'diff %':>9}")
for F in [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]:
    a = polynomial(F, p)
    b = float(dsyhs.compute_Rw_dsyhs(Fr=F, rho=rho, g=g, **p)) / (rho * g * p["volume"])
    print(f"  {F:<8.2f}{1e3 * a:>9.3f}{1e3 * b:>10.3f}{100. * (b - a) / a:>9.1f}")

fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))

mask = sysser == 1.
axes[0].plot(Fr[mask], 1e3 * measured[mask], "o", color="black", markersize=5, label="Measured, Sysser 1", zorder=3)
axes[0].plot(Fr_fine, [1e3 * polynomial(F, p) for F in Fr_fine], color=BLUE, linewidth=2, label="Polynomial from the table")
axes[0].plot(
    Fr_fine,
    [1e3 * float(dsyhs.compute_Rw_dsyhs(Fr=F, rho=rho, g=g, **p)) / (rho * g * p["volume"]) for F in Fr_fine],
    color=ORANGE,
    linewidth=2,
    label="compute_Rw_dsyhs",
)
style(axes[0], title="Parent hull: Rr / (rho g volume) x 1000", xlabel="Froude number")
axes[0].legend(frameon=False)

top = 1e3 * max(measured.max(), shipped.max()) * 1.05
axes[1].plot([0, top], [0, top], color=GREY, linewidth=1)
axes[1].plot(1e3 * measured, 1e3 * shipped, "o", color=ORANGE, markersize=3, alpha=0.6, label="compute_Rw_dsyhs")
axes[1].plot(1e3 * measured, 1e3 * poly, "o", color=BLUE, markersize=3, alpha=0.6, label="Polynomial from the table")
axes[1].set_aspect("equal")
axes[1].set_xlim(0, top)
axes[1].set_ylim(0, top)
style(axes[1], title=f"All {len(onp.unique(sysser)):.0f} hulls, Fn 0.125 to 0.60", xlabel="Measured x 1000", ylabel="Predicted x 1000")
axes[1].legend(frameon=False)

fig.tight_layout()
finish(fig, "dsyhs_towing_tank")
