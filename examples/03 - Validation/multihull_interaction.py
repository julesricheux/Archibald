# -*- coding: utf-8 -*-
"""
Catamaran interference: Jamaluddin regressions, Michell, Miyazawa, Insel & Molland.

Reference: Iqbal & Trimulyono, "Numerical study of the Cb influence on catamaran resistance",
KAPAL 11(3), 2014, which quote the regressions of Jamaluddin (2012) and print their result for the
parent demihull. Tables 5 and 6 of that paper are compared here with the module. This is a check of
the coefficients through a secondary source, not of the regression against tank data.

Parent demihull of the paper: Lwl 19.963 m, B 2.223 m, T 1.113 m, Cb 0.394, volume 19.443 m3,
slenderness L / volume^(1/3) = 7.42.

!! Michell needs Cp and Cx, which the paper does not give. Cp = 0.693 is the value recalled for the NPL
5b hull of Molland et al. (1994), not read in a source: please check it. Cx follows from Cb = Cp Cx.
The script prints how much the factor depends on it, which is a lot near a hollow of the demihull Cw.

Michell's factor is a theoretical prediction of the same quantity. The regressions were fitted on
measurements with a thin ship Cw as reference, so their tau also absorbs the error of thin ship
theory. Agreement should only be expected to be qualitative.

The Insel & Molland method is a table reader with no data shipped: ``insel_molland_tables.csv`` is
a blank for measured factors. Its interpolation is checked here on the table of Jamaluddin.
"""
import matplotlib.pyplot as plt
import numpy as onp

from archibald.dynamics.hydro import insel_molland, jamaluddin, michell, miyazawa
from archibald.toolbox import units as u

from validation_common import BLUE, GREY, ORANGE, AQUA, comparison_table, finish, read_table, style


#%% DEMIHULL AND PUBLISHED TABLES

Lwl, Bwl, T, volume, Cb = 19.963, 2.223, 1.113, 19.443, 0.394
Cp = 0.693  # Not from the paper, see above
Cx = Cb / Cp
rho, g = 1025., 9.81

SL_TABLE = [0.2, 0.3, 0.4]
FORM_FACTOR_TABLE = [1.4047, 1.3942, 1.3882]  # Table 5, 1 + beta k
FN_TABLE = jamaluddin.FN_TAU
TAU_TABLE = onp.array([                       # Table 6, rows Fn, columns S/L
    [0.627, 0.358, 0.241],
    [1.456, 1.023, 0.797],
    [0.976, 0.854, 0.777],
    [0.990, 0.935, 0.898],
    [0.830, 0.732, 0.670],
    [0.669, 0.622, 0.591],
])


def michell_tau(Fn, SL, Cp=Cp, Cx=Cx, n_u=401):
    """Wave interference factor R_cat / (2 R_demihull) of the Michell matrix."""
    stw = Fn * onp.sqrt(g * Lwl) / u.kt
    S = SL * Lwl
    R = michell.compute_Rw_matrix_michell(
        stw=stw, Lwl=[Lwl] * 2, Bwl=[Bwl] * 2, T=[T] * 2, Cp=[Cp] * 2, Cx=[Cx] * 2,
        x=[0., 0.], y=[-S / 2., S / 2.], rho=rho, g=g, n_u=n_u,
    )
    return float((R[0][0] + R[0][1] + R[1][0] + R[1][1]) / (R[0][0] + R[1][1]))


#%% TABLES OF THE PAPER

# The module gives the interference part. The base form factor of the same regression restores 1 + beta k.
base_form_factor = 3.03 * (Lwl / volume ** (1. / 3.)) ** (-0.40)
comparison_table("Jamaluddin, form factor 1 + beta k, Table 5", [
    (f"S/L = {SL}", published, base_form_factor * (1. + float(jamaluddin.viscous_interference_ratio(Lwl, volume, SL))), "")
    for SL, published in zip(SL_TABLE, FORM_FACTOR_TABLE)
], tolerance=0.1)

differences = comparison_table("Jamaluddin, wave interference tau, Table 6", [
    (f"Fn = {Fn}, S/L = {SL}", TAU_TABLE[i, j], jamaluddin.tau_jamaluddin(Fn, SL), "")
    for i, Fn in enumerate(FN_TABLE) for j, SL in enumerate(SL_TABLE)
], tolerance=1.0)
print(f"  Largest difference {max(abs(d) for d in differences):.2f} %")


#%% TABLE READER AND LIMITS

# The table reader, on the published table: at its nodes it must return Rw (tau - 1), in the two directions
Rw = [1000., 1000.]
check = []
for i, Fn in enumerate(FN_TABLE):
    for j, SL in enumerate(SL_TABLE):
        S = SL * Lwl
        dRw = insel_molland.compute_dRw_insel_molland(
            stw=Fn * onp.sqrt(g * Lwl) / u.kt, Lwl=[Lwl] * 2, y=[-S / 2., S / 2.], g=g, Rw=Rw,
            fn_grid=FN_TABLE, sl_grid=SL_TABLE, tau_table=TAU_TABLE.tolist(),
        )
        check.append(abs(float(dRw[0][1]) - 1000. * (TAU_TABLE[i, j] - 1.)))
print(f"\nInsel & Molland table reader, largest error at the nodes: {max(check):.2e} N")

print("Michell, tau - 1 for hulls far apart at Fn 0.4 (theory: tends to 0 as the transverse waves spread):")
print(f"  {'S/L':<6}{'n_u = 401':>11}{'n_u = 40001':>13}")
for SL in [1., 5., 20., 50.]:
    print(f"  {SL:<6g}{michell_tau(0.4, SL) - 1.:>11.4f}{michell_tau(0.4, SL, n_u=40001) - 1.:>13.4f}")
print(f"Michell at Fn 0.37, S/L 0.3: tau = {michell_tau(0.37, 0.3):.3f}, "
      f"with Cp 0.65: {michell_tau(0.37, 0.3, Cp=0.65, Cx=Cb / 0.65):.3f}, "
      f"with Cp 0.75: {michell_tau(0.37, 0.3, Cp=0.75, Cx=Cb / 0.75):.3f}")

print("\nViscous interference of the parent demihull, increase of the viscous resistance [%]")
print(f"  {'S/L':<6}{'S/B':>7}{'Jamaluddin':>12}{'Miyazawa':>11}")
SL_fine = onp.linspace(0.2, 0.6, 41)
jam, miy = [], []
for SL in SL_fine:
    S = SL * Lwl
    jam.append(100. * jamaluddin.compute_dRf_jamaluddin(Lwl=[Lwl] * 2, volume=[volume] * 2, y=[-S / 2., S / 2.], Rf=[1., 1.])[0])
    miy.append(100. * miyazawa.compute_dRf_miyazawa(Bwl=[Bwl] * 2, y=[-S / 2., S / 2.], Rf=[1., 1.])[0])
for SL in SL_TABLE:
    k = int(onp.argmin(abs(SL_fine - SL)))
    print(f"  {SL:<6}{SL * Lwl / Bwl:>7.2f}{float(jam[k]):>12.2f}{float(miy[k]):>11.2f}")

measured = read_table("insel_molland_tables.csv")


#%% PLOTS

Fn_fine = onp.linspace(0.17, 0.70, 107)
fig, axes = plt.subplots(1, 4, figsize=(14, 3.9))
for j, SL in enumerate(SL_TABLE):
    ax = axes[j]
    ax.axhline(1., color=GREY, linewidth=1)
    ax.plot(Fn_fine, [michell_tau(Fn, SL) for Fn in Fn_fine], color=ORANGE, linewidth=2, label="Michell (theory)")
    ax.plot(Fn_fine, jamaluddin.tau_jamaluddin(Fn_fine, SL), color=BLUE, linewidth=2, label="Jamaluddin, module")
    ax.plot(FN_TABLE, TAU_TABLE[:, j], "o", color="black", markersize=5, label="Iqbal & Trimulyono, Table 6")
    if measured is not None:
        mask = onp.array(measured["SL"]) == SL
        ax.plot(onp.array(measured["Fn"])[mask], onp.array(measured["tau"])[mask], "s", color=AQUA, label="Measured")
    ax.set_ylim(0., 2.5)
    style(ax, title=f"Wave interference tau, S/L = {SL}", xlabel="Froude number")
axes[0].legend(frameon=False, fontsize=8)

axes[3].plot(SL_fine, jam, color=BLUE, linewidth=2, label="Jamaluddin")
axes[3].plot(SL_fine, miy, color=AQUA, linewidth=2, label="Miyazawa (B/S)^2")
style(axes[3], title="Viscous interference [%]", xlabel="S/L")
axes[3].set_ylim(bottom=0.)
axes[3].legend(frameon=False, fontsize=8)

fig.tight_layout()
finish(fig, "multihull_interaction")
