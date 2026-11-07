# -*- coding: utf-8 -*-
"""
Holtrop & Mennen (1982), "An approximate power prediction method", International
Shipbuilding Progress 29, p. 170: the numerical example, with every intermediate value.

Archibald implements the 1984 revision of the method, not the 1982 one, so:
- coefficients shared by both versions must match the example closely;
- the form factor, the wave term m4 (m2 in 1982), the correlation allowance and the
  wake were revised in 1984 and are expected to differ a little. They are labelled so.

The published sum: 869.63 x 1.156 + 8.83 + 557.11 + 0.049 + 0 + 221.98 = 1793.26 kN.
RAPP (8.83 kN) is not modelled by the method here, as appendages are computed apart.

The example gives the transom area (16 m2) but no transom draught. Holtrop uses
FnT = V / sqrt(2 g AT / (B + B Cwp)), and ``compute_Rtr_holtrop`` uses V / sqrt(g Ttr).
Both are shown below: with the equivalent draught, and with a transom draught of 4 m.
"""
import math

import matplotlib.pyplot as plt
import numpy as onp

import archibald.toolbox.units as u
from archibald.dynamics.hydro import holtrop

from validation_common import BLUE, ORANGE, comparison_table, finish, style


#%% DATA (published, p. 170)

g, rho, nu = 9.81, 1025., 1.1883e-6  # sea water at 15 deg C, nu gives the published CF
stw = 25.  # [kt]

ship = dict(
    Lwl=205.,
    Lbp=205.,  # the wave terms of Holtrop use the waterline length L, not Lpp = 200
    Bwl=32.,
    T=10.,
    volume=37500.,
    Aws=7381.45,
    Atr=16.,
    Abt=20.,
    hB=4.,
    Cx=0.980,
    Cwp=0.750,
    Cp=0.5833,
    Cb=37500. / (205. * 32. * 10.),
    ie=12.08,
    lcb=-0.75,  # [%] relative to L/2
    rho=rho,
    g=g,
    nu=nu,
    stw=stw,
)
Cstern = 10.
propeller = dict(D=8., T_A=10., Ae_Ao=0.7393)
Ttr_equivalent = 2. * ship["Atr"] / (ship["Bwl"] * (1. + ship["Cwp"]))


#%% COMPUTATION

Vms = stw * u.kt
Fr = Vms / math.sqrt(g * ship["Lwl"])
Rf_bare = 0.5 * rho * Vms ** 2 * ship["Aws"] * 0.075 / (math.log10(Vms * ship["Lwl"] / nu) - 2.) ** 2

components = dict(
    Rf=float(holtrop.compute_Rf_holtrop(**ship, Csternchoice=4)),
    Rw=float(holtrop.compute_Rw_holtrop(**ship)),
    Rb=float(holtrop.compute_Rb_holtrop(**ship, Bulbchoice=1)),
    Rtr=float(holtrop.compute_Rtr_holtrop(**ship, Ttr=Ttr_equivalent)),
    Ra=float(holtrop.compute_Ra_holtrop(**ship)),
)
Rtr_real_draught = float(holtrop.compute_Rtr_holtrop(**ship, Ttr=4.))


#%% RESULTS

comparison_table(
    "Shared with the 1982 method: must match",
    [
        ("Fn", 0.2868, Fr, ""),
        ("Bare hull RF [kN]", 869.63, Rf_bare / 1e3, "ITTC-57 on the full S"),
        ("c7", 0.1561, holtrop.c_7(**ship), ""),
        ("c1", 1.398, holtrop.c_1(**ship), ""),
        ("c3", 0.02119, holtrop.c_3(**ship), "was 0.01586 before the fix of (0.56 Abt)^1.5"),
        ("c2", 0.7595, holtrop.c_2(**ship), "was 0.7882"),
        ("c5", 0.9592, holtrop.c_5(**ship), ""),
        ("m1", -2.1274, holtrop.m_1(**ship), "the code divides by T + 0.1, see README"),
        ("c15 (printed unsigned)", -1.69385, holtrop.c_15(**ship), ""),
        ("lambda", 0.6513, holtrop.lambda_(**ship), ""),
        ("LR [m]", 81.385, holtrop.L_R(**ship), ""),
        ("CP1", 0.5477, holtrop.C_P1(**ship), ""),
        ("c11", 1.250, holtrop.c_11(T=propeller["T_A"], D=propeller["D"]), ""),
        ("eta_R", 0.9931, holtrop.eta_R_single(**ship, Ae_Ao=propeller["Ae_Ao"]), ""),
        ("Pb", 0.6261, holtrop.p_b(ship["T"], ship["hB"], ship["Abt"]), ""),
        ("Fni", 1.5084, holtrop.Fr_i(stw, ship["T"], ship["hB"], ship["Abt"], g), ""),
        ("RB [kN]", 0.049, components["Rb"] / 1e3, ""),
        ("FnT with equivalent Ttr", 5.433, holtrop.Fr_T(stw, Ttr_equivalent, g), "equivalent draught, see docstring"),
    ],
)
comparison_table(
    "Revised in 1984: small differences are expected",
    [
        ("1 + k1", 1.156, holtrop.one_plus_k(**ship, Cstern=Cstern), "1984 form with c14"),
        ("(1 + k1) RF [kN]", 869.63 * 1.156, components["Rf"] / 1e3, "code also removes Atr from S"),
        ("RW [kN]", 557.11, components["Rw"] / 1e3, "m4 replaces m2"),
        ("CA", 0.000352, holtrop.C_A(ship["Lwl"]), "other formula in the code"),
        ("RA [kN]", 221.98, components["Ra"] / 1e3, ""),
        ("c9", 14.500, holtrop.c_9(**ship, D=propeller["D"]), "code removes Atr from S"),
        ("w", 0.2584, holtrop.w_single(**ship, **propeller, Cstern=Cstern), "1984 regression"),
    ],
    tolerance=5.,
)
print(f"\n  RTR, published 0.00 kN: {components['Rtr'] / 1e3:.2f} kN with the equivalent draught, "
      f"{Rtr_real_draught / 1e3:.1f} kN with a real transom draught of 4 m <--")
total = sum(components.values()) / 1e3 + 8.83
print(f"  Rtotal, published 1793.26 kN: {total:.1f} kN (with the published RAPP and the equivalent draught)")

names = ["(1+k)RF", "RW", "RA", "RB", "RTR"]
published = [869.63 * 1.156, 557.11, 221.98, 0.049, 0.]
computed = [components[key] / 1e3 for key in ("Rf", "Rw", "Ra", "Rb", "Rtr")]

fig, ax = plt.subplots(figsize=(7.5, 4))
positions = onp.arange(len(names))
ax.bar(positions - 0.19, published, width=0.36, color=BLUE, label="Holtrop & Mennen (1982)")
ax.bar(positions + 0.19, computed, width=0.36, color=ORANGE, label="Archibald (1984 revision)")
ax.set_xticks(positions, names)
style(ax, title="Resistance components at 25 kn [kN]")
ax.grid(axis="x", visible=False)
ax.legend(frameon=False)
fig.tight_layout()
finish(fig, "holtrop_1982_example")
