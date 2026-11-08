# -*- coding: utf-8 -*-
"""
Hull lateral force: Jones (1946), Inoue et al. (1981), Kijima & Nakiri (2003).

No published force or derivative could be read for these, so this is not a validation yet.
The formulas are checked by hand in ``archibald/dynamics/hydro/test_lateral_force.py``. Here:

- the three methods at their common linear limit, on a ship in their range of validity;
- the derivatives of the KVLCC2, ready to compare with published captive tests when they are in
  data/kvlcc2_derivatives.csv. For instance Yasukawa & Yoshimura (2015), J. Mar. Sci. Technol. 20,
  or the SIMMAN benchmark, with the sign and non dimensionalisation of the source.

The DSYHS data also has yawed tests, in 4TU data set 10.4121/21501402, but only with a keel:
it cannot test a bare hull method, and would serve a future yacht side force method.
"""
import matplotlib.pyplot as plt
import numpy as onp

from archibald.dynamics.hydro import inoue, jones, kijima

from validation_common import BLUE, GREY, ORANGE, AQUA, comparison_table, finish, read_table, style


#%% KVLCC2

# Particulars of Hizir et al. (2019), Table 1. Cp and Cwp are not printed there, and only enter
# through the aft hull terms of Kijima, whose default assumes the whole hull. They are noted as guesses.
ship = dict(Lwl=320., Bwl=58., T=20.8, Cb=0.8098, Cp=0.811, Cwp=0.90, rho=1025., V=7.97)
Q = 0.5 * ship["rho"] * ship["V"] ** 2 * ship["Lwl"] * ship["T"]
methods = [
    ("Jones", jones.compute_L_jones, jones.compute_Ri_jones, BLUE),
    ("Inoue", inoue.compute_L_inoue, inoue.compute_Ri_inoue, ORANGE),
    ("Kijima", kijima.compute_L_kijima, kijima.compute_Ri_kijima, AQUA),
]

small = onp.radians(0.01)
kappa = 2. * ship["T"] / ship["Lwl"]
print("Linear lift derivative, L / (0.5 rho V^2 L T beta), KVLCC2")
derivatives = {}
for name, lift, _, _ in methods:
    derivatives[name] = float(lift(beta=small, **ship)) / (Q * small)
    print(f"  {name:<8}{derivatives[name]:>8.4f}")

comparison_table("By hand", [
    ("Jones, pi k / 2", onp.pi / 2. * kappa, derivatives["Jones"], ""),
    ("Inoue, pi k / 2 + 1.4 Cb B / L", onp.pi / 2. * kappa + 1.4 * ship["Cb"] * ship["Bwl"] / ship["Lwl"], derivatives["Inoue"], ""),
])

N_beta = float(kijima.compute_N_kijima(beta=small, **ship)) / (Q * ship["Lwl"] * small)
print(f"\nYaw derivative N / (0.5 rho V^2 L^2 T beta), Kijima: {N_beta:.4f}")

published = read_table("kvlcc2_derivatives.csv")
if published is not None:
    comparison_table("KVLCC2 derivatives", [
        (name, value, derivatives.get(name, N_beta), source)
        for name, value, source in zip(published["method"], published["value"], published["source"])
    ], tolerance=10.)


#%% PLOTS

angles = onp.linspace(0., 20., 81)
fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
for name, lift, drag, color in methods:
    beta = onp.radians(angles)
    axes[0].plot(angles, [float(lift(beta=b, **ship)) / Q for b in beta], color=color, linewidth=2, label=name)
    axes[1].plot(angles, [float(drag(beta=b, **ship)) / Q for b in beta], color=color, linewidth=2, label=name)
style(axes[0], title="KVLCC2: lift / (0.5 rho V^2 L T)", xlabel="Leeway [deg]")
style(axes[1], title="KVLCC2: added resistance / (0.5 rho V^2 L T)", xlabel="Leeway [deg]")
axes[0].legend(frameon=False)
axes[1].legend(frameon=False)

fig.tight_layout()
finish(fig, "lateral_force")
