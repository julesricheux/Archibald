# -*- coding: utf-8 -*-
"""
Wageningen B-series open water polynomials, ``BSeriesPropeller``.

Two independent uses of the same polynomials:

1. Holtrop & Mennen (1982), worked example of the propulsion of a single screw ship, p. 171.
   Z = 4, AE/A0 = 0.7393, D = 8 m, J = 0.7184. It prints KTs, KQo and the efficiency, with the
   viscous correction of the ITTC-1978 method, but not the pitch ratio. So P/D is solved here from the
   printed KT, and the printed KQ and efficiency are then a true check.

2. MIT 2.611 course notes, "KT-KQ design" (Kerwin): B5-75, P/D = 1, intersection of the polynomial
   with KT = 0.544 J^2 and KT = 0.544 J^4. The values are read from a chart and given to 2 digits.

Tabulated polynomial values, as in Bernitsas et al. (1981), are a blank: data/bernitsas_kt_kq_points.csv.
"""
import matplotlib.pyplot as plt
import numpy as onp

from archibald.geometry.propeller import BSeriesPropeller

from validation_common import BLUE, GREY, ORANGE, AQUA, YELLOW, comparison_table, finish, read_table, style


def propeller(blades, area_ratio, pitch_ratio):
    return BSeriesPropeller(blades_number=blades, blade_area_ratio=area_ratio, diameter=1., pitch_ratio=pitch_ratio)


def scalar(value):
    """The polynomials return one element arrays."""
    return onp.asarray(value, dtype=float).item()


def bisection(function, low, high, n=60):
    f_low = function(low)
    for _ in range(n):
        mid = 0.5 * (low + high)
        f_mid = function(mid)
        if f_mid * f_low > 0.:
            low, f_low = mid, f_mid
        else:
            high = mid
    return 0.5 * (low + high)


#%% 1, HOLTROP & MENNEN (1982)

J = 0.7184
Z, area_ratio, D = 4, 0.7393, 8.
KT_ship, KQ_ship, eta_published = 0.18802, 0.033275, 0.6461
dCD, chord = 0.000956, 3.065  # Viscous correction of the example, chord at 0.75 R [m]


def thrust_error(pitch_ratio):
    KT = scalar(propeller(Z, area_ratio, pitch_ratio).compute_kt(J))
    return KT + dCD * 0.3 * pitch_ratio * chord * Z / D - KT_ship


pitch_ratio = bisection(thrust_error, 0.6, 1.4)
p = propeller(Z, area_ratio, pitch_ratio)
KT = scalar(p.compute_kt(J)) + dCD * 0.3 * pitch_ratio * chord * Z / D
KQ = scalar(p.compute_kq(J)) - dCD * 0.25 * chord * Z / D
eta = J * KT / (2. * onp.pi * KQ)

print(f"Pitch ratio that gives the printed KT: P/D = {pitch_ratio:.4f} (not printed in the paper)")
# The paper prints KTs at ship scale but KQo with an index "o", and does not say whether the viscous
# correction is in KQo. So the three possible readings are given. The polynomial is the same in all.
KQ_polynomial = scalar(p.compute_kq(J))
pitch_ratio_uncorrected = bisection(lambda x: scalar(propeller(Z, area_ratio, x).compute_kt(J)) - KT_ship, 0.6, 1.4)
KQ_uncorrected = scalar(propeller(Z, area_ratio, pitch_ratio_uncorrected).compute_kq(J))
comparison_table("Holtrop & Mennen (1982), J = 0.7184", [
    ("KT at ship scale (imposed)", KT_ship, KT, ""),
    ("KQo, KQ corrected", KQ_ship, KQ, "KT and KQ both at ship scale"),
    ("KQo, KQ not corrected", KQ_ship, KQ_polynomial, "closest reading, KT at ship scale"),
    ("KQo, neither corrected", KQ_ship, KQ_uncorrected, "KT and KQ both open water"),
    ("efficiency, KQ corrected", eta_published, eta, ""),
    ("efficiency, KQ not corrected", eta_published, J * KT / (2. * onp.pi * KQ_polynomial), ""),
], tolerance=1.0)


#%% 2, MIT 2.611 NOTES

mit = propeller(5, 0.75, 1.0)
points = []
rows = []
for exponent, J_published, eta_notes in [(2, 0.64, 0.61), (4, 0.74, 0.67)]:
    J_found = bisection(lambda x: scalar(mit.compute_kt(x)) - 0.544 * x ** exponent, 0.2, 1.2)
    KT_found, KQ_found, eta_found = [scalar(value) for value in mit.compute_performance(J_found)]
    points.append((J_found, KT_found, eta_found))
    rows += [
        (f"J, KT = 0.544 J^{exponent}", J_published, J_found, "2 digits"),
        (f"efficiency, KT = 0.544 J^{exponent}", eta_notes, eta_found, "2 digits"),
    ]
comparison_table("MIT 2.611 notes, B5-75, P/D = 1", rows, tolerance=1.5)

tabulated = read_table("bernitsas_kt_kq_points.csv")
if tabulated is not None:
    rows = []
    for Z, area, pitch, J, KT, KQ in zip(*(tabulated[key] for key in ("Z", "AE_A0", "P_D", "J", "KT", "KQ"))):
        tested = propeller(int(Z), area, pitch)
        rows.append((f"KT, Z{Z:g} A{area:g} P{pitch:g} J{J:g}", KT, scalar(tested.compute_kt(J)), ""))
        rows.append((f"KQ, Z{Z:g} A{area:g} P{pitch:g} J{J:g}", KQ, scalar(tested.compute_kq(J)), ""))
    comparison_table("Bernitsas et al. (1981)", rows)


#%% PLOT

J_range = onp.linspace(0.02, 1.3, 130)
fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))

for pitch, color in zip([0.6, 0.8, 1.0, 1.2], [BLUE, ORANGE, AQUA, YELLOW]):
    prop = propeller(5, 0.75, pitch)
    KT_curve = onp.array([scalar(prop.compute_kt(x)) for x in J_range])
    KQ_curve = onp.array([scalar(prop.compute_kq(x)) for x in J_range])
    keep = KT_curve > 0.
    eta_curve = J_range * KT_curve / (2. * onp.pi * KQ_curve)
    axes[0].plot(J_range[keep], KT_curve[keep], color=color, linewidth=2, label=f"P/D = {pitch}")
    axes[0].plot(J_range[keep], 10. * KQ_curve[keep], color=color, linewidth=1.2, linestyle="--")
    axes[1].plot(J_range[keep], eta_curve[keep], color=color, linewidth=2, label=f"P/D = {pitch}")

axes[0].plot([x[0] for x in points], [x[1] for x in points], "o", color="black", markersize=5, label="Solved points of the notes")
axes[1].plot([0.64, 0.74], [0.61, 0.67], "o", color="black", markersize=5, label="MIT 2.611 notes")
axes[1].plot(J, eta_published, "s", color=GREY, markersize=6, label="Holtrop & Mennen, B4-74")
style(axes[0], title="B5-75: KT (solid), 10 KQ (dashed)", xlabel="Advance coefficient J")
style(axes[1], title="Open water efficiency", xlabel="Advance coefficient J")
axes[0].set_ylim(bottom=0.)
axes[1].set_ylim(0., 0.9)
axes[0].legend(frameon=False, fontsize=8)
axes[1].legend(frameon=False, fontsize=8)

fig.tight_layout()
finish(fig, "propeller_b_series")
