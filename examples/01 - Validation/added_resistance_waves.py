# -*- coding: utf-8 -*-
"""
Added resistance in waves: Mittendorf et al. (2022) and the DSYHS table.

Savitsky & Brown (1976) is checked in ``savitsky_planing.py``.

1. Mittendorf, Nielsen, Bingham & Liu, "Towards the uncertainty quantification of semi-empirical
   formulas applied to the added resistance of ships in waves of arbitrary heading", Ocean Engineering
   251, 2022, on the KVLCC2 in head seas at Fn = 0.142:
   compared with the 3-D linear potential code PRECAL as computed by Hizir, Kim, Turan & Day,
   Int. J. Naval Architecture and Ocean Engineering 11, 2019, Table 3, with the mesh convergence of
   that code. This is another method and not a measurement: towing tank values are a blank,
   data/kvlcc2_caw_experiment.csv.
   Particulars from Hizir et al., Table 1. Le and Lr are not given there: 60 m is the value of the
   commented KVLCC2 example at the end of ``mittendorf.py``.

2. ``compute_Raw_dsyhs``: at the nodes of its table the function must return the formula with
   the printed a and b. This checks the reading of the table and its interpolation, not the
   table itself, whose source could not be read: data/dsyhs_added_resistance_points.csv is a blank.
"""
import matplotlib.pyplot as plt
import numpy as onp

from archibald.dynamics.hydro import dsyhs, mittendorf

from validation_common import BLUE, GREY, ORANGE, AQUA, YELLOW, comparison_table, finish, read_table, style


#%% 1, KVLCC2

Lpp, Bwl, T, Cb, Le, Lr, kyy, Fn = 320., 58., 20.8, 0.8098, 60., 60., 0.25, 0.142

ratios = onp.linspace(0.3, 2.2, 96)
# mm_raw returns the mean, the upper bound (set shifted by +0.19967) and the lower bound.
mean, upper, lower = [
    onp.array(values, dtype=float)
    for values in mittendorf.mm_raw(ratios * Lpp, 180., Lpp, Bwl, T, T, Cb, Le, Lr, kyy, Fn)[:3]
]
assert onp.all(upper >= lower)

potential = {  # Hizir et al. (2019), Table 3, added resistance coefficient R / (rho g A^2 B^2 / L)
    0.5: dict(coarse=2.257, base=2.457, fine=2.421),
    1.2: dict(coarse=5.906, base=6.038, fine=5.937),
}
rows = []
for ratio, values in potential.items():
    result = mittendorf.mm_raw([ratio * Lpp], 180., Lpp, Bwl, T, T, Cb, Le, Lr, kyy, Fn)
    mean_i, upper_i, lower_i = float(result[0][0]), float(result[1][0]), float(result[2][0])
    interval = f"PRECAL; interval of the method [{lower_i:.2f}, {upper_i:.2f}]"
    rows.append((f"lambda/L = {ratio}, base mesh", values["base"], mean_i, interval))
    rows.append((f"lambda/L = {ratio}, fine mesh", values["fine"], mean_i, interval))
comparison_table("KVLCC2, head seas, Fn = 0.142, against a potential code", rows, tolerance=20.)
print("  PRECAL itself moves by 8 % between its coarse and base meshes at lambda/L = 0.5.")

experiment = read_table("kvlcc2_caw_experiment.csv")


#%% 2, DSYHS TABLE

rho, g, Lwl, swh, volume, gyration = 1025., 9.81, 10., 1., 3., 2.5
term = 100. * volume ** (1. / 3.) / Lwl * gyration / Lwl


def Raw(T1_prime, mu, Fr):
    T1 = T1_prime / onp.sqrt(g / Lwl)
    return float(dsyhs.compute_Raw_dsyhs(rho=rho, g=g, Lwl=Lwl, swh=swh, volume=volume, kyy=gyration, T1=T1, mu=mu, Fr=Fr))


FR_NODES = [0.15, 0.25, 0.35, 0.45, 0.60]
worst = 0.
zeros = []
for row in dsyhs._wave_coefs:
    for k, Fr in enumerate(FR_NODES):
        a, b = row[2 + 2 * k], row[3 + 2 * k]
        expected = a * term ** b * rho * g * Lwl * swh ** 2 / 100.
        if a == 0.:
            zeros.append((row[0], row[1], Fr))
            continue
        worst = max(worst, abs(Raw(row[0], row[1], Fr) / expected - 1.))
print(f"\nDSYHS added resistance at the {dsyhs._wave_coefs.shape[0] * 5} nodes of the table: largest error {100. * worst:.2e} %")
print(f"  {len(zeros)} entries of the table are zero (a = b = 0): {[tuple(float(x) for x in z) for z in zeros]}")

dense = [
    (Raw(x, mu, Fr), x, mu, Fr)
    for x in onp.linspace(2., 6., 41) for mu in [100., 115., 125., 135., 145.] for Fr in onp.linspace(0.15, 0.6, 46)
]
lowest = min(dense)
negative = sum(value[0] < 0. for value in dense)
print(
    f"  Between the nodes, {negative} of {len(dense)} points are negative. Lowest: {lowest[0]:.1f} N "
    f"at T1' = {lowest[1]:.2f}, mu = {lowest[2]:g}, Fn = {lowest[3]:.2f} (the interpolation meets the zeros of the table)."
)

published = read_table("dsyhs_added_resistance_points.csv")
if published is not None:
    rows = []
    for i, (x, mu, Fr) in enumerate(zip(published["T1_prime"], published["mu"], published["Fn"])):
        point = f"T1' = {x:g}, mu = {mu:g}, Fn = {Fr:g}"
        for name in ("a", "b"):
            if isinstance(published[name][i], float):  # A blank cell stays a string
                table_value = float(dsyhs._wave_interpolators[name]({"T1_prime": x, "mu": mu, "Fr": Fr}))
                rows.append((f"{name}, {point}", published[name][i], table_value, ""))
        if isinstance(published["Raw_N"][i], float):
            L_i = published["Lwl"][i]
            rows.append((f"Raw [N], {point}", published["Raw_N"][i], dsyhs.compute_Raw_dsyhs(
                rho=rho, g=g, Lwl=L_i, swh=published["swh"][i], volume=published["volume"][i], kyy=published["kyy"][i],
                T1=x / onp.sqrt(g / L_i), mu=mu, Fr=Fr,
            ), f"rho = {rho:g}"))
    comparison_table("DSYHS table against its source", rows, tolerance=0.5)


#%% PLOTS

fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))

axes[0].fill_between(ratios, lower, upper, color=BLUE, alpha=0.15, linewidth=0, label="90 % prediction interval")
axes[0].plot(ratios, mean, color=BLUE, linewidth=2, label="Mittendorf et al.")
axes[0].plot(list(potential), [v["base"] for v in potential.values()], "o", color="black", markersize=5, label="PRECAL, base mesh (Hizir et al.)")
axes[0].plot(list(potential), [v["fine"] for v in potential.values()], "s", color=GREY, markersize=5, label="PRECAL, fine mesh (Hizir et al.)")
if experiment is not None:
    axes[0].plot(experiment["lambda_L"], experiment["Caw"], "D", color=AQUA, markersize=5, label="Experiments")
style(axes[0], title="KVLCC2, head seas: R / (rho g A^2 B^2 / L)", xlabel="Wavelength / Lpp")
axes[0].legend(frameon=False, fontsize=8)

T_prime = onp.linspace(2., 6., 81)
T_nodes = [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0]
for Fr, color, k in zip([0.25, 0.35, 0.45], [BLUE, ORANGE, AQUA], [1, 2, 3]):
    values = onp.array([Raw(x, 125., Fr) for x in T_prime])
    axes[1].plot(T_prime, values / 1e3, color=color, linewidth=2, label=f"Fn = {Fr}")
    nodes = dsyhs._wave_coefs[dsyhs._wave_coefs[:, 1] == 125.]
    axes[1].plot(
        T_nodes,
        [row[2 + 2 * k] * term ** row[3 + 2 * k] * rho * g * Lwl * swh ** 2 / 100. / 1e3 for row in nodes],
        "o", color=color, markersize=4,
    )
style(axes[1], title="compute_Raw_dsyhs, mu = 125 deg: curve and table nodes [kN]", xlabel="T1 sqrt(g / Lwl)")
axes[1].legend(frameon=False, fontsize=8)

fig.tight_layout()
finish(fig, "added_resistance_waves")
