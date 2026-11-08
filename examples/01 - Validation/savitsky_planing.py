# -*- coding: utf-8 -*-
"""
Savitsky (1964) and Savitsky & Brown (1976) planing method.

Case 1, the worked example of Savitsky & Brown (1976) as computed by OpenPlaning:
    27 kt class hull, beam 7.315 m, deadrise 15 deg, 827.4 kN, at the attitude OpenPlaning found.
    Reference: Castro-Feliciano et al., python-openplaning, README, MIT licence,
    https://github.com/elcf/python-openplaning. This is a cross-code check of the same equations,
    not the values printed in the paper. Those go in data/savitsky_brown_1976_example.csv.
    OpenPlaning includes the gravity term of the lift and the speed dependent centre of pressure;
    ``savitsky.py`` keeps only the dynamic limit by design, because the buoyancy comes from the
    hydrostatics module. That difference is shown rather than hidden: the complete formulas
    are written here, in ``complete_lift`` and ``complete_lcp``, using the same equations as Savitsky (1964).
    OpenPlaning also evaluates the friction force with the free stream speed U^2, where Savitsky and
    this module use the mean bottom velocity V_m^2, which is 1.6 % lower here. Both are printed.

Case 2, equilibrium of a prismatic model:
    Beam 16 in, deadrise 20 deg, 80 lb, from ``archibald/dynamics/hydro/validation_planing.py``, where
    it is labelled "Shoemaker Model 29 (Savitsky & Brown / Alourdas)". Its provenance could not be
    confirmed in a primary source, so it is an indication only.

Case 3, measured prismatic hulls of Chambliss & Boyd, NACA TN 2876: blank, data/chambliss_boyd_tn2876.csv.

All trims in degrees. Nothing here adds a hull mesh: the wetted lengths come from the method itself.
"""
import matplotlib.pyplot as plt
import numpy as onp

import archibald as arb
import archibald.numpy as np
from archibald.dynamics.hydro import savitsky as sv
from archibald.dynamics.hydro import strip_theory as st
from archibald.toolbox import units as u

from validation_common import BLUE, GREY, ORANGE, AQUA, comparison_table, finish, read_table, style


#%% COMPLETE LIFT AND CENTRE OF PRESSURE (Savitsky 1964)

def complete_lift(stw, trim_true, lam, b, deadrise, rho, g):
    """Total lift [N], with the gravity term of C_L0 that ``savitsky.py`` leaves to the hydrostatics."""
    V = stw * u.kt
    Cv = V / np.sqrt(g * b)
    CL0 = trim_true ** 1.1 * (0.012 * np.sqrt(lam) + 0.0055 * lam ** 2.5 / Cv ** 2)
    CLb = CL0 - 0.0065 * deadrise * CL0 ** 0.6
    return CLb * 0.5 * rho * V ** 2 * b ** 2, CLb


def complete_lcp(stw, lam, b, g):
    """Centre of pressure from the transom [m], with its speed dependence."""
    Cv = stw * u.kt / np.sqrt(g * b)
    return lam * b * (0.75 - 1. / (5.21 * Cv ** 2 / lam ** 2 + 2.39))


def trim_giving_tau(tau_true):
    """Input trim such that ``savitsky.tau`` returns ``tau_true``: it inverts the soft floor of the module."""
    return onp.log(onp.exp(tau_true) - 1.)


#%% CASE 1, S&B 1976 EXAMPLE THROUGH OPENPLANING

open_planing = dict(
    stw=13.07 / u.kt, weight=827400., b=7.315, loa=24.38, deadrise=15., swh=1.402,
    rho=1025.87, nu=1.19e-6, g=9.8066,
    tau=2.878945, L_K=28.69256, L_C=17.67617, lam=3.199428, x_s=11.0164, alpha=18.36643,
    lcp=11.05546, area=175.5762, V_m=12.96912, dCf=0.2485087e-3, F_friction=31893.98,
    Fx_pressure=39245.86, Fz=780400.3, R_AW=29716.1,
)
o = open_planing


def case_1(trim_input, label):
    kw = dict(trim=trim_input, Lwl=o["L_K"], Bwl=o["b"], deadrise=o["deadrise"])
    tau = float(sv.tau(trim_input))
    lam = float(sv.lambda_(**kw))
    L_complete, CLb = complete_lift(o["stw"], tau, lam, o["b"], o["deadrise"], o["rho"], o["g"])
    Df = sv.D_f(stw=o["stw"], rho=o["rho"], nu=o["nu"], dCf=o["dCf"], **kw)
    V_m = o["stw"] * u.kt * float(sv.V_m_ratio(trim=trim_input, lam=lam, deadrise=o["deadrise"]))
    # OpenPlaning multiplies Cf by U^2, Savitsky and this module by V_m^2. Same force otherwise.
    Df_free_stream = float(Df) * (o["stw"] * u.kt / V_m) ** 2
    # The regression of the added resistance reads the trim directly, it does not pass through tau
    R_AW = sv.compute_Raw_savitsky(
        stw=o["stw"], rho=o["rho"], g=o["g"], trim=o["tau"], Bwl=o["b"], deadrise=o["deadrise"],
        swh=o["swh"], Loa=o["loa"], displacement=o["weight"],
    )
    return comparison_table(label, [
        ("trim seen by the method", o["tau"], tau, "deg"),
        ("chine length L_C", o["L_C"], sv.L_C(**kw), "m"),
        ("lambda", o["lam"], lam, ""),
        ("x_s", o["x_s"], sv.x_s(**kw), "m"),
        ("spray root angle", o["alpha"], onp.degrees(onp.arctan(o["b"] / (2. * float(sv.x_s(**kw))))), "deg"),
        ("wetted bottom area", o["area"], sv.A_wetted(**kw), "m2"),
        ("V_m", o["V_m"], V_m, "m/s"),
        ("friction Df cos(tau), U^2", o["F_friction"], Df_free_stream * onp.cos(onp.radians(tau)), "N, OpenPlaning's velocity"),
        ("friction Df cos(tau), V_m^2", o["F_friction"], float(Df) * onp.cos(onp.radians(tau)), "N, Savitsky's velocity, as shipped"),
        ("lift, complete", o["Fz"], L_complete, "N"),
        ("lift, module (dynamic)", o["Fz"], sv.L_dynamic(stw=o["stw"], rho=o["rho"], **kw), "N, dynamic only by design"),
        ("pressure drag, complete", o["Fx_pressure"], float(L_complete) * onp.tan(onp.radians(tau)), "N"),
        ("LCP, complete", o["lcp"], complete_lcp(o["stw"], lam, o["b"], o["g"]), "m"),
        ("LCP, module (dynamic)", o["lcp"], sv.lcp(lam=lam, Bwl=o["b"]), "m, dynamic only by design"),
        ("added resistance R_AW", o["R_AW"], R_AW, "N"),
    ])


shipped = case_1(o["tau"], "Case 1, input trim = published trim (as shipped, through the soft floor of tau)")
exact = case_1(trim_giving_tau(o["tau"]), "Case 1, input trim corrected so that tau = published trim")

published_example = read_table("savitsky_brown_1976_example.csv")
if published_example is not None:
    trim_exact = trim_giving_tau(o["tau"])
    kw_exact = dict(trim=trim_exact, Lwl=o["L_K"], Bwl=o["b"], deadrise=o["deadrise"])
    lam_exact = float(sv.lambda_(**kw_exact))
    tau_rad = onp.radians(o["tau"])
    friction = float(sv.D_f(stw=o["stw"], rho=o["rho"], nu=o["nu"], dCf=o["dCf"], **kw_exact))
    computed = {
        "tau_deg": o["tau"],  # Inputs of OpenPlaning: a difference here means that the case itself differs
        "L_K_m": o["L_K"],
        "lambda": lam_exact,
        "L_C_m": float(sv.L_C(**kw_exact)),
        "V_m_ms": o["stw"] * u.kt * float(sv.V_m_ratio(trim=trim_exact, lam=lam_exact, deadrise=o["deadrise"])),
        "R_N": o["weight"] * onp.tan(tau_rad) + friction / onp.cos(tau_rad),  # Savitsky (1964), R = W tan(tau) + Df / cos(tau)
        "R_AW_N": float(sv.compute_Raw_savitsky(
            stw=o["stw"], rho=o["rho"], g=o["g"], trim=o["tau"], Bwl=o["b"], deadrise=o["deadrise"],
            swh=o["swh"], Loa=o["loa"], displacement=o["weight"],
        )),
    }
    quantities = list(zip(published_example["quantity"], published_example["value"]))
    comparison_table("Case 1, values printed by Savitsky & Brown (1976)", [
        (name, value, computed[name], "") for name, value in quantities if name in computed
    ])
    unknown = [name for name, _ in quantities if name not in computed]
    if unknown:
        print(f"  Not compared, unknown quantities: {unknown}. Known: {list(computed)}")


#%% CASE 2, EQUILIBRIUM OF A PRISMATIC MODEL

inch, lb = 0.0254, 4.4482216
b, deadrise, W, rho, nu, g = 16. * inch, 20., 80. * lb, 1000., 1.14e-6, 9.81
cases = [
    dict(stw=18.01, lcg=18.5 * inch, trim=4.21, R=15.12),
    dict(stw=20.92, lcg=18.3 * inch, trim=3.46, R=16.76),
]


def solve_module(case):
    """Dynamic lift only, as ``savitsky.py`` is used when no hydrostatics are added."""
    opti = arb.Opti()
    trim = opti.variable(init_guess=4., lower_bound=0.5, upper_bound=15.)
    L_K = opti.variable(init_guess=1., lower_bound=0.1, upper_bound=5.)
    kw = dict(stw=case["stw"], rho=rho, nu=nu, trim=trim, Lwl=L_K, Bwl=b, deadrise=deadrise)
    opti.subject_to([
        sv.compute_Fz_savitsky(**kw) / W == 1.,
        sv.compute_My_savitsky(lcg_tr=case["lcg"], **kw) / (W * b) == 0.,
    ])
    sol = opti.solve(verbose=False)
    trim, L_K = float(sol(trim)), float(sol(L_K))
    kw.update(trim=trim, Lwl=L_K)
    R = sv.compute_Rf_savitsky(**kw) + sv.compute_Rp_savitsky(**kw)
    return float(trim), float(R)


def solve_complete(case):
    """Complete Savitsky lift and centre of pressure, using the wetted geometry of the module."""
    opti = arb.Opti()
    trim = opti.variable(init_guess=4., lower_bound=0.5, upper_bound=15.)
    L_K = opti.variable(init_guess=1., lower_bound=0.1, upper_bound=5.)
    kw = dict(trim=trim, Lwl=L_K, Bwl=b, deadrise=deadrise)
    tau = sv.tau(trim)
    lam = sv.lambda_(**kw)
    lift, _ = complete_lift(case["stw"], tau, lam, b, deadrise, rho, g)
    arm = complete_lcp(case["stw"], lam, b, g) - case["lcg"]
    opti.subject_to([lift / W == 1., arm / b == 0.])
    sol = opti.solve(verbose=False)
    trim, L_K = float(sol(trim)), float(sol(L_K))
    tau = float(sv.tau(trim))
    Rf = sv.compute_Rf_savitsky(stw=case["stw"], rho=rho, nu=nu, trim=trim, Lwl=L_K, Bwl=b, deadrise=deadrise)
    return tau, float(W * onp.tan(onp.radians(tau)) + Rf)


def solve_strip(case):
    """2D+T strip theory. Length 1.6 m and CoG height 0.10 m are the guesses of the original script."""
    sections = st.prismatic_sections(1.6, b, deadrise, n=40)
    opti = arb.Opti()
    zeta = opti.variable(init_guess=0.03, lower_bound=-0.5, upper_bound=1.)
    trim = opti.variable(init_guess=4., lower_bound=-3., upper_bound=20.)
    out = st.strip_wrench(case["stw"], rho, g, nu, zeta, trim, sections, case["lcg"], 0.10)
    opti.subject_to([
        (out["Fz_dyn"] + out["Fz_hs"]) / W == 1.,
        (out["My_dyn"] + out["My_hs"]) / (W * 0.8) == 0.,
    ])
    sol = opti.solve(verbose=False)
    zeta, trim = float(sol(zeta)), float(sol(trim))
    out = st.strip_wrench(case["stw"], rho, g, nu, zeta, trim, sections, case["lcg"], 0.10)
    return trim, float(out["Rp_dyn"] + out["Rp_hs"] + out["Rf"])


labels = ["Module, dynamic lift", "Complete Savitsky", "Strip theory 2D+T"]
solvers = [solve_module, solve_complete, solve_strip]
results = onp.array([[solver(case) for case in cases] for solver in solvers])

print("\nCase 2, equilibrium of the 16 in model (reference data of unconfirmed provenance)")
print(f"  {'':<24}{'trim [deg]':>12}{'R [lb]':>10}")
for i, case in enumerate(cases):
    print(f"  {case['stw']} kt, reference{'':<9}{case['trim']:>10.2f}{case['R']:>10.2f}")
    for label, row in zip(labels, results[:, i]):
        print(f"    {label:<22}{row[0]:>10.2f}{row[1] / lb:>10.2f}")

measured = read_table("chambliss_boyd_tn2876.csv")
if measured is not None:
    print("\nCase 3, Chambliss & Boyd, NACA TN 2876, at the measured trim and wetted length, fresh water")
    print(f"  {'deadrise':>9}{'trim':>7}{'V':>7}{'lift':>10}{'computed':>10}{'drag':>10}{'computed':>10}")
    for row in zip(*(measured[key] for key in ("deadrise", "b", "trim", "V", "lambda", "lift", "drag"))):
        beta_i, b_i, trim_i, V_i, lam_i, lift_i, drag_i = row
        lift_computed = float(complete_lift(V_i / u.kt, trim_i, lam_i, b_i, beta_i, rho, g)[0])
        friction = sv.D_f(stw=V_i / u.kt, rho=rho, nu=nu, trim=trim_giving_tau(trim_i), Lwl=1., Bwl=b_i, deadrise=beta_i, lam=lam_i)
        drag_computed = lift_computed * onp.tan(onp.radians(trim_i)) + float(friction) / onp.cos(onp.radians(trim_i))
        print(f"  {beta_i:>9g}{trim_i:>7g}{V_i:>7g}{lift_i:>10.1f}{lift_computed:>10.1f}{drag_i:>10.1f}{drag_computed:>10.1f}")


#%% PLOTS

names = [
    "L_C", "lambda", "x_s", "spray angle", "wetted area", "V_m", "friction (U^2)",
    "lift (complete)", "pressure drag (complete)", "LCP (complete)", "R_AW",
]
index = [1, 2, 3, 4, 5, 6, 7, 9, 11, 12, 14]

fig, axes = plt.subplots(1, 3, figsize=(13, 4.4), gridspec_kw=dict(width_ratios=[1.5, 1, 1]))

y = onp.arange(len(names))[::-1]
axes[0].barh(y + 0.19, [shipped[i] for i in index], height=0.36, color=ORANGE, label="As shipped")
axes[0].barh(y - 0.19, [exact[i] for i in index], height=0.36, color=BLUE, label="tau floor removed")
axes[0].axvline(0., color=GREY, linewidth=1)
axes[0].set_yticks(y, names)
style(axes[0], title="S&B 1976 example: difference to OpenPlaning", xlabel="Difference [%]")
axes[0].grid(axis="y", visible=False)
axes[0].legend(frameon=False, title=f"Floor removed: largest {max(abs(exact[i]) for i in index):.2f} %", title_fontsize=8)

x = onp.arange(len(cases))
width = 0.2
reference = [[c["trim"] for c in cases], [c["R"] for c in cases]]
titles = ["16 in model: trim [deg]", "16 in model: resistance [lb]"]
for k, ax in enumerate(axes[1:]):
    ax.bar(x - 1.5 * width, reference[k], width, color="black", label="Reference")
    for i, (label, color) in enumerate(zip(labels, [ORANGE, BLUE, AQUA])):
        values = results[i, :, k] / (lb if k == 1 else 1.)
        ax.bar(x + (i - 0.5) * width, values, width, color=color, label=label)
    ax.set_xticks(x, [f"{c['stw']} kt" for c in cases])
    style(ax, title=titles[k])
    ax.grid(axis="x", visible=False)
axes[1].legend(frameon=False, fontsize=8)

fig.tight_layout()
finish(fig, "savitsky_planing")
