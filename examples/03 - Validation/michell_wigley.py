# -*- coding: utf-8 -*-
"""
Michell (1898) thin ship wave resistance on the Wigley hull.

The Wigley hull is the one hull that the equivalent hull of ``michell.py`` represents exactly:
    y = B/2 (1 - (2x/L)^2) (1 - (z/T)^2),  so p = q = 2 and Cp = Cx = 2/3.

Three checks:
1. Hydrostatics: a mesh of the analytical hull against its exact coefficients, because this is
   where a real hull enters the method. The same mesh is also scaled to several sizes, because
   ``ArchibaldMesh.hydrostatics`` smooths with lengths fixed in metres: a large face that touches the
   waterline from above is partly counted as wet on a small hull.
2. Numerics: ``compute_Rw_michell`` against an independent, much finer evaluation of the
   original integral, written here without the integration by parts or the separated
   amplitude used in the module:
       R = 4 rho g^2 / (pi U^2) * Integral[1, inf] (I^2 + J^2) l^2 / sqrt(l^2 - 1) dl
       I + iJ = Integral Integral dy/dx exp(l^2 k0 z + i l k0 x) dx dz,  k0 = g / U^2
3. Physics: measured total resistance, Kajitani et al. (1983) as tabulated by Ahmed, Yaakob,
   Rashid & Elbatran, Journal of Transport System Engineering 2(1), 2015, Table 5.
   CT - CF contains the viscous form effect as well as the wave resistance, so Michell is not
   expected to reach it. The form factor it would imply is printed. It should be small, positive
   and steady if the wave resistance has the right size.

Published thin ship curves would be a better reference than CT - CF:
data/wigley_wave_resistance.csv is a blank for them.
"""
import matplotlib.pyplot as plt
import numpy as onp
from scipy.integrate import trapezoid

from archibald.dynamics.hydro import michell
from archibald.geometry.hull import Hull
from archibald.geometry.mesh import ArchibaldMesh
from archibald.performance import OperatingPoint
from archibald.toolbox import units as u

from validation_common import BLUE, GREY, ORANGE, comparison_table, finish, read_table, style


#%% HULL

L, B, T = 1., 0.1, 0.0625
rho, g, nu = 1000., 9.81, 1e-6

# Kajitani et al. (1983), through Ahmed et al. (2015), Table 5. L = 1 m is the size used by Ahmed et al.
# for their computation. The model size of the experiment is not given there, it only affects CF here.
FN_MEASURED = onp.array([0.250, 0.267, 0.289, 0.316, 0.354, 0.408])
CT_MEASURED = onp.array([5.92, 5.84, 6.16, 6.32, 6.12, 6.91]) * 1e-3


def half_breadth(x, z):
    """Half breadth of the Wigley hull [m], vertical above the water like Ahmed et al. (2015)."""
    zeta = onp.minimum(z, 0.) / T
    return B / 2. * (1. - (2. * x / L) ** 2) * (1. - zeta ** 2)


def wigley_mesh(scale=1., n_x=160, n_z=60, n_top=12):
    """Closed triangular mesh of the hull with the keel at z = -T * scale, up to z = +T * scale / 2, with n_top rows above water."""
    x = L / 2. * onp.cos(onp.linspace(0., onp.pi, n_x + 1))
    z = onp.concatenate([onp.linspace(T / 2., 0., n_top + 1)[:-1], -T * onp.sin(onp.linspace(0., onp.pi / 2., n_z + 1))])
    X, Z = onp.meshgrid(x, z, indexing="ij")
    Y = half_breadth(X, Z)
    n_j = len(z)
    index = lambda i, j, side: side * (n_x + 1) * n_j + i * n_j + j
    vertices = onp.concatenate([
        onp.stack([X, Y, Z], axis=-1).reshape(-1, 3),
        onp.stack([X, -Y, Z], axis=-1).reshape(-1, 3),
    ])
    faces = []
    for i in range(n_x):
        for j in range(n_j - 1):
            a, b, c, d = index(i, j, 0), index(i + 1, j, 0), index(i + 1, j + 1, 0), index(i, j + 1, 0)
            faces += [[a, b, c], [a, c, d]]
            a, b, c, d = index(i, j, 1), index(i + 1, j, 1), index(i + 1, j + 1, 1), index(i, j + 1, 1)
            faces += [[a, c, b], [a, d, c]]
        a, b, c, d = index(i, 0, 1), index(i + 1, 0, 1), index(i + 1, 0, 0), index(i, 0, 0)
        faces += [[a, b, c], [a, c, d]]
    # Reversed so that the normals point out of the hull
    return ArchibaldMesh(scale * vertices, onp.array(faces, dtype=int)[:, ::-1])


def wetted_surface():
    """Exact wetted surface of the hull [m2], from the slopes of the surface."""
    x = onp.linspace(-L / 2., L / 2., 2001)
    z = onp.linspace(-T, 0., 801)
    X, Z = onp.meshgrid(x, z, indexing="ij")
    dy_dx = -B / 2. * 8. * X / L ** 2 * (1. - (Z / T) ** 2)
    dy_dz = -B / 2. * (1. - (2. * X / L) ** 2) * 2. * Z / T ** 2
    return 2. * trapezoid(trapezoid(onp.sqrt(1. + dy_dx ** 2 + dy_dz ** 2), z, axis=1), x)


#%% INDEPENDENT EVALUATION OF THE MICHELL INTEGRAL

def michell_integral(Fn, n_l=3001, n_x=4001, n_z=2001):
    """Wave resistance [N] from the original integral, with trapezoidal rules on dense grids."""
    U = Fn * onp.sqrt(g * L)
    k0 = g / U ** 2
    t = onp.linspace(0., 1., n_l)
    # l = 1 + w^2 removes the square root singularity at l = 1. The module truncates at
    # k0 (1 + u^2) T = 16, which costs it 0.1 to 0.4 % (1 % at Fn = 0.15). Here it is 100.
    w_max = onp.sqrt(onp.sqrt(100. / (k0 * T)) + 1.)
    w = w_max * t
    l = 1. + w ** 2
    x = onp.linspace(-L / 2., L / 2., n_x)
    z = onp.linspace(-T, 0., n_z)
    slope_x = -B / 2. * 8. * x / L ** 2
    shape_z = 1. - (z / T) ** 2
    amplitude = onp.empty(n_l)
    for i, li in enumerate(l):
        # The slope is odd in x, so only the sine part (J) survives. The cosine part (I) is zero.
        J = trapezoid(slope_x * onp.sin(li * k0 * x), x)
        depth = trapezoid(shape_z * onp.exp(li ** 2 * k0 * z), z)
        amplitude[i] = (J * depth) ** 2
    weight = l ** 2 / onp.sqrt(l + 1.) * 2.  # l^2 / sqrt(l^2 - 1) dl with dl = 2 w dw and sqrt(l - 1) = w
    return 4. * rho * g ** 2 / (onp.pi * U ** 2) * trapezoid(amplitude * weight, w)


def Rw_module(Fn, scale=1., **kwargs):
    stw = Fn * onp.sqrt(g * L * scale) / u.kt
    return float(michell.compute_Rw_michell(
        stw=stw, Lwl=L * scale, Bwl=B * scale, T=T * scale, Cp=2. / 3., Cx=2. / 3., rho=rho, g=g, **kwargs,
    ))


def dynamic_pressure(Fn):
    return 0.5 * rho * (Fn * onp.sqrt(g * L)) ** 2


#%% 1, HYDROSTATICS OF THE MESH

S = wetted_surface()


def mesh_hydrostatics(scale):
    hull = Hull("Wigley", mesh=wigley_mesh(scale))
    op_point = OperatingPoint(stw=0.3 * onp.sqrt(g * L * scale) / u.kt)
    hull.compute_hydrostatics_properties(op_point)
    h = hull.hydrostatics_data
    comparison_table(f"1. Mesh hydrostatics against the analytical hull, L = {scale * L:g} m (lengths divided by the scale)", [
        ("volume", 4. / 9. * L * B * T, float(h["volume"]) / scale ** 3, "m3"),
        ("Lwl", L, float(h["Lwl"]) / scale, "m"),
        ("Bwl", B, float(h["Bwl"]) / scale, "m"),
        ("T", T, float(h["T"]) / scale, "m"),
        ("Cb", 4. / 9., h["Cb"], ""),
        ("Cp", 2. / 3., h["Cp"], ""),
        ("Cx", 2. / 3., h["Cx"], ""),
        ("Cwp", 2. / 3., h["Cwp"], ""),
        ("wetted surface", S, float(h["Aws"]) / scale ** 2, "m2"),
    ])
    return hull, op_point


hull, op_point = mesh_hydrostatics(100.)
print(f"  Ahmed et al. (2015) list S = 0.135 m2 for the 1 m hull, the integral of its equation gives {S:.4f} m2.")


def size_errors(scale, n_top):
    """Errors [%] of the mesh hydrostatics. The mesh is the same, only its size changes."""
    h = wigley_mesh(scale, n_top=n_top).hydrostatics(onp.zeros(3), "z")
    return (
        100. * (float(h["volume"]) / scale ** 3 / (4. / 9. * L * B * T) - 1.),
        100. * (float(h["Cx"]) / (2. / 3.) - 1.),
        100. * (float(h["Lwl"]) / scale / L - 1.),
    )


SIZES = onp.array([1., 1.5, 2., 3., 5., 7., 10., 15., 20., 30., 50., 100.])
resolved = onp.array([size_errors(scale, 12) for scale in SIZES])
one_row = onp.array([size_errors(scale, 1) for scale in SIZES])
print("\n  Effect of the size of the hull, errors [%]. Same mesh, with 12 rows above water, or a single large row")
print(f"  {'L [m]':<8}{'volume 12':>11}{'volume 1':>10}{'Cx 12':>8}{'Cx 1':>8}{'Lwl 12':>8}")
for scale, row, row_1 in zip(SIZES, resolved, one_row):
    print(f"  {scale:<8g}{row[0]:>11.2f}{row_1[0]:>10.2f}{row[1]:>8.2f}{row_1[1]:>8.2f}{row[2]:>8.2f}")


#%% 2, NUMERICS OF THE INTEGRAL

Fn_check = onp.array([0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 0.80, 1.00])
reference = onp.array([michell_integral(Fn) for Fn in Fn_check])
shipped = onp.array([Rw_module(Fn) for Fn in Fn_check])
fine = onp.array([Rw_module(Fn, n_u=1601, n_x=401, n_z=201) for Fn in Fn_check])

print("\n2. compute_Rw_michell against the independent integral, Cw x 1000 on the exact surface")
print(f"  {'Fn':<7}{'integral':>11}{'shipped':>11}{'diff %':>9}{'finer grid':>12}{'diff %':>9}")
for i, Fn in enumerate(Fn_check):
    q = dynamic_pressure(Fn) * S
    print(
        f"  {Fn:<7.2f}{1e3 * reference[i] / q:>11.4f}{1e3 * shipped[i] / q:>11.4f}"
        f"{100. * (shipped[i] / reference[i] - 1.):>9.2f}{1e3 * fine[i] / q:>12.4f}"
        f"{100. * (fine[i] / reference[i] - 1.):>9.2f}"
    )

# Linear superposition: two coincident hulls have twice the sources, so four times the resistance
stw = 0.4 * onp.sqrt(g * L) / u.kt
R = michell.compute_Rw_matrix_michell(
    stw=stw, Lwl=[L, L], Bwl=[B, B], T=[T, T], Cp=[2. / 3.] * 2, Cx=[2. / 3.] * 2,
    x=[0., 0.], y=[0., 0.], rho=rho, g=g,
)
print(f"\n  Two coincident hulls, R / R_alone = {sum(map(sum, R)) / R[0][0]:.6f} (theory 4)")

# The route through a Hull and its mesh
route = float(hull.compute_resistance_components(op_point, method={"Rw": michell.compute_Rw_michell})["Rw"])
direct = Rw_module(0.3, scale=100.)
print(f"  Through Hull with the mesh, L = 100 m, Fn = 0.30: {route / 1e3:.2f} kN, from the exact coefficients: {direct / 1e3:.2f} kN")


#%% 3, MEASUREMENTS

Fn_curve = onp.linspace(0.15, 0.70, 111)
Cw_curve = onp.array([Rw_module(Fn) / (dynamic_pressure(Fn) * S) for Fn in Fn_curve])
Fn_ref = onp.linspace(0.15, 0.70, 23)
Cw_ref = onp.array([michell_integral(Fn, n_l=1501, n_x=2001, n_z=1001) / (dynamic_pressure(Fn) * S) for Fn in Fn_ref])

Re = FN_MEASURED * onp.sqrt(g * L) * L / nu
CF = 0.075 / (onp.log10(Re) - 2.) ** 2
Cw_at_measured = onp.array([Rw_module(Fn) / (dynamic_pressure(Fn) * S) for Fn in FN_MEASURED])
implied_k = (CT_MEASURED - Cw_at_measured) / CF - 1.

print("\n3. Measured total resistance, x 1000")
print(f"  {'Fn':<8}{'CT':>8}{'CF':>8}{'CT-CF':>8}{'Cw':>8}{'implied 1+k':>13}")
for i, Fn in enumerate(FN_MEASURED):
    print(
        f"  {Fn:<8.3f}{1e3 * CT_MEASURED[i]:>8.2f}{1e3 * CF[i]:>8.2f}{1e3 * (CT_MEASURED[i] - CF[i]):>8.2f}"
        f"{1e3 * Cw_at_measured[i]:>8.2f}{1. + implied_k[i]:>13.3f}"
    )

published = read_table("wigley_wave_resistance.csv")


#%% PLOTS

fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))

axes[2].axhline(0., color=GREY, linewidth=1)
axes[2].plot(SIZES, resolved[:, 0], "-o", color=BLUE, markersize=4, linewidth=2, label="12 rows above water")
axes[2].plot(SIZES, one_row[:, 0], "-o", color=ORANGE, markersize=4, linewidth=2, label="1 large row above water")
axes[2].set_xscale("log")
style(axes[2], title="Mesh volume error [%], same shape at any size", xlabel="Hull length [m]")
axes[2].legend(frameon=False)

axes[0].plot(Fn_curve, 1e3 * Cw_curve, color=ORANGE, linewidth=2, label="compute_Rw_michell")
axes[0].plot(Fn_ref, 1e3 * Cw_ref, "o", color=BLUE, markersize=4, label="Independent integral")
if published is not None:
    axes[0].plot(published["Fn"], published["Cw_x1000"], "s", color=GREY, markersize=5, label="Published")
style(axes[0], title="Wigley hull: wave resistance Cw x 1000", xlabel="Froude number")
axes[0].legend(frameon=False)

axes[1].plot(FN_MEASURED, 1e3 * (CT_MEASURED - CF), "o", color="black", markersize=5, label="Measured CT - CF")
axes[1].plot(Fn_curve, 1e3 * Cw_curve, color=ORANGE, linewidth=2, label="Michell Cw")
axes[1].set_xlim(0.2, 0.45)
axes[1].set_ylim(0., 3.5)
style(axes[1], title="Against Kajitani et al. (1983), x 1000", xlabel="Froude number")
axes[1].legend(frameon=False)

fig.tight_layout()
finish(fig, "michell_wigley")
