# -*- coding: utf-8 -*-
"""
Free-floating hydrostatic equilibrium of the Molenez hull.

Immersion, heel and trim are released together and solved so that the weight
torsor of the masses balances the buoyancy torsor of the hull.
"""
from archibald.optimization import Opti
from archibald.performance import OperatingPoint
from archibald.geometry.hull import Hull
from archibald.geometry.sailboat import Sailboat
from archibald.weights import MassProperties


#%% DATA

hullStl = r'data//molenez2_data//hull.stl'

Lref = 25.9  # hull length, used to scale the moments [m]

boat = Sailboat(
    hulls=[Hull(name='hull', mesh=hullStl)],
    masses=[
        MassProperties(mass=100e3, x_cg=12.3, y_cg=0., z_cg=2.2),  # lightship
        MassProperties(mass=14e3, x_cg=12.0, y_cg=0., z_cg=0.5),   # ballast
        MassProperties(mass=10e3, x_cg=15.0, y_cg=6.0, z_cg=3.0),   # deck cargo, to port
    ],
)

mass_names = ['lightship', 'ballast', 'deck cargo']


#%% EQUILIBRIUM

opti = Opti()

T = opti.variable(init_guess=1.3, lower_bound=-10)                       # immersion [m]
heel = opti.variable(init_guess=0., lower_bound=-180., upper_bound=180.)  # positive to starboard [deg]
trim = opti.variable(init_guess=0., lower_bound=-90., upper_bound=90.)  # positive bow down [deg]

op_point = OperatingPoint(dz=-T, heel=heel, trim=trim)

boat.compute_weight(op_point)
boat.compute_buoyancy(op_point)
Fw, Mw = boat.forces["Fw"], boat.moments["Mw"]
Fb, Mb = boat.forces["Fb"], boat.moments["Mb"]

Ftot = Fw + Fb
Mtot = Mw + Mb

# Three equations for three unknowns, scaled by the weight of the boat
weight = sum(boat.masses).mass * op_point.environment.gravity

opti.subject_to([
    Ftot[0, 2] == 0.,
    Mtot[0, 0] == 0.,
    Mtot[0, 1] == 0.,
])

sol = opti.solve(verbose=True)


#%% RESULTS

print(f"Immersion : {sol(T):.3f} m")
print(f"Heel      : {sol(heel):+.2f} deg (positive to starboard)")
print(f"Trim      : {sol(trim):+.2f} deg (positive bow down)")
print(f"Volume    : {sol(boat.hulls[0].hydrostatics_data['volume']):.2f} m3")

print("\nResiduals")
print(f"Fz : {sol(Ftot[0, 2]):+.2e} N")
print(f"Mx : {sol(Mtot[0, 0]):+.2e} N.m")
print(f"My : {sol(Mtot[0, 1]):+.2e} N.m")

print("\nWeight torsor at equilibrium")
print(f"{'':>12} {'Fz [kN]':>10} {'Mx [kN.m]':>11} {'My [kN.m]':>11}")
for i, name in enumerate(mass_names):
    F_i = sol(boat.forces[f"Fw_{i}"]).ravel()
    M_i = sol(boat.moments[f"Mw_{i}"]).ravel()
    print(f"{name:>12} {F_i[2] / 1e3:10.1f} {M_i[0] / 1e3:11.1f} {M_i[1] / 1e3:11.1f}")

F_w = sol(Fw).ravel()
M_w = sol(Mw).ravel()
print(f"{'total':>12} {F_w[2] / 1e3:10.1f} {M_w[0] / 1e3:11.1f} {M_w[1] / 1e3:11.1f}")


#%% DRAWING

# The hull mesh stays in boat axes during the computation: the water plane is
# brought to the hull instead. To draw, move a solved copy of the hull to the water.
boat_sol = sol(boat)
boat_sol.hulls[0].mesh.transform(sol(op_point))
boat_sol.hulls[0].mesh.draw(
    draw_plane=True,
    # point=np.array([0, 0, T0]),
    # normal=np.array([0., 1., 0.]),
    # backend="matplotlib",
    # backend="plotly",
    set_axis_visibility=True,
)


# boat_sol.draw()
