# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 10:32:06 2026

@author: jrich
"""
    
import os
import archibald.numpy as np
import archibald.dynamics.hydro.dsyhs as dsyhs
import archibald.dynamics.hydro.holtrop as holtrop

from archibald.optimization import Opti
from archibald.geometry import Hull, Sailboat
from archibald.performance import OperatingPoint


stw = 10.
# T = 1.3
# T = 0.5
# T = -4.483367256512539
T = -0.1
# T = -0.
# T = 0.1
# T = 0.2
heel = 0.
trim = 0.
leeway = 0.

opti = Opti()

stw = opti.variable(init_guess=stw, lower_bound=0.)
T = opti.variable(init_guess=T)
# heel = opti.variable(init_guess=heel)
# trim = opti.variable(init_guess=trim)
# leeway = opti.parameter(leeway)

op_point = OperatingPoint(
    stw=stw,
    dz=-T,
    heel=heel,
    trim=trim,
    leeway=leeway,
)

hull = Hull(
    mesh=os.path.abspath(r"..\..\examples\02 - Geometry\data\molenez2_data\hull.stl")
)

sailboat = Sailboat(
    displacement=130e3,
    cog=[12.2, 0., 1.],
    hulls=[hull],
)


custom_process = {
    "Rf": dsyhs.compute_Rf_dsyhs,
    # "Rw": dsyhs.compute_Rw_dsyhs,
    "Rtr": dsyhs.compute_Rtr_dsyhs,
    # "Rf": holtrop.compute_Rf_holtrop,
    # "Rw": holtrop.compute_Rw_holtrop,
    # "Rb": holtrop.compute_Rb_holtrop,
    # "Rtr": holtrop.compute_Rtr_holtrop,
    # "Ra": holtrop.compute_Ra_holtrop,
}

# hull.compute_resistance(
#     op_point,
#     method=custom_process,
#     **{'Csternchoice': 1, 'Bulbchoice': 0}
# )

Ftot, Mtot = sailboat.compute_torsor(
    op_point,
    # method="dsyhs",
    # method="holtrop",
    method=custom_process,
    **{'Csternchoice': 1, 'Bulbchoice': 0}
)

Fprop = 13e3
Ffoil = 10000./(1+np.abs(T)**3) * stw**2. * (1+trim/10.) * 2.
# Ffoil = 1e4 / (1+np.abs(T-2))
Fhull = stw**2. * 1e2 * np.softplus(T)
Fdrag = Ffoil/100. + stw**2.

opti.subject_to((Ftot[0] + Fprop - Fdrag) == 0)
# opti.subject_to((Fprop - Fdrag - Fhull) == 0)
opti.subject_to((Ftot[2] + Ffoil) == 0)
# opti.subject_to((Ftot[0] + Fprop - Fdrag) == 0)
# opti.subject_to((Ftot[2] + Ffoil) == 0)
# opti.subject_to((np.softplus(T, beta=1e3)*1e6 + sailboat.forces["Fw"][:,2] + Ffoil) == 0)
# opti.subject_to(Mtot[0] == 0)
# opti.subject_to(Mtot[1] == 0)

opti.minimize(0.)
# opti.minimize(T)

sol = opti.solve()

forces = sol(sailboat.forces)
moments = sol(sailboat.moments)

print(sol((T, heel, trim)))

final_op = sol(op_point)

print("\n--- EQUILIBRIUM REACHED ---")
print(f"Fhull:    {forces['Fh'][0]:.0f} kN")
print(f"STW:        {final_op.stw:.1f} kts")
print(f"dz:         {final_op.dz:.2f} m")
print(f"Heel:       {final_op.heel:.3f} deg")
print(f"Trim:       {final_op.trim:.3f} deg")
print(f"Volume:       {sol(sailboat.hulls[0].hydrostatics_data['volume']):.1f} m3")
print(f"Foil%:       {sol(-Ffoil/forces['Fw'][2])*100.:.1f} %")
print(f"Final Ftot: {sol(Ftot)}")
print(f"Final Mtot: {sol(Mtot)}")

# hull.draw(final_op, set_axis_visibility=True)