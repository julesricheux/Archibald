# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 17:21:43 2025

@author: jrich
"""

import archibald2.numpy as np
import archibald2.tools.units as u

from archibald2.optimization import Opti

from archibald2.performance.operating_point import OperatingPoint

from archibald2.geometry.hull import Hull
from archibald2.geometry.airfoil import Airfoil
from archibald2.geometry.wing import Sail, Fin
from archibald2.geometry.lifting_set import Appendage, Rig
from archibald2.geometry.sailboat import Sailboat
from archibald2.geometry.propeller import BSeriesPropeller

import trimesh

import casadi as ca

from archibald2.tools.geom_utils import dxf_to_le_chords
from archibald2.tools.dyn_utils import holtrop_correction_factor
from archibald2.tools.env_utils import Hs_approx, WA_approx
from archibald2.tools.math_utils import build_casadi_interpolant_from_path, GeLU
from archibald2.tools.control_utils import aoa_corr_from_awa, boom_correction_from_awa, twist_corr_from_awa, fin_init, camber_corr_from_aoa

np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)

#%% ADDITIONNAL FUNCTIONS


def fi(x):
    return np.sin(2*np.pi*x - np.pi/2)


def gi(x):
    return 1.366 * ((fi(x**0.36) + 1)/2)**1.


def compute_windage(hull,
                    op_point,
                    fCX,
                    fCY
                    ):
    
    center = hull.aerostaticData['caa'] + np.array([hull.aerostaticData['Laa']/5., 0., 0.]) # center of effort approximately at 33% of the body chord
    
    z = center[2]
    Ax = hull.aerostaticData['Ax']
    Ay = hull.aerostaticData['Ay']
    
    rho = op_point.environment.air.density
    aws = op_point._aws(z) # in m/s
    awa = op_point.awa(z) # in degrees
    leeway = op_point.leeway # in degrees
    
    k = 1/2 * rho * aws**2
    
    Fx = k * Ax * fCX(awa-leeway)
    Fy = k * Ay * fCY(awa-leeway)
    
    Faa = np.array([Fx * np.cosd(leeway) + Fy * np.sind(leeway),
                    Fx * np.sind(leeway) + Fy * np.cosd(leeway),
                    0.])
    Maa = np.cross(center, Faa)
    
    return Faa, Maa


#%% HULL PARAMETERS



AWAwindage = np.array([0., 15., 25., 35., 45., 60., 90., 120., 150., 180.])

CXwindage = np.array([-0.15, -0.31, -0.20, -0.12, -0.12, -0.32, -0.25, 0.28, 0.15, 0.20])
CYwindage = np.array([0.0, 0.38, 0.64, 0.83, 0.86, 1.01, 1.22, 1.01, 0.80, 0.0])

CXaa = ca.interpolant('LUT',
                      'bspline',
                      [AWAwindage],
                      CXwindage)
CYaa = ca.interpolant('LUT',
                      'bspline',
                      [AWAwindage],
                      CYwindage)


#%% GENERAL CONFIGURATION

nSections = 10

viz = 1
opt = 0
draw2D = 0
draw3D = 1 * (1-opt)
disp_op_point = 1.

z0 = 10. # m
tws0 = 25. # kts
twa0 = 60. # deg
# stw = tws0*np.sind(twa0)**2 / 2. + 1. # kts
# stw = tws0*(np.sind(twa0)**2 + twa0/180.) / 2. + 1. # kts
stw = 11. # kts

stwi = stw
tws0i = tws0
twa0i = twa0

WA = WA_approx(twa0i)
Hs = Hs_approx(tws0i)

# sailingVessel = False
sailingVessel = True

# setSail = False
setSail = True

setSail *= sailingVessel

# finsOut = False
finsOut = True

finsOut *= sailingVessel

# engineOn = False
engineOn = True

engineOn += (1-sailingVessel)

aoa0 = 15. # deg # to be varied depending on awa 
# aoa0 = 22. * setSail # deg # to be varied depending on awa 
depowering = 0.

# Compute apparent wind speed (aws0i)
aws0i = np.sqrt(stwi**2 + tws0i**2 + 2 * stwi * tws0i * np.cosd(twa0i))

# Compute apparent wind angle (awa0i in radians)
awa0i = np.arccosd((stwi + tws0i * np.cosd(twa0i)) / aws0i)

# if awa0i < 20.:
#     setSail = False
#     finsOut = False

if awa0i > 85. or awa0i < aoa0:
    # aero_method = 'buildup'
    aero_method = 'vlm-quick'
else:
    aero_method = 'vlm'
# aero_method = 'vlm-quick'

twistFac = 0.5
twistAngle = aoa0 * twistFac # deg # to be varied depending on awa 
# twistAngle = 3.
twistPower = 0.912

mainTwistAngle = twistAngle * twistFac
jibTwistAngle = mainTwistAngle

# balExp = 1 - 0.15*aoa/25
balExp = 1.

mainSheeting = 1.
jibSheeting = 6.

leeway = 0. # deg # positive for windward / negative for leeward
# heel = -4. * setSail # deg # positive for windward / negative for leeward
heel = - tws0i/6. * setSail # deg # positive for windward / negative for leeward
trim = 0. # deg # positive for nose down / negative for nose up

# dz = - 5.075 # m # positive for higher / negative for lower
dz = - 5.50 # m

rudderOffset = leeway * (1 - 0.33) # angle to have the rudder with approx zero lift
# rudderOffset = 0. # angle to have the rudder with approx zero lift
# rudderAngle = -1.5 * setSail # positive to luff (push the helm) / negative to bear away (pull the helm)
rudderAngle = -1. # positive to luff (push the helm) / negative to bear away (pull the helm)
finAngle = 15. * setSail

displacement = 10.5e6 # kg

lcg = 65.25
tcg = 0.0
vcg = 8.8

propellerDiam = 3.8
areaRatio = 0.8
pitchRatio = 0.839
nBlades = 4
rpm = 150.
nPropeller = 1


engineBkp = 3184. # kW
elecBkp = 900. # kW
limitBkp = elecBkp + engineBkp


#%% OPTI PARAMETERS

opti = Opti()

if opt:
    
    tws0 = opti.parameter(tws0i)
    twa0 = opti.parameter(twa0i)
    
    # objective = 'vmg'
    objective = 'bkp'
    # objective = 'sc'
    tol = 1e0
    
    # stw = opti.variable(init_guess=tws0*(np.sind(twa0)**2 + twa0/180.) / 3. + 1.)

    WA = opti.parameter(WA)
    Hs = opti.parameter(Hs)
    
    # trim = opti.variable(init_guess=0., lower_bound = -1., upper_bound = 1.)
    # heel = opti.variable(init_guess=0., lower_bound = -10., upper_bound = 10.)
    leeway = opti.variable(init_guess=0., lower_bound=-1., upper_bound=twa0)
    
    # twa0 = opti.variable(init_guess=45., lower_bound=20., upper_bound=150)
    
    if not sailingVessel or not setSail:
        pitchRatio = opti.variable(init_guess=1., lower_bound=0.5, upper_bound=1.4)
        rpm = opti.variable(init_guess=1000.*stwi/propellerDiam, lower_bound=1e-3)
    
    elif engineOn:
        stw = opti.parameter(stwi)
        pitchRatio = opti.variable(init_guess=1.4, lower_bound=0.5, upper_bound=1.4)
        # rpm = opti.variable(init_guess=250.*stwi/propellerDiam, lower_bound=1e-3)
        rpm = opti.variable(init_guess=200., lower_bound=1e-3)
        
    else:
        # stw = opti.variable(init_guess=stwi)
        stw = opti.variable(init_guess=tws0i*(np.sind(twa0i)**2 + twa0i/180.) / 3. + 1.)
        # aoa0 = opti.variable(init_guess=15., upper_bound=22.)
        objective = 'stw'
        
        
    # mainTwistAngle = opti.variable(init_guess=1., lower_bound=0.)
    # jibTwistAngle = mainTwistAngle * 1.3
    
    # dz = opti.variable(init_guess= - 5.)
    # rudderOffset = 0.
    # rudderOffset = leeway * (1 - 0.33)
    # finAngle = opti.variable(init_guess=10., lower_bound=0.)#, upper_bound=15.)
    
    if finsOut:
        # finAngle = opti.variable(init_guess=fin_init(stwi, tws0i, twa0i) * setSail, lower_bound=0., upper_bound=15.)
        finAngle = opti.variable(init_guess=10. * setSail, lower_bound=0., upper_bound=15.)
        # finAngle = 15.
    else:
        finAngle = 0.
    
    # finAngle = 15.
    rudderAngle = opti.variable(init_guess=-1.)
    # rudderAngle = - finAngle * 1/3.
    # rudderAngle = 10.
    
    Tx, Ty, Tz = (1, 1, 0)
    Rx, Ry, Rz = (0, 0, 1)
    
    method = 'constraint'
    # method = 'objective'
    
    
# op_point = OperatingPoint(
#          stw=stw, # kts
#          tws0=tws0, # kts
#          twa=twa0, # deg
#          z0=10., # m
#          a=1/20,
#          # a=1/10,
#          # a=1/7,
#          heel=heel,
#          trim=trim,
#          leeway=leeway,
#          immersion=dz,
#          )

op_point = OperatingPoint(
    stw=7.6488826126501595,
    tws0=25.0,
    twa=45.0,
    z0=10.0,
    a=0.05,
    heel=-4.166666666666667,
    trim=0.0,
    leeway=9.259349126902611,
    immersion=-5.5
)
    
cog = np.array([lcg, 
                tcg,
                vcg]) # m

aws0 = op_point.aws0
awa0 = op_point.awa0

# aoa_corr = aoa0
aoa_corr = aoa_corr_from_awa(awa0, aoa0)

a = 15.
b = 2.
# fac = 5/20
fac = 3/20

aoa = np.array([1.+ boom_correction_from_awa(awa0, a=a, b=b, fac=fac), # boom incidence correction
                1.]) * aoa_corr # deg

heel *= gi(awa0/180.)

#%% HULL PARAMETERS
# hullStl = 'C:/Users/jrich/NEOLINE DEVELOPPEMENT/NeoDev - ND-PROJETS - NAVIRES-LIGNES - ND-PROJETS - NAVIRES-LIGNES/NEOLINER PCTC 169 (PCTC169)/03_APS/carenes/airbus.stl'
hullStl = 'C:/Users/jrich/Documents/archibald-main/Private/n136_data/n136.stl' # path to a STL mesh of the hull
addStl = 'C:/Users/jrich/Documents/archibald-main/Private/n136_data/superstructures.stl' # path to a STL mesh of the hull
hull = Hull('n136pctc_hull', displacement, cog, hullStl)


#%% PROPELLER PARAMETERS

propeller = BSeriesPropeller(
    Z=nBlades,
    Ae_Ao=areaRatio,
    P_D=pitchRatio,
    D=propellerDiam
    )


#%% SAILS PARAMETERS

# Put this in vanilla or Private

# Define rig initial geometry
nSections = 12

mainLe = np.array([[-1.4,  0. ,  4.4],
                   [-1.4,  0. ,  7.7],
                   [-1.4,  0. , 14. ],
                   [-1.4,  0. , 20.2],
                   [-1.4,  0. , 26.5],
                   [-1.4,  0. , 32.7],
                   [-1.4,  0. , 39. ],
                   [-1.4,  0. , 45.2],
                   [-1.4,  0. , 51.5],
                   [-1.4,  0. , 57.7],
                   [-1.4,  0. , 61.3],
                   [-1.4,  0. , 71.3]])

mainChords = np.array([20.4,
                       20.3,
                       19.8,
                       19.1,
                       18.3,
                       17.4,
                       16.3,
                       14.9,
                       13.4,
                       11.7,
                       10.6,
                       0.7])

mastDiam = 2.
mainLe[:, 0] += mastDiam
mainChords += mastDiam

# Jibsail rig 2
jibTackPt = np.array([19.6, 0.0,  4.5])
jibHeadPt = np.array([ 3.2, 0.0, 60.0])

jibFoot = 16.4
jibHead = 0.1

jibLe = np.linspace(jibTackPt, jibHeadPt, nSections)
jibChords = np.linspace(jibFoot, jibHead, nSections)

if not setSail:
    mainChords = np.ones(nSections) * mastDiam
    jibChords = np.ones(nSections) * 0.5


balAxis = np.array([[-22.2, 0, 2.6],
                    [-22.2, 0, 2.6],
                    [-3.4/2, 0, 1.6],
                    [3.4/2, 0, 1.6],
                    [19.6, 0, 2.8],
                    [19.6, 0, 2.8],
                    ])

balNormal = np.array([[1., 0., 0.] for i in range(len(balAxis))])

balRadius = np.array([0., 1.4, 3.3, 3.3, 1.1, 0.])/2

balAR = 1.5

# zSpardeck = 25.2

# nSolidsail = 3

# solidsailCoords = np.array([[ 36.,          0., zSpardeck],
#                             [ 36.+48.,      0., zSpardeck],
#                             [ 36.+48.+48.,  0., zSpardeck]])

zSpardeck = 20.9

nSolidsail = 2

solidsailCoords = np.array([[ 43.2, 0.0, zSpardeck],
                            [ 98.4, 0.0, zSpardeck]])


balFac = GeLU(awa0 - aoa) - leeway
balSheeting = balFac

mainTwist = - np.linspace(0,1,nSections)**twistPower * mainTwistAngle
jibTwist = - np.linspace(0,1,nSections)**twistPower * jibTwistAngle

mainAirfoilVec = []
jibAirfoilVec = []

# mainAirfoil = asb.Airfoil('e376').repanel().normalize()
# jibAirfoil = asb.Airfoil('e376').repanel().normalize()

if setSail:
    mainAF = 'naca5401'
    jibAF = 'naca7301'
    # mainAF = 'e376'
    # jibAF = 'e376'
else:
    mainAF = 'naca0032'
    jibAF = 'naca0032'

mainAirfoil = Airfoil(mainAF).repanel().normalize()
jibAirfoil = Airfoil(jibAF).repanel().normalize()

# camber_corr = camber_corr_from_aoa(aoa_corr)
camber_corr = 1.
twist_corr = twist_corr_from_awa(awa0)

for i in range(nSections):
    
    mainTempAirfoil = Airfoil(mainAF).repanel().normalize()
    jibTempAirfoil = Airfoil(jibAF).repanel().normalize()
    
    # mainTempAirfoil.coordinates[:,1] = mainAirfoil.coordinates[:,1] * ((i+0.001)/(nSections))**(1/5) * camber_corr
    # jibTempAirfoil.coordinates[:,1] = jibAirfoil.coordinates[:,1] * ((i+0.001)/(nSections))**(1/5) * camber_corr
    mainTempAirfoil.coordinates *= np.ones((mainTempAirfoil.coordinates.shape[0], 1)) @ \
                                    np.array([[1., camber_corr]]) * ((i+0.001)/(nSections))**(1/5)
    jibTempAirfoil.coordinates *= np.ones((jibTempAirfoil.coordinates.shape[0], 1)) @ \
                                  np.array([[1., camber_corr]]) * ((i+0.001)/(nSections))**(1/5)
                                       
    mainAirfoilVec.append(mainTempAirfoil)
    jibAirfoilVec.append(jibTempAirfoil)
    
mainAirfoilVec.reverse()
jibAirfoilVec.reverse()

solidsailCoords[:,0] *= -1.
mainLe[:,0] *= -1.
jibLe[:,0] *= -1.

# mainChords *= fac

n136rig = Rig(
    name="n136pctc_rig",
    xyz_ref=[0, 0, 0],  # CG location
    wings=[
        Sail(
            name=f'mainsail {i}',
            xyz_le = mainLe + solidsailCoords[i,:],  # Coordinates of the XSec's leading edge, relative to the wing's leading edge.
            chords = mainChords,
            twists = mainTwist * twist_corr,
            airfoils = mainAirfoilVec,  # Airfoils are blended between a given XSec and the next one
            deflection_axis = np.array([0.,0.,1.]),
            deflection_center = solidsailCoords[i,:],
        ) for i in range(nSolidsail) ] +\
        [Sail(
            name=f'jibsail {i}',
            xyz_le = jibLe + solidsailCoords[i,:],  # Coordinates of the XSec's leading edge, relative to the wing's leading edge.
            chords = jibChords,
            twists = jibTwist * twist_corr,
            airfoils = jibAirfoilVec,  # Airfoils are blended between a given XSec and the next one
            deflection_axis = np.array([0.,0.,1.]),
            deflection_center = solidsailCoords[i,:],
        )  for i in range(nSolidsail) 
    ],
)


# n136rig.draw(thin_wings=True)

# solidsailPath = 'C:/Users/jrich/Documents/archibald-main/Private/n169airbus_data/solidsail_cda_polar.nc'

# Fxsail = build_casadi_interpolant_from_path(solidsailPath, 'solidsail_X')
# Fysail = build_casadi_interpolant_from_path(solidsailPath, 'solidsail_Y')
# Mxsail = build_casadi_interpolant_from_path(solidsailPath, 'solidsail_K')
# Mzsail = build_casadi_interpolant_from_path(solidsailPath, 'solidsail_N')
    

#%% APPENDAGE PARAMETERS

finDx = 10.

rudderDxf = 'C:/Users/jrich/Documents/archibald-main/Private/n136_data/rudder.dxf'
finSbdDxf = 'C:/Users/jrich/Documents/archibald-main/Private/n136_data/fin_sbd.dxf'
finPrtDxf = 'C:/Users/jrich/Documents/archibald-main/Private/n136_data/fin_prt.dxf'

rudderLe, rudderChords = dxf_to_le_chords(rudderDxf, nSections, method='bezier')
finSbdLe, finSbdChords = dxf_to_le_chords(finSbdDxf, nSections, method='bezier')
finPrtLe, finPrtChords = dxf_to_le_chords(finPrtDxf, nSections, method='bezier')

chordsThreshold = 1e-3
rudderChords[rudderChords < chordsThreshold] = chordsThreshold

# finSections = 'D:/Documents/archibald-main/archibald/data/airfoils/thick_sections/asym_ah85l120_mod.dat'
# finSections = 'e854.dat'
finSections = 'e169.dat'
rudderSections = 'e169.dat'

finAirfoilVec = []

finRootAirfoil = Airfoil('naca0030').repanel().normalize()
finTipAirfoil = Airfoil('naca0015').repanel().normalize()

for i in range(nSections):
    
    tempAirfoil = finTipAirfoil.blend_with_another_airfoil(finRootAirfoil,
                                                           blend_fraction = i/(nSections-1))
    finAirfoilVec.append(tempAirfoil.repanel())

rudderAirfoil = Airfoil('e862').repanel()

finSbdLe[:,0] += finDx
finPrtLe[:,0] += finDx
finSbdLe[:,0] *= -1.
finPrtLe[:,0] *= -1.
rudderLe[:,0] *= -1.

n136app = Appendage(
    name="n136_app",
    xyz_ref=[0, 0, 0],  # CG location
    wings=[
        Fin(
            name='rudder',
            xyz_le = rudderLe,  # Coordinates of the XSec's leading edge, relative to the wing's leading edge.
            chords = rudderChords,
            airfoils = [rudderAirfoil]*nSections  # Airfoils are blended between a given XSec and the next one
        ), ] + [
        Fin(
            name='daggerboard_starboard',
            xyz_le = finSbdLe,  # Coordinates of the XSec's leading edge, relative to the wing's leading edge.
            chords = finSbdChords,
            airfoils = finAirfoilVec  # Airfoils are blended between a given XSec and the next one
        ),
        Fin(
            name='daggerboard_portside',
            xyz_le = finPrtLe,  # Coordinates of the XSec's leading edge, relative to the wing's leading edge.
            chords = finPrtChords,
            airfoils = finAirfoilVec  # Airfoils are blended between a given XSec and the next one
        ),
    ] * finsOut
)

#%% SAILBOAT PARAMETERS

hullMesh = trimesh.load(hullStl)
addMesh = trimesh.load(addStl)

n136 = Sailboat(
    name="n136",
    displacement = displacement,
    cog = cog,
    xyz_ref=[0, 0, 0],  # CG location
    app = n136app,
    rig = n136rig,
    mesh = [addMesh],
    hull = hull,
    propeller=propeller,
    )

n136.transform(heel=heel,
               trim=trim,
               leeway=leeway,
               dz=dz)

n136.app.wings[0].local_rotation(deflection=rudderOffset + rudderAngle)

for i in range(finsOut):
    n136.app.wings[1].local_rotation(deflection=-finAngle)
    n136.app.wings[2].local_rotation(deflection=-finAngle)


for i in range(nSolidsail):
    # MAINSAIL SETTING
    main_i = n136.rig.wings[i]
    main_i.global_rotation(angle=balSheeting[i],
                            axis=main_i.deflection_axis,
                            center=main_i.deflection_center,
                            offset_deflection=True)
    main_i.set_sheeting(-mainSheeting * camber_corr)
    
    # JIBSAIL SETTING
    jib_i = n136.rig.wings[nSolidsail+i]
    jib_i.global_rotation(angle=balSheeting[i],
                          axis=jib_i.deflection_axis,
                          center=jib_i.deflection_center,
                          offset_deflection=True)
    jib_i.set_sheeting(-jibSheeting * camber_corr)

# n136.draw()

#%% HULL COMPUTATION

hull.compute_statics(op_point=op_point)

Fb, Mb = hull.compute_buoyancy(
    op_point,
    recompute_statics=False
    )

Fhull, Mhull = hull.compute_resistance_force(
    op_point,
    method='holtrop'
    # method='dsyhs'
    )
                                             
Fr = stw * u.knot / np.sqrt(hull.hydrostaticData['Lwl'] * op_point.environment.gravity)

holtropCorr = holtrop_correction_factor(Fr)

Fhull *= holtropCorr
Mhull *= holtropCorr
                                     
Faa, Maa = compute_windage(hull, op_point, CXaa, CYaa)
# Faa, Maa = hull.compute_wind_loads(
#     op_point,
#     recompute_statics=False,
#     method='richeux',
#     )

Fdrift, Mdrift = hull.compute_lateral_force(
    op_point,
    method='dice',
    )

Fwave, Mwave = np.zeros(3), np.zeros(3)

#%% PROPELLER COMPUTATION

etaS = 0.98

if engineOn:
    
    Fprop, Mprop, prop = n136.compute_propeller(
        rpm,
        op_point, 
        recompute_statics=False,
        full_output=True,
        shaft_efficiency=etaS
        )
    
    J = prop['J']
    Kt = prop['Kt']
    Kq = prop['Kq']
    
    etaH = prop['etaH']
    eta0 = prop['eta0']
    etaR = prop['etaR']
    w = prop['w']
    t = prop['t']
    P = prop['P']
    
else:
    P = 0.
    
    J = 0.
    Kt = 0.
    Kq = 0.
    
    etaH = 1.
    eta0 = float('nan')
    etaR = 1.
    w = 0.
    t = 0.
    P = 0.
    
    Fprop = np.array([0., 0., 0.])
    Mprop = np.array([0., 0., 0.])

#%% SAIL COMPUTATION


# AR = n136rig.wings[0].aspect_ratio()
# span = n136rig.wings[0].span()

cellAR = 2

nSpanwise = 1
nChordwise = 3

zHead = 92.

# if setSail:
    
Fsail, Msail, aero = n136.compute_sails(
    op_point,
    nSpanwise=1,
    nChordwise=5,
    full_output=True,
    run_symmetric=True,
    Zsym=0.,
    )


# %% APPENDAGE COMPUTATION

# AR = n136app.wings[0].aspect_ratio()
# span = n136app.wings[0].span()

cellAR = 2

if finsOut:
    Fapp, Mapp, hydro = n136.compute_appendages(
        op_point,
        nSpanwise=1,
        nChordwise=3,
        full_output=True
        )
    
else:
    Fapp, Mapp, hydro = np.zeros(3), np.zeros(3), {}

#%% SAILBOAT COMPUTATION

Fw, Mw = n136.compute_weight(op_point)


#%% EQUILIBRIUM

totalDrag = -(Fhull[0] + Fdrift[0] + Faa[0] + Fapp[0] + Fwave[0])
windThrust = Fsail[0]
    
Ftot = Fsail + Fapp + Fw + Fb + Faa + Fhull + Fdrift + Fwave + Fprop
Mtot = Msail + Mapp + Mw + Mb + Maa + Mhull + Mdrift + Mwave + Mprop
    
# bkp = (totalDrag - windThrust) *stwi*u.knot/(eta0 * etaR * etaS * etaH) * needEngine
bkp = P / etaS
    
sailsContribution = windThrust / totalDrag * 100.


cons = []

if opt:
    
    if method == 'constraint':
        
        if Tx:
            cons.append(Ftot[0] == 0.)
        if Ty:
            cons.append(Ftot[1] == 0.)
        if Tz:
            cons.append(Ftot[2] == 0.)
        if Rx:
            cons.append(Mtot[0] == 0.)
        if Ry:
            cons.append(Mtot[1] == 0.)
        if Rz:
            cons.append(Mtot[2] == 0.)
        if engineOn:
            # cons.append(bkp/1e3 < limitBkp)
            cons.append(eta0 > 1e-3)
            # cons.append(eta0 < 1.0)
            cons.append(J < 1.5)
            cons.append(Kq >= 0.0)
        
        if objective == 'stw':
            opti.maximize(stw)
        elif objective == 'vmg':
            opti.maximize(stw * np.cosd(twa0))
        elif objective == 'bkp':
            opti.minimize(bkp)
        elif objective == 'sc':
            opti.maximize(sailsContribution)
    
    elif method == 'objective':
    
        obj = 0.
        
        if Tx:
            obj += (Ftot[0])**2
        if Ty:
            obj += (Ftot[1])**2
        if Tz:
            obj += (Ftot[2])**2
        if Rx:
            obj += (Mtot[0])**2
        if Ry:
            obj += (Mtot[1])**2
        if Rz:
            obj += (Mtot[2])**2
    
        opti.minimize(obj)

opti.subject_to(cons)

sol = opti.solve(max_iter=500,
                 # max_runtime=60.,
                 )

# print(Fsail[0])

#%% Vizzz

g = op_point.environment.gravity

vmg = stw*np.cosd(twa0)


totalDrag = -(Fhull[0] + Fdrift[0] + Faa[0] + Fapp[0] + Fwave[0])
totalThrust = Fsail[0] + Fprop[0]

totalDrift = Fsail[1] + Faa[1]
totalAntiDrift = -(Fapp[1] + Fdrift[1] + Fprop[1])
    
if viz:
    geometryColor = 'grey'
    
    print()
    print('=== CONDITIONS ===')
    print('TWS10 =', np.round(sol(tws0),2), 'kts / TWA10 =', np.round(sol(twa0),2), '° / STW =', np.round(sol(stw),2), 'kts')
    print('AWS10 =', np.round(sol(aws0),2), 'kts / AWA10 =', np.round(sol(awa0),2), '° / VMG =', np.round(sol(vmg),2), 'kts')
    print()
    print('=== STATE ===')
    print('HEEL =', np.round(sol(heel),2), '° / TRIM =', np.round(sol(trim),2), '° / LEEWAY =', np.round(sol(leeway),2), '°')
    print('BKP =', np.round(sol(bkp)/1e3,2), 'kW / Propeller efficiency =', np.round(sol(eta0),2), '/ Mechanical efficiency =', np.round(sol(eta0*etaS*etaH*etaR),2))
    print('Sails contribution =', np.round(sol(sailsContribution)))
    print()
    print('=== CONTROLS ===')
    print('Fins angle =', np.round(sol(finAngle),2), '° / Rudder angle =', np.round(sol(rudderAngle),2), '°')
    print('Propeller speed =', np.round(sol(rpm),2), 'RPM / Propeller pitch =', np.round(sol(pitchRatio),2))
    print()
    print('=== SAILS ===')
    print(sol(Fsail)/1e3, 'kN')
    print()
    print('=== APPENDAGE ===')
    print(sol(Fapp)/1e3, 'kN')
    print()
    print('===PROPELLER ===')
    print(sol(Fprop)/1e3, 'kN')
    print()
    print('=== WINDAGE ===')
    print(sol(Faa)/1e3, 'kN')
    print()
    print('=== HULL ===')
    print('Calm water:', sol(Fhull)/1e3, 'kN')
    print('Drift:', sol(Fdrift)/1e3, 'kN')
    print('Added wave:', sol(Fwave)/1e3, 'kN')    
    print()
    print('=== TOTAL ===')
    print('Total thrust = ',np.round(sol(Fsail[0] + Fprop[0])/1e3,2), 'kN')
    print('Total drag = ',np.round(sol(Fhull[0] + Fdrift[0] + Faa[0] + Fapp[0] + Fwave[0])/1e3,2), 'kN')
    print()
    print('Total anti-drift = ',np.round(sol(Fapp[1] + Fhull[1] + Fdrift[1] + Fprop[1])/1e3,2), 'kN')
    print('Total drift = ',np.round(sol(Fsail[1] + Faa[1])/1e3,2), 'kN')
    print()
    print('=== NET BUOYANCY ===')
    print(np.round(sol(Fb[2] + Fw[2])/1e3/g,2), 'tons')
    print()
    print('=== ARMS ===')
    print('Heeling, Pitching, Yawing')
    print(sol(Mtot) / hull.displacement / g, 'm')
    
    # print()
    # print(Fsail)
    # print(n136.aeroVLMdata.lift_force_underway)
    # print(Fsail/n136.aeroVLMdata.lift_force_underway)
    
    if draw3D:
        # n136.draw_vlm()
        n136.draw_vlm(z_water= 0.,
                      recalculate_streamlines=True, show_edges=False,
                      cmap='autumn', streamlines_c='orange', plot_axes=False)

    if disp_op_point:
        attrs = {
        "stw": sol(op_point).stw,
          "tws0": sol(op_point).tws0,
          "twa": sol(op_point).twa,
          "z0": sol(op_point).z0,
          "a": sol(op_point).a,
          "heel": sol(op_point).heel,
          "trim": sol(op_point).trim,
          "leeway": sol(op_point).leeway,
          "immersion": sol(op_point).immersion,
          }
        
        script = "op_point = OperatingPoint(\n" + ",\n".join(
        f"    {key}={repr(value)}" for key, value in attrs.items()) + "\n)"
        
        print()
        print(script) 