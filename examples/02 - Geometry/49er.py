# -*- coding: utf-8 -*-
import archibald.numpy as np
import casadi as ca

from archibald.geometry.differentiable_mesh import DifferentiableMesh, ramp

import trimesh

from archibald2.optimization import Opti


def tall(array):
    return np.reshape(array, (-1, 1))


def wide(array):
    return np.reshape(array, (1, -1))

hullStl = r'data/49er_data/hull.stl'

rho = 1025.
displacement = 290. # kg

z = 0.
leeway = 0.
heel = 0.
trim = 0.

cog = np.array([2.40,
                .1,
                0.14]) # m

hullMesh = trimesh.load(hullStl)
vertices, faces = hullMesh.vertices, hullMesh.faces



leewayRot = np.rotation_matrix_3D((leeway)*np.pi/180,
                                  np.array([0.,0.,1.]))

antiLeewayRot = np.rotation_matrix_3D((leeway)*np.pi/180,
                                      np.array([0.,0.,1.]))

vertices = vertices @ (antiLeewayRot) - z

point = np.array([0.0, 0.0, 0.0])
normal = np.array([0., 0., 1.])

initMesh = DifferentiableMesh(vertices, faces)

initVertDist = initMesh.vertices_distances_to_plane(point, 'z')
initFaceDist = initMesh.faces_distances_to_plane(point, 'z')

# NB
# if np.sum(ramp(dist)) == 0 : fully immersed
# if np.prod(ramp(dist)) == 1 : fully emerged

fully_immersed = ca.if_else(np.abs(np.sum(ramp(initVertDist))) < 1e-5, True, False)
fully_emerged = ca.if_else(np.abs(np.prod(ramp(initVertDist)) - 1.) < 1e-5, True, False)

if fully_immersed:
    ### HYDROSTATICS COMPUTATION
    
    volume = initMesh.volume
    cob = initMesh.volume_centroid
    
    Wsa = initMesh.area
    cow = initMesh.area_centroid
    
    Ax = initMesh.frontal_area('x')
    Ay = initMesh.frontal_area('y')
    
    bounds = initMesh.bounds
    
    X0 = bounds[0,0]
    T = bounds[2,1] - bounds[2,0]
    
    Ax = initMesh.frontal_area('x')
    Ay = initMesh.frontal_area('y')
    
    dl = (bounds[0,1] - bounds[0,0])/100.
    
    transomMesh = initMesh.compress_mesh(np.array([1.,0.,0.])*(X0+dl), 'x',
                                         tol=1e-5, scale_fac=10.)
    
    Atr = transomMesh.frontal_area('x')
    # Atr = compute_volume_and_center_of_mass(transom, faces)[0] / dl # alternate approximate way to compute Atr
    
    z_min = initMesh.bounds[2,0]

    Ttr = wide(point)[0,2] - z_min
    
    hydrostaticData = {}
    
    hydrostaticData['volume'] = initMesh.volume
    
    points = {'cob': cob, 'cof': cob*0., 'cow': cow, '0L': X0}
    lengths = {'Lwl': 0., 'Bwl': 0., 'T': T, 'Ttr': Ttr, '0L': X0}
    areas = {'Wsa': Wsa, 'Wpa': 0., 'Ax': Ax, 'Ay': Ay, 'Atr': Atr, 'Abt': 0.}
    coefs = {'Cb': 1., 'Cp': 1., 'Cwp': 1., 'Cx': 1., 'Cy': 1.}

    hydrostaticData |= points | lengths | areas | coefs

    hydrostaticData['immersion'] = z
    hydrostaticData['heel'] = heel
    hydrostaticData['trim'] = trim
    hydrostaticData['ie'] = None
    
    ### AEROSTATICS COMPUTATION
    
    points = {'caa': cob*0.}
    areas = {'Dsa': 0., 'Ax': 0., 'Ay': 0.}
    
    aerostaticData = {}
    
    aerostaticData |= points | areas
    
    aerostaticData['immersion'] = z
    aerostaticData['heel'] = heel
    aerostaticData['trim'] = trim

elif fully_emerged:
    ### AEROSTATICS COMPUTATION
    
    dryBounds = initMesh.bounds
    
    Taa = dryBounds[2,1] - wide(point)[0,2]
    Axaa = initMesh.frontal_area('x')
    Ayaa = initMesh.frontal_area('y')
    Dsa = initMesh.weighted_area(weight=ramp(initFaceDist-1e-3, tol=1e-3))
    
    caa = initMesh.area_centroid
    
    points = {'caa': caa}
    areas = {'Dsa': Dsa, 'Ax': Axaa, 'Ay': Ayaa}
    
    aerostaticData = {}
    
    aerostaticData |= points | areas
    
    aerostaticData['immersion'] = z
    aerostaticData['heel'] = heel
    aerostaticData['trim'] = trim
    
    ### HYDROSTATICS COMPUTATION
    
    hydrostaticData = {}
    
    hydrostaticData['volume'] = 0.
    
    points = {'cob': caa*0., 'cof': caa*0., 'cow': caa*0., '0L': caa*0.}
    lengths = {'Lwl': 0., 'Bwl': 0., 'T': 0., 'Ttr': 0., '0L': 0.}
    areas = {'Wsa': 0., 'Wpa': 0., 'Ax': 0., 'Ay': 0., 'Atr': 0., 'Abt': 0.}
    coefs = {'Cb': 0., 'Cp': 0., 'Cwp': 0., 'Cx': 0., 'Cy': 0.}

    hydrostaticData |= points | lengths | areas | coefs

    hydrostaticData['immersion'] = z
    hydrostaticData['heel'] = heel
    hydrostaticData['trim'] = trim
    hydrostaticData['ie'] = None

else:
    ### HYDROSTATICS COMPUTATION
    
    wetMesh = initMesh.compress_mesh(point, normal, dist=initVertDist,
                                     tol=1e-5, scale_fac=2.)

    dryMesh = initMesh.compress_mesh(point, -normal, dist=-initVertDist,
                                     tol=1e-5, scale_fac=2.)
    
    waterplaneMesh = wetMesh.compress_mesh(point, -normal, dist=-initVertDist,
                                           tol=1e-5, scale_fac=100.)
    
    # transomMesh = wetMesh.compress_mesh()
    
    volume = wetMesh.volume
    cob = wetMesh.volume_centroid
    
    Wsa = wetMesh.weighted_area(weight=ramp(initFaceDist))
    cow = wetMesh.weighted_area_centroid(weight=ramp(initFaceDist, tol=1e-5))
    
    Wpa = waterplaneMesh.weighted_area(weight=ramp(initFaceDist))
    cof = waterplaneMesh.weighted_area_centroid(weight=ramp(initFaceDist, tol=1e-5))
    
    wetBounds = wetMesh.bounds
    
    X0 = wetBounds[0,0]
    Lwl = wetBounds[0,1] - wetBounds[0,0]
    Bwl = wetBounds[1,1] - wetBounds[1,0]
    T = wetBounds[2,1] - wetBounds[2,0]
    
    Ax = wetMesh.frontal_area('x')
    Ay = wetMesh.frontal_area('y')
    Awp = waterplaneMesh.frontal_area('z')
    
    dl = Lwl/100.
    
    transomMesh = wetMesh.compress_mesh(np.array([1.,0.,0.])*(X0+dl), 'x',
                                        tol=1e-5, scale_fac=10.)
    
    Atr = transomMesh.frontal_area('x')
    # Atr = compute_volume_and_center_of_mass(transom, faces)[0] / dl # alternate approximate way to compute Atr
    
    v = transomMesh.vertices
    
    z_min = np.min(v[:,2])
    z_max = np.max(v[:,2])

    Ttr = z_max - z_min
    
    Cb = volume / (Lwl * Bwl * T)
    Cp = volume / (Ax * Lwl)
    Cx = Ax / (Bwl * T)
    Cy = Ay / (Lwl * T)
    Cwp = Awp / (Lwl * Bwl)
    
    hydrostaticData = {}
    
    hydrostaticData['volume'] = volume
    
    points = {'cob': cob, 'cof': cof, 'cow': cow, '0L': X0}
    lengths = {'Lwl': Lwl, 'Bwl': Bwl, 'T': T, 'Ttr': Ttr, '0L': X0}
    areas = {'Wsa': Wsa, 'Wpa': Awp, 'Ax': Ax, 'Ay': Ay, 'Atr': Atr, 'Abt': 0.}
    coefs = {'Cb': Cb, 'Cp': Cp, 'Cwp': Cwp, 'Cx': Cx, 'Cy': Cy}

    hydrostaticData |= points | lengths | areas | coefs

    hydrostaticData['immersion'] = z
    hydrostaticData['heel'] = heel
    hydrostaticData['trim'] = trim
    hydrostaticData['ie'] = None
    
    ### AEROSTATICS COMPUTATION
    
    dryBounds = dryMesh.bounds
    
    Taa = dryBounds[2,1] - dryBounds[2,0]
    Axaa = dryMesh.frontal_area('x')
    Ayaa = dryMesh.frontal_area('y')
    Dsa = dryMesh.weighted_area(weight=ramp(initFaceDist-1e-3, tol=1e-3))
    
    caa = dryMesh.area_centroid
    
    points = {'caa': caa}
    areas = {'Dsa': Dsa, 'Ax': Axaa, 'Ay': Ayaa}
    
    aerostaticData = {}
    
    aerostaticData |= points | areas
    
    aerostaticData['immersion'] = z
    aerostaticData['heel'] = heel
    aerostaticData['trim'] = trim

diff_mesh.draw()
