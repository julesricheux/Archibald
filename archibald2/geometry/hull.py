# -*- coding: utf-8 -*-
"""
SHIP HULLS DESCRIPTION AND COMPUTATION
(hydrostatics, resistance prediction, appendage hydrodynamics)

Created: 30/05/2023
Last update: 24/12/2023

@author: Jules Richeux
@university: ENSA Nantes, FRANCE
@contributors: -

Further development plans:
    
    - Stabilise the hydrostatics computation. It can currently return an error
      when the immersion is initialised so that the hull has no waterplane (i.e.
      is fully dry or immersed, i.e. z is outside the mesh vertical bounds)
    
    - Implement a direct method to compute multihulls hull resistance. After
      hydrostatics computation, the mesh of each hull needs to be computed separately.
      Spacing and positionning between hulls should also be evaluated to apply
      interactions coefficients.
    
    - Implement Savisky planing and pre-planing methods for fast vessels (this
      already has been investigated independently, see experimental features)
    
    - Migrate the vortex lattice method from AVL to AeroSandbox (see experimental
      features). AeroSandbox offers better geometry management and enhanced graphics.
      The main difficulty is to keep consistent the viscous XFoil coupling.
"""

#%% DEPENDENCIES

import os

import scipy.optimize as opt

import trimesh as trimesh
import trimesh.transformations as tf

import copy

from typing import Union

#from archibald2.geometry.legacy.lifting_planes import Centreboard, Rudder, CantingKeel
from archibald2.environment.environment import Environment

from archibald2.geometry.differentiable_mesh import DifferentiableMesh

from archibald2.tools.dyn_utils import Cf_hull
from archibald2.tools.math_utils import set_normal, ReLU, ramp, rotate_single_vector, read_coefs, build_interpolation, wide, tall
import archibald2.tools.units as u
from archibald2.performance.operating_point import OperatingPoint
import archibald2.dynamics.hydro.holtrop as holtrop

import archibald2.numpy as np

import casadi as ca



#%% CLASSES

class Hull():
    def __init__(self,
                 name: str = 'hull',
                 displacement: float = 0.0,
                 cog: np.array = np.zeros(3),
                 mesh: str = None,
                 vertices = None,
                 faces = None,
                 inv_x: bool = True,
                 env: Environment = Environment()):
        
        self.name = name
        self.nAppendages = 0
        self.appendages = {}
        
        self.cog = cog
        self.displacement = displacement
        
        if mesh and os.path.exists(mesh):
            self.mesh = trimesh.load(mesh)
            if inv_x:
                self.mesh.vertices *= np.array([[-1., 1., 1.]])
                self.mesh.faces = self.mesh.faces[:, [0, 2, 1]]
            self.vertices0 = copy.copy(self.mesh.vertices)
            self.diff_mesh = DifferentiableMesh(self.vertices0, self.mesh.faces)
            self.Loa = self.mesh.bounds[1,0] - self.mesh.bounds[0,0]
            self.Boa = self.mesh.bounds[1,1] - self.mesh.bounds[0,1]
            self.D = self.mesh.bounds[1,2] - self.mesh.bounds[0,2]
        elif vertices is not None and faces is not None:
            self.mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
            if inv_x:
                self.mesh.vertices *= np.array([[-1., 1., 1.]])
            self.vertices0 = copy.copy(self.mesh.vertices)
            self.diff_mesh = DifferentiableMesh(self.vertices0, self.mesh.faces)
            self.Loa = self.mesh.bounds[1,0] - self.mesh.bounds[0,0]
            self.Boa = self.mesh.bounds[1,1] - self.mesh.bounds[0,1]
            self.D = self.mesh.bounds[1,2] - self.mesh.bounds[0,2]
        else:
            self.mesh = None
            self.Loa = None
            self.Boa = None
            self.D = None
        
        self.area = 0.0
        
        self.environment = env
        
        self._globalData = None
        self._localData = None
        
        # Coefficients for DSYHS computation
        current = os.path.dirname(os.path.realpath(__file__))
        _keunig_coefs_file = os.path.join(current, '../data/coefs_keunig_2008.csv')
        
        if os.path.exists(_keunig_coefs_file):
            _coefs = read_coefs(_keunig_coefs_file, skipRows=1, delim='\t')
            self._keunig_np = build_interpolation(_coefs, method='cubic')
            self._keunig_ca = [ca.interpolant('LUT',
                                              'bspline',
                                              # 'linear',
                                              [_coefs[:,0]],
                                              _coefs[:,i]) 
                               for i in range(1, _coefs.shape[1])]
        else:
            self._keunig_np = None
            self._keunig_ca = None
            
        # Coefficients for additionnal drift resistance computation
        _drift_coefs = np.array([[0.        , 0.        ],
                                 [0.78539816, 0.47      ],
                                 [1.        , 1.05      ]])
        self._drift = build_interpolation(_drift_coefs, method='linear')[0]
        
        self.hydrostaticData = dict()
        self.aerostaticData = dict()
        
    def reset_differentiable_vertices(self):
        self.diff_mesh.vertices = copy.copy(self.vertices0)*np.array([[-1., 1., 1.]])


    def compute_minimal_hydrostatics(self,                             
                                     z: float = 0., # m
                                     heel:float = 0., # degrees
                                     trim: float = 0., # degrees
                                     ):       
        n = set_normal(heel, trim)

        # Extract the immersed hull
        underwater = self.mesh.slice_plane(plane_origin=np.array([0,0,z]), plane_normal=n, cap=True)

        # Compute main properties
        volume = underwater.volume #volume
        cob = underwater.center_mass #center of buoyancy
        
        return volume, cob

    def compute_hydrostatics(self,
                             z: float = 0., # m
                             heel:float = 0., # degrees
                             trim: float = 0., # degrees
                             point = None,
                             normal = None,
                             disp: bool = False):
        
        if point is None:
            point = np.array([0,0,z])
        if normal is None:
            normal = set_normal(heel, trim)
        
        # try:
        # Extract the immersed hull
        underwater = self.mesh.slice_plane(plane_origin=point, plane_normal=-normal, cap=True)
        ws = self.mesh.slice_plane(plane_origin=point, plane_normal=normal, cap=False)    
        floatation = self.mesh.section(plane_origin=point, plane_normal=normal)
        
        
        # Main hyrostatics computation
        volume = underwater.volume #volume
        
        uw_bounds = underwater.bounds
        f_bounds = floatation.bounds
        
        if uw_bounds is None:
            uw_bounds = np.zeros((3,3))
        if f_bounds is None:
            f_bounds = np.zeros((3,3))
        
        Lwl = (f_bounds[1,0] - f_bounds[0,0])/np.cosd(trim)
        Bwl = (f_bounds[1,1] - f_bounds[0,1])/np.cosd(heel)
        T = (uw_bounds[1,2] - uw_bounds[0,2])/np.cosd(heel)
        
        
        # BMt computation
        c = floatation.centroid
        v = floatation.vertices - c
        
        rot_matrix_trim = tf.rotation_matrix(np.radians(-trim), np.array([0,1,0]))
        rot_matrix_heel = tf.rotation_matrix(np.radians(-heel), np.array([1,0,0]))

        
        section = tf.transform_points(v, rot_matrix_trim)
        section = tf.transform_points(section, rot_matrix_heel)
        
        peak = section[np.argmax(section[:, 0]), :]
        
        section = section - peak + np.array([Lwl, 0, 0])

        # path2d = trimesh.path.path.Path2D(floatation.entities, vertices=section)
        planar = trimesh.path.Path2D(entities=floatation.entities.copy(),
                                     vertices=section[:, :2],
                                     metadata=floatation.metadata.copy(),
                                     process=False)
        
        if disp:
            planar.show()
        
        # Half-entry angle computation
        # x_interval = [Lwl*389/400, 399/400*Lwl]
        # sel = (section[:, 0] >= x_interval[0]) & (section[:, 0] <= x_interval[1])
        bow_pts = section[section[:, 0].argsort()][-10:] - np.array([Lwl, 0, 0])
        
        star_bow_pts = bow_pts[(bow_pts[:, 1] < 0.)]
        port_bow_pts = bow_pts[(bow_pts[:, 1] >= 0.)]
        
        star_offset = star_bow_pts[np.argmax(star_bow_pts[:, 0]), :]
        port_offset = port_bow_pts[np.argmax(port_bow_pts[:, 0]), :]
        
        star_bow_pts -= star_offset
        port_bow_pts -= port_offset
        
        star_bow_pts = star_bow_pts[(star_bow_pts[:, 0] != 0.)]
        port_bow_pts = port_bow_pts[(port_bow_pts[:, 0] != 0.)]
        
        star_ie = np.degrees(np.mean(np.arctan(star_bow_pts[:,1]/-star_bow_pts[:,0])))
        port_ie = np.degrees(np.mean(np.arctan(port_bow_pts[:,1]/-port_bow_pts[:,0])))
        
        ie = (port_ie - star_ie)/2
        
        Ixx = 0.0
        Iyy = 0.0
        for poly in planar.polygons_closed:
            Ixx += trimesh.path.polygons.second_moments(poly)[0]
            Iyy += trimesh.path.polygons.second_moments(poly)[1]

        # Compute main properties
        volume = underwater.volume #volume
        cob = underwater.center_mass #center of buoyancy
        BMt = Ixx / volume # transverse metacentric radius
        BMl = Iyy / volume # longitudinal metacentric radius
        
        # RMt computation
        yrot = np.array([0, np.cosd(heel), np.sind(heel)])
        zrot = np.array([0 ,-np.sind(heel), np.cosd(heel)])
        
        metaT = cob + BMt*zrot
        
        # BMtvec = metaT - cob
        
        GMtvec = metaT - self.cog
        GZtvec = GMtvec - np.dot(GMtvec, zrot.T) * zrot
        
        GMt = np.linalg.norm(GMtvec[1:]) * np.sign(np.dot(GMtvec[1:], zrot[1:]))
        GZt = np.linalg.norm(GZtvec[1:]) * np.sign(np.dot(GZtvec[1:], yrot[1:]))
        
        # RMl computation
        xrot = np.array([np.cosd(heel), 0, np.sind(heel)])
        zrot = np.array([-np.sind(heel), 0, np.cosd(heel)])
        
        metaL = cob + BMl*zrot
        
        # BMlvec = metaL - cob

        GMlvec = metaL - self.cog
        GZlvec = GMlvec - np.dot(GMlvec, zrot.T) * zrot
        
        GMl = np.linalg.norm(GMlvec[[0,2]]) * np.sign(np.dot(GMlvec[[0,2]], zrot[[0,2]]))
        GZl = np.linalg.norm(GZlvec[[0,2]]) * np.sign(np.dot(GZlvec[[0,2]], xrot[[0,2]]))
        
        # Compute floatation
        Uwa = underwater.area
        Wsa = ws.area
        Wpa = Uwa - Wsa
        
        Uw = underwater.centroid
        Ws = ws.centroid
        
        cof = (Uwa * Uw - Wsa * Ws) / Wpa #center of floatation
        
        # Compute midship area
        ms = underwater.section(plane_origin=cof, plane_normal=[1,0,0])
        if ms is None:
            Ax = 0.0
        else:
            ms = ms.to_2D()
            Ax = ms[0].area
            
        # Compute transversal area
        cl = underwater.section(plane_origin=cof, plane_normal=[0,1,0])
        if cl is None:
            Ay = 0.0
        else:
            cl = cl.to_2D()
            Ay = cl[0].area
            
        # Compute transom area
        xap = max(f_bounds[0,0], uw_bounds[0,0])
        tr = underwater.section(plane_origin=[xap+min(Lwl/100, .1),0,0], plane_normal=[1,0,0])
        if tr is None:
            Atr = 0.0
        else:
            tr = tr.to_2D()
            Atr = tr[0].area
            Ttr=np.min(np.abs(tr[0].bounds[1]-tr[0].bounds[0]))
            
        # Compute transverse bulb area
        xfp = f_bounds[1,1]
        bt = underwater.section(plane_origin=[xfp+min(Lwl/100, .1),0,0], plane_normal=[1,0,0])
        if bt is None:
            Abt = 0.0
        else:
            bt = bt.to_planar()
            Abt = bt[0].area
        
        # compute hydro coefficients
        Cb = volume / (Lwl*Bwl*T)
        Cp = volume / (Lwl*Ax)
        Cwp = Wpa / (Lwl*Bwl)
        Cx = Ax / (Bwl*T)
        Cy = Ay / (Lwl*T)
        
        # lengths = np.array([BMt, GMt, GZt, BMl, GMl, GZl, Lwl, Bwl, T])
        # areas = np.array([Wsa, Wpa, Amc, Atr, Abt])
        # coefs = np.array([Cb, Cp, Cwp, Cm])
        
        lengths = {'BMt': BMt, 'GMt': GMt, 'GZt': GZt, 'BMl': BMl, 'GMl': GMl, 'GZl': GZl, 'Lwl': Lwl, 'Bwl': Bwl, 'T': T, 'Ttr': Ttr, '0L': f_bounds[0,0]}
        areas = {'Wsa': Wsa, 'Wpa': Wpa, 'Ax': Ax, 'Ay': Ay, 'Atr': Atr, 'Abt': Abt}
        coefs = {'Cb': Cb, 'Cp': Cp, 'Cwp': Cwp, 'Cx': Cx, 'Cy': Cy}
        
        self.hydrostaticData = lengths | areas | coefs
        
        # cow = ws.centroid
        # cdyn = cow - np.array([Lwl/4., 0., 0.]) # center of pressure condidered at 25% chord
        # X0 = f_bounds[0,0]
        
        # self.cob = cob
        # self.cof = cof
        # self.cows = cow
        # self.volume = volume
        
        # self.hydrostaticData = {}
        
        # self.hydrostaticData['volume'] = volume
        
        # points = {'cob': cob, 'cof': cof, 'cow': cow, 'cdyn': cdyn, '0L': X0}
        # lengths = {'Lwl': Lwl, 'Bwl': Bwl, 'T': T, 'Ttr': Ttr, '0L': X0}
        # areas = {'Wsa': Wsa, 'Wpa': Wpa, 'Ax': Ax, 'Ay': Ay, 'Atr': Atr, 'Abt': 0.}
        # coefs = {'Cb': Cb, 'Cp': Cp, 'Cwp': Cwp, 'Cx': Cx, 'Cy': Cy}
    
        # self.hydrostaticData |= points | lengths | areas | coefs
        
        self.hydrostaticData['immersion'] = z
        self.hydrostaticData['heel'] = heel
        self.hydrostaticData['trim'] = trim
        self.hydrostaticData['ie'] = ie
        self.hydrostaticData['areaCentroid'] = Ws
        
        return volume, cob, cof, lengths, areas, coefs
        
        # except:
        #     self.hydrostaticData = {}
        #     return 0., np.array([0.,0.,0.]), np.array([0.,0.,0.]), {}, {}, {}
        
    def compute_aerostatics(self,
                            z: float = 0., # m
                            heel:float = 0., # degrees
                            trim: float = 0., # degrees
                            disp: bool = False):
        
        n = set_normal(heel, trim)
        
        try:
            # Extract the emerged hull
            overwater = self.mesh.slice_plane(plane_origin=[0.,0.,z], plane_normal=-n, cap=True)
            ws = self.mesh.slice_plane(plane_origin=[0.,0.,z], plane_normal=-n, cap=False)  
    
            # Compute main properties
            volume = overwater.volume #volume
            cow = overwater.center_mass #center of buoyancy
            
            Uwa = overwater.area
            Wsa = ws.area
            Wpa = Uwa - Wsa
            
            Ws = ws.centroid
            
            ms = overwater.section(plane_origin=cow, plane_normal=[1.,0.,0.])
            if ms is None:
                Ax = 0.0
            else:
                ms = ms.to_planar()
                Ax = ms[0].area
                
            # Compute transversal area
            cl = overwater.section(plane_origin=cow, plane_normal=[0.,1.,0.])
            if cl is None:
                Ay = 0.0
            else:
                cl = cl.to_planar()
                Ay = cl[0].area
            
            areas = {'Wsa': Wsa, 'Ax': Ax, 'Ay': Ay}
            
            self.aerostaticData = areas
            
            self.aerostaticData['volume'] = volume
            self.aerostaticData['immersion'] = z
            self.aerostaticData['heel'] = heel
            self.aerostaticData['trim'] = trim
            self.aerostaticData['volumeCentroid'] = cow
            self.aerostaticData['areaCentroid'] = Ws
            
            return volume, cow, areas
        
        except:
            self.aerostaticData = {}
            return 0., np.array([0.,0.,0.]), np.array([0.,0.,0.]), {}, {}, {}
    
    def free_heel_trim_immersion(self, heel0=0., trim0=0., z0=0., disp=False):
        
        rho = self.environment.water.density
        
        def heel_trim_immersion(X, rho):
            # initialize parameters
            z, heel, trim = X
            n = set_normal(heel, trim)

            # compute current hydrostatics
            volume, cob = self.compute_minimal_hydrostatics(z, heel, trim)
            
            print(volume)
            
            BG = self.cog - cob # CoB-CoG vector
            prod = np.cross(BG, n) # cross product to check colinearity
            prod = np.linalg.norm(prod)
            
            det1 = np.linalg.det([BG[1:], n[1:]])
            det2 = np.linalg.det([BG[[0,2]], n[[0,2]]])
            
            eq = np.array([volume*rho/self.displacement - 1,
                           det1,
                           det2])
            
            return np.linalg.norm(eq)
        
        x0 = np.array([z0, heel0, trim0])
        param = (rho)
        
        # limits = [(0,10), (-60, 60), (-10,10)]
        
        Xopt = opt.minimize(heel_trim_immersion, x0, args=param, tol=1e-5)
        
        volume, cob, cof, lengths, areas, coefs = self.compute_hydrostatics(Xopt.x[0], heel=Xopt.x[1], trim=Xopt.x[2], disp=disp)

        return volume, cob, cof, lengths, areas, coefs, Xopt.x

    def free_trim_immersion(self, heel=0, trim0=0., z0=0., disp=False):
        
        rho = self.environment.water.density
        
        def trim_immersion_eq(X, heel, rho):
            # initialize parameters
            z, trim = X
            n = set_normal(heel, trim)

            # compute current hydrostatics
            volume, cob = self.compute_minimal_hydrostatics(z, heel, trim)
            
            BG = self.cog - cob # CoB-CoG vector
            
            det = np.linalg.det([BG[[0,2]], n[[0,2]]])
            
            eq = np.array([volume*rho/self.displacement - 1,
                           det])
            
            return np.linalg.norm(eq)
        
        x0 = np.array([z0, trim0])
        param = (heel, rho)
        
        Xopt = opt.minimize(trim_immersion_eq, x0, args=param, tol=1e-5)        
        
        volume, cob, cof, lengths, areas, coefs = self.compute_hydrostatics(Xopt.x[0], heel=heel, trim=Xopt.x[1], disp=disp)
        
        return volume, cob, cof, lengths, areas, coefs, Xopt.x

    def free_heel_immersion(self, trim=0, heel0=1., z0=0., disp=False):
        
        rho = self.environment.water.density
        
        def heel_immersion_eq(X, trim, rho):
            # initialize parameters
            z, heel = X
            n = set_normal(heel, trim)

            # compute current hydrostatics
            volume, cob = self.compute_minimal_hydrostatics(z, heel, trim)
            
            BG = self.cog - cob # CoB-CoG vector
            
            det = np.linalg.det([BG[1:], n[1:]])
            
            eq = np.array([volume*rho/self.displacement - 1,
                           det])
            
            return np.linalg.norm(eq)
        
        
        x0 = np.array([z0, heel0])
        param = (trim, rho)
        
        Xopt = opt.minimize(heel_immersion_eq, x0, args=param, tol=1e-5)
        
        volume, cob, cof, lengths, areas, coefs = self.compute_hydrostatics(Xopt.x[0], heel=Xopt.x[1], trim=trim, disp=disp)
        
        return volume, cob, cof, lengths, areas, coefs, Xopt.x

    def free_immersion(self, heel=0., trim=0., z0=0., disp=False):
        
        rho = self.environment.water.density
        
        def immersion_eq(X, heel, trim, rho):
            # initialize parameters
            z = X[0]

            # compute current hydrostatics
            volume, cob = self.compute_minimal_hydrostatics(z, heel, trim)
            
            eq = volume*rho-self.displacement
            
            return eq
        
        x0 = np.array([z0])
        param = (heel, trim, rho)
        
        # xBounds = [(self.mesh.bounds[0,2], self.mesh.bounds[1,2])]
        
        # Xopt = opt.minimize(immersion_eq, x0, args=param, bounds=xBounds, tol=1e-5)
        Xopt = opt.fsolve(immersion_eq, x0, args=param, xtol=1e-5)
        
        volume, cob, cof, lengths, areas, coefs = self.compute_hydrostatics(Xopt[0], heel=heel, trim=trim, disp=disp)
        
        return volume, cob, cof, lengths, areas, coefs, Xopt
        
    
    def compute_resistance_dsyhs(self, V):
        """
        Computes the total resistance of a ship given its velocity, geometric
        parameters and fluid properties following DSYHS method

        Args:
            V (float): ship speed (kts)

        Returns:
            float: DSYHS bare hull resistance (N)
        """
        
        # Physical data
        rho = self.environment.water.density
        nu = self.environment.water.kinematic_viscosity
        g = self.environment.gravity
        
        volume = self.hydrostaticData['volume']
        cob = self.hydrostaticData['cob']
        cof = self.hydrostaticData['cob']
        Lwl = self.hydrostaticData['Lwl']
        X0 = self.hydrostaticData['0L']
        
        displacement = volume * rho * g
        
        lcbfpp = X0 + Lwl - wide(cob)[0,0]
        LCFfpp = X0 + Lwl - wide(cof)[0,0]
        
        Lwl = self.hydrostaticData['Lwl']
        Bwl = self.hydrostaticData['Bwl']
        T = self.hydrostaticData['T']
        Ttr = self.hydrostaticData['Ttr']
        Cx = self.hydrostaticData['Cx']
        Cp = self.hydrostaticData['Cp']
        Wpa = self.hydrostaticData['Wpa']
        Wsa = self.hydrostaticData['Wsa']
        
        Atr = self.hydrostaticData['Atr']
        
        Vms = V * u.knot
        
        Vms_ReLU = ReLU(Vms)

        # Transom additionnal resistance (Holtrop&Mennen, 1978)
        Fr_T = Vms / np.sqrt(g * Ttr + 1e-6)
        
        # ctr = np.max((0.2 * (1 - (0.2 * Fr_T)), 0.0))
        
        # if Fr_T < 5:
        ctr = 0.2 * (1 - (0.2 * Fr_T))
        ctr_ReLu = ReLU(ctr)
        Rtr = 0.5 * rho * (Vms ** 2) * Atr * ctr_ReLu
        
        Fr = Vms_ReLU / np.sqrt(g*Lwl)
        Re = Vms_ReLU * Lwl / nu
        # Fr = Vms / np.sqrt(g*Lwl)
        # Re = Vms * Lwl / nu
        
        # Bare hull viscous resistance
        Rvh = 1/2 * rho * (Wsa - Atr) * Cf_hull(Re+10.) * Vms**2
        
        # Bare hull upright residuary resistance DSYHS
        if np.is_casadi_type(Fr):
            K = self._keunig_ca
        else:
            K = self._keunig_np
            
        lcbfpp = 0.5
            
        Rrh = displacement * (K[0](Fr) + volume**(1/3)/Lwl * (K[1](Fr) * lcbfpp/Lwl + \
                                                              K[2](Fr) * Cp + \
                                                              K[3](Fr) * volume**(2/3)/Wpa + \
                                                              K[4](Fr) * Bwl/Lwl + \
                                                              K[5](Fr) * lcbfpp/LCFfpp + \
                                                              K[6](Fr) * Bwl/T + \
                                                              K[7](Fr) * Cx)) 
            
        return (Rvh + Rrh + Rtr) * np.sign(V)
        # return (Rvh + Rtr) * np.sign(V)

    def compute_resistance_holtrop(self, V, uInterval=False):
        """
        Computes the total resistance of a ship given its velocity, geometric
        parameters and fluid properties following Holtrop's 1978 method

        Args:
            V (float): ship speed (kts)

        Returns:
            float: Holtrop bare hull resistance (N)
        """
        
        # Physical data
        rho = self.environment.water.density
        nu = self.environment.water.kinematic_viscosity
        g = self.environment.gravity
        
        self.Loa = self.mesh.bounds[1,0] - self.mesh.bounds[0,0]
        # Boa = self.mesh.bounds[1,1] - self.mesh.bounds[0,1]
        volume = self.hydrostaticData['volume']
        cob = self.hydrostaticData['cob']
        cof = self.hydrostaticData['cof']
        X0 = self.hydrostaticData['0L']
        
        Lwl = self.hydrostaticData['Lwl']
        Bwl = self.hydrostaticData['Bwl']
        T = self.hydrostaticData['T']
        Ttr = self.hydrostaticData['Ttr']
        # Ttr = lengths['Ttr']
        Cx = self.hydrostaticData['Cx']
        Cp = self.hydrostaticData['Cp']
        Cb = self.hydrostaticData['Cb']
        Cwp = self.hydrostaticData['Cwp']
        Wpa = self.hydrostaticData['Wpa']
        Wsa = self.hydrostaticData['Wsa']
        
        Atr = self.hydrostaticData['Atr']
        Abt = self.hydrostaticData['Abt']
        
        Loa = self.Loa
        Lbp = self.Loa # length between perpendiculars
        lcb = cob[0]
        lcb = (lcb-X0)/Lwl - .5
        
        origin = self.hydrostaticData['0L']
        ie = self.hydrostaticData['ie']
        
        if ie is None:
            Lr = Lwl*(1 - Cp + 0.06*Cp*lcb/(4*Cp-1))
            
            ie = 1 + (89 * (np.exp(
                (-(Lbp / Bwl) ** 0.80856) * ((1 - Cwp) ** 0.30484) * ((1 - Cp - (0.0225 * lcb)) ** 0.6367) * (
                            (Lr / Bwl) ** 0.34574) * (((100 * volume) / (Lbp ** 3)) ** 0.16302))))
        
        # hB = cob[2] - self.mesh.bounds[0,2] # bulb center above keel line
        hB = T/2
        
        # Csternchoice = np.fmax(1, np.round(Cp * 4))
        Csternchoice = 3
        Bulbchoice = 0
        
        Vms = V * u.knot
        
        Vms_ReLU = ReLU(Vms)
        
        Lbp = 136.
        Lwl = 136.
        lcb = 0.5
        
        Rf = holtrop.compute_Rf_holtrop(Vms, Lbp, Loa, Lwl, volume, Bwl, T, Wsa, Cp, lcb, Csternchoice, nu, rho, origin, uInterval)
        Rw = holtrop.compute_Rw_holtrop(Vms, Lwl, Lbp, Bwl, T, Bwl, Abt, Cp, Cwp, Atr, lcb, hB, Cx, ie, rho, g)
        Rb = holtrop.compute_Rb_holtrop(Vms, T, hB, Abt, Bulbchoice, rho, g)
        Rtr = holtrop.compute_Rtr(Vms, Ttr, Atr, Bwl, Cwp, rho, g)
        
        Ra = holtrop.compute_Ra_holtrop(Vms, Lwl, Bwl, T, Cb, hB, rho, Wsa, Abt, uInterval)
        
        if uInterval:
            return Rf[0] + Rw + Rb + Rtr + Ra[0], Rf[1] + Rw + Rb + Rtr + Ra[1]
        
        # print(Rf, Rw, Rb, Rtr, Ra)
        
        # return (Rtr + Ra) * np.sign(V)
        return (Rf + Rw + Rb + Rtr + Ra) * np.sign(V)
    
    def compute_propulsion_coefficients(self, stw, Ae_Ao):
        
        volume = self.hydrostaticData['volume']
        L = self.hydrostaticData['Lwl']
        B = self.hydrostaticData['Bwl']
        T = self.hydrostaticData['T']
        Taa = self.aerostaticData['Taa']
        D = T + Taa
        Cb = self.hydrostaticData['Cb']
        Cp = self.hydrostaticData['Cp']
        Cx = self.hydrostaticData['Cx']
        
        Wsa = self.hydrostaticData['Wsa']
        
        X0 = self.hydrostaticData['0L']
        cob = self.hydrostaticData['cob']
        lcb = cob[0]
        lcb = (lcb-X0)/L - .5
        
        Cstern = 0.
            
        c14 = 1 + 0.011*Cstern
        Lr = L*(1 - Cp + 0.06*Cp*lcb/(4*Cp-1))

        k = .93 + .487118*c14*(B/L)**1.06806 * (T/L)**.46106 * (L/Lr)**.121563 *\
               (L**3/volume)**.36486 * (1-Cp)**(-.604247) - 1
               
        CA = 0.00675 * (L +100)**(-1/3) - 0.00064
             
        Re_ReLU = (ReLU(stw*u.knot) * L) / 1.2e-6
        
        CV = (1+k) * Cf_hull(Re_ReLU+10.) + CA
        
        # prediction of delivered power:
        
        c8 = ca.if_else(
            (B / T) < 5.0,
            B * Wsa / (L * D * T),
            Wsa * (7.0 * (B / T) - 25.0) / (L * D * ((B / T) - 3.0))
        )
        
        c9 = ca.if_else(
            c8 < 28.0,
            c8,
            32.0 - 16.0 / (c8 - 24.0)
        )
        
        # c11
        c11 = ca.if_else(
            T / D < 2.0,
            T / D,
            0.0833333 * ca.power(T / D, 3.0) + 1.33333
        )
            
        # c19
        c19 = ca.if_else(
            Cp < 0.7,
            0.12997 / (0.95 - Cb) - 0.11056 / (0.95 - Cp),
            0.18567 / (1.3571 - Cx) - 0.71276 + 0.38648 * Cp
        )


        #c20
        c20 = 1 + 0.015*Cstern

        # Cp1
        Cp1 = 1.45 * Cp - 0.315 - 0.0225 * lcb
        
        r"""Wake prediction for single screw ships according to :cite:`holtropStatisticalReAnalysisResistance1984`, p.  273:
    
        .. math::
    
            w = c_9 c{20} C_V \frac L{T_A} \left( 0.050776 + 0.93405 c_{11} \frac{C_V}
                {\left(1 - C_{P1} \right)} \right) + 0.27915 c_{20} \sqrt{\frac B{L\left(
                1 - C_{P1} \right)}} c_{19} c_{20}
    
        """
        w = (
            c9
            * c20
            * CV
            * (L / T)
            * (0.050776 + 0.93405 * c11 * (CV / (1 - Cp1)))
        ) + (
            0.27915 * c20 * np.sqrt(B / (L * (1 - Cp1)))
            + c19 * c20
        )
            
        r"""Thrust decuction prediction for single screw ships according to
        :cite:`holtropStatisticalReAnalysisResistance1984`, p.  274:
    
        .. math::
    
            t = \frac{0.25014 \left(\frac B L \right)^{0.28956} \left( \frac{\sqrt{B T}}D
                      \right)^{0.2624}}
                     {\left(1 - C_P + 0.0225 \text{lcb}\right)^{0.01762}} +
                0.0015 C_{\text{stern}}
    
        """
        t = (
            0.25014
            * np.power(B / L, 0.28956)
            * np.power(np.sqrt((B * T)) / D, 0.2624)
            / np.power(1 - Cp + 0.0225 * lcb, 0.01762)
            + 0.0015 * Cstern
        )
        
        r"""The relatigve-rotative efficiency prediction for single screw
        ships according according to :cite:`holtrop1982approximate`, pp.  168:
    
        .. math::
    
            η_R = 0.9922 - 0.05908 \frac{A_E}{A_O} +
               0.07424 \left( C_P - 0.0225 \text{lcb} \right)
        """
        eta_R = (
            0.9922
            - 0.05908 * Ae_Ao
            + 0.07424 * (Cp - 0.0225 * lcb)
        )
        
        return w, t, eta_R
        
        # def w_single_open_stern(speed, ship):
        #     r"""Wake prediction for single screw ships with open stern (as
        #     sometimes applied on slender, fast sailing ships) according to
        #     :cite:`holtrop1982approximate`, p.  169:
        
        #     .. math::
        
        #         w = 0.3 C_B + 10 C_V C_B - 0.23 \frac{D}{\sqrt{B T}}
        
        #     """
        #     return (
        #         0.3 * Cb
        #         + 10.0 * CV * Cb
        #         - 0.23 * D / np.sqrt(B * T)
        #     )
        
        
        # def t_single_open_stern(speed, ship):
        #     r"""Thrust decuction prediction for single screw ships with open stern
        #     (as sometimes applied on slender, fast sailing ships) according to
        #     :cite:`holtrop1982approximate`, p.  169:
        
        #     .. math::
        
        #         t = 0.10
        
        #     """
        #     return 0.1
        
        
        # def eta_R_single_open_stern(ship):
        #     r"""The relatigve-rotative efficiency prediction for single screw
        #     ships with open stern (as sometimes applied on slender, fast
        #     sailing ships) according according to :cite:`holtrop1982approximate`, pp.  168:
        
        #     .. math::
        
        #         η_R = 0.98
        
        #     """
        #     return 0.98
        
        
        # def w_twin(speed, ship):
        #     r"""Wake prediction for twin screw ships according to :cite:`holtrop1982approximate`, p.  169:
        
        #     .. math::
        
        #         w = 0.3095 C_B + 10 C_V C_B - 0.23 \frac D{\sqrt{B T}}
        
        #     """
        #     return (
        #         0.3095 * Cb
        #         + 10.0 * CV * Cb
        #         - 0.23 * D / np.sqrt(B * T)
        #     )
        
        
        # def t_twin(ship):
        #     r"""Thrust decuction prediction for twin screw ships according to :cite:`holtrop1982approximate`,
        #     p.  169:
        
        #     .. math::
        
        #         t = 0.325 C_B - 0.1885 \frac D {\sqrt{B T}}
        
        #     """
        #     return 0.325 * Cb - 0.1885 * D / np.sqrt(B * T)
        
        
        # def eta_R_twin(ship):
        #     r"""The relatigve-rotative efficiency prediction for twin screw ships
        #     according according to :cite:`holtrop1982approximate`, pp.  168:
        
        #     .. math::
        
        #         η_R = 0.9737 + 0.111 \left( C_P - 0.0225 \text{lcb} \right) + 0.06325 \frac P D
        #     """
        #     return (
        #         0.9737
        #         + 0.111 * (Cp - 0.0225 * lcb)
        #         + 0.06325 * ship.P() / D
        #     )

    
    
    def compute_hull_resistance(self, V, z=0, heel=None, trim=None, delta=0, method=str(), uInterval=False):
        """
        

        Parameters
        ----------
        V : ship's velocity (in kts).
        z : TYPE, optional
            DESCRIPTION. The default is 0.
        heel : TYPE, optional
            DESCRIPTION. The default is None.
        trim : TYPE, optional
            DESCRIPTION. The default is None.
        delta : TYPE, optional
            DESCRIPTION. The default is 0.
        method : TYPE, optional
            DESCRIPTION. The default is str().
        uInterval : TYPE, optional
            DESCRIPTION. The default is False.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        # Physical data
        rho = self.environment.water.density
        
        Vx = V * np.cosd(delta) * u.knot
        Vy = V * np.sind(delta) * u.knot
        Cb = self.hydrostaticData['Cb']
        Ay = self.hydrostaticData['Ay']
        
        if method == "dsyhs" and uInterval:
            print("Warning: uncertainty computation not yet implemented for DSYHS method")
            Rhull = 0.0
        elif method == "dsyhs":
            Rhull = self.compute_resistance_dsyhs(V)
        elif method == "holtrop":
            Rhull = self.compute_resistance_holtrop(V, uInterval)
        else:
            print("Warning: invalid method given for the hull resistance computation")
            Rhull = 0.0
        
        Rdrift = 1/2 * rho * Ay * Vy**2 * Cb # self._drift(Cb)
        
        if uInterval:
            return np.sqrt(Rhull[0]**2 + Rdrift**2), np.sqrt(Rhull[1]**2 + Rdrift**2)
            
        return np.sqrt(Rhull**2 + Rdrift**2)
    
    
    def compute_lateral_force(self,
                              op_point: OperatingPoint,
                              recompute_statics: bool = False,
                              method: str = 'dsyhs',
                              ):
        """
        Compute hull lateral force.
        
        """
        if not(self.hydrostaticData) or recompute_statics:
            self.compute_statics(op_point)
        
        stw = op_point.stw # in kts
        leeway = op_point.leeway # in deg
        
        Lwl = self.hydrostaticData['Lwl']
        Bwl = self.hydrostaticData['Bwl']
        T = self.hydrostaticData['T']
        Wsa = self.hydrostaticData['Wsa']
        Ay = self.hydrostaticData['Ay']
        Cb = self.hydrostaticData['Cb']
        Cx = self.hydrostaticData['Cx']
        
        AR = 2.*T/Lwl
        
        rho = self.environment.water.density
        
        center = self.hydrostaticData['cdyn'] * np.array([-1., -1., 1.])
        
        # if method == 'dsyhs':
        if False:
            """
            Adapted from Fossati.
            
            """
            # Fossati (Table 7.1)
            b1 = 2.025
            b2 = 9.551
            b3 = 0.631
            b4 = -6.575
            
            Flat = 1/2 * rho * (stw*u.knot)**2 * Wsa * leeway*u.rad * np.cosd(leeway) * \
                (b1 * T**2/Wsa +\
                 b2 * (T**2/Wsa)**2 +\
                 b3 * T/T +\
                 b4 * T/T * T**2/Wsa
                )
                    
            Flat = Flat * Ay / Wsa
                    
            return np.array([0., -1., 0.]) * Flat * Cb**2, np.zeros(3) # ARBITRARY CORR COEF. LATERAL HULL FORCE IS CURRENTLY NOT CONSISTENT
        
        # elif method == 'holtrop':
        elif False:
            """
            According to S. Inoue, A practical calculation method of ship maneuvering motion, Int. Shipbuild. Progr. 28, 207-222 (1981)
            
            """
            abs_leeway = np.sqrt(leeway**2 + 1e-15) # adding femto for derivability at zero
            # abs_leeway = leeway
            
            cl = 0.8 * 0.5 * np.pi * AR * np.sind(leeway) +\
                 0.6541 * np.sind(abs_leeway) * np.sind(leeway) * np.cosd(leeway)
            
            abs_cl = np.sqrt(cl**2 + 1e-15) # adding femto for derivability at zero
            
            cd = 0.66 * abs_cl * abs_leeway**0.6 #+\
                 # 0.6541 * np.sind(abs_leeway)**3 +\
                 
            # fl = 0.5 * rho * cl * Wsa * (stw*u.knot)**2
            # fd = 0.5 * rho * cd * Wsa * (stw*u.knot)**2
            fl = 0.5 * rho * cl * Ay * (stw*u.knot)**2 * Cb
            fd = 0.5 * rho * cd * Ay * (stw*u.knot)**2 * Cb
            
            corr = 2.5 # ARBITRARY CORR COEF. LATERAL HULL FORCE IS CURRENTLY NOT CONSISTENT
            
            Fdrift = np.array([-fd/corr,
                               -fl*corr,
                               0.])
            Mdrift = np.cross(center, Fdrift)
            
            return Fdrift, Mdrift
        
        elif method == 'holtrop' or method == 'dsyhs':
            
            # cl = np.pi/2 * AR * leeway
            # cd = cl**2 / (np.pi * AR)
            
            abs_leeway = np.sqrt(leeway**2 + 1e-15)
            
            # b = -np.log(2.)/np.log(np.pi/5.)
            # c = -1./2.
            
            # coef_Cx = (np.cos(2.*np.pi*(Cx**b + c)) + 1.)/2.
            coef_Cx = (1-2*Cx*(1-Cx))
            
            cl = 1.5 * np.pi * np.sind(leeway) * AR * coef_Cx
            cd = cl**2 / (np.pi * AR) * abs_leeway ** 0.6
            # cd = 0.66 * cl * abs_leeway**0.6
            
            corr_dyn_press = 0.5 * rho * Ay * (stw*u.knot)**2 
            
            fl = cl * corr_dyn_press
            fd = cd * corr_dyn_press
            
            Fdrift = np.array([-fd,
                               -fl,
                               0.])
            Mdrift = np.cross(center, Fdrift)
            
            return Fdrift, Mdrift
        
        elif method == 'dice':
            
            # TODO: Approximate to avoid copyright issues
             
            # Hydrodynamic derivatives from D-ICE's N169 AIRBUS study
            Xvv = -0.5565
            Xvvvv = 0.6294

            Yv = -0.1348
            Yvvv = -3.1545

            Kv = 0.2746
            Kvvv = -1.5372

            Nv = -0.07181
            Nvvv = -0.4713
            
            L = self.hydrostaticData['Lwl']
            T = self.hydrostaticData['T']
            
            rho = op_point.environment.water.density
            
            stw = op_point._stw # in m/s
            leeway = op_point.leeway # in degrees

            # Compute the modified velocity (v' and v)
            v = stw * np.sind(leeway)
            v_prime = v / stw

            # Compute dyn_pres (dynamic pressure term)
            dyn_pres = 0.5 * rho * stw**2

            # Modified forces and moments using the provided equations
            Fx = dyn_pres * L * T * (Xvv * v_prime**2 + Xvvvv * v_prime**4)
            Fy = dyn_pres * L * T * (Yv * v_prime + Yvvv * v_prime**3)
            Mx = dyn_pres * L * T * (Kv * v_prime + Kvvv * v_prime**3)
            Mz = dyn_pres * L * T * (Nv * v_prime + Nvvv * v_prime**3)  # Assuming Nvv = 0
            
            Fdrift = np.array([Fx, Fy, 0.])
            # Mdrift = np.array([Mx, 0., Mz])
            Mdrift = np.cross(center, Fdrift)
            # Mdrift = Fdrift

            return Fdrift, Mdrift
        
        else:
            raise(ValueError(f'unsupported method {method} for hull side force computation'))
            return np.zeros(3), np.zeros(3)
        
        
    def compute_resistance_force(self,
                                 op_point: OperatingPoint,
                                 method: str = 'dsyhs',
                                 ):
        stw = op_point.stw # in kts
        
        if method == "dsyhs":
            Rt = self.compute_resistance_dsyhs(stw)
        elif method == "holtrop":
            Rt = self.compute_resistance_holtrop(stw)
        else:
            raise(ValueError(f'unsupported method {method} for hull resistance computation'))
            return np.zeros(3), np.zeros(3)
        
        center = self.hydrostaticData['cow'] * np.array([-1., -1., 1.])
        # Cb = self.hydrostaticData['Cb']
        # Ay = self.hydrostaticData['Ay']
        
        # R90 = 1/2 * rho * Ay * (stw*u.knot)**2 * Cb # self._drift(Cb)
        
        # fac90 = (1. - np.cosd(leeway))/2.
        
        # RtotHull = Rt * (1-fac90) + R90 * fac90
        
        # Fhull = np.array([-RtotHull,
        #                   0.,
        #                   0.]) +\
        #         Fdrift
        
        Fhull = np.array([-Rt,
                          0.,
                          0.])
        
        Mhull = np.cross(center, Fhull)
        
        return Fhull, Mhull
    
    
    def compute_hydrodynamics(self,
                              op_point: OperatingPoint,
                              method: str = 'dsyhs',
                              ):
        
        Fhull, Mhull = self.compute_resistance_force(op_point, 
                                                     method=method)
        Fdrift, Mdrift = self.compute_lateral_force(op_point,
                                                    method=method)

        return Fhull + Fdrift, Mhull + Mdrift
        
    def compute_windage_area(self,
                             awa: float = 0., # degrees
                             leeway: float = 0., # degrees
                             method: str = 'fossati'
                             ):
        """
        Compute the effective exposed windage area.
        
        Adapted from Fossati (eq. 5.20 and 5.21) using (Ueno et al., 2012) (Fig. 5)

        """
        
        Ax = self.aerostaticData['Ax']
        Ay = self.aerostaticData['Ay']
        
        if method == 'fossati':
        
            # b0 = np.degrees(np.arctan(BOA/LOA)) # Fossati (eq. 5.19)
            b0 = (np.arctan(self.Boa/self.Loa)*u.deg) * 2 # modified from Fossati (eq. 5.19) using (Ueno et al., 2012) (fig. 5)
            
            k = .1
            b0fac = 1 / (1 + np.exp(-(awa-b0)*k)) * 1 / (1 + np.exp((awa-(180 - b0))*k))
            
            return Ay * np.sind(awa) * b0fac + Ax * (1-b0fac) # modified from Fossati (eq. 5.20 and 5.21) using (Ueno et al., 2012) (fig. 5) for continuous differenciation
        
        elif method == 'richeux':
            return Ax*np.cosd(awa-leeway) + Ay*np.sind(awa-leeway)
            
        else:
            raise ValueError('Unsupported method. Should be "fossati" or "richeux".')
        
        
    def compute_wind_loads(self,
                           op_point: OperatingPoint,
                           windage_cd: float = 0.65,
                           recompute_statics: bool = False,
                           method='fossati'):
        """
        Compute the wind loads on the hull superstructures for a given operating point.
        
        From Fossati (eq. 5.22)
        
        "Cr usually between 0.6 and 0.7 for monohull sailing yachts", Fossati (p.124)

        """
        
        if not(self.aerostaticData) or recompute_statics:
            self.compute_statics(op_point)
            
        center = self.aerostaticData['cdyn'] * np.array([-1., -1., 1.])
        
        z = wide(center)[0,2]
        
        aws = op_point._aws(z) # in m/s
        awa = op_point.awa(z) # in degrees
        
        leeway = op_point.leeway
        
        area = self.compute_windage_area(awa, leeway, method=method)
        
        rho = self.environment.air.density
        
        if method == 'fossati':
            
            FtotAA = 1/2 * rho * windage_cd * area * aws**2  # Fossati (eq. 5.18)
            
            Faa = np.array([- FtotAA * np.cosd(awa),
                            FtotAA * np.sind(awa),
                            0.]) # Fossati (eq. 5.22)
            
            Maa = np.cross(center, Faa)
            
            return Faa, Maa
        
        elif method == 'richeux':
            
            CL = np.sind(2*awa-leeway)
            CD = 0.2 - awa/450. + np.sind(awa-leeway)
            
            CX = CL * np.sind(awa) - CD * np.cosd(awa)
            CY = CL * np.cosd(awa) + CD * np.sind(awa)
            
            Fx = 0.5 * rho * area * CX * aws**2
            Fy = 0.5 * rho * area * CY * aws**2
            
            Faa = np.array([Fx,
                            Fy,
                            0.])
            
            Maa = np.cross(center, Faa)
            
            return Faa, Maa
            
        else:
            raise ValueError('Unsupported method. Should be "fossati" or "richeux".')
            
        
    def compute_crew_resistance(self,
                                op_point: OperatingPoint,
                                n_crew: int = 0,
                                center: Union[float, np.ndarray] = None,
                                crew_area: float = 1., # m2
                                crew_cd: float = 0.9,
                                ):
        if center is None:
            center = self.cog.copy()
            
        z = center[2] # TODO: make the crew center move with the ship
        aws = op_point._aws(z) # in m/s
        awa = op_point.awa(z) # in degrees

        FtotCrew = 0.5 * 1.225 * crew_cd * crew_area * n_crew * aws**2

        Fcrew = np.array([-FtotCrew * np.cosd(awa),
                          FtotCrew * np.sind(awa),
                          0.])
        
        Mcrew = np.cross(center, Fcrew)
        
        return Fcrew, Mcrew
    
    
    def compute_statics(self,
                        op_point: OperatingPoint):
        
        leeway = op_point.leeway
        
        antiLeewayRot = np.rotation_matrix_3D((-leeway)*np.pi/180, 'z')
        
        vertices = copy.copy(self.diff_mesh.vertices)
        # vertices[:,0] *= -1
        faces = copy.copy(self.diff_mesh.faces)

        vertices = vertices @ antiLeewayRot

        point = np.array([0.0, 0.0, 0.0])

        initMesh = DifferentiableMesh(vertices, faces)

        initVertDist = initMesh.vertices_distances_to_plane(point, 'z')
        initFaceDist = initMesh.faces_distances_to_plane(point, 'z')

        # TODO: continuously handle fully wet and fully dry cases

        # NB
        # if np.sum(ramp(dist)) == 0 : fully immersed
        # if np.prod(ramp(dist)) == 1 : fully emerged
        
        # fully_immersed = ca.if_else(np.abs(np.sum(ramp(initVertDist))) < 1e-5, True, False)
        # fully_emerged = ca.if_else(np.abs(np.prod(ramp(initVertDist)) - 1.) < 1e-5, True, False)
        
        # if fully_immersed:
        #     ### HYDROSTATICS COMPUTATION
            
        #     volume = initMesh.volume
        #     cob = initMesh.volume_centroid
            
        #     Wsa = initMesh.area
        #     cow = initMesh.area_centroid
            
        #     Ax = initMesh.frontal_area('x')
        #     Ay = initMesh.frontal_area('y')
            
        #     bounds = initMesh.bounds
            
        #     X0 = bounds[0,0]
        #     T = bounds[2,1] - bounds[2,0]
            
        #     Ax = initMesh.frontal_area('x')
        #     Ay = initMesh.frontal_area('y')
            
        #     dl = (bounds[0,1] - bounds[0,0])/100.
            
        #     transomMesh = initMesh.compress_mesh(np.array([1.,0.,0.])*(X0+dl), 'x',
        #                                          tol=1e-5, scale_fac=10.)
            
        #     Atr = transomMesh.frontal_area('x')
        #     # Atr = compute_volume_and_center_of_mass(transom, faces)[0] / dl # alternate approximate way to compute Atr
            
        #     z_min = initMesh.bounds[2,0]
        
        #     Ttr = wide(point)[0,2] - z_min
            
        #     self.hydrostaticData = {}
            
        #     self.hydrostaticData['volume'] = initMesh.volume
            
        #     points = {'cob': cob, 'cof': cob*0., 'cow': cow, '0L': X0}
        #     lengths = {'Lwl': 0., 'Bwl': 0., 'T': T, 'Ttr': Ttr, '0L': X0}
        #     areas = {'Wsa': Wsa, 'Wpa': 0., 'Ax': Ax, 'Ay': Ay, 'Atr': Atr, 'Abt': 0.}
        #     coefs = {'Cb': 1., 'Cp': 1., 'Cwp': 1., 'Cx': 1., 'Cy': 1.}
        
        #     self.hydrostaticData |= points | lengths | areas | coefs
            
        #     self.hydrostaticData['ie'] = None
            
        #     ### AEROSTATICS COMPUTATION
            
        #     points = {'caa': cob*0.}
        #     areas = {'Dsa': 0., 'Ax': 0., 'Ay': 0.}
            
        #     self.aerostaticData = {}
            
        #     self.aerostaticData |= points | areas

        # elif fully_emerged:
        #     ### AEROSTATICS COMPUTATION
            
        #     dryBounds = initMesh.bounds
            
        #     Taa = dryBounds[2,1] - wide(point)[0,2]
        #     Axaa = initMesh.frontal_area('x')
        #     Ayaa = initMesh.frontal_area('y')
        #     Dsa = initMesh.weighted_area(weight=ramp(initFaceDist-1e-3, tol=1e-3))
            
        #     caa = initMesh.area_centroid
            
        #     points = {'caa': caa}
        #     areas = {'Dsa': Dsa, 'Ax': Axaa, 'Ay': Ayaa}
            
        #     self.aerostaticData = {}
            
        #     self.aerostaticData |= points | areas
            
        #     ### HYDROSTATICS COMPUTATION
            
        #     self.hydrostaticData = {}
            
        #     self.hydrostaticData['volume'] = 0.
            
        #     points = {'cob': caa*0., 'cof': caa*0., 'cow': caa*0., '0L': caa*0.}
        #     lengths = {'Lwl': 0., 'Bwl': 0., 'T': 0., 'Ttr': 0., '0L': 0.}
        #     areas = {'Wsa': 0., 'Wpa': 0., 'Ax': 0., 'Ay': 0., 'Atr': 0., 'Abt': 0.}
        #     coefs = {'Cb': 0., 'Cp': 0., 'Cwp': 0., 'Cx': 0., 'Cy': 0.}
        
        #     self.hydrostaticData |= points | lengths | areas | coefs
            
        #     self.hydrostaticData['ie'] = None
        
        # else:
        if True:
            ### HYDROSTATICS COMPUTATION
            
            wetMesh = initMesh.compress_mesh(point, 'z', dist=initVertDist,
                                             tol=1e-5, scale_fac=2.)
            
            self.wet_mesh = wetMesh
        
            dryMesh = initMesh.compress_mesh(point, '-z', dist=-initVertDist,
                                             tol=1e-5, scale_fac=2.)
            
            self.dry_mesh = dryMesh
            
            waterplaneMesh = wetMesh.compress_mesh(point, '-z', dist=-initVertDist,
                                                   tol=1e-5, scale_fac=1000.)
            
            self.waterplane = waterplaneMesh
            
            # transomMesh = wetMesh.compress_mesh()
            
            volume = wetMesh.volume
            cob = wetMesh.volume_centroid
            
            # Wsa = wetMesh.weighted_area(weight=ramp(initFaceDist))
            cow = wetMesh.weighted_area_centroid(weight=ramp(initFaceDist, tol=1e-5))
            
            # Wpa = waterplaneMesh.weighted_area(weight=ramp(initFaceDist))
            cof = waterplaneMesh.weighted_area_centroid(weight=ramp(initFaceDist, tol=1e-5))
            # cof = waterplaneMesh.area_centroid
            
            dryBounds = dryMesh.bounds
            wetBounds = wetMesh.bounds
            
            Laa = dryBounds[0,1] - dryBounds[0,0]
            
            X0 = wetBounds[0,0]
            Lwl = wetBounds[0,1] - wetBounds[0,0]
            Bwl = wetBounds[1,1] - wetBounds[1,0]
            T = wetBounds[2,1] - wetBounds[2,0]
            
            cdyn = cow - np.array([Lwl/4., 0., 0.]) # center of pressure condidered at 25% chord
            
            Ax = wetMesh.frontal_area('x')
            Ay = wetMesh.frontal_area('y')
            Wpa = waterplaneMesh.frontal_area('z')
            
            Wsa = wetMesh.area - Wpa
            
            dl = Lwl/1000.
            
            transomMesh = wetMesh.compress_mesh(np.array([1.,0.,0.])*(X0+dl), 'x',
                                                tol=1e-5, scale_fac=1000.)
            
            self.transom_mesh = transomMesh
            
            Atr = transomMesh.frontal_area('x')
            # Atr = compute_volume_and_center_of_mass(transom, faces)[0] / dl # alternate approximate way to compute Atr
            
            v = transomMesh.vertices
            
            z_min = np.min(v[:,2])
            z_max = np.max(v[:,2])
        
            Ttr = z_max - z_min
            
            Cb = np.clip(volume / (Lwl * Bwl * T + 1e-8), 0., 1.)
            Cp = np.clip(volume / (Ax * Lwl + 1e-8), 0., 1.)
            Cx = np.clip(Ax / (Bwl * T + 1e-8), 0., 1.)
            Cy = np.clip(Ay / (Lwl * T + 1e-8), 0., 1.)
            Cwp = np.clip(Wpa / (Lwl * Bwl + 1e-8), 0., 1.)
            
            cob = rotate_single_vector(cob, antiLeewayRot.T, np.zeros(3)) # bring back Cob in underway axes
            cof = rotate_single_vector(cof, antiLeewayRot.T, np.zeros(3)) # bring back Cof in underway axes
            cow = rotate_single_vector(cow, antiLeewayRot.T, np.zeros(3)) # bring back Cow in underway axes
            cdyn = rotate_single_vector(cdyn, antiLeewayRot.T, np.zeros(3)) # bring back Cow in underway axes
            
            self.hydrostaticData = {}
            
            self.hydrostaticData['volume'] = volume
            
            points = {'cob': cob, 'cof': cof, 'cow': cow, 'cdyn': cdyn, '0L': X0}
            lengths = {'Lwl': Lwl, 'Bwl': Bwl, 'T': T, 'Ttr': Ttr, '0L': X0}
            areas = {'Wsa': Wsa, 'Wpa': Wpa, 'Ax': Ax, 'Ay': Ay, 'Atr': Atr, 'Abt': 0.}
            coefs = {'Cb': Cb, 'Cp': Cp, 'Cwp': Cwp, 'Cx': Cx, 'Cy': Cy}
        
            self.hydrostaticData |= points | lengths | areas | coefs
            
            self.hydrostaticData['ie'] = None
            
            ### AEROSTATICS COMPUTATION
            
            dryBounds = dryMesh.bounds
            
            Taa = dryBounds[2,1] - dryBounds[2,0]
            Axaa = dryMesh.frontal_area('x')
            Ayaa = dryMesh.frontal_area('y')
            Dsa = dryMesh.weighted_area(weight=ramp(initFaceDist-1e-3, tol=1e-3))
            
            caa = dryMesh.area_centroid
            cdyn = caa - np.array([Laa/4., 0., 0.]) # center of pressure condidered at 25% chord
            
            caa = rotate_single_vector(caa, antiLeewayRot.T, np.zeros(3)) # bring back Caa in underway axes
            cdyn = rotate_single_vector(cdyn, antiLeewayRot.T, np.zeros(3))
            
            points = {'caa': caa, 'cdyn': cdyn}
            lengths = {'Laa': Laa, 'Taa': Taa}
            areas = {'Dsa': Dsa, 'Ax': Axaa, 'Ay': Ayaa}
            
            self.aerostaticData = {}
            
            self.aerostaticData |= points | lengths | areas
            
    
    def compute_static_data(self,
                            point,
                            mat,
                            xn,
                            yn,
                            normal,
                            ):

        vdist = self.diff_mesh.vertices_distances_to_plane(point, normal)
        fdist = self.diff_mesh.faces_distances_to_plane(point, normal)
        
        dryVertFac = np.fmax(np.fmin(+vdist-0.5, 1.), 0.)
        wetVertFac = np.fmax(np.fmin(-vdist+0.5, 1.), 0.)
        
        dryFaceFac = np.fmax(np.fmin(+fdist-0.5, 1.), 0.)
        wetFaceFac = np.fmax(np.fmin(-fdist+0.5, 1.), 0.)
        
        # mat = np.vstack((xn, yn, normal))


        ### HYDROSTATICS COMPUTATION
        
        # wetMesh = initMesh.compress_mesh(point, 'z', dist=initVertDist,
        #                                  tol=1e-5, scale_fac=2.)
        
        # self.wet_mesh = wetMesh
    
        # dryMesh = initMesh.compress_mesh(point, '-z', dist=-initVertDist,
        #                                  tol=1e-5, scale_fac=2.)
        
        # self.dry_mesh = dryMesh
        
        # waterplaneMesh = wetMesh.compress_mesh(point, '-z', dist=-initVertDist,
        #                                        tol=1e-5, scale_fac=1000.)
        
        # self.waterplane = waterplaneMesh
        self.waterplane = self.diff_mesh.slice_mesh(point, normal)
        
        self.waterplane._vertices = (self.waterplane.vertices - point) @ mat
        self.waterplane.reset_data()
        
        # transomMesh = wetMesh.compress_mesh()
        
        volume, cob = self.diff_mesh.hydrostatics(point, normal)
        
        # Wsa = wetMesh.weighted_area(weight=ramp(initFaceDist))
        cow = self.diff_mesh.weighted_area_centroid(weight=wetFaceFac)
        
        # Wpa = waterplaneMesh.weighted_area(weight=ramp(initFaceDist))
        cof = self.waterplane.area_centroid
        # cof = waterplaneMesh.area_centroid
        
        # dryBounds = dryMesh.bounds
        # wetBounds = wetMesh.bounds
        
        bounds = self.waterplane.bounds
        
        X0 = bounds[0,0] + (point @ mat)[0]
        Lwl = bounds[0,1] - bounds[0,0]
        Bwl = bounds[1,1] - bounds[1,0]
        T = -np.min(vdist)
        
        Laa = Lwl
        
        cdyn = cow - np.array([Lwl/4., 0., 0.]) # center of pressure condidered at 25% chord
        
        Ax = self.diff_mesh.frontal_area(xn, weight=wetFaceFac)
        Ay = self.diff_mesh.frontal_area(yn, weight=wetFaceFac)
        Wpa = self.waterplane.area
        
        Wsa = self.diff_mesh.weighted_area(weight=wetFaceFac)
        
        dl = Lwl/10.
        
        transom_point = np.array([1.,0.,0.])*(X0+dl)
        self.transom_mesh = self.diff_mesh.slice_mesh(
                point=transom_point,
                normal=xn,
            )
        self.transom_mesh._vertices = (self.waterplane.vertices - point) @ mat
        self.transom_mesh.reset_data()
        
        Atr = self.transom_mesh.area
        # Atr = compute_volume_and_center_of_mass(transom, faces)[0] / dl # alternate approximate way to compute Atr
        
        trBounds = self.transom_mesh.bounds
    
        Ttr = trBounds[2,1] - trBounds[2,0]
        
        Atr = 0.
        Ttr = 0.
        
        Cb = np.clip(volume / (Lwl * Bwl * T + 1e-8), 0., 1.)
        Cp = np.clip(volume / (Ax * Lwl + 1e-8), 0., 1.)
        Cx = np.clip(Ax / (Bwl * T + 1e-8), 0., 1.)
        Cy = np.clip(Ay / (Lwl * T + 1e-8), 0., 1.)
        Cwp = np.clip(Wpa / (Lwl * Bwl + 1e-8), 0., 1.)
        
        # cob = rotate_single_vector(cob, antiLeewayRot.T, np.zeros(3)) # bring back Cob in underway axes
        # cof = rotate_single_vector(cof, antiLeewayRot.T, np.zeros(3)) # bring back Cof in underway axes
        # cow = rotate_single_vector(cow, antiLeewayRot.T, np.zeros(3)) # bring back Cow in underway axes
        # cdyn = rotate_single_vector(cdyn, antiLeewayRot.T, np.zeros(3)) # bring back Cow in underway axes
        
        self.hydrostaticData = {}
        
        self.hydrostaticData['volume'] = volume
        
        points = {'cob': cob, 'cof': cof, 'cow': cow, 'cdyn': cdyn, '0L': X0}
        lengths = {'Lwl': Lwl, 'Bwl': Bwl, 'T': T, 'Ttr': Ttr, '0L': X0}
        areas = {'Wsa': Wsa, 'Wpa': Wpa, 'Ax': Ax, 'Ay': Ay, 'Atr': Atr, 'Abt': 0.}
        coefs = {'Cb': Cb, 'Cp': Cp, 'Cwp': Cwp, 'Cx': Cx, 'Cy': Cy}
    
        self.hydrostaticData |= points | lengths | areas | coefs
        
        self.hydrostaticData['ie'] = None
        
        ### AEROSTATICS COMPUTATION
        
        Taa = np.max(vdist)
        Axaa = self.diff_mesh.frontal_area(xn, weight=dryFaceFac)
        Ayaa = self.diff_mesh.frontal_area(yn, weight=dryFaceFac)
        Dsa = self.diff_mesh.weighted_area(weight=dryFaceFac)
        
        caa = self.diff_mesh.weighted_area_centroid(weight=dryFaceFac)
        cdyn = caa - np.array([Laa/4., 0., 0.]) # center of pressure condidered at 25% chord
        
        # caa = rotate_single_vector(caa, antiLeewayRot.T, np.zeros(3)) # bring back Caa in underway axes
        # cdyn = rotate_single_vector(cdyn, antiLeewayRot.T, np.zeros(3))
        
        points = {'caa': caa, 'cdyn': cdyn}
        lengths = {'Laa': Laa, 'Taa': Taa}
        areas = {'Dsa': Dsa, 'Ax': Axaa, 'Ay': Ayaa}
        
        self.aerostaticData = {}
        
        self.aerostaticData |= points | lengths | areas
        
        print(self.hydrostaticData)
            
                
    def compute_buoyancy(self,
                         op_point: OperatingPoint = OperatingPoint(),
                         recompute_statics: bool = True
                         ):
        
        if not(self.hydrostaticData) or recompute_statics:
            self.compute_statics(op_point)
        
        center = self.hydrostaticData['cob'] * np.array([-1., -1., 1.])
        # center[1] *= -1 # TODO: find why lateral inversion is necessary. Heel rotation seems not consistent
        volume = self.hydrostaticData['volume']
        rho = op_point.environment.water.density
        g = op_point.environment.gravity
        
        Fb = np.array([0.,
                       0.,
                       volume*rho*g])
        
        Mb = np.cross(center, Fb)
        
        return Fb, Mb
    
#%% TEST
if __name__ == '__main__':
    
    import archibald2 as asb
    from archibald2.performance.operating_point import OperatingPoint
    
    opti = asb.Opti()
    
    archibald_root = 'C:/Users/jrich/NEOLINE DEVELOPPEMENT/NeoDev - ND-DÉVELOPPEMENT - Documents/ND-DÉVELOPPEMENT/01_Développement/08_Outils'
    # hullStl = archibald_root+'/archibald-main/Private/n136_data/n136.stl' # path to a STL mesh of the hull
    hullStl = archibald_root+'/archibald-main/Private/n136_data/n136_quad.stl' # path to a STL mesh of the hull
    hullMesh = trimesh.load(hullStl)
    
    hullMesh.fix_normals()
    
    diff_mesh = DifferentiableMesh(hullMesh.vertices, hullMesh.faces)

    rho = 1025.
    # displacement = 290. # kg
    
    displacement = 13.1e6 # kg
    
    lcg = 65.25
    tcg = 0.0
    vcg = 8.8

    dz = 0.
    # dz = opti.variable(init_guess=-5.3)
    stw = 8.
    leeway = 0.
    heel = 0.
    trim = 0.
    
    hull = Hull('forty', displacement, np.zeros(3),
                vertices=hullMesh.vertices, faces=hullMesh.faces,
                inv_x=False)
    
    op_point = OperatingPoint(
             stw = stw,
             heel=heel,
             trim=trim,
             leeway=leeway,
             immersion=dz,
             )
    
    hull.diff_mesh.vertices = hull.diff_mesh.vertices @ np.rotation_matrix_3D(heel, 'x')
    
    hull.diff_mesh.vertices = np.add(hull.diff_mesh.vertices, wide(np.array([0.,0.,dz])))
    
    hull.compute_statics(op_point)
    
    obj = (hull.hydrostaticData['volume']*op_point.environment.water.density - displacement)**2
    # obj = (hull.hydrostaticData['cob'][2] + .1)**2
    # obj = (hull.hydrostaticData['Wsa'] - 3.)**2
    # obj = (hull.hydrostaticData['Wpa'] - 1.)**2
    # obj = (hull.aerostaticData['caa'][2] + .05)**2
    # obj = (hull.hydrostaticData['cof'][0] -2.)**2
    # obj = (hull.hydrostaticData['cow'][0] - 2.)**2
    opti.minimize(obj)
    
    sol = opti.solve(
        max_iter=50,
        )
    