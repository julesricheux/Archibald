# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 14:59:26 2024

@author: jrich
"""

import os

from archibald import _archibald_root
from archibald.modeling import InterpolatedModel
from archibald.toolbox.math_utils import read_coefs, rotate_single_vector
import archibald.numpy as np
# from archibald.geometry.airfoil import Airfoil

from typing import Union, List

import copy

"""
According to https://deepblue.lib.umich.edu/bitstream/handle/2027.42/91702/Publication_No_237.pdf?sequence=1

Evaluation ranges :
2 <= Z <= 7
0.30 <= Ae/Ao <= 1.05
0.50 <= P/D <= 1.40

"""

def tall(array):
    return np.reshape(array, (-1, 1))


def wide(array):
    return np.reshape(array, (1, -1))


def b_series_polynomial(J, P_D, Ae_Ao, Z, c, stuv):
    
    is_casadi = np.is_casadi_type(J) or \
                np.is_casadi_type(P_D) or \
                np.is_casadi_type(Ae_Ao) or \
                np.is_casadi_type(Z) or \
                np.is_casadi_type(c) or \
                np.is_casadi_type(stuv)
    
    if isinstance(J, (int, float, np.number)):
        J = np.array([J])
        
    elif isinstance(J, (list, tuple)):
        J = np.array(J)
        
    JMat = tall(copy.copy(J))
    
    propertyMat = np.ones((JMat.shape[0], 1)) @ wide(np.array([P_D, Ae_Ao, Z]))
        
    if is_casadi:
        paramMat = np.stack([JMat] + [propertyMat[:,i] for i in range(3)], axis=1)
        
        k = []
        
        for i in range(JMat.shape[0]):
            line = np.ones((stuv.shape[0], 1)) @ wide(paramMat[i,:])
            
            powMat = np.power(line, stuv)
            
            prodMat = np.prod(powMat, axis=1)
            
            k.append(np.dot(prodMat, tall(c)))
            
    else:
        # Reshape a to be of shape (n, 4, 1)
        paramMat = np.stack([JMat] + [tall(propertyMat[:,i]) for i in range(3)], axis=1)
        
        # Perform element-wise exponentiation of a and b
        powMat = np.power(paramMat, np.transpose(stuv))
        
        # Compute the product along the second axis
        prodMat = np.prod(powMat, axis=1)
        
        k = np.dot(prodMat, tall(c)).T[0]
    
    return np.array(k)


def b_series_sections(r_R, tMax,
                      V1fun, V2fun,
                      tLE=1e-3, tTE=1e-3,
                      n=200, disp=False):
    
    nSections = len(r_R)
    
    xtMax = 0.4
    
    # Computing abs positions from required resolution
    nLead = int(n*xtMax/2)
    nTrail = n//2 - nLead
    
    Pneg = np.linspace(-1., 0., nTrail)
    Ppos = np.linspace(1/nLead, 1., nLead)
    
    P = np.hstack((Pneg, Ppos))
    
    rr, PP = np.meshgrid(r_R, P)
    rrFlat = rr.ravel()
    PPflat = PP.ravel()
    
    # V1 and V2 interpolation
    V1 = V1fun(rrFlat, PPflat)
    V2 = V2fun(rrFlat, PPflat)
    
    V1neg = V1[:nTrail*nSections]
    V2neg = V2[:nTrail*nSections]
    
    V1pos = V1[nTrail*nSections:]
    V2pos = V2[nTrail*nSections:]
    
    # X-coordinates computation
    xLead = xtMax*(1 - Ppos)
    xTrail = (1-xtMax)*(xtMax/(1-xtMax) - Pneg)
    
    xUpper = np.hstack((xTrail, xLead))
    xLower = np.hstack((np.flip(xLead), np.flip(xTrail)))
    xSection = np.hstack((xUpper, 
                          xLower))
    
    AF = []
    
    for iSection in range(nSections):
        # Intrados / Lower side
        yyFaceTrail = V1neg*(tMax[iSection] - tTE) # for p <= 0
        yyFaceLead = V1pos*(tMax[iSection] - tLE) # for p > 0
        
        # Extrados / Upper side
        yyBackTrail = (V1neg + V2neg)*(tMax[iSection] - tTE) + tTE # for p <= 0
        yyBackLead = (V1pos + V2pos)*(tMax[iSection] - tLE) + tLE # for p > 0
        
        # Intrados / Lower side
        yFaceTrail = yyFaceTrail[iSection::nSections]
        yFaceLead = yyFaceLead[iSection::nSections]
        
        # Extrados / Upper side
        yBackTrail = yyBackTrail[iSection::nSections]
        yBackLead = yyBackLead[iSection::nSections]
        
        yUpper = np.hstack((yBackTrail,
                           yBackLead))
        
        yLower = np.hstack((np.flip(yFaceLead), 
                           np.flip(yFaceTrail)))
        
        ySection = np.hstack((yUpper, 
                              yLower))
        
        coords = np.transpose(np.vstack((xSection, ySection)))
        
        AF.append(Airfoil(name='B-SERIES-'+str(r_R[iSection]),
                         coordinates=coords))
        
        if disp:
            AF[-1].draw()
            
    return AF


def write_airfoil_polars(airfoil, aMin=-180., aMax=180., n=361, Re=1e6,
                         directory='./pyBEMT/pybemt/airfoils/', suffix=''):
    
    af = airfoil.copy()
    
    if suffix=='-gen':
        af.coordinates[:,1] = -af.coordinates[:,1]
        
        np.flip(af.coordinates, axis=0)
    
    alpha = np.linspace(aMin, aMax, n)
    data = af.get_aero_from_neuralfoil(alpha, Re)
    
    with open(directory+af.name+suffix+'.dat', 'w') as file:
        for i in range(13):
            file.write("---\n")
        file.write("Alpha CL CD\n")
        for a, cl, cd in zip(alpha, data['CL'], data['CD']):
            file.write(f"{a} {cl} {cd}\n")


class Propeller():
    def __init__(self,
                 name : str = 'Untitled_propeller',
                 xyz_center : Union[np.ndarray, List] = None,
                 axis : Union[np.ndarray, List] = None,
                 Z : int = 2,
                 Ae_Ao : float = 1.,
                 D : float = 1.,
                 P_D : float = 1.):
        
        if xyz_center is None:
            xyz_center = np.zeros(3)
            
        if axis is None:
            axis = np.array([1., 0., 0.])
        
        self.name = name
        self._center = xyz_center
        self._center_0 = copy.copy(xyz_center)
        self._axis = axis
        self._axis_0 = copy.copy(axis)
        self._Z = Z
        self._Ae_Ao = Ae_Ao
        self._D = D
        self._R = D/2
        self._P_D = P_D
        self._P = np.degrees(np.arctan(self._P_D))
        
    @property
    def center(self):
        return self._center
    
    @property
    def axis(self):
        return self._axis
    
    @property
    def blades_number(self):
        return self._Z
        
    @property
    def diameter(self):
        return self._D
        
    @property
    def radius(self):
        return self._R
        
    @property
    def blade_area_ratio(self):
        return self._Ae_Ao
        
    @property
    def pitch_ratio(self):
        return self._P_D
    
    @property
    def pitch_angle(self):
        return self._P
    
    @property
    def area(self):
        return np.pi * self._R**2
    
    def _reset(self):
        """
        Cancel all previous transformations.

        """
        self._center = copy.copy(self._center_0)
        self._axis = copy.copy(self._axis_0)
        
    def reset(self):
        """
        Cancel all previous transformations.

        """
        self._reset()
    
    @center.setter
    def center(self, xyz):
        xyz = np.array(xyz)
        if np.array(xyz).shape not in [3, (3,), (1,3), (3,1)]:
            raise ValueError("Passed array is not of the right shape. Should be a 3D point.")
        self._center = xyz
        
    def translate(self, xyz):
        xyz = np.array(xyz)
        if xyz.shape not in [3, (3,), (1,3), (3,1)]:
            raise ValueError("Passed array is not of the right shape. Should be a 3D point.")
        self._center += xyz
        
    def _global_rotation_from_matrix(self,
                                     rot_mat: np.ndarray = None,
                                     center: Union[np.ndarray, List[float]] = None,
                                     ):
        """
        Rotate the propeller from the given rotation matrix and center point.
        
        """
        if center is None:
            center = self._center.copy()
        self._center = rotate_single_vector(self._center, rot_mat, center)
        self._axis = rotate_single_vector(self._axis, rot_mat, center)
        
    def global_rotation(self,
                        angle: float,
                        axis: Union[np.ndarray, List[float], str],
                        center: Union[np.ndarray, List[float]] = None,
                        matrix: np.ndarray = None,
                        ):
        """
        Rotate the propeller from the given angle/axis couple or matrix, and center point.

        """
        if matrix is None:
            matrix = np.rotation_matrix_3D((angle)*np.pi/180., axis)
        
        self._global_rotation_from_matrix(matrix, center)
        
    def local_rotation(self,
                       angle: float,
                       axis: Union[np.ndarray, List[float], str],
                       matrix: np.ndarray = None,
                       ):
        """
        Rotate the leading edge from the given angle, axis and center point.

        """
        if matrix is None:
            matrix = np.rotation_matrix_3D((angle)*np.pi/180., axis)
        
        self._global_rotation_from_matrix(matrix, None)


class BSeriesPropeller(Propeller):
    def __init__(self,
                 name : str = 'Untitled_Bseries_propeller',
                 xyz_center : Union[np.ndarray, List] = None,
                 axis : Union[np.ndarray, List] = None,
                 Z : int = 3,
                 Ae_Ao : float = 1.,
                 D : float = 1.,
                 P_D : float = 1.):
        
        super().__init__(name, xyz_center, axis, Z, Ae_Ao, D, P_D)
        
        _data_path = os.path.join(_archibald_root, "data", "propeller")
        ktDatapath = os.path.join(_data_path, "b-series-kt.csv")
        kqDatapath = os.path.join(_data_path, "b-series-kq.csv")
        V1Datapath = os.path.join(_data_path, "b-series-v1.csv")
        V2Datapath = os.path.join(_data_path, "b-series-v2.csv")
        geometryDatapath = os.path.join(_data_path, "b-series-geometry.csv")
                
        ktData = read_coefs(ktDatapath, delim='\t', skipRows=2)
        kqData = read_coefs(kqDatapath, delim='\t', skipRows=2)
        
        if self._Z == 3:
            _coefs = read_coefs(geometryDatapath, delim='\t',
                                skipRows=2, columns=[0,4,5,6,7,8,9]).T
            
        elif 4 <= self._Z <= 7:
            _coefs = read_coefs(geometryDatapath, delim='\t',
                                skipRows=2, columns=[0,1,2,3,7,8,9]).T
            
        else:
            raise ValueError("Number of blades is "+str(self._Z)+" but should be between 3 and 7")
        
        # _interp = build_interpolation(_coefs, method='cubic') LEGACY
        geometryKeys = ['c/D*Z/(AE/AO)', 'a/c', 'b/c', 'Ar', 'Br', 'pitch factor']
        geometryData = {}
        
        for i, k in enumerate(geometryKeys):
            # geometryData[k] = _interp[i] LEGACY
            geometryData[k] = InterpolatedModel(
                x_data_coordinates=_coefs[0, :],
                y_data_structured=_coefs[i+1, :],
                method="bspline",
            )
        
        """
        geometryData['V1'] = build_2D_interpolator_from_csv(V1Datapath)
        geometryData['V2'] = build_2D_interpolator_from_csv(V2Datapath)
        """
        self._geometryData = geometryData
        
        self._cT = ktData[:, 0]
        self._stuvT = ktData[:, 1:].astype('int32')
        
        self._cQ = kqData[:, 0]
        self._stuvQ = kqData[:, 1:].astype('int32')
        
        
    def chord_law(self, r):
        normLaw = self._geometryData['c/D*Z/(AE/AO)'](r)
        
        return normLaw * self._D * (self._Ae_Ao) / self._Z
    
    def sweep_law(self, r, chordLaw=None):
        if chordLaw is None:
            chordLaw = self.chord_law(r)
        normLaw = self._geometryData['a/c'](r)
        
        return normLaw * chordLaw
    
    def pitch_law(self, r):
        normLaw = self._geometryData['pitch factor'](r)
        
        return normLaw
    
    def thickness_law(self, r, chordLaw=None):
        if chordLaw is None:
            chordLaw = self.chord_law(r)
        thicknessLaw = self._D * (self._geometryData['Ar'](r) - self._Z * self._geometryData['Br'](r)) / chordLaw
        
        return thicknessLaw
    
    
    def compute_geometric_laws(self, r=np.linspace(0.0+1e-3, 1.-1e-3, 10)):
        chordLaw = self.chord_law(r)
        sweepLaw = self.sweep_law(r, chordLaw)
        pitchLaw = self.pitch_law(r)
        thicknessLaw = self.thickness_law(r, chordLaw)
        
        return chordLaw, sweepLaw, pitchLaw, thicknessLaw
    
    # EXPERIMENTAL
    def build_blade_sections(self, r, thicknessLaw=None, chordLaw=None):
        if thicknessLaw is None:
            if not chordLaw is None:
                chordLaw = self.chord_law(r)
            thicknessLaw = self.thickness_law(r, chordLaw)
            
        return b_series_sections(r, self.thickness_law(r),
                                 self._geometryData['V1'], self._geometryData['V2'], disp=False)
    
    # EXPERIMENTAL
    def write_sections_polars(self, r=None, sections=None, thicknessLaw=None, chordLaw=None, suffix=''):
        if r or sections:
            if sections is None:
                sections = self.build_blade_sections(r, thicknessLaw, chordLaw)
            for s in sections:
                write_airfoil_polars(s, suffix=suffix)
        else:
            print("Could not write sections polars. No radius nor sections were specified.")
    
    # EXPERIMENTAL
    def write_pyBEMT_input(self, nSections=10,
                           directory='./pyBEMT/pybemt/airfoils/', generator=False):
        
        r = np.linspace(0.20, 0.999, nSections)
        nSections = len(r)
        
        if generator:
            suffix = '-gen'
        else:
            suffix = ''
        
        chords = self.chord_law(r)
        pitches = self.pitch_angle * self.pitch_law(r) + 90 * (1-self.pitch_law(r))
        sections = self.build_blade_sections(r, chordLaw=chords)
        self.write_sections_polars(sections=sections, suffix=suffix)
        
        filename = directory+self.name+suffix+'.ini'
        
        with open(filename, 'w') as file:
            file.write("[case]\n")
            file.write("rpm = 100\n")
            file.write("v_inf = 5.144\n")
            
            if generator:
                file.write("\n[turbine]\n")
            else:
                file.write("\n[rotor]\n")
                
            file.write(f"nblades = {self._Z}\n")
            file.write(f"diameter = {self._D}\n")
            file.write(f"radius_hub = {r[0]}\n")
            
            file.write("section = ")
            
            for i in range(nSections):
                file.write(f" {sections[i].name}{suffix}")
            
            file.write("\nradius = ")
            for i in range(nSections):
                file.write(f" {r[i]}")
            
            file.write("\nchord = ")
            for i in range(nSections):
                file.write(f" {chords[i]}")
            
            file.write("\npitch = ")
            for i in range(nSections):
                file.write(f" {pitches[i]}")
                
            file.write("\n\n[fluid]\n")
            
            file.write("rho = 1025.0\n")
            file.write("mu = 1.0e-3\n")
            file.write("pvap = 1700\n")
            file.write("patm = 101325\n")
            file.write("depth = 1000.\n")
            
        return filename
       
    # TODO rewrite for to Archibald
    # def build_aerosanbox_geometry(self, nSections=10):
        
    #     r = np.linspace(0.20, .99, nSections)
    #     # r = r * 0.85 + 0.15 + 1e-4
        
    #     # r = np.linspace(0.01, .999, nSections)
        
    #     chords, sweeps, pitchFac, thicknessLaw = self.compute_geometric_laws(r)
        
    #     nBlades = self._Z
    #     rHub = 0.4
    #     nHub = nSections

    #     # sections = self.build_blade_sections(r)
    #     sections = [asb.Airfoil('naca2402')] * nSections

    #     twists = (90. - self._P) * pitchFac

    #     le = np.array([[-sweeps[i],
    #                     0.,
    #                     r[i]*self._R] for i in range(nSections)])
        
    #     rotLe = [rotate(le[i], np.array([0., 0., 1.]), twists[i], np.zeros(3)) for i in range(nSections)]
    #     # le = rotate(le, np.array([0., 0., 1.]), twists, np.zeros(3))
    #     # le = np.array([[0., 0., r[i]*self._R] for i in range(nSections)])
    #     # twists = (270. - np.array([np.degrees(np.arctan(prop._P_D)) for i in range(nSections)])) * 1.

    #     # varPitch = s.twist

    #     angles = np.linspace(0., 360*(1-1/nBlades), nBlades)

    #     xyzHub = np.cosspace(0., rHub, nHub).reshape((nHub,1)) * np.array([1., 0., 0.])
    #     radiusHub = np.sqrt(rHub**2 - xyzHub[:,0]**2)
        
    #     propeller = asb.Airplane(
    #         name="propeller",
    #         xyz_ref=[0, 0, 0],  # CG location
    #         wings=[
    #             asb.Wing(
    #                 name="blade "+str(j),
    #                 symmetric=False,  # Should this wing be mirrored across the XZ plane?
    #                 xsecs=[ asb.WingXSec(  # Root
    #                         # xyz_le = le[i],  # Coordinates of the XSec's leading edge, relative to the wing's leading edge.
    #                         xyz_le = rotate(rotLe[i], np.array([1., 0., 0.]), angles[j], np.zeros(3)),  # Coordinates of the XSec's leading edge, relative to the wing's leading edge.
    #                         chord = chords[i],
    #                         twist = -twists[i], # degrees
    #                         airfoil = sections[i]
    #                         )
    #                 for i in range(nSections)]
    #             )
    #             for j in range(nBlades)
    #             ],
    #         fuselages=[
    #             asb.Fuselage(
    #                 name="hub",
    #                 xsecs=[
    #                     asb.FuselageXSec(
    #                         xyz_c=2*xyzHub[i],
    #                         radius=radiusHub[i],
    #                     )
    #                 for i in range(nHub)
    #                 ]
    #             )
    #         ]
    #         )
        
    #     return propeller
    
        
    def compute_Kt(self, J):
        """
        Computes Kt for a given J set using B-series propeller open-water
        polynamial performance.

        Parameters
        ----------
        J : 1-D array_like. Advance parameter

        Returns
        -------
        1-D array_like. Thrust coefficient Kt

        """
        return b_series_polynomial(J, self._P_D, self._Ae_Ao, self._Z, self._cT, self._stuvT)


    def compute_Kq(self, J):
        """
        Computes Kq for a given J set using B-series propeller open-water
        polynamial performance.

        Parameters
        ----------
        J : 1-D array_like. Advance parameter

        Returns
        -------
        1-D array_like. Torque coefficient Kq

        """
        return b_series_polynomial(J, self._P_D, self._Ae_Ao, self._Z, self._cQ, self._stuvQ)

    
    def compute_performance(self, J):
        """
        Compute Kt, Kq and eta for a given set of J using B-series propeller
        open-water polynamial performance.

        Parameters
        ----------
        J : 1-D array_like. Advance parameter

        Returns
        -------
        1-D array_like. Torque coefficient Kt

        Returns
        -------
        Kt : 1-D array_like. Thrust coefficient
        Kq : 1-D array_like. Torque coefficient
        eta : 1-D array_like. Propeller open-water efficiency

        """
        Kt = self.compute_Kt(J)
        Kq = self.compute_Kq(J)
        
        eta = Kt/Kq * J/(2*np.pi)
        
        return Kt, Kq, eta
    
    
    def compute_forces(self, J, rho=1025., Va=None, rpm=None):
        """
        Compute Kt, Kq and eta for a given set of J using B-series propeller
        open-water polynamial performance.

        Parameters
        ----------
        J : 1-D array_like. Advance parameter

        Returns
        -------
        1-D array_like. Torque coefficient Kt

        Returns
        -------
        Kt : 1-D array_like. Thrust coefficient
        Kq : 1-D array_like. Torque coefficient
        eta : 1-D array_like. Propeller open-water efficiency

        """
        if rpm is not None:
            n = rpm/60
        elif Va is None:
            raise ValueError('Va or rpm need to be specified')
        else:
            n = Va/(J*self._D)
        
        Kt = self.compute_Kt(J)
        Kq = self.compute_Kq(J)
        
        T = Kt * rho * n**2 * self._D**4
        Q = Kq * rho * n**2 * self._D**5
        P = Q * n * 2*np.pi
        
        return T, Q, P, n
    
    
    def compute_coefficients(self, J, rho=1025., Va=None, rpm=None):
        """
        Compute Kt, Kq and eta for a given set of J using B-series propeller
        open-water polynamial performance.

        Parameters
        ----------
        J : 1-D array_like. Advance parameter

        Returns
        -------
        1-D array_like. Torque coefficient Kt

        Returns
        -------
        Kt : 1-D array_like. Thrust coefficient
        Kq : 1-D array_like. Torque coefficient
        eta : 1-D array_like. Propeller open-water efficiency

        """
        if rpm:
            n = rpm/60
            if Va:
                print('rpm value kept for all calculations, specified Va is ignored')
            Va = n/(J*self._D)
        elif not(Va):
            raise ValueError('Va or rpm need to be specified')
        else:
            n = Va/(J*self._D)
        
        Kt = self.compute_Kt(J)
        Kq = self.compute_Kq(J)
        
        # T, Q, P, n = compute_forces(self, J, rho=rho, Va=Va, rpm=rpm)
        
        # Ap = self.area
        # Ap = np.pi*self.radius**2
        
        # Cth = T/(0.5*rho*Ap*Va**2)
        # Cp = P/(0.5*rho*Ap*Va**3)
        
        Cth = 8/np.pi * Kt/J**2
        Cq = 8/np.pi * Kq/J**2
        Cp = Cq * n * 2*np.pi
        
        return Cth, Cq, Cp
    
    # TODO implement properly
    # def compute_forces(self,
    #                    op_point,
    #                    hull = None,
    #                    ):
    #     if hull:
    #                       rpm: float,
    #                       op_point: OperatingPoint = OperatingPoint(),
    #                       recompute_statics: bool = True,
    #                       full_output: bool = False,
    #                       shaft_efficiency: float = 0.98,
    #                       ):

    #     if self.propeller:
    #     if self.hull:
    #         if recompute_statics:
    #             self.hull.compute_statics(op_point)
    #         w, t, etaR = self.hull.compute_propulsion_coefficients(op_point.stw, self.propeller.blade_area_ratio)
    #     else:
    #         w, t, etaR = 0., 0., 1.
        
    #     etaS = shaft_efficiency
    #     etaH = (1-t)/(1-w)
        
    #     leeway = op_point.leeway
        
    #     n = rpm / 60. # rps
    #     V = op_point._stw # m/s
    #     D = self.propeller.diameter # m
    #     J = (1 - w) * V / (n*D)
        
    #     Kt, Kq, eta0 = self.propeller.compute_performance(J)
        
    #     T, Q, P, n = self.propeller.compute_forces(
    #         J,
    #         op_point.environment.water.density,
    #         rpm=rpm
    #     )
        
    #     F = np.sum(T*(1-t)*etaR)
        
    #     center = self.propeller.center
    #     axis = tall(self.propeller.axis)
        
    #     prop = {
    #         'J': J,
    #         'Kt': Kt,
    #         'Kq': Kq,
    #         'eta0': eta0,
    #         'etaH': etaH,
    #         'etaS': etaS,
    #         'etaR': etaR,
    #         'T': T,
    #         'Q': Q,
    #         'P': P,
    #         'n': n,
    #         'w': w,
    #         't': t,
    #         'F_b': F * np.array([1.,0.,0.]),
    #         'M_b': np.zeros(3),
    #         'F_ab': F * axis,
    #         'M_ab': np.cross(center, F * axis)
    #     }
        
    #     Fprop = prop['F_ab']
    #     Mprop = prop['M_ab']


if __name__=="__main__":
    
    # TODO clean
    
    import casadi as ca
    import matplotlib.pyplot as plt
    import csv
    
    def slice_before_negative(arr):
        for i, num in enumerate(arr):
            if num < 0:
                return i + 1
        return i + 1
    
    def generator_speed_torque_law(rpm):
        return 1.493e-3 * rpm**2
        
    
    STW = 11. # kts
    Rt = 400. # kN
    Fx = 0. # kN
    rho = 1025.
    
    D = 4.
    
    nSections = 10
    
    plt.figure()
    
    J = np.linspace(0.1, 1.3, 100)
    # J = ca.linspace(0., 3.0, 100)
    TSR = np.pi/J
    rpm = 60 * STW * 0.5144 / (J*D)
    
    # Define the colormap
    cmap_name = 'autumn'
    # cmap_name = 'hot'
    cmap = plt.get_cmap(cmap_name)
    
    nPitch = 3
    
    # Get 5 colors from the colormap
    colors = [cmap(i) for i in np.linspace(0, 1, nPitch+2)]
    
    # PITCHES = np.linspace(0.4, 1.4, nPitch)
    PITCHES = [0.8]
    
    P0 = [1.4, 1.2, 1.0, 0.8, 0.5,
        ]
    J0 = [1.5, 1.25, 1.1, 0.9, 0.6,
        ]
    
    # Specify the output file name
    output_file = "stacked_vectors.csv"
    
    for i, P_D in enumerate(PITCHES):
        # prop = BSeriesPropeller(Z=4, Ae_Ao=0.8, P_D=P_D, D=3.72)
        prop = BSeriesPropeller(Z=4, Ae_Ao=0.8, P_D=P_D, D=4.)
        # prop = BSeriesPropeller(Z=7, Ae_Ao=.66, P_D=P_D, D=3.800)
        
        # asb_prop = prop.build_aerosanbox_geometry(nSections)
        
        # Tvlm = []
        # Qvlm = []
        
        # asb_prop.draw()
        
        # for Ji in J:
        #     # print(Ji)
        #     rot = STW * u.knot / (Ji * prop.diameter) * 60. # rotation per minute
        #     print(rot)
        #     vlm = asb.VortexLatticeMethod(airplane=asb_prop,
        #                                   spanwise_resolution=1,
        #                                   op_point=asb.OperatingPoint(
        #                                       velocity = 5.144,
        #                                       p = rot * u.rpm
        #                                       ),
        #                                   align_trailing_vortices_with_wind=False)
            
        #     aero = vlm.run()
            
        #     # vlm.draw()
            
        #     Tvlm.append(aero['F_g'][0])
        #     Qvlm.append(aero['M_g'][0])
            
        # Tvlm = np.array(Tvlm)
        # Qvlm = np.array(Qvlm)

        # vlm.calculate_streamlines(n_steps=100,
        #                           length=10.)

        # vlm.draw(recalculate_streamlines=False)
    
        Kt, Kq, eta = prop.compute_performance(J)
        T, Q, P, n = prop.compute_forces(J, Va=STW*0.5144)
        Cth, Cq, Cp = prop.compute_coefficients(J, Va=STW*0.5144)
        
        # print(T/Tvlm)
        # print(Q/Qvlm)
        
        # iMaxT = slice_before_negative(Kt)
        # iMaxQ = slice_before_negative(Kq)
        # # iMaxQ = -1
        # iMaxE = slice_before_negative(eta) - 1
        
        # iQ = np.where(Kq > 0., True, False)
        # iQ = np.argwhere(Kq > 0.)[:,0]
        iQ = np.argwhere(Kq)[:,0]
        # plt.plot(J[:iMaxE], eta[:iMaxE], color=colors[i], ls='-.')
        # plt.plot(J[iMaxQ:], -1/eta[iMaxQ:], color=colors[i], ls='-.')
        # plt.plot(J[:iMaxQ], Kt[:iMaxQ] ,color=colors[i], label='Pitch ratio '+str(round(P_D, 2)))
        # plt.plot(J[:iMaxQ], 10*Kq[:iMaxQ], color=colors[i], ls='--')
        # plt.plot(J[:iMaxE], eta[:iMaxE], color=colors[i], ls='-.')
        
        plt.plot(J[iQ], Kt[iQ] ,color=colors[i], label='Pitch ratio '+str(round(P_D, 2)))
        plt.plot(J[iQ], 10*Kq[iQ], color=colors[i], ls='--')
        
        # plt.plot(J[iQ], eta[iQ], color=colors[i], ls='-.')
        
        # plt.plot(J, Kt ,color=colors[i], label='Pitch ratio '+str(round(P_D, 2)))
        # plt.plot(J, 10*Kq, color=colors[i], ls='--')
        # plt.plot(J, eta, color=colors[i], ls='-.')
        
        # with open(output_file, mode='a', newline='') as file:
        #     writer = csv.writer(file)
            
        #     # Stack and write rows
        #     for k in range(len(iQ)):
        #         writer.writerow([J[iQ][k], P_D, Kt[iQ][k], Kq[iQ][k], eta[iQ][k]])
        
        
        # plt.plot(TSR, Cth ,color=colors[i], label='Pitch ratio '+str(round(P_D, 2)))
        # plt.plot(TSR, Cp, color=colors[i], ls='--')
        
        # plt.plot(TSR[:iMaxE], eta[:iMaxE], color=colors[i], ls='-.')
        # plt.plot(TSR[iMaxQ:], -1/eta[iMaxQ:], color=colors[i], ls='-.')
        

    
    
    
    D = prop.diameter

    Va = rpm/60/(J*D)   

    Teff = (Rt - Fx) * 1000.
    KtEff = Teff / (rho * (STW * 0.5144)**2 * D**2) * J**2
    CthEff = Teff / (0.5*rho*prop.area*Va**2)
    
    # plt.plot(J, KtEff, color='black', lw=1.)
    
    rpmMax = 155.
    Jmin = STW * 0.5144 / (rpmMax/60. * D)
    
    # plt.plot([Jmin, Jmin], [-10., 10.], color='black', ls=':', lw=1., label='RPM '+str(rpmMax))
    # plt.plot([0., 10.], [0., 0.], color='black', lw=1., label='RPM '+str(rpmMax))
    
    plt.grid()
    plt.legend()
    
    plt.xlabel('J')
    plt.ylabel('eta, Kt, 10Kq')
    
    # plt.title('STW = '+str(STW)+' kts \ Rt = '+str(Rt)+' kN \ sail% = '+str(round(Fx/Rt*100,1))+' %')
    
    # plt.xlim((J.min(), J.max()))
    # plt.ylim((0., 1.))
    
    # plt.xlim((float(np.min(J)), float(np.max(J))))
    # plt.ylim((0., 1.0))
    
    plt.show()
