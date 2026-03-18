# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 23:38:18 2024

@author: jrich
"""

import os
import sys

__root_dir = os.path.dirname(os.path.abspath(__file__))
if __root_dir not in sys.path:
    sys.path.append(os.path.dirname(__root_dir))
    
# import numpy as np

from typing import Union, List
import copy

import archibald2.numpy as np

import casadi as ca

np.set_printoptions(precision=2)

#%%

def tall(array):
    return np.reshape(array, (-1, 1))


def wide(array):
    return np.reshape(array, (1, -1))


def relu(x):
    # import casadi as ca
    # return ca.fmax(0,x)
    
    return np.fmax(x, 0.)


def ramp(x, tol=1e-3):
    
    return np.fmin(1.,
                    np.fmax(0.,
                            (x+tol)/tol))

def axis_str_to_array(direction):
    if direction == 'x':
        return np.array([1.,0.,0.])
    elif direction == 'y':
        return np.array([0.,1.,0.])
    elif direction == 'z':
        return np.array([0.,0.,1.])
    if direction == '-x':
        return -np.array([1.,0.,0.])
    elif direction == '-y':
        return -np.array([0.,1.,0.])
    elif direction == '-z':
        return -np.array([0.,0.,1.])
    else:
        raise ValueError(f"'{direction}' could not be interpreted as a direction."+\
                         " Should be 'x', 'y', 'z' (with possibly a '-' sign) or a (3,) array.")
        return None


class DifferentiableMesh():
    """
    Differentiable mesh description. Vertices may be described by both numpy or CasADI objects.
    
    Allows approximated though differentiable hydrostatics computations.
    
    """
    
    # TODO: add a refine_mesh method to adapt coarser meshes
    
    def __init__(self,
                 vertices: Union[np.ndarray, List] = np.array([]),
                 faces: List[int] = None):
        
        if vertices is None or faces is None:
            vertices = np.zeros((3,3))
            faces = wide(np.arange(3))
            
        self._vertices = vertices
        self._faces = faces
        
        self.reset_data() # Always needs to be called when vertices are modified in place
        
    @property
    def vertices(self):
        return self._vertices
    
    @property
    def faces(self):
        return self._faces
    
    @vertices.setter
    def vertices(self, value):
        # Check if value is a numpy array, CasADi MX, or CasADi DM
        if not isinstance(value, (np.ndarray, ca.MX, ca.DM)):
            raise ValueError("Vertices must be a numpy array or a CasADi array (MX or DM).")
        
        # Check if the shape matches the expected number of vertices and 3D coordinates
        if value.shape != (self.faces.max() + 1, 3):
            raise ValueError(f"Shape mismatch between vertices and faces. Vertices should have shape {(self.faces.max() + 1, 3)} to match the defined mesh topology.")
        
        # Assign the value and reset data
        self._vertices = value
        self.reset_data()
        
    def reset_data(self):
        # Extract vertices of each face for later operations
        self._v0 = self.vertices[self.faces[:, 0], :]
        self._v1 = self.vertices[self.faces[:, 1], :]
        self._v2 = self.vertices[self.faces[:, 2], :]
        
        self._data = {'cross_product': None,
                      'triangle_centers': None,
                      'triangle_areas': None,
                      'tetrahedron_centers': None,
                      'tetrahedron_volumes': None,
                      'area': None,
                      'area_centroid': None,
                      'volume': None,
                      'volume_centroid': None,
                      }
        
    def compute_cross_product(self):
        """
        Compute the cross products between the edges of all triangular faces.

        """
        # Extract vertices of each face
        v0, v1, v2 = self._v0, self._v1, self._v2
        
        # Vectorized edge vectors
        edge1 = v1 - v0
        edge2 = v2 - v0
        
        # Cross product of edge vectors for all triangles at once
        self._data['cross_product'] = np.cross(edge1, edge2)
        
        
    def compute_triangle_centers(self):
        """
        Compute the centers of all triangular faces.

        """
        # Extract vertices of each face
        v0, v1, v2 = self._v0, self._v1, self._v2
        
        self._data['triangle_centers'] = (v0 + v1 + v2) / 3.0
        
        
    def compute_triangle_areas(self):
        """
        Compute the areas of all triangular faces.

        """        
        if self._data['cross_product'] is None:
            self.compute_cross_product()
            
        cross_prod = self._data['cross_product']
        
        # Area of each triangular face
        self._data['triangle_areas'] = np.linalg.norm(cross_prod + 1e-15, axis=1) / 2.0 # added a femto to avoid gradient error at x=0 for sqrt(0)
        
        
    def compute_tetrahedron_centers(self):
        """
        Compute the centers of all tetrahedrons formed by faces and origin.

        """
        # Extract vertices of each face
        v0, v1, v2 = self._v0, self._v1, self._v2
        
        self._data['tetrahedron_centers'] = (v0 + v1 + v2) / 4.0
        
            
    def compute_tetrahedron_volumes(self):
        """
        Compute the signed volumes of all tetrahedrons formed by faces and origin.

        """        
        if self._data['cross_product'] is None:
            self.compute_cross_product()
            
        cross_prod = self._data['cross_product']
        v0 = self._v0
        
        # Signed volume of tetrahedron formed with origin for each face
        self._data['tetrahedron_volumes'] = np.sum(v0 * cross_prod / 6.0, axis=1)
    
    def vertices_distances_to_plane(self, point, normal):
        """
        Compute the oriented distances of each vertex to a given plane, represented by a point and a normal.

        """
        if type(normal) is str:
            normal = axis_str_to_array(normal)
        # NB
        # if dist.all < 0 : fully immersed
        # if dist.all > 0 : fully emerged
        
        return np.add(self.vertices, -wide(point)) @ normal
    
    def faces_distances_to_plane(self, point, normal):
        """
        Compute the oriented distances of each vertex to a given plane, represented by a point and a normal.

        """
        if type(normal) is str:
            normal = axis_str_to_array(normal)
        # NB
        # if dist.all < 0 : fully immersed
        # if dist.all > 0 : fully emerged
        
        if self._data['triangle_centers'] is None:
            self.compute_triangle_centers()
        
        return np.add(self._data['triangle_centers'], -wide(point)) @ normal
    
    def frontal_area(self,
                     direction: Union[str, np.ndarray]):
        """
        Compute the frontal area of a mesh in a given direction.

        Parameters:
        - direction: str or (3,) array representing the frontal direction (str, numpy array or CasADi MX/SX).

        Returns:
        - frontal_area: Scalar representing the frontal area of the mesh.
        
        """
        if type(direction) is str:
            direction = axis_str_to_array(direction)
        
        if self._data['cross_product'] is None:
            self.compute_cross_product()
            
        cross_prod = self._data['cross_product']
        
        # Normalize the direction vector
        direction = direction / np.linalg.norm(direction)

        # Compute the product between the cosine of the angle between the triangle normal and the projection direction and the triangle area
        cos_theta_area = (cross_prod @ tall(direction))

        frontal_area = np.sum(np.fabs(cos_theta_area)) / 4.0
        
        return frontal_area
        
    def weighted_volume(self, weight=1.):
        """
        Compute the mesh volume. May be weighted.

        """
        if self._data['tetrahedron_volumes'] is None:
            self.compute_tetrahedron_volumes()
        
        # Signed volume of tetrahedron formed with origin for each face
        tetra_volumes = self._data['tetrahedron_volumes'] 
        
        # Total volume
        return np.abs(np.sum(tetra_volumes * weight)) * np.sign(weight)
        
    def weighted_volume_centroid(self, weight=1.):
        """
        Compute the mesh volume centroid. May be weighted.

        """
        if self._data['tetrahedron_centers'] is None:
            self.compute_tetrahedron_centers()
        if self._data['tetrahedron_volumes'] is None:
            self.compute_tetrahedron_volumes()
        
        tetra_centers = self._data['tetrahedron_centers']
        tetra_volumes = self._data['tetrahedron_volumes'] * weight
        
        # Weighted sum of centroids by volumes to get the total center of mass
        return np.sum(wide(tetra_volumes) @ tetra_centers / np.sum(tetra_volumes), axis=0)
    
    def weighted_area(self, weight=1.):
        """
        Compute the mesh area. May be weighted.

        """
        if self._data['triangle_areas'] is None:
            self.compute_triangle_areas()
        
        # Areas of each triangular faces
        tri_areas = self._data['triangle_areas'] * weight
        
        # Total area
        return np.sum(tri_areas)

    def weighted_area_centroid(self, weight=1.):
        """
        Compute the mesh volume centroid. May be weighted.

        """
        if self._data['triangle_centers'] is None:
            self.compute_triangle_centers()
        if self._data['triangle_areas'] is None:
            self.compute_triangle_areas()
        
        tri_centers = self._data['triangle_centers']
        tri_areas = self._data['triangle_areas'] * weight
        
        # Weighted sum of centroids by volumes to get the total center of mass
        return np.sum((wide(tri_areas) @ tri_centers / np.sum(tri_areas)), axis=0)
    
    @property
    def volume(self):
        if self._data['volume'] is None:
            self._data['volume'] = self.weighted_volume(weight=1.)
        
        return self._data['volume']
    
    @property
    def volume_centroid(self):
        if self._data['volume_centroid'] is None:
            self._data['volume_centroid'] = self.weighted_volume_centroid(weight=1.)
        
        return self._data['volume_centroid']
    
    @property
    def area(self):
        if self._data['area'] is None:
            self._data['area'] = self.weighted_area(weight=1.)
        
        return self._data['area']
    
    @property
    def area_centroid(self):
        if self._data['area_centroid'] is None:
            self._data['area_centroid'] = self.weighted_area_centroid(weight=1.)
        
        return self._data['area_centroid']
    
    @property
    def bounds(self):
        """
        Compute the mesh bounding box in all 3 directions.

        """
        v = self.vertices
        
        x_min = np.min(v[:,0])
        x_max = np.max(v[:,0])
        y_min = np.min(v[:,1])
        y_max = np.max(v[:,1])
        z_min = np.min(v[:,2])
        z_max = np.max(v[:,2])
        
        return np.array([[x_min, x_max],
                         [y_min, y_max],
                         [z_min, z_max],
                         ])
    
        
    def compress_mesh(self,
                      point,
                      normal,
                      dist=None,
                      tol=1e-5,
                      scale_fac=2.,
                      inplace: bool = False):
        """
        Compress the mesh.
        TODO: write the complete doc

        Parameters
        ----------
        point : TYPE
            DESCRIPTION.
        normal : TYPE
            DESCRIPTION.
        tol : TYPE, optional
            DESCRIPTION. The default is 1e-5. The higher tol, the further the plane projection influence.
        scale_fac : TYPE, optional
            DESCRIPTION. The default is 2. The higher scale_fac, the sharper the scaling ramp.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        ### PROJECTING STEP
        
        vertices = self.vertices
        faces = self.faces
        
        if type(normal) is str:
            normal = axis_str_to_array(normal)
        
        # Compute the vertices distances from the plane, if not already provided
        if dist is None:
            dist = self.vertices_distances_to_plane(point, normal)
        
        # Projected vertices = vert - (vert.normal) @ normal
        # NB: the ramp function allows to only affect the vertices on one side on the plane
        vert_proj = vertices - tall(dist*ramp(dist, tol)) @ wide(normal)
        
        ### SCALING STEP
        # NB: the previously moves vertices are scaled towards the center of the projecting plane
        # to get a cleaner external mesh shape
        
        # min_fac = -np.min(scal) # NB: this is water draft
        max_fac = np.max(dist)/scale_fac
        
        proximity = ramp(dist-max_fac, tol=max_fac)
        fac = 1 - proximity
        
        center = np.mean(vertices, axis=0)
        center_proj = wide(center) - (np.add(wide(center), -wide(point)) @ normal) @ wide(normal)
        
        # Scaled vertices = (vert - center) * scale_factor + center
        vert_proj_scal = np.add(np.add(vert_proj, -wide(center_proj)) * tall(fac),
                                wide(center_proj)
                                )
        
        if inplace:
            self.vertices = vert_proj_scal
            self.reset_data()
        else:
            return DifferentiableMesh(vert_proj_scal, copy.copy(faces))
        
        
    def __repr__(self):
        
        return f"{self.__class__.__name__} instance "+\
                " with {self.vertices.shape[0]} vertices "+\
                "and {self.faces.shape[0]} faces."
        
        
    def draw(self,
             color = 'cyan',
             opacity = 0.3,
             show_edges = True,
             draw_plane = False,
             point = np.zeros(3),
             normal = np.array([0., 0., 1.]),
             plane_color = 'orange',
             plane_opacity = 0.2,
             backend: str = 'pyvista',
             ):
        
        if backend == 'pyvista':
            import pyvista as pv
            
            # Create a PyVista plotter
            plotter = pv.Plotter()
            
            # Create the mesh
            mesh = pv.PolyData(self.vertices, np.hstack([[3, *face] for face in self.faces]))
            plotter.add_mesh(mesh, color=color, opacity=opacity, show_edges=show_edges)
            
            if draw_plane:
                # Add the plane
                plane = pv.Plane(center=point, direction=normal, i_size=2, j_size=2)
                plotter.add_mesh(plane, color=plane_color, opacity=plane_opacity)
            
            # Display the plot
            plotter.show()
            
        else:
            raise NotImplementedError(f'{backend} is not a supported drawing module.')
            
            
if __name__ == '__main__':
    
    from trimesh import load
    
    hullStl = '../../Private/49er_data/hull.stl'

    rho = 1025.
    displacement = 290. # kg

    z = 0.
    leeway = 0.
    heel = 0.
    trim = 0.

    cog = np.array([2.40,
                    .1,
                    0.14]) # m

    hullMesh = load(hullStl)
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
    