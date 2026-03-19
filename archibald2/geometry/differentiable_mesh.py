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
from archibald2.tools.math_utils import ReLU

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
    


class DifferentiablePolygon():
    """
    Polygon description.
    
    Allows approximated though differentiable hydrostatics computations.
    
    """
    def __init__(self,
                 vertices: Union[np.ndarray, List] = np.array([]),
                 edges: List[int] = None):
        
        if vertices is None or edges is None:
            vertices = np.zeros((2,3))
            faces = wide(np.arange(2))
        
        self._vertices = vertices
        self._edges = edges
                
        self.reset_data() # Always needs to be called when vertices are modified in place
        
    @property
    def vertices(self):
        return self._vertices
    
    @property
    def edges(self):
        return self._edges
    
    def reset_data(self):
        # Extract vertices of each face for later operations
        self._nV = self.vertices.shape[0]
        self._nE = self.edges.shape[0]
        self._c = np.mean(self.vertices, axis=0)
        
        self._v0 = self.vertices[self.edges[:, 0], :]
        self._v1 = self.vertices[self.edges[:, 1], :]
        
        self._v2 = np.tile(self._c, (self._nE, 1))
        
        self._data = {'cross_product': None,
                      'triangle_areas': None,
                      'triangle_centers': None,
                      'edge_lengths': None,
                      'edge_centers': None,
                      'area': None,
                      'area_centroid': None,
                      'perimeter': None,
                      'normals': None,
                      'bounds': None,
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
        
    def compute_triangle_normals(self):
        """
        Compute the normals of all triangular faces.

        """        
        if self._data['cross_product'] is None:
            self.compute_cross_product()
            
        self._data['normals'] = self._data['cross_product'] / 2.0
        
        
    def compute_edge_centers(self):
        """
        Compute the center of each polygon edge.

        """
        self._data['edge_centers'] = (self._v0 + self._v1) / 2.0
    
    
    def compute_edge_lengths(self):
        """
        Compute the length of each polygon edge.

        """
        v0, v1 = self._v0, self._v1
        
        self._data['edge_lengths'] = np.linalg.norm(v1 - v0, axis=1)
    

    def compute_perimeter(self):
        """
        Compute the polygon perimeter.

        """
        if self._data['edge_lengths'] is None:
            self.compute_edge_lengths()
            
        self._data['perimeter'] = np.sum(self._data['edge_lengths'])
    
    
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
        if self._data['normals'] is None:
            self.compute_triangle_normals()
            
        normals = self._data['normals']
        
        # Area of each triangular face
        self._data['triangle_areas'] = np.linalg.norm(normals + 1e-15, axis=1) # added a femto to avoid gradient error at x=0 for sqrt(0)
        
        
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
    def normals(self):
        if self._data['normals'] is None:
            self.compute_triangle_normals()
        
        return self._data['normals']
    
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
    def perimeter(self):
        if self._data['perimeter'] is None:
            self.compute_perimeter()
        
        return self._data['perimeter']
    
    def compute_bounds(self):
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
        
        self._data['bounds'] = np.array([[x_min, x_max],
                                         [y_min, y_max],
                                         [z_min, z_max],
                                         ])
        
    @property
    def bounds(self):
        """
        Return the mesh bounding box in all 3 directions.

        """
        if self._data['bounds'] is None:
            self.compute_bounds()
            
        return self._data['bounds']
    

    def draw(self,
            color = 'cyan',
            # mesh_color = 'grey',
            # plane_color = 'orange',
            # plane_opacity = 0.01,
            backend: str = 'pyvista',
            show: bool = True,
            ):
        
        if backend=="pyvista":
        
            import pyvista as pv
            
            # Create a PyVista plotter
            plotter = pv.Plotter()
                    
            for u, v in self.edges:
                # Extract endpoints
                pts = self.vertices[[u, v]]
                
                # Each line cell format: [number_of_points, id0, id1]
                # For 2-point line it's always [2, 0, 1]
                line = pv.PolyData(pts, lines=np.array([2, 0, 1]))
                
                # Add to plotter
                plotter.add_mesh(line, color=color, line_width=3)
                
                # Show axes triad and bounding box axes
            plotter.show_axes()            # 3D axes in corner
            plotter.show_grid()            # grid + bounding box
            
            # Display the plot
            if show:
                plotter.show()
            return plotter


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
    
    @property
    def edges(self):
        if self._data['edges'] is None:
            self.compute_edges()
            
        return self._data['edges']
    
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
                      'edges': None,
                      'triangle_centers': None,
                      'triangle_areas': None,
                      'tetrahedron_centers': None,
                      'tetrahedron_volumes': None,
                      'area': None,
                      'area_centroid': None,
                      'volume': None,
                      'volume_centroid': None,
                      'normals': None,
                      'bounds': None,
                      }
        
    def compute_edges(self):
        # Generate all edges for each triangle (3 edges per face)
        edges = np.concatenate([
            self.faces[:, [0, 1]],
            self.faces[:, [1, 2]],
            self.faces[:, [2, 0]]
        ], axis=0)

        # Sort each edge so that [a, b] and [b, a] are treated the same
        edges = np.sort(edges, axis=1)
        
        # Remove duplicates
        self._data['edges'] = np.unique(edges, axis=0)
        
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
        
    def compute_triangle_normals(self):
        """
        Compute the normals of all triangular faces.

        """        
        if self._data['cross_product'] is None:
            self.compute_cross_product()
            
        self._data['normals'] = self._data['cross_product'] / 2.0
        
        
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
        if self._data['normals'] is None:
            self.compute_triangle_normals()
            
        normals = self._data['normals']
        
        # Area of each triangular face
        self._data['triangle_areas'] = np.linalg.norm(normals + 1e-15, axis=1) # added a femto to avoid gradient error at x=0 for sqrt(0)
        
        
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
                     direction: Union[str, np.ndarray],
                     weight: Union[float, np.ndarray]=1.
                     ):
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
            
        cross_prod = self._data['cross_product'] # * weight
        
        # Normalize the direction vector
        direction = direction / np.linalg.norm(direction)

        # Compute the product between the cosine of the angle between the triangle normal and the projection direction and the triangle area
        cos_theta_area = (cross_prod @ tall(direction))

        frontal_area = np.sum(np.fabs(cos_theta_area) * tall(weight)) / 4.0
        
        return frontal_area
        
    def weighted_volume(self, weight=1.):
        """
        Compute the mesh volume. May be weighted.

        """
        if self._data['tetrahedron_volumes'] is None:
            self.compute_tetrahedron_volumes()
        
        # Signed volume of tetrahedron formed with origin for each face
        tetra_volumes = self._data['tetrahedron_volumes'] * weight
        
        # Total volume
        return np.sum(tetra_volumes)
        
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
    def normals(self):
        if self._data['normals'] is None:
            self.compute_triangle_normals()
        
        return self._data['normals']
    
    # @property
    # def area_Ixx(self):
    #     dist = self._data['triangle_centers'] - self.area_centroid
    #     Ixx = np.sum(
    #         self._data['triangle_areas'] * (dist[:,1]**2 + dist[:,2]**2)
    #     )
    #     return Ixx
    
    def compute_bounds(self):
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
        
        self._data['bounds'] = np.array([[x_min, x_max],
                                         [y_min, y_max],
                                         [z_min, z_max],
                                         ])
        
    @property
    def bounds(self):
        """
        Return the mesh bounding box in all 3 directions.

        """
        if self._data['bounds'] is None:
            self.compute_bounds()
            
        return self._data['bounds']
    
        
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
                f"with {self.vertices.shape[0]} vertices, "+\
                f"{self.edges.shape[0]} edges "+\
                f"and {self.faces.shape[0]} faces."
                
        
    def hydrostatics(self,
                     point,
                     normal,
                     offset=0.,
                     factor=1e6,
                     ):
        
        if self._data['cross_product'] is None:
            self.compute_cross_product()
            
        if self._data['triangle_centers'] is None:
            self.compute_triangle_centers()
        
        fdist = self.faces_distances_to_plane(point, normal)
        
        n = self.faces.shape[0]
        
        ppoint = np.tile(wide(point), (n,1))
        
        # Extract vertices of each face
        v0, v1, v2 = self._v0 - ppoint, self._v1 - ppoint, self._v2 - ppoint
        
        centers = (v0 + v1 + v2) / 4.0
        
        # ndist = np.fmax(np.fmin(-fdist+0.5, 1.), 0.)
        ndist = np.fmax(np.fmin((-fdist+offset)*factor, 1.), 0.)
        
        vols = np.sum(v0 * self._data["cross_product"] / 6.0, axis=1)
        volume = np.sum(vols * ndist)
        cob = np.sum(wide(vols * ndist) @ centers / np.sum(vols * ndist), axis=0) + point
        
        return volume, cob # TODO find a solution to compute cob. Volume very quick and precise
    
    
    def slice_mesh(self,
                   point,
                   normal
                   ):
        
        dist = self.vertices_distances_to_plane(point, normal)
        
        # edges = diff_mesh.edges
        edges_raw = np.concatenate([
            self.faces[:, [0, 1]],
            self.faces[:, [1, 2]],
            self.faces[:, [2, 0]],
        ], axis=0)
        
        edges = np.sort(edges_raw, axis=1)
        
        edge_dist = dist[edges]
        edge_prod = np.prod(edge_dist, axis=1)
        edge_sign = np.sign(edge_prod)
        edge_mask = ((1-edge_sign)/2)
        
        
        selected_edges = edges[edge_mask == 1.]
        
        if len(selected_edges) > 0:
        
            face_mask = np.sum(edge_mask.reshape((3,-1)).T, axis=1)//2 # sliced faces
            
            # Get coordinates of the selected edge endpoints
            v0 = self.vertices[selected_edges[:, 0]]  # shape: (n_crossing, 3)
            v1 = self.vertices[selected_edges[:, 1]]
            
            # Get distances of the endpoints to the plane
            d0 = edge_dist[edge_mask == 1.][:, 0]
            d1 = edge_dist[edge_mask == 1.][:, 1]
            
            # Compute interpolation factor t (clip to avoid numerical instability)
            t = d0 / (d0 - d1 + 1e-12)
            
            # Compute intersection points
            intersections = v0 + tall(t) * (v1 - v0)
        
            intersection_indices = (np.full(edges.shape[0], -1) +\
                np.cumsum(edge_mask)*edge_mask).reshape((3, -1)).T
            
            II = intersection_indices[np.where(face_mask, True, False)].astype(int)
            
            slice_edges = []
            
            for i in range(len(II)):
                I = II[i,:]
                
                shift = np.argwhere(I==-1)[0,0]
                i1 = (1 + shift) % 3
                i2 = (2 + shift) % 3
                
                slice_edges.append([I[i2], I[i1]])
                                   
            slice_edges = np.array(slice_edges)
            
            return DifferentiablePolygon(intersections, slice_edges)
        
        return DifferentiablePolygon(None, None)
        
        
        
        
    def draw(self,
             color = 'cyan',
             opacity = 0.3,
             show_edges = True,
             draw_plane = False,
             point = np.zeros(3),
             normal = np.array([0., 0., 1.]),
             mesh_color = 'grey',
             plane_color = 'orange',
             plane_opacity = 0.2,
             backend: str = 'pyvista',
             show: bool = True,
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
                plane = pv.Plane(center=point, direction=normal, i_size=200, j_size=200)
                plotter.add_mesh(plane, color=plane_color, opacity=plane_opacity)
            
            if show:
                # Display the plot
                plotter.show()
            return plotter
            
        elif backend == 'plotly':
            import plotly.graph_objects as go
            
            fig = go.Figure()
            
            x, y, z = self.vertices.T
            i, j, k = self.faces.T
            
            fig.add_trace(
                go.Mesh3d(
                    x=x, y=y, z=z,
                    i=i, j=j, k=k,
                    opacity=1.,
                    color=mesh_color,
                )
            )
            
            fig.update_layout(
                scene=dict(
            #         xaxis=dict(showbackground=False, showspikes = False, showticklabels=False, title=''),
            #         yaxis=dict(showbackground=False, showspikes = False, showticklabels=False, title=''),
            #         zaxis=dict(showbackground=False, showspikes = False, showticklabels=False, title=''),
            #         # xaxis = list(title = '', autorange = TRUE, showspikes = FALSE, showgrid = FALSE, zeroline = FALSE, showline = FALSE, autotick = TRUE, ticks = '', showticklabels = FALSE),
            #         # yaxis = list(title = '', autorange = TRUE, showspikes = FALSE, showgrid = FALSE, zeroline = FALSE, showline = FALSE, autotick = TRUE, ticks = '', showticklabels = FALSE),
            #         # zaxis = list(title = '', autorange = TRUE, showspikes = FALSE, showgrid = FALSE, zeroline = FALSE, showline = FALSE, autotick = TRUE, ticks = '', showticklabels = FALSE),
                    aspectmode='data'
                ),
            #     showlegend=False,
            #     margin=dict(l=0, r=0, t=0, b=0),
            #     template='plotly_dark',
            )
            
            # fig.layout.scene.camera.projection.type = "orthographic"
            # fig.layout.scene.dragmode="pan"
            
            # camera = dict(
            #     up=dict(x=0, y=0, z=1),
            #     center=dict(x=0, y=0, z=0),
            #     eye=dict(x=-10, y=0, z=0)
            # )
            
            # fig.update_layout(scene_camera=camera)
            

            if show:
                from plotly.offline import plot
                plot(fig)
            return fig
            
        else:
            raise NotImplementedError(f'{backend} is not a supported drawing module.')
            
            
# if __name__ == '__main__':
    
#     from trimesh import load
    
#     hullStl = '../../Private/49er_data/hull.stl'

#     rho = 1025.
#     displacement = 290. # kg

#     z = 0.
#     leeway = 0.
#     heel = 0.
#     trim = 0.

#     cog = np.array([2.40,
#                     .1,
#                     0.14]) # m

#     hullMesh = load(hullStl)
#     vertices, faces = hullMesh.vertices, hullMesh.faces



#     leewayRot = np.rotation_matrix_3D((leeway)*np.pi/180,
#                                       np.array([0.,0.,1.]))
    
#     antiLeewayRot = np.rotation_matrix_3D((leeway)*np.pi/180,
#                                           np.array([0.,0.,1.]))

#     vertices = vertices @ (antiLeewayRot) - z

#     point = np.array([0.0, 0.0, 0.0])
#     normal = np.array([0., 0., 1.])

#     initMesh = DifferentiableMesh(vertices, faces)

#     initVertDist = initMesh.vertices_distances_to_plane(point, 'z')
#     initFaceDist = initMesh.faces_distances_to_plane(point, 'z')

#     # NB
#     # if np.sum(ramp(dist)) == 0 : fully immersed
#     # if np.prod(ramp(dist)) == 1 : fully emerged
    
#     fully_immersed = ca.if_else(np.abs(np.sum(ramp(initVertDist))) < 1e-5, True, False)
#     fully_emerged = ca.if_else(np.abs(np.prod(ramp(initVertDist)) - 1.) < 1e-5, True, False)
    
#     if fully_immersed:
#         ### HYDROSTATICS COMPUTATION
        
#         volume = initMesh.volume
#         cob = initMesh.volume_centroid
        
#         Wsa = initMesh.area
#         cow = initMesh.area_centroid
        
#         Ax = initMesh.frontal_area('x')
#         Ay = initMesh.frontal_area('y')
        
#         bounds = initMesh.bounds
        
#         X0 = bounds[0,0]
#         T = bounds[2,1] - bounds[2,0]
        
#         Ax = initMesh.frontal_area('x')
#         Ay = initMesh.frontal_area('y')
        
#         dl = (bounds[0,1] - bounds[0,0])/100.
        
#         transomMesh = initMesh.compress_mesh(np.array([1.,0.,0.])*(X0+dl), 'x',
#                                              tol=1e-5, scale_fac=10.)
        
#         Atr = transomMesh.frontal_area('x')
#         # Atr = compute_volume_and_center_of_mass(transom, faces)[0] / dl # alternate approximate way to compute Atr
        
#         z_min = initMesh.bounds[2,0]
    
#         Ttr = wide(point)[0,2] - z_min
        
#         hydrostaticData = {}
        
#         hydrostaticData['volume'] = initMesh.volume
        
#         points = {'cob': cob, 'cof': cob*0., 'cow': cow, '0L': X0}
#         lengths = {'Lwl': 0., 'Bwl': 0., 'T': T, 'Ttr': Ttr, '0L': X0}
#         areas = {'Wsa': Wsa, 'Wpa': 0., 'Ax': Ax, 'Ay': Ay, 'Atr': Atr, 'Abt': 0.}
#         coefs = {'Cb': 1., 'Cp': 1., 'Cwp': 1., 'Cx': 1., 'Cy': 1.}
    
#         hydrostaticData |= points | lengths | areas | coefs
    
#         hydrostaticData['immersion'] = z
#         hydrostaticData['heel'] = heel
#         hydrostaticData['trim'] = trim
#         hydrostaticData['ie'] = None
        
#         ### AEROSTATICS COMPUTATION
        
#         points = {'caa': cob*0.}
#         areas = {'Dsa': 0., 'Ax': 0., 'Ay': 0.}
        
#         aerostaticData = {}
        
#         aerostaticData |= points | areas
        
#         aerostaticData['immersion'] = z
#         aerostaticData['heel'] = heel
#         aerostaticData['trim'] = trim

#     elif fully_emerged:
#         ### AEROSTATICS COMPUTATION
        
#         dryBounds = initMesh.bounds
        
#         Taa = dryBounds[2,1] - wide(point)[0,2]
#         Axaa = initMesh.frontal_area('x')
#         Ayaa = initMesh.frontal_area('y')
#         Dsa = initMesh.weighted_area(weight=ramp(initFaceDist-1e-3, tol=1e-3))
        
#         caa = initMesh.area_centroid
        
#         points = {'caa': caa}
#         areas = {'Dsa': Dsa, 'Ax': Axaa, 'Ay': Ayaa}
        
#         aerostaticData = {}
        
#         aerostaticData |= points | areas
        
#         aerostaticData['immersion'] = z
#         aerostaticData['heel'] = heel
#         aerostaticData['trim'] = trim
        
#         ### HYDROSTATICS COMPUTATION
        
#         hydrostaticData = {}
        
#         hydrostaticData['volume'] = 0.
        
#         points = {'cob': caa*0., 'cof': caa*0., 'cow': caa*0., '0L': caa*0.}
#         lengths = {'Lwl': 0., 'Bwl': 0., 'T': 0., 'Ttr': 0., '0L': 0.}
#         areas = {'Wsa': 0., 'Wpa': 0., 'Ax': 0., 'Ay': 0., 'Atr': 0., 'Abt': 0.}
#         coefs = {'Cb': 0., 'Cp': 0., 'Cwp': 0., 'Cx': 0., 'Cy': 0.}
    
#         hydrostaticData |= points | lengths | areas | coefs
    
#         hydrostaticData['immersion'] = z
#         hydrostaticData['heel'] = heel
#         hydrostaticData['trim'] = trim
#         hydrostaticData['ie'] = None
    
#     else:
#         ### HYDROSTATICS COMPUTATION
        
#         wetMesh = initMesh.compress_mesh(point, normal, dist=initVertDist,
#                                          tol=1e-5, scale_fac=2.)
    
#         dryMesh = initMesh.compress_mesh(point, -normal, dist=-initVertDist,
#                                          tol=1e-5, scale_fac=2.)
        
#         waterplaneMesh = wetMesh.compress_mesh(point, -normal, dist=-initVertDist,
#                                                tol=1e-5, scale_fac=100.)
        
#         # transomMesh = wetMesh.compress_mesh()
        
#         volume = wetMesh.volume
#         cob = wetMesh.volume_centroid
        
#         Wsa = wetMesh.weighted_area(weight=ramp(initFaceDist))
#         cow = wetMesh.weighted_area_centroid(weight=ramp(initFaceDist, tol=1e-5))
        
#         Wpa = waterplaneMesh.weighted_area(weight=ramp(initFaceDist))
#         cof = waterplaneMesh.weighted_area_centroid(weight=ramp(initFaceDist, tol=1e-5))
        
#         wetBounds = wetMesh.bounds
        
#         X0 = wetBounds[0,0]
#         Lwl = wetBounds[0,1] - wetBounds[0,0]
#         Bwl = wetBounds[1,1] - wetBounds[1,0]
#         T = wetBounds[2,1] - wetBounds[2,0]
        
#         Ax = wetMesh.frontal_area('x')
#         Ay = wetMesh.frontal_area('y')
#         Awp = waterplaneMesh.frontal_area('z')
        
#         dl = Lwl/100.
        
#         transomMesh = wetMesh.compress_mesh(np.array([1.,0.,0.])*(X0+dl), 'x',
#                                             tol=1e-5, scale_fac=10.)
        
#         Atr = transomMesh.frontal_area('x')
#         # Atr = compute_volume_and_center_of_mass(transom, faces)[0] / dl # alternate approximate way to compute Atr
        
#         v = transomMesh.vertices
        
#         z_min = np.min(v[:,2])
#         z_max = np.max(v[:,2])
    
#         Ttr = z_max - z_min
        
#         Cb = volume / (Lwl * Bwl * T)
#         Cp = volume / (Ax * Lwl)
#         Cx = Ax / (Bwl * T)
#         Cy = Ay / (Lwl * T)
#         Cwp = Awp / (Lwl * Bwl)
        
#         hydrostaticData = {}
        
#         hydrostaticData['volume'] = volume
        
#         points = {'cob': cob, 'cof': cof, 'cow': cow, '0L': X0}
#         lengths = {'Lwl': Lwl, 'Bwl': Bwl, 'T': T, 'Ttr': Ttr, '0L': X0}
#         areas = {'Wsa': Wsa, 'Wpa': Awp, 'Ax': Ax, 'Ay': Ay, 'Atr': Atr, 'Abt': 0.}
#         coefs = {'Cb': Cb, 'Cp': Cp, 'Cwp': Cwp, 'Cx': Cx, 'Cy': Cy}
    
#         hydrostaticData |= points | lengths | areas | coefs
    
#         hydrostaticData['immersion'] = z
#         hydrostaticData['heel'] = heel
#         hydrostaticData['trim'] = trim
#         hydrostaticData['ie'] = None
        
#         ### AEROSTATICS COMPUTATION
        
#         dryBounds = dryMesh.bounds
        
#         Taa = dryBounds[2,1] - dryBounds[2,0]
#         Axaa = dryMesh.frontal_area('x')
#         Ayaa = dryMesh.frontal_area('y')
#         Dsa = dryMesh.weighted_area(weight=ramp(initFaceDist-1e-3, tol=1e-3))
        
#         caa = dryMesh.area_centroid
        
#         points = {'caa': caa}
#         areas = {'Dsa': Dsa, 'Ax': Axaa, 'Ay': Ayaa}
        
#         aerostaticData = {}
        
#         aerostaticData |= points | areas
        
#         aerostaticData['immersion'] = z
#         aerostaticData['heel'] = heel
#         aerostaticData['trim'] = trim
    
    
if __name__=='__main__':
    
    #%% COMPUTING
    
    import trimesh
    
    from archibald2.optimization import Opti
    
    def ReLU(x):
        return x * (x > 0)
    
    def waterplane_normal(heel_deg: float, trim_deg: float) -> np.ndarray:
        """
        Compute the unit normal vector of the waterplane given heel and trim angles.
    
        Parameters
        ----------
        heel_deg : float
            Heel angle in degrees (rotation around ship longitudinal axis, roll).
            Positive heel = starboard down.
        trim_deg : float
            Trim angle in degrees (rotation around ship transverse axis, pitch).
            Positive trim = bow down.
    
        Returns
        -------
        np.ndarray
            Normal vector of the waterplane (unit vector, shape (3,)).
        """
    
        # Convert to radians
        heel = np.deg2rad(heel_deg)
        trim = np.deg2rad(trim_deg)
    
        # Start with upright ship: waterplane normal is along +z
        n = np.array([0.0, 0.0, 1.0])
    
        # Rotation matrices
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(heel), -np.sin(heel)],
            [0, np.sin(heel), np.cos(heel)]
        ])
    
        Ry = np.array([
            [np.cos(trim), 0, np.sin(trim)],
            [0, 1, 0],
            [-np.sin(trim), 0, np.cos(trim)]
        ])
    
        # Apply rotations: first trim (pitch), then heel (roll)
        n_rot = Ry @ (Rx @ n)
    
        # Normalize
        return n_rot / np.linalg.norm(n_rot)
    
    archibald_root = 'C:/Users/jrich/NEOLINE DEVELOPPEMENT/NeoDev - ND-DÉVELOPPEMENT - Documents/ND-DÉVELOPPEMENT/01_Développement/08_Outils'
    # hullStl = archibald_root+'/archibald-main/Private/n136_data/n136.stl' # path to a STL mesh of the hull
    hullStl = archibald_root+'/archibald-main/Private/n136_data/n136_quad.stl' # path to a STL mesh of the hull
    hullMesh = trimesh.load(hullStl)
    
    diff_mesh = DifferentiableMesh(hullMesh.vertices, hullMesh.faces)
    
    opti = Opti()

    T0 = 5.332476916708436
    
    # T = opti.variable(init_guess=T0, lower_bound=1e-3)
    # T = opti.parameter(T0)
    T = T0
    
    X0 = np.mean(diff_mesh.vertices, axis=0)[0]
    
    normal = waterplane_normal(heel_deg = 0., trim_deg = 0.00)
    point = np.array([X0, 0.0, 0.0]) + T * normal
    # point = np.array([X0, T, 0.0])
    # normal = np.array([-0.0, 0.0, 1.0])
    
    
    dist = diff_mesh.vertices_distances_to_plane(point, normal)
    
    # edges = diff_mesh.edges
    edges_raw = np.concatenate([
        diff_mesh.faces[:, [0, 1]],
        diff_mesh.faces[:, [1, 2]],
        diff_mesh.faces[:, [2, 0]],
    ], axis=0)
    
    face_indices = np.repeat(np.arange(len(diff_mesh.faces)), 3)  # each face repeated 3 times
    
    # edges_sorted = np.sort(edges, axis=1)  # shape (3*n_faces, 2)
    
    edges_raw = np.sort(edges_raw, axis=1)
    
    # edges, inverse_indices = np.unique(edges_raw, axis=0, return_inverse=True)
    edges = edges_raw
    
    edge_dist = dist[edges]
    edge_prod = np.prod(edge_dist, axis=1)
    edge_sign = np.sign(edge_prod)
    edge_mask = ((1-edge_sign)/2)
    
    
    selected_edges = edges[edge_mask == 1.]
    
    # face_mask = np.sum(edge_mask[inverse_indices].reshape((3,-1)).T, axis=1)//2
    face_mask = np.sum(edge_mask.reshape((3,-1)).T, axis=1)//2 # sliced faces
    
    # Get coordinates of the selected edge endpoints
    v0 = diff_mesh.vertices[selected_edges[:, 0]]  # shape: (n_crossing, 3)
    v1 = diff_mesh.vertices[selected_edges[:, 1]]
    
    v0p = diff_mesh.vertices[edges[:, 0]]  # shape: (n_crossing, 3)
    v1p = diff_mesh.vertices[edges[:, 1]]
    
    # Get distances of the endpoints to the plane
    d0 = edge_dist[edge_mask == 1.][:, 0]
    d1 = edge_dist[edge_mask == 1.][:, 1]
    
    d0p = edge_dist[:, 0]
    d1p = edge_dist[:, 1]
    
    # Compute interpolation factor t (clip to avoid numerical instability)
    t = d0 / (d0 - d1 + 1e-12)
    
    tp = d0p / (d0p - d1p + 1e-12)
    
    # inter_mask = ca.if_else(ca.logic_and(tp > 0, tp < 1), 1.0, np.nan)
    # inter_mask = ca.if_else(ca.logic_and(tp > 0, tp < 1), 1.0, np.nan)
    inter_mask = (tp>0) * (tp<1)
    
    inter_blow = ca.if_else(ca.logic_and(tp > 0, tp < 1), 1.0, np.nan)
    
    # Compute intersection points
    intersections = v0 + tall(t) * (v1 - v0)
    intersections_p = v0p + tall(tp*edge_mask) * (v1p - v0p) # WARNING, approx 100x slower
    
    vertices = diff_mesh.vertices.copy()
    faces = diff_mesh.faces.copy()
    nV = len(vertices)
    nF = len(faces)
    
    # full_result = np.full((edges.shape[0]), -1)
    # full_result[edge_mask==1.] = np.arange(len(intersections))
    
    # intersection_indices = full_result[inverse_indices].reshape((3,-1)).T
    # intersection_indices = full_result.reshape((3,-1)).T
    # intersection_indices = ((np.cumsum(inter_mask)-1)).reshape((3,-1)).T
    # intersection_indices = ((np.cumsum(inter_mask)-1)*(1/inter_mask)).reshape((3,-1)).T
    # intersection_indices = np.repeat(np.array([np.nan, 0, 1]), nF).reshape((3,-1)).T *\
    intersection_indices = (np.full(edges.shape[0], -1) +\
        np.cumsum(edge_mask)*edge_mask).reshape((3, -1)).T
    intersection_indices_p = (np.full(edges.shape[0], -1) +\
        (np.arange(len(edge_mask))+1)*edge_mask).reshape((3, -1)).T

    new_vertices = np.concatenate((vertices, intersections))
    new_vertices_p = np.concatenate((vertices, intersections_p))
    data = []
    
    new_faces_p1 = []
    new_faces_p2 = []
    
    real_mask = np.full(4*nF, np.nan)
    # real_mask1 = []
    # real_mask2 = []
    
    for i in range(nF):
        V = faces[i,:]
        I = intersection_indices[i,:]
        Ip = intersection_indices_p[i,:]
        
        # initial
        a = [V]
        
        # sliced
        # shift = np.argwhere(np.isinf(I)|np.isnan(I))[0,0]
        shift = np.argwhere(I==-1)[0,0]
        # shift = 0
        
        i0 = (0 + shift) % 3
        i1 = (1 + shift) % 3
        i2 = (2 + shift) % 3
        
        b = [
            np.array([V[i2], nV+I[i2], nV+I[i1]]),
            np.array([V[i1], nV+I[i1], nV+I[i2]]),
            np.array([V[i0], V[i1], nV+I[i2]])
        ]
        
        bp = [
            np.array([V[i2], nV+Ip[i2], nV+Ip[i1]]),
            np.array([V[i1], nV+Ip[i1], nV+Ip[i2]]), 
            np.array([V[i0], V[i1], nV+Ip[i2]])    
        ]
        
        new_faces_p1.append(a)
        new_faces_p2.append(bp)
        
        # real_mask1.append(np.array([1-face_mask[i]]))
        # real_mask2.append(np.array([face_mask[i]]*3))
        
        real_mask[i] = 1-face_mask[i]
        real_mask[nF+3*i:nF+3*i+3] = face_mask[i]
        
        data.append(
            a * (1-int(face_mask[i])) +\
            b * int(face_mask[i])
            )
    
    new_faces = np.concatenate(data)
    new_faces_p = np.concatenate(new_faces_p1+new_faces_p2)
    # real_mask = np.concatenate(real_mask1+real_mask2)
    
    new_mesh = DifferentiableMesh(vertices=new_vertices, faces=new_faces.astype(int))
    # new_mesh_p = DifferentiableMesh(vertices=new_vertices, faces=new_faces_p[real_mask==1].astype(int))
    new_mesh_p = DifferentiableMesh(vertices=new_vertices_p, faces=new_faces_p.astype(int))
    
    # center = np.mean(intersections, axis=0)
    center = point
    centers = np.repeat(wide(center), new_mesh.vertices.shape[0], axis=0)
    
    new_dist = np.concatenate((dist, np.zeros(len(intersections))))
    
    wet_weight = (1 - np.tanh(1e6*(new_dist-1e-4)))/2
    dry_weight = (1 + np.tanh(1e6*(new_dist+1e-4)))/2
        
    wet_vertices = centers * tall(1-wet_weight) + new_mesh.vertices * tall(wet_weight)
    dry_vertices = centers * tall(1-dry_weight) + new_mesh.vertices * tall(dry_weight)
    
    cross = dry_weight * wet_weight
    
    wp_vertices = centers * tall(1 - cross) + new_mesh.vertices * tall(cross)
    
    wet_mesh = DifferentiableMesh(wet_vertices, faces=new_faces.astype(int))
    dry_mesh = DifferentiableMesh(dry_vertices, faces=new_faces.astype(int))
    
    wp_mesh = DifferentiableMesh(wp_vertices, faces=new_faces.astype(int))
    
    Vaa = dry_mesh.volume
    Caa = dry_mesh.volume_centroid
    
    Vw = wet_mesh.volume
    cob = wet_mesh.volume_centroid
    
    Awp = wp_mesh.area / 2
    cof = wp_mesh.area_centroid

    
    rho = 1.025
    displacement = 10e3
    volume = displacement / rho
    
    opti.minimize((Vw - volume)**2)
    
    # sol = opti.solve()
    
        
    print()
    # print('Vaa', Vaa)
    print('Vw', Vw)
    print('cob', cob)
    # print('cof', cof)
    # print('Awp', Awp)
    
    # print((Vw+Vaa)/new_mesh.volume)
    
    # TODO: OPTIMIZE EXECUTION TIME
    
    fdist = new_mesh.faces_distances_to_plane(point, normal)
    
    zfac = ReLU(-fdist)# * (1-face_mask) #+ fac1 * face_mask
    
    # p = (rho * g * zfac)
    p = zfac
    # dS = tall(diff_mesh._data['triangle_areas']) * normals
    dS = new_mesh.normals
    
    pi = - tall(p) * dS
    
    volume = np.sum(pi[:,2], axis=0)
    
    # f = np.linalg.norm(pi, axis=1)
    # f = np.linalg.norm(pi[:,:2], axis=1)
    f = pi[:,2]
    
    lcb = np.sum(f * new_mesh._data['triangle_centers'][:,0], axis=0)/ volume
    tcb = np.sum(f * new_mesh._data['triangle_centers'][:,1], axis=0)/ volume
    vcb = np.sum(f * new_mesh._data['triangle_centers'][:,2], axis=0)/ volume
    
    result = diff_mesh.hydrostatics(point, normal)
    
    print()
    print(result)
    
    # # print(lcb, tcb, vcb)
    # v0, v1, v2 = diff_mesh._v0, diff_mesh._v1, diff_mesh._v2
    # cross_prod = np.cross(v2-v0, v1-v0)
    # vols = np.sum((v0-point) * cross_prod / 6.0, axis=1)
    
    # fdist = diff_mesh.faces_distances_to_plane(point, -normal)
    
    # print(np.sum(vols * np.fmax(np.fmin(fdist+0.5,1.),0.)))
    # print(np.sum(vols * (fdist>0)))


    #%% DRAWING
    
    import pyvista as pv
    
    # Create a PyVista plotter
    plotter = pv.Plotter()
    
    color = 'cyan'
    opacity = 0.5
    show_edges = True
    draw_plane = True
    # point = np.zeros(3)
    # normal = np.array([0., 0., 1.])
    mesh_color = 'grey'
    plane_color = 'orange'
    plane_opacity = 0.01
    backend: str = 'pyvista'
    show: bool = False
    # show: bool = True
    
    
    if show:
        # Create the mesh
        
        mesh = pv.PolyData(new_mesh.vertices, np.hstack([[3, *face] for face in new_mesh.faces]))
        plotter.add_mesh(mesh, 
                          # color=color, 
                          scalars=wet_weight,       # This lets PyVista color using a colormap
                          cmap='viridis',
                          opacity=opacity,
                          show_edges=show_edges
                          )
        
        # mesh = pv.PolyData(new_mesh_p.vertices, np.hstack([[3, *face] for face in new_mesh_p.faces]))
        # plotter.add_mesh(mesh, 
        #                   # color=color, 
        #                   scalars=real_mask,       # This lets PyVista color using a colormap
        #                  cmap='viridis',
        #                  opacity=real_mask,
        #                  show_edges=show_edges
        #                  )
        
        if draw_plane:
            # Add the plane
            plane = pv.Plane(center=point, direction=normal, i_size=200, j_size=200)
            plotter.add_mesh(plane, color=plane_color, opacity=plane_opacity)
            
        if len(selected_edges) > 0:
            lines = np.hstack([[2, *edge] for edge in selected_edges]).astype(np.int32)
            line_mesh = pv.PolyData()
            line_mesh.points = diff_mesh.vertices
            line_mesh.lines = lines
            plotter.add_mesh(line_mesh, color="red", line_width=3)
            
        if len(intersections) > 0:
            # Add intersection points
            # intersection_cloud = pv.plot(intersections)
            plotter.add_points(intersections,
                                   style='points_gaussian',
                                    render_points_as_spheres=True,
                                   # scalars=rgba,
                                   # rgba=True,
                                   color='red',
                                   point_size=0.5,
                                   )
        
        
        # Display the plot
        plotter.show()
        
        volume = np.sum(np.sum((diff_mesh._v0 - wide(point)) * diff_mesh._data["cross_product"] / 6.0, axis=1) * np.fmax(np.fmin(-(diff_mesh._data['triangle_centers'] - wide(point)) @ normal + 0.5, 1.0), 0.0))
