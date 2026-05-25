# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 23:38:18 2024

@author: jrich
"""

from typing import Union, List
import copy

import archibald.numpy as np
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