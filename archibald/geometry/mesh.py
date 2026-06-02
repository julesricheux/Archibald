# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 23:38:18 2024

@author: jrich
"""
import casadi as ca
import archibald.numpy as np

from typing import Union, List
from archibald.common import ArchibaldObject
from archibald.performance import OperatingPoint
from archibald.toolbox.string_formatting import axis_string_to_array

#%%

def tall(array):
    return np.reshape(array, (-1, 1))


def wide(array):
    return np.reshape(array, (1, -1))


def rotation_matrix(
        heel: float = 0.,
        trim: float = 0.,
        leeway: float = 0.,
    ):
    """
    Computes the standard naval/aeronautics rotation matrix (Z-Y-X convention)
    to rotate points based on leeway, trim, and heel angles.
    
    Parameters:
    -----------
    heel_deg : float
        Heel angle (Roll) in degrees. Positive heels to starboard.
    trim_deg : float
        Trim angle (Pitch) in degrees. Positive is bow down.
    leeway_deg : float
        Leeway angle (Yaw) in degrees. Positive is leeway to starboard.
        
    Returns:
    --------
    R : numpy.ndarray
        A 3x3 rotation matrix.
    """
    # Convert angles to radians
    phi = np.radians(heel)      # Roll
    theta = np.radians(trim)    # Pitch
    psi = np.radians(leeway)    # Yaw / Leeway

    # Pre-compute sine and cosine values
    c_phi, s_phi = np.cos(phi), np.sin(phi)
    c_theta, s_theta = np.cos(theta), np.sin(theta)
    c_psi, s_psi = np.cos(psi), np.sin(psi)

    # Rotation around Z-axis (Leeway / Yaw)
    R_z = np.array([
        [c_psi, -s_psi, 0],
        [s_psi,  c_psi, 0],
        [0,      0,     1]
    ])

    # Rotation around Y-axis (Trim / Pitch)
    R_y = np.array([
        [ c_theta, 0, s_theta],
        [ 0,       1, 0      ],
        [-s_theta, 0, c_theta]
    ])

    # Rotation around X-axis (Heel / Roll)
    R_x = np.array([
        [1, 0,       0      ],
        [0, c_phi, -s_phi],
        [0, s_phi,  c_phi]
    ])

    # Combined matrix: R = Rz * Ry * Rx
    R = R_z @ R_y @ R_x

    return R


class ArchibaldPolygon(ArchibaldObject):
    """
    Polygon description.
    
    Allows approximated though differentiable hydrostatics computations.
    
    """
    def __init__(
        self,
        vertices: Union[np.ndarray, List] = np.array([]),
        edges: List[int] = None
    ):
        
        if vertices is None or edges is None:
            vertices = np.zeros((2,3))
            # faces = wide(np.arange(2))
        
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
        
        self._data = {
            'cross_product': None,
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
        
        self._data['bounds'] = np.array([
            [x_min, x_max],
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
    

    def draw(
            self,
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


class ArchibaldMesh(ArchibaldObject):
    """
    Differentiable mesh description. Vertices may be described by both numpy or CasADI objects.
    
    Allows approximated though differentiable hydrostatics computations.
    
    """
    
    # TODO: add a refine_mesh method to adapt coarser meshes
    
    def __init__(
        self,
        vertices: Union[np.ndarray, List] = np.array([]),
        faces: List[int] = None,
    ):
        
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
    
    @property
    def edges_lengths(self):
        if self._data['edges_lengths'] is None:
            self.compute_edges_lengths()
            
        return self._data['edges_lengths']
    
    @property
    def average_length(self):
        if self._data['average_length'] is None:
            self.compute_average_length()
            
        return self._data['average_length']
    
    @vertices.setter
    def vertices(self, value):
        # Check if value is a numpy array, CasADi MX, or CasADi DM
        if not (np.is_casadi_type(value) or type(value) == np.ndarray):
            raise ValueError(f"Vertices must be a numpy array or a CasADi array (MX or DM), not {type(value)}.")
        
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
        
        self._data = {
            'cross_product': None,
            'edges': None,
            'edges_lengths': None,
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
            'average_length': None,
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
        
    def compute_edges_lengths(self):
        """
        Compute the length of each mesh edge.

        """
        if self._data['edges'] is None:
            self.compute_edges()
            
        # edges_pts = self.vertices[self.edges[:, 0]]
        v0, v1 = self.vertices[self.edges[:, 0]], self.vertices[self.edges[:, 1]]
        
        self._data['edges_lengths'] = np.linalg.norm(v1 - v0, axis=1)
        
    def compute_average_length(self):
        """
        Compute the average mesh edge length.

        """
        if self._data['edges_lengths'] is None:
            self.compute_edges_lengths()
        
        self._data['average_length'] = np.mean(self.edges_lengths)
        
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
        
            
    def compute_tetrahedron_volumes(
        self,
        ref_point: Union[np.ndarray, List] = [0., 0., 0.],
    ):
        """
        Compute the signed volumes of all tetrahedrons formed by faces and origin.

        """
        if self._data['cross_product'] is None:
            self.compute_cross_product()
        
        # Signed volume of tetrahedron formed with origin for each face
        self._data['tetrahedron_volumes'] = np.sum(
            np.add(
                self._v0,
                -wide(ref_point)
            ) * self._data["cross_product"] / 6.0,
            axis=1
        )
    
    def vertices_distances_to_plane(
        self,
        point: Union[np.ndarray, List],
        normal: Union[np.ndarray, List, str],
    ):
        """
        Compute the oriented distances of each vertex to a given plane, represented by a point and a normal.

        """
        if type(normal) is str:
            normal = axis_string_to_array(normal)
        # NB
        # if dist.all < 0 : fully immersed
        # if dist.all > 0 : fully emerged
        
        return np.add(
            self.vertices,
            -wide(point)
        ) @ tall(normal)
    
    def faces_distances_to_plane(
        self,
        point: Union[np.ndarray, List],
        normal: Union[np.ndarray, List, str],
    ):
        """
        Compute the oriented distances of each vertex to a given plane, represented by a point and a normal.

        """
        if type(normal) is str:
            normal = axis_string_to_array(normal)
        # NB
        # if dist.all < 0 : fully immersed
        # if dist.all > 0 : fully emerged
        
        if self._data['triangle_centers'] is None:
            self.compute_triangle_centers()
        
        return np.add(self._data['triangle_centers'], -wide(point)) @ normal
    
    def frontal_area(
        self,
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
            direction = axis_string_to_array(direction)
        
        if self._data['cross_product'] is None:
            self.compute_cross_product()
            
        cross_prod = self._data['cross_product'] # * weight
        
        # Normalize the direction vector
        direction = direction / np.linalg.norm(direction)

        # Compute the product between the cosine of the angle between the triangle normal and the projection direction and the triangle area
        cos_theta_area = (cross_prod @ tall(direction))

        frontal_area = np.sum(np.fabs(cos_theta_area) * tall(weight)) / 4.0
        
        return frontal_area
        
    def weighted_volume(
        self,
        weight: Union[np.ndarray, float] = 1.,
        ref_point: Union[np.ndarray, List] = [0., 0., 0.],
        recompute_tetrahedron_volumes: bool = False,
    ):
        """
        Compute the mesh volume. May be weighted.

        """
        if recompute_tetrahedron_volumes or self._data['tetrahedron_volumes'] is None:
            self.compute_tetrahedron_volumes(ref_point)
        
        # Signed volume of tetrahedron formed with origin for each face
        tetra_volumes = self._data['tetrahedron_volumes'] * weight
        
        # Total volume
        return np.sum(tetra_volumes)
        
    def weighted_volume_centroid(
        self,
        weight: Union[np.ndarray, float] = 1.,
        ref_point: Union[np.ndarray, List] = [0., 0., 0.],
        recompute_tetrahedron_volumes: bool = False,
    ):
        """
        Compute the mesh volume centroid. May be weighted.

        """
        if recompute_tetrahedron_volumes or self._data['tetrahedron_volumes'] is None:
            self.compute_tetrahedron_volumes(ref_point)
            
        if self._data['tetrahedron_centers'] is None:
            self.compute_tetrahedron_centers()
        
        tetra_centers = self._data['tetrahedron_centers']
        tetra_volumes = self._data['tetrahedron_volumes'] * weight
        
        # Weighted sum of centroids by volumes to get the total center of mass
        # return np.sum(wide(tetra_volumes) @ tetra_centers / np.sum(tetra_volumes), axis=0)
    
        return np.sum(
            tetra_centers * tall(tetra_volumes) / np.sum(tetra_volumes),
            axis=0
        )
    
    def weighted_area(
        self,
        weight: Union[np.ndarray, float] = 1.,
    ):
        """
        Compute the mesh area. May be weighted.

        """
        if self._data['triangle_areas'] is None:
            self.compute_triangle_areas()
        
        # Areas of each triangular faces
        tri_areas = self._data['triangle_areas'] * weight
        
        # Total area
        return np.sum(tri_areas)

    def weighted_area_centroid(
            self,
            weight: Union[np.ndarray, float] = 1.,
        ):
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
        
        self._data['bounds'] = np.array([
            [x_min, x_max],
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
        
    def __repr__(self):
        
        return (
            f"{self.__class__.__name__} instance "+\
            f"with {self.vertices.shape[0]} vertices, "+\
            f"{self.edges.shape[0]} edges "+\
            f"and {self.faces.shape[0]} faces."
        )
    
    def hydrostatics(
            self,
            point: Union[np.ndarray, List] = np.zeros(3),
            normal: Union[np.ndarray, List, str] = "z",
        ):
        
        if type(normal) is str:
            normal = axis_string_to_array(normal)
        
        if self._data['cross_product'] is None:
            self.compute_cross_product()
            
        if self._data['tetrahedron_centers'] is None:
            self.compute_tetrahedron_centers()
        
        # vertices signed distances from the waterplane, shape similar to self.faces
        # > 0 for wet, < 0 for dry
        vdist = -self.vertices_distances_to_plane(
            point,
            normal
        )[self.faces]
        
        # avg_dist = self.average_length
        avg_dist = 1.
        
        mix_weights = np.sigmoid(
            np.mean(
                vdist/avg_dist, 
                axis=1
            ) * 10./3.
            # ) * np.sqrt(10.)
        ) # smooth clipping weight
        # calibrated from the wet area variation of an equilateral triangle from
        # the average of its vertices signed distances
        
        highest_v = np.min(vdist, axis=1) # signed distance of the highest vertex of each face
        deepest_v = np.max(vdist, axis=1) # signed distance of the deepest vertex of each face
        
        corr = 1e3
        wet = np.sigmoid(highest_v*corr) * np.sigmoid(deepest_v*corr) # boolean for fully wet faces
        dry = np.sigmoid(-highest_v*corr) * np.sigmoid(-deepest_v*corr) # boolean for fully dry faces
        
        # weight = 1 for fully wet faces, 0 for fully dry faces,  
        weights = np.fmax(
            wet,
            mix_weights
        ) * (1-dry) 
        
        # compute tetrahedron volumes with reference point on the waterplane
        self.compute_tetrahedron_volumes(point)
        
        # total volume
        volume = self.weighted_volume(
            weight=weights,
            ref_point=point,
            recompute_tetrahedron_volumes=False,
        )
        
        # average weighted tetrahedron center i.e. center of buoyancy
        cob = self.weighted_volume_centroid(
            weight=weights,
            ref_point=point,
            recompute_tetrahedron_volumes=False,
        )
        
        return volume, cob
    
    
    def slice_mesh(
            self,
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
            
            return ArchibaldPolygon(intersections, slice_edges)
        
        return ArchibaldPolygon(None, None)
    
    def transform(
            self,
            op_point: OperatingPoint,
            inverse: bool = False,
        ):
    
        self.vertices = op_point.apply_transformations(self.vertices, inverse=inverse)
            
    def draw(
        self,
        color = 'orange',
        opacity = 0.3,
        show_edges = True,
        draw_plane = False,
        point = np.zeros(3),
        normal = np.array([0., 0., 1.]),
        mesh_color = 'grey',
        cmap = "RdYlGn",
        plane_color = 'blue',
        plane_opacity = 0.2,
        backend: str = 'pyvista',
        show: bool = True,
        ax = None,                      # Added for matplotlib compatibility
        set_axis_visibility = None,     # Added for matplotlib compatibility
    ):
        
        if backend == 'pyvista':
            import pyvista as pv
            
            # Create a PyVista plotter
            plotter = pv.Plotter()
            
            # Create the mesh
            mesh = pv.PolyData(self.vertices, np.hstack([[3, *face] for face in self.faces]))
            
            if draw_plane:
                weight = np.sigmoid(
                    self.vertices_distances_to_plane(
                        point,
                        normal
                    ) * 10./3.
                )
                plotter.add_mesh(
                    mesh,
                    show_edges=show_edges,
                    scalars=weight,
                    cmap=cmap,
                )
                # Add the plane
                b = self.bounds
                diag = np.linalg.norm(b[:, 1] - b[:, 0]) * 1.2
                projected_centroid = (
                    self.area_centroid
                    - np.dot(
                        wide(self.area_centroid) - wide(point),
                        tall(normal),
                    )
                )
                plane = pv.Plane(
                    center=projected_centroid,
                    direction=normal,
                    i_size=diag, j_size=diag,
                )
                plotter.add_mesh(
                    plane,
                    color=plane_color,
                    opacity=plane_opacity
                )
            else:
                plotter.add_mesh(
                    mesh,
                    show_edges=show_edges,
                    color=color,
                    opacity=opacity,
                )
                
            if set_axis_visibility:
                plotter.add_axes()
                plotter.show_grid(color='gray')
            
            if show:
                # Display the plot
                plotter.show()
            return plotter
            
        elif backend == 'plotly':
            import plotly.graph_objects as go
            
            fig = go.Figure()
            
            x, y, z = self.vertices.T
            i, j, k = self.faces.T
            
            if draw_plane:
                # Compute weights
                weight = np.sigmoid(
                    self.vertices_distances_to_plane(point, normal) * 10. / 3.
                )
                
                # Draw the main mesh mapped to the colorscale
                fig.add_trace(
                    go.Mesh3d(
                        x=x, y=y, z=z,
                        i=i, j=j, k=k,
                        intensity=weight,
                        colorscale=cmap,
                        showscale=False,
                    )
                )
                
                # Calculate the plane boundaries
                b = self.bounds
                diag = np.linalg.norm(b[:, 1] - b[:, 0])
                projected_centroid = (
                    self.area_centroid
                    - np.dot(
                        wide(self.area_centroid) - wide(point),
                        tall(normal),
                    )
                ).flatten()
                
                if np.allclose(normal[:2], 0):
                    v1 = np.array([1., 0., 0.])
                else:
                    v1 = np.array([-normal[1], normal[0], 0.])
                v1 /= np.linalg.norm(v1)
                v2 = np.cross(normal, v1)
                v2 /= np.linalg.norm(v2)
                
                half_size = diag / 2.0 * 1.2
                p1 = projected_centroid - half_size * v1 - half_size * v2
                p2 = projected_centroid + half_size * v1 - half_size * v2
                p3 = projected_centroid + half_size * v1 + half_size * v2
                p4 = projected_centroid - half_size * v1 + half_size * v2
                
                # Draw the plane as two triangles (0,1,2 and 0,2,3)
                fig.add_trace(
                    go.Mesh3d(
                        x=[p1[0], p2[0], p3[0], p4[0]],
                        y=[p1[1], p2[1], p3[1], p4[1]],
                        z=[p1[2], p2[2], p3[2], p4[2]],
                        i=[0, 0],
                        j=[1, 2],
                        k=[2, 3],
                        color=plane_color,
                        opacity=plane_opacity,
                    )
                )
            else:
                # Standard drawing without a plane
                fig.add_trace(
                    go.Mesh3d(
                        x=x, y=y, z=z,
                        i=i, j=j, k=k,
                        opacity=opacity,
                        color=color,
                    )
                )
            
            # Setup scene aspect and axis visibility
            scene_layout = dict(aspectmode='data')
            
            if set_axis_visibility is False:
                scene_layout.update(
                    xaxis=dict(visible=False),
                    yaxis=dict(visible=False),
                    zaxis=dict(visible=False)
                )
            elif set_axis_visibility is True:
                scene_layout.update(
                    xaxis=dict(visible=True),
                    yaxis=dict(visible=True),
                    zaxis=dict(visible=True)
                )
                
            fig.update_layout(scene=scene_layout)

            if show:
                from plotly.offline import plot
                plot(fig)
            return fig
            
        elif backend == 'matplotlib':
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d.art3d import Poly3DCollection
            
            # 1. Setup or verify the 3D axis
            if ax is None:
                fig = plt.figure(figsize=(8, 8))
                ax = fig.add_subplot(111, projection='3d')
                
            # Convert faces to a safe list of arrays for inhomogeneous shapes
            mesh_polys = [self.vertices[face] for face in self.faces]
            
            # Disable default shading to prevent immediate ValueError
            mesh_collection = Poly3DCollection(mesh_polys, shade=False)
            
            # 3. Handle Coloring and Shading Styles
            if draw_plane:
                weight = np.sigmoid(
                    self.vertices_distances_to_plane(point, normal) * 10. / 3.
                )
                # Compute face weights safely using list comprehension
                face_weights = np.array([np.mean(weight[face]) for face in self.faces])
                cmap_callable = plt.get_cmap(cmap)
                face_colors = cmap_callable(face_weights)
                mesh_collection.set_facecolor(face_colors)
            else:
                mesh_collection.set_facecolor(color)
                mesh_collection.set_alpha(opacity)
                
            if show_edges:
                mesh_collection.set_edgecolor('grey')
                mesh_collection.set_linewidth(0.5)
            else:
                mesh_collection.set_edgecolor('none')
                
            ax.add_collection3d(mesh_collection)
            
            # 4. Handle Waterplane Drawing (if requested)
            if draw_plane:
                b = self.bounds
                diag = np.linalg.norm(b[:, 1] - b[:, 0]) * 1.2
                projected_centroid = (
                    self.area_centroid
                    - np.dot(
                        wide(self.area_centroid) - wide(point),
                        tall(normal),
                    )
                )
                
                if np.allclose(normal[:2], 0):
                    v1 = np.array([1., 0., 0.])
                else:
                    v1 = np.array([-normal[1], normal[0], 0.])
                v1 /= np.linalg.norm(v1)
                v2 = np.cross(normal, v1)
                v2 /= np.linalg.norm(v2)
                
                half_size = diag / 2.0
                p1 = projected_centroid - half_size * v1 - half_size * v2
                p2 = projected_centroid + half_size * v1 - half_size * v2
                p3 = projected_centroid + half_size * v1 + half_size * v2
                p4 = projected_centroid - half_size * v1 + half_size * v2
                
                plane_collection = Poly3DCollection(
                    [p1, p2, p3, p4], 
                    facecolors=plane_color, 
                    alpha=plane_opacity,
                    shade=False
                )
                ax.add_collection3d(plane_collection)
            
            # 5. Apply view limits & visibility toggles requested by draw_three_view
            ax.auto_scale_xyz(self.vertices[:, 0], self.vertices[:, 1], self.vertices[:, 2])
            
            # FORCE UNCONSTRAINED ASPECT RATIO
            ax.set_box_aspect(None)  # Resets any forced 1:1:1 box aspects
                
            # ax.auto_scale_xyz(self.vertices[:, 0], self.vertices[:, 1], self.vertices[:, 2])
            
            if set_axis_visibility is False:
                ax.set_axis_off()
            elif set_axis_visibility is True:
                ax.set_axis_on()
            
            # set adequate zoom and scale
            b = self.bounds
            c = self.area_centroid
            main_dim = np.max(b[:, 1] - b[:, 0])
            
            ax.set_xlim(c[0] - main_dim/2., c[0] + main_dim/2.)
            ax.set_ylim(c[1] - main_dim/2., c[1] + main_dim/2.)
            ax.set_zlim(c[2] - main_dim/2., c[2] + main_dim/2.)
                
            if show:
                plt.show()
                
            return ax
            
        else:
            raise NotImplementedError(f'{backend} is not a supported drawing module.')